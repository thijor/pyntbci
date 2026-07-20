import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta, norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

import pyntbci.classifiers
from pyntbci.utilities import itr


def _supports_running(estimator: ClassifierMixin) -> bool:
    """Whether estimator supports running (incremental) decision_function()/predict(), i.e. is an eCCA or rCCA with
    ensemble=False (see their decision_function() docstrings). Used to decide, in the segment-by-segment loops
    below, whether to feed the estimator only the newly observed segment each time (cheap) or, for any other
    estimator (a user-supplied classifier, or ensemble=True), fall back to passing the whole growing prefix and
    letting it recompute from scratch each time (safe, works for any ClassifierMixin, but O(n_segments^2)).

    Parameters
    ----------
    estimator: ClassifierMixin
        The estimator to check.

    Returns
    -------
    supported: bool
        Whether estimator supports running decision_function()/predict().
    """
    return isinstance(estimator, (pyntbci.classifiers.eCCA, pyntbci.classifiers.rCCA)) and not getattr(
        estimator, "ensemble", False
    )


def _iter_segment_scores(
    estimator: ClassifierMixin,
    X: NDArray,
    segment_samples: int,
    n_segments: int,
):
    """Yield (i_segment, scores) for each growing segment of X (i.e., scores as returned by decision_function() on
    X[:, :, :(1 + i_segment) * segment_samples], for i_segment in range(n_segments)), using running=True (feeding
    only the newly observed segment, not the whole growing prefix, each time) when estimator supports it (see
    _supports_running()), and falling back to a from-scratch recompute on the full prefix otherwise.

    Parameters
    ----------
    estimator: ClassifierMixin
        The (fitted) estimator to compute segment scores with.
    X: NDArray
        The matrix of EEG data of shape (n_trials, n_channels, n_samples).
    segment_samples: int
        The size of a segment in samples.
    n_segments: int
        The number of segments to yield scores for.

    Yields
    ------
    i_segment: int
        The segment index.
    scores: NDArray
        The scores as returned by estimator.decision_function() for the growing prefix up to and including this
        segment.
    """
    use_running = _supports_running(estimator)
    prev = 0
    for i_segment in range(n_segments):
        idx = (1 + i_segment) * segment_samples
        if use_running:
            scores = estimator.decision_function(X[:, :, prev:idx], running=True, reset=(i_segment == 0))
        else:
            scores = estimator.decision_function(X[:, :, :idx])
        prev = idx
        yield i_segment, scores


def _iter_segment_predictions(
    estimator: ClassifierMixin,
    X: NDArray,
    segment_samples: int,
    n_segments: int,
):
    """Like _iter_segment_scores(), but yielding (i_segment, yh) from estimator.predict() instead of scores from
    decision_function().

    Parameters
    ----------
    estimator: ClassifierMixin
        The (fitted) estimator to compute segment predictions with.
    X: NDArray
        The matrix of EEG data of shape (n_trials, n_channels, n_samples).
    segment_samples: int
        The size of a segment in samples.
    n_segments: int
        The number of segments to yield predictions for.

    Yields
    ------
    i_segment: int
        The segment index.
    yh: NDArray
        The predicted labels as returned by estimator.predict() for the growing prefix up to and including this
        segment.
    """
    use_running = _supports_running(estimator)
    prev = 0
    for i_segment in range(n_segments):
        idx = (1 + i_segment) * segment_samples
        if use_running:
            yh = estimator.predict(X[:, :, prev:idx], running=True, reset=(i_segment == 0))
        else:
            yh = estimator.predict(X[:, :, :idx])
        prev = idx
        yield i_segment, yh


def _running_predict(
    stopping: ClassifierMixin,
    X_chunk: NDArray,
    reset: bool,
    use_decision_function: bool,
) -> NDArray:
    """Get cumulative decision_function() (or predict()) results for a growing trial, given only the newest chunk
    of data each call, for use by a *Stopping class's own predict()'s running=True mode. Works for any wrapped
    estimator, not just eCCA/rCCA: if the estimator supports running scoring itself (see _supports_running()), the
    chunk is forwarded directly and the estimator does the incremental work; otherwise, the raw chunks are
    buffered here (in stopping, not in the estimator) and the estimator recomputes from scratch on the full
    buffered prefix each call, exactly as if running=False had been used throughout, i.e. still correct, only not
    faster. Either way, the caller only ever needs to supply the new chunk, and the running state (which estimator
    predict()/decision_function() calls need, in the fallback case, or how many samples have been observed, in
    both cases, to resolve the current segment index) lives in stopping._running_.

    Parameters
    ----------
    stopping: ClassifierMixin
        The *Stopping instance whose running state (stopping._running_) to use and update.
    X_chunk: NDArray
        The newly observed chunk of EEG data of shape (n_trials, n_channels, n_new_samples).
    reset: bool
        Whether to discard any existing running state before processing this call.
    use_decision_function: bool
        Whether to call the estimator's decision_function() (True) or predict() (False).

    Returns
    -------
    result: NDArray
        The cumulative decision_function() scores or predict() labels, as if computed from scratch on the full
        trial observed so far (not just the new chunk).
    """
    if reset or stopping._running_ is None:
        stopping._running_ = {"n_samples": 0, "raw_buffer": None}
    r = stopping._running_
    method_name = "decision_function" if use_decision_function else "predict"
    method = getattr(stopping.estimator, method_name)
    if _supports_running(stopping.estimator):
        result = method(X_chunk, running=True, reset=reset)
    else:
        r["raw_buffer"] = X_chunk if r["raw_buffer"] is None else np.concatenate((r["raw_buffer"], X_chunk), axis=2)
        result = method(r["raw_buffer"])
    r["n_samples"] += X_chunk.shape[2]
    return result


class BayesStopping(ClassifierMixin, BaseEstimator):
    """Bayesian dynamic stopping. Fits Gaussian distributions for target and non-target responses, and calculates a
    stopping threshold using these and a cost criterion [1]_.

    Parameters
    ----------
    estimator: ClassifierMixin
        The classifier object that performs the classification.
    segment_time: float
        The size of a segment of data at which classification is performed in seconds.
    fs: int
        The sampling frequency of the EEG data in Hz.
    method: str (default: "bds0")
        The method to use for Bayesian dynamic stopping: bds0, bds1, bds2.
    cr: float (default: 1.0)
        The cost ratio.
    target_pf: float (default: 0.05)
        The targeted probability for error.
    target_pd: float (default: 0.80)
        The targeted probability for detection.
    max_time: float (default: None)
        The maximum time in seconds at which to force a stop, i.e., a classification. Trials will not be longer than
        this maximum time. If None, the algorithm will always emit -1 if it cannot stop.
    min_time: float (default: None)
        The minimum time in seconds at which a stop is possible, i.e., a classification. Before the minimum time, the
        algorithm will always emit -1. If None, the algorithm allows a stop already after the first segment of data.
    approach: str (default: "template_inner")
        The approach used to fit the BDS model. Either an analytic template-based method using the inner product, or an
        approach using the empirical scores from the estimator object.

    Attributes
    ----------
    classes_: NDArray
        The classes that can be predicted, taken from the wrapped estimator's classes_ after fitting. Note, predict()
        may additionally return -1 to indicate a trial has not yet been stopped, which is not itself a class.
    alpha_: float
        The scaling parameter between observed and predicted responses.
    sigma_: float
        The standard deviation of the noise.
    b0_: NDArray
        The mean of the non-target Gaussian distribution of shape (n_segments).
    b1_: NDArray
        The mean of the target Gaussian distribution of shape (n_segments).
    s0_: NDArray
        The standard deviation of the non-target Gaussian distribution of shape (n_segments).
    s1_: NDArray
        The standard deviation of the target Gaussian distribution of shape (n_segments).
    eta_: NDArray
        The decision boundary of shape (n_segments).
    pf_: NDArray
        The predicted probability of a false detection of shape (n_segments).
    pm_: NDArray
        The predicted probability of a miss of shape (n_segments).

    References
    ----------
    .. [1] Ahmadi, S., Desain, P., & Thielen, J. (2024). A Bayesian dynamic stopping method for evoked response
           brain-computer interfacing. Frontiers in Human Neuroscience, 18, 1437965.
    """

    classes_: NDArray
    alpha_: float
    sigma_: float
    b0_: NDArray
    b1_: NDArray
    s0_: NDArray
    s1_: NDArray
    eta_: NDArray
    pf_: NDArray
    pm_: NDArray
    _running_: dict = None

    def __init__(
        self,
        estimator: ClassifierMixin,
        segment_time: float,
        fs: int,
        method: str = "bds0",
        cr: float = 1.0,
        target_pf: float = 0.05,
        target_pd: float = 0.80,
        max_time: float = None,
        min_time: float = None,
        approach: str = "template_inner",
    ) -> None:
        self.estimator = estimator
        self.segment_time = segment_time
        self.fs = fs
        self.method = method
        self.cr = cr
        self.target_pf = target_pf
        self.target_pd = target_pd
        self.max_time = max_time
        self.min_time = min_time
        self.approach = approach

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> ClassifierMixin:
        """The training procedure to fit the dynamic procedure on supervised EEG data.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: NDArray
            The vector of ground-truth labels of the trials in X of shape (n_trials).

        Returns
        -------
        self: ClassifierMixin
            Returns the instance itself.
        """
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_

        if self.approach == "template_inner":
            self._fit_template_inner(X, y)
        else:
            self._fit_score(X, y)

        self._running_ = None
        return self

    def _fit_template_inner(
        self,
        X: NDArray,
        y: NDArray,
    ) -> None:
        """Fit the Bayesian dynamic stopping model using the analytic template-based approach, i.e., using the inner
        product between the rCCA templates. Sets alpha_, sigma_, b0_, b1_, s0_, s1_, eta_, pf_, and pm_.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: NDArray
            The vector of ground-truth labels of the trials in X of shape (n_trials).
        """
        assert isinstance(self.estimator, pyntbci.classifiers.rCCA), "Approach template_inner works only for rCCA."
        n_samples = X.shape[2]
        n_classes = self.estimator.Ts_.shape[0]

        # Spatially filter data and flatten
        X = self.estimator.cca_[0].transform(X=X)[0].reshape((-1, 1))

        # Get templates
        if n_samples < self.estimator.Ts_.shape[2]:
            T = self.estimator.Ts_
        else:
            T = np.concatenate(
                (self.estimator.Ts_, np.tile(self.estimator.Tw_, (1, 1, n_samples // self.estimator.Tw_.shape[2]))),
                axis=2,
            )
        T = T[:, :, :n_samples]

        # Obtain alpha from least squares
        T_ = T[y, :, :].reshape((-1, 1))
        model = LinearRegression()
        model.fit(T_, X)
        self.alpha_ = model.coef_[0, 0]

        # Obtain sigma from Gaussian fit to residuals
        residuals = X - model.predict(T_)
        self.sigma_ = norm.fit(residuals)[1]

        # Calculate b0, b1, s0, s1
        n_segments = int(n_samples / int(self.segment_time * self.fs))
        self.b0_ = np.zeros(n_segments)
        self.b1_ = np.zeros(n_segments)
        self.s0_ = np.zeros(n_segments)
        self.s1_ = np.zeros(n_segments)
        for i_segment in range(n_segments):
            idx = (1 + i_segment) * int(self.segment_time * self.fs)
            inner = np.inner(T[:, 0, :idx], T[:, 0, :idx])  # select component
            inner_xy = inner[~np.eye(n_classes, dtype=bool)]
            inner_xx = inner[np.eye(n_classes, dtype=bool)]
            self.b0_[i_segment] = inner_xy.mean()
            self.b1_[i_segment] = inner_xx.mean()
            self.s0_[i_segment] = np.sqrt(
                self.b1_[i_segment] * self.sigma_**2
                + np.mean((self.alpha_ * inner_xy - self.alpha_ * self.b0_[i_segment]) ** 2)
            )
            self.s1_[i_segment] = np.sqrt(
                self.b1_[i_segment] * self.sigma_**2
                + np.mean((self.alpha_ * inner_xx - self.alpha_ * self.b1_[i_segment]) ** 2)
            )

        # Calculate eta
        a = self.s1_**2 - self.s0_**2
        b = -2 * self.alpha_ * (self.s1_**2 * self.b0_ - self.s0_**2 * self.b1_)
        c = -(self.alpha_**2) * (
            self.s1_**2 * self.b0_**2 + self.s0_**2 * self.b1_**2
        ) + 2 * self.s0_**2 * self.s1_**2 * np.log(self.s0_ / (self.s1_ * (n_classes - 1) * self.cr))
        self.eta_ = (-b + np.sqrt(np.clip(b**2 - 4 * a * c, 0, None))) / (2 * a)

        # Calculate predicted error vectors (corrected for multiple comparisons)
        self.pf_ = ((n_classes - 1) / n_classes) * (1 - norm.cdf(self.eta_, self.alpha_ * self.b0_, self.s0_))
        self.pf_ = 1 - (1 - self.pf_) ** n_classes
        self.pm_ = (1 / n_classes) * norm.cdf(self.eta_, self.alpha_ * self.b1_, self.s1_)
        self.pm_ = 1 - (1 - self.pm_) ** n_classes

    def _fit_score(
        self,
        X: NDArray,
        y: NDArray,
    ) -> None:
        """Fit the Bayesian dynamic stopping model using the empirical approach, i.e., using the empirical scores
        obtained from the estimator object. Sets b0_, b1_, s0_, s1_, eta_, pf_, and pm_.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: NDArray
            The vector of ground-truth labels of the trials in X of shape (n_trials).
        """
        n_samples = X.shape[2]

        # Calculate b0, b1, s0, s1
        n_segments = int(n_samples / int(self.segment_time * self.fs))
        self.b0_ = np.zeros(n_segments)
        self.b1_ = np.zeros(n_segments)
        self.s0_ = np.zeros(n_segments)
        self.s1_ = np.zeros(n_segments)
        segment_samples = int(self.segment_time * self.fs)
        for i_segment, scores in _iter_segment_scores(self.estimator, X, segment_samples, n_segments):
            n_classes = scores.shape[1]
            mask = np.full(scores.shape, False)
            mask[np.arange(y.size), y] = True
            self.b0_[i_segment] = scores[~mask].mean()
            self.b1_[i_segment] = scores[mask].mean()
            self.s0_[i_segment] = scores[~mask].std()
            self.s1_[i_segment] = scores[mask].std()

        # Calculate eta
        a = self.s1_**2 - self.s0_**2
        b = -2 * (self.s1_**2 * self.b0_ - self.s0_**2 * self.b1_)
        c = -(self.s1_**2 * self.b0_**2 + self.s0_**2 * self.b1_**2) + 2 * self.s0_**2 * self.s1_**2 * np.log(
            self.s0_ / (self.s1_ * (n_classes - 1) * self.cr)
        )
        self.eta_ = (-b + np.sqrt(np.clip(b**2 - 4 * a * c, 0, None))) / (2 * a)

        # Calculate predicted error vectors (corrected for multiple comparisons)
        self.pf_ = ((n_classes - 1) / n_classes) * (1 - norm.cdf(self.eta_, self.b0_, self.s0_))
        self.pf_ = 1 - (1 - self.pf_) ** n_classes
        self.pm_ = (1 / n_classes) * norm.cdf(self.eta_, self.b1_, self.s1_)
        self.pm_ = 1 - (1 - self.pm_) ** n_classes

    def predict(
        self,
        X: NDArray,
        running: bool = False,
        reset: bool = False,
    ) -> NDArray:
        """The testing procedure to apply the estimator to novel EEG data using Bayesian dynamic stopping.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples). If running=True, this is only the
            newly observed samples since the previous call (not the full trial so far), see running below.
        running: bool (default: False)
            Whether to use running (incremental) scoring. If False (default), predict() behaves exactly as without
            this parameter: X is the complete trial data seen so far, and everything is recomputed from scratch
            (safe to call in any order, e.g. repeatedly with the same or a shorter X). If True, X is only the
            newly observed samples since the previous call, and a running state (kept internally, not a fitted
            attribute) is reused and updated; if the wrapped estimator supports running scoring itself (an eCCA or
            rCCA with ensemble=False), each call only does O(n_new_samples) work instead of reprocessing the whole
            trial, otherwise the raw data is buffered here and recomputed from scratch each call (still correct,
            just not faster). Use reset=True on the first call of a new running sequence (e.g. for a new trial or
            a new batch of trials); the running state is otherwise unaffected by (and does not affect) running=False
            calls, and is cleared by fit().
        reset: bool (default: False)
            Whether to discard any existing running state before processing this call. Only relevant if
            running=True; a never-yet-used instance already starts fresh without it, so it only needs to be set
            explicitly to start a new sequence before the previous one naturally ended.

        Returns
        -------
        y: NDArray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """
        if self.approach == "template_inner":
            check_is_fitted(self, ["alpha_", "sigma_", "b0_", "b1_", "s0_", "s1_", "pf_", "pm_"])
        else:
            check_is_fitted(self, ["b0_", "b1_", "s0_", "s1_", "pf_", "pm_"])

        if running:
            n_prev = 0 if (reset or self._running_ is None) else self._running_["n_samples"]
            ctime = (n_prev + X.shape[2]) / self.fs
        else:
            ctime = X.shape[2] / self.fs

        if self.min_time is not None and ctime <= self.min_time:
            if running:
                _running_predict(self, X, reset, use_decision_function=True)  # advance state, result unused
            yh = np.full(X.shape[0], -1, dtype="int64")

        elif self.max_time is not None and ctime >= self.max_time:
            if running:
                yh = _running_predict(self, X, reset, use_decision_function=False)
            else:
                yh = self.estimator.predict(X)

        else:
            i_segment = int(np.round(ctime / self.segment_time)) - 1
            i_segment = np.max([0, i_segment])  # lower bound 0

            # Compute the scores
            if running:
                scores = _running_predict(self, X, reset, use_decision_function=True)
            else:
                scores = self.estimator.decision_function(X)

            # Check if stopped
            if self.method == "bds0":
                # Stop if eta threshold of this segment is reached
                not_stopped = np.max(scores, axis=1) <= self.eta_[i_segment]

            elif self.method == "bds1":
                # Change target pf/pd with min/max of learned pf/pd (locally, for this prediction only)
                target_pf, target_pd = self.target_pf, self.target_pd
                if np.min(self.pf_) > target_pf:
                    target_pf = np.min(self.pf_)
                    warnings.warn(f"target_pf is not reachable, using {target_pf:.3} for this prediction instead.")
                if np.min(self.pm_) > 1 - target_pd:
                    target_pd = 1 - np.min(self.pm_)
                    warnings.warn(f"target_pd is not reachable, using {target_pd:.3} for this prediction instead.")

                # Stop if eta threshold is reached, and if both pf and pd targets are reached
                c1 = np.max(scores, axis=1) <= self.eta_[i_segment]
                c2 = self.pf_[i_segment] >= target_pf
                c3 = self.pm_[i_segment] >= (1 - target_pd)
                not_stopped = np.logical_or(np.logical_or(c1, c2), c3)

            elif self.method == "bds2":
                # Change target pf/pd with min/max of learned pf/pd (locally, for this prediction only)
                target_pf, target_pd = self.target_pf, self.target_pd
                if np.min(self.pf_) > target_pf:
                    target_pf = np.min(self.pf_)
                    warnings.warn(f"target_pf is not reachable, using {target_pf:.3} for this prediction instead.")
                if np.min(self.pm_) > 1 - target_pd:
                    target_pd = 1 - np.min(self.pm_)
                    warnings.warn(f"target_pd is not reachable, using {target_pd:.3} for this prediction instead.")

                # Find intersection of target pf and pm to find "optimal" eta
                idx_pf = np.where(self.pf_ <= target_pf)[0]
                idx_pm = np.where(self.pm_ <= 1 - target_pd)[0]
                idx = np.intersect1d(idx_pf, idx_pm)
                # Fall back to the most conservative (last) segment if no segment satisfies both targets at once
                eta = self.eta_[idx[0]] if idx.size > 0 else self.eta_[-1]

                # Stop if "optimal" eta is reached
                not_stopped = np.max(scores, axis=1) <= eta

            else:
                raise Exception("Unknown method:", self.method)

            # Classify and set not-stopped-trials to -1
            yh = np.argmax(scores, axis=1)
            yh[not_stopped] = -1

        return yh


class CriterionStopping(ClassifierMixin, BaseEstimator):
    """Criterion static stopping. Fits an optimal stopping time given some criterion to optimize.

    Parameters
    ----------
    estimator: ClassifierMixin
        The classifier object that performs the classification.
    segment_time: float
        The size of a segment of data at which classification is performed in seconds.
    fs: int
        The sampling frequency of the EEG data in Hz.
    criterion: str (default: "accuracy")
        The criterion to use: accuracy, itr.
    optimization: str (default: "max")
        The optimization to use: max, target.
    n_folds: int (default: 4)
        The number of folds to evaluate the optimization.
    target: float (default: None)
        The targeted value for the criterion to optimize for. Only used if optimization="target".
    smooth_width: float (default: None)
        The width of the smoothing applied in seconds. If None, the values of the criterion are not smoothened.
    max_time: float (default: None)
        The maximum time in seconds at which to force a stop, i.e., a classification. Trials will not be longer than
        this maximum time. If None, the algorithm will always emit -1 if it cannot stop.
    min_time: float (default: None)
        The minimum time in seconds at which a stop is possible, i.e., a classification. Before the minimum time, the
        algorithm will always emit -1. If None, the algorithm allows a stop already after the first segment of data.

    Attributes
    ----------
    classes_: NDArray
        The classes that can be predicted, taken from the wrapped estimator's classes_ after fitting (i.e., after the
        internal cross-validation, fit on the last fold, matching what predict() uses). Note, predict() may
        additionally return -1 to indicate a trial has not yet been stopped, which is not itself a class.
    stop_time_: float
        The trained static stopping time.
    """

    classes_: NDArray
    stop_time_: float
    _running_: dict = None

    def __init__(
        self,
        estimator: ClassifierMixin,
        segment_time: float,
        fs: int,
        criterion: str = "accuracy",
        optimization: str = "max",
        n_folds: int = 4,
        target: float = None,
        smooth_width: float = None,
        max_time: float = None,
        min_time: float = None,
    ) -> None:
        self.estimator = estimator
        self.segment_time = segment_time
        self.fs = fs
        self.criterion = criterion
        self.optimization = optimization
        self.n_folds = n_folds
        self.target = target
        self.smooth_width = smooth_width
        self.max_time = max_time
        self.min_time = min_time

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> ClassifierMixin:
        """The training procedure to fit the static procedure on supervised EEG data.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: NDArray
            The vector of ground-truth labels of the trials in X of shape (n_trials).

        Returns
        -------
        self: ClassifierMixin
            Returns the instance itself.
        """
        n_trials = X.shape[0]
        n_samples = X.shape[2]
        n_segments = int(n_samples / int(self.segment_time * self.fs))

        assert n_trials >= self.n_folds, "n_trials must be at least n_folds, otherwise some folds have no test data."

        folds = np.arange(self.n_folds).repeat(np.ceil(n_trials / self.n_folds))[:n_trials]
        scores = np.zeros((self.n_folds, n_segments), dtype="float32")

        for i_fold in range(self.n_folds):
            X_trn = X[folds != i_fold, :, :]
            X_tst = X[folds == i_fold, :, :]
            y_trn = y[folds != i_fold]
            y_tst = y[folds == i_fold]

            # Fit estimator
            self.estimator.fit(X_trn, y_trn)

            segment_samples = int(self.segment_time * self.fs)
            for i_segment, yh in _iter_segment_predictions(self.estimator, X_tst, segment_samples, n_segments):
                # Compute criterion
                if self.criterion.lower() == "accuracy":
                    scores[i_fold, i_segment] = np.mean(yh == y_tst)
                elif self.criterion.lower() == "itr":
                    # Note, the number of classes does not affect the optimum
                    scores[i_fold, i_segment] = itr(10, np.mean(yh == y_tst), (1 + i_segment) * self.segment_time)
                else:
                    raise Exception("Unknown criterion:", self.criterion)

        # Smoothen
        if self.smooth_width is not None:
            width = int(self.smooth_width / self.segment_time)
            kernel = np.full(width, 1 / width)
            for i_fold in range(self.n_folds):
                scores[i_fold, :] = np.convolve(scores[i_fold, :], kernel, mode="same")

        # Average folds
        scores = scores.mean(axis=0)

        # Optimize the criterion
        if self.optimization.lower() == "max":
            self.stop_time_ = (1 + np.argmax(scores)) * self.segment_time
        elif self.optimization.lower() == "target":
            if self.target is None:
                raise Exception("For optimization target one should set the target")
            if np.any(scores >= self.target):
                self.stop_time_ = (1 + np.argmax(scores >= self.target)) * self.segment_time
            else:
                self.stop_time_ = X.shape[2] / self.fs
        else:
            raise Exception("Unknown optimization:", self.optimization)

        self.classes_ = self.estimator.classes_
        self._running_ = None
        return self

    def predict(
        self,
        X: NDArray,
        running: bool = False,
        reset: bool = False,
    ) -> NDArray:
        """The testing procedure to apply the estimator to novel EEG data using criterion static stopping.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples). If running=True, this is only the
            newly observed samples since the previous call (not the full trial so far), see running below.
        running: bool (default: False)
            Whether to use running (incremental) scoring. If False (default), predict() behaves exactly as without
            this parameter: X is the complete trial data seen so far, and everything is recomputed from scratch
            (safe to call in any order, e.g. repeatedly with the same or a shorter X). If True, X is only the
            newly observed samples since the previous call, and a running state (kept internally, not a fitted
            attribute) is reused and updated; if the wrapped estimator supports running scoring itself (an eCCA or
            rCCA with ensemble=False), each call only does O(n_new_samples) work instead of reprocessing the whole
            trial, otherwise the raw data is buffered here and recomputed from scratch each call (still correct,
            just not faster). Use reset=True on the first call of a new running sequence (e.g. for a new trial or
            a new batch of trials); the running state is otherwise unaffected by (and does not affect) running=False
            calls, and is cleared by fit().
        reset: bool (default: False)
            Whether to discard any existing running state before processing this call. Only relevant if
            running=True; a never-yet-used instance already starts fresh without it, so it only needs to be set
            explicitly to start a new sequence before the previous one naturally ended.

        Returns
        -------
        y: NDArray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """
        check_is_fitted(self, ["stop_time_"])

        if running:
            n_prev = 0 if (reset or self._running_ is None) else self._running_["n_samples"]
            ctime = (n_prev + X.shape[2]) / self.fs
        else:
            ctime = X.shape[2] / self.fs

        if self.min_time is not None and ctime <= self.min_time:
            if running:
                _running_predict(self, X, reset, use_decision_function=False)  # advance state, result unused
            yh = np.full(X.shape[0], -1, dtype="int64")

        elif (self.max_time is not None and ctime >= self.max_time) or ctime >= self.stop_time_:
            if running:
                yh = _running_predict(self, X, reset, use_decision_function=False)
            else:
                yh = self.estimator.predict(X)

        else:
            if running:
                _running_predict(self, X, reset, use_decision_function=False)  # advance state, result unused
            yh = np.full(X.shape[0], -1, dtype="int64")

        return yh


class DistributionStopping(ClassifierMixin, BaseEstimator):
    """Distribution dynamic stopping. Fits a distribution to non-target / non-maximum scores, and tests the probability
    of the target / maximum score to be an outlier of that distribution [2]_.

    Parameters
    ----------
    estimator: ClassifierMixin
        The classifier object that performs the classification.
    segment_time: float
        The size of a segment of data at which classification is performed in seconds.
    fs: int
        The sampling frequency of the EEG data in Hz.
    trained: bool (default: False)
        Whether to calibrate the beta distributions on training data.
    distribution: str (default: "beta")
        The distribution to use for the non-target / non-maximum distribution. Either beta or norm.
    target_p: float (default: 0.95)
        The targeted probability of correct classification.
    max_time: float (default: None)
        The maximum time in seconds at which to force a stop, i.e., a classification. Trials will not be longer than
        this maximum time. If None, the algorithm will always emit -1 if it cannot stop.
    min_time: float (default: None)
        The minimum time in seconds at which a stop is possible, i.e., a classification. Before the minimum time, the
        algorithm will always emit -1. If None, the algorithm allows a stop already after the first segment of data.

    Attributes
    ----------
    classes_: NDArray
        The classes that can be predicted, taken from the wrapped estimator's classes_ after fitting. Note, predict()
        may additionally return -1 to indicate a trial has not yet been stopped, which is not itself a class.
    distributions_: list[dict]
        A list of dictionaries containing the parameters of the distribution for each data segment. Only used if
        trained=True.

    References
    ----------
    .. [2] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
           code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
           056007. doi: 10.1088/1741-2552/abecef
    """

    classes_: NDArray
    distributions_: list[dict]
    _running_: dict = None

    def __init__(
        self,
        estimator: ClassifierMixin,
        segment_time: float,
        fs: int,
        trained: bool = False,
        distribution: str = "beta",
        target_p: float = 0.95,
        max_time: float = None,
        min_time: float = None,
    ) -> None:
        self.estimator = estimator
        self.segment_time = segment_time
        self.fs = fs
        self.trained = trained
        self.distribution = distribution
        self.target_p = target_p
        self.max_time = max_time
        self.min_time = min_time

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> ClassifierMixin:
        """The training procedure to fit the dynamic procedure on supervised EEG data.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: NDArray
            The vector of ground-truth labels of the trials in X of shape (n_trials).

        Returns
        -------
        self: ClassifierMixin
            Returns the instance itself.
        """
        assert self.distribution in ["beta", "norm"], "Distribution must be beta or norm."

        # Fit estimator
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_

        # Fit beta distributions
        if self.trained:
            self.distributions_ = []
            n_segments = int(X.shape[2] / int(self.segment_time * self.fs))
            segment_samples = int(self.segment_time * self.fs)
            for i_segment, scores in _iter_segment_scores(self.estimator, X, segment_samples, n_segments):
                # Put target score at index 0
                for i_trial in range(X.shape[0]):
                    scores[i_trial, [0, y[i_trial]]] = scores[i_trial, [y[i_trial], 0]]

                # Fit distribution to non-target scores
                if self.distribution == "beta":
                    a, b, loc, scale = beta.fit(scores[:, 1:].flatten(), floc=-1, fscale=2)
                    self.distributions_.append(dict(a=a, b=b, loc=loc, scale=scale))
                elif self.distribution == "norm":
                    loc, scale = norm.fit(scores[:, 1:].flatten())
                    self.distributions_.append(dict(loc=loc, scale=scale))

        self._is_fitted = True
        self._running_ = None
        return self

    def predict(
        self,
        X: NDArray,
        running: bool = False,
        reset: bool = False,
    ) -> NDArray:
        """The testing procedure to apply the estimator to novel EEG data using distribution dynamic stopping.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples). If running=True, this is only the
            newly observed samples since the previous call (not the full trial so far), see running below.
        running: bool (default: False)
            Whether to use running (incremental) scoring. If False (default), predict() behaves exactly as without
            this parameter: X is the complete trial data seen so far, and everything is recomputed from scratch
            (safe to call in any order, e.g. repeatedly with the same or a shorter X). If True, X is only the
            newly observed samples since the previous call, and a running state (kept internally, not a fitted
            attribute) is reused and updated; if the wrapped estimator supports running scoring itself (an eCCA or
            rCCA with ensemble=False), each call only does O(n_new_samples) work instead of reprocessing the whole
            trial, otherwise the raw data is buffered here and recomputed from scratch each call (still correct,
            just not faster). Use reset=True on the first call of a new running sequence (e.g. for a new trial or
            a new batch of trials); the running state is otherwise unaffected by (and does not affect) running=False
            calls, and is cleared by fit().
        reset: bool (default: False)
            Whether to discard any existing running state before processing this call. Only relevant if
            running=True; a never-yet-used instance already starts fresh without it, so it only needs to be set
            explicitly to start a new sequence before the previous one naturally ended.

        Returns
        -------
        y: NDArray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """
        check_is_fitted(self)

        if running:
            n_prev = 0 if (reset or self._running_ is None) else self._running_["n_samples"]
            ctime = (n_prev + X.shape[2]) / self.fs
        else:
            ctime = X.shape[2] / self.fs

        if self.min_time is not None and ctime <= self.min_time:
            if running:
                _running_predict(self, X, reset, use_decision_function=True)  # advance state, result unused
            yh = np.full(X.shape[0], -1, dtype="int64")

        elif self.max_time is not None and ctime >= self.max_time:
            if running:
                yh = _running_predict(self, X, reset, use_decision_function=False)
            else:
                yh = self.estimator.predict(X)

        else:
            i_segment = int(np.round(ctime / self.segment_time)) - 1
            i_segment = np.max([0, i_segment])  # lower bound 0

            # Compute the scores
            if running:
                scores = _running_predict(self, X, reset, use_decision_function=True)
            else:
                scores = self.estimator.decision_function(X)

            # Sort the scores (ascending)
            scores_sorted = np.sort(scores, axis=1)

            # Calculate probability of maximum score being a "true" maximum
            p = np.zeros(scores.shape[0])
            for i_trial in range(scores.shape[0]):
                if self.trained:
                    # Look-up pre-fit distribution parameters
                    if self.distribution == "beta":
                        a, b, loc, scale = [self.distributions_[i_segment][key] for key in ["a", "b", "loc", "scale"]]
                    elif self.distribution == "norm":
                        loc, scale = [self.distributions_[i_segment][key] for key in ["loc", "scale"]]
                else:
                    # Fit distribution to current non-max scores
                    try:
                        if self.distribution == "beta":
                            a, b, loc, scale = beta.fit(scores_sorted[i_trial, :-1], floc=-1, fscale=2)
                        elif self.distribution == "norm":
                            loc, scale = norm.fit(scores_sorted[i_trial, :-1])
                    except Exception:
                        p[i_trial] = 0.0
                        continue

                # Look up probability of maximum score in distribution
                if self.distribution == "beta":
                    p[i_trial] = beta.cdf(scores_sorted[i_trial, -1], a, b, loc, scale) ** scores.shape[1]
                elif self.distribution == "norm":
                    p[i_trial] = norm.cdf(scores_sorted[i_trial, -1], loc, scale) ** scores.shape[1]

            # Check if stopped
            not_stopped = p <= self.target_p

            # Classify and set not-stopped-trials to -1
            yh = np.argmax(scores, axis=1)
            yh[not_stopped] = -1

        return yh

    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status and return a Boolean value.

        Returns
        -------
        fitted: bool
            Whether the classifier is fitted.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


class MarginStopping(ClassifierMixin, BaseEstimator):
    """Margin dynamic stopping. Learns threshold margins (difference between best and second-best score) to stop at
    such that a targeted accuracy is reached [3]_.

    Parameters
    ----------
    estimator: ClassifierMixin
        The classifier object that performs the classification.
    segment_time: float
        The size of a segment of data at which classification is performed in seconds.
    fs: int
        The sampling frequency of the EEG data in Hz.
    target_p: float (default: 0.95)
        The targeted probability of correct classification.
    margin_min: float (default: 0.0)
        The minimum value for the possible threshold margin to stop at.
    margin_max: float (default: 1.0)
        The maximum value for the possible threshold margin to stop at.
    margin_step: float (default: 0.05)
        The step size defining the resolution of the threshold margins at which to stop.
    max_time: float (default: None)
        The maximum time in seconds at which to force a stop, i.e., a classification. Trials will not be longer than
        this maximum time. If None, the algorithm will always emit -1 if it cannot stop.
    min_time: float (default: None)
        The minimum time in seconds at which a stop is possible, i.e., a classification. Before the minimum time, the
        algorithm will always emit -1. If None, the algorithm allows a stop already after the first segment of data.

    Attributes
    ----------
    classes_: NDArray
        The classes that can be predicted, taken from the wrapped estimator's classes_ after fitting. Note, predict()
        may additionally return -1 to indicate a trial has not yet been stopped, which is not itself a class.
    margins_: NDArray
        The trained stopping margins of shape (n_segments).

    References
    ----------
    .. [3] Thielen, J., van den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials:
           re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797. doi: 10.1371/journal.pone.0133797
    """

    classes_: NDArray
    margins_: NDArray
    _running_: dict = None

    def __init__(
        self,
        estimator: ClassifierMixin,
        segment_time: float,
        fs: int,
        target_p: float = 0.95,
        margin_min: float = 0.0,
        margin_max: float = 1.0,
        margin_step: float = 0.05,
        max_time: float = None,
        min_time: float = None,
    ) -> None:
        self.estimator = estimator
        self.segment_time = segment_time
        self.fs = fs
        self.target_p = target_p
        self.margin_min = margin_min
        self.margin_max = margin_max
        self.margin_step = margin_step
        self.max_time = max_time
        self.min_time = min_time

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> ClassifierMixin:
        """The training procedure to fit the dynamic procedure on supervised EEG data.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: NDArray
            The vector of ground-truth labels of the trials in X of shape (n_trials).

        Returns
        -------
        self: ClassifierMixin
            Returns the instance itself.
        """
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_

        # Set margin axis (possible margins to stop at)
        margin_axis = np.arange(self.margin_min, self.margin_max, self.margin_step)

        # Calculate a margin per segment
        n_samples = X.shape[2]
        n_segments = int(n_samples / int(self.segment_time * self.fs))
        self.margins_ = np.zeros(n_segments)
        segment_samples = int(self.segment_time * self.fs)
        for i_segment, scores in _iter_segment_scores(self.estimator, X, segment_samples, n_segments):
            # Compute margins (best - second best)
            scores_sorted = np.sort(scores, axis=1)
            margins = scores_sorted[:, -1] - scores_sorted[:, -2]

            # Compute correctness (best is label)
            correct = np.argmax(scores, axis=1) == y

            # Compute histograms
            wrong_hist = np.histogram(margins[~correct], margin_axis)[0]
            right_hist = np.histogram(margins[correct], margin_axis)[0]

            # Reverse cumulative (how many stop is margin>x)
            wrong_hist = np.cumsum(wrong_hist[::-1])[::-1]
            right_hist = np.cumsum(right_hist[::-1])[::-1]

            # Compute accuracy (plus eps to prevent division by zero)
            accuracy = right_hist / (wrong_hist + right_hist + np.finfo("float").eps)

            # Select margin that makes the accuracy reach the targeted accuracy
            idx = np.where(accuracy >= self.target_p)[0]
            if len(idx) == 0:
                self.margins_[i_segment] = margin_axis[-1]
            else:
                self.margins_[i_segment] = margin_axis[idx[0]]

        self._running_ = None
        return self

    def predict(
        self,
        X: NDArray,
        running: bool = False,
        reset: bool = False,
    ) -> NDArray:
        """The testing procedure to apply the estimator to novel EEG data using margin dynamic stopping.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples). If running=True, this is only the
            newly observed samples since the previous call (not the full trial so far), see running below.
        running: bool (default: False)
            Whether to use running (incremental) scoring. If False (default), predict() behaves exactly as without
            this parameter: X is the complete trial data seen so far, and everything is recomputed from scratch
            (safe to call in any order, e.g. repeatedly with the same or a shorter X). If True, X is only the
            newly observed samples since the previous call, and a running state (kept internally, not a fitted
            attribute) is reused and updated; if the wrapped estimator supports running scoring itself (an eCCA or
            rCCA with ensemble=False), each call only does O(n_new_samples) work instead of reprocessing the whole
            trial, otherwise the raw data is buffered here and recomputed from scratch each call (still correct,
            just not faster). Use reset=True on the first call of a new running sequence (e.g. for a new trial or
            a new batch of trials); the running state is otherwise unaffected by (and does not affect) running=False
            calls, and is cleared by fit().
        reset: bool (default: False)
            Whether to discard any existing running state before processing this call. Only relevant if
            running=True; a never-yet-used instance already starts fresh without it, so it only needs to be set
            explicitly to start a new sequence before the previous one naturally ended.

        Returns
        -------
        y: NDArray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """
        check_is_fitted(self, ["margins_"])

        if running:
            n_prev = 0 if (reset or self._running_ is None) else self._running_["n_samples"]
            ctime = (n_prev + X.shape[2]) / self.fs
        else:
            ctime = X.shape[2] / self.fs

        if self.min_time is not None and ctime <= self.min_time:
            if running:
                _running_predict(self, X, reset, use_decision_function=True)  # advance state, result unused
            yh = np.full(X.shape[0], -1, dtype="int64")

        elif self.max_time is not None and ctime >= self.max_time:
            if running:
                yh = _running_predict(self, X, reset, use_decision_function=False)
            else:
                yh = self.estimator.predict(X)

        else:
            i_segment = int(np.round(ctime / self.segment_time)) - 1
            i_segment = np.max([0, i_segment])  # lower bound 0

            # Compute the scores
            if running:
                scores = _running_predict(self, X, reset, use_decision_function=True)
            else:
                scores = self.estimator.decision_function(X)

            # Sort the scores (ascending)
            scores_sorted = np.sort(scores, axis=1)

            # Compute margins (best minus second best)
            margins = scores_sorted[:, -1] - scores_sorted[:, -2]

            # Check stopped
            not_stopped = margins <= self.margins_[i_segment]

            # Classify and set not-stopped-trials to -1
            yh = np.argmax(scores, axis=1)
            yh[not_stopped] = -1

        return yh


class ValueStopping(ClassifierMixin, BaseEstimator):
    """Value dynamic stopping. Learns threshold values to stop at such that a targeted accuracy is reached.

    Parameters
    ----------
    estimator: ClassifierMixin
        The classifier object that performs the classification.
    segment_time: float
        The size of a segment of data at which classification is performed in seconds.
    fs: int
        The sampling frequency of the EEG data in Hz.
    target_p: float (default: 0.95)
        The targeted probability of correct classification.
    value_min: float (default: 0.0)
        The minimum value for the possible threshold value to stop at.
    value_max: float (default: 1.0)
        The maximum value for the possible threshold value to stop at.
    value_step: float (default: 0.05)
        The step size defining the resolution of the threshold values at which to stop.
    max_time: float (default: None)
        The maximum time in seconds at which to force a stop, i.e., a classification. Trials will not be longer than
        this maximum time. If None, the algorithm will always emit -1 if it cannot stop.
    min_time: float (default: None)
        The minimum time in seconds at which a stop is possible, i.e., a classification. Before the minimum time, the
        algorithm will always emit -1. If None, the algorithm allows a stop already after the first segment of data.

    Attributes
    ----------
    classes_: NDArray
        The classes that can be predicted, taken from the wrapped estimator's classes_ after fitting. Note, predict()
        may additionally return -1 to indicate a trial has not yet been stopped, which is not itself a class.
    values_: NDArray
        The trained stopping values of shape (n_segments).
    """

    classes_: NDArray
    values_: NDArray
    _running_: dict = None

    def __init__(
        self,
        estimator: ClassifierMixin,
        segment_time: float,
        fs: int,
        target_p: float = 0.95,
        value_min: float = 0.0,
        value_max: float = 1.0,
        value_step: float = 0.05,
        max_time: float = None,
        min_time: float = None,
    ) -> None:
        self.estimator = estimator
        self.segment_time = segment_time
        self.fs = fs
        self.target_p = target_p
        self.value_min = value_min
        self.value_max = value_max
        self.value_step = value_step
        self.max_time = max_time
        self.min_time = min_time

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> ClassifierMixin:
        """The training procedure to fit the dynamic procedure on supervised EEG data.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: NDArray
            The vector of ground-truth labels of the trials in X of shape (n_trials).

        Returns
        -------
        self: ClassifierMixin
            Returns the instance itself.
        """
        # Set value axis (possible values to stop at)
        value_axis = np.arange(self.value_min, self.value_max, self.value_step)

        # Fit estimator
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_

        # Calculate a value per segment
        n_samples = X.shape[2]
        n_segments = int(n_samples / int(self.segment_time * self.fs))
        self.values_ = np.zeros(n_segments)
        segment_samples = int(self.segment_time * self.fs)
        for i_segment, scores in _iter_segment_scores(self.estimator, X, segment_samples, n_segments):
            # Compute values
            values = np.max(scores, axis=1)

            # Compute correctness (max is label)
            correct = np.argmax(scores, axis=1) == y

            # Compute histograms
            wrong_hist = np.histogram(values[~correct], value_axis)[0]
            right_hist = np.histogram(values[correct], value_axis)[0]

            # Reverse cumulative (how many stop is margin>x)
            wrong_hist = np.cumsum(wrong_hist[::-1])[::-1]
            right_hist = np.cumsum(right_hist[::-1])[::-1]

            # Compute accuracy (plus eps to prevent division by zero)
            accuracy = right_hist / (wrong_hist + right_hist + np.finfo("float").eps)

            # Select margin that makes the accuracy reach the targeted accuracy
            idx = np.where(accuracy >= self.target_p)[0]
            if len(idx) == 0:
                self.values_[i_segment] = value_axis[-1]
            else:
                self.values_[i_segment] = value_axis[idx[0]]

        self._running_ = None
        return self

    def predict(
        self,
        X: NDArray,
        running: bool = False,
        reset: bool = False,
    ) -> NDArray:
        """The testing procedure to apply the estimator to novel EEG data using value dynamic stopping.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples). If running=True, this is only the
            newly observed samples since the previous call (not the full trial so far), see running below.
        running: bool (default: False)
            Whether to use running (incremental) scoring. If False (default), predict() behaves exactly as without
            this parameter: X is the complete trial data seen so far, and everything is recomputed from scratch
            (safe to call in any order, e.g. repeatedly with the same or a shorter X). If True, X is only the
            newly observed samples since the previous call, and a running state (kept internally, not a fitted
            attribute) is reused and updated; if the wrapped estimator supports running scoring itself (an eCCA or
            rCCA with ensemble=False), each call only does O(n_new_samples) work instead of reprocessing the whole
            trial, otherwise the raw data is buffered here and recomputed from scratch each call (still correct,
            just not faster). Use reset=True on the first call of a new running sequence (e.g. for a new trial or
            a new batch of trials); the running state is otherwise unaffected by (and does not affect) running=False
            calls, and is cleared by fit().
        reset: bool (default: False)
            Whether to discard any existing running state before processing this call. Only relevant if
            running=True; a never-yet-used instance already starts fresh without it, so it only needs to be set
            explicitly to start a new sequence before the previous one naturally ended.

        Returns
        -------
        y: NDArray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """
        check_is_fitted(self, ["values_"])

        if running:
            n_prev = 0 if (reset or self._running_ is None) else self._running_["n_samples"]
            ctime = (n_prev + X.shape[2]) / self.fs
        else:
            ctime = X.shape[2] / self.fs

        if self.min_time is not None and ctime <= self.min_time:
            if running:
                _running_predict(self, X, reset, use_decision_function=True)  # advance state, result unused
            yh = np.full(X.shape[0], -1, dtype="int64")

        elif self.max_time is not None and ctime >= self.max_time:
            if running:
                yh = _running_predict(self, X, reset, use_decision_function=False)
            else:
                yh = self.estimator.predict(X)

        else:
            i_segment = int(np.round(ctime / self.segment_time)) - 1
            i_segment = np.max([0, i_segment])  # lower bound 0

            # Compute the scores
            if running:
                scores = _running_predict(self, X, reset, use_decision_function=True)
            else:
                scores = self.estimator.decision_function(X)

            # Compute values
            values = np.max(scores, axis=1)

            # Check stopped
            not_stopped = values <= self.values_[i_segment]

            # Classify and set not-stopped-trials to -1
            yh = np.argmax(scores, axis=1)
            yh[not_stopped] = -1

        return yh
