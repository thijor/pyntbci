import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta, norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

import pyntbci.classifiers
from pyntbci.utilities import itr


class BayesStopping(BaseEstimator, ClassifierMixin):
    """Bayesian dynamic stopping. Fits Gaussian distributions for target and non-target responses, and calculates a
    stopping threshold using these and a cost criterion [1]_.

    Parameters
    ----------
    estimator: ClassifierMixin
        The classifier object that performs the classification.
    segment_time: float
        The size of a segment of data at which classification is performed ins seconds.
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
    alpha_: float
    sigma_: float
    b0_: NDArray
    b1_: NDArray
    s0_: NDArray
    s1_: NDArray
    eta_: NDArray
    pf_: NDArray
    pm_: NDArray

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

        if self.approach == "template_inner":
            self._fit_template_inner(X, y)
        else:
            self._fit_score(X, y)

        return self

    def _fit_template_inner(self, X, y):
        assert isinstance(self.estimator, pyntbci.classifiers.rCCA), "Approach template_inner works only for rCCA."
        n_samples = X.shape[2]
        n_classes = self.estimator.Ts_.shape[0]

        # Spatially filter data and flatten
        X = self.estimator.cca_[0].transform(X=X)[0].reshape((-1, 1))

        # Get templates
        if n_samples < self.estimator.Ts_.shape[2]:
            T = self.estimator.Ts_
        else:
            T = np.concatenate((self.estimator.Ts_, np.tile(self.estimator.Tw_, (1, 1, n_samples // self.estimator.Tw_.shape[2]))), axis=2)
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
            self.s0_[i_segment] = np.sqrt(self.b1_[i_segment] * self.sigma_ ** 2 +
                                          np.mean((self.alpha_ * inner_xy - self.alpha_ * self.b0_[i_segment]) ** 2))
            self.s1_[i_segment] = np.sqrt(self.b1_[i_segment] * self.sigma_ ** 2 +
                                          np.mean((self.alpha_ * inner_xx - self.alpha_ * self.b1_[i_segment]) ** 2))

        # Calculate eta
        a = self.s1_ ** 2 - self.s0_ ** 2
        b = -2 * self.alpha_ * (self.s1_ ** 2 * self.b0_ - self.s0_ ** 2 * self.b1_)
        c = -self.alpha_ ** 2 * (self.s1_ ** 2 * self.b0_ ** 2 + self.s0_ ** 2 * self.b1_ ** 2) + \
            2 * self.s0_ ** 2 * self.s1_ ** 2 * np.log(self.s0_ / (self.s1_ * (n_classes - 1) * self.cr))
        self.eta_ = (-b + np.sqrt(np.clip(b ** 2 - 4 * a * c, 0, None))) / (2 * a)

        # Calculate predicted error vectors (corrected for multiple comparisons)
        self.pf_ = ((n_classes - 1) / n_classes) * (1 - norm.cdf(self.eta_, self.alpha_ * self.b0_, self.s0_))
        self.pf_ = 1 - (1 - self.pf_) ** n_classes
        self.pm_ = (1 / n_classes) * norm.cdf(self.eta_, self.alpha_ * self.b1_, self.s1_)
        self.pm_ = 1 - (1 - self.pm_) ** n_classes

    def _fit_score(self, X, y):
        n_samples = X.shape[2]

        # Calculate b0, b1, s0, s1
        n_segments = int(n_samples / int(self.segment_time * self.fs))
        self.b0_ = np.zeros(n_segments)
        self.b1_ = np.zeros(n_segments)
        self.s0_ = np.zeros(n_segments)
        self.s1_ = np.zeros(n_segments)
        for i_segment in range(n_segments):
            idx = (1 + i_segment) * int(self.segment_time * self.fs)
            scores = self.estimator.decision_function(X[:, :, :idx])
            n_classes = scores.shape[1]
            mask = np.full(scores.shape, False)
            mask[np.arange(y.size), y] = True
            self.b0_[i_segment] = scores[~mask].mean()
            self.b1_[i_segment] = scores[mask].mean()
            self.s0_[i_segment] = scores[~mask].std()
            self.s1_[i_segment] = scores[mask].std()

        # Calculate eta
        a = self.s1_ ** 2 - self.s0_ ** 2
        b = -2 * (self.s1_ ** 2 * self.b0_ - self.s0_ ** 2 * self.b1_)
        c = -(self.s1_ ** 2 * self.b0_ ** 2 + self.s0_ ** 2 * self.b1_ ** 2) + \
            2 * self.s0_ ** 2 * self.s1_ ** 2 * np.log(self.s0_ / (self.s1_ * (n_classes - 1) * self.cr))
        self.eta_ = (-b + np.sqrt(np.clip(b ** 2 - 4 * a * c, 0, None))) / (2 * a)

        # Calculate predicted error vectors (corrected for multiple comparisons)
        self.pf_ = ((n_classes - 1) / n_classes) * (1 - norm.cdf(self.eta_, self.b0_, self.s0_))
        self.pf_ = 1 - (1 - self.pf_) ** n_classes
        self.pm_ = (1 / n_classes) * norm.cdf(self.eta_, self.b1_, self.s1_)
        self.pm_ = 1 - (1 - self.pm_) ** n_classes

    def predict(
            self,
            X: NDArray,
    ) -> NDArray:
        """The testing procedure to apply the estimator to novel EEG data using Bayesian dynamic stopping.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: NDArray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """
        check_is_fitted(self, ["alpha_", "sigma_", "b0_", "b1_", "s0_", "s1_", "pf_", "pm_"])

        ctime = X.shape[2] / self.fs

        if self.min_time is not None and ctime <= self.min_time:
            yh = np.full(X.shape[0], -1, dtype="int64")

        elif self.max_time is not None and ctime >= self.max_time:
            yh = self.estimator.predict(X)

        else:
            i_segment = int(np.round(ctime / self.segment_time)) - 1
            i_segment = np.max([0, i_segment])  # lower bound 0

            # Compute the scores
            scores = self.estimator.decision_function(X)

            # Check if stopped
            if self.method == "bds0":
                # Stop if eta threshold of this segment is reached
                not_stopped = np.max(scores, axis=1) <= self.eta_[i_segment]

            elif self.method == "bds1":
                # Change target pf/pd with min/max of learned pf/pd
                if np.min(self.pf_) > self.target_pf:
                    self.target_pf = np.min(self.pf_)
                    print(f"Warning: changed target_pf to {self.target_pf:.3}")
                if np.min(self.pm_) > 1 - self.target_pd:
                    self.target_pd = 1 - np.min(self.pm_)
                    print(f"Warning: changed target_pd to {self.target_pd:.3}")

                # Stop if eta threshold is reached, and if both pf and pd targets are reached
                c1 = np.max(scores, axis=1) <= self.eta_[i_segment]
                c2 = self.pf_[i_segment] >= self.target_pf
                c3 = self.pm_[i_segment] >= (1 - self.target_pd)
                not_stopped = np.logical_or(np.logical_or(c1, c2), c3)

            elif self.method == "bds2":
                # Change target pf/pd with min/max of learned pf/pd
                if np.min(self.pf_) > self.target_pf:
                    self.target_pf = np.min(self.pf_)
                    print(f"Warning: changed target_pf to {self.target_pf:.3}")
                if np.min(self.pm_) > 1 - self.target_pd:
                    self.target_pd = 1 - np.min(self.pm_)
                    print(f"Warning: changed target_pd to {self.target_pd:.3}")

                # Find intersection of target pf and pm to find "optimal" eta
                idx_pf = np.where(self.pf_ <= self.target_pf)[0]
                idx_pm = np.where(self.pm_ <= 1 - self.target_pd)[0]
                idx = np.intersect1d(idx_pf, idx_pm)
                eta = self.eta_[idx[0]]

                # Stop if "optimal" eta is reached
                not_stopped = np.max(scores, axis=1) <= eta

            else:
                raise Exception("Unknown method:", self.method)

            # Classify and set not-stopped-trials to -1
            yh = np.argmax(scores, axis=1)
            yh[not_stopped] = -1

        return yh


class CriterionStopping(BaseEstimator, ClassifierMixin):
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
    n_folds: int (n_folds: 4)
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
    stop_time_: float
        The trained static stopping time.
    """
    stop_time_: float

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

        folds = np.arange(self.n_folds).repeat(np.ceil(n_trials / self.n_folds))[:n_trials]
        scores = np.zeros((self.n_folds, n_segments))

        for i_fold in range(self.n_folds):

            X_trn = X[folds != i_fold, :, :]
            X_tst = X[folds == i_fold, :, :]
            y_trn = y[folds != i_fold]
            y_tst = y[folds == i_fold]

            # Fit estimator
            self.estimator.fit(X_trn, y_trn)

            for i_segment in range(n_segments):

                # Predict labels for this segment
                idx = (1 + i_segment) * int(self.segment_time * self.fs)
                yh = self.estimator.predict(X_tst[:, :, :idx])

                # Compute criterion
                if self.criterion == "accuracy":
                    scores[i_fold, i_segment] = np.mean(yh == y_tst)
                elif self.criterion == "itr":
                    # Note, the number of classes does not affect the optimum
                    scores[i_fold, i_segment] = itr(10, np.mean(yh == y_tst), (1 + i_segment) * self.segment_time)
                else:
                    raise Exception("Unknown criterion:", self.criterion)

        # Smoothen
        if self.smooth_width is not None:
            width = int(self.smooth_width / self.segment_time)
            kernel = np.full(width, 1/width)
            for i_fold in range(self.n_folds):
                scores[i_fold, :] = np.convolve(scores[i_fold, :], kernel, mode="same")

        # Average folds
        scores = scores.mean(axis=0)

        # Optimize the criterion
        if self.optimization == "max":
            self.stop_time_ = (1 + np.argmax(scores)) * self.segment_time
        elif self.optimization == "target":
            if self.target is None:
                raise Exception("For optimization target one should set the target")
            idx = np.where(scores >= self.target)[0]
            if len(idx) == 0:
                self.stop_time_ = X.shape[2] / self.fs
            else:
                self.stop_time_ = (1 + idx[0]) * self.segment_time
        else:
            raise Exception("Unknown optimization:", self.optimization)

        return self

    def predict(
            self,
            X: NDArray,
    ) -> NDArray:
        """The testing procedure to apply the estimator to novel EEG data using criterion static stopping.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: NDArray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """
        check_is_fitted(self, ["stop_time_"])

        ctime = X.shape[2] / self.fs
        if self.min_time is not None and ctime <= self.min_time:
            yh = np.full(X.shape[0], -1, dtype="int64")

        elif (self.max_time is not None and ctime >= self.max_time) or ctime >= self.stop_time_:
            yh = self.estimator.predict(X)

        else:
            yh = -1 * np.ones(X.shape[0])

        return yh

class DistributionStopping(BaseEstimator, ClassifierMixin):
    """Distribution dynamic stopping. Fits a distribution to non-target / non-maximum scores, and tests the probability
    of the target / maximum score to be an outlier of that distribution [2]_.

    Parameters
    ----------
    estimator: ClassifierMixin
        The classifier object that performs the classification.
    segment_time: float
        The size of a segment of data at which classification is performed ins seconds.
    fs: int
        The sampling frequency of the EEG data in Hz. Required for max_time.
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
    distributions_: list[dict]
        A list of dictionaries containing the parameters of the distribution for each data segment. Only used if
        trained=True.

    References
    ----------
    .. [2] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
           code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
           056007. doi: 10.1088/1741-2552/abecef
    """

    distributions_: list[dict]

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

        assert self.distribution in ["beta", "norm"], "Distribution must be beta or norm."

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
        # Fit estimator
        self.estimator.fit(X, y)

        # Fit beta distributions
        if self.trained:
            self.distributions_ = []
            n_segments = int(X.shape[2] / int(self.segment_time * self.fs))
            for i_segment in range(n_segments):

                # Estimate scores for this segment
                idx = (1 + i_segment) * int(self.segment_time * self.fs)
                scores = self.estimator.decision_function(X[:, :, :idx])

                # Put target score at index 0
                for i_trial in range(X.shape[0]):
                    scores[[0, y[i_trial]], :] = scores[[y[i_trial], 0], :]

                # Fit distribution to non-target scores
                if self.distribution == "beta":
                    a, b, loc, scale = beta.fit(scores[:, 1:].flatten(), floc=-1, fscale=2)
                    self.distributions_.append(dict(a=a, b=b, loc=loc, scale=scale))
                elif self.distribution == "norm":
                    loc, scale = norm.fit(scores[:, 1:].flatten())
                    self.distributions_.append(dict(loc=loc, scale=scale))

        return self

    def predict(
            self,
            X: NDArray,
    ) -> NDArray:
        """The testing procedure to apply the estimator to novel EEG data using dynamic stopping.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: NDArray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """

        ctime = X.shape[2] / self.fs

        if self.min_time is not None and ctime <= self.min_time:
            yh = np.full(X.shape[0], -1, dtype="int64")

        elif self.max_time is not None and ctime >= self.max_time:
            yh = self.estimator.predict(X)

        else:
            i_segment = int(np.round(ctime / self.segment_time)) - 1
            i_segment = np.max([0, i_segment])  # lower bound 0

            # Compute the scores
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
                    except:
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


class MarginStopping(BaseEstimator, ClassifierMixin):
    """Margin dynamic stopping. Learns threshold margins (difference between best and second-best score) to stop at
    such that a targeted accuracy is reached [3]_.

    Parameters
    ----------
    estimator: ClassifierMixin
        The classifier object that performs the classification.
    segment_time: float
        The size of a segment of data at which classification is performed ins seconds.
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
    margins_: NDArray
        The trained stopping margins of shape (n_segments).

    References
    ----------
    .. [3] Thielen, J., van den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials:
           re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797. doi: 10.1371/journal.pone.0133797
    """
    margins_: NDArray

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

        # Set margin axis (possible margins to stop at)
        margin_axis = np.arange(self.margin_min, self.margin_max, self.margin_step)

        # Calculate a margin per segment
        n_samples = X.shape[2]
        n_segments = int(n_samples / int(self.segment_time * self.fs))
        self.margins_ = np.zeros(n_segments)
        for i_segment in range(n_segments):

            # Compute scores for this segment
            idx = (1 + i_segment) * int(self.segment_time * self.fs)
            scores = self.estimator.decision_function(X[:, :, :idx])

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

        return self

    def predict(
            self,
            X: NDArray,
    ) -> NDArray:
        """The testing procedure to apply the estimator to novel EEG data using margin dynamic stopping.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: NDArray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """
        check_is_fitted(self, ["margins_"])

        ctime = X.shape[2] / self.fs

        if self.min_time is not None and ctime <= self.min_time:
            yh = np.full(X.shape[0], -1, dtype="int64")

        elif self.max_time is not None and ctime >= self.max_time:
            yh = self.estimator.predict(X)

        else:
            i_segment = int(np.round(ctime / self.segment_time)) - 1
            i_segment = np.max([0, i_segment])  # lower bound 0

            # Compute the scores
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


class ValueStopping(BaseEstimator, ClassifierMixin):
    """Value dynamic stopping. Learns threshold values to stop at such that a targeted accuracy is reached.

    Parameters
    ----------
    estimator: ClassifierMixin
        The classifier object that performs the classification.
    segment_time: float
        The size of a segment of data at which classification is performed ins seconds.
    fs: int
        The sampling frequency of the EEG data in Hz.
    target_p: float (default: 0.95)
        The targeted probability of correct classification.
    value_min: float (default: 0.0)
        The minimum value for the possible threshold margin to stop at.
    value_max: float (default: 1.0)
        The maximum value for the possible threshold margin to stop at.
    value_step: float (default: 0.05)
        The step size defining the resolution of the threshold margins at which to stop.
    max_time: float (default: None)
        The maximum time in seconds at which to force a stop, i.e., a classification. Trials will not be longer than
        this maximum time. If None, the algorithm will always emit -1 if it cannot stop.
    min_time: float (default: None)
        The minimum time in seconds at which a stop is possible, i.e., a classification. Before the minimum time, the
        algorithm will always emit -1. If None, the algorithm allows a stop already after the first segment of data.

    Attributes
    ----------
    values_: float
        The trained stopping values of shape (n_segments).
    """
    values_: NDArray

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

        # Calculate a value per segment
        n_samples = X.shape[2]
        n_segments = int(n_samples / int(self.segment_time * self.fs))
        self.values_ = np.zeros(n_segments)
        for i_segment in range(n_segments):

            # Compute scores for this segment
            idx = (1 + i_segment) * int(self.segment_time * self.fs)
            scores = self.estimator.decision_function(X[:, :, :idx])

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

        return self

    def predict(
            self,
            X: NDArray,
    ) -> NDArray:
        """The testing procedure to apply the estimator to novel EEG data using margin dynamic stopping.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: NDArray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """
        check_is_fitted(self, ["values_"])

        ctime = X.shape[2] / self.fs

        if self.min_time is not None and ctime <= self.min_time:
            yh = np.full(X.shape[0], -1, dtype="int64")

        elif self.max_time is not None and ctime >= self.max_time:
            yh = self.estimator.predict(X)

        else:
            i_segment = int(np.round(ctime / self.segment_time)) - 1
            i_segment = np.max([0, i_segment])  # lower bound 0

            # Compute the scores
            scores = self.estimator.decision_function(X)

            # Compute values
            values = np.max(scores, axis=1)

            # Check stopped
            not_stopped = values <= self.values_[i_segment]

            # Classify and set not-stopped-trials to -1
            yh = np.argmax(scores, axis=1)
            yh[not_stopped] = -1

        return yh
