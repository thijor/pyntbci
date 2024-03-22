import numpy as np
from scipy.stats import beta, norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import pyntbci.classifiers


class BayesStopping(BaseEstimator, ClassifierMixin):
    """Bayesian dynamic stopping. Fits Gaussian distributions for target and non-target responses, and calculates a
    stopping threshold using these and a cost criterion [1]_.

    Parameters
    ----------
    estimator: BaseEstimator
        The classifier object that performs the classification.
    segment_time: float
        The size of a segment of data at which classification is performed ins seconds.
    fs: int
        The sampling frequency of the EEG data in Hz.
    method: str (default: "bes0")
        The method to use for Bayesian dynamic stopping: bes0, bes1, bes2.
    cr: float (default: 1.0)
        The cost ratio.
    target_pf: float (default: 0.05)
        The targeted probability for error.
    target_pd: float (default: 0.80)
        The targeted probability for detection.
    max_time: float (default: None)
        The maximum time at which to force a stop, i.e., a classification. If None, the algorithm will always emit -1 if
        it cannot stop, otherwise it will emit a classification regardless of the certainty after that maximum time.

    References
    ----------
    .. [1] Ahmadi, S., Thielen, J., Farquhar, J., & Desain, P. (in prep.) A model driven Bayesian dynamic stopping
           method for parallel stimulation evoked response BCIs.
    """

    def __init__(self, estimator, segment_time, fs, method="bes0", cr=1.0, target_pf=0.05, target_pd=0.80,
                 max_time=None):
        self.estimator = estimator
        self.segment_time = segment_time
        self.fs = fs
        self.method = method
        self.cr = cr
        self.target_pf = target_pf
        self.target_pd = target_pd
        self.max_time = max_time

    def fit(self, X, y):
        """The training procedure to fit the dynamic procedure on supervised EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples). Note, must be full cycles, such that
            n_samples % (cycle_size * fs) == 0.
        y: np.ndarray
            The vector of ground-truth labels of the trials in X of shape (n_trials). These denote the index at which to
            find the associated codes in latency!

        Returns
        -------
        self: BayesStopping
            An instance of the stopping procedure.
        """
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True, y_numeric=True)
        y = y.astype(np.uint)

        # TODO: pyntbci.classifier.BayesStopping does not yet work with pyntbci.classifiers.Ensemble
        # N.B. Ensemble does not implement _apply_w and _get_T
        assert not isinstance(self.estimator, pyntbci.classifiers.Ensemble), "Not yet implemented for Ensemble!"

        # Fit estimator
        self.estimator.fit(X, y)

        # Spatially filter data
        X = np.sum(self.estimator._cca.transform(X=X)[0], axis=1)
        n_samples = X.shape[1]

        # Get templates
        T = self.estimator._get_T(n_samples)
        n_classes = T.shape[0]

        # TODO: pyntbci.classifier.BayesStopping does not work with multi-component templates
        # N.B. Ensemble assumes n-components=1
        assert T.shape[1] == 1, "Not yet implemented for multiple components!"
        T = T[:, 0, :]  # remove singular dimension

        # Obtain alpha from least squares
        model = LinearRegression()
        model.fit(T[y, :].reshape(-1, 1), X.reshape(-1, 1))
        self.alpha_ = model.coef_[0, 0]

        # Obtain sigma from Gaussian fit to residuals
        residuals = X.reshape(-1, 1) - model.predict(T[y, :].reshape(-1, 1))
        self.sigma_ = norm.fit(residuals)[1]

        # Calculate b0, b1, s0, s1
        n_segments = int(n_samples / int(self.segment_time * self.fs))
        self.b0_ = np.zeros(n_segments)
        self.b1_ = np.zeros(n_segments)
        self.s0_ = np.zeros(n_segments)
        self.s1_ = np.zeros(n_segments)
        for i_segment in range(n_segments):
            idx = (1 + i_segment) * int(self.segment_time * self.fs)
            inner = np.inner(T[:, :idx], T[:, :idx])
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

        return self

    def predict(self, X):
        """The testing procedure to apply the estimator to novel EEG data using Bayesian dynamic stopping.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: np.ndarray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """
        check_is_fitted(self, ["alpha_", "sigma_", "b0_", "b1_", "s0_", "s1_", "pf_", "pm_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)

        if self.max_time is None or X.shape[2] < self.max_time * self.fs:

            # Compute the scores
            scores = self.estimator.decision_function(X)

            # Check if stopped
            i_segment = int(X.shape[2] / int(self.segment_time * self.fs)) - 1
            if self.method == "bes0":
                # Stop if eta threshold of this segment is reached
                not_stopped = np.max(scores, axis=1) <= self.eta_[i_segment]

            elif self.method == "bes1":
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

            elif self.method == "bes2":
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

        else:
            yh = self.estimator.predict(X)

        return yh


class BetaStopping(BaseEstimator, ClassifierMixin):
    """Beta dynamic stopping. Fits a beta distribution to non-max (correlation+1)/2, and tests the probability of
    the maximum correlation to belong to that same distribution [2]_.

    Parameters
    ----------
    estimator: BaseEstimator
        The classifier object that performs the classification.
    target_p: float (default: 0.95)
        The targeted probability of correct classification.
    fs: int (default None)
        The sampling frequency of the EEG data in Hz. Required for max_time.
    max_time: float (default: None)
        The maximum time at which to force a stop, i.e., a classification. If None, the algorithm will always emit -1 if
        it cannot stop, otherwise it will emit a classification regardless of the certainty after that maximum time.

    References
    ----------
    .. [2] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
           code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
           056007. doi: 10.1088/1741-2552/abecef
    """

    def __init__(self, estimator, target_p=0.95, fs=None, max_time=None):
        self.estimator = estimator
        self.target_p = target_p
        self.fs = fs
        self.max_time = max_time
        if self.max_time is not None:
            assert self.fs is not None, "If max_time is specified, then also fs should be specified."

    def fit(self, X, y):
        """The training procedure to fit the dynamic procedure on supervised EEG data. Note, BetaStopping itself does
        not require training, it only trains the estimator.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples). Note, must be full cycles, such that
            n_samples % (cycle_size * fs) == 0.
        y: np.ndarray
            The vector of ground-truth labels of the trials in X of shape (n_trials). These denote the index at which
            to find the associated codes in latency!

        Returns
        -------
        self: BayesStopping
            An instance of the stopping procedure.
        """
        # Fit estimator
        self.estimator.fit(X, y)

    def predict(self, X):
        """The testing procedure to apply the estimator to novel EEG data using beta dynamic stopping.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: np.ndarray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """
        X = check_array(X, ensure_2d=False, allow_nd=True)

        if self.max_time is None or X.shape[2] < self.max_time * self.fs:

            # Compute the scores and translate to range 0 to 1
            scores = (self.estimator.decision_function(X) + 1) / 2

            # Sort the scores (ascending)
            scores_sorted = np.sort(scores, axis=1)

            # Calculate probability of maximum score being a "true" maximum
            p = np.zeros(scores.shape[0])
            for i_trial in range(scores.shape[0]):
                # Fit beta to non-max scores
                a, b, loc, scale = beta.fit(scores_sorted[i_trial, :-1], floc=0, fscale=1)

                # Look up probability of maximum score in beta
                p[i_trial] = beta.cdf(scores_sorted[i_trial, -1], a, b, loc, scale) ** scores.shape[1]

            # Check if stopped
            not_stopped = p <= self.target_p

            # Classify and set not-stopped-trials to -1
            yh = np.argmax(scores, axis=1)
            yh[not_stopped] = -1

        else:
            yh = self.estimator.predict(X)

        return yh


class MarginStopping(BaseEstimator, ClassifierMixin):
    """Margin dynamic stopping. Learns threshold margins (difference between best and second best score) to stop at
    such that a targeted accuracy is reached [3]_.

    Parameters
    ----------
    estimator: BaseEstimator
        The classifier object that performs the classification.
    segment_time; float
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
        The maximum time at which to force a stop, i.e., a classification. If None, the algorithm will always emit -1 if
        it cannot stop, otherwise it will emit a classification regardless of the certainty after that maximum time.

    References
    ----------
    .. [3] Thielen, J., van den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials:
           re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797. doi: 10.1371/journal.pone.0133797
    """

    def __init__(self, estimator, segment_time, fs, target_p=0.95, margin_min=0.0, margin_max=1.0, margin_step=0.05,
                 max_time=None):
        self.estimator = estimator
        self.segment_time = segment_time
        self.fs = fs
        self.target_p = target_p
        self.margin_min = margin_min
        self.margin_max = margin_max
        self.margin_step = margin_step
        self.max_time = max_time

    def fit(self, X, y):
        """The training procedure to fit the dynamic procedure on supervised EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples). Note, must be full cycles, such that
            n_samples % (cycle_size * fs) == 0.
        y: np.ndarray
            The vector of ground-truth labels of the trials in X of shape (n_trials). These denote the index at
            which to find the associated codes in latency!

        Returns
        -------
        self: MarginStopping
            An instance of the stopping procedure.
        """
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True, y_numeric=True)
        y = y.astype(np.uint)

        # Fit estimator
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

    def predict(self, X):
        """The testing procedure to apply the estimator to novel EEG data using margin dynamic stopping.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: np.ndarray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, the value equals -1 if the
            trial cannot yet be stopped.
        """
        check_is_fitted(self, ["margins_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)

        if self.max_time is None or X.shape[2] < self.max_time * self.fs:

            # Compute the scores
            scores = self.estimator.decision_function(X)

            # Sort the scores (ascending)
            scores_sorted = np.sort(scores, axis=1)

            # Compute margins (best minus second best)
            margins = scores_sorted[:, -1] - scores_sorted[:, -2]

            # Check stopped
            i_segment = int(X.shape[2] / int(self.segment_time * self.fs)) - 1
            not_stopped = margins <= self.margins_[i_segment]

            # Classify and set not-stopped-trials to -1
            yh = np.argmax(scores, axis=1)
            yh[not_stopped] = -1

        else:
            yh = self.estimator.predict(X)

        return yh
