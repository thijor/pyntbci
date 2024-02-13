import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from pyntbci.transformers import CCA, TRCA
from pyntbci.utilities import correct_latency, correlation, euclidean, event_matrix, structure_matrix


class eCCA(BaseEstimator, ClassifierMixin):
    """ERP CCA pipeline. Also called the "reference" method [1]_. It computes ERPs as templates for full sequences and
    performs a CCA for spatial filtering.

    Parameters
    ----------
    lags: np.ndarray
        A vector of latencies in seconds per class relative to the first code if codes are circularly shifted versions
        of the first code, or None if all codes are different or this circular shift feature should be ignored.
    fs: int
        The sampling frequency of the EEG data in Hz.
    cycle_size: float (default: None)
        The time that one cycle of the code takes in seconds. If None, takes the full data length.
    template_metric: str (default: "mean")
        Metric to use to compute templates: mean, median, OCSM.
    score_metric: str (default: "correlation")
        Metric to use to compute the overlap of templates and single-trials during testing: correlation, euclidean,
        inner.
    cca_channels: list (default: None)
        A list of channel indexes that need to be included in the estimation of a spatial filter at the template side of
        the CCA, i.e. CCA(X, T[:, cca_channels, :]). If None is given, all channels are used.
    lx: float | list (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along X (channels). If
        None, no regularization is applied.
    ly: float | list (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along T (channels). If
        None, no regularization is applied.
    latency: np.ndarray (default: None)
        The raster latencies of each of the classes of shape (n_classes,) that the data/templates need to be corrected
        for.
    ensemble: bool (default: False)
        Whether or not to use an ensemble classifier, that is, a separate spatial filter for each class.
    cov_estimator: object (default: None)
        Estimator object with a fit method that estimates a covariance matrix of the EEG data. If None, a custom
        empirical covariance is used.

    References
    ----------
    .. [1] Martínez-Cagigal, V., Thielen, J., Santamaria-Vazquez, E., Pérez-Velasco, S., Desain, P., & Hornero, R.
           (2021). Brain–computer interfaces based on code-modulated visual evoked potentials (c-VEP): A literature
           review. Journal of Neural Engineering, 18(6), 061002. doi: 10.1088/1741-2552/ac38cf
    """

    def __init__(self, lags, fs, cycle_size=None, template_metric="mean", score_metric="correlation", cca_channels=None,
                 lx=None, ly=None, latency=None, ensemble=False, cov_estimator=None):
        self.lags = lags
        self.fs = fs
        self.cycle_size = cycle_size
        self.template_metric = template_metric
        self.score_metric = score_metric
        self.cca_channels = cca_channels
        self.lx = lx
        self.ly = ly
        self.latency = latency
        self.ensemble = ensemble
        self.cov_estimator = cov_estimator

    def _fit_T(self, X):
        """Fit the templates.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        T: np.ndarray
            The matrix of one EEG template of shape (n_channels, n_samples).
        """
        n_trials, n_channels, n_samples = X.shape
        if self.template_metric == "mean":
            T = np.mean(X, axis=0)
        elif self.template_metric == "median":
            T = np.median(X, axis=0)
        elif self.template_metric == "ocsvm":
            ocsvm = OneClassSVM(kernel="linear", nu=0.5)
            T = np.zeros((n_channels, n_samples))
            for i_channel in range(n_channels):
                ocsvm.fit(X[:, i_channel, :])
                T[i_channel, :] = ocsvm.coef_
        else:
            raise Exception(f"Unknown template metric:", self.template_metric)
        return T

    def _get_T(self, n_samples=None):
        """Get the templates.

        Parameters
        ----------
        n_samples: int (default: None)
            The number of samples requested. If None, one code cycle is given.

        Returns
        -------
        T: np.ndarray
            The templates of shape (n_classes, n_samples).
        """
        if n_samples is None or self.T_.shape[1] == n_samples:
            T = self.T_
        else:
            n = int(np.ceil(n_samples / self.T_.shape[1]))
            T = np.tile(self.T_, (1, n))[:, :n_samples]
        T -= T.mean(axis=1, keepdims=True)
        return T

    def decision_function(self, X):
        """Applies the routines to arrive at raw unthresholded classification scores for X.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        scores: np.ndarray
            The matrix of scores of shape (n_trials, n_classes).
        """
        # Set templates to trial length
        T = self._get_T(X.shape[2])

        if self.ensemble:
            scores = np.zeros((X.shape[0], T.shape[0]))
            for i_class in range(T.shape[0]):

                # Apply spatial filter
                Xi = np.sum(self._cca[i_class].transform(X=X)[0], axis=1)

                # Scoring
                if self.score_metric == "correlation":
                    scores[:, i_class] = correlation(Xi, T[i_class, :])[:, 0]
                elif self.score_metric == "euclidean":
                    scores[:, i_class] = 1 / (1 + euclidean(Xi, T[i_class, :]))[:, 0]  # conversion to similarity
                elif self.score_metric == "inner":
                    scores[:, i_class] = np.inner(Xi, T[i_class, :])
                else:
                    raise Exception(f"Unknown score metric: {self.score_metric}")

        else:
            # Apply spatial filter
            X = np.sum(self._cca.transform(X=X)[0], axis=1)

            # Scoring
            if self.score_metric == "correlation":
                scores = correlation(X, T)
            elif self.score_metric == "euclidean":
                scores = 1 / (1 + euclidean(X, T))  # conversion to similarity
            elif self.score_metric == "inner":
                scores = np.inner(X, T)
            else:
                raise Exception(f"Unknown score metric: {self.score_metric}")

        return scores

    def fit(self, X, y):
        """The training procedure to fit eCCA on supervised EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: numpy.ndarray
            The vector of ground-truth labels of the trials in X of shape (n_trials), i.e., the index of the
            attended code.

        Returns
        -------
        self: eCCA
            An instance of the classifier.
        """
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True, y_numeric=True)
        y = y.astype(np.uint)
        n_trials, n_channels, n_samples = X.shape

        # Correct for raster latency
        if self.latency is not None:
            X = correct_latency(X, y, -self.latency, self.fs, axis=-1)

        # Synchronize all classes
        if self.lags is not None:
            X = correct_latency(X, y, -self.lags, self.fs, axis=-1)
            y = np.zeros(y.shape, y.dtype)

        # Cut trials to cycles
        if self.cycle_size is not None:
            cycle_size = int(self.cycle_size * self.fs)
            assert n_samples % cycle_size == 0, "X must be full cycles."
            n_cycles = int(n_samples / cycle_size)
            X = X.reshape((n_trials, n_channels, n_cycles, cycle_size))
            X = X.transpose((0, 2, 1, 3))
            X = X.reshape((n_trials * n_cycles, n_channels, cycle_size))
            n_trials, n_channels, n_samples = X.shape
            y = np.repeat(y, n_cycles)

        # Compute templates
        if self.lags is None:
            # Compute a template per class separately
            n_classes = np.unique(y).size
            T = np.zeros((n_classes, n_channels, n_samples))
            for i_class in range(n_classes):
                T[i_class, :, :] = self._fit_T(X[y == i_class, :, :])
        else:
            # Compute a template for latency 0 and shift for all others
            n_classes = len(self.lags)
            T = np.tile(self._fit_T(X)[np.newaxis, :, :], (n_classes, 1, 1))
            T = correct_latency(T, np.arange(n_classes), self.lags, self.fs, axis=-1)
            if self.latency is not None:
                T = correct_latency(T, np.arange(n_classes), self.latency, self.fs, axis=-1)

        # Fit CCA
        if self.ensemble:
            self.w_ = np.zeros((n_channels, n_classes))
            self._cca = []
            for i_class in range(n_classes):
                S = np.reshape(X[y == i_class, :, :].transpose((0, 2, 1)), (-1, n_channels))  # Concatenate trials
                R = np.tile(T[i_class, :, :].T, (np.sum(y == i_class), 1))  # Concatenate templates
                if self.cca_channels is not None:
                    R = R[:, self.cca_channels]
                self._cca.append(CCA(n_components=1, lx=self.lx, ly=self.ly, estimator_x=self.cov_estimator,
                                     estimator_y=self.cov_estimator))
                self._cca[i_class].fit(S, R)
                self.w_[:, i_class] = self._cca[i_class].w_x_.flatten()
        else:
            S = np.reshape(X.transpose((0, 2, 1)), (-1, n_channels))  # Concatenate trials
            R = np.reshape(T[y, :, :].transpose((0, 2, 1)), (-1, n_channels))  # Concatenate templates
            if self.cca_channels is not None:
                R = R[:, self.cca_channels]
            self._cca = CCA(n_components=1, lx=self.lx, ly=self.ly, estimator_x=self.cov_estimator,
                            estimator_y=self.cov_estimator)
            self._cca.fit(S, R)
            self.w_ = self._cca.w_x_

        # Spatially filter templates
        if self.ensemble:
            self.T_ = np.zeros((n_classes, n_samples))
            for i_class in range(n_classes):
                self.T_[i_class, :] = np.sum(self._cca[i_class].transform(X=T[i_class, :, :].T)[0], axis=1)
        else:
            self.T_ = np.sum(self._cca.transform(X=T)[0], axis=1)

        return self

    def predict(self, X):
        """The testing procedure to apply eCCA to novel EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: np.ndarray
            The vector of predicted labels of the trials in X of shape (n_trials).
        """
        check_is_fitted(self, ["w_", "T_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)
        return np.argmax(self.decision_function(X), axis=1)


class eTRCA(BaseEstimator, ClassifierMixin):
    """ERP TRCA pipeline. It computes ERPs as templates for full sequences and performs a TRCA for spatial filtering
    [2]_.

    Parameters
    ----------
    lags: np.ndarray
        A vector of latencies in seconds per class relative to the first code if codes are circularly shifted versions
        of the first code, or None if all codes are different or this circular shift feature should be ignored.
    fs: int
        The sampling frequency of the EEG data in Hz.
    cycle_size: float (default: None)
        The time that one cycle of the code takes in seconds. If None, takes the full data length.
    template_metric: str (default: "mean")
        Metric to use to compute templates: mean, median, OCSM.
    score_metric: str (default: "correlation")
        Metric to use to compute the overlap of templates and single-trials during testing: correlation, euclidean,
        inner.
    latency: np.ndarray (default: None)
        The raster latencies of each of the classes of shape (n_classes,) that the data/templates need to be corrected
        for.
    ensemble: bool (default: False)
        Whether or not to use an ensemble classifier, that is, a separate spatial filter for each class.

    References
    ----------
    .. [2] Nakanishi, M., Wang, Y., Chen, X., Wang, Y. T., Gao, X., & Jung, T. P. (2017). Enhancing detection of SSVEPs
           for a high-speed brain speller using task-related component analysis. IEEE Transactions on Biomedical
           Engineering, 65(1), 104-112. doi: 10.1109/TBME.2017.2694818
    """

    def __init__(self, lags, fs, cycle_size=None, template_metric="mean", score_metric="correlation", latency=None,
                 ensemble=False):
        self.lags = lags
        self.fs = fs
        self.cycle_size = cycle_size
        self.template_metric = template_metric
        self.score_metric = score_metric
        self.latency = latency
        self.ensemble = ensemble

    def _fit_T(self, X):
        """Fit the templates.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        T: np.ndarray
            The matrix of one EEG template of shape (n_channels, n_samples).
        """
        if self.template_metric == "mean":
            T = np.mean(X, axis=0)
        elif self.template_metric == "median":
            T = np.median(X, axis=0)
        elif self.template_metric == "ocsvm":
            ocsvm = OneClassSVM(kernel="linear", nu=0.5)
            ocsvm.fit(X)
            T = ocsvm.coef_
        else:
            raise Exception(f"Unknown template metric:", self.template_metric)
        return T

    def _get_T(self, n_samples=None):
        """Get the templates.

        Parameters
        ----------
        n_samples: int (default: None)
            The number of samples requested. If None, one code cycle is given.

        Returns
        -------
        T: np.ndarray
            The templates of shape (n_classes, n_samples).
        """
        if n_samples is None or self.T_.shape[1] == n_samples:
            T = self.T_
        else:
            n = int(np.ceil(n_samples / self.T_.shape[1]))
            T = np.tile(self.T_, (1, n))[:, :n_samples]
        T -= T.mean(axis=1, keepdims=True)
        return T

    def decision_function(self, X):
        """Applies the routines to arrive at raw unthresholded classification scores for X.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        scores: np.ndarray
            The matrix of scores of shape (n_trials, n_classes).
        """

        # Apply spatial filter
        X = np.sum(self._trca.transform(X), axis=1)

        # Set templates to trial length
        T = self._get_T(X.shape[1])

        # Scoring
        if self.ensemble:
            n_trials = X.shape[0]
            n_classes = T.shape[0]
            scores = np.zeros((n_trials, n_classes))
            for i_class in range(n_classes):
                if self.score_metric == "correlation":
                    scores[:, i_class] = correlation(X[..., i_class], T[i_class, :])[:, 0]
                elif self.score_metric == "euclidean":
                    scores[:, i_class] = 1 / (1 + euclidean(X[..., i_class], T[i_class, :]))[:, 0]
                elif self.score_metric == "inner":
                    scores[:, i_class] = np.inner(X[..., i_class], T[i_class, :])
                else:
                    raise Exception(f"Unknown score metric: {self.score_metric}")
            return scores
        else:
            if self.score_metric == "correlation":
                return correlation(X, T)
            elif self.score_metric == "euclidean":
                return 1 / (1 + euclidean(X, T))
            elif self.score_metric == "inner":
                return np.inner(X, T)
            else:
                raise Exception(f"Unknown score metric: {self.score_metric}")

    def fit(self, X, y):
        """The training procedure to fit eTRCA on supervised EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: numpy.ndarray
            The vector of ground-truth labels of the trials in X of shape (n_trials), i.e., the index of the attended
            code.

        Returns
        -------
        self: eTRCA
            An instance of the classifier.
        """
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True, y_numeric=True)
        y = y.astype(np.uint)
        n_trials, n_channels, n_samples = X.shape

        # Correct for raster latency
        if self.latency is not None:
            X = correct_latency(X, y, -self.latency, self.fs, axis=-1)

        # Learn spatial filter
        self._trca = TRCA(n_components=1)
        if self.ensemble:
            self._trca.fit(X, y)
        else:
            self._trca.fit(X)
        self.w_ = self._trca.w_

        # Spatially filter data
        if self.ensemble:
            X = np.sum(self._trca.transform(X, y), axis=1)
        else:
            X = np.sum(self._trca.transform(X), axis=1)

        # Synchronize all classes
        if self.lags is not None:
            X = correct_latency(X, y, -self.lags, self.fs, axis=-1)
            y = np.zeros(y.shape, y.dtype)

        # Cut trials to cycles
        if self.cycle_size is not None:
            cycle_size = int(self.cycle_size * self.fs)
            assert n_samples % cycle_size == 0, "X must be full cycles."
            n_cycles = int(n_samples / cycle_size)
            X = X.reshape((n_trials * n_cycles, cycle_size))
            n_trials, n_samples = X.shape
            y = np.repeat(y, n_cycles)

        # Compute templates
        if self.lags is None:
            # Compute a template per class separately
            classes = np.unique(y)
            n_classes = classes.size
            T = np.zeros((n_classes, n_samples))
            for i_class in range(n_classes):
                T[i_class, :] = self._fit_T(X[y == classes[i_class], :])
        else:
            # Compute a template for latency 0 and shift for all others
            n_classes = len(self.lags)
            T = np.tile(self._fit_T(X)[np.newaxis, :], (n_classes, 1))
            T = correct_latency(T, np.arange(n_classes), self.lags, self.fs, axis=-1)
            if self.latency is not None:
                T = correct_latency(T, np.arange(n_classes), self.latency, self.fs, axis=-1)
        self.T_ = T

        return self

    def predict(self, X):
        """The testing procedure to apply eTRCA to novel EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: np.ndarray
            The vector of predicted labels of the trials in X of shape (n_trials).
        """
        check_is_fitted(self, ["w_", "T_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)
        return np.argmax(self.decision_function(X), axis=1)


class FilterBank(BaseEstimator, ClassifierMixin):
    """Filterbank pipeline. It wraps a filterbank around a classifier object, where the classifier is combined to each
    passband in the filterbank separately, and a gating function combines the outputs.

    Parameters
    ----------
    estimator: BaseEstimator
        The classifier object that is applied to each of the passbands in the filterbank.
    gating: str (default: "mean")
        The gating function that is used to combine the scores obtained from each of the passbands: mean, max.
    """

    def __init__(self, estimator, gating="mean"):
        self.estimator = estimator
        self.gating = gating

    def decision_function(self, X):
        """Applies the routines to arrive at raw unthresholded classification scores for X.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples, n_passbands).

        Returns
        -------
        scores: np.ndarray
            The matrix of scores of shape (n_trials, n_classes).
        """

        # Get separate raw scores for each passband
        scores = [
            self.models_[i].decision_function(X[:, :, :, i])
            for i in range(X.shape[3])]
        scores = np.stack(scores, axis=2)

        # Combine passbands
        if self.gating == "mean":
            return np.mean(scores, axis=2)
        elif self.gating == "max":
            return np.max(scores, axis=2)

    def fit(self, X, y):
        """The training procedure to apply a filterbank on supervised EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples, n_passbands).
        y: np.ndarray
            The vector of ground-truth labels of the trials in X of shape (n_trials). Note, these denote the index at
            which to find the associated codes!

        Returns
        -------
        self: FilterBank
            An instance of the classifier.
        """
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True, y_numeric=True)
        y = y.astype(np.uint)
        assert X.ndim == 4

        # Fit separate model for each passband
        self.models_ = [
            copy.deepcopy(self.estimator).fit(X[:, :, :, i], y)
            for i in range(X.shape[3])]

        return self

    def predict(self, X):
        """The testing procedure to apply the filterbank to novel EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples, n_passbands).

        Returns
        -------
        y: np.ndarray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, these denote the index at which
            to find the associated codes!
        """
        check_is_fitted(self, ["models_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)
        assert X.ndim == 4
        return np.argmax(self.decision_function(X), axis=1)


class rCCA(BaseEstimator, ClassifierMixin):
    """Reconvolution CCA pipeline. It performs a spatial and temporal decomposition (reconvolution [3]_) within a
    CCA [4]_ to perform spatial filtering as well as template prediction [5]_.

    Parameters
    ----------
    codes: np.ndarray
        The pseudo-random noise-codes used for stimulation of shape (n_classes, n_samples). Should be sampled at fs and
        one cycle (i.e., code-repetition) is sufficient.
    fs: int
        The sampling frequency of the EEG data in Hz.
    event: str (default: "duration")
        The event type to perform the transformation of codes to events with.
    transient_size: float | list (default: 0.3)
        The length of the transient response(s) for each of the events in seconds.
    onset_event: bool (default: False)
        Whether or not to add an event for the onset of stimulation. Added as last event.
    score_metric: str (default: "correlation")
        Metric to use to compute the overlap of templates and single-trials during testing: correlation, euclidean.
    lx: float | list (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along X (channels). If
        None, no regularization is applied.
    ly: float | list (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along M (samples). If None,
        no regularization is applied.
    latency: np.ndarray (default: None)
        The raster latencies of each of the classes of shape (n_classes,) that the data/templates need to be corrected
        for.
    ensemble: bool (default: False)
        Whether or not to use an ensemble classifier, that is, a separate spatial filter for each class.
    amplitudes: np.ndarray
        The amplitudes of the stimulation of shape (n_classes, n_samples). Should be sampled at fs similar to codes.
    cov_estimator_x: object (default: None)
        Estimator object with a fit method that estimates a covariance matrix of the EEG data. If None, a custom
        empirical covariance is used.
    cov_estimator_m: object (default: None)
        Estimator object with a fit method that estimates a covariance matrix of the stimulus structure matrix. If None,
        a custom empirical covariance is used.

    References
    ----------
    .. [3] Thielen, J., van den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials:
           re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797. doi: 10.1371/journal.pone.0133797
    .. [4] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2017). Re(con)volution: accurate response prediction
           for broad-band evoked potentials-based brain computer interfaces. Brain-Computer Interface Research: A
           State-of-the-Art Summary 6, 35-42. doi: 10.1007/978-3-319-64373-1_4
    .. [5] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
           code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
           056007. doi: 10.1088/1741-2552/abecef
    """

    def __init__(self, codes, fs, event="duration", transient_size=0.3, onset_event=False, score_metric="correlation",
                 lx=None, ly=None, latency=None, ensemble=False, amplitudes=None, cov_estimator_x=None,
                 cov_estimator_m=None):
        self.codes = codes
        self.fs = fs
        self.event = event
        self.transient_size = transient_size
        self.onset_event = onset_event
        self.score_metric = score_metric
        self.lx = lx
        self.ly = ly
        self.latency = latency
        self.ensemble = ensemble
        self.amplitudes = amplitudes
        self.cov_estimator_x = cov_estimator_x
        self.cov_estimator_m = cov_estimator_m

    def _get_M(self, n_samples=None):
        """Get the structure matrix of a particular length.

        Parameters
        ----------
        n_samples: int (default: None)
            The number of samples that the structure matrix should be. If none, one code cycle will be used.

        Returns
        -------
        M: np.ndarray
            The structure matrix denoting event timings of shape (n_classes, transient_size, n_samples).
        """
        # Repeat the codes to n_samples
        if n_samples is None or self.codes.shape[1] == n_samples:
            codes = self.codes
        else:
            n = int(np.ceil(n_samples / self.codes.shape[1]))
            codes = np.tile(self.codes, (1, n))

        # Get structure matrices
        E, self.events_ = event_matrix(codes, self.event, self.onset_event)
        M = structure_matrix(E, int(self.transient_size * self.fs), self.amplitudes)
        return M[:, :, :n_samples]

    def _get_T(self, n_samples=None):
        """Get the templates.

        Parameters
        ----------
        n_samples: int (default: None)
            The number of samples requested. If None, one code cycle is given.

        Returns
        -------
        T: np.ndarray
            The templates of shape (n_classes, n_samples).
        """
        if n_samples is None or self.Ts_.shape[1] == n_samples:
            T = self.Ts_
        else:
            n = int(np.ceil(n_samples / self.Ts_.shape[1]))
            if (n - 1) > 1:
                Tw = np.tile(self.Tw_, (1, (n - 1)))
            else:
                Tw = self.Tw_
            T = np.concatenate((self.Ts_, Tw), axis=1)[:, :n_samples]
        T -= T.mean(axis=1, keepdims=True)
        return T

    def decision_function(self, X):
        """Applies the routines to arrive at raw unthresholded classification scores for X.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        scores: np.ndarray
            The matrix of scores of shape (n_trials, n_classes).
        """
        # Set templates to trial length
        T = self._get_T(X.shape[2])

        if self.ensemble:
            scores = np.zeros((X.shape[0], T.shape[0]))
            for i_class in range(T.shape[0]):

                # Spatially filter data
                Xi = np.sum(self._cca[i_class].transform(X=X)[0], axis=1)

                # Scoring
                if self.score_metric == "correlation":
                    scores[:, i_class] = correlation(Xi, T[i_class, :])[:, 0]
                elif self.score_metric == "euclidean":
                    scores[:, i_class] = 1 / (1 + euclidean(Xi, T[i_class, :]))[:, 0]  # conversion to similarity
                elif self.score_metric == "inner":
                    scores[:, i_class] = np.inner(Xi, T[i_class, :])
                else:
                    raise Exception(f"Unknown score metric: {self.score_metric}")

        else:
            # Spatially filter data
            X = np.sum(self._cca.transform(X=X)[0], axis=1)

            # Scoring
            if self.score_metric == "correlation":
                scores = correlation(X, T)
            elif self.score_metric == "euclidean":
                scores = 1 / (1 + euclidean(X, T))  # conversion to similarity
            elif self.score_metric == "inner":
                scores = np.inner(X, T)
            else:
                raise Exception(f"Unknown score metric: {self.score_metric}")

        return scores

    def fit(self, X, y):
        """The training procedure to fit a rCCA on supervised EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: np.ndarray
            The vector of ground-truth labels of the trials in X of shape (n_trials). Note, these denote the index at
            which to find the associated codes!

        Returns
        -------
        self: rCCA
            An instance of the classifier.
        """
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True, y_numeric=True)
        y = y.astype(np.uint)
        n_classes = np.unique(y).size

        # Correct for raster latency
        if self.latency is not None:
            X = correct_latency(X, y, -self.latency, self.fs, axis=-1)

        # Get structure matrix
        M = self._get_M(X.shape[2])

        # Fit w and r
        if self.ensemble:
            self.w_ = np.zeros((X.shape[1], n_classes))
            self.r_ = np.zeros((M.shape[1], n_classes))
            self._cca = []
            for i_class in range(n_classes):
                self._cca.append(CCA(n_components=1, lx=self.lx, ly=self.ly, estimator_x=self.cov_estimator_x,
                                     estimator_y=self.cov_estimator_m))
                self._cca[i_class].fit(X[y == i_class, :, :], np.tile(M[[i_class], :, :], (np.sum(y == i_class), 1, 1)))
                self.w_[:, i_class] = self._cca[i_class].w_x_.flatten()
                self.r_[:, i_class] = self._cca[i_class].w_y_.flatten()
        else:
            self._cca = CCA(n_components=1, lx=self.lx, ly=self.ly, estimator_x=self.cov_estimator_x,
                            estimator_y=self.cov_estimator_m)
            self._cca.fit(X, M[y, :, :])
            self.w_ = self._cca.w_x_
            self.r_ = self._cca.w_y_

        # Set templates (start and wrapper)
        M = self._get_M(2 * self.codes.shape[1])
        if self.ensemble:
            T = np.zeros((n_classes, M.shape[2]))
            for i_class in range(n_classes):
                T[i_class, :] = np.sum(self._cca[i_class].transform(X=None, Y=M[[i_class], :, :])[1], axis=1)
        else:
            T = np.sum(self._cca.transform(X=None, Y=M)[1], axis=1)
        self.Ts_ = T[:, :self.codes.shape[1]]
        self.Tw_ = T[:, self.codes.shape[1]:]

        # Correct for raster latency
        if self.latency is not None:
            self.Ts_ = correct_latency(self.Ts_, np.arange(len(self.latency)), self.latency, self.fs, axis=-1)
            self.Tw_ = correct_latency(self.Tw_, np.arange(len(self.latency)), self.latency, self.fs, axis=-1)

        return self

    def predict(self, X):
        """The testing procedure to apply rCCA to novel EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: np.ndarray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, these denote the index at which
            to find the associated codes!
        """
        check_is_fitted(self, ["w_", "r_", "Ts_", "Tw_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)
        return np.argmax(self.decision_function(X), axis=1)

    def set_codes(self, codes):
        """Set the codes and as such change the templates.

        Parameters
        ----------
        codes: np.ndarray
            The pseudo-random noise-codes used for stimulation of shape (n_classes, n_samples). Should be sampled at fs
            and one code-repetition is sufficient.
        """
        self.codes = codes
        T = np.sum(self._cca.transform(X=None, Y=self._get_M(2 * self.codes.shape[1]))[1], axis=1)
        self.Ts_ = T[:, :self.codes.shape[1]]
        self.Tw_ = T[:, self.codes.shape[1]:]

    def set_amplitudes(self, amplitudes):
        """Set the codes and as such change the templates.

        Parameters
        ----------
        amplitudes: np.ndarray
            The amplitudes of the stimulation of shape (n_classes, n_samples). Should be sampled at fs similar to codes.
        """
        self.amplitudes = amplitudes
        self.set_codes(self.codes)
