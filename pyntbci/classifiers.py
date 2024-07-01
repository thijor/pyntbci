import copy
from typing import Union

import numpy as np
import sklearn.base
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from pyntbci.transformers import CCA, TRCA
from pyntbci.utilities import correct_latency, correlation, decoding_matrix, encoding_matrix, euclidean, event_matrix


class eCCA(BaseEstimator, ClassifierMixin):
    """ERP CCA classifier. Also called the "reference" method [1]_. It computes ERPs as templates for full sequences and
    performs a CCA for spatial filtering.

    Parameters
    ----------
    lags: None | np.ndarray
        A vector of latencies in seconds per class relative to the first stimulus if stimuli are circularly shifted
        versions of the first stimulus, or None if all stimuli are different or this circular shift feature should be
        ignored.
    fs: int
        The sampling frequency of the EEG data in Hz.
    cycle_size: float (default: None)
        The time that one cycle of the code takes in seconds. If None, takes the full data length.
    template_metric: str (default: "mean")
        Metric to use to compute templates: mean, median, OCSVM.
    score_metric: str (default: "correlation")
        Metric to use to compute the overlap of templates and single-trials during testing: correlation, euclidean,
        inner.
    cca_channels: list[int] (default: None)
        A list of channel indexes that need to be included in the estimation of a spatial filter at the template side
        of the CCA, i.e. CCA(X, T[:, cca_channels, :]). If None is given, all channels are used.
    gamma_x: float | list[float] | np.ndarray (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along X (channels). If
        None, no regularization is applied. The gamma_x ranges from 0 (no regularization) to 1 (full regularization).
    gamma_y: float | list[float] | np.ndarray (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along Y (channels). If
        None, no regularization is applied. The gamma_y ranges from 0 (no regularization) to 1 (full regularization).
    latency: np.ndarray (default: None)
        The raster latencies of each of the classes of shape (n_classes,) that the data/templates need to be corrected
        for.
    ensemble: bool (default: False)
        Whether or not to use an ensemble classifier, that is, a separate spatial filter for each class.
    cov_estimator_x: object (default: None)
        Estimator object with a fit method that estimates a covariance matrix of the EEG data. If None, a custom
        empirical covariance is used.
    cov_estimator_t: object (default: None)
        Estimator object with a fit method that estimates a covariance matrix of the EEG templates. If None, a custom
        empirical covariance is used.
    n_components: int (default: 1)
        The number of CCA components to use.

    Attributes
    ----------
    w_: np.ndarray
        The weight vector representing a spatial filter of shape (n_channels, n_components). If ensemble=True, then the
        shape is (n_channels, n_components, n_classes).
    T_: np.ndarray
        The template matrix representing the expected responses of shape (n_classes, n_components, n_samples).

    References
    ----------
    .. [1] Martínez-Cagigal, V., Thielen, J., Santamaria-Vazquez, E., Pérez-Velasco, S., Desain, P., & Hornero, R.
           (2021). Brain–computer interfaces based on code-modulated visual evoked potentials (c-VEP): A literature
           review. Journal of Neural Engineering, 18(6), 061002. doi: 10.1088/1741-2552/ac38cf
    """
    w_: np.ndarray
    T_: np.ndarray

    def __init__(
            self,
            lags: Union[None, np.ndarray],
            fs: int,
            cycle_size: float = None,
            template_metric: str = "mean",
            score_metric: str = "correlation",
            cca_channels: list[int] = None,
            gamma_x: Union[float, list[float], np.ndarray] = None,
            gamma_y: Union[float, list[float], np.ndarray] = None,
            latency: np.ndarray = None,
            ensemble: bool = False,
            cov_estimator_x: sklearn.base.BaseEstimator = None,
            cov_estimator_t: sklearn.base.BaseEstimator = None,
            n_components: int = 1,
    ) -> None:
        self.lags = lags
        self.fs = fs
        self.cycle_size = cycle_size
        self.template_metric = template_metric
        self.score_metric = score_metric
        self.cca_channels = cca_channels
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.latency = latency
        self.ensemble = ensemble
        self.cov_estimator_x = cov_estimator_x
        self.cov_estimator_t = cov_estimator_t
        self.n_components = n_components

    def _fit_T(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
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

    def decision_function(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """Apply the classifier to get classification scores for X.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        scores: np.ndarray
            The similarity scores of shape (n_trials, n_classes, n_components).
        """
        check_is_fitted(self, ["w_", "T_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)

        # Set templates to trial length
        T = self.get_T(X.shape[2])

        # Compute scores
        scores = np.zeros((X.shape[0], T.shape[0], self.n_components))
        if self.ensemble:
            for i_class in range(T.shape[0]):
                Xi = self._cca[i_class].transform(X=X)[0]
                for i_component in range(self.n_components):
                    if self.score_metric == "correlation":
                        scores[:, i_class, i_component] = correlation(Xi[:, i_component, :],
                                                                      T[i_class, i_component, :])[:, 0]
                    elif self.score_metric == "euclidean":
                        scores[:, i_class, i_component] = 1 / (1 + euclidean(Xi[:, i_component, :],
                                                                             T[i_class, i_component, :]))[:, 0]
                    elif self.score_metric == "inner":
                        scores[:, i_class, i_component] = np.inner(Xi[:, i_component, :],
                                                                   T[i_class, i_component, :])
                    else:
                        raise Exception(f"Unknown score metric: {self.score_metric}")

        else:
            X = self._cca.transform(X=X)[0]
            for i_component in range(self.n_components):
                if self.score_metric == "correlation":
                    scores[:, :, i_component] = correlation(X[:, i_component, :], T[:, i_component, :])
                elif self.score_metric == "euclidean":  # includes conversion to similarity
                    scores[:, :, i_component] = 1 / (1 + euclidean(X[:, i_component, :], T[:, i_component, :]))
                elif self.score_metric == "inner":
                    scores[:, :, i_component] = np.inner(X[:, i_component, :], T[:, i_component, :])
                else:
                    raise Exception(f"Unknown score metric: {self.score_metric}")

        return scores

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> sklearn.base.BaseEstimator:
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
            self.w_ = np.zeros((n_channels, self.n_components, n_classes))
            self._cca = []
            for i_class in range(n_classes):
                S = np.reshape(X[y == i_class, :, :].transpose((0, 2, 1)), (-1, n_channels))  # Concatenate trials
                R = np.tile(T[i_class, :, :].T, (np.sum(y == i_class), 1))  # Concatenate templates
                if self.cca_channels is not None:
                    R = R[:, self.cca_channels]
                self._cca.append(CCA(n_components=self.n_components, gamma_x=self.gamma_x, gamma_y=self.gamma_y,
                                     estimator_x=self.cov_estimator_x, estimator_y=self.cov_estimator_t))
                self._cca[i_class].fit(S, R)
                self.w_[:, :, i_class] = self._cca[i_class].w_x_
        else:
            S = np.reshape(X.transpose((0, 2, 1)), (-1, n_channels))  # Concatenate trials
            R = np.reshape(T[y, :, :].transpose((0, 2, 1)), (-1, n_channels))  # Concatenate templates
            if self.cca_channels is not None:
                R = R[:, self.cca_channels]
            self._cca = CCA(n_components=self.n_components, gamma_x=self.gamma_x, gamma_y=self.gamma_y,
                            estimator_x=self.cov_estimator_x, estimator_y=self.cov_estimator_t)
            self._cca.fit(S, R)
            self.w_ = self._cca.w_x_

        # Spatially filter templates
        if self.ensemble:
            self.T_ = np.zeros((n_classes, self.n_components, n_samples))
            for i_class in range(n_classes):
                self.T_[i_class, :, :] = self._cca[i_class].transform(X=None, Y=T[[i_class], :, :])[1]
        else:
            self.T_ = self._cca.transform(X=None, Y=T)[1]

        return self

    def get_T(
            self,
            n_samples: int = None,
    ) -> np.ndarray:
        """Get the templates.

        Parameters
        ----------
        n_samples: int (default: None)
            The number of samples requested. If None, one code cycle is given.

        Returns
        -------
        T: np.ndarray
            The templates of shape (n_classes, n_components, n_samples).
        """
        if n_samples is None or self.T_.shape[2] == n_samples:
            T = self.T_
        else:
            n = int(np.ceil(n_samples / self.T_.shape[2]))
            T = np.tile(self.T_, (1, 1, n))[:, :, :n_samples]
        T -= T.mean(axis=2, keepdims=True)
        return T

    def predict(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """The testing procedure to apply eCCA to novel EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: np.ndarray
            The predicted labels of shape (n_trials, n_components).
        """
        check_is_fitted(self, ["w_", "T_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)
        return np.argmax(self.decision_function(X), axis=1)


class Ensemble(BaseEstimator, ClassifierMixin):
    """Ensemble classifier. It wraps an ensemble classifier around another classifier object. The classifiers are
    applied to each item in a databank separately. A gating function combines the outputs of the individual
    classifications to arrive at a single final combined classification.

    Parameters
    ----------
    estimator: sklearn.base.BaseEstimator
        The classifier object that is applied to each item in the databank.
    gate: sklearn.base.BaseEstimator
        The gate that is used to combine the scores obtained from each individual estimator.

    Attributes
    ----------
    models_: list[sklearn.base.BaseEstimator]
        A list containing all models learned for each of the databanks.
    """
    models_: list[sklearn.base.BaseEstimator]

    def __init__(
            self,
            estimator: sklearn.base.BaseEstimator,
            gate: sklearn.base.BaseEstimator,
    ) -> None:
        self.estimator = estimator
        self.gate = gate

    def decision_function(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """Apply the classifier to get classification scores for X.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples, n_items).

        Returns
        -------
        scores: np.ndarray
            The matrix of scores of shape (n_trials, n_classes).
        """
        scores = np.stack([
            self.models_[i].decision_function(X[:, :, :, i])
            for i in range(X.shape[3])], axis=2)
        return self.gate.decision_function(scores)

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> sklearn.base.BaseEstimator:
        """The training procedure to apply an ensemble classifier on supervised EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples, n_items).
        y: np.ndarray
            The vector of ground-truth labels of the trials in X of shape (n_trials). Note, these denote the index at
            which to find the associated stimulus!

        Returns
        -------
        self: Ensemble
            An instance of the classifier.
        """
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True, y_numeric=True)
        y = y.astype(np.uint)
        assert X.ndim == 4

        # Fit separate models for each databank
        self.models_ = [
            copy.deepcopy(self.estimator).fit(X[:, :, :, i], y)
            for i in range(X.shape[3])]

        # Fit gating
        scores = np.stack([
            self.models_[i].decision_function(X[:, :, :, i])
            for i in range(X.shape[3])], axis=2)
        self.gate.fit(scores, y)

        return self

    def predict(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """The testing procedure to apply the ensemble classifier to novel EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples, n_items).

        Returns
        -------
        y: np.ndarray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, these denote the index at which
            to find the associated stimulus!
        """
        check_is_fitted(self, ["models_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)
        return self.gate.predict(self.decision_function(X))


class eTRCA(BaseEstimator, ClassifierMixin):
    """ERP TRCA classifier. It computes ERPs as templates for full sequences and performs a TRCA for spatial filtering
    [2]_.

    Parameters
    ----------
    lags: None | np.ndarray
        A vector of latencies in seconds per class relative to the first stimulus if stimuli are circularly shifted
        versions of the first stimulus, or None if all stimuli are different or this circular shift feature should be
        ignored.
    fs: int
        The sampling frequency of the EEG data in Hz.
    cycle_size: float (default: None)
        The time that one cycle of the code takes in seconds. If None, takes the full data length.
    template_metric: str (default: "mean")
        Metric to use to compute templates: mean, median, OCSVM.
    score_metric: str (default: "correlation")
        Metric to use to compute the overlap of templates and single-trials during testing: correlation, euclidean,
        inner.
    latency: np.ndarray (default: None)
        The raster latencies of each of the classes of shape (n_classes,) that the data/templates need to be corrected
        for.
    ensemble: bool (default: False)
        Whether or not to use an ensemble classifier, that is, a separate spatial filter for each class.

    Attributes
    ----------
    w_: np.ndarray
        The weight vector representing a spatial filter of shape (n_channels, n_components). If ensemble=True, then the
        shape is (n_channels, n_components, n_classes).
    T_: np.ndarray
        The template matrix representing the expected responses of shape (n_classes, n_samples, n_components).

    References
    ----------
    .. [2] Nakanishi, M., Wang, Y., Chen, X., Wang, Y. T., Gao, X., & Jung, T. P. (2017). Enhancing detection of SSVEPs
           for a high-speed brain speller using task-related component analysis. IEEE Transactions on Biomedical
           Engineering, 65(1), 104-112. doi: 10.1109/TBME.2017.2694818
    """
    w_: np.ndarray
    T_: np.ndarray

    def __init__(
            self,
            lags: Union[None, np.ndarray],
            fs: int,
            cycle_size: float = None,
            template_metric: str = "mean",
            score_metric: str = "correlation",
            latency: np.ndarray = None,
            ensemble: bool = False,
            n_components: int = 1,
    ) -> None:
        self.lags = lags
        self.fs = fs
        self.cycle_size = cycle_size
        self.template_metric = template_metric
        self.score_metric = score_metric
        self.latency = latency
        self.ensemble = ensemble
        self.n_components = n_components

    def _fit_T(
            self,
            X: np.ndarray,
    ) -> sklearn.base.BaseEstimator:
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

    def decision_function(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """Apply the classifier to get classification scores for X.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        scores: np.ndarray
            The similarity scores of shape (n_trials, n_classes, n_components).
        """
        check_is_fitted(self, ["w_", "T_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)

        # Set templates to trial length
        T = self.get_T(X.shape[2])

        # Compute scores
        scores = np.zeros((X.shape[0], T.shape[0], self.n_components))
        if self.ensemble:
            for i_class in range(T.shape[0]):
                Xi = self._trca[i_class].transform(X)
                for i_component in range(self.n_components):
                    if self.score_metric == "correlation":
                        scores[:, i_class, i_component] = correlation(Xi[:, i_component, :],
                                                                      T[i_class, i_component, :])[:, 0]
                    elif self.score_metric == "euclidean":  # includes conversion to similarity
                        scores[:, i_class, i_component] = 1 / (1 + euclidean(Xi[:, i_component, :],
                                                                             T[i_class, i_component, :]))[:, 0]
                    elif self.score_metric == "inner":
                        scores[:, i_class, i_component] = np.inner(Xi[:, i_component, :],
                                                                   T[i_class, i_component, :])
                    else:
                        raise Exception(f"Unknown score metric: {self.score_metric}")
        else:
            X = self._trca.transform(X)
            for i_component in range(self.n_components):
                if self.score_metric == "correlation":
                    scores[:, :, i_component] = correlation(X[:, i_component, :], T[:, i_component, :])
                elif self.score_metric == "euclidean":  # includes conversion to similarity
                    scores[:, :, i_component] = 1 / (1 + euclidean(X[:, i_component, :], T[:, i_component, :]))
                elif self.score_metric == "inner":
                    scores[:, :, i_component] = np.inner(X[:, i_component, :], T[:, i_component, :])
                else:
                    raise Exception(f"Unknown score metric: {self.score_metric}")

        return scores

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> sklearn.base.BaseEstimator:
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
        if self.latency is None:
            n_classes = np.unique(y).size
        else:
            n_classes = len(self.latency)
            X = correct_latency(X, y, -self.latency, self.fs, axis=-1)

        # Learn spatial filter
        if self.ensemble:
            self.w_ = np.zeros((n_channels, self.n_components, n_classes))
            self._trca = []
            for i_class in range(n_classes):
                self._trca.append(TRCA(n_components=self.n_components))
                self._trca[i_class].fit(X[y == i_class, :, :])
                self.w_[:, :, i_class] = self._trca[i_class].w_
        else:
            self._trca = TRCA(n_components=self.n_components)
            self._trca.fit(X)
            self.w_ = self._trca.w_

        # Spatially filter data
        if self.ensemble:
            Z = np.copy(X)
            X = np.zeros((Z.shape[0], self.n_components, Z.shape[2]))
            for i_class in range(n_classes):
                X[y == i_class, :, :] = self._trca[i_class].transform(Z[y == i_class, :, :])
            del Z
        else:
            X = self._trca.transform(X)

        # Synchronize all classes
        if self.lags is not None:
            X = correct_latency(X, y, -self.lags, self.fs, axis=-1)
            y = np.zeros(y.shape, y.dtype)

        # Cut trials to cycles
        if self.cycle_size is not None:
            cycle_size = int(self.cycle_size * self.fs)
            assert n_samples % cycle_size == 0, "X must be full cycles."
            n_cycles = int(n_samples / cycle_size)
            X = X.reshape((n_trials, self.n_components, n_cycles, cycle_size))
            X = X.transpose((0, 2, 1, 3))
            X = X.reshape((n_trials * n_cycles, self.n_components, cycle_size))
            n_trials, _, n_samples = X.shape
            y = np.repeat(y, n_cycles)

        # Compute templates
        if self.lags is None:
            # Compute a template per class separately
            classes = np.unique(y)
            n_classes = classes.size
            T = np.zeros((n_classes, self.n_components, n_samples))
            for i_class in range(n_classes):
                T[i_class, :, :] = self._fit_T(X[y == classes[i_class], :, :])
        else:
            # Compute a template for latency 0 and shift for all others
            n_classes = len(self.lags)
            T = np.tile(self._fit_T(X)[np.newaxis, :], (n_classes, 1, 1))
            T = correct_latency(T, np.arange(n_classes), self.lags, self.fs, axis=-1)
            if self.latency is not None:
                T = correct_latency(T, np.arange(n_classes), self.latency, self.fs, axis=-1)
        self.T_ = T

        return self

    def get_T(
            self,
            n_samples: int = None,
    ) -> np.ndarray:
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
        if n_samples is None or self.T_.shape[2] == n_samples:
            T = self.T_
        else:
            n = int(np.ceil(n_samples / self.T_.shape[2]))
            T = np.tile(self.T_, (1, 1, n))[:, :, :n_samples]
        T -= T.mean(axis=2, keepdims=True)
        return T

    def predict(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """The testing procedure to apply eTRCA to novel EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: np.ndarray
            The predicted labels of shape (n_trials, n_components).
        """
        check_is_fitted(self, ["w_", "T_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)
        return np.argmax(self.decision_function(X), axis=1)


class rCCA(BaseEstimator, ClassifierMixin):
    """Reconvolution CCA classifier. It performs a spatial and temporal decomposition (reconvolution [3]_) within a
    CCA [4]_ to perform spatial filtering as well as template prediction [5]_.

    Parameters
    ----------
    stimulus: np.ndarray
        The stimulus used for stimulation of shape (n_classes, n_samples). Should be sampled at fs. One cycle (i.e.,
        one stimulus-repetition) is sufficient.
    fs: int
        The sampling frequency of the EEG data in Hz.
    event: str (default: "duration")
        The event definition to map stimulus to events.
    onset_event: bool (default: False)
        Whether or not to add an event for the onset of stimulation. Added as last event.
    decoding_length: float (default: None)
        The length of the spectral filter for each data channel in seconds. If None, it is set to 1/fs, equivalent to 1
        sample, such that no phase-shifting is performed.
        thus no (spatio-)spectral filter is learned.
    decoding_stride: float (default: None)
        The stride of the spectral filter for each data channel in seconds. If None, it is set to 1/fs, equivalent to 1
        sample, such that no stride is used.
    encoding_length: float | list[float] (default: None)
        The length of the transient response(s) for each of the events in seconds. If None, it is set to 1/fs,
        equivalent to 1 sample, such that no phase-shifting is performed.
    encoding_stride: float | list[float] (default: None)
        The stride of the transient response(s) for each of the events in seconds. If None, it is set to 1/fs,
        equivalent to 1 sample, such that no stride is used.
    score_metric: str (default: "correlation")
        Metric to use to compute the overlap of templates and single-trials during testing: correlation, euclidean.
    latency: np.ndarray (default: None)
        The raster latencies of each of the classes of shape (n_classes,) that the data/templates need to be corrected
        for.
    ensemble: bool (default: False)
        Whether or not to use an ensemble classifier, that is, a separate spatial filter for each class.
    amplitudes: np.ndarray
        The amplitude of the stimulus of shape (n_classes, n_samples). Should be sampled at fs.
    gamma_x: float | list[float] | np.ndarray (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along X (channels). If
        None, no regularization is applied. The gamma_x ranges from 0 (no regularization) to 1 (full regularization).
    gamma_m: float | list[float] np.ndarray (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along M (samples). If None,
        no regularization is applied. The gamma_m ranges from 0 (no regularization) to 1 (full regularization).
    cov_estimator_x: BaseEstimator (default: None)
        A BaseEstimator object with a fit method that estimates a covariance matrix of the EEG data, the decoding
        matrix. If None, a custom empirical covariance is used.
    cov_estimator_m: BaseEstimator (default: None)
        A BaseEstimator object with a fit method that estimates a covariance matrix of the stimulus encoding matrix. If
        None, a custom empirical covariance is used.
    n_components: int (default: 1)
        The number of CCA components to use.

    Attributes
    ----------
    w_: np.ndarray
        The weight vector representing a spatial filter of shape (n_channels, n_components). If ensemble=True, then the
        shape is (n_channels, n_components, n_classes).
    r_: np.ndarray
        The weight vector representing a temporal filter of shape (n_events * n_event_samples, n_components). If
        ensemble=True, then the shape is (n_events * n_event_samples, n_components, n_classes).
    Ts_: np.ndarray
        The template matrix representing the expected responses of shape (n_classes, n_components, n_samples) for
        stimulus cycle 1 (i.e., it includes the onset of stimulation and does not contain the tails of previous cycles).
    Tw_: np.ndarray
        The template matrix representing the expected responses of shape (n_classes, n_components, n_samples) for
        stimulus cycles 2 and further (i.e., it does not include the onset of stimulation but does include the tails of
        previous cycles).

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
    w_: np.ndarray
    r_: np.ndarray
    Ts_: np.ndarray
    Tw_: np.ndarray

    def __init__(
            self,
            stimulus: np.ndarray,
            fs: int,
            event: str = "duration",
            onset_event: bool = False,
            decoding_length: float = None,
            decoding_stride: float = None,
            encoding_length: Union[float, list[float]] = None,
            encoding_stride: Union[float, list[float]] = None,
            score_metric: str = "correlation",
            latency: np.ndarray = None,
            ensemble: bool = False,
            amplitudes: np.ndarray = None,
            gamma_x: Union[float, list[float], np.ndarray] = None,
            gamma_m: Union[float, list[float], np.ndarray] = None,
            cov_estimator_x: sklearn.base.BaseEstimator = None,
            cov_estimator_m: sklearn.base.BaseEstimator = None,
            n_components: int = 1,
    ) -> None:
        self.stimulus = stimulus
        self.fs = fs
        self.event = event
        self.onset_event = onset_event
        if decoding_length is None:
            self.decoding_length = 1 / fs
        else:
            self.decoding_length = decoding_length
        if decoding_stride is None:
            self.decoding_stride = 1 / fs
        else:
            self.decoding_stride = decoding_stride
        if encoding_length is None:
            self.encoding_length = 1 / fs
        else:
            self.encoding_length = encoding_length
        if encoding_stride is None:
            self.encoding_stride = 1 / fs
        else:
            self.encoding_stride = encoding_stride
        self.score_metric = score_metric
        self.latency = latency
        self.ensemble = ensemble
        self.amplitudes = amplitudes
        self.gamma_x = gamma_x
        self.gamma_m = gamma_m
        self.cov_estimator_x = cov_estimator_x
        self.cov_estimator_m = cov_estimator_m
        self.n_components = n_components

    def _get_M(
            self,
            n_samples: int = None,
    ) -> np.ndarray:
        """Get the encoding matrix of a particular length.

        Parameters
        ----------
        n_samples: int (default: None)
            The number of samples that the encoding matrix should be. If none, one stimulus cycle is used.

        Returns
        -------
        M: np.ndarray
            The encoding matrix denoting event timings of shape (n_classes, encoding_length, n_samples).
        """
        # Repeat the stimulus to n_samples
        if n_samples is None or self.stimulus.shape[1] == n_samples:
            stimulus = self.stimulus
        else:
            n = int(np.ceil(n_samples / self.stimulus.shape[1]))
            stimulus = np.tile(self.stimulus, (1, n))

        # Repeat the amplitudes to n_samples
        if n_samples is None or self.amplitudes is None or self.amplitudes.shape[1] == n_samples:
            amplitudes = self.amplitudes
        else:
            n = int(np.ceil(n_samples / self.amplitudes.shape[1]))
            amplitudes = np.tile(self.amplitudes, (1, n))

        # Get encoding matrices
        E, self.events_ = event_matrix(stimulus, self.event, self.onset_event)
        M = encoding_matrix(E, int(self.encoding_length * self.fs), int(self.encoding_stride * self.fs), amplitudes)
        return M[:, :, :n_samples]

    def decision_function(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """Apply the classifier to get classification scores for X.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        scores: np.ndarray
            The similarity scores of shape (n_trials, n_classes, n_components).
        """
        check_is_fitted(self, ["w_", "r_", "Ts_", "Tw_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)

        # Set decoding matrix
        X = decoding_matrix(X, int(self.decoding_length * self.fs), int(self.decoding_stride * self.fs))

        # Set templates to trial length
        T = self.get_T(X.shape[2])

        # Compute scores
        scores = np.zeros((X.shape[0], T.shape[0], self.n_components))
        if self.ensemble:
            for i_class in range(T.shape[0]):
                Xi = self._cca[i_class].transform(X=X)[0]
                for i_component in range(self.n_components):
                    if self.score_metric == "correlation":
                        scores[:, i_class, i_component] = correlation(Xi[:, i_component, :],
                                                                      T[i_class, i_component, :])[:, 0]
                    elif self.score_metric == "euclidean":  # includes conversion to similarity
                        scores[:, i_class, i_component] = 1 / (1 + euclidean(Xi[:, i_component, :],
                                                                             T[i_class, i_component, :]))[:, 0]
                    elif self.score_metric == "inner":
                        scores[:, i_class, i_component] = np.inner(Xi[:, i_component, :],
                                                                   T[i_class, i_component, :])
                    else:
                        raise Exception(f"Unknown score metric: {self.score_metric}")

        else:
            X = self._cca.transform(X=X)[0]
            for i_component in range(self.n_components):
                if self.score_metric == "correlation":
                    scores[:, :, i_component] = correlation(X[:, i_component, :], T[:, i_component, :])
                elif self.score_metric == "euclidean":  # includes conversion to similarity
                    scores[:, :, i_component] = 1 / (1 + euclidean(X[:, i_component, :], T[:, i_component, :]))
                elif self.score_metric == "inner":
                    scores[:, :, i_component] = np.inner(X[:, i_component, :], T[:, i_component, :])
                else:
                    raise Exception(f"Unknown score metric: {self.score_metric}")

        return scores

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> sklearn.base.BaseEstimator:
        """The training procedure to fit a rCCA on supervised EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: np.ndarray
            The vector of ground-truth labels of the trials in X of shape (n_trials). Note, these denote the index at
            which to find the associated stimulus!

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
            X = correct_latency(X, y, -self.latency, self.fs, axis=2)

        # Set decoding matrix
        X = decoding_matrix(X, int(self.decoding_length * self.fs), int(self.decoding_stride * self.fs))

        # Get encoding window
        M = self._get_M(X.shape[2])

        # Fit w and r
        if self.ensemble:
            self.w_ = np.zeros((X.shape[1], self.n_components, n_classes))
            self.r_ = np.zeros((M.shape[1], self.n_components, n_classes))
            self._cca = []
            for i_class in range(n_classes):
                self._cca.append(CCA(n_components=self.n_components, gamma_x=self.gamma_x, gamma_y=self.gamma_m,
                                     estimator_x=self.cov_estimator_x, estimator_y=self.cov_estimator_m))
                self._cca[i_class].fit(X[y == i_class, :, :], np.tile(M[[i_class], :, :], (np.sum(y == i_class), 1, 1)))
                self.w_[:, :, i_class] = self._cca[i_class].w_x_
                self.r_[:, :, i_class] = self._cca[i_class].w_y_
        else:
            self._cca = CCA(n_components=self.n_components, gamma_x=self.gamma_x, gamma_y=self.gamma_m,
                            estimator_x=self.cov_estimator_x, estimator_y=self.cov_estimator_m)
            self._cca.fit(X, M[y, :, :])
            self.w_ = self._cca.w_x_
            self.r_ = self._cca.w_y_

        # Set templates (start and wrapper)
        M = self._get_M(2 * self.stimulus.shape[1])
        if self.ensemble:
            T = np.zeros((n_classes, self.n_components, M.shape[2]))
            for i_class in range(n_classes):
                T[i_class, :, :] = self._cca[i_class].transform(X=None, Y=M[[i_class], :, :])[1]
        else:
            T = self._cca.transform(X=None, Y=M)[1]
        self.Ts_ = T[:, :, :self.stimulus.shape[1]]
        self.Tw_ = T[:, :, self.stimulus.shape[1]:]

        # Correct for raster latency
        if self.latency is not None:
            self.Ts_ = correct_latency(self.Ts_, np.arange(len(self.latency)), self.latency, self.fs, axis=2)
            self.Tw_ = correct_latency(self.Tw_, np.arange(len(self.latency)), self.latency, self.fs, axis=2)

        return self

    def get_T(
            self,
            n_samples: int = None,
    ) -> np.ndarray:
        """Get the templates.

        Parameters
        ----------
        n_samples: int (default: None)
            The number of samples requested. If None, one stimulus cycle is used.

        Returns
        -------
        T: np.ndarray
            The templates of shape (n_classes, n_components, n_samples).
        """
        if n_samples is None or self.Ts_.shape[2] == n_samples:
            T = self.Ts_
        else:
            n = int(np.ceil(n_samples / self.Ts_.shape[2]))
            if (n - 1) > 1:
                Tw = np.tile(self.Tw_, (1, 1, (n - 1)))
            else:
                Tw = self.Tw_
            T = np.concatenate((self.Ts_, Tw), axis=2)[:, :, :n_samples]
        T -= T.mean(axis=2, keepdims=True)
        return T

    def predict(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """The testing procedure to apply rCCA to novel EEG data.

        Parameters
        ----------
        X: np.ndarray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        y: np.ndarray
            The predicted labels of shape (n_trials, n_components).
        """
        check_is_fitted(self, ["w_", "r_", "Ts_", "Tw_"])
        X = check_array(X, ensure_2d=False, allow_nd=True)
        return np.argmax(self.decision_function(X), axis=1)

    def set_stimulus(
            self,
            stimulus: np.ndarray,
    ) -> None:
        """Set the stimulus, and as such change the templates.

        Parameters
        ----------
        stimulus: np.ndarray
            The stimulus used for stimulation of shape (n_classes, n_samples). Should be sampled at fs. One cycle (i.e.,
            one stimulus-repetition) is sufficient. If None, it is not changed.
        """
        self.set_stimulus_amplitudes(stimulus, self.amplitudes)

    def set_amplitudes(
            self,
            amplitudes: np.ndarray,
    ) -> None:
        """Set the amplitudes, and as such change the templates.

        Parameters
        ----------
        amplitudes: np.ndarray
            The amplitude of the stimulus of shape (n_classes, n_samples). Should be sampled at fs. If None, it is not
            changed.
        """
        self.set_stimulus_amplitudes(self.stimulus, amplitudes)

    def set_stimulus_amplitudes(
            self,
            stimulus: np.ndarray,
            amplitudes: np.ndarray,
    ) -> None:
        """Set the stimulus and/or the amplitudes, and as such change the templates.

        Parameters
        ----------
        stimulus: np.ndarray
            The stimulus used for stimulation of shape (n_classes, n_samples). Should be sampled at fs. One cycle (i.e.,
            one stimulus-repetition) is sufficient. If None, it is not changed.
        amplitudes: np.ndarray
            The amplitude of the stimulus of shape (n_classes, n_samples). Should be sampled at fs. If None, it is not
            changed.
        """
        self.stimulus = stimulus
        self.amplitudes = amplitudes
        T = self._cca.transform(X=None, Y=self._get_M(2 * self.stimulus.shape[1]))[1]
        self.Ts_ = T[:, :, :self.stimulus.shape[1]]
        self.Tw_ = T[:, :, self.stimulus.shape[1]:]
