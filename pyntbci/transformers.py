from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, sqrtm, svd, inv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pyntbci.utilities import covariance, pinv


class CCA(BaseEstimator, TransformerMixin):
    """Canonical correlation analysis (CCA). Maximizes the correlation between two variables in their projected spaces.
    Here, CCA is implemented as the SVD of (cross)covariance matrices [1]_.

    Parameters
    ----------
    n_components: int
        The number of CCA components to use.
    gamma_x: float | list[float] | NDArray (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along n_features_x. If
        None, no regularization is applied. The gamma_x ranges from 0 (no regularization) to 1 (full regularization).
    gamma_y: float | list[float] | NDArray (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along n_features_y. If
        None, no regularization is applied. The gamma_y ranges from 0 (no regularization) to 1 (full regularization).
    estimator_x: BaseEstimator (default: None)
        A BaseEstimator object that estimates a covariance matrix for X using a fit method. If None, a custom
        implementation of the empirical covariance is used.
    estimator_y: BaseEstimator (default: None)
        A BaseEstimator object that estimates a covariance matrix for Y using a fit method. If None, a custom
        implementation of the empirical covariance is used.
    running: bool (default: False)
        If False, the CCA is instantaneous, only fit to the current data. If True, the CCA is incremental and keeps
        track of previous data to update a running average and covariance for the CCA.
    alpha_x: float (default: None)
        Amount of variance to retain in computing the inverse of the covariance matrix of X. If None, all variance.
    alpha_y: float (default: None)
        Amount of variance to retain in computing the inverse of the covariance matrix of Y. If None, all variance.

    Attributes
    ----------
    w_x_: NDArray
        The weight vector to project X of shape (n_features_x, n_components).
    w_y_: NDArray
        The weight vector to project Y of shape (n_features_y, n_components).
    n_x_: int
        The number of samples used to estimate avg_x_ and cov_x.
    avg_x_: NDArray
        The (running) average of X of shape (n_features_x).
    cov_x_: NDArray
        The (running) covariance of X of shape (n_features_x, n_features_x).
    n_y_: int
        The number of samples used to estimate avg_y_ and cov_y.
    avg_y_: NDArray
        The (running) average of Y of shape (n_features_y).
    cov_y_: NDArray
        The (running) covariance of Y of shape (n_features_y, n_features_y).
    n_xy_: int
        The number of samples used to estimate avg_xy_ and cov_xy.
    avg_xy_: NDArray
        The (running) average of concat(X, Y) of shape (n_features_x + n_features_y).
    cov_xy_: NDArray
        The (running) cross-covariance of X and Y of shape (n_features_x, n_features_y).

    References
    ----------
    .. [1] Hotelling, H. (1992). Relations between two sets of variates. In Breakthroughs in statistics: methodology and
           distribution (pp. 162-190). New York, NY: Springer New York. doi: 10.1007/978-1-4612-4380-9_14
    """
    w_x_: NDArray
    w_y_: NDArray
    n_x_: int = 0
    avg_x_: NDArray = None
    cov_x_: NDArray = None
    n_y_: int = 0
    avg_y_: NDArray = None
    cov_y_: NDArray = None
    n_xy_: int = 0
    avg_xy_: NDArray = None
    cov_xy_: NDArray = None

    def __init__(
            self,
            n_components: int,
            gamma_x: Union[float, list[float], NDArray] = None,
            gamma_y: Union[float, list[float], NDArray] = None,
            estimator_x: BaseEstimator = None,
            estimator_y: BaseEstimator = None,
            running: bool = False,
            alpha_x: float = None,
            alpha_y: float = None,
    ) -> None:
        self.n_components = n_components
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.estimator_x = estimator_x
        self.estimator_y = estimator_y
        self.running = running
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

    def _fit_X2D_Y2D(
            self,
            X: NDArray,
            Y: NDArray,
    ) -> None:
        """Fit the CCA for a 2D X data matrix and 2D Y data matrix.

        Parameters
        ----------
        X: NDArray
            Data matrix of shape (n_samples, n_features_x).
        Y: NDArray
            Data matrix of shape (n_samples, n_features_y).
        """
        assert X.shape[0] == Y.shape[0], f"Unequal samples in X ({X.shape[0]}) and Y ({Y.shape[0]})!"

        # Compute covariances
        Z = np.concatenate((X, Y), axis=1)
        self.n_xy_, self.avg_xy_, self.cov_xy_ = covariance(Z, self.n_xy_, self.avg_xy_, self.cov_xy_,
                                                            estimator=None, running=self.running)
        if self.estimator_x is None:
            self.n_x_ = self.n_xy_
            self.avg_x_ = self.avg_xy_[:, :X.shape[1]]
            self.cov_x_ = self.cov_xy_[:X.shape[1], :X.shape[1]]
        else:
            self.n_x_, self.avg_x_, self.cov_x_ = covariance(X, self.n_x_, self.avg_x_, self.cov_x_,
                                                             estimator=self.estimator_x, running=self.running)
        if self.estimator_y is None:
            self.n_y_ = self.n_xy_
            self.avg_y_ = self.avg_xy_[:, X.shape[1]:]
            self.cov_y_ = self.cov_xy_[X.shape[1]:, X.shape[1]:]
        else:
            self.n_y_, self.avg_y_, self.cov_y_ = covariance(Y, self.n_y_, self.avg_y_, self.cov_y_,
                                                             estimator=self.estimator_y, running=self.running)
        Cxx = self.cov_x_
        Cyy = self.cov_y_
        Cxy = self.cov_xy_[:X.shape[1], X.shape[1]:]

        # Regularization
        if self.gamma_x is not None:
            nu_x = np.trace(Cxx) / Cxx.shape[1]
            if isinstance(self.gamma_x, int) or isinstance(self.gamma_x, float):
                gamma_x = np.full(Cxx.shape[1], self.gamma_x)
            elif np.array(self.gamma_x).ndim == 1:
                gamma_x = np.array(self.gamma_x)[np.newaxis, :]
            else:
                gamma_x = self.gamma_x
            Cxx = (1 - gamma_x) * Cxx + nu_x * np.diag(gamma_x)
        if self.gamma_y is not None:
            nu_y = np.trace(Cyy) / Cyy.shape[1]
            if isinstance(self.gamma_y, int) or isinstance(self.gamma_y, float):
                gamma_y = np.full(Cyy.shape[1], self.gamma_y)
            elif np.array(self.gamma_y).ndim == 1:
                gamma_y = np.array(self.gamma_y)[np.newaxis, :]
            else:
                gamma_y = self.gamma_y
            Cyy = (1 - gamma_y) * Cyy + nu_y * np.diag(gamma_y)

        # Inverse square root
        if self.alpha_x is None:
            iCxx = np.real(sqrtm(inv(Cxx)))
        else:
            iCxx = np.real(sqrtm(pinv(Cxx, self.alpha_x)))
        if self.alpha_y is None:
            iCyy = np.real(sqrtm(inv(Cyy)))
        else:
            iCyy = np.real(sqrtm(pinv(Cyy, self.alpha_y)))

        # SVD
        U, self.rho_, V = svd(iCxx @ Cxy @ iCyy)

        # Compute projection vectors
        Wx = iCxx @ U
        Wy = iCyy @ V.T

        # Select components
        self.w_x_ = Wx[:, :self.n_components]
        self.w_y_ = Wy[:, :self.n_components]

    def _fit_X3D_Y3D(
            self,
            X: NDArray,
            Y: NDArray,
    ) -> None:
        """Fit the CCA for a 3D X data matrix and 3D Y data matrix.

        Parameters
        ----------
        X: NDArray
            Data matrix of shape (n_trials, n_features_x, n_samples).
        Y: NDArray
            Data matrix of shape (n_trials, n_features_y, n_samples).
        """
        assert X.shape[0] == Y.shape[0], f"Unequal trials in X ({X.shape[0]}) and Y ({Y.shape[0]})!"
        assert X.shape[2] == Y.shape[2], f"Unequal samples in X ({X.shape[2]}) and Y ({Y.shape[2]})!"

        n_trials, n_features_x, n_samples = X.shape
        n_features_y = Y.shape[1]

        # Create aligned matrices
        X = X.transpose((0, 2, 1)).reshape((n_samples * n_trials, n_features_x))
        Y = Y.transpose((0, 2, 1)).reshape((n_samples * n_trials, n_features_y))

        # CCA
        self._fit_X2D_Y2D(X, Y)

    def _fit_X3D_Y1D(
            self,
            X: NDArray,
            Y: NDArray,
    ) -> None:
        """Fit the CCA for a 3D X data matrix and 1D Y label vector.

        Parameters
        ----------
        X: NDArray
            Data matrix of shape (n_trials, n_features_x, n_samples).
        Y: NDArray
            Label vector of shape (n_trials,).
        """
        assert X.shape[0] == Y.shape[0], f"Unequal trials in X ({X.shape[0]}) and Y ({Y.shape[0]})!"

        n_trials, n_channels, n_samples = X.shape
        labels = np.unique(Y)
        n_classes = labels.size

        # Compute templates
        T = np.zeros((n_classes, n_channels, n_samples))
        for i, label in enumerate(labels):
            T[i, :, :] = np.mean(X[Y == labels[i], :, :], axis=0)

        # CCA
        self._fit_X3D_Y3D(X, T[Y, :, :])

    def fit(
            self,
            X: NDArray,
            Y: NDArray,
    ) -> TransformerMixin:
        """Fit the CCA in one of 3 ways: (1) X (data) is 3D and y (labels) is 1D, (2) X (data) is 3D and Y (data) is 3D,
        or (3) X (data) is 2D and Y (data) is 2D.

        Parameters
        ----------
        X: NDArray
            Data matrix of shape (n_trials, n_features_x, n_samples) or (n_samples, n_features_x).
        Y: NDArray
            Data matrix of shape (n_trials, n_features_y, n_samples) or (n_samples, n_features_y), or label vector of
            shape (n_trials,).

        Returns
        -------
        self: TransformerMixin
            Returns the instance itself.
        """
        if X.ndim == 3 and Y.ndim == 1:
            self._fit_X3D_Y1D(X, Y)
        elif X.ndim == 3 and Y.ndim == 3:
            self._fit_X3D_Y3D(X, Y)
        elif X.ndim == 2 and Y.ndim == 2:
            self._fit_X2D_Y2D(X, Y)
        else:
            raise Exception(f"Dimensions of X ({X.shape}) and/or Y ({Y.shape}) are not valid.")

        return self

    def _transform_X2D(
            self,
            X: NDArray = None,
            Y: NDArray = None,
    ) -> tuple[NDArray, NDArray]:
        """Transform the 2D data matrix from feature space to component space.

        Parameters
        ----------
        X: NDArray
            Data matrix of shape (n_samples, n_features_x).
        Y: NDArray
            Data matrix of shape (n_samples, n_features_y).

        Returns
        -------
        X: NDArray
            Projected data matrix of shape (n_samples, n_components).
        Y: NDArray
            Projected data matrix of shape (n_samples, n_components).
        """
        if X is not None:
            X -= self.avg_x_
            X = np.dot(X, self.w_x_)
        if Y is not None:
            Y -= self.avg_y_
            Y = np.dot(Y, self.w_y_)

        return X, Y

    def _transform_X3D(
            self,
            X: NDArray = None,
            Y: NDArray = None,
    ) -> tuple[NDArray, NDArray]:
        """Transform the 3D data matrix from feature space to component space.

        Parameters
        ----------
        X: NDArray
            Data matrix of shape (n_trials, n_features_x, n_samples).
        Y: NDArray
            Data matrix of shape (n_trials, n_features_y, n_samples).

        Returns
        -------
        X: NDArray
            Projected data matrix of shape (n_trials, n_components, n_samples).
        Y: NDArray
            Projected data matrix of shape (n_trials, n_components, n_samples).
        """
        if X is not None:
            n_trials, n_features_x, n_samples = X.shape
            X = X.transpose((0, 2, 1)).reshape((n_trials * n_samples, n_features_x))
        if Y is not None:
            n_trials, n_features_y, n_samples = Y.shape
            Y = Y.transpose((0, 2, 1)).reshape((n_trials * n_samples, n_features_y))

        X, Y = self._transform_X2D(X, Y)

        if X is not None:
            X = X.reshape((n_trials, n_samples, self.n_components)).transpose(0, 2, 1)
        if Y is not None:
            Y = Y.reshape((n_trials, n_samples, self.n_components)).transpose(0, 2, 1)

        return X, Y

    def transform(
            self,
            X: NDArray = None,
            Y: NDArray = None,
    ) -> tuple[NDArray, NDArray]:
        """Transform the data matrix from feature space to component space. Note, works with both 2D and 3D data, and
        can operate on both X and Y if both are not None, or on each separately if the other is None.

        Parameters
        ----------
        X: NDArray (default: None)
            Data matrix of shape (n_samples, n_features_x) or (n_trials, n_features_x, n_samples). If None, only
            performs projection of Y.
        Y: NDArray (default: None)
            Data matrix of shape (n_samples, n_features_y) or (n_trials, n_features_y, n_samples). If None, only
            performs projection of X.

        Returns
        -------
        X: NDArray
            Projected data matrix of shape (n_samples, n_components) or (n_trials, n_components, n_samples). None if the
            input was None.
        Y: NDArray
            Projected data matrix of shape (n_samples, n_components) or (n_trials, n_components, n_samples). None if the
            input was None.
        """
        check_is_fitted(self, ["w_x_", "w_y_", "rho_"])

        if (X is None or X.ndim == 3) and (Y is None or Y.ndim == 3):
            X, Y = self._transform_X3D(X, Y)
        elif (X is None or X.ndim == 2) and (Y is None or Y.ndim == 2):
            X, Y = self._transform_X2D(X, Y)
        else:
            raise Exception("X and Y must be both 3D or 2D, or None.")

        return X, Y


class TRCA(BaseEstimator, TransformerMixin):
    """Task related component analysis (TRCA). Maximizes the intra-class covariances, i.e., the intra-class consistency
    [2]_. TRCA was applied to (SSVEP) BCI [3]_. Alternative implementations, also used as example for this code, see
    Matlab code in [2]_ for the original, Matlab code in [4]_ for the SSVEP BCI introduction, and two Python
    implementation in MOABB [5]_, and MEEGKit [6]_.

    Parameters
    ----------
    n_components: int (default: 1)
        The number of TRCA components to use.

    Attributes
    ----------
    w_: NDArray
        The weight vector to project X of shape (n_features, n_components).

    References
    ----------
    .. [2] Tanaka, H., Katura, T., & Sato, H. (2013). Task-related component analysis for functional neuroimaging and
           application to near-infrared spectroscopy data. NeuroImage, 64, 308-327.
           doi: 10.1016/j.neuroimage.2012.08.044
    .. [3] Nakanishi, M., Wang, Y., Chen, X., Wang, Y. T., Gao, X., & Jung, T. P. (2017). Enhancing detection of SSVEPs
           for a high-speed brain speller using task-related component analysis. IEEE Transactions on Biomedical
           Engineering, 65(1), 104-112. doi: 10.1109/TBME.2017.2694818
    .. [4] https://github.com/mnakanishi/TRCA-SSVEP/blob/master/src/train_trca.m
    .. [5] https://github.com/NeuroTechX/moabb/blob/develop/moabb/pipelines/classification.py
    .. [6] https://github.com/nbara/python-meegkit/blob/master/meegkit/trca.py
    """
    w_: NDArray

    def __init__(
            self,
            n_components: int = 1,
    ) -> None:
        self.n_components = n_components

    def fit(
            self,
            X: NDArray,
            y: NDArray = None,
    ) -> TransformerMixin:
        """Fit TRCA.

        Parameters
        ----------
        X: NDArray
            Data matrix of shape (n_trials, n_features, n_samples).
        y: NDArray (default: None)
            Not used.

        Returns
        -------
        self: TransformerMixin
            Returns the instance itself.
        """
        n_trials, n_channels, n_samples = X.shape

        # Covariance of all data
        Xa = X.transpose((1, 0, 2)).reshape((n_channels, n_trials * n_samples))
        Xa -= Xa.mean(axis=1, keepdims=True)
        Q = Xa.dot(Xa.T)

        # Covariance of pairs of trials
        S = np.zeros((n_channels, n_channels))
        for i_trial in range(n_trials - 1):
            Xi = X[i_trial, :, :]
            Xi -= Xi.mean(axis=1, keepdims=True)
            for j_trial in range(1 + i_trial, n_trials):
                Xj = X[j_trial, :, :]
                Xj -= Xj.mean(axis=1, keepdims=True)
                S += (Xi.dot(Xj.T) + Xj.dot(Xi.T))

        # Eigenvalue decomposition
        D, V = eigh(S, Q)
        self.w_ = V[:, np.argsort(D)[::-1][:self.n_components]]

        return self

    def transform(
            self,
            X: NDArray,
            y: NDArray = None,
    ) -> NDArray:
        """Transform the data matrix from feature space to component space. Note, can operate on both X and y or just X.
        If X and y are provided, X is filtered with class-specific filters. If only X is provided and a multi-class
        filter was learned, all trials are filtered with all filters. If only one filter was learned, then only this
        filter is applied.

        Parameters
        ----------
        X: NDArray
            Data matrix of shape (n_trials, n_features, n_samples).
        y: NDArray (default: None)
            Not used.

        Returns
        -------
        X: NDArray
            Projected data matrix of shape (n_trials, n_components, n_samples).
        """
        check_is_fitted(self, ["w_"])
        n_trials, n_channels, n_samples = X.shape

        X = X.transpose((0, 2, 1))
        X = X.reshape((n_trials * n_samples, n_channels))
        X = np.dot(X, self.w_)
        X = X.reshape((n_trials, n_samples, self.n_components))
        X = X.transpose((0, 2, 1))

        return X
