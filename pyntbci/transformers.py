import numpy as np
from scipy.linalg import eigh, inv, sqrtm, svd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from pyntbci.utilities import covariance


class CCA(BaseEstimator, TransformerMixin):
    """Canonical correlation analysis (CCA). Maximizes the correlation between two variables in their projected spaces.
    Here, CCA is implemented as the SVD of (cross)covariance matrices [1]_.

    Parameters
    ----------
    n_components: int
        The number of CCA components to use.
    lx: float | list (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along n_features_x. If
        None, no regularization is applied.
    ly: float | list (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along n_features_y. If
        None, no regularization is applied.
    estimator_x: object (Default: None)
        An object that estimates a covariance matrix for X using a fit method. If None, a custom implementation of the
        empirical covariance is used.
    estimator_y
        An object that estimates a covariance matrix for Y using a fit method. If None, a custom implementation of the
        empirical covariance is used.
    running: bool (default: False)
        If False, the CCA is instantaneous, only fit to the current data. If True, the CCA is incremental and keeps
        track of previous data to update a running average and covariance for the CCA.

    References
    ----------
    .. [1] Hotelling, H. (1992). Relations between two sets of variates. In Breakthroughs in statistics: methodology and
           distribution (pp. 162-190). New York, NY: Springer New York. doi: 10.1007/978-1-4612-4380-9_14
    """

    def __init__(self, n_components, lx=None, ly=None, estimator_x=None, estimator_y=None, running=False):
        self.n_components = n_components
        self.lx = lx
        self.ly = ly
        self.estimator_x = estimator_x
        self.estimator_y = estimator_y
        self.running = running
        self.n_x_ = 0
        self.avg_x_ = None
        self.cov_x_ = None
        self.n_y_ = 0
        self.avg_y_ = None
        self.cov_y_ = None
        self.n_xy_ = 0
        self.avg_xy_ = None
        self.cov_xy_ = None

    def _fit_X2D_Y2D(self, X, Y):
        """Fit the CCA for a 2D X data matrix and 2D Y data matrix.

        Parameters
        ----------
        X: np.ndarray
            Data matrix of shape (n_samples, n_features_x).
        Y: np.ndarray
            Data matrix of shape (n_samples, n_features_y).
        """
        X = check_array(X, ensure_2d=True, allow_nd=False)
        Y = check_array(Y, ensure_2d=True, allow_nd=False)
        assert X.shape[0] == Y.shape[0], f"Unequal samples in X ({X.shape[0]}) and Y ({Y.shape[0]})!"
        X = X.astype("float")
        Y = Y.astype("float")

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
        if self.lx is None:
            lx = np.zeros((1, X.shape[1]))
        elif isinstance(self.lx, int) or isinstance(self.lx, float):
            lx = self.lx * np.ones((1, X.shape[1]))
        elif np.array(self.lx).ndim == 1:
            lx = np.array(self.lx)[np.newaxis, :]
        else:
            lx = self.lx
        if self.ly is None:
            ly = np.zeros((1, Y.shape[1]))
        elif isinstance(self.ly, int) or isinstance(self.ly, float):
            ly = self.ly * np.ones((1, Y.shape[1]))
        elif np.array(self.ly).ndim == 1:
            ly = np.array(self.ly)[np.newaxis, :]
        else:
            ly = self.ly
        Cxx += lx @ np.eye(X.shape[1])
        Cyy += ly @ np.eye(Y.shape[1])

        # Inverse square root
        iCxx = np.real(inv(sqrtm(Cxx)))
        iCyy = np.real(inv(sqrtm(Cyy)))

        # SVD
        U, self.rho_, V = svd(iCxx @ Cxy @ iCyy)

        # Compute projection vectors
        Wx = iCxx @ U
        Wy = iCyy @ V.T

        # Select components
        self.w_x_ = Wx[:, :self.n_components]
        self.w_y_ = Wy[:, :self.n_components]

    def _fit_X3D_Y3D(self, X, Y):
        """Fit the CCA for a 3D X data matrix and 3D Y data matrix.

        Parameters
        ----------
        X: np.ndarray
            Data matrix of shape (n_trials, n_features_x, n_samples).
        Y: np.ndarray
            Data matrix of shape (n_trials, n_features_y, n_samples).
        """
        X = check_array(X, ensure_2d=False, allow_nd=True)
        Y = check_array(Y, ensure_2d=False, allow_nd=True)
        X = X.astype("float")
        Y = Y.astype("float")
        assert X.shape[0] == Y.shape[0], f"Unequal trials in X ({X.shape[0]}) and Y ({Y.shape[0]})!"
        assert X.shape[2] == Y.shape[2], f"Unequal samples in X ({X.shape[2]}) and Y ({Y.shape[2]})!"

        n_trials, n_features_x, n_samples = X.shape
        n_features_y = Y.shape[1]

        # Create aligned matrices
        X = X.transpose((0, 2, 1)).reshape((n_samples * n_trials, n_features_x))
        Y = Y.transpose((0, 2, 1)).reshape((n_samples * n_trials, n_features_y))

        # CCA
        self._fit_X2D_Y2D(X, Y)

    def _fit_X3D_Y1D(self, X, Y):
        """Fit the CCA for a 3D X data matrix and 1D Y label vector.

        Parameters
        ----------
        X: np.ndarray
            Data matrix of shape (n_trials, n_features_x, n_samples).
        Y: np.ndarray
            Label vector of shape (n_trials,).
        """
        X, Y = check_X_y(X, Y, ensure_2d=False, allow_nd=True, y_numeric=True)
        X = X.astype("float")
        Y = Y.astype(np.uint)
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

    def fit(self, X, Y):
        """Fit the CCA in one of 3 ways: (1) X (data) is 3D and y (labels) is 1D, (2) X (data) is 3D and Y (data) is 3D,
        or (3) X (data) is 2D and Y (data) is 2D.

        Parameters
        ----------
        X: np.ndarray
            Data matrix of shape (n_trials, n_features_x, n_samples) or (n_samples, n_features_x).
        Y: np.ndarray
            Data matrix of shape (n_trials, n_features_y, n_samples) or (n_samples, n_features_y), or label vector of
            shape (n_trials,).

        Returns
        -------
        self: CCA
            An instance of the transformer.
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

    def _transform_X2D(self, X=None, Y=None):
        """Transform the 2D data matrix from feature space to component space.

        Parameters
        ----------
        X: np.ndarray
            Data matrix of shape (n_samples, n_features_x).
        Y: np.ndarray
            Data matrix of shape (n_samples, n_features_y).

        Returns
        -------
        X: np.ndarray
            Projected data matrix of shape (n_samples, n_components).
        Y: np.ndarray
            Projected data matrix of shape (n_samples, n_components).
        """
        if X is not None:
            X = check_array(X, ensure_2d=True, allow_nd=False)
            X = X.astype("float")
            X -= self.avg_x_
            X = np.dot(X, self.w_x_)
        if Y is not None:
            Y = check_array(Y, ensure_2d=True, allow_nd=False)
            Y = Y.astype("float")
            Y -= self.avg_y_
            Y = np.dot(Y, self.w_y_)

        return X, Y

    def _transform_X3D(self, X=None, Y=None):
        """Transform the 3D data matrix from feature space to component space.

        Parameters
        ----------
        X: np.ndarray
            Data matrix of shape (n_trials, n_features_x, n_samples).
        Y: np.ndarray
            Data matrix of shape (n_trials, n_features_y, n_samples).

        Returns
        -------
        X: np.ndarray
            Projected data matrix of shape (n_trials, n_components, n_samples).
        Y: np.ndarray
            Projected data matrix of shape (n_trials, n_components, n_samples).
        """
        if X is not None:
            X = check_array(X, ensure_2d=False, allow_nd=True)
            X = X.astype("float")
            n_trials, n_features_x, n_samples = X.shape
            X = X.transpose((0, 2, 1)).reshape((n_trials * n_samples, n_features_x))
        if Y is not None:
            Y = check_array(Y, ensure_2d=False, allow_nd=True)
            Y = Y.astype("float")
            n_trials, n_features_y, n_samples = Y.shape
            Y = Y.transpose((0, 2, 1)).reshape((n_trials * n_samples, n_features_y))

        X, Y = self._transform_X2D(X, Y)

        if X is not None:
            X = X.reshape((n_trials, n_samples, self.n_components)).transpose(0, 2, 1)
        if Y is not None:
            Y = Y.reshape((n_trials, n_samples, self.n_components)).transpose(0, 2, 1)

        return X, Y

    def transform(self, X=None, Y=None):
        """Transform the data matrix from feature space to component space. Note, works with both 2D and 3D data, and
        can operate on both X and Y if both are not None, or on each separately if the other is None.

        Parameters
        ----------
        X: np.ndarray (default: None)
            Data matrix of shape (n_samples, n_features_x) or (n_trials, n_features_x, n_samples). If None, only
            performs projection of Y.
        Y: np.ndarray (default: None)
            Data matrix of shape (n_samples, n_features_y) or (n_trials, n_features_y, n_samples). If None, only
            performs projection of X.

        Returns
        -------
        X: np.ndarray
            Projected data matrix of shape (n_samples, n_components) or (n_trials, n_components, n_samples). None if the
            input was None.
        Y: np.ndarray
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
    n_components: int
        The number of TRCA components to use.

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

    def __init__(self, n_components):
        self.n_components = n_components

    def _fit_X(self, X):
        """Fit TRCA without labels by computing one filter across all trials.

        Parameters
        ----------
        X: np.ndarray
            Data matrix of shape (n_trials, n_features, n_samples).

        Returns
        -------
        w: np.ndarray:
            The learned weights of shape (n_features, n_components).
        """
        X = check_array(X, ensure_2d=False, allow_nd=True)
        X = X.astype("float")
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
        return V[:, np.argsort(D)[::-1][:self.n_components]]

    def _fit_X_y(self, X, y):
        """Fit TRCA with labels by computing a filter across all trials of the same label, for each class.

        Parameters
        ----------
        X: np.ndarray
            Data matrix of shape (n_trials, n_features, n_samples).
        y: np.ndarray
            Label vector of shape (n_trials,).

        Returns
        -------
        w: np.ndarray:
            The learned weights of shape (n_features, n_components, n_classes).
        """
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True, y_numeric=True)
        X = X.astype("float")
        y = y.astype(np.uint)

        n_trials, n_channels, n_samples = X.shape
        classes = np.unique(y)
        n_classes = classes.size
        W = np.zeros((n_channels, self.n_components, n_classes))
        for i_class in range(n_classes):
            W[:, :, i_class] = self._fit_X(X[y == classes[i_class], :, :])
        return W

    def fit(self, X, y=None):
        """Fit TRCA in one of 2 ways: (1) without labels (y=None) or with labels (y=vector). If no labels are provided,
        TRCA will compute one filter across all labels. If labels are provided, one filter will be computed for each of
        the classes.

        Parameters
        ----------
        X: np.ndarray
            Data matrix of shape (n_trials, n_features, n_samples) or (n_samples, n_features).
        y: np.ndarray
            Label vector of shape (n_trials,).

        Returns
        -------
        self: TRCA
            An instance of the transformer.
        """
        if y is None:
            self.w_ = self._fit_X(X)
        else:
            self.w_ = self._fit_X_y(X, y)

        return self

    def transform(self, X, y=None):
        """Transform the data matrix from feature space to component space. Note, can operate on both X and y or just X.
        If X and y are provided, data are filtered with class-specific filters. If only X is provided and a multi-class
        filter was learned, all trials are filtered with all filters. If only one filter was learned, then only this
        filter is applied.

        Parameters
        ----------
        X: np.ndarray
            Data matrix of shape (n_trials, n_features, n_samples).
        y: np.ndarray (default: None)
            Label vector of shape (n_trials,). Can be None.

        Returns
        -------
        X: np.ndarray
            Projected data matrix of shape (n_trials, n_components, n_samples) if X and y were provided, of shape
            (n_trials, n_components, n_samples, n_classes) if y=None and multi-class filters were learned, and
            (n_trials, n_components, n_samples) if y=None and one pooled filter was learned.
        """
        check_is_fitted(self, ["w_"])
        n_trials, n_channels, n_samples = X.shape
        if y is None:
            X = check_array(X, ensure_2d=False, allow_nd=True)
            X = X.astype("float")
            if self.w_.ndim == 2:
                Y = np.dot(
                    X.transpose((0, 2, 1)).reshape((n_trials * n_samples, n_channels)), self.w_
                ).reshape((n_trials, n_samples, self.n_components)).transpose((0, 2, 1))
            else:
                n_classes = self.w_.shape[2]
                Y = np.zeros((n_trials, self.n_components, n_samples, n_classes))
                for i_class in range(n_classes):
                    Y[:, :, :, i_class] = np.dot(
                        X.transpose((0, 2, 1)).reshape((n_trials * n_samples, n_channels)), self.w_[:, :, i_class]
                    ).reshape((n_trials, n_samples, self.n_components)).transpose((0, 2, 1))
        else:
            X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True, y_numeric=True)
            X = X.astype("float")
            y = y.astype(np.uint)
            Y = np.zeros((n_trials, self.n_components, n_samples))
            for i_trial in range(n_trials):
                Y[i_trial, :, :] = np.dot(self.w_[:, :, y[i_trial]].T, X[i_trial, :, :])
        return Y
