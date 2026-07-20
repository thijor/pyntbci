from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd, inv, eigh, LinAlgError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pyntbci.utilities import covariance, pinv


def _ill_conditioned_message(name: str, shape: tuple) -> str:
    """Build an error message for a singular or too ill-conditioned covariance matrix.

    Parameters
    ----------
    name: str
        The name of the side ("X" or "Y") the covariance matrix was derived from.
    shape: tuple
        The shape of the covariance matrix.

    Returns
    -------
    message: str
        The error message.
    """
    other = "y" if name.lower() == "x" else "x"
    return (
        f"The {name} covariance matrix of shape {shape} is singular or too ill-conditioned to invert. This usually "
        f"means there is too little data relative to the number of features, e.g. when fitting a separate CCA per "
        f"class on a small per-class subset of the data (ensemble=True in eCCA/rCCA). Consider setting "
        f"gamma_{name.lower()}/alpha_{name.lower()} (or gamma_{other}/alpha_{other} if the other side is the "
        f"problem) to regularize the covariance matrix, or providing more training data."
    )


def _sym_sqrt(M: NDArray, name: str = "X") -> NDArray:
    """Compute the square root of a symmetric positive semi-definite matrix via its eigendecomposition. Equivalent
    to scipy.linalg.sqrtm(M) for such matrices, but exploits symmetry to be faster, more numerically stable, and
    guaranteed real (avoiding the small imaginary numerical noise that the general-purpose scipy.linalg.sqrtm can
    introduce). Also checks, using the already-computed eigenvalues, that M is not so ill-conditioned that inverting
    it has produced a numerically corrupted (i.e., no longer positive semi-definite) result.

    Parameters
    ----------
    M: NDArray
        A symmetric positive semi-definite matrix of shape (n, n).
    name: str (default: "X")
        The name of the side ("X" or "Y") that M was derived from, used only to phrase the error message if M turns
        out not to be (numerically) positive semi-definite.

    Returns
    -------
    sqrtM: NDArray
        The square root of M of shape (n, n), such that sqrtM @ sqrtM == M.
    """
    w, V = eigh(M)
    if w.min() < -1e-8 * w.max():
        raise LinAlgError(_ill_conditioned_message(name, M.shape))
    w = np.clip(w, 0, None)
    return (V * np.sqrt(w)) @ V.T


def _safe_inv(C: NDArray, name: str) -> NDArray:
    """Invert a covariance matrix, raising a clear, actionable error instead of a bare "singular matrix" if it
    cannot be inverted.

    Parameters
    ----------
    C: NDArray
        A covariance matrix of shape (n, n) to invert.
    name: str
        The name of the side ("X" or "Y") that C was derived from, used only to phrase the error message if C turns
        out to be singular.

    Returns
    -------
    iC: NDArray
        The inverse of C of shape (n, n).
    """
    try:
        return inv(C)
    except LinAlgError as e:
        raise LinAlgError(_ill_conditioned_message(name, C.shape)) from e


class CCA(TransformerMixin, BaseEstimator):
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
    rho_: NDArray
        The singular values (canonical correlations) of shape (min(n_features_x, n_features_y),).
    n_x_: int
        The number of samples used to estimate avg_x_ and cov_x_.
    avg_x_: NDArray
        The (running) average of X of shape (n_features_x).
    cov_x_: NDArray
        The (running) covariance of X of shape (n_features_x, n_features_x).
    n_y_: int
        The number of samples used to estimate avg_y_ and cov_y_.
    avg_y_: NDArray
        The (running) average of Y of shape (n_features_y).
    cov_y_: NDArray
        The (running) covariance of Y of shape (n_features_y, n_features_y).
    n_xy_: int
        The number of samples used to estimate avg_xy_ and cov_xy_.
    avg_xy_: NDArray
        The (running) average of concat(X, Y) of shape (n_features_x + n_features_y).
    cov_xy_: NDArray
        The (running) covariance of concat(X, Y) of shape (n_features_x + n_features_y, n_features_x + n_features_y),
        from which the cross-covariance of X and Y is taken as the (n_features_x, n_features_y) off-diagonal block.

    References
    ----------
    .. [1] Hotelling, H. (1992). Relations between two sets of variates. In Breakthroughs in statistics: methodology and
           distribution (pp. 162-190). New York, NY: Springer New York. doi: 10.1007/978-1-4612-4380-9_14
    """

    w_x_: NDArray
    w_y_: NDArray
    rho_: NDArray
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
        running: bool = False,
        alpha_x: float = None,
        alpha_y: float = None,
    ) -> None:
        self.n_components = n_components
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
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

        # Compute covariances. Cxx/Cyy are taken as blocks of the joint covariance of concat(X, Y) rather than
        # computed separately, since they are already sub-computations of it.
        Z = np.concatenate((X, Y), axis=1)
        self.n_xy_, self.avg_xy_, self.cov_xy_ = covariance(
            Z, self.n_xy_, self.avg_xy_, self.cov_xy_, running=self.running
        )
        self.n_x_ = self.n_xy_
        self.avg_x_ = self.avg_xy_[:, : X.shape[1]]
        self.cov_x_ = self.cov_xy_[: X.shape[1], : X.shape[1]]
        self.n_y_ = self.n_xy_
        self.avg_y_ = self.avg_xy_[:, X.shape[1] :]
        self.cov_y_ = self.cov_xy_[X.shape[1] :, X.shape[1] :]
        Cxx = self.cov_x_
        Cyy = self.cov_y_
        Cxy = self.cov_xy_[: X.shape[1], X.shape[1] :]

        # Regularization. Shrink towards nu * I, blending row-wise and column-wise scaling by (1 - gamma) so the
        # result stays symmetric for a per-feature gamma (for scalar gamma this is identical to (1 - gamma) * Cxx,
        # since both terms are then equal).
        if self.gamma_x is not None:
            nu_x = np.trace(Cxx) / Cxx.shape[1]
            if isinstance(self.gamma_x, (int, float)):
                gamma_x = np.full(Cxx.shape[1], self.gamma_x)
            else:
                gamma_x = np.asarray(self.gamma_x).flatten()
            Cxx = 0.5 * ((1 - gamma_x)[:, np.newaxis] * Cxx + Cxx * (1 - gamma_x)[np.newaxis, :]) + nu_x * np.diag(
                gamma_x
            )
        if self.gamma_y is not None:
            nu_y = np.trace(Cyy) / Cyy.shape[1]
            if isinstance(self.gamma_y, (int, float)):
                gamma_y = np.full(Cyy.shape[1], self.gamma_y)
            else:
                gamma_y = np.asarray(self.gamma_y).flatten()
            Cyy = 0.5 * ((1 - gamma_y)[:, np.newaxis] * Cyy + Cyy * (1 - gamma_y)[np.newaxis, :]) + nu_y * np.diag(
                gamma_y
            )

        # Inverse square root
        if self.alpha_x is None:
            iCxx = _sym_sqrt(_safe_inv(Cxx, "X"), name="X")
        else:
            iCxx = _sym_sqrt(pinv(Cxx, self.alpha_x), name="X")
        if self.alpha_y is None:
            iCyy = _sym_sqrt(_safe_inv(Cyy, "Y"), name="Y")
        else:
            iCyy = _sym_sqrt(pinv(Cyy, self.alpha_y), name="Y")

        # SVD
        U, self.rho_, V = svd(iCxx @ Cxy @ iCyy)

        # Compute projection vectors
        Wx = iCxx @ U
        Wy = iCyy @ V.T

        # Select components
        self.w_x_ = Wx[:, : self.n_components]
        self.w_y_ = Wy[:, : self.n_components]

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

        self._is_fitted = True
        return self

    def _transform_X2D(
        self,
        X: NDArray = None,
        Y: NDArray = None,
    ) -> tuple[NDArray, NDArray]:
        """Transform the 2D data matrix from feature space to component space.

        Parameters
        ----------
        X: NDArray (default: None)
            Data matrix of shape (n_samples, n_features_x).
        Y: NDArray (default: None)
            Data matrix of shape (n_samples, n_features_y).

        Returns
        -------
        X: NDArray
            Projected data matrix of shape (n_samples, n_components).
        Y: NDArray
            Projected data matrix of shape (n_samples, n_components).
        """
        if X is not None:
            X = np.dot(X - self.avg_x_, self.w_x_)
        if Y is not None:
            Y = np.dot(Y - self.avg_y_, self.w_y_)

        return X, Y

    def _transform_X3D(
        self,
        X: NDArray = None,
        Y: NDArray = None,
    ) -> tuple[NDArray, NDArray]:
        """Transform the 3D data matrix from feature space to component space.

        Parameters
        ----------
        X: NDArray (default: None)
            Data matrix of shape (n_trials, n_features_x, n_samples).
        Y: NDArray (default: None)
            Data matrix of shape (n_trials, n_features_y, n_samples).

        Returns
        -------
        X: NDArray
            Projected data matrix of shape (n_trials, n_components, n_samples).
        Y: NDArray
            Projected data matrix of shape (n_trials, n_components, n_samples).
        """
        # Batched matmul: (n_components, n_features) @ (n_trials, n_features, n_samples) broadcasts the weight
        # matrix across trials, avoiding the transpose+reshape (and the copy it forces) needed to route through
        # _transform_X2D.
        if X is not None:
            X = self.w_x_.T @ (X - self.avg_x_[:, :, np.newaxis])
        if Y is not None:
            Y = self.w_y_.T @ (Y - self.avg_y_[:, :, np.newaxis])

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
        check_is_fitted(self)

        if (X is None or X.ndim == 3) and (Y is None or Y.ndim == 3):
            X, Y = self._transform_X3D(X, Y)
        elif (X is None or X.ndim == 2) and (Y is None or Y.ndim == 2):
            X, Y = self._transform_X2D(X, Y)
        else:
            raise Exception("X and Y must be both 3D or 2D, or None.")

        return X, Y

    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status and return a Boolean value.

        Returns
        -------
        fitted: bool
            Whether the transformer is fitted.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


class Vectorizer(TransformerMixin, BaseEstimator):
    """Vectorizer. Flattens a multi-dimensional data matrix per trial into a single feature vector, e.g. to use
    multi-channel time-series data with generic (non multi-dimensional) scikit-learn estimators.

    Parameters
    ----------
    channel_prime: bool (default: False)
        Whether the channels are the fastest-varying (i.e., contiguous) dimension in the flattened feature vector.
        If False, the samples are the fastest-varying dimension instead.
    """

    def __init__(
        self,
        channel_prime: bool = False,
    ) -> None:
        self.channel_prime = channel_prime

    def fit(
        self,
        X: NDArray,
        y: NDArray = None,
    ) -> TransformerMixin:
        """Fit the vectorizer. Note, does not involve learning.

        Parameters
        ----------
        X: NDArray
            Data matrix of shape (n_trials, n_channels, n_samples).
        y: NDArray (default: None)
            Not used.

        Returns
        -------
        self: TransformerMixin
            Returns the instance itself.
        """
        self._is_fitted = True
        return self

    def transform(
        self,
        X: NDArray,
        y: NDArray = None,
    ) -> NDArray:
        """Flatten the data matrix per trial into a single feature vector.

        Parameters
        ----------
        X: NDArray
            Data matrix of shape (n_trials, n_channels, n_samples).
        y: NDArray (default: None)
            Not used.

        Returns
        -------
        X: NDArray
            Flattened data matrix of shape (n_trials, n_channels * n_samples).
        """
        check_is_fitted(self)
        if self.channel_prime:
            X = X.transpose((0, 2, 1))
        return X.reshape((X.shape[0], -1))

    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status and return a Boolean value.

        Returns
        -------
        fitted: bool
            Whether the transformer is fitted.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
