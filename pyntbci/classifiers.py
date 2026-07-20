from copy import deepcopy
from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from pyntbci.transformers import CCA
from pyntbci.utilities import (
    correct_latency,
    correlation,
    decoding_matrix,
    encoding_matrix,
    euclidean,
    event_matrix,
    inner,
)


def _running_score(
    X_chunk: NDArray,
    T_raw_chunk: NDArray,
    T_raw_mean: NDArray,
    score_metric: str,
    state: dict = None,
) -> tuple[NDArray, dict]:
    """Compute similarity scores between a newly observed chunk of (spatially filtered) signal and the
    corresponding chunk of a template, incrementally, without recomputing from scratch over samples already seen.
    Used by decision_function()'s running=True mode in eCCA and rCCA.

    Correlation is shift-invariant, so it is computed directly via correlation()'s own running mode on the raw
    (not de-meaned) template chunk. Euclidean distance and the inner product are not shift-invariant, and get_T()
    de-means the template by the mean over the *current full window*, which itself changes every call as more
    samples arrive, so the de-meaned values cannot simply be appended to as n_samples grows. Instead, the running
    sums here are accumulated on the raw (not de-meaned) template, and get_T()'s de-meaning is applied afterwards
    as a cheap correction at query time, using the freshly (and cheaply, since it needs no spatial filtering or
    matrix products) computed T_raw_mean:
        euclidean:  d(X, T)^2 = sum((X - T_raw)^2) + 2 * mu * sum(X - T_raw) + n * mu^2
        inner:      inner(X, T) = sum(X * T_raw) - mu * sum(X)
    where mu = T_raw_mean and T = T_raw - mu (i.e., what get_T() would return). Both identities were verified
    numerically against get_T()-based batch computation (max abs error ~1e-13, i.e., floating-point noise).

    Parameters
    ----------
    X_chunk: NDArray
        The new chunk of (spatially filtered) signal of shape (n_trials, n_new_samples).
    T_raw_chunk: NDArray
        The corresponding new chunk of the template, not yet de-meaned (see _get_T_raw()), of shape
        (n_classes, n_new_samples).
    T_raw_mean: NDArray
        The mean of the not-yet-de-meaned template over the full window observed so far (i.e., what get_T()'s
        de-meaning would subtract), of shape (1, n_classes). Only used for score_metric in {"euclidean", "inner"}.
    score_metric: str
        The score metric: correlation, euclidean, inner.
    state: dict (default: None)
        The running state returned by a previous call, or None for the first chunk of a new sequence.

    Returns
    -------
    scores: NDArray
        The similarity scores of shape (n_trials, n_classes), cumulative over all chunks observed so far (not just
        the new chunk).
    state: dict
        The updated running state, to pass as state on the next call.
    """
    state = {} if state is None else state

    if score_metric.lower() == "correlation":
        if X_chunk.shape[1] == 0:
            # A zero-sample chunk carries no information; short-circuit rather than feed an empty array into
            # covariance()'s running update, which (unlike euclidean()/inner(), for which summing zero samples is
            # a well-defined, warning-free no-op) would corrupt the running mean/covariance with NaN (its mean of
            # the empty new observation is NaN, and NaN * 0 is still NaN, not the "no change" one might expect).
            n_a = X_chunk.shape[0]
            if state.get("cov") is None:
                return np.full((n_a, T_raw_chunk.shape[0]), np.nan), state
            cov = state["cov"]
            var_a = np.diag(cov)[:n_a, np.newaxis]
            var_b = np.diag(cov)[np.newaxis, n_a:]
            scores = cov[:n_a, n_a:] / np.sqrt(var_a * var_b)
            return scores, state
        scores, n, avg, cov = correlation(
            X_chunk, T_raw_chunk, state.get("n", 0), state.get("avg"), state.get("cov"), running=True
        )
        return scores, {"n": n, "avg": avg, "cov": cov}

    sum_x_obs = X_chunk.sum(axis=1, keepdims=True)
    sum_x = sum_x_obs if state.get("sum_x") is None else state["sum_x"] + sum_x_obs
    sum_t_obs = T_raw_chunk.sum(axis=1, keepdims=True).T
    sum_t = sum_t_obs if state.get("sum_t") is None else state["sum_t"] + sum_t_obs
    n_obs = state.get("n", 0) + X_chunk.shape[1]

    if score_metric.lower() == "euclidean":
        _, sum_xx, sum_tt, sum_xt = euclidean(
            X_chunk, T_raw_chunk, state.get("sum_xx"), state.get("sum_tt"), state.get("sum_xt"), running=True
        )
        d2 = (sum_xx - 2 * sum_xt + sum_tt) + 2 * T_raw_mean * (sum_x - sum_t) + n_obs * T_raw_mean**2
        scores = np.sqrt(np.clip(d2, 0, None))
        return scores, {
            "n": n_obs,
            "sum_x": sum_x,
            "sum_t": sum_t,
            "sum_xx": sum_xx,
            "sum_tt": sum_tt,
            "sum_xt": sum_xt,
        }

    elif score_metric.lower() == "inner":
        sum_xt = inner(X_chunk, T_raw_chunk, state.get("sum_xt"), running=True)
        scores = sum_xt - T_raw_mean * sum_x
        return scores, {"n": n_obs, "sum_x": sum_x, "sum_t": sum_t, "sum_xt": sum_xt}

    else:
        raise Exception(f"Unknown score metric: {score_metric}")


class eCCA(ClassifierMixin, BaseEstimator):
    """ERP CCA classifier. Also called the "reference" method [1]_. It computes ERPs as templates for full sequences and
    performs a CCA for spatial filtering.

    Parameters
    ----------
    lags: None | NDArray
        A vector of latencies in seconds per class relative to the first stimulus if stimuli are circularly shifted
        versions of the first stimulus, or None if all stimuli are different or this circular shift feature should be
        ignored.
    fs: int
        The sampling frequency of the EEG data in Hz.
    cycle_size: float (default: None)
        The time that one cycle of the code takes in seconds. If None, takes the full data length.
    template_metric: str (default: "mean")
        Metric to use to compute templates: mean, median, ocsvm.
    score_metric: str (default: "correlation")
        Metric to use to compute the overlap of templates and single-trials during testing: correlation, Euclidean,
        inner.
    cca_channels: list[int] (default: None)
        A list of channel indexes that need to be included in the estimation of a spatial filter at the template side
        of the CCA, i.e. CCA(X, T[:, cca_channels, :]). If None is given, all channels are used.
    gamma_x: float | list[float] | NDArray (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along X (channels). If
        None, no regularization is applied. The gamma_x ranges from 0 (no regularization) to 1 (full regularization).
    gamma_t: float | list[float] | NDArray (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along T (channels). If
        None, no regularization is applied. The gamma_t ranges from 0 (no regularization) to 1 (full regularization).
    latency: NDArray (default: None)
        The raster latencies of each of the classes of shape (n_classes,) that the data/templates need to be corrected
        for.
    ensemble: bool (default: False)
        Whether to use an ensemble classifier, that is, a separate spatial filter for each class. Note, each filter
        is then fit on only that class's trials, so its covariance matrices are estimated from substantially less
        data than in the non-ensemble case; this can make them singular or too ill-conditioned to invert, especially
        with few trials per class or many channels/features. If this occurs, set gamma_x/gamma_t or alpha_x/alpha_t
        to regularize the covariance matrix.
    n_components: int (default: 1)
        The number of CCA components to use.
    squeeze_components: bool (default: True)
        Remove the component dimension when n_components=1.
    alpha_x: float (default: None)
        Amount of variance to retain in computing the inverse of the covariance matrix of X. If None, all variance.
    alpha_t: float (default: None)
        Amount of variance to retain in computing the inverse of the covariance matrix of T. If None, all variance.
    running: bool (default: False)
        Whether fit() is incremental: if False, each fit() call replaces the previous fit, using only the trials
        passed to that call. If True, each fit() call instead adds its trials to the ones seen in all previous
        fit() calls (i.e., keeps the spatial filter's running covariance from CCA(running=True), and the
        template's running mean, instead of discarding them), so a model can be trained gradually as more trials
        become available. Requires lags to be set (a fixed, known-upfront class count and a single, un-split
        running template -- see lags above), template_metric="mean" (the only template metric with an exact
        incremental update), and ensemble=False. Unlike rCCA(running=True), this is only an approximation of the
        equivalent batch fit, not exact: the (running) template is itself used as the CCA fit's target on every
        call, so earlier calls see an earlier, less complete estimate of it than later calls do; it converges
        towards the batch result as more trials accumulate, but is not expected to equal it. To start a new
        running fit from scratch, use a new instance (or call set_params(running=False) once, fit(), then
        set_params(running=True) again).

    Attributes
    ----------
    classes_: NDArray
        The classes that can be predicted, of shape (n_classes). Equal to numpy.arange(len(lags)) if lags is set
        (i.e., all circularly shifted classes can be predicted, whether or not they were observed in y during fit),
        otherwise the sorted unique labels observed in y.
    cca_: list[TransformerMixin]
        The CCA used to fit the spatial filters. If ensemble=False, len(cca_)=1, otherwise len(cca_)=n_classes.
    w_: NDArray
        The weight vector representing a spatial filter of shape (n_channels, n_components). If ensemble=True, then the
        shape is (n_channels, n_components, n_classes).
    T_: NDArray
        The template matrix representing the expected responses of shape (n_classes, n_components, n_samples).

    References
    ----------
    .. [1] Martínez-Cagigal, V., Thielen, J., Santamaria-Vazquez, E., Pérez-Velasco, S., Desain, P., & Hornero, R.
           (2021). Brain–computer interfaces based on code-modulated visual evoked potentials (c-VEP): A literature
           review. Journal of Neural Engineering, 18(6), 061002. doi: 10.1088/1741-2552/ac38cf
    """

    classes_: NDArray
    cca_: list[TransformerMixin]
    w_: NDArray
    T_: NDArray
    _running_: dict = None
    _template_n_: int = 0
    _template_avg_: NDArray = None

    def __init__(
        self,
        lags: Union[None, NDArray],
        fs: int,
        cycle_size: float = None,
        template_metric: str = "mean",
        score_metric: str = "correlation",
        cca_channels: list[int] = None,
        gamma_x: Union[float, list[float], NDArray] = None,
        gamma_t: Union[float, list[float], NDArray] = None,
        latency: NDArray = None,
        ensemble: bool = False,
        n_components: int = 1,
        squeeze_components: bool = True,
        alpha_x: float = None,
        alpha_t: float = None,
        running: bool = False,
    ) -> None:
        self.lags = lags
        self.fs = fs
        self.cycle_size = cycle_size
        self.template_metric = template_metric
        self.score_metric = score_metric
        self.cca_channels = cca_channels
        self.gamma_x = gamma_x
        self.gamma_t = gamma_t
        self.latency = latency
        self.ensemble = ensemble
        self.n_components = n_components
        self.squeeze_components = squeeze_components
        self.alpha_x = alpha_x
        self.alpha_t = alpha_t
        self.running = running

    def _fit_T(
        self,
        X: NDArray,
    ) -> NDArray:
        """Fit the templates.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        T: NDArray
            The matrix of one EEG template of shape (n_channels, n_samples).
        """
        n_trials, n_channels, n_samples = X.shape
        if self.template_metric.lower() == "mean":
            T = X.mean(axis=0)
        elif self.template_metric.lower() == "median":
            T = np.median(X, axis=0)
        elif self.template_metric.lower() == "ocsvm":
            ocsvm = OneClassSVM(kernel="linear", nu=0.5)
            T = np.zeros((n_channels, n_samples))
            for i_channel in range(n_channels):
                ocsvm.fit(X[:, i_channel, :])
                T[i_channel, :] = ocsvm.coef_
        else:
            raise Exception(f"Unknown template metric: {self.template_metric}")
        return T

    def decision_function(
        self,
        X: NDArray,
        running: bool = False,
        reset: bool = False,
    ) -> NDArray:
        """Apply the classifier to get classification scores for X.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples). If running=True, this is only the
            newly observed samples since the previous call (not the full trial), see running below.
        running: bool (default: False)
            Whether to use running (incremental) scoring. If False (default), decision_function behaves exactly as
            without this parameter: X is the complete trial data seen so far, and everything is recomputed from
            scratch. If True, X is only the newly observed samples since the previous call, and a running state
            (kept internally, not a fitted attribute) is reused and updated; this is much cheaper when called
            repeatedly on a growing trial, e.g. from a dynamic stopping simulation loop, since each call only does
            O(n_new_samples) work instead of reprocessing the whole trial. Use reset=True on the first call of a
            new running sequence (e.g. for a new trial or a new batch of trials); the running state is otherwise
            unaffected by (and does not affect) running=False calls, and is cleared by fit(). Only supported for
            ensemble=False.
        reset: bool (default: False)
            Whether to discard any existing running state before processing this call. Only relevant if
            running=True; a never-yet-used instance already starts fresh without it, so it only needs to be set
            explicitly to start a new sequence before the previous one naturally ended.

        Returns
        -------
        scores: NDArray
            The similarity scores of shape (n_trials, n_classes, n_components) or (n_trials, n_classes) if
            n_components=1 and squeeze_components=True. If running=True, this is the cumulative score over all
            samples observed so far in the running sequence (not just the new chunk).
        """
        check_is_fitted(self)

        if not running:
            # Set templates to trial length
            T = self.get_T(X.shape[2])

            # Compute scores
            scores = np.zeros((X.shape[0], T.shape[0], self.n_components))
            if self.ensemble:
                for i_class in range(T.shape[0]):
                    Xi = self.cca_[i_class].transform(X=X)[0]
                    for i_component in range(self.n_components):
                        if self.score_metric.lower() == "correlation":
                            scores[:, i_class, i_component] = correlation(
                                Xi[:, i_component, :], T[i_class, i_component, :]
                            )[:, 0]
                        elif self.score_metric.lower() == "euclidean":
                            scores[:, i_class, i_component] = (
                                1 / (1 + euclidean(Xi[:, i_component, :], T[i_class, i_component, :]))[:, 0]
                            )
                        elif self.score_metric.lower() == "inner":
                            scores[:, i_class, i_component] = np.inner(
                                Xi[:, i_component, :], T[i_class, i_component, :]
                            )
                        else:
                            raise Exception(f"Unknown score metric: {self.score_metric}")

            else:
                X = self.cca_[0].transform(X=X)[0]
                for i_component in range(self.n_components):
                    if self.score_metric.lower() == "correlation":
                        scores[:, :, i_component] = correlation(X[:, i_component, :], T[:, i_component, :])
                    elif self.score_metric.lower() == "euclidean":  # includes conversion to similarity
                        scores[:, :, i_component] = 1 / (1 + euclidean(X[:, i_component, :], T[:, i_component, :]))
                    elif self.score_metric.lower() == "inner":
                        scores[:, :, i_component] = np.inner(X[:, i_component, :], T[:, i_component, :])
                    else:
                        raise Exception(f"Unknown score metric: {self.score_metric}")

            if self.n_components == 1 and self.squeeze_components:
                scores = scores[:, :, 0]

            return scores

        assert not self.ensemble, "running=True decision_function is not supported for ensemble=True."

        if reset or self._running_ is None:
            self._running_ = {"n_trials": X.shape[0], "n_samples": 0, "component_state": [None] * self.n_components}
        assert X.shape[0] == self._running_["n_trials"], (
            f"running=True decision_function was called with {X.shape[0]} trials, but the running sequence was "
            f"started (or last continued) with {self._running_['n_trials']}; call with reset=True to start a new "
            f"sequence."
        )

        Xf = self.cca_[0].transform(X=X)[0]
        n_prev = self._running_["n_samples"]
        n_new = self._running_["n_samples"] + X.shape[2]
        assert n_new > 0, "running=True decision_function requires at least 1 sample on the first call of a sequence."
        scores = np.zeros((X.shape[0], len(self.classes_), self.n_components))
        for i_component in range(self.n_components):
            T_raw_full = self._get_T_raw(n_new)[:, i_component, :]
            T_raw_chunk = T_raw_full[:, n_prev:n_new]
            T_raw_mean = T_raw_full.mean(axis=1, keepdims=True).T

            component_scores, self._running_["component_state"][i_component] = _running_score(
                Xf[:, i_component, :],
                T_raw_chunk,
                T_raw_mean,
                self.score_metric,
                self._running_["component_state"][i_component],
            )
            if self.score_metric.lower() == "euclidean":  # includes conversion to similarity
                component_scores = 1 / (1 + component_scores)
            scores[:, :, i_component] = component_scores
        self._running_["n_samples"] = n_new

        if self.n_components == 1 and self.squeeze_components:
            scores = scores[:, :, 0]

        return scores

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> ClassifierMixin:
        """The training procedure to fit eCCA on supervised EEG data.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: NDArray
            The vector of ground-truth labels of the trials in X of shape (n_trials), i.e., the index of the
            attended code.

        Returns
        -------
        self: ClassifierMixin
            Returns the instance itself.
        """
        n_trials, n_channels, n_samples = X.shape

        if self.running:
            assert self.lags is not None, "running=True requires lags to be set."
            assert self.template_metric.lower() == "mean", "running=True only supports template_metric='mean'."
            assert not self.ensemble, "running=True is not supported for ensemble=True."

        # Whether this call continues a running fit already in progress (see cca_[0].running below), as opposed to
        # (re)starting one (or, if running=False, always).
        continuing = self.running and getattr(self, "cca_", None) and self.cca_[0].running

        # Correct for raster latency
        if self.latency is not None:
            X = correct_latency(X, y, -self.latency, self.fs, axis=-1)

        # Cut trials to cycles
        if self.cycle_size is not None:
            cycle_size = int(self.cycle_size * self.fs)
            n_cycles = int(n_samples / cycle_size)
            if n_samples % cycle_size > 0:
                X = X[:, :, : int(n_cycles * cycle_size)]
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
            Z = correct_latency(X, y, -self.lags, self.fs, axis=-1)
            if self.running:
                # A single running mean over all (latency-corrected) trials observed so far, regardless of class,
                # since the circular-shift model assumes all classes are shifted versions of the same underlying
                # response. Note this template (base_T, used as R below) is itself a moving target across calls,
                # unlike rCCA's stimulus-derived (and thus fixed) R -- so, unlike rCCA, this is an approximation
                # of the batch fit, not exact; see the running docstring entry.
                if continuing:
                    assert Z.shape[1:] == self._template_avg_.shape, (
                        f"running=True requires every fit() call to have the same number of channels and samples "
                        f"per trial (after latency correction/cycle-cutting); got {Z.shape[1:]}, expected "
                        f"{self._template_avg_.shape}."
                    )
                    n_obs = Z.shape[0]
                    avg_obs = Z.mean(axis=0)
                    n_new = self._template_n_ + n_obs
                    self._template_avg_ = self._template_avg_ + (avg_obs - self._template_avg_) * (n_obs / n_new)
                    self._template_n_ = n_new
                else:
                    self._template_n_ = Z.shape[0]
                    self._template_avg_ = Z.mean(axis=0)
                base_T = self._template_avg_
            else:
                base_T = self._fit_T(Z)
            T = np.tile(base_T[np.newaxis, :, :], (n_classes, 1, 1))
            T = correct_latency(T, np.arange(n_classes), self.lags, self.fs, axis=-1)
            if self.latency is not None:
                T = correct_latency(T, np.arange(n_classes), self.latency, self.fs, axis=-1)

        # Fit CCA
        if self.ensemble:
            self.w_ = np.zeros((n_channels, self.n_components, n_classes))
            self.cca_ = []
            for i_class in range(n_classes):
                S = np.reshape(X[y == i_class, :, :].transpose((0, 2, 1)), (-1, n_channels))
                R = np.tile(T[i_class, :, :].T, ((y == i_class).sum(), 1))
                if self.cca_channels is not None:
                    R = R[:, self.cca_channels]
                self.cca_.append(
                    CCA(
                        n_components=self.n_components,
                        gamma_x=self.gamma_x,
                        gamma_y=self.gamma_t,
                        alpha_x=self.alpha_x,
                        alpha_y=self.alpha_t,
                    )
                )
                self.cca_[i_class].fit(S, R)
                self.w_[:, :, i_class] = self.cca_[i_class].w_x_
        else:
            S = np.reshape(X.transpose((0, 2, 1)), (-1, n_channels))
            R = np.reshape(T[y, :, :].transpose((0, 2, 1)), (-1, n_channels))
            if self.cca_channels is not None:
                R = R[:, self.cca_channels]
            if continuing:
                self.cca_[0].set_params(
                    n_components=self.n_components,
                    gamma_x=self.gamma_x,
                    gamma_y=self.gamma_t,
                    alpha_x=self.alpha_x,
                    alpha_y=self.alpha_t,
                )
            else:
                self.cca_ = [
                    CCA(
                        n_components=self.n_components,
                        gamma_x=self.gamma_x,
                        gamma_y=self.gamma_t,
                        alpha_x=self.alpha_x,
                        alpha_y=self.alpha_t,
                        running=self.running,
                    )
                ]
            self.cca_[0].fit(S, R)
            self.w_ = self.cca_[0].w_x_

        # Spatially filter templates
        if self.ensemble:
            self.T_ = np.zeros((n_classes, self.n_components, n_samples))
            for i_class in range(n_classes):
                self.T_[i_class, :, :] = self.cca_[i_class].transform(T[[i_class], :, :])[0]
        else:
            self.T_ = self.cca_[0].transform(T)[0]

        self.classes_ = np.arange(n_classes)
        self._running_ = None
        self._is_fitted = True
        return self

    def _get_T_raw(
        self,
        n_samples: int = None,
    ) -> NDArray:
        """Get the templates, tiled to the requested length, without the de-meaning applied by get_T(). Used by
        get_T() itself, and by the running (running=True) path of decision_function(), which needs to apply the
        equivalent of get_T()'s de-meaning as a cheap correction at query time (see decision_function()), since the
        de-meaned values are not simply appendable as n_samples grows (the mean itself changes).

        Parameters
        ----------
        n_samples: int (default: None)
            The number of samples requested. If None, one code cycle is given.

        Returns
        -------
        T: NDArray
            The (not de-meaned) templates of shape (n_classes, n_components, n_samples).
        """
        if n_samples is None or self.T_.shape[2] == n_samples:
            return self.T_.copy()
        n = int(np.ceil(n_samples / self.T_.shape[2]))
        return np.tile(self.T_, (1, 1, n))[:, :, :n_samples]

    def get_T(
        self,
        n_samples: int = None,
    ) -> NDArray:
        """Get the templates.

        Parameters
        ----------
        n_samples: int (default: None)
            The number of samples requested. If None, one code cycle is given.

        Returns
        -------
        T: NDArray
            The templates of shape (n_classes, n_components, n_samples).
        """
        T = self._get_T_raw(n_samples)
        T -= T.mean(axis=2, keepdims=True)
        return T

    def predict(
        self,
        X: NDArray,
        running: bool = False,
        reset: bool = False,
    ) -> NDArray:
        """The testing procedure to apply eCCA to novel EEG data.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples). If running=True, this is only the
            newly observed samples since the previous call, see decision_function().
        running: bool (default: False)
            Whether to use running (incremental) scoring, see decision_function().
        reset: bool (default: False)
            Whether to discard any existing running state before processing this call, see decision_function().

        Returns
        -------
        y: NDArray
            The predicted labels of shape (n_trials, n_components) or (n_trials) if n_components=1 and
            squeeze_components=True.
        """
        check_is_fitted(self)
        return np.argmax(self.decision_function(X, running=running, reset=reset), axis=1)

    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status and return a Boolean value.

        Returns
        -------
        fitted: bool
            Whether the classifier is fitted.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


class Ensemble(ClassifierMixin, BaseEstimator):
    """Ensemble classifier. It wraps an ensemble classifier around another classifier object. The classifiers are
    applied to each item in a databank separately. A gating function combines the outputs of the individual
    classifications to arrive at a single final combined classification.

    Parameters
    ----------
    estimator: ClassifierMixin
        The classifier object that is applied to each item in the databank.
    gate: ClassifierMixin
        The gate that is used to combine the scores obtained from each individual estimator.

    Attributes
    ----------
    classes_: NDArray
        The classes that can be predicted, taken from the gate's classes_ after fitting.
    models_: list[ClassifierMixin]
        A list containing all models learned for each of the databanks.
    """

    classes_: NDArray
    models_: list[ClassifierMixin]

    def __init__(
        self,
        estimator: ClassifierMixin,
        gate: ClassifierMixin,
    ) -> None:
        self.estimator = estimator
        self.gate = gate

    def decision_function(
        self,
        X: NDArray,
    ) -> NDArray:
        """Apply the classifier to get classification scores for X.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples, n_items).

        Returns
        -------
        scores: NDArray
            The matrix of scores of shape (n_trials, n_classes).
        """
        check_is_fitted(self)
        scores = np.stack([self.models_[i].decision_function(X[:, :, :, i]) for i in range(X.shape[3])], axis=2)
        return self.gate.decision_function(scores)

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> ClassifierMixin:
        """The training procedure to apply an ensemble classifier on supervised EEG data.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples, n_items).
        y: NDArray
            The vector of ground-truth labels of the trials in X of shape (n_trials). Note, these denote the index at
            which to find the associated stimulus!

        Returns
        -------
        self: ClassifierMixin
            Returns the instance itself.
        """
        assert X.ndim == 4

        # Fit separate models for each databank
        self.models_ = [deepcopy(self.estimator).fit(X[:, :, :, i], y) for i in range(X.shape[3])]

        # Fit gating
        scores = np.stack([self.models_[i].decision_function(X[:, :, :, i]) for i in range(X.shape[3])], axis=2)
        self.gate.fit(scores, y)

        self.classes_ = self.gate.classes_
        self._is_fitted = True
        return self

    def predict(
        self,
        X: NDArray,
    ) -> NDArray:
        """The testing procedure to apply the ensemble classifier to novel EEG data.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples, n_items).

        Returns
        -------
        y: NDArray
            The vector of predicted labels of the trials in X of shape (n_trials). Note, these denote the index at which
            to find the associated stimulus!
        """
        check_is_fitted(self)
        scores = np.stack([self.models_[i].decision_function(X[:, :, :, i]) for i in range(X.shape[3])], axis=2)
        return self.gate.predict(scores)

    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status and return a Boolean value.

        Returns
        -------
        fitted: bool
            Whether the classifier is fitted.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


class rCCA(ClassifierMixin, BaseEstimator):
    """Reconvolution CCA classifier. It performs a spatial and temporal decomposition (reconvolution [3]_) within a
    CCA [4]_ to perform spatial filtering as well as template prediction [5]_.

    Parameters
    ----------
    stimulus: NDArray
        The stimulus used for stimulation of shape (n_classes, n_samples). Should be sampled at fs. One cycle (i.e.,
        one stimulus-repetition) is sufficient.
    fs: int
        The sampling frequency of the EEG data in Hz.
    event: str (default: "duration")
        The event definition to map stimulus to events.
    onset_event: bool (default: False)
        Whether to add an event for the onset of stimulation. Added as last event.
    decoding_length: float (default: None)
        The length of the spectral filter for each data channel in seconds. If None, it is set to 1/fs, equivalent to 1
        sample, such that no phase-shifting is performed and thus no (spatio-)spectral filter is learned.
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
        Metric to use to compute the overlap of templates and single-trials during testing: correlation, Euclidean,
        inner.
    latency: NDArray (default: None)
        The raster latencies of each of the classes of shape (n_classes,) that the data/templates need to be corrected
        for.
    ensemble: bool (default: False)
        Whether to use an ensemble classifier, that is, a separate spatial filter for each class. Note, each filter
        is then fit on only that class's trials, so its covariance matrices are estimated from substantially less
        data than in the non-ensemble case; this can make them singular or too ill-conditioned to invert, especially
        with a wide encoding matrix (multiple events and/or a long encoding_length) relative to the trial length. If
        this occurs, set gamma_x/gamma_m or alpha_x/alpha_m to regularize the covariance matrix.
    amplitudes: NDArray (default: None)
        The amplitude of the stimulus of shape (n_classes, n_samples). Should be sampled at fs.
    gamma_x: float | list[float] | NDArray (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along X (channels). If
        None, no regularization is applied. The gamma_x ranges from 0 (no regularization) to 1 (full regularization).
    gamma_m: float | list[float] | NDArray (default: None)
        Regularization on the covariance matrix for CCA for all or each individual parameter along M (samples). If None,
        no regularization is applied. The gamma_m ranges from 0 (no regularization) to 1 (full regularization).
    n_components: int (default: 1)
        The number of CCA components to use.
    squeeze_components: bool (default: True)
        Remove the component dimension when n_components=1.
    alpha_x: float (default: None)
        Amount of variance to retain in computing the inverse of the covariance matrix of X. If None, all variance.
    alpha_m: float (default: None)
        Amount of variance to retain in computing the inverse of the covariance matrix of M. If None, all variance.
    tmin: float (default: 0)
        The start of stimulation in seconds. Can be used if there was a delay in the marker.
    running: bool (default: False)
        Whether fit() is incremental: if False, each fit() call replaces the previous fit, using only the trials
        passed to that call. If True, each fit() call instead adds its trials to the ones seen in all previous
        fit() calls (i.e., keeps the spatial/temporal filter's running covariance from CCA(running=True) instead
        of discarding it), so a model can be trained gradually as more trials become available without redoing
        the full computation on all trials so far. Since rCCA's templates (Ts_/Tw_) are already always recomputed
        from the stimulus and the current filter, not the training trials themselves, this is mathematically
        exact: two calls fit(X1, y1) then fit(X2, y2) give the same filter as one call fit(concat(X1, X2),
        concat(y1, y2)). Not supported for ensemble=True, since each class's covariance would then be running on
        its own, and a class absent from an early batch would otherwise silently never get initialized. To start
        a new running fit from scratch, use a new instance (or call set_params(running=False) once, fit(), then
        set_params(running=True) again).
    cca_: list[TransformerMixin]
        The CCA used to fit the spatial and temporal filters. If ensemble=False, len(cca_)=1, otherwise
        len(cca_)=n_classes.
    events_: list
        The list of events used to map the stimulus to, as set by set_encoding_matrix().
    w_: NDArray
        The weight vector representing a spatial filter of shape (n_channels, n_components). If ensemble=True, then the
        shape is (n_channels, n_components, n_classes).
    r_: NDArray
        The weight vector representing a temporal filter of shape (n_events * n_event_samples, n_components). If
        ensemble=True, then the shape is (n_events * n_event_samples, n_components, n_classes).
    Ms_: NDArray
        The encoding matrix representing the events of shape (n_classes, n_features, n_samples) for stimulus cycle 1
        (i.e., it includes the onset of stimulation and does not contain the tails of previous cycles).
    Mw_: NDArray
        The encoding matrix representing the events of shape (n_classes, n_features, n_samples) for stimulus cycles 2
        and further (i.e., it does not include the onset of stimulation but does include the tails of previous
        cycles).
    Ts_: NDArray
        The template matrix representing the expected responses of shape (n_classes, n_components, n_samples) for
        stimulus cycle 1 (i.e., it includes the onset of stimulation and does not contain the tails of previous cycles).
    Tw_: NDArray
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

    classes_: NDArray
    cca_: list[TransformerMixin]
    events_: list
    w_: NDArray
    r_: NDArray
    Ms_: NDArray
    Mw_: NDArray
    Ts_: NDArray
    Tw_: NDArray
    _running_: dict = None

    def __init__(
        self,
        stimulus: NDArray,
        fs: int,
        event: str = "duration",
        onset_event: bool = False,
        decoding_length: float = None,
        decoding_stride: float = None,
        encoding_length: Union[float, list[float]] = None,
        encoding_stride: Union[float, list[float]] = None,
        score_metric: str = "correlation",
        latency: NDArray = None,
        ensemble: bool = False,
        amplitudes: NDArray = None,
        gamma_x: Union[float, list[float], NDArray] = None,
        gamma_m: Union[float, list[float], NDArray] = None,
        n_components: int = 1,
        squeeze_components: bool = True,
        alpha_x: float = None,
        alpha_m: float = None,
        tmin: float = 0,
        running: bool = False,
    ) -> None:
        self.stimulus = stimulus
        self.fs = fs
        self.event = event
        self.onset_event = onset_event
        self.decoding_length = decoding_length
        self.decoding_stride = decoding_stride
        self.encoding_length = encoding_length
        self.encoding_stride = encoding_stride
        self.score_metric = score_metric
        self.latency = latency
        self.ensemble = ensemble
        self.amplitudes = amplitudes
        self.gamma_x = gamma_x
        self.gamma_m = gamma_m
        self.n_components = n_components
        self.squeeze_components = squeeze_components
        self.alpha_x = alpha_x
        self.alpha_m = alpha_m
        self.tmin = tmin
        self.running = running

    def _resolve_decoding_length_stride(self) -> tuple[float, float]:
        """Resolve decoding_length and decoding_stride, defaulting to 1/fs (i.e., no phase-shifting) if None.

        Returns
        -------
        decoding_length: float
            The resolved decoding length in seconds.
        decoding_stride: float
            The resolved decoding stride in seconds.
        """
        decoding_length = 1 / self.fs if self.decoding_length is None else self.decoding_length
        decoding_stride = 1 / self.fs if self.decoding_stride is None else self.decoding_stride
        return decoding_length, decoding_stride

    def _get_T_full(
        self,
        n_samples: int,
    ) -> NDArray:
        """Get the templates, tiled (Ts_ followed by repeated Tw_) to the requested length. Used by decision_function()
        for both the batch and the running path, since (unlike eCCA's get_T()) no de-meaning is applied here, so a
        chunk at any given position range has the same value regardless of how many more samples are requested.

        Parameters
        ----------
        n_samples: int
            The number of samples requested.

        Returns
        -------
        T: NDArray
            The templates of shape (n_classes, n_components, n_samples).
        """
        if n_samples < self.Ts_.shape[2]:
            T = self.Ts_
        else:
            T = np.concatenate((self.Ts_, np.tile(self.Tw_, (1, 1, n_samples // self.Ts_.shape[2]))), axis=2)
        return T[:, :, :n_samples]

    def decision_function(
        self,
        X: NDArray,
        running: bool = False,
        reset: bool = False,
    ) -> NDArray:
        """Apply the classifier to get classification scores for X.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples). If running=True, this is only the
            newly observed samples since the previous call (not the full trial), see running below.
        running: bool (default: False)
            Whether to use running (incremental) scoring. If False (default), decision_function behaves exactly as
            without this parameter: X is the complete trial data seen so far, and everything is recomputed from
            scratch. If True, X is only the newly observed samples since the previous call, and a running state
            (kept internally, not a fitted attribute) is reused and updated; this is much cheaper when called
            repeatedly on a growing trial, e.g. from a dynamic stopping simulation loop, since each call only does
            O(n_new_samples) work instead of reprocessing the whole trial (this includes decoding_matrix's spatio-
            spectral filtering, not just the final score). Use reset=True on the first call of a new running
            sequence (e.g. for a new trial or a new batch of trials); the running state is otherwise unaffected by
            (and does not affect) running=False calls, and is cleared by fit(). Only supported for ensemble=False.
        reset: bool (default: False)
            Whether to discard any existing running state before processing this call. Only relevant if
            running=True; a never-yet-used instance already starts fresh without it, so it only needs to be set
            explicitly to start a new sequence before the previous one naturally ended.

        Returns
        -------
        scores: NDArray
            The similarity scores of shape (n_trials, n_classes, n_components) or (n_trials, n_classes) if
            n_components=1 and squeeze_components=True. If running=True, this is the cumulative score over all
            samples observed so far in the running sequence (not just the new chunk).
        """
        check_is_fitted(self)

        if not running:
            # Set decoding matrix
            decoding_length, decoding_stride = self._resolve_decoding_length_stride()
            if int(decoding_length * self.fs) > 1:
                X = decoding_matrix(X, int(decoding_length * self.fs), int(decoding_stride * self.fs))

            # Set templates to trial length
            T = self._get_T_full(X.shape[2])

            # Compute scores
            scores = np.zeros((X.shape[0], T.shape[0], self.n_components), dtype="float32")
            if self.ensemble:
                for i_class in range(T.shape[0]):
                    Xi = self.cca_[i_class].transform(X=X)[0]
                    for i_component in range(self.n_components):
                        if self.score_metric.lower() == "correlation":
                            scores[:, i_class, i_component] = correlation(
                                Xi[:, i_component, :], T[i_class, i_component, :]
                            )[:, 0]
                        elif self.score_metric.lower() == "euclidean":  # includes conversion to similarity
                            scores[:, i_class, i_component] = (
                                1 / (1 + euclidean(Xi[:, i_component, :], T[i_class, i_component, :]))[:, 0]
                            )
                        elif self.score_metric.lower() == "inner":
                            scores[:, i_class, i_component] = np.inner(
                                Xi[:, i_component, :], T[i_class, i_component, :]
                            )
                        else:
                            raise Exception(f"Unknown score metric: {self.score_metric}")

            else:
                X = self.cca_[0].transform(X=X)[0]
                for i_component in range(self.n_components):
                    if self.score_metric.lower() == "correlation":
                        scores[:, :, i_component] = correlation(X[:, i_component, :], T[:, i_component, :])
                    elif self.score_metric.lower() == "euclidean":  # includes conversion to similarity
                        scores[:, :, i_component] = 1 / (1 + euclidean(X[:, i_component, :], T[:, i_component, :]))
                    elif self.score_metric.lower() == "inner":
                        scores[:, :, i_component] = np.inner(X[:, i_component, :], T[:, i_component, :])
                    else:
                        raise Exception(f"Unknown score metric: {self.score_metric}")

            if self.n_components == 1 and self.squeeze_components:
                scores = scores[:, :, 0]

            return scores

        assert not self.ensemble, "running=True decision_function is not supported for ensemble=True."

        if reset or self._running_ is None:
            decoding_length, decoding_stride = self._resolve_decoding_length_stride()
            length = int(decoding_length * self.fs)
            stride = int(decoding_stride * self.fs)
            self._running_ = {
                "n_trials": X.shape[0],
                "n_samples": 0,
                "n_stable": 0,
                "length": length if length > 1 else 1,
                "stride": stride if length > 1 else 1,
                "raw_buffer": None,
                "component_state": [None] * self.n_components,
            }
        r = self._running_
        assert X.shape[0] == r["n_trials"], (
            f"running=True decision_function was called with {X.shape[0]} trials, but the running sequence was "
            f"started (or last continued) with {r['n_trials']}; call with reset=True to start a new sequence."
        )
        assert r["n_samples"] + X.shape[2] > 0, (
            "running=True decision_function requires at least 1 sample on the first call of a sequence."
        )

        # Extend the raw buffer with the new chunk, and run decoding_matrix (if used) over [buffer + new chunk]. Only
        # the leading part of this local, bounded-size window is far enough from the trailing (still unobserved)
        # edge to be unaffected by future samples (i.e., "stable"); see decision_function docstring.
        boundary = r["length"] - r["stride"]
        raw = X if r["raw_buffer"] is None else np.concatenate((r["raw_buffer"], X), axis=2)
        if r["length"] > 1:
            Xd = decoding_matrix(raw, r["length"], r["stride"])
        else:
            Xd = raw
        Xf = self.cca_[0].transform(X=Xd)[0]
        n_stable_new = max(0, r["n_samples"] + X.shape[2] - boundary)
        n_local_stable = n_stable_new - r["n_stable"]
        r["raw_buffer"] = raw[:, :, raw.shape[2] - min(boundary, raw.shape[2]) :]

        scores = np.zeros((X.shape[0], len(self.classes_), self.n_components), dtype="float32")
        T_zero_mean = np.zeros((1, len(self.classes_)))
        for i_component in range(self.n_components):
            T_chunk = self._get_T_full(n_stable_new)[:, i_component, r["n_stable"] : n_stable_new]
            # Only the updated state is used here, not the returned scores: a newly-stabilized chunk can be as
            # small as a single sample (whenever n_stable just crossed into positive territory), which for
            # score_metric="correlation" can make its (immediately discarded) instantaneous correlation degenerate
            # (0/0, from too few samples to estimate a variance) without affecting the (still exact) running state.
            with np.errstate(invalid="ignore"):
                _, r["component_state"][i_component] = _running_score(
                    Xf[:, i_component, :n_local_stable],
                    T_chunk,
                    T_zero_mean,
                    self.score_metric,
                    r["component_state"][i_component],
                )
            # Combine the (committed) stable state with the (uncommitted) provisional tail for this query's answer
            T_tail = self._get_T_full(r["n_samples"] + X.shape[2])[:, i_component, n_stable_new:]
            component_scores, _ = _running_score(
                Xf[:, i_component, n_local_stable:],
                T_tail,
                T_zero_mean,
                self.score_metric,
                r["component_state"][i_component],
            )
            if self.score_metric.lower() == "euclidean":  # includes conversion to similarity
                component_scores = 1 / (1 + component_scores)
            scores[:, :, i_component] = component_scores
        r["n_samples"] += X.shape[2]
        r["n_stable"] = n_stable_new

        if self.n_components == 1 and self.squeeze_components:
            scores = scores[:, :, 0]

        return scores

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> ClassifierMixin:
        """The training procedure to fit a rCCA on supervised EEG data.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples).
        y: NDArray
            The vector of ground-truth labels of the trials in X of shape (n_trials). Note, these denote the index at
            which to find the associated stimulus!

        Returns
        -------
        self: ClassifierMixin
            Returns the instance itself.
        """

        # Set encoding matrix
        self.set_encoding_matrix()
        n_classes = self.Ms_.shape[0]

        # Set decoding matrix
        decoding_length, decoding_stride = self._resolve_decoding_length_stride()
        if int(decoding_length * self.fs) > 1:
            X = decoding_matrix(X, int(decoding_length * self.fs), int(decoding_stride * self.fs))

        # Set structure matrices to trial length
        if X.shape[2] < self.Ms_.shape[2]:
            M = self.Ms_
        else:
            M = np.concatenate((self.Ms_, np.tile(self.Mw_, (1, 1, X.shape[2] // self.Ms_.shape[2]))), axis=2)
        M = M[:, :, : X.shape[2]]

        assert not (self.running and self.ensemble), "running=True is not supported for ensemble=True."

        # Fit w and r. If running=True and a running fit is already in progress (i.e. this is not the first fit()
        # call since the running sequence was last (re)started), reuse and add to it via CCA's own running=True
        # mechanism instead of discarding it; otherwise (running=False, or the first call of a new sequence) fit
        # fresh, exactly as before. Params that CCA reads on every fit() call (not just at construction) are kept
        # in sync in case they were changed via set_params() between calls.
        continuing = self.running and getattr(self, "cca_", None) and self.cca_[0].running
        if self.ensemble:
            self.w_ = np.zeros((X.shape[1], self.n_components, n_classes), dtype=X.dtype)
            self.r_ = np.zeros((M.shape[1], self.n_components, n_classes), dtype=X.dtype)
            self.cca_ = []
            for i_class in range(n_classes):
                self.cca_.append(
                    CCA(
                        n_components=self.n_components,
                        gamma_x=self.gamma_x,
                        gamma_y=self.gamma_m,
                        alpha_x=self.alpha_x,
                        alpha_y=self.alpha_m,
                    )
                )
                self.cca_[i_class].fit(X[y == i_class, :, :], M[y[y == i_class], :, :])
                self.w_[:, :, i_class] = self.cca_[i_class].w_x_
                self.r_[:, :, i_class] = self.cca_[i_class].w_y_
        else:
            if continuing:
                self.cca_[0].set_params(
                    n_components=self.n_components,
                    gamma_x=self.gamma_x,
                    gamma_y=self.gamma_m,
                    alpha_x=self.alpha_x,
                    alpha_y=self.alpha_m,
                )
            else:
                self.cca_ = [
                    CCA(
                        n_components=self.n_components,
                        gamma_x=self.gamma_x,
                        gamma_y=self.gamma_m,
                        alpha_x=self.alpha_x,
                        alpha_y=self.alpha_m,
                        running=self.running,
                    )
                ]
            self.cca_[0].fit(X, M[y, :, :])
            self.w_ = self.cca_[0].w_x_
            self.r_ = self.cca_[0].w_y_

        self.classes_ = np.arange(n_classes)
        self._is_fitted = True
        self.set_templates()
        return self

    def predict(
        self,
        X: NDArray,
        running: bool = False,
        reset: bool = False,
    ) -> NDArray:
        """The testing procedure to apply rCCA to novel EEG data.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples). If running=True, this is only the
            newly observed samples since the previous call, see decision_function().
        running: bool (default: False)
            Whether to use running (incremental) scoring, see decision_function().
        reset: bool (default: False)
            Whether to discard any existing running state before processing this call, see decision_function().

        Returns
        -------
        y: NDArray
            The predicted labels of shape (n_trials, n_components) or (n_trials) if n_components=1 and
            squeeze_components=True.
        """
        check_is_fitted(self)
        return np.argmax(self.decision_function(X, running=running, reset=reset), axis=1)

    def set_encoding_matrix(
        self,
    ) -> None:
        """Set the encoding matrix."""
        if self.encoding_length is None:
            encoding_length = 1
        else:
            encoding_length = (np.atleast_1d(self.encoding_length) * self.fs).astype("int")
        if self.encoding_stride is None:
            encoding_stride = 1
        else:
            encoding_stride = (np.atleast_1d(self.encoding_stride) * self.fs).astype("int")
        if self.amplitudes is None or self.amplitudes.shape[1] == 2 * self.stimulus.shape[1]:
            amplitude = self.amplitudes
        else:
            n = int(np.ceil(2 * self.stimulus.shape[1] / self.amplitudes.shape[1]))
            amplitude = np.tile(self.amplitudes, (1, n))[:, : 2 * self.stimulus.shape[1]]
        E, self.events_ = event_matrix(np.tile(self.stimulus, (1, 2)), self.event, self.onset_event)
        M = encoding_matrix(E, encoding_length, encoding_stride, amplitude, int(self.tmin * self.fs))
        self.Ms_ = M[:, :, : self.stimulus.shape[1]]
        self.Mw_ = M[:, :, self.stimulus.shape[1] :]

        # Correct for raster latency
        if self.latency is not None:
            self.Ms_ = correct_latency(self.Ms_, np.arange(len(self.latency)), self.latency, self.fs, axis=2)
            self.Mw_ = correct_latency(self.Mw_, np.arange(len(self.latency)), self.latency, self.fs, axis=2)

    def set_templates(self) -> None:
        """Set the templates."""
        try:
            check_is_fitted(self)
            M = np.concatenate((self.Ms_, self.Mw_), axis=2)
            if self.ensemble:
                T = np.zeros((M.shape[0], self.n_components, M.shape[2]))
                for i_class in range(M.shape[0]):
                    T[i_class, :, :] = self.cca_[i_class].transform(X=None, Y=M[[i_class], :, :])[1]
            else:
                T = self.cca_[0].transform(X=None, Y=M)[1]
            self.Ts_ = T[:, :, : self.stimulus.shape[1]]
            self.Tw_ = T[:, :, self.stimulus.shape[1] :]
            self._running_ = None
        except NotFittedError:
            pass

    def set_stimulus(
        self,
        stimulus: NDArray,
    ) -> None:
        """Set the stimulus, and as such change the templates.

        Parameters
        ----------
        stimulus: NDArray
            The stimulus used for stimulation of shape (n_classes, n_samples). Should be sampled at fs. One cycle (i.e.,
            one stimulus-repetition) is sufficient.
        """
        self.stimulus = stimulus
        self.set_encoding_matrix()
        self.set_templates()

    def set_amplitudes(
        self,
        amplitudes: NDArray,
    ) -> None:
        """Set the amplitudes, and as such change the templates.

        Parameters
        ----------
        amplitudes: NDArray
            The amplitude of the stimulus of shape (n_classes, n_samples). Should be sampled at fs.
        """
        self.amplitudes = amplitudes
        self.set_encoding_matrix()
        self.set_templates()

    def set_stimulus_amplitudes(
        self,
        stimulus: NDArray,
        amplitudes: NDArray,
    ) -> None:
        """Set the stimulus and the amplitudes, and as such change the templates.

        Parameters
        ----------
        stimulus: NDArray
            The stimulus used for stimulation of shape (n_classes, n_samples). Should be sampled at fs. One cycle (i.e.,
            one stimulus-repetition) is sufficient.
        amplitudes: NDArray
            The amplitude of the stimulus of shape (n_classes, n_samples). Should be sampled at fs.
        """
        self.stimulus = stimulus
        self.amplitudes = amplitudes
        self.set_encoding_matrix()
        self.set_templates()

    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status and return a Boolean value.

        Returns
        -------
        fitted: bool
            Whether the classifier is fitted.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
