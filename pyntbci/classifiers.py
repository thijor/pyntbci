from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from pyntbci.transformers import CCA, _solve_cca
from pyntbci.utilities import (
    RunningCovariance,
    correct_latency,
    correlation,
    decoding_matrix,
    encoding_matrix,
    euclidean,
    event_matrix,
    inner,
    smoothness_matrix,
)


SCORE_METRICS = ("correlation", "euclidean", "inner")
TEMPLATE_METRICS = ("mean", "median", "ocsvm")


def _score(
    score_metric: str,
    X: NDArray,
    T: NDArray,
) -> NDArray:
    """Compute similarity scores between (spatially filtered) single-trials and templates.

    Euclidean distance is converted to a similarity (1 / (1 + distance)) so that, like correlation and the inner
    product, a higher score means more similar. Shared by the batch (non-running) path of eCCA and rCCA, for both
    the ensemble and non-ensemble cases.

    Parameters
    ----------
    score_metric: str
        The score metric: one of SCORE_METRICS (correlation, euclidean, inner).
    X: NDArray
        The (spatially filtered) single-trials of shape (n_trials, n_samples).
    T: NDArray
        The templates of shape (n_classes, n_samples), or a single template of shape (n_samples,).

    Returns
    -------
    scores: NDArray
        The similarity scores of shape (n_trials, n_classes), or (n_trials, 1) if a single template was given.
    """
    metric = score_metric.lower()
    if metric == "correlation":
        return correlation(X, T)
    if metric == "euclidean":  # convert distance to a similarity so higher is more similar, as for the others
        return 1 / (1 + euclidean(X, T))
    if metric == "inner":
        return inner(X, T)
    raise ValueError(f"Unknown score metric: {score_metric}. Options are {SCORE_METRICS}.")


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
        raise ValueError(f"Unknown score metric: {score_metric}. Options are {SCORE_METRICS}.")


def _resolve_response_prior(response_prior: NDArray, n_features: int, n_events: int) -> NDArray:
    """Resolve a response_prior to the full temporal-feature length (matching the temporal filter r), or None.

    response_prior may be given either as one response (of length n_features // n_events), applied to every event,
    or as the full concatenation of the per-event responses (of length n_features), matching the layout of the
    temporal filter (see encoding_length). Shared by rCCA and UnsupervisedRCCA in classifiers.

    Parameters
    ----------
    response_prior: NDArray
        The prior on the expected transient response, sampled at fs. If None, None is returned.
    n_features: int
        The number of temporal features (the length of the temporal filter r).
    n_events: int
        The number of events the response is modeled for.

    Returns
    -------
    prior: NDArray
        The prior of shape (n_features,), or None if response_prior is None.
    """
    if response_prior is None:
        return None
    prior = np.asarray(response_prior, dtype="float64").ravel()
    if prior.size == n_features:
        return prior
    if n_features % n_events == 0 and prior.size == n_features // n_events:
        return np.tile(prior, n_events)  # one response for all events
    raise ValueError(
        f"response_prior has length {prior.size}, but must be either the per-event response length "
        f"({n_features // n_events}, applied to all {n_events} events) or the full concatenated length "
        f"({n_features})."
    )


def _apply_temporal_prior(
    prior: NDArray,
    gamma: float,
    smoothness: float,
    L: NDArray,
    w: NDArray,
    r: NDArray,
    Cxm: NDArray,
    Cmm: NDArray,
) -> tuple[NDArray, NDArray]:
    """Regularize a temporal response with a Gaussian prior (a prior mean and/or a smoothness prior), component-wise.

    Given the spatial filter w (kept fixed), the temporal response is re-estimated as the ridge/generalized-Tikhonov
    solution r = (Cmm + P)^-1 (Cmx w + lambda * mean), where the prior precision P = lambda I + kappa L combines two
    optional priors, both scaled to the temporal covariance so their strengths are dimensionless:

    - a prior *mean* (prior, e.g. an expected VEP shape): P gets lambda = gamma * trace(Cmm) / n_features on the
      identity, and the right-hand side gets lambda * mean. This interpolates the response from the data-driven
      estimate (lambda = 0) to the prior mean (lambda -> infinity). It anchors the response's absolute phase, which
      is what makes circularly-shifted codes decodable (an unconstrained response can otherwise slide to make any
      candidate fit); the (jointly sign-ambiguous) filters w and r are first sign-flipped to agree with the prior,
      else the blend is destructive.
    - a *smoothness* prior (L, the second-difference operator from smoothness_matrix): P gets kappa = smoothness *
      trace(Cmm) / n_features on L, penalizing sum_t (r[t] - r[t-1])^2 so r is temporally smooth. This is a
      zero-mean prior (it favors smoothness, not any particular shape), so it only enters the precision.

    Either prior may be omitted (prior=None, or smoothness=None/L=None). Shared by rCCA and UnsupervisedRCCA.

    Parameters
    ----------
    prior: NDArray
        The prior mean response of shape (n_features,), as resolved by _resolve_response_prior(), or None.
    gamma: float
        The strength of the prior-mean regularization (only used if prior is not None).
    smoothness: float
        The strength of the smoothness regularization (only used if not None).
    L: NDArray
        The smoothness penalty matrix of shape (n_features, n_features), see smoothness_matrix(). Only used if
        smoothness is not None.
    w: NDArray
        The spatial filter of shape (n_channels, n_components).
    r: NDArray
        The unregularized temporal response of shape (n_features, n_components).
    Cxm: NDArray
        The cross-covariance of the (decoded) EEG and the structure matrix of shape (n_channels, n_features).
    Cmm: NDArray
        The auto-covariance of the structure matrix of shape (n_features, n_features).

    Returns
    -------
    w: NDArray
        The (possibly sign-flipped) spatial filter of shape (n_channels, n_components).
    r: NDArray
        The prior-regularized temporal response of shape (n_features, n_components).
    """
    A = Cmm.astype("float64", copy=True)
    lam = 0.0
    if prior is not None:  # prior-mean ridge: identity precision scaled by the mean feature variance
        lam = gamma * np.trace(Cmm) / Cmm.shape[0]
        A += lam * np.eye(Cmm.shape[0])
    if smoothness is not None:  # smoothness precision scaled so its total (trace) matches smoothness * trace(Cmm)
        A += smoothness * np.trace(Cmm) / np.trace(L) * L
    w_new = np.array(w, dtype="float64")
    r_new = np.zeros_like(r, dtype="float64")
    for c in range(r.shape[1]):
        if prior is not None and r[:, c] @ prior < 0:  # align the joint sign ambiguity to agree with the prior mean
            w_new[:, c] = -w[:, c]
        rhs = Cxm.T @ w_new[:, c]
        if prior is not None:
            rhs = rhs + lam * (prior / np.linalg.norm(prior) * np.linalg.norm(r[:, c]))
        r_new[:, c] = np.linalg.solve(A, rhs)
    return w_new, r_new


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
            raise ValueError(f"Unknown template metric: {self.template_metric}. Options are {TEMPLATE_METRICS}.")
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
                        scores[:, i_class, i_component] = _score(
                            self.score_metric, Xi[:, i_component, :], T[i_class, i_component, :]
                        )[:, 0]

            else:
                X = self.cca_[0].transform(X=X)[0]
                for i_component in range(self.n_components):
                    scores[:, :, i_component] = _score(self.score_metric, X[:, i_component, :], T[:, i_component, :])

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
        if self.score_metric.lower() not in SCORE_METRICS:
            raise ValueError(f"Unknown score metric: {self.score_metric}. Options are {SCORE_METRICS}.")
        if self.template_metric.lower() not in TEMPLATE_METRICS:
            raise ValueError(f"Unknown template metric: {self.template_metric}. Options are {TEMPLATE_METRICS}.")

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
        A list containing all models learned for each of the databanks (clones of estimator).
    gate_: ClassifierMixin
        The fitted clone of gate. The passed-in estimator and gate are never mutated.
    """

    classes_: NDArray
    models_: list[ClassifierMixin]
    gate_: ClassifierMixin

    def __init__(
        self,
        estimator: ClassifierMixin,
        gate: ClassifierMixin,
    ) -> None:
        self.estimator = estimator
        self.gate = gate

    def _stack_scores(
        self,
        X: NDArray,
    ) -> NDArray:
        """Stack each databank model's decision_function scores along a new last axis.

        Parameters
        ----------
        X: NDArray
            The matrix of EEG data of shape (n_trials, n_channels, n_samples, n_items).

        Returns
        -------
        scores: NDArray
            The stacked scores of shape (n_trials, n_classes, n_items).
        """
        return np.stack([self.models_[i].decision_function(X[:, :, :, i]) for i in range(X.shape[3])], axis=2)

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
        return self.gate_.decision_function(self._stack_scores(X))

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
        if X.ndim != 4:
            raise ValueError(f"X must be 4D (n_trials, n_channels, n_samples, n_items); got {X.ndim}D.")

        # Fit a separate (cloned) model for each databank, so the passed-in estimator is never mutated
        self.models_ = [clone(self.estimator).fit(X[:, :, :, i], y) for i in range(X.shape[3])]

        # Fit gating on a clone, so the passed-in gate is never mutated
        self.gate_ = clone(self.gate)
        self.gate_.fit(self._stack_scores(X), y)

        self.classes_ = self.gate_.classes_
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
        return self.gate_.predict(self._stack_scores(X))


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
    encoding_length: float | list[float] (default: 0.3)
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
    response_prior: NDArray (default: None)
        A prior on the expected transient response (e.g. a flash-VEP: a negative peak near 75 ms, a positive peak
        near 100 ms, and a negative peak near 125 ms), sampled at fs, toward which the learned temporal filter r_ is
        softly regularized (see response_prior_gamma). Given either as one response of length n_event_samples
        (applied to every event) or as the full concatenation of the per-event responses of length n_features
        (matching r_; see encoding_length). Unlike in the unsupervised case, the (labeled) fit already estimates the
        response at the correct phase, so this acts as a regularizer toward a physiologically plausible shape, which
        can stabilize the response when little/noisy data is available. If None (default), no prior is used.
    response_prior_gamma: float (default: 1.0)
        The strength of the soft regularization toward response_prior, from 0 (ignore the prior, purely data-driven)
        upwards (larger pulls the response more strongly toward the prior; in the limit the response equals the
        prior). Only used if response_prior is not None.
    smoothness_m: float (default: None)
        The strength of a temporal-smoothness prior on the response r_: after the CCA, the temporal filter is
        re-estimated with a second-difference penalty (see smoothness_matrix in utilities) that penalizes the squared
        differences between adjacent response samples, favoring a smooth response (as commonly used when estimating
        temporal response functions). Applied per event (smoothness is not enforced across event boundaries). Ranges
        from 0 (no smoothing) upwards; the strength is scaled by the response covariance so it is dimensionless,
        comparable to gamma_m. If None (default), no smoothness prior is used. Composes with response_prior.
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
        response_prior: NDArray = None,
        response_prior_gamma: float = 1.0,
        smoothness_m: float = None,
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
        self.response_prior = response_prior
        self.response_prior_gamma = response_prior_gamma
        self.smoothness_m = smoothness_m

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
                        scores[:, i_class, i_component] = _score(
                            self.score_metric, Xi[:, i_component, :], T[i_class, i_component, :]
                        )[:, 0]

            else:
                X = self.cca_[0].transform(X=X)[0]
                for i_component in range(self.n_components):
                    scores[:, :, i_component] = _score(self.score_metric, X[:, i_component, :], T[:, i_component, :])

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
        if self.score_metric.lower() not in SCORE_METRICS:
            raise ValueError(f"Unknown score metric: {self.score_metric}. Options are {SCORE_METRICS}.")

        # Set encoding matrix
        self.set_encoding_matrix()
        n_classes = self.Ms_.shape[0]
        prior = _resolve_response_prior(self.response_prior, self.Ms_.shape[1], len(self.events_))
        L = smoothness_matrix(self._response_feature_lengths()) if self.smoothness_m is not None else None

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
                if prior is not None or L is not None:
                    self._apply_temporal_prior_to_cca(self.cca_[i_class], prior, L)
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
            if prior is not None or L is not None:
                self._apply_temporal_prior_to_cca(self.cca_[0], prior, L)
            self.w_ = self.cca_[0].w_x_
            self.r_ = self.cca_[0].w_y_

        self.classes_ = np.arange(n_classes)
        self.set_templates()
        return self

    def _response_feature_lengths(self) -> NDArray:
        """The number of temporal features (response samples) per event, i.e. the sizes of the per-event blocks of
        the temporal filter r_ (see encoding_length), as used to build the smoothness matrix."""
        n_events = len(self.events_)
        if self.encoding_length is None:
            length = np.ones(n_events, dtype=int)
        else:
            length = np.atleast_1d((np.asarray(self.encoding_length, dtype=float) * self.fs)).astype(int)
            if length.size == 1:
                length = np.repeat(length, n_events)
        if self.encoding_stride is None:
            stride = np.ones(n_events, dtype=int)
        else:
            stride = np.atleast_1d((np.asarray(self.encoding_stride, dtype=float) * self.fs)).astype(int)
            if stride.size == 1:
                stride = np.repeat(stride, n_events)
        return length // stride

    def _apply_temporal_prior_to_cca(self, cca: TransformerMixin, prior: NDArray, L: NDArray) -> None:
        """Overwrite a fitted CCA's spatial and temporal filters with the temporally regularized response (a
        prior-mean and/or smoothness ridge, see _apply_temporal_prior), so the templates built from them inherit the
        prior."""
        n_channels = cca.w_x_.shape[0]
        cca.w_x_, cca.w_y_ = _apply_temporal_prior(
            prior,
            self.response_prior_gamma,
            self.smoothness_m,
            L,
            cca.w_x_,
            cca.w_y_,
            cca.cov_xy_[:n_channels, n_channels:],
            cca.cov_y_,
        )

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


class UnsupervisedRCCA(ClassifierMixin, BaseEstimator):
    """Unsupervised adaptive reconvolution CCA classifier for calibration-free c-VEP decoding [6]_.

    Instead of a supervised calibration, each trial is decoded by fitting a separate rCCA per candidate stimulus (as
    a hypothesis) and selecting the stimulus whose model best fits the trial, i.e. yields the highest correlation
    between the spatially filtered EEG and the temporally filtered stimulus structure matrix. This is the
    instantaneous mode (`cumulative=False`), which treats every trial independently.

    Three cumulative extensions build a model from previously decoded trials, using their predicted labels as
    pseudo-labels (there are no ground-truth labels in a calibration-free setting):

    - `cumulative=True`: every hypothesis is fit on all previously decoded trials (at their pseudo-labels) plus the
      current trial (hypothesized as each candidate). This is mathematically identical to refitting from scratch on
      the full history every trial, but is done efficiently by keeping a single running covariance of the
      pseudo-labeled history (see `RunningCovariance` in `utilities`) shared across all hypotheses, so each trial
      only adds its own (bounded) contribution rather than reprocessing the whole history. The first trial, with no
      history, reduces to the instantaneous mode.
    - `confidence=True` (implies `cumulative`): each trial is weighted by a confidence, so that high-confidence
      trials drive the model updates and low-confidence trials are suppressed. The confidence is the normalized
      correlation margin `(rho_winner - rho_runner_up) / std(rho_except_winner)`, estimated from an instantaneous
      pass (as in [6]_), and used as a per-trial weight in the running covariance.
    - `posthoc=True` (implies `cumulative`): after each trial, all previously decoded trials are re-decoded with the
      just-updated (presumably better) model and their pseudo-labels are corrected, which then affects subsequent
      updates. This is the only mode that must retain the past trials' EEG (in `X_hist_`), since re-decoding needs
      the raw data; a changed label is applied to the running covariance as an exact remove-then-re-add, avoiding a
      full refit. The other modes keep no raw data (only the running covariance and the list of pseudo-labels).

    These flags reproduce the four variants of [6]_: instantaneous (all False), cumulative (`cumulative`),
    confidence-weighted cumulative (`cumulative`, `confidence`), and confidence-weighted cumulative with post hoc
    re-analysis (`cumulative`, `confidence`, `posthoc`).

    Note, decoding is inherently online and stateful: `predict()` streams trials in their given (chronological)
    order, decoding each with the model learned from the ones before it, and the internal session persists across
    calls. Decoding trials one at a time therefore accumulates exactly as decoding them in one call does (as needed
    for real-time use where trials arrive one by one): `[predict(X[[i]]) for i in range(n_trials)]` gives the same
    result as `predict(X)`. Because state persists, `predict()`/`decision_function()` are not pure functions; call
    `fit()` (or pass `reset=True`) to start a fresh session, e.g. before an independent replay. The single-trial
    `partial_fit_predict()` is the same online step exposed directly. The structure-matrix machinery of `rCCA`
    (event and encoding matrices, latency correction, optional spatio-spectral decoding matrix) is reused, and the
    core CCA is solved with the same `_solve_cca` as `CCA` in `transformers`. With short trials or a wide encoding
    matrix, the per-hypothesis covariances can be ill-conditioned; set `gamma_x`/`gamma_m` (or `alpha_x`/`alpha_m`)
    to regularize, as for supervised `rCCA`.

    Parameters
    ----------
    stimulus: NDArray
        The stimulus used for stimulation of shape (n_classes, n_samples). Should be sampled at fs. One cycle (i.e.,
        one stimulus-repetition) is sufficient.
    fs: int
        The sampling frequency of the EEG data in Hz.
    event: str (default: "duration")
        The event definition to map stimulus to events.
    onset_event: bool (default: True)
        Whether to add an event for the onset of stimulation. Added as last event.
    decoding_length: float (default: None)
        The length of the spectral filter for each data channel in seconds. If None, it is set to 1/fs, equivalent to 1
        sample, such that no phase-shifting is performed and thus no (spatio-)spectral filter is learned.
    decoding_stride: float (default: None)
        The stride of the spectral filter for each data channel in seconds. If None, it is set to 1/fs.
    encoding_length: float | list[float] (default: 0.3)
        The length of the transient response(s) for each of the events in seconds.
    encoding_stride: float | list[float] (default: None)
        The stride of the transient response(s) for each of the events in seconds. If None, it is set to 1/fs.
    latency: NDArray (default: None)
        The raster latencies of each of the classes of shape (n_classes,) that the templates need to be corrected for.
    tmin: float (default: 0)
        The start of stimulation in seconds. Can be used if there was a delay in the marker.
    n_components: int (default: 1)
        The number of CCA components to use. Decoding and confidence use the first component only, matching [6]_.
    gamma_x: float | list[float] | NDArray (default: None)
        Regularization on the covariance matrix for CCA along X (channels), see `rCCA`.
    gamma_m: float | list[float] | NDArray (default: None)
        Regularization on the covariance matrix for CCA along M (samples), see `rCCA`.
    alpha_x: float (default: None)
        Amount of variance to retain in computing the inverse of the covariance matrix of X. If None, all variance.
    alpha_m: float (default: None)
        Amount of variance to retain in computing the inverse of the covariance matrix of M. If None, all variance.
    cumulative: bool (default: True)
        Whether to learn cumulatively from previously decoded trials (using their pseudo-labels). If False, each
        trial is decoded instantaneously and independently.
    confidence: bool (default: False)
        Whether to weight each trial by its confidence during cumulative updates. Implies cumulative.
    posthoc: bool (default: False)
        Whether to re-decode and relabel all previous trials after each update. Implies cumulative, and retains the
        past trials' EEG in X_hist_.
    response_prior: NDArray (default: None)
        A prior on the expected transient response (e.g. a flash-VEP: a negative peak near 75 ms, a positive peak
        near 100 ms, and a negative peak near 125 ms), sampled at fs. Either one response of length n_event_samples
        (applied to every event) or the full concatenation of the per-event responses of length n_features (matching
        the temporal filter r_; see encoding_length). If given, the learned response is softly regularized toward it
        (see response_prior_gamma), which anchors the response's absolute phase. This is what makes decoding work for
        circularly-shifted codes (e.g. shifted m-sequences): without it, an unconstrained response can circularly
        slide to make every candidate stimulus fit equally well (the more so the longer encoding_length), so the
        classes become indistinguishable. If None (default), no prior is used.
    response_prior_gamma: float (default: 1.0)
        The strength of the soft regularization toward response_prior, ranging from 0 (ignore the prior, purely
        data-driven) upwards (larger pulls the response more strongly toward the prior; in the limit the response
        equals the prior). Only used if response_prior is not None.
    smoothness_m: float (default: None)
        The strength of a temporal-smoothness prior on the response, penalizing the squared differences between
        adjacent response samples (see smoothness_matrix in utilities and rCCA's smoothness_m), so the response is
        smooth. Unlike response_prior it makes no assumption about the response shape, so it does not by itself
        resolve circularly-shifted codes; it composes with response_prior (which anchors the phase) and reduces
        overfitting. Ranges from 0 (no smoothing) upwards. If None (default), no smoothness prior is used.

    Attributes
    ----------
    classes_: NDArray
        The class labels of shape (n_classes,).
    events_: list
        The list of events used to map the stimulus to, as set by the internal rCCA.
    labels_: list
        The pseudo-labels (predicted labels) of the decoded trials, in order.
    confidences_: list
        The confidence of each decoded trial, in order.
    w_: NDArray
        The spatial filter of the most recently winning model of shape (n_channels, n_components).
    r_: NDArray
        The temporal filter of the most recently winning model of shape (n_features, n_components).
    cov_: RunningCovariance
        The running covariance of the pseudo-labeled history (only populated if cumulative).
    X_hist_: list
        The (decoded) EEG of the decoded trials, retained only if posthoc, for re-decoding.

    References
    ----------
    .. [6] Thielen, J. (2026). Confidence-weighted cumulative rCCA with post hoc re-analysis: unsupervised adaptive
           learning for calibration-free c-VEP BCI. 10th Graz Brain-Computer Interface Conference 2026.
    """

    classes_: NDArray
    events_: list
    labels_: list
    confidences_: list
    w_: NDArray
    r_: NDArray
    cov_: RunningCovariance
    X_hist_: list

    def __init__(
        self,
        stimulus: NDArray,
        fs: int,
        event: str = "duration",
        onset_event: bool = True,
        decoding_length: float = None,
        decoding_stride: float = None,
        encoding_length: Union[float, list[float]] = 0.3,
        encoding_stride: Union[float, list[float]] = None,
        latency: NDArray = None,
        tmin: float = 0,
        n_components: int = 1,
        gamma_x: Union[float, list[float], NDArray] = None,
        gamma_m: Union[float, list[float], NDArray] = None,
        alpha_x: float = None,
        alpha_m: float = None,
        cumulative: bool = True,
        confidence: bool = False,
        posthoc: bool = False,
        response_prior: NDArray = None,
        response_prior_gamma: float = 1.0,
        smoothness_m: float = None,
    ) -> None:
        self.stimulus = stimulus
        self.fs = fs
        self.event = event
        self.onset_event = onset_event
        self.decoding_length = decoding_length
        self.decoding_stride = decoding_stride
        self.encoding_length = encoding_length
        self.encoding_stride = encoding_stride
        self.latency = latency
        self.tmin = tmin
        self.n_components = n_components
        self.gamma_x = gamma_x
        self.gamma_m = gamma_m
        self.alpha_x = alpha_x
        self.alpha_m = alpha_m
        self.cumulative = cumulative
        self.confidence = confidence
        self.posthoc = posthoc
        self.response_prior = response_prior
        self.response_prior_gamma = response_prior_gamma
        self.smoothness_m = smoothness_m

    def _setup(self) -> None:
        """Build the internal rCCA (for the structure matrices) and reset the running state."""
        self._rcca = rCCA(
            stimulus=self.stimulus,
            fs=self.fs,
            event=self.event,
            onset_event=self.onset_event,
            decoding_length=self.decoding_length,
            decoding_stride=self.decoding_stride,
            encoding_length=self.encoding_length,
            encoding_stride=self.encoding_stride,
            latency=self.latency,
            tmin=self.tmin,
            n_components=self.n_components,
            gamma_x=self.gamma_x,
            gamma_m=self.gamma_m,
            alpha_x=self.alpha_x,
            alpha_m=self.alpha_m,
        )
        self._rcca.set_encoding_matrix()
        self.events_ = self._rcca.events_
        self.classes_ = np.arange(self._rcca.Ms_.shape[0])
        self._response_prior_ = _resolve_response_prior(
            self.response_prior, self._rcca.Ms_.shape[1], len(self._rcca.events_)
        )
        self._L_ = smoothness_matrix(self._rcca._response_feature_lengths()) if self.smoothness_m is not None else None
        self.cov_ = RunningCovariance()
        self.labels_ = []
        self.confidences_ = []
        self.X_hist_ = []
        self.w_ = None
        self.r_ = None

    def _ensure_setup(self) -> None:
        """Set up on first use (so partial_fit_predict can be called without an explicit fit)."""
        if not hasattr(self, "cov_"):
            self._setup()

    def _structure_matrix(self, n_samples: int) -> NDArray:
        """Get the structure matrices for all classes, tiled to the requested length (as in rCCA.fit)."""
        Ms, Mw = self._rcca.Ms_, self._rcca.Mw_
        if n_samples < Ms.shape[2]:
            M = Ms[:, :, :n_samples]
        else:
            M = np.concatenate((Ms, np.tile(Mw, (1, 1, n_samples // Ms.shape[2]))), axis=2)[:, :, :n_samples]
        return M.astype("float64")

    def _decode(self, X: NDArray) -> NDArray:
        """Apply the spatio-spectral decoding matrix to a single trial (n_channels, n_samples) if used."""
        decoding_length, decoding_stride = self._rcca._resolve_decoding_length_stride()
        length = int(decoding_length * self.fs)
        if length > 1:
            return decoding_matrix(X[np.newaxis], length, int(decoding_stride * self.fs))[0]
        return X

    def _score(self, w: NDArray, r: NDArray, Xd: NDArray, Mi: NDArray) -> float:
        """The correlation between the spatially filtered EEG and the temporally filtered structure matrix."""
        xf = w[:, 0] @ Xd  # (n_samples,)
        tf = r[:, 0] @ Mi  # (n_samples,)
        return correlation(xf[np.newaxis, :], tf[np.newaxis, :])[0, 0]

    @staticmethod
    def _margin(rho: NDArray) -> float:
        """The confidence: normalized margin between the winning and runner-up correlations (see [6]_)."""
        order = np.sort(rho)
        denom = order[:-1].std()
        return float((order[-1] - order[-2]) / denom) if denom > 0 else 0.0

    def _fit_and_score(
        self,
        base: RunningCovariance,
        Xd: NDArray,
        M: NDArray,
        weight: float,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Fit one CCA per candidate class off a shared base covariance and score the current trial.

        Parameters
        ----------
        base: RunningCovariance
            The covariance of the history the candidates share (empty for instantaneous, the committed history for
            cumulative). Not mutated; each candidate peeks at base plus its own (Xd, M[i]) contribution.
        Xd: NDArray
            The (decoded) current trial of shape (n_channels, n_samples).
        M: NDArray
            The structure matrices of all classes of shape (n_classes, n_features, n_samples).
        weight: float
            The weight of the current trial's contribution (None or 1 for unweighted).

        Returns
        -------
        rho: NDArray
            The per-class correlation scores of shape (n_classes,).
        w_all: NDArray
            The per-class spatial filters of shape (n_classes, n_channels, n_components).
        r_all: NDArray
            The per-class temporal filters of shape (n_classes, n_features, n_components).
        """
        n_channels, n_features = Xd.shape[0], M.shape[1]
        Xt = Xd.T
        rho = np.zeros(self.classes_.size)
        w_all = np.zeros((self.classes_.size, n_channels, self.n_components))
        r_all = np.zeros((self.classes_.size, n_features, self.n_components))
        for i in range(self.classes_.size):
            cov = base.peek(np.concatenate((Xt, M[i].T), axis=1), weights=weight).covariance
            Cxm = cov[:n_channels, n_channels:]
            Cmm = cov[n_channels:, n_channels:]
            w_all[i], r_all[i], _ = _solve_cca(
                cov[:n_channels, :n_channels],
                Cxm,
                Cmm,
                self.n_components,
                self.gamma_x,
                self.gamma_m,
                self.alpha_x,
                self.alpha_m,
            )
            if self._response_prior_ is not None or self._L_ is not None:  # anchor the phase and/or smooth the response
                w_all[i], r_all[i] = _apply_temporal_prior(
                    self._response_prior_,
                    self.response_prior_gamma,
                    self.smoothness_m,
                    self._L_,
                    w_all[i],
                    r_all[i],
                    Cxm,
                    Cmm,
                )
            rho[i] = self._score(w_all[i], r_all[i], Xd, M[i])
        return rho, w_all, r_all

    def _relabel(self, M: NDArray) -> None:
        """Post hoc re-analysis: re-decode all previous trials with the current model and correct their labels."""
        j = len(self.X_hist_) - 1  # index of the just-decoded (current) trial; only relabel the ones before it
        if j <= 0:
            return
        Tproj = np.einsum("l,nlt->nt", self.r_[:, 0], M)  # (n_classes, n_samples), the filtered templates
        for k in range(j):
            xf = self.w_[:, 0] @ self.X_hist_[k]  # (n_samples,)
            new = int(np.argmax(correlation(xf[np.newaxis, :], Tproj)))
            old = self.labels_[k]
            if new != old:
                weight = self.confidences_[k] if self.confidence else 1.0
                Xt = self.X_hist_[k].T
                self.cov_.update(np.concatenate((Xt, M[old].T), axis=1), weights=weight, sign=-1)
                self.cov_.update(np.concatenate((Xt, M[new].T), axis=1), weights=weight, sign=1)
                self.labels_[k] = new

    def partial_fit_predict(self, X: NDArray, update: bool = True) -> tuple[int, float, NDArray]:
        """Decode a single trial online, optionally committing it to the model with its pseudo-label.

        Parameters
        ----------
        X: NDArray
            The EEG data of a single trial of shape (n_channels, n_samples) or (1, n_channels, n_samples).
        update: bool (default: True)
            Whether to commit this trial to the online model (updating the running covariance and the
            pseudo-label/confidence/history state) after decoding it. If False, the trial is decoded against the
            current model but no state is changed, i.e. a pure, side-effect-free query that can be repeated any
            number of times on the same or a growing trial (as a dynamic-stopping loop does, decoding growing
            segments of a trial until it decides to stop) without polluting the model; commit the decided trial once
            afterwards with a single update=True call. Only meaningful for cumulative variants (with cumulative=False
            there is no cross-trial state to update).

        Returns
        -------
        label: int
            The predicted (pseudo-) label of the trial.
        confidence: float
            The confidence of the prediction.
        scores: NDArray
            The per-class correlation scores of shape (n_classes,).
        """
        self._ensure_setup()
        if X.ndim == 3:
            assert X.shape[0] == 1, "partial_fit_predict decodes a single trial at a time."
            X = X[0]
        Xd = self._decode(X)
        M = self._structure_matrix(Xd.shape[1])

        # An instantaneous pass is needed either as the prediction itself (not cumulative) or to derive the
        # confidence weight before the cumulative update (confidence)
        confidence = None
        if not self.cumulative or self.confidence:
            rho, w_all, r_all = self._fit_and_score(RunningCovariance(), Xd, M, None)
            confidence = self._margin(rho)

        if self.cumulative:
            weight = confidence if self.confidence else 1.0
            rho, w_all, r_all = self._fit_and_score(self.cov_, Xd, M, weight)
            if confidence is None:  # vanilla cumulative: report a confidence from the cumulative scores
                confidence = self._margin(rho)

        label = int(np.argmax(rho))

        if update:  # commit this trial to the online model (else this is a read-only query, leaving state untouched)
            self.w_, self.r_ = w_all[label], r_all[label]
            self.labels_.append(label)
            self.confidences_.append(confidence)
            if self.cumulative:
                weight = confidence if self.confidence else 1.0
                self.cov_.update(np.concatenate((Xd.T, M[label].T), axis=1), weights=weight)
                if self.posthoc:
                    self.X_hist_.append(Xd)
                    self._relabel(M)

        return label, confidence, rho

    def fit(self, X: NDArray = None, y: NDArray = None) -> ClassifierMixin:
        """Set up the classifier (calibration-free: no training data or labels are used).

        Parameters
        ----------
        X: NDArray (default: None)
            Not used, present for scikit-learn API consistency.
        y: NDArray (default: None)
            Not used, present for scikit-learn API consistency.

        Returns
        -------
        self: ClassifierMixin
            Returns the instance itself.
        """
        self._setup()
        return self

    def _run(self, X: NDArray, reset: bool, update: bool) -> tuple[NDArray, NDArray]:
        """Stream all trials in order, returning predictions and scores. Continues the current online session
        (accumulating onto whatever has already been decoded), or starts a fresh one if reset=True. If update=False,
        the trials are only decoded (no state is changed), see partial_fit_predict."""
        if reset:
            self._setup()
        else:
            self._ensure_setup()
        yh = np.zeros(X.shape[0], dtype="int64")
        scores = np.zeros((X.shape[0], self.classes_.size))
        for j in range(X.shape[0]):
            yh[j], _, scores[j, :] = self.partial_fit_predict(X[j], update=update)
        return yh, scores

    def predict(self, X: NDArray, reset: bool = False, update: bool = True) -> NDArray:
        """Decode a sequence of trials online, in the given (chronological) order.

        Each trial is decoded with the model learned from all trials decoded so far, and then folds into that model
        (using its own prediction as a pseudo-label) if cumulative. This is a stateful, online operation: the
        internal session persists across calls, so decoding trials one at a time is equivalent to decoding them in a
        single call, i.e. ``[predict(X[[i]]) for i in range(n_trials)]`` gives the same result as ``predict(X)``, as
        needed for real-time use where trials arrive one by one. Because it persists state, predict() is not a pure
        function: call fit() (or pass reset=True) to start a fresh session, e.g. before an independent replay.

        Parameters
        ----------
        X: NDArray
            The EEG data of shape (n_trials, n_channels, n_samples), in chronological order.
        reset: bool (default: False)
            Whether to discard the current online session and start fresh before decoding X. Use reset=True (or a
            fresh instance, or fit()) for a self-contained replay; leave False to continue an ongoing session.
        update: bool (default: True)
            Whether to commit the decoded trials to the online model. Use update=False for a pure, side-effect-free
            decode (nothing is committed), e.g. to repeatedly probe growing segments of a trial in a dynamic-stopping
            loop without polluting the model, then commit the decided trial once with update=True. See
            partial_fit_predict.

        Returns
        -------
        y: NDArray
            The predicted labels of shape (n_trials,).
        """
        return self._run(X, reset, update)[0]

    def decision_function(self, X: NDArray, reset: bool = False, update: bool = True) -> NDArray:
        """Decode a sequence of trials online and return the per-trial per-class correlation scores.

        Stateful and online, see predict() (of which this is the score-returning counterpart).

        Parameters
        ----------
        X: NDArray
            The EEG data of shape (n_trials, n_channels, n_samples), in chronological order.
        reset: bool (default: False)
            Whether to discard the current online session and start fresh before decoding X, see predict().
        update: bool (default: True)
            Whether to commit the decoded trials to the online model, see predict(). Use update=False for a pure,
            side-effect-free scoring (e.g. probing growing segments of a trial in a dynamic-stopping loop).

        Returns
        -------
        scores: NDArray
            The per-trial per-class correlation scores of shape (n_trials, n_classes).
        """
        return self._run(X, reset, update)[1]
