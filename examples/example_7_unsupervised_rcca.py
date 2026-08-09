"""
Unsupervised rCCA
=================
This script shows how to use unsupervised adaptive rCCA from PyntBCI for calibration-free decoding of c-VEP trials [4]_.
Unlike the supervised rCCA (see the rCCA example), the unsupervised variant needs no calibration data: each trial is
decoded by fitting a separate rCCA per candidate stimulus (as a hypothesis) and selecting the stimulus whose model
best fits the data. This "instantaneous" mode can be improved by cumulatively learning from previously decoded trials,
using their predicted labels as pseudo-labels [1,2,3]_. Two further extensions are shown: confidence-weighting (updates
are driven mostly by confidently decoded trials) and post hoc re-analysis (past trials are re-decoded, and their
pseudo-labels corrected, with a later, better model).

References
----------
.. [1] Thielen, J. (2026). Confidence-weighted cumulative rCCA with post hoc re-analysis: unsupervised adaptive
       learning for calibration-free c-VEP BCI. 10th Graz Brain-Computer Interface Conference 2026.
.. [2] Thielen, J., & Tangermann, M. (2025, October). Exploring new territory II: Calibration-free decoding for ERP BCI.
       In 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC) (pp. 3788-3793). IEEE.
.. [3] Thielen, J., Sosulski, J., & Tangermann, M. (2024). Exploring new territory: Calibration-free decoding for c-VEP
       BCI. 9th Graz Brain-Computer Interface Conference 2024.
.. [4] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
       code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
       056007.
"""

import matplotlib.pyplot as plt
import numpy as np

import pyntbci

# %%
# Simulate data
# -------------
# The cell below simulates synthetic c-VEP data in response to a set of Gold codes (as in [1]_). The trials are put in
# a random (chronological) order to mimic an online copy-spelling session, in which the unsupervised classifier sees
# one trial at a time and adapts as it goes.

FS = 120  # sampling frequency
PR = 60  # presentation rate

V = pyntbci.stimulus.make_gold_codes()[:12]
V = np.repeat(V, FS // PR, axis=1)
N_CLASSES = V.shape[0]
CYCLE_SIZE = V.shape[1] / FS

N_TRIALS = 3 * N_CLASSES
N_CHANNELS = 8
N_SAMPLES = int(3 * CYCLE_SIZE * FS)
SEED = 42

X, y, V = pyntbci.eeg.generate_c_vep(
    N_TRIALS, N_CHANNELS, N_SAMPLES, FS, n_classes=N_CLASSES, stimulus=V, primary_channels=4, random_state=SEED
)

# Shuffle the trials into a random chronological order
rng = np.random.default_rng(SEED)
order = rng.permutation(N_TRIALS)
X, y = X[order, :, :], y[order]

# All variants share the same rCCA configuration. A "refe" event with an onset event is used (as in [1]_). Because the
# once-per-trial onset response makes the response covariance rank-deficient (and, at short trial durations, there are
# few samples relative to the response length), alpha_m truncates the near-degenerate directions of the response
# covariance (as for supervised rCCA with little data).
RCCA_KWARGS = dict(stimulus=V, fs=FS, event="refe", onset_event=True, encoding_length=0.3, alpha_m=0.99)

# %%
# Inspect data
# ------------

print("X", X.shape, "(trials x channels x samples)", X.dtype)  # EEG
print("y", y.shape, "(trials)", y.dtype)  # labels
print("V", V.shape, "(classes, samples)", V.dtype)  # codes
print("fs", FS, "Hz")  # sampling frequency
print("fr", PR, "Hz")  # presentation rate

# %%
# Unsupervised decoding
# ---------------------
# `UnsupervisedRCCA` is calibration-free, so there is no supervised `fit(X, y)`. Instead, `predict(X)` streams the
# trials in the given (chronological) order and decodes each one with the model learned from the trials before it. The
# four variants of [1]_ are selected with the `cumulative`, `confidence`, and `posthoc` flags. Shown on short
# (one-cycle) trials, where a single trial carries too little information to decode well on its own, cumulative
# learning from previously decoded trials clearly helps.

n_samples = int(CYCLE_SIZE * FS)  # one cycle

# Instantaneous: every trial decoded independently, no learning across trials
rcca_i = pyntbci.classifiers.UnsupervisedRCCA(**RCCA_KWARGS, cumulative=False)
yh_i = rcca_i.predict(X[:, :, :n_samples])
print(f"Instantaneous accuracy: {np.mean(yh_i == y):.2f}")

# Cumulative: learn from previously decoded trials using their (pseudo-)labels
rcca_c = pyntbci.classifiers.UnsupervisedRCCA(**RCCA_KWARGS, cumulative=True)
yh_c = rcca_c.predict(X[:, :, :n_samples])
print(f"Cumulative accuracy:    {np.mean(yh_c == y):.2f}")

# %%
# Decoding curve
# --------------
# Following [1]_, the four variants are compared with a decoding curve: the classification accuracy as a function of
# the single-trial duration. For each trial duration, the full online session is re-run from scratch on the truncated
# trials. Cumulative learning clearly improves over the instantaneous baseline, especially at shorter trial durations,
# and the post hoc re-analysis squeezes out a bit more at the short end; the variants converge once single trials are
# long enough to decode well on their own.

variants = {
    "instantaneous": dict(cumulative=False),
    "cumulative": dict(cumulative=True),
    "confidence": dict(cumulative=True, confidence=True),
    "posthoc": dict(cumulative=True, confidence=True, posthoc=True),
}

trial_sizes = np.array([0.5, 1.0, 1.5, 2.0, 3.0]) * CYCLE_SIZE  # in seconds
accuracy = np.zeros((len(variants), trial_sizes.size))
for i_variant, (name, flags) in enumerate(variants.items()):
    for i_size, trial_size in enumerate(trial_sizes):
        rcca = pyntbci.classifiers.UnsupervisedRCCA(**RCCA_KWARGS, **flags)
        yh = rcca.predict(X[:, :, : int(trial_size * FS)])
        accuracy[i_variant, i_size] = np.mean(yh == y)

# Plot decoding curves
plt.figure(figsize=(15, 4))
for i_variant, name in enumerate(variants):
    plt.plot(trial_sizes, accuracy[i_variant, :], linestyle="-", marker="o", label=name)
plt.axhline(1 / N_CLASSES, color="k", linestyle="--", alpha=0.5, label="chance")
plt.xlabel("trial duration [s]")
plt.ylabel("accuracy")
plt.legend()
plt.title("Unsupervised rCCA decoding curve")
plt.tight_layout()
