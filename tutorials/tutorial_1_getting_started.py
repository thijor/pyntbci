"""
Getting started
===============

This getting started tutorial shows an example of how to use the PyntBCI library for analysing code-modulated responses.
This tutorial makes use of a small dataset of EEG data, recorded from one participant during a single session. This data
is already minimally preprocessed following a spectral filter with a band-pass at 2-30 Hz and a downsample from 2 kHz to
240 Hz. The stimuli that were shown are a circularly shifted modulated Gold code (shift register length 6, shift between
stimuli 2 bits), presented at 60 Hz. The participant focused on each of the 32 stimuli once in a 4x8 matrix speller,
where each presentation lasted 4.2 seconds (2 code cycles) after a 0.8 second cue.

In this notebook, the reconvolution CCA (rCCA) method for decoding EEG is demonstrated, see [1]_ and [2]_. Additionally,
rCCA is compared to eCCA, the so-called reference pipeline as discussed in the c-VEP review [3]_.

References
----------
.. [1] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
       code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
       056007. DOI: https://doi.org/10.1088/1741-2552/abecef
.. [2] Thielen, J., van den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials:
       re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797.
       DOI: https://doi.org/10.1371/journal.pone.0133797
.. [3] Martínez-Cagigal, V., Thielen, J., Santamaria-Vazquez, E., Pérez-Velasco, S., Desain, P., & Hornero, R. (2021).
       Brain–computer interfaces based on code-modulated visual evoked potentials (c-VEP): A literature review. Journal
       of Neural Engineering, 18(6), 061002. DOI: https://doi.org/10.1088/1741-2552/ac38cf
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn

import pyntbci

seaborn.set_context("paper", font_scale=1.5)

# %%
# The data
# --------
# The dataset consists of: (1) The EEG data X that is a matrix of k trials, c channels, and m samples; (2) The labels y
# that is a vector of k trials; (3) The pseudo-random noise-codes V that is a matrix of stimuli with n classes and m
# samples. Note, the stimuli are upsampled to the EEG sampling frequency and contain only one stimulus-cycle. During a
# trial, however, the stimuli were repeated 2 times (2 stimulus cycles).

# Path to pyntbci (to read the tutorial data and standard cap files)
path = os.path.join(os.path.dirname(pyntbci.__file__))

# Load tutorial data
tmp = np.load(os.path.join(path, "data", "tutorial.npz"))
X = tmp["X"]
y = tmp["y"]
V = tmp["V"]
fs = int(tmp["fs"])
fr = 60
print("X", X.shape, "(trials x channels x samples)")  # EEG
print("y", y.shape, "(trials)")  # labels
print("V", V.shape, "(classes, samples)")  # codes
print("fs", fs, "Hz")  # sampling frequency
print("fr", fr, "Hz")  # presentation rate

# Extract data dimensions
n_trials, n_channels, n_samples = X.shape
n_classes = V.shape[0]

# Read cap file
capfile = os.path.join(path, "capfiles", "biosemi64.loc")
with open(capfile, "r") as fid:
    channels = []
    for line in fid.readlines():
        channels.append(line.split("\t")[-1].strip())
print("Channels:", ", ".join(channels))

# Visualize EEG data
i_trial = 0  # the trial to visualize
plt.figure(figsize=(15, 15))
plt.plot(np.arange(0, n_samples) / fs, 25e-6 * np.arange(n_channels) + X[i_trial, :, :].T)
plt.xlim([0, 1])  # limit to 1 second EEG data
plt.yticks(25e-6 * np.arange(n_channels), channels)
plt.xlabel("time [s]")
plt.ylabel("channel")
plt.title(f"Single-trial multi-channel EEG time-series (trial {i_trial})")
plt.tight_layout()

# Visualize labels
plt.figure(figsize=(15, 3))
hist = np.histogram(y, bins=np.arange(n_classes + 1))[0]
plt.bar(np.arange(n_classes), hist)
plt.xticks(np.arange(n_classes))
plt.xlabel("label")
plt.ylabel("count")
plt.title("Single-trial labels")
plt.tight_layout()

# Visualize stimuli
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
pyntbci.plotting.stimplot(V, fs=fs, ax=ax, plotfs=False)
fig.tight_layout()
ax.set_title("Stimulus time-series")

# %%
# The event matrix
# ----------------
# The first step for reconvolution is to find within the sequences the repetitive events. This can be imposed "manually"
# by choosing the event definition that we believe the brain responds to. Here, the so-called "duration" event is used,
# which marks the length of a flash as the important piece of information. As the sequences in this dataset were
# modulated, there are only two events: a short and a long flash. Additionally, a third event is added that will account
# for the onset of a trial, during which all of a sudden the screen started flashing. The event matrix is a matrix of n
# classes, e events, and m samples.
#
# Please, note that more event definitions exist, which can be explored with the `event` variable of `rCCA`. For
# instance, `event="contrast"` is a useful event definition as well, which looks at rising and falling edges,
# generalising over the length of a flash.

# Create event matrix
E, events = pyntbci.utilities.event_matrix(V, event="duration", onset_event=True)
print("E:", E.shape, "(classes x events x samples)")
print("Events:", ", ".join([str(event) for event in events]))

# Visualize event time-series
i_class = 0  # the class to visualize
fig, ax = plt.subplots(1, 1, figsize=(15, 3))
pyntbci.plotting.eventplot(V[i_class, ::int(fs/fr)], E[i_class, :, ::int(fs/fr)], fs=fr, ax=ax, events=events)
ax.set_title(f"Event time-series (code {i_class})")
plt.tight_layout()

# Visualize event matrix
i_class = 0
plt.figure(figsize=(15, 3))
plt.imshow(E[i_class, :, :], cmap="gray")
plt.gca().set_aspect(10)
plt.xticks(np.arange(0, E.shape[2], 60), np.arange(0, E.shape[2], 60) / fs)
plt.yticks(np.arange(E.shape[1]), events)
plt.xlabel("time [s]")
plt.title(f"Event matrix (class {i_class})")
plt.tight_layout()

# %%
# The structure matrix
# --------------------
# The second step for reconvolution is to model the expected responses associated to each of the events and their
# overlap. This is done in the so-called structure matrix (or design matrix). The structure matrix is essentially a
# Toeplitz version of the event matrix. It allows to model the c-VEP as the dot product of r (the transient response to
# an event) and M (the structure matrix for a specific class) for the ith class. The structure matrix is a matrix of n
# classes, l response samples, and m samples.
#
# An important parameter here is the `encoding_length` argument. An easy abstraction is to assume the same length for
# the responses to each of the events. However, one could also set different lengths for each of the events.

# Create structure matrix
encoding_length = int(0.3 * fs)  # 300 ms responses
M = pyntbci.utilities.encoding_matrix(E, encoding_length)
print("M:", M.shape, "(classes x encoding_length*events x samples)")

# Plot structure matrix
i_class = 0  # the class to visualize
plt.figure(figsize=(15, 6))
plt.imshow(M[i_class, :, :], cmap="gray")
plt.xticks(np.arange(0, M.shape[2], 60), np.arange(0, M.shape[2], 60) / fs)
plt.yticks(np.arange(0, E.shape[1] * encoding_length, 12), np.tile(np.arange(0, encoding_length, 12) / fs, E.shape[1]))
plt.xlabel("time [s]")
plt.ylabel(events[::-1])
plt.title(f"Structure matrix (class {i_class})")
plt.tight_layout()

# %%
# Reconvolution CCA
# -----------------
# The full reconvolution CCA (rCCA) pipeline is implemented as a scikit-learn compatible class in PyntBCI in
# `pyntbci.classifiers.rCCA`. All it needs are the binary sequences `stimulus`, the sampling frequency `fs`, the event
# definition `event`, the transient response size `encoding_length` and whether to include an event for the onset of a
# trial `onset_event`.
#
# When calling `rCCA.fit(X, y)` with training data `X` and labels `y`, the full decomposition is performed to obtain
# spatial filters `rCCA.w_` and temporal filter `rCCA.r_`.
#
# Please note that the transient responses are concatenated in this temporal filter `rCCA.r_`. One can use
# `rCCA.events_` to disentangle these and find which response is associated to which event.

# Perform CCA decomposition with duration event
encoding_length = 0.3  # 300 ms responses
rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=encoding_length, onset_event=True)
rcca.fit(X, y)
print("w: ", rcca.w_.shape, "(channels)")
print("r: ", rcca.r_.shape, "(encoding_length*events)")

# Plot CCA filters
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
pyntbci.plotting.topoplot(rcca.w_, capfile, ax=ax[0])
ax[0].set_title("Spatial filter")
tmp = np.reshape(rcca.r_, (len(rcca.events_), -1))
for i in range(len(rcca.events_)):
    ax[1].plot(np.arange(int(encoding_length * fs)) / fs, tmp[i, :])
ax[1].legend(rcca.events_)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("amplitude [a.u.]")
ax[1].set_title("Transient responses")
fig.tight_layout()

# %%
# Cross-validation
# ----------------
# To perform decoding, one can call `rCCA.fit(X_trn, y_trn)` on training data `X_trn` and labels `y_trn` and
# `rCCA.predict(X_tst)` on testing data `X_tst`. In this section, a chronological cross-validation is set up to evaluate
# the performance of rCCA.
#
# Additionally, a second classifier is introduced, `eCCA`, which is the so-called "reference" method for c-VEP decoding.
# Instead of using reconvolution for template generation (rCCA), eCCA computes templates by computing average responses
# to repeated trials. As in this dataset a single circularly shifted code was used, we can compute one template for this
# code, and circularly shift it to generate templates for all other classes. Therefore, eCCA requires a `lags` parameter
# that specifies the relationship between the different classes.

# Chronological cross-validation
n_folds = 4
n_trials = int(X.shape[0] / n_folds)
folds = np.repeat(np.arange(n_folds), n_trials)

# Loop folds
accuracy = np.zeros((2, n_folds))
for i_fold in range(n_folds):
    # Split data to train and test set
    X_trn, y_trn = X[folds != i_fold, ...], y[folds != i_fold]
    X_tst, y_tst = X[folds == i_fold, ...], y[folds == i_fold]

    # rCCA
    rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="contrast", encoding_length=0.3, onset_event=True)
    rcca.fit(X_trn, y_trn)
    yh_tst = rcca.predict(X_tst)
    accuracy[0, i_fold] = np.mean(yh_tst == y_tst)

    # eCCA
    ecca = pyntbci.classifiers.eCCA(lags=np.arange(0, 2 * 63, 4) / 60, fs=fs, cycle_size=2 * 63 / 60)
    ecca.fit(X_trn, y_trn)
    yh_tst = ecca.predict(X_tst)
    accuracy[1, i_fold] = np.mean(yh_tst == y_tst)

# Plot accuracy (over folds)
plt.figure(figsize=(15, 3))
plt.bar(-0.2 + np.arange(n_folds), accuracy[0, :], 0.4, label="rCCA")
plt.bar(0.2 + np.arange(n_folds), accuracy[1, :], 0.4, label="eCCA")
plt.axhline(1 / n_classes, color="k", linestyle="--", label="chance", alpha=0.5)
plt.xticks(np.arange(n_folds))
plt.xlabel("(test) fold")
plt.ylabel("accuracy")
plt.legend()
plt.title("Chronological cross-validation")
plt.tight_layout()

# %%
# Learning curve
# --------------
# When comparing eCCA and rCCA, one can appreciate that rCCA typically requires fewer data than eCCA. The reason for
# this is that rCCA reduce the number of free parameters to those of the transient responses instead of the full c-VEP,
# which at the same time allows to increase the amount of data to perform a kind of average over. This can be observed
# in the so-called learning curve, which shows the performance as a function of the amount of training data.

# Chronological cross-validation
n_folds = 4
n_trials = int(X.shape[0] / n_folds)
folds = np.repeat(np.arange(n_folds), n_trials)

# Loop folds
accuracy = np.zeros((2, n_trials * (n_folds - 1), n_folds))
for i_fold in range(n_folds):

    # Split data to train and test set
    X_trn, y_trn = X[folds != i_fold, ...], y[folds != i_fold]
    X_tst, y_tst = X[folds == i_fold, ...], y[folds == i_fold]

    # Loop trials for the learning curve
    for i_trial in range(n_trials * (n_folds - 1)):
        # rCCA
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=0.3, onset_event=True)
        rcca.fit(X_trn[:1 + i_trial, ...], y_trn[:1 + i_trial])
        yh_tst = rcca.predict(X_tst)
        accuracy[0, i_trial, i_fold] = np.mean(yh_tst == y_tst)

        # eCCA
        ecca = pyntbci.classifiers.eCCA(lags=np.arange(0, 2 * 63, 4) / 60, fs=fs, cycle_size=63 / 60)
        ecca.fit(X_trn[:1 + i_trial, ...], y_trn[:1 + i_trial])
        yh_tst = ecca.predict(X_tst)
        accuracy[1, i_trial, i_fold] = np.mean(yh_tst == y_tst)

# Plot learning curve
plt.figure(figsize=(15, 3))
avg = accuracy[0, ...].mean(axis=-1)
std = accuracy[0, ...].std(axis=-1)
plt.plot(np.arange(n_trials * (n_folds - 1)), avg, label="rCCA")
plt.fill_between(np.arange(n_trials * (n_folds - 1)), avg + std, avg - std, alpha=0.2)
avg = accuracy[1, ...].mean(axis=-1)
std = accuracy[1, ...].std(axis=-1)
plt.plot(np.arange(n_trials * (n_folds - 1)), avg, label="eCCA")
plt.fill_between(np.arange(n_trials * (n_folds - 1)), avg + std, avg - std, alpha=0.2)
plt.axhline(1 / n_classes, color="k", linestyle="--", label="chance", alpha=0.5)
plt.xlabel("train trials [#]")
plt.ylabel("accuracy")
plt.legend()
plt.title("Learning curve")
plt.tight_layout()

plt.show()
