"""
Thielen 2021 dynamic stopping
=============================
This script shows an example of how to use dynamic stopping using the Thielen et al. 2021 dataset. This dataset can be
downloaded from [1]_, and was recorded as part of [2]_. Note, this notebook does not involve, but does require, the
reading and preprocessing of the raw data. Reading and preprocessing is outlined in the
example_thielen2021_preprocessing.py script.

Disclaimer: This notebook does not aim to replicate the original work, instead it provides an example how to use PyntBCI
for this dataset using dynamic stopping.

References
----------
.. [1] Thielen et al. (2021) From full calibration to zero training for a code-modulated visual evoked potentials brain
       computer interface. DOI: https://doi.org/10.34973/9txv-z787
.. [2] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
       code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
       056007. DOI: https://doi.org/10.1088/1741-2552/abecef
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import pyntbci
import seaborn


seaborn.set_context("paper", font_scale=1.5)


# %%
# Set the data path
# -----------------
# The cell below specifies where the dataset has been downloaded to. Please, make sure it is set correctly according to
# the specification of your device. If none of the folder structures in the dataset were changed, the cells below should
# work just as fine.

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset
subject = "sub-01"  # the subject to analyse

# %%
# The data
# --------
# The dataset consists of (1) the EEG data X that is a matrix of k trials, c channels, and m samples; (2) the labels y
# that is a vector of k trials; (3) the pseudo-random noise-codes V that is a matrix of n classes and m samples. Note,
# the codes are upsampled to match the EEG sampling frequency and contain only one code-cycle.

# Load data
fn = os.path.join(path, "derivatives", "offline", subject, f"{subject}_gdf.npz")
tmp = np.load(fn)
X = tmp["X"]
y = tmp["y"]
V = tmp["V"]
fs = tmp["fs"]
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
capfile = os.path.join(path, "resources", "nt_cap8.loc")
with open(capfile, "r") as fid:
    channels = []
    for line in fid.readlines():
        channels.append(line.split("\t")[-1].strip())
print("Channels:", ", ".join(channels))

# Visualize EEG data
i_trial = 0  # the trial to visualize
plt.figure(figsize=(15, 5))
plt.plot(np.arange(0, n_samples) / fs, 25e-6 * np.arange(n_channels) + X[i_trial, :, :].T)
plt.xlim([0, 1])  # limit to 1 second EEG data
plt.yticks(25e-6 * np.arange(n_channels), channels)
plt.xlabel("time [s]")
plt.ylabel("channel")
plt.title(f"Single-trial multi-channel EEG time-series (trial {i_trial})")
plt.tight_layout()

# Visualize labels
plt.figure(figsize=(15, 3))
hist = np.histogram(y, bins=np.arange(n_classes+1))[0]
plt.bar(np.arange(n_classes), hist)
plt.xticks(np.arange(n_classes))
plt.xlabel("label")
plt.ylabel("count")
plt.title("Single-trial labels")
plt.tight_layout()

# Visualize stimuli
Vup = V.repeat(20, axis=1)  # upsample to better visualize the sharp edges
plt.figure(figsize=(15, 8))
plt.plot(np.arange(Vup.shape[1]) / (20 * fs), 2 * np.arange(n_classes) + Vup.T)
for i in range(1 + int(V.shape[1] / (fs / fr))):
    plt.axvline(i / fr, c="k", alpha=0.1)
plt.yticks(2 * np.arange(n_classes), np.arange(n_classes))
plt.xlabel("time [s]")
plt.ylabel("code")
plt.title("Code time-series")
plt.tight_layout()

# ##
# Settings
# --------
# Some general settings for the following sections

# Set trial duration
trial_time = 4.2  # limit trials to a certain duration in seconds
inter_trial_time = 1.0  # ITI in seconds for computing ITR
n_samples = int(trial_time * fs)

# Setup rCCA
encoding_length = 0.3
onset_event = True

# Set stopping
segment_time = 0.1
n_segments = int(trial_time / segment_time)

# Set chronological cross-validation
n_folds = 5
folds = np.repeat(np.arange(n_folds), int(n_trials / n_folds))

# %%
# Margin dynamic stopping
# -----------------------
# The margin method learns threshold margins (i.e., the difference between the best and second best score) to stop.
# These margins are defined as such that a targeted accuracy is reached.
#
# References:
# .. [3] Thielen, J., van den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials:
#        re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797.
#        doi: https://doi.org/10.1371/journal.pone.0133797

# Target accuracy
target_p = 0.95 ** (1 / n_segments)

# Fit classifier
rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=encoding_length,
                                onset_event=onset_event, score_metric="correlation")
margin = pyntbci.stopping.MarginStopping(rcca, segment_time, fs, target_p=target_p, max_time=trial_time)
margin.fit(X, y)

# Plot dynamic stopping
plt.figure(figsize=(15, 3))
plt.plot(np.arange(1, 1 + margin.margins_.size) * segment_time, margin.margins_, c="k")
plt.xlabel("time [s]")
plt.ylabel("margin")
plt.title("Margin dynamic stopping")

# Loop folds
accuracy_margin = np.zeros(n_folds)
duration_margin = np.zeros(n_folds)
for i_fold in range(n_folds):

    # Split data to train and valid set
    X_trn, y_trn = X[folds != i_fold, :, :n_samples], y[folds != i_fold]
    X_tst, y_tst = X[folds == i_fold, :, :n_samples], y[folds == i_fold]

    # Train template-matching classifier
    rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=encoding_length,
                                    onset_event=onset_event, score_metric="correlation")
    margin = pyntbci.stopping.MarginStopping(rcca, segment_time, fs, target_p=target_p)
    margin.fit(X_trn, y_trn)

    # Loop segments
    yh_tst = np.zeros(X_tst.shape[0])
    dur_tst = np.zeros(X_tst.shape[0])
    for i_segment in range(n_segments):

        # Apply template-matching classifier
        tmp = margin.predict(X_tst[:, :, :int((1 + i_segment) * segment_time * fs)])

        # Check stopped
        idx = np.logical_and(tmp >= 0, dur_tst == 0)
        yh_tst[idx] = tmp[idx]
        dur_tst[idx] = (1 + i_segment) * segment_time
        if np.all(dur_tst > 0):
            break

    # Compute accuracy
    accuracy_margin[i_fold] = np.mean(yh_tst == y_tst)
    duration_margin[i_fold] = np.mean(dur_tst)

# Compute ITR
itr_margin = pyntbci.utilities.itr(n_classes, accuracy_margin, duration_margin + inter_trial_time)

# Plot accuracy (over folds)
fig, ax = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
ax[0].bar(np.arange(n_folds), accuracy_margin)
ax[0].hlines(np.mean(accuracy_margin), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[1].bar(np.arange(n_folds), duration_margin)
ax[1].hlines(np.mean(duration_margin), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[2].bar(np.arange(n_folds), itr_margin)
ax[2].hlines(np.mean(itr_margin), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[2].set_xlabel("(test) fold")
ax[0].set_ylabel("accuracy")
ax[1].set_ylabel("duration [s]")
ax[2].set_ylabel("itr [bits/min]")
ax[0].set_title(f"Margin dynamic stopping: avg acc {accuracy_margin.mean():.2f} | " +
                f"avg dur {duration_margin.mean():.2f} | avg itr {itr_margin.mean():.1f}")

# Print accuracy (average and standard deviation over folds)
print("Margin:")
print(f"\tAccuracy: avg={accuracy_margin.mean():.2f} with std={accuracy_margin.std():.2f}")
print(f"\tDuration: avg={duration_margin.mean():.2f} with std={duration_margin.std():.2f}")
print(f"\tITR: avg={itr_margin.mean():.1f} with std={itr_margin.std():.2f}")

# %%
# Beta dynamic stopping
# ---------------------
# The beta method fits a beta distribution to the non-maximum scores (i.e., if correlation, then correlation+1)/2), and
# tests the probability of the maximum correlation to belong to that beta distribution.
#
# References:
# .. [4] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
#        code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
#        056007. doi: http://doi.org/10.1088/1741-2552/abecef

# Target accuracy
target_p = 0.95 ** (1 / n_segments)

# Loop folds
accuracy_beta = np.zeros(n_folds)
duration_beta = np.zeros(n_folds)
for i_fold in range(n_folds):

    # Split data to train and valid set
    X_trn, y_trn = X[folds != i_fold, :, :n_samples], y[folds != i_fold]
    X_tst, y_tst = X[folds == i_fold, :, :n_samples], y[folds == i_fold]

    # Train template-matching classifier
    rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=encoding_length,
                                    onset_event=onset_event, score_metric="correlation")
    beta = pyntbci.stopping.BetaStopping(rcca, target_p=target_p, fs=fs, max_time=trial_time)
    beta.fit(X, y)

    # Loop segments
    yh_tst = np.zeros(X_tst.shape[0])
    dur_tst = np.zeros(X_tst.shape[0])
    for i_segment in range(n_segments):

        # Apply template-matching classifier
        tmp = beta.predict(X_tst[:, :, :int((1 + i_segment) * segment_time * fs)])

        # Check stopped
        idx = np.logical_and(tmp >= 0, dur_tst == 0)
        yh_tst[idx] = tmp[idx]
        dur_tst[idx] = (1 + i_segment) * segment_time
        if np.all(dur_tst > 0):
            break

    # Compute accuracy
    accuracy_beta[i_fold] = np.mean(yh_tst == y_tst)
    duration_beta[i_fold] = np.mean(dur_tst)

# Compute ITR
itr_beta = pyntbci.utilities.itr(n_classes, accuracy_beta, duration_beta + inter_trial_time)

# Plot accuracy (over folds)
fig, ax = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
ax[0].bar(np.arange(n_folds), accuracy_beta)
ax[0].hlines(np.mean(accuracy_beta), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[1].bar(np.arange(n_folds), duration_beta)
ax[1].hlines(np.mean(duration_beta), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[2].bar(np.arange(n_folds), itr_beta)
ax[2].hlines(np.mean(itr_beta), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[2].set_xlabel("(test) fold")
ax[0].set_ylabel("accuracy")
ax[1].set_ylabel("duration [s]")
ax[2].set_ylabel("itr [bits/min]")
ax[0].set_title(f"Beta dynamic stopping: avg acc {accuracy_beta.mean():.2f} | " +
                f"avg dur {duration_beta.mean():.2f} | avg itr {itr_beta.mean():.1f}")

# Print accuracy (average and standard deviation over folds)
print("Beta:")
print(f"\tAccuracy: avg={accuracy_beta.mean():.2f} with std={accuracy_beta.std():.2f}")
print(f"\tDuration: avg={duration_beta.mean():.2f} with std={duration_beta.std():.2f}")
print(f"\tITR: avg={itr_beta.mean():.1f} with std={itr_beta.std():.2f}")

# %%
# Bayesian dynamic stopping (BDS0)
# --------------------------------
# The Bayesian method fits Gaussian distributions for target and non-target responses, and calculates a stopping
# threshold using these and a cost criterion. This method comes in three flavours: bds0, bds1, and bds2.
#
# References:
# .. [5] Ahmadi, S., Thielen, J., Farquhar, J., & Desain, P. (in prep.) A model driven Bayesian dynamic stopping method
#        for parallel stimulation evoked response BCIs.

# Cost ratio and target probabilities
cr = 1.0

# Fit classifier
rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=encoding_length,
                                onset_event=onset_event, score_metric="inner")
bayes = pyntbci.stopping.BayesStopping(rcca, segment_time, fs, cr=cr, max_time=trial_time)
bayes.fit(X, y)

# Plot dynamic stopping
fig, ax = plt.subplots(2, 1, figsize=(15, 4), sharex=True)
ax[0].plot(np.arange(1, 1 + bayes.eta_.size) * segment_time, bayes.eta_, c="k", label="eta")
ax[0].plot(np.arange(1, 1 + bayes.eta_.size) * segment_time, bayes.alpha_ * bayes.b0_, "--b", label="b0")
ax[0].plot(np.arange(1, 1 + bayes.eta_.size) * segment_time, bayes.alpha_ * bayes.b1_, "--g", label="b1")
ax[0].plot(np.arange(1, 1 + bayes.eta_.size) * segment_time, bayes.alpha_ * bayes.b0_ - bayes.s0_, "b")
ax[0].plot(np.arange(1, 1 + bayes.eta_.size) * segment_time, bayes.alpha_ * bayes.b1_ - bayes.s1_, "g")
ax[0].plot(np.arange(1, 1 + bayes.eta_.size) * segment_time, bayes.alpha_ * bayes.b0_ + bayes.s0_, "b")
ax[0].plot(np.arange(1, 1 + bayes.eta_.size) * segment_time, bayes.alpha_ * bayes.b1_ + bayes.s1_, "g")
ax[0].legend()
ax[1].plot(np.arange(1, 1 + bayes.eta_.size) * segment_time, bayes.pf_, label="pf")
ax[1].plot(np.arange(1, 1 + bayes.eta_.size) * segment_time, bayes.pm_, label="pm")
ax[1].legend()
ax[1].set_xlabel("time [s]")
ax[0].set_title("Bayesian dynamic stopping")

# Loop folds
accuracy_bds0 = np.zeros(n_folds)
duration_bds0 = np.zeros(n_folds)
for i_fold in range(n_folds):

    # Split data to train and valid set
    X_trn, y_trn = X[folds != i_fold, :, :n_samples], y[folds != i_fold]
    X_tst, y_tst = X[folds == i_fold, :, :n_samples], y[folds == i_fold]

    # Train template-matching classifier
    rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=encoding_length,
                                    onset_event=onset_event, score_metric="inner")
    bayes = pyntbci.stopping.BayesStopping(rcca, segment_time, fs, method="bds0", cr=cr, max_time=trial_time)
    bayes.fit(X_trn, y_trn)

    # Apply template-matching classifier
    yh_tst = np.zeros(X_tst.shape[0])
    dur_tst = np.zeros(X_tst.shape[0])
    for i_segment in range(n_segments):
        tmp = bayes.predict(X_tst[:, :, :int((1 + i_segment) * segment_time * fs)])
        idx = np.logical_and(tmp >= 0, dur_tst == 0)
        yh_tst[idx] = tmp[idx]
        dur_tst[idx] = (1 + i_segment) * segment_time
        if np.all(dur_tst > 0):
            break

    # Compute accuracy
    accuracy_bds0[i_fold] = np.mean(yh_tst == y_tst)
    duration_bds0[i_fold] = np.mean(dur_tst)

# Compute ITR
itr_bds0 = pyntbci.utilities.itr(n_classes, accuracy_bds0, duration_bds0 + inter_trial_time)

# Plot accuracy (over folds)
fig, ax = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
ax[0].bar(np.arange(n_folds), accuracy_bds0)
ax[0].hlines(np.mean(accuracy_bds0), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[1].bar(np.arange(n_folds), duration_bds0)
ax[1].hlines(np.mean(duration_bds0), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[2].bar(np.arange(n_folds), itr_bds0)
ax[2].hlines(np.mean(itr_bds0), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[2].set_xlabel("(test) fold")
ax[0].set_ylabel("accuracy")
ax[1].set_ylabel("duration [s]")
ax[2].set_ylabel("itr [bits/min]")
ax[0].set_title(f"BDS0 dynamic stopping: avg acc {accuracy_bds0.mean():.2f} | " +
                f"avg dur {duration_bds0.mean():.2f} | avg itr {itr_bds0.mean():.1f}")

# Print accuracy (average and standard deviation over folds)
print("BDS0:")
print(f"\tAccuracy: avg={accuracy_bds0.mean():.2f} with std={accuracy_bds0.std():.2f}")
print(f"\tDuration: avg={duration_bds0.mean():.2f} with std={duration_bds0.std():.2f}")
print(f"\tITR: avg={itr_bds0.mean():.1f} with std={itr_bds0.std():.2f}")

# %%
# Bayesian dynamic stopping (BDS1)
# --------------------------------
# The Bayesian method fits Gaussian distributions for target and non-target responses, and calculates a stopping
# threshold using these and a cost criterion. This method comes in three flavours: bds0, bds1, and bds2.
#
# References:
# .. [6] Ahmadi, S., Thielen, J., Farquhar, J., & Desain, P. (in prep.) A model driven Bayesian dynamic stopping method
#        for parallel stimulation evoked response BCIs.

# Cost ratio and target probabilities
cr = 1.0
target_pf = 0.05
target_pd = 0.80

# Loop folds
accuracy_bds1 = np.zeros(n_folds)
duration_bds1 = np.zeros(n_folds)
for i_fold in range(n_folds):

    # Split data to train and valid set
    X_trn, y_trn = X[folds != i_fold, :, :n_samples], y[folds != i_fold]
    X_tst, y_tst = X[folds == i_fold, :, :n_samples], y[folds == i_fold]

    # Train template-matching classifier
    rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=encoding_length,
                                    onset_event=onset_event, score_metric="inner")
    bayes = pyntbci.stopping.BayesStopping(rcca, segment_time, fs, method="bds1", cr=cr, target_pf=target_pf,
                                           target_pd=target_pd, max_time=trial_time)
    bayes.fit(X_trn, y_trn)

    # Apply template-matching classifier
    yh_tst = np.zeros(X_tst.shape[0])
    dur_tst = np.zeros(X_tst.shape[0])
    for i_segment in range(n_segments):
        tmp = bayes.predict(X_tst[:, :, :int((1 + i_segment) * segment_time * fs)])
        idx = np.logical_and(tmp >= 0, dur_tst == 0)
        yh_tst[idx] = tmp[idx]
        dur_tst[idx] = (1 + i_segment) * segment_time
        if np.all(dur_tst > 0):
            break

    # Compute accuracy
    accuracy_bds1[i_fold] = np.mean(yh_tst == y_tst)
    duration_bds1[i_fold] = np.mean(dur_tst)

# Compute ITR
itr_bds1 = pyntbci.utilities.itr(n_classes, accuracy_bds1, duration_bds1 + inter_trial_time)

# Plot accuracy (over folds)
fig, ax = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
ax[0].bar(np.arange(n_folds), accuracy_bds1)
ax[0].hlines(np.mean(accuracy_bds1), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[1].bar(np.arange(n_folds), duration_bds1)
ax[1].hlines(np.mean(duration_bds1), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[2].bar(np.arange(n_folds), itr_bds1)
ax[2].hlines(np.mean(itr_bds1), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[2].set_xlabel("(test) fold")
ax[0].set_ylabel("accuracy")
ax[1].set_ylabel("duration [s]")
ax[2].set_ylabel("itr [bits/min]")
ax[0].set_title(f"BDS1 dynamic stopping: avg acc {accuracy_bds1.mean():.2f} | " +
                f"avg dur {duration_bds1.mean():.2f} | avg itr {itr_bds1.mean():.1f}")

# Print accuracy (average and standard deviation over folds)
print("BDS1:")
print(f"\tAccuracy: avg={accuracy_bds1.mean():.2f} with std={accuracy_bds1.std():.2f}")
print(f"\tDuration: avg={duration_bds1.mean():.2f} with std={duration_bds1.std():.2f}")
print(f"\tITR: avg={itr_bds1.mean():.1f} with std={itr_bds1.std():.2f}")

# %%
# Bayesian dynamic stopping (BDS2)
# --------------------------------
# The Bayesian method fits Gaussian distributions for target and non-target responses, and calculates a stopping
# threshold using these and a cost criterion. This method comes in three flavours: bds0, bds1, and bds2.
#
# References:
# .. [7] Ahmadi, S., Thielen, J., Farquhar, J., & Desain, P. (in prep.) A model driven Bayesian dynamic stopping method
#        for parallel stimulation evoked response BCIs.

# Cost ratio and target probabilities
cr = 1.0
target_pf = 0.05
target_pd = 0.80

# Loop folds
accuracy_bds2 = np.zeros(n_folds)
duration_bds2 = np.zeros(n_folds)
for i_fold in range(n_folds):

    # Split data to train and valid set
    X_trn, y_trn = X[folds != i_fold, :, :n_samples], y[folds != i_fold]
    X_tst, y_tst = X[folds == i_fold, :, :n_samples], y[folds == i_fold]

    # Train template-matching classifier
    rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=encoding_length,
                                    onset_event=onset_event, score_metric="inner")
    bayes = pyntbci.stopping.BayesStopping(rcca, segment_time, fs, method="bds2", cr=cr, target_pf=target_pf,
                                           target_pd=target_pd, max_time=trial_time)
    bayes.fit(X_trn, y_trn)

    # Apply template-matching classifier
    yh_tst = np.zeros(X_tst.shape[0])
    dur_tst = np.zeros(X_tst.shape[0])
    for i_segment in range(n_segments):
        tmp = bayes.predict(X_tst[:, :, :int((1 + i_segment) * segment_time * fs)])
        idx = np.logical_and(tmp >= 0, dur_tst == 0)
        yh_tst[idx] = tmp[idx]
        dur_tst[idx] = (1 + i_segment) * segment_time
        if np.all(dur_tst > 0):
            break

    # Compute accuracy
    accuracy_bds2[i_fold] = np.mean(yh_tst == y_tst)
    duration_bds2[i_fold] = np.mean(dur_tst)

# Compute ITR
itr_bds2 = pyntbci.utilities.itr(n_classes, accuracy_bds2, duration_bds2 + inter_trial_time)

# Plot accuracy (over folds)
fig, ax = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
ax[0].bar(np.arange(n_folds), accuracy_bds2)
ax[0].hlines(np.mean(accuracy_bds2), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[1].bar(np.arange(n_folds), duration_bds2)
ax[1].hlines(np.mean(duration_bds2), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[2].bar(np.arange(n_folds), itr_bds2)
ax[2].hlines(np.mean(itr_bds2), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
ax[2].set_xlabel("(test) fold")
ax[0].set_ylabel("accuracy")
ax[1].set_ylabel("duration [s]")
ax[2].set_ylabel("itr [bits/min]")
ax[0].set_title(f"BDS2 dynamic stopping: avg acc {accuracy_bds2.mean():.2f} | " +
                f"avg dur {duration_bds2.mean():.2f} | avg itr {itr_bds2.mean():.1f}")

# Print accuracy (average and standard deviation over folds)
print("BDS2:")
print(f"\tAccuracy: avg={accuracy_bds2.mean():.2f} with std={accuracy_bds2.std():.2f}")
print(f"\tDuration: avg={duration_bds2.mean():.2f} with std={duration_bds2.std():.2f}")
print(f"\tITR: avg={itr_bds2.mean():.1f} with std={itr_bds2.std():.2f}")

# %%
# Overall comparison
# ------------------
# Comparison of the presented stopping methods. Note, each of these use default parameters that might need fine-tuning.
# Additionally, the evaluation is performed on a single participant only.

# Plot accuracy
width = 0.8
fig, ax = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
ax[0].bar(0, accuracy_margin.mean(), width=width, yerr=accuracy_margin.std(), label="margin")
ax[0].bar(1, accuracy_beta.mean(), width=width, yerr=accuracy_beta.std(), label="beta")
ax[0].bar(2, accuracy_bds0.mean(), width=width, yerr=accuracy_bds0.std(), label="bds0")
ax[0].bar(3, accuracy_bds1.mean(), width=width, yerr=accuracy_bds1.std(), label="bds1")
ax[0].bar(4, accuracy_bds2.mean(), width=width, yerr=accuracy_bds2.std(), label="bds2")
ax[1].bar(0, duration_margin.mean(), width=width, yerr=duration_margin.std(), label="margin")
ax[1].bar(1, duration_beta.mean(), width=width, yerr=duration_beta.std(), label="beta")
ax[1].bar(2, duration_bds0.mean(), width=width, yerr=duration_bds0.std(), label="bds0")
ax[1].bar(3, duration_bds1.mean(), width=width, yerr=duration_bds1.std(), label="bds1")
ax[1].bar(4, duration_bds2.mean(), width=width, yerr=duration_bds2.std(), label="bds2")
ax[2].bar(0, itr_margin.mean(), width=width, yerr=itr_margin.std(), label="margin")
ax[2].bar(1, itr_beta.mean(), width=width, yerr=itr_beta.std(), label="beta")
ax[2].bar(2, itr_bds0.mean(), width=width, yerr=itr_bds0.std(), label="bds0")
ax[2].bar(3, itr_bds1.mean(), width=width, yerr=itr_bds1.std(), label="bds1")
ax[2].bar(4, itr_bds2.mean(), width=width, yerr=itr_bds2.std(), label="bds2")
ax[2].set_xticks(np.arange(5), ["margin", "beta", "bds0", "bds1", "bds2"])
ax[2].set_xlabel("dynamic stopping method")
ax[0].set_ylabel("accuracy")
ax[1].set_ylabel("duration [s]")
ax[2].set_ylabel("itr [bits/min]")
ax[1].legend(bbox_to_anchor=(1.0, 1.0))
ax[0].set_title("Comparison of dynamic stopping methods averaged across folds")

# plt.show()
