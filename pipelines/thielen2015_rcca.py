"""
Thielen 2015 rCCA
=================
This script shows an example of how to classify the Thielen et al. 2015 dataset using rCCA. This dataset can be
downloaded from [1]_, and was recorded as part of [2]_. Note, this notebook does not involve, but does require, the
reading and preprocessing of the raw data. Reading and preprocessing is outlined in the
example_thielen2015_preprocessing.py script.

Disclaimer: This notebook does not aim to replicate the original work, instead it provides an example how to use PyntBCI
for this dataset using rCCA.

References
----------
.. [1] Thielen et al. (2015) Broad-Band visually evoked potentials: re(con)volution in brain-computer interfacing.
       DOI: https://doi.org/10.34973/1ecz-1232
.. [2] Thielen, J., Van Den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials:
       re(con)volution in brain-computer interfacing. PloS one, 10(7), e0133797.
       DOI: https://doi.org/10.1371/journal.pone.0133797
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn

import pyntbci

seaborn.set_context("paper")

# %%
# Set the data path
# -----------------
# The cell below specifies where the dataset has been downloaded to. Please, make sure it is set correctly according to
# the specification of your device. If none of the folder structures in the dataset were changed, the cells below should
# work just as fine.

home = os.path.expanduser("~")
path = os.path.join(home, "data", "thielen2015")  # the path to the dataset
subject = "sub-01"  # the subject to analyse

# %%
# The data
# --------
# The dataset consists of (1) the EEG data X that is a matrix of k trials, c channels, and m samples; (2) the labels y
# that is a vector of k trials; (3) the pseudo-random noise-codes V that is a matrix of n classes and m samples. Note,
# the codes are upsampled to match the EEG sampling frequency and contain only one code-cycle.

# Load data
fn = os.path.join(path, "derivatives", subject, f"{subject}_gdf.npz")
tmp = np.load(fn)
X_trn = tmp["X_train"]
y_trn = tmp["y_train"]
V = tmp["V"]
X_tst = tmp["X_test"]
y_tst = tmp["y_test"]
U = tmp["U"]
fs = tmp["fs"]
fr = 120

# Print data dimensions
print("X_train: shape:", X_trn.shape, ", type:", X_trn.dtype)  # EEG time-series: trials x channels x samples
print("y_train: shape:", y_trn.shape, ", type:", y_trn.dtype)  # Labels: trials
print("V: shape:", V.shape, ", type:", V.dtype)  # Codes: classes x samples
print("X_test: shape:", X_tst.shape, ", type:", X_tst.dtype)  # EEG time-series: trials x channels x samples
print("y_test: shape:", y_tst.shape, ", type:", y_tst.dtype)  # Labels: trials
print("U: shape:", U.shape, ", type:", U.dtype)  # Codes: classes x samples

# Extract data dimensions
n_trials, n_channels, n_samples = X_trn.shape
n_classes = V.shape[0]

# Print sample rate
print("fs:", fs)

# Read cap file
capfile = os.path.join(os.path.dirname(pyntbci.__file__), "capfiles", "biosemi64.loc")
fid = open(capfile, "r")
channels = []
for line in fid.readlines():
    channels.append(line.split("\t")[-1].strip())

# Print channels
print("channels:", channels)

# Visualize EEG data
i_trial = 0
plt.figure(figsize=(15, 15))
plt.plot(np.arange(0, n_samples) / fs, 25e-6 * np.arange(n_channels) + X_trn[i_trial, :, :].T)
plt.yticks(25e-6 * np.arange(n_channels), channels)
plt.xlabel("time [sec]")
plt.ylabel("channel")
plt.title(f"Single-trial multi-channel EEG time-series (trial {i_trial})")

# Visualize labels
plt.figure(figsize=(15, 3))
hist = np.histogram(y_trn, bins=np.arange(n_classes + 1))[0]
plt.bar(np.arange(n_classes), hist)
plt.xticks(np.arange(n_classes))
plt.xlabel("label")
plt.ylabel("count")
plt.title("Single-trial labels")

# Visualize stimuli
Vup = V.repeat(20, axis=1)  # upsample to better visualize the sharp edges
plt.figure(figsize=(15, 8))
plt.plot(np.arange(Vup.shape[1]) / (20 * fs), 2 * np.arange(n_classes) + Vup.T)
for i in range(1 + int(V.shape[1] / (fs / fr))):
    plt.axvline(i / fr, c="k", alpha=0.1)
plt.yticks(2 * np.arange(n_classes), np.arange(n_classes))
plt.xlabel("time [sec]")
plt.ylabel("code")
plt.title("Code time-series")

# %%
# Hold-out classification
# -----------------------
# Given that we can optimize a spatial filter and a response vector, we can define the classification criterion, which
# is a template matching classifier. Specifically, for a new single-trial that we want to classify, we first spatially
# filter it, and then compare it to the predicted responses which we use as templates. The comparison is done by
# computing the correlation of the spatially filtered single-trial with all templates, and selecting the one with the
# highest correlation, as that is the one that the single-trial is most similar to.
#
# Note, the dataset contains single-trials of 4.2 seconds long. For many participants in the dataset, if all data is
# used, this leads to 100% accuracy. A new parameter is introduced here, that cuts the single-trials to shorter lengths.
# Ideally, this parameter is explored, to estimate a so-called decoding curve.
#
# Please note that in this dataset, different stimulus sets were used during the training (V) and testing (U) phase.
# Therefore, after fitting the classifier on the training data, we need to set the stimulus set to the testing stimuli.

# Set trial duration
n_samples = int(4.2 * fs)

# Train template-matching classifier
rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=0.2, onset_event=True)
rcca.fit(X_trn, y_trn)

# Change stimuli to test set
rcca.set_stimulus(U)

# Apply template-matching classifier
yh_tst = rcca.predict(X_tst)

# Compute accuracy
accuracy = np.mean(yh_tst == y_tst)

# Print accuracy
print("Accuracy: {:.1f}".format(accuracy))

# %%
# Analyse all participants in the dataset
# ---------------------------------------

# Set paths
home = os.path.expanduser("~")
path = os.path.join(home, "data", "thielen2015")  # the path to the dataset
n_subjects = 12
subjects = [f"sub-{1 + i:02d}" for i in range(n_subjects)]

# Set trial duration
trial_time = 4.2  # limit trials to a certain duration in seconds
n_trials = 108  # limit the number of trials in the dataset

# Chronological cross-validation
n_folds = 6
folds = np.repeat(np.arange(n_folds), int(n_trials / n_folds))

# Loop participants
accuracy = np.zeros((n_subjects, n_folds))
for i_subject in range(n_subjects):
    subject = subjects[i_subject]

    # Load data
    fn = os.path.join(path, "derivatives", subject, f"{subject}_gdf.npz")
    tmp = np.load(fn)
    fs = tmp["fs"]
    X_tst = tmp["X_test"][:n_trials, :, :int(trial_time * fs)]
    y_tst = tmp["y_test"][:n_trials]
    U = tmp["U"]

    # Cross-validation
    for i_fold in range(n_folds):
        # Split data to train and test set
        X_trn_, y_trn_ = X_tst[folds != i_fold, :, :], y_tst[folds != i_fold]
        X_tst_, y_tst_ = X_tst[folds == i_fold, :, :], y_tst[folds == i_fold]

        # Train classifier
        rcca = pyntbci.classifiers.rCCA(stimulus=U, fs=fs, event="duration", encoding_length=0.2, onset_event=True)
        rcca.fit(X_trn_, y_trn_)

        # Apply classifier
        yh_tst_ = rcca.predict(X_tst_)

        # Compute accuracy
        accuracy[i_subject, i_fold] = np.mean(yh_tst_ == y_tst_)

# Add average to accuracies
subjects += ["avg"]
avg = np.mean(accuracy, axis=0, keepdims=True)
accuracy = np.concatenate((accuracy, avg), axis=0)

# Plot accuracy
plt.figure(figsize=(15, 5))
avg = accuracy.mean(axis=1)
std = accuracy.std(axis=1)
plt.bar(np.arange(1 + n_subjects) + 0.3, avg, 0.5, yerr=std, label="rCCA")
plt.axhline(accuracy.mean(), linestyle="--", alpha=0.5, label="average")
plt.axhline(1 / n_classes, linestyle="--", color="k", alpha=0.5, label="chance")
plt.table(cellText=[np.round(avg, 2), np.round(std, 2)], loc='bottom', rowLabels=["avg", "std"], colLabels=subjects,
          cellLoc="center")
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.xticks([])
plt.ylabel("accuracy")
plt.xlim([-0.25, n_subjects + 0.75])
plt.legend()
plt.title("Decoding performance full dataset")
plt.tight_layout()

# Print accuracy
print(f"Average accuracy: {avg.mean():.2f}")

plt.show()
