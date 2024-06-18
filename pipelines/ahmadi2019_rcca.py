"""
Ahmadi 2019 rCCA
=================
This script shows an example of how to classify the Ahmadi et al. 2019 dataset using rCCA. This dataset can be
downloaded from [1]_, and was recorded as part of [2]_. Note, this dataset is already preprocessed.

Disclaimer: This notebook does not aim to replicate the original work, instead it provides an example how to use PyntBCI
for this dataset using rCCA.

References
----------
.. [1] Ahmadi et al. (2019). Sensor tying. DOI: https://doi.org/10.34973/ehq6-b836
.. [2] Ahmadi, S., Borhanazad, M., Tump, D., Farquhar, J., & Desain, P. (2019). Low channel count montages using sensor
       tying for VEP-based BCI. Journal of Neural Engineering, 16(6), 066038.
       DOI: https://doi.org/10.1088/1741-2552/ab4057
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.io import loadmat
from scipy.signal import resample

import pyntbci

seaborn.set_context("paper")

# %%
# Set the data path
# -----------------
# The cell below specifies where the dataset has been downloaded to. Please, make sure it is set correctly according to
# the specification of your device. If none of the folder structures in the dataset were changed, the cells below should
# work just as fine.

home = os.path.expanduser("~")
path = os.path.join(home, "data", "ahmadi2019")  # the path to the dataset
subject = "S1"  # the subject to analyse
session = "32"  # 8 or 32 (channels) (8 not supported, capfile missing)

# %%
# The data
# --------
# The dataset consists of (1) the EEG data X that is a matrix of k trials, c channels, and m samples; (2) the labels y
# that is a vector of k trials; (3) the pseudo-random noise-codes V that is a matrix of n classes and m samples. Note,
# the codes are upsampled to match the EEG sampling frequency and contain only one code-cycle.

# Load data
fn = os.path.join(path, "sourcedata", f"{subject}-{session}.mat")
tmp = loadmat(fn)
X = tmp["data"]["X"][0, 0].transpose((2, 0, 1))
y = tmp["data"]["y"][0, 0]
V = tmp["data"]["V"][0, 0].T

# Process data
fs_original = 360
fs = 120
fr = 60
X = resample(X, int(X.shape[2] / (fs_original / fs)), axis=2)
y = y.flatten() - 1
subset = np.unique(y)
for i in range(y.size):
    y[i] = np.where(subset == y[i])[0][0]
V = V[subset, ::3]

# Print data dimensions
print("X: shape:", X.shape, ", type:", X.dtype)  # EEG time-series: trials x channels x samples
print("y: shape:", y.shape, ", type:", y.dtype)  # Labels: trials
print("V: shape:", V.shape, ", type:", V.dtype)  # Codes: classes x samples

# Extract data dimensions
n_trials, n_channels, n_samples = X.shape
n_classes = V.shape[0]

# Print sample rate
print("fs:", fs)

# Read cap file
capfile = os.path.join(os.path.dirname(pyntbci.__file__), "capfiles", "biosemi32.loc")
fid = open(capfile, "r")
channels = []
for line in fid.readlines():
    channels.append(line.split("\t")[-1].strip())

# Print channels
print("channels:", channels)

# Visualize EEG data
i_trial = 0
plt.figure(figsize=(15, 25))
plt.plot(np.arange(0, n_samples) / fs, 60 * np.arange(n_channels) + X[i_trial, :, :].T)
plt.yticks(60 * np.arange(n_channels), channels)
plt.xlabel("time [s]")
plt.ylabel("channel")
plt.title(f"Single-trial multi-channel EEG time-series (trial {i_trial})")

# Visualize labels
plt.figure(figsize=(15, 3))
hist = np.histogram(y, bins=np.arange(n_classes + 1))[0]
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
plt.xlabel("time [s]")
plt.ylabel("code")
plt.title("Code time-series")

# %%
# Analyse all participants in the dataset
# ---------------------------------------

# Set paths
home = os.path.expanduser("~")
path = os.path.join(home, "data", "ahmadi2019")  # the path to the dataset
n_subjects = 10
subjects = [f"S{1 + i}" for i in range(n_subjects)]
session = "8"  # 8 or 32 (channels)

# Set trial duration
trialtime = 4.2  # limit trials to a certain duration in seconds
n_trials = 60  # limit the number of trials in the dataset

# Chronological cross-validation
n_folds = 5

# Loop participants
accuracy = np.zeros((n_subjects, n_folds))
for i_subject in range(n_subjects):
    subject = subjects[i_subject]

    # Load data
    fn = os.path.join(path, "sourcedata", f"{subject}-{session}.mat")
    tmp = loadmat(fn)
    X = tmp["data"]["X"][0, 0].transpose((2, 0, 1))
    y = tmp["data"]["y"][0, 0]
    V = tmp["data"]["V"][0, 0].T

    # Chronological cross-validation
    # Note, number of trials changes over participants
    n_trials = y.size
    folds = np.repeat(np.arange(n_folds), int(n_trials / n_folds))

    # Process data
    fs_original = 360
    fs = 120
    X = resample(X, int(X.shape[2] / (fs_original / fs)), axis=2)
    y = y.flatten() - 1
    subset = np.unique(y)
    for i in range(y.size):
        y[i] = np.where(subset == y[i])[0][0]
    V = V[subset, ::3]

    X = X[:n_trials, :, :int(trialtime * fs)]

    # Cross-validation
    for i_fold in range(n_folds):
        # Split data to train and test set
        X_trn, y_trn = X[folds != i_fold, :, :], y[folds != i_fold]
        X_tst, y_tst = X[folds == i_fold, :, :], y[folds == i_fold]

        # Train classifier
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=0.3, onset_event=True)
        rcca.fit(X_trn, y_trn)

        # Apply classifier
        yh_tst = rcca.predict(X_tst)[:, 0]  # select component

        # Compute accuracy
        accuracy[i_subject, i_fold] = np.mean(yh_tst == y_tst)

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

# plt.show()
