"""
Thielen 2015 preprocessing
==========================
This script shows how to read and (minimally) preprocess the Thielen et al. 2015 dataset. This dataset can be
downloaded from [1]_ and was recorded as part of [2]_. This script makes use of the MNE Python library for human
neurophysiological data.

Disclaimer: This notebook does not aim to replicate the original work, instead it provides an example how to read this
dataset.

References
----------
.. [1] Thielen et al. (2015) Broad-Band visually evoked potentials: re(con)volution in brain-computer interfacing.
       DOI: https://doi.org/10.34973/1ecz-1232
.. [2] Thielen, J., Van Den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials:
       re(con)volution in brain-computer interfacing. PloS one, 10(7), e0133797.
       DOI: https://doi.org/10.1371/journal.pone.0133797
"""


import os
import numpy as np
import scipy.io as sio
import mne


# %%
# Set the data path
# -----------------
# The cell below specifies where the dataset has been downloaded to. Please, make sure it is set correctly according to
# the specification of your device. If none of the folder structures in the dataset were changed, the cells below should
# work just as fine.

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2015")  # the path to the dataset

# %%
# Set the preprocessing settings
# ------------------------------
# The preprocessing could be performed for all 12 participants in the dataset, or a subset of these. This can be
# specified with the `subjects` variable. The preprocessing will then go through several steps, including (1) spectral
# band-pass and notch filter to limit the spectral content; (2) resampling to lower the sampling frequency from the raw
# 512 Hz to a lower one that matches (a multiple of) the screen refresh rate (120 Hz); (3) slicing to cut the raw
# continuous data into single-trials. Each of these steps have accompanying parameters that can be adjusted with the
# variables below.

# Subjects to read and preprocess
subjects = [f"sub-{1+i:02d}" for i in range(12)]  # all participants

# Configuration for the preprocessing
fs = 240  # target sampling frequency in Hz for resampling
bandpass = [1.0, 65.0]  # bandpass filter cutoffs in Hz for spectral filtering
notch = 50.0  # notch filter cutoff in Hz for spectral filtering
tmin = 0.0  # trial onset in seconds for slicing
tmax = 4.2  # trial end in seconds for slicing

# %%
# Reading and upsampling the stimulation sequences
# ------------------------------------------------
# The noise-tags used in this dataset were modulated Gold codes with a linear shift register of length 6 and
# feedback-tap positions at [6, 1] and [6, 5, 2, 1] for the train set, and [6, 5] and [6, 5, 3, 2] for the testing set.
# For more details, see the original publication. In addition to reading the codes, the codes are upsampled to the EEG
# sampling frequency. Note, the reading is done in the preprocessing code, as each participant had a uniquely optimized
# subset and layout of testing codes, as defined on the online training data.

# The screen refresh rate
FR = 120

# %%
# Reading and preprocessing the EEG data
# --------------------------------------
# The cell below performs the reading and preprocessing of the EEG data, given the configuration above. The experiment
# consisted of 4 blocks during each of which 36 trials were recorded, one for each of the codes in random order. The
# cell results into (1) the data `X` that is a matrix of k trials c channels, and m samples, (2) the ground-truth labels
# `y` that is a vector of k trials, (3) the codes `V` that is a matrix of n classes and m samples, and (4) the sampling
# frequency `fs`.
#
# Note, the labels in `y` refer to the index in `V` at which to find the target (i.e., attended) code for a particular
# trial.
#
# Also note that there is a distinction between training data and testing data in this dataset, as the noise-codes
# changed from the one phase to the other.
#
# The cell reads the data from the sourcedata folder in the dataset, and saves the processed data for each participant
# in a derivatives folder with a folder structure identical to the sourcedata.

# The experimental blocks
BLOCKS = ["train", "test_sync_1", "test_sync_2", "test_sync_3"]

# Loop over subjects
for subject in subjects:

    epochs = []
    labels = []
    codes = []

    # Loop over blocks
    for block in BLOCKS:
        # Read raw file
        raw = mne.io.read_raw_gdf(os.path.join(path, "sourcedata", subject, block, f"{subject}_{block}.gdf"),
                                  stim_channel="status", preload=True,
                                  exclude=["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8",
                                           "ANA1", "ANA2", "ANA3", "ANA4", "ANA5", "ANA6", "ANA7", "ANA8",
                                           "ANA9", "ANA10", "ANA11", "ANA12", "ANA13", "ANA14", "ANA15",
                                           "ANA16", "ANA17", "ANA18", "ANA19", "ANA20", "ANA21", "ANA22",
                                           "ANA23", "ANA24", "ANA25", "ANA26", "ANA27", "ANA28", "ANA29",
                                           "ANA30", "ANA31", "ANA32"])

        # Read events
        events = mne.find_events(raw)

        # Spectral notch filter
        raw.notch_filter(freqs=np.arange(notch, raw.info['sfreq'] / 2, notch))

        # Spectral band-pass filter
        raw = raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])

        # Slice data to trials
        # N.B. 500 ms pre-trial is added which is removed later. This is added to catch filter artefacts from the
        # subsequent resample. The resample is done after slicing to maintain accurate marker timing.
        epo = mne.Epochs(raw, events=events, tmin=tmin - 0.5, tmax=tmax, baseline=None, picks="eeg", preload=True)

        # Downsample
        epo.resample(sfreq=fs)

        # Add to dataset
        epochs.append(epo)

        # Read codes and labels
        # N.B. Minus 1 to convert Matlab counting from 1 to Python counting from 0
        f = sio.loadmat(os.path.join(path, "sourcedata", subject, block, f"{subject}_{block}.mat"))
        labels.append(np.array(f["labels"]).astype("uint8").flatten() - 1)
        subset = f["subset"].astype("uint8").flatten() - 1
        layout = f["layout"].astype("uint8").flatten() - 1
        codes.append(np.repeat(f["codes"][:, subset[layout]], fs // FR, axis=0).astype("uint8"))

    # Extract data and concatenate runs
    X_train = epochs[0].get_data(tmin=tmin, tmax=tmax).astype("float32")
    X_test = mne.concatenate_epochs(epochs[1:]).get_data(tmin=tmin, tmax=tmax).astype("float32")
    y_train = np.array(labels[0]).flatten().astype("uint8")
    y_test = np.array(labels[1:]).flatten().astype("uint8")
    V = codes[0].T
    U = codes[1].T

    # Create output folder
    if not os.path.exists(os.path.join(path, "derivatives", subject)):
        os.makedirs(os.path.join(path, "derivatives", subject))

    # Save data
    np.savez(os.path.join(path, "derivatives", subject, f"{subject}_gdf.npz"),
             X_train=X_train, y_train=y_train, V=V, X_test=X_test, y_test=y_test, U=U, fs=fs)

    # Print summary
    print("Subject: ", subject)
    print("\tX_train shape:", X_train.shape, "(trials, channels, samples)")
    print("\ty_train shape:", y_train.shape, "(trials)")
    print("\tV shape:", V.shape, "(classes, samples)")
    print("\tX_test shape:", X_test.shape, "(trials, channels, samples)")
    print("\ty_test shape:", y_test.shape, "(trials)")
    print("\tU shape:", V.shape, "(classes, samples)")
    print("\tfs: ", fs)
