"""
Thielen 2021 preprocessing
==========================
This script reads and preprocess the Thielen et al. 2021 dataset for the purpose of the tutorial data of PyntBCI. This
dataset can be downloaded from [1]_ and was recorded as part of [2]_. This script makes use of the MNE Python library
for human neurophysiological data.

Disclaimer: This notebook does not aim to replicate the original work, instead it provides an example how to read this
dataset.

References
----------
.. [1] Thielen et al. (2021) From full calibration to zero training for a code-modulated visual evoked potentials brain
       computer interface. DOI: https://doi.org/10.34973/9txv-z787
.. [2] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
       code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
       056007. DOI: https://doi.org/10.1088/1741-2552/abecef
"""

import glob
import os

import h5py
import mne
import numpy as np
import scipy.io as sio

import pyntbci

# %%
# Set the data path
# -----------------
# The cell below specifies where the dataset has been downloaded to. Please, make sure it is set correctly according to
# the specification of your device. If none of the folder structures in the dataset were changed, the cells below should
# work just as fine.

data_path = os.path.join(os.path.expanduser("~"), "data", "thielen2021")  # the path to the dataset
out_path = os.path.join(os.path.dirname(pyntbci.__file__), "data")

# %%
# Set the preprocessing settings
# ------------------------------
# The preprocessing could be performed for all 30 participants in the dataset, or a subset of these. This can be
# specified with the `subjects` variable. The preprocessing will then go through several steps, including (1) spectral
# band-pass and notch filter to limit the spectral content; (2) resampling to lower the sampling frequency from the raw
# 512 Hz to a lower one that matches (a multiple of) the screen refresh rate (60 Hz); (3) slicing to cut the raw
# continuous data into single-trials. Each of these steps have accompanying parameters that can be adjusted with the
# variables below.

# Subjects to read and preprocess
subjects = [f"sub-{1 + i:02d}" for i in range(5)]  # all participants

# Configuration for the preprocessing
fs = 240  # target sampling frequency in Hz for resampling
bandpass = [1.0, 65.0]  # bandpass filter cutoffs in Hz for spectral filtering
notch = 50.0  # notch filter cutoff in Hz for spectral filtering
tmin = 0  # trial onset in seconds for slicing
tmax = 10.5  # trial end in seconds for slicing

# %%
# Reading and upsampling the stimulation sequences
# ------------------------------------------------
# The noise-tags used in this dataset were modulated Gold codes with a linear shift register of length 6 and
# feedback-tap positions at [6, 1] and [6, 5, 2, 1]. They were flip-balanced optimized for the iPad screen, and a subset
# of 20 codes was selected. For more details, see the original publication. In addition to reading the codes, the codes
# are upsampled to the EEG sampling frequency.

# The screen refresh rate
FR = 60

# Load codes
V = sio.loadmat(os.path.join(data_path, "resources", "mgold_61_6521_flip_balanced_20.mat"))["codes"].T

# Upsample codes from screen framerate to EEG sampling rate
V = np.repeat(V, fs // FR, axis=1).astype("uint8")

# %%
# Reading and preprocessing the EEG data
# --------------------------------------
# The cell below performs the reading and preprocessing of the EEG data, given the configuration above. The experiment
# consisted of 5 blocks during each of which 20 trials were recorded, one for each of the codes in random order. The
# cell results into (1) the data `X` that is a matrix of k trials c channels, and m samples, (2) the ground-truth labels
# `y` that is a vector of k trials, (3) the codes `V` that is a matrix of n classes and m samples, and (4) the sampling
# frequency `fs`.
#
# Note, the labels in `y` refer to the index in `V` at which to find the target (i.e., attended) code for a particular
# trial.
#
# The cell reads the data from the sourcedata folder in the dataset, and saves the processed data for each participant
# in a derivatives folder with a folder structure identical to the sourcedata.

# The experimental blocks
BLOCKS = ["block_1", "block_2", "block_3", "block_4", "block_5"]

# Loop over subjects
for subject in subjects:

    epochs = []
    labels = []

    # Loop over blocks
    for block in BLOCKS:
        # Find gdf file
        folder = os.path.join(data_path, "sourcedata", "offline", subject, block, f"{subject}_*_{block}_main_eeg.gdf")
        listing = glob.glob(folder)
        assert len(listing) == 1, f"Found none or multiple files for {subject}_{block}, should be a single file!"
        fname = listing[0]

        # Read raw file
        raw = mne.io.read_raw_gdf(fname, stim_channel="status", preload=True, verbose=False)

        # Read events
        events = mne.find_events(raw, verbose=False)

        # Select only the start of a trial
        # N.B. Every 2.1 seconds a trigger was generated (15 times per trial, plus one 16th "leaking trigger")
        # N.B. This "leaking trigger" is not always present, so taking epoch[::16, :] won't work, unfortunately
        cond = np.logical_or(np.diff(events[:, 0]) < 1.8 * raw.info['sfreq'],
                             np.diff(events[:, 0]) > 2.4 * raw.info['sfreq'])
        idx = np.concatenate(([0], 1 + np.where(cond)[0]))
        onsets = events[idx, :]

        # Visualize events
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1, figsize=(17, 3))
        # ax.scatter(events[:, 0] / raw.info['sfreq'], events[:, 2], marker=".")
        # ax.scatter(onsets[:, 0] / raw.info['sfreq'], onsets[:, 2], marker="x")

        # Spectral notch filter
        raw.notch_filter(freqs=np.arange(notch, raw.info['sfreq'] / 2, notch), verbose=False)

        # Spectral band-pass filter
        raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], verbose=False)

        # Slice data to trials
        # N.B. 500 ms pre-trial is added which is removed later. This is added to catch filter artefacts from the
        # subsequent resample. The resample is done after slicing to maintain accurate marker timing.
        epo = mne.Epochs(raw, events=onsets, tmin=tmin - 0.5, tmax=tmax, baseline=None, picks="eeg", preload=True,
                         verbose=False)

        # Downsample
        epo.resample(sfreq=fs, verbose=False)

        # Add to dataset
        epochs.append(epo)

        # Read labels
        # N.B. Minus 1 to convert Matlab counting from 1 to Python counting from 0
        f = h5py.File(os.path.join(data_path, "sourcedata", "offline", subject, block, "trainlabels.mat"), "r")
        labels.append(np.array(f["v"]).astype("uint8").flatten() - 1)

    # Extract data and concatenate runs
    X = mne.concatenate_epochs(epochs, verbose=False).get_data(tmin=tmin, tmax=tmax).astype("float32")
    y = np.array(labels).flatten().astype("uint8")

    # Save data
    np.savez(os.path.join(out_path, f"thielen2021_{subject}.npz"), X=X, y=y, V=V, fs=fs)

    # Print summary
    print("Subject: ", subject)
    print("\tX shape:", X.shape, "(trials, channels, samples)")
    print("\ty shape:", y.shape, "(trials)")
    print("\tV shape:", V.shape, "(classes, samples)")
    print("\tfs: ", fs)
