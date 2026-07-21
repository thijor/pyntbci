"""
MOABB
=====
This script shows how to use rCCA from PyntBCI for decoding c-VEP trials from real (not synthetic) EEG data, obtained
from the Thielen et al. (2015) c-VEP dataset [1]_ via MOABB (Mother of All BCI Benchmarks) [2]_. Cross-validation is performed
with a few lines of standard scikit-learn code, since `rCCA` is itself a scikit-learn compatible estimator.

Note, this example requires `moabb` and `mne` to be installed (`pip install moabb`), which are not dependencies of
PyntBCI itself. It also downloads several hundred MB of EEG data from https://data.ru.nl/ on first use, cached by
MOABB in `~/mne_data/`. For these reasons, this script is excluded from execution when building the documentation
(see `filename_pattern` in `doc/conf.py`); its code and output below were generated separately.

References
----------
.. [1] Thielen, J., van den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials:
       re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797.
       DOI: https://doi.org/10.1371/journal.pone.0133797
.. [2] MOABB Thielen2015 dataset: https://moabb.neurotechx.com/docs/generated/moabb.datasets.Thielen2015.html
"""

import mne
import numpy as np
from moabb.datasets import Thielen2015
from sklearn.model_selection import StratifiedKFold, cross_val_score

import pyntbci

mne.set_log_level("WARNING")

# %%
# Load data
# ---------
# MOABB's `CVEP` paradigm decodes individual bits, which does not match PyntBCI's trial-level reconvolution
# CCA. Instead, the raw MNE recordings are read directly with `Thielen2015().get_data()`, and trials are epoched by
# hand using the dataset's own `stim_trial` (trial onset and class label) and `stim_epoch` (per-sample stimulus bit)
# event channels. A single subject is used here to keep the example minimal and fast; loop over
# `Thielen2015().subject_list` to evaluate all subjects.

FS = 120  # target sampling frequency after resampling
PR = 120  # stimulus presentation rate
TRIAL_DURATION = 4.2  # seconds
N_BITS = int(round(PR * TRIAL_DURATION))  # 504

dataset = Thielen2015()
sessions = dataset.get_data(subjects=[1])[1]["0"]

X_list, y_list, codes = [], [], {}
for run_name, raw in sessions.items():
    trial_events = mne.find_events(raw, stim_channel="stim_trial", verbose=False)
    epoch_events = mne.find_events(raw, stim_channel="stim_epoch", verbose=False)
    onsets, labels = trial_events[:, 0], trial_events[:, 2] - 200  # dataset encodes labels as 200 + class

    # Reconstruct each class's binary stimulus code from the per-sample event channel, the first time it is seen
    for onset, label in zip(onsets, labels):
        if label not in codes:
            start = np.searchsorted(epoch_events[:, 0], onset)
            codes[label] = epoch_events[start : start + N_BITS, 2] - 100  # dataset encodes bits as 100 + value

    # Band-pass filter and epoch the continuous EEG into trials
    raw_eeg = raw.copy().pick("eeg")
    raw_eeg.filter(l_freq=6.0, h_freq=21.0, verbose=False)
    events = np.stack([onsets, np.zeros_like(onsets), labels], axis=1)
    epochs = mne.Epochs(
        raw_eeg, events, tmin=-0.5, tmax=TRIAL_DURATION + 0.5, baseline=None, preload=True, verbose=False
    )
    epochs.resample(FS, verbose=False)

    X_list.append(epochs.get_data(tmin=0, tmax=TRIAL_DURATION)[:, :, : N_BITS + 1].astype("float32"))
    y_list.append(labels.astype("int64"))

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
V = np.stack([codes[label] for label in range(len(codes))], axis=0).astype("float64")

# %%
# Inspect data
# ------------

print("X", X.shape, "(trials x channels x samples)", X.dtype)  # EEG
print("y", y.shape, "(trials)", y.dtype)  # labels
print("V", V.shape, "(classes, samples)", V.dtype)  # codes
print("fs", FS, "Hz")  # sampling frequency
print("fr", PR, "Hz")  # presentation rate

# %%
# Cross-validation
# ----------------
# Because `stimulus` is known upfront (it does not depend on the training split), `rCCA` can be plugged directly
# into scikit-learn's model selection routines without a wrapper. A `StratifiedKFold` is used here to match the
# dataset's 3 repetitions per class.

rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="duration", encoding_length=0.3, onset_event=True)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
accuracy = cross_val_score(rcca, X, y, cv=cv)

print(f"Accuracy: avg={accuracy.mean():.2f} with std={accuracy.std():.2f}")
