import numpy as np
from scipy.signal import butter, buttord, cheby1, cheb1ord, filtfilt


EVENTS = ("id", "on", "off", "onoff", "dur", "re", "fe", "refe")


def correct_latency(X, y, latency, fs, axis=-1):
    """Correct for a latency in data. This is done by shifting data according to the class-specific latencies.

    Parameters
    ----------
    X: np.ndarray
        The EEG data of shape (n_trials, ...)
    y: np.ndarray
        The labels of the EEG data of shape (n_trials) that indicate which latency to use for which trials.
    latency: np.ndarray
        The list with latencies in seconds of shape (n_classes).
    fs: int
        The sampling frequency of the EEG data in Hz.
    axis: int (default: -1)
        The axis in X to perform the correction to.
    
    Returns
    -------
    X: np.ndarray
        The latency-corrected EEG data of shape (n_trials, ...).
    """
    Z = np.zeros(X.shape, dtype=X.dtype)
    for label in np.unique(y):
        idx = label == y
        if latency[label] == 0:
            Z[idx, ...] = X[idx, ...]
        else:
            shift = int(np.round(latency[label] * fs))
            Z[idx, ...] = np.roll(X[idx, ...], shift, axis=axis)
    return Z


def correlation(A, B):
    """Compute the correlation coefficient. Computed between two sets of variables.

    Parameters
    ----------
    A: np.ndarray
        The first set of variables of shape (n_A, n_samples).
    B: np.ndarray
        The second set of variables of shape (n_B, n_samples).

    Returns
    -------
    scores: np.ndarray
        The correlation matrix of shape (n_A, n_B).
    """
    if A.ndim == 1:
        A = A[np.newaxis, :]
    if B.ndim == 1:
        B = B[np.newaxis, :]
    assert A.shape[1] == B.shape[1], f"Number of samples in A ({A.shape[1]}) does not equal B ({B.shape[1]})"

    # Demean
    Az = A - np.mean(A, axis=1, keepdims=True)
    Bz = B - np.mean(B, axis=1, keepdims=True)

    # Sum of squares
    ssA = np.sum(Az ** 2, axis=1, keepdims=True)
    ssB = np.sum(Bz ** 2, axis=1, keepdims=True)

    # Correlation coefficient
    outer = np.dot(ssA, ssB.T)
    scores = np.dot(Az, Bz.T) / np.sqrt(outer)

    return scores


def covariance(X, n_old=0, avg_old=None, cov_old=None, estimator=None, running=False):
    """Compute the covariance matrix.

    Parameters
    ----------
    X: np.ndarray
        Data matrix of shape (n_samples, n_features).
    n_old: int (default: 0)
        Number of already observed samples.
    avg_old: np.ndarray (default: None)
        Already observed average of shape (n_features).
    cov_old: np.ndarray (default: None)
        Already observed covariance of shape (n_features, n_features).
    estimator: object (default: None)
        An object that estimates a covariance matrix using a fit method. If None, a custom implementation of the
        empirical covariance is used.
    running: bool (default: False)
        Whether or not to use a running covariance. If False, the covariance matrix is computed instantaneously using
        only X, such that n_old, avg_old, and cov_old are not used.

    Returns
    -------
    n_new: int
        Number of samples.
    avg_new: np.ndarray
        The average of shape (1, n_features).
    cov_new: np.ndarray
        The covariance of shape (n_features, n_features).
    """
    n_obs = X.shape[0]
    avg_obs = np.mean(X, axis=0, keepdims=True)
    if n_old == 0 or not running:
        n_new = n_obs
        avg_new = avg_obs
        X1 = X - avg_obs
        if estimator is None:
            cov_obs = np.dot(X1.T, X1) / (n_new - 1)
        else:
            cov_obs = estimator.fit(X1).covariance_
        cov_new = cov_obs
    else:
        n_new = n_old + n_obs
        X1 = X - avg_old
        avg_new = avg_old + (avg_obs - avg_old) * (n_obs / n_new)
        X2 = X - avg_new
        if estimator is None:
            cov_obs = np.dot(X1.T, X2) / (n_new - 1)
        else:
            # TODO: Compute the cumulative cross-covariance X1 and X2 using estimator
            raise NotImplementedError
        cov_new = cov_obs + cov_old * ((n_new - n_obs - 1) / (n_new - 1))
    return n_new, avg_new, cov_new


def decoding_matrix(data, length, stride=1):
    """Make a Hankel-like decoding matrix. Used to phase-shift the data (i.e., backward / decoding model), to learn a
    spatio-spectral filter (i.e., a spectral filter per channel).

    Parameters
    ----------
    data: np.ndarray
        Data matrix of shape (n_trials, n_channels, n_samples).
    length: int
        The length in samples of the spectral filter, i.e., the number of phase-shifted data per channel.
    stride: int (default: 1)
        The step size in samples over the length of the spectral filter.

    Returns
    -------
    dmatrix: np.ndarray
        Decoding matrix of shape (n_trials, n_channels * length / stride, n_samples).
    """
    n_trials, n_channels, n_samples = data.shape
    n_windows = int(length / stride)

    # Create Toeplitz structure
    dmatrix = np.zeros((n_trials, n_windows, n_channels, n_samples), dtype=data.dtype)
    dmatrix[:, 0, :, :] = data
    for i_window in range(1, n_windows):
        dmatrix[:, i_window, :, :] = np.roll(dmatrix[:, i_window - 1, :, :], -stride, axis=2)
        dmatrix[:, i_window, :, -stride:] = 0

    # Reshape to channel-prime
    dmatrix = dmatrix.reshape((n_trials, n_windows * n_channels, n_samples))
    return dmatrix


def encoding_matrix(stimulus, length, stride=1, amplitude=None):
    """Make a Toeplitz-like encoding matrix. Used to phase-shift the stimulus (forward / encoding model), per event to
    learn a (or several) temporal filter(s). Also called a "structure matrix" or "design matrix".

    Parameters
    ----------
    stimulus: np.ndarray
        Stimulus matrix of shape (n_classes, n_events, n_samples).
    length: int | list
        The length in samples of the temporal filter, i.e., the number of phase-shifted stimulus per event. If a list is
        provided, it denotes the length per event.
    stride: int (default: 1)
        The step size in samples over the length of the temporal filter.
    amplitude: np.ndarray (default: None)
        Amplitude information to embed in the encoding matrix of shape (n_classes, n_samples). If None, it is ignored.
    Returns
    -------
    ematrix: np.ndarray
        Encoding matrix of shape (n_trials, n_channels * length / stride, n_samples).
    """
    n_classes, n_events, n_samples = stimulus.shape

    if isinstance(length, int):
        length = n_events * [length]
    elif isinstance(length, list) or isinstance(length, tuple):
        assert len(length) == n_events, "len(encoding_length) does not match S.shape[1]."
    else:
        raise Exception("encoding_length should be (int, list, tuple).")

    # Create encoding window per event
    ematrix = []
    for i_event in range(n_events):

        # Add amplitude information
        if amplitude is not None:
            stimulus[:, i_event, :] *= amplitude

        # Create Toeplitz structure
        n_windows = int(length[i_event] / stride)
        tmp = np.zeros((n_classes, n_windows, n_samples), dtype=stimulus.dtype)
        tmp[:, 0, :] = stimulus[:, i_event, :]
        for i_window in range(1, n_windows):
            tmp[:, i_window, :] = np.roll(tmp[:, i_window - 1, :], stride, axis=1)
            tmp[:, i_window, :stride] = 0
        ematrix.append(tmp)

    # Concatenate matrices per event
    ematrix = np.concatenate(ematrix, axis=1)
    return ematrix


def euclidean(A, B):
    """Compute the Euclidean distance. Computed between two sets of variables.

    Parameters
    ----------
    A: np.ndarray
        The first set of variables of shape (n_A, n_samples).
    B: np.ndarray
        The second set of variables of shape (n_B, n_samples).

    Returns
    -------
    scores: np.ndarray
        The correlation matrix of shape (n_A, n_B).
    """
    if A.ndim == 1:
        A = A[np.newaxis, :]
    if B.ndim == 1:
        B = B[np.newaxis, :]
    assert A.shape[1] == B.shape[1], f"Number of samples in A ({A.shape[1]}) does not equal B ({B.shape[1]})"

    scores = np.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            c = A[i, :] - B[j, :]
            scores[i, j] = np.sqrt(np.inner(c, c))
    return scores


def event_matrix(stimulus, event, onset_event=False):
    """Make an event matrix. The event matrix describes the onset of events in a stimulus sequence, given a particular
    event definition.

    Parameters
    ----------
    stimulus: np.ndarray
        The stimulus used for stimulation of shape (n_stims, n_samples). Note, this can be any continuous time-series
        per stimulus, it is not limited to binary sequences.
    event: str
        The event type to perform the transformation of codes to events with:
            "id" | "identity": the events are continuous, the stimulus itself.
            "dur" | "duration": each run-length of the same value is an event (e.g., 1, 11, 111, etc.).
            "re" | "rise" | "risingedge": each transition of a lower to a higher value is an event (e.g., 01).
            "fe" | "fall" | "fallingedge": each transition of a higher to a lower value is an event (i.e., 10).
            "refe" | "risefall" | "risingedgefallingedge" | "contrast": one event for rising edges and one event for
            falling edges (i.e., 01 and 10).
    onset_event: bool (default: False)
        Whether or not to model the onset of stimulation. This "onset" event is added as last event.

    Returns
    -------
    events: np.ndarray
        An event matrix of zeros and ones denoting the onsets of events of shape (n_stims, n_events, n_samples).
    labels: tuple
        A tuple of event labels of shape (n_events).
    """
    if stimulus.ndim == 1:
        stimulus = stimulus[np.newaxis, :]
    n_stims, n_samples = stimulus.shape

    if event == "id" or event == "identity":
        events = stimulus[:, np.newaxis, :]
        labels = ("id",)

    elif event == "on":
        events = stimulus[:, np.newaxis, :] > 0
        labels = ("on",)

    elif event == "off":
        events = stimulus[:, np.newaxis, :] == 0
        labels = ("off",)

    elif event == "onoff":
        on = stimulus > 0
        off = stimulus == 0
        events = np.concatenate((on[:, np.newaxis, :], off[:, np.newaxis, :]), axis=1)
        labels = ("on", "off")

    elif event == "dur" or event == "duration":

        # Get rising and falling edges
        diff = np.diff(stimulus, axis=1)
        change = diff != 0

        # Create event dictionary
        events = dict()
        for i_code in range(n_stims):

            # Get rising and falling edge locations
            idx = 1 + np.concatenate(([-1], np.where(change[i_code, :])[0], [n_samples - 1]))

            # Get durations of events
            durations = np.diff(idx)

            # Ignore inactive periods
            durations = durations[stimulus[i_code, idx[:-1]] > 0]
            idx = idx[:-1][stimulus[i_code, idx[:-1]] > 0]

            # Fill out durations in dictionary
            unique_durations = np.unique(durations)
            for duration in unique_durations:
                if duration not in events:
                    events[duration] = np.zeros((n_stims, n_samples), dtype="bool_")
                events[duration][i_code, idx] = durations == duration

        # Extract unique events (sorted numerically or alphabetically)
        labels = tuple(sorted(events.keys()))

        # Convert dictionary to event matrix
        events = np.array([events[duration] for duration in labels]).transpose((1, 0, 2)).astype("bool_")

    # Get rising edges as event
    elif event == "re" or event == "rise" or event == "risingedge":
        diff = np.diff(np.concatenate((np.zeros((n_stims, 1)), stimulus), axis=1), axis=1)
        rise = diff > 0
        events = rise[:, np.newaxis, :]
        labels = ("rise",)

    # Get falling edges as event
    elif event == "fe" or event == "fall" or event == "fallingedge":
        diff = np.diff(np.concatenate((np.zeros((n_stims, 1)), stimulus), axis=1), axis=1)
        fall = diff < 0
        events = fall[:, np.newaxis, :]
        labels = ("fall",)

    # Get rising and falling edges as separate events
    elif event == "refe" or event == "risefall" or event == "risingedgefallingedge" or event == "contrast":
        diff = np.diff(np.concatenate((np.zeros((n_stims, 1)), stimulus), axis=1), axis=1)
        rise = diff > 0
        fall = diff < 0
        events = np.concatenate((rise[:, np.newaxis, :], fall[:, np.newaxis, :]), axis=1)
        labels = ("rise", "fall")

    else:
        raise Exception(f"Unknown event: {event}.")

    # Add onset response as separate event
    if onset_event:
        events = np.concatenate((events, np.zeros((n_stims, 1, n_samples))), axis=1)
        events[:, -1, 0] = 1
        labels += ("onset",)

    return events.astype("float32"), labels


def filterbank(X, passbands, fs, tmin=None, ftype="butterworth", N=None, stopbands=None, gpass=3.0, gstop=40.0):
    """Apply a filterbank. Note, the order of the filter is set according to the maximum loss in the passband and the
    minimum loss in the stopband.

    Parameters
    ----------
    X: np.ndarray
        The matrix of EEG data of shape (n_trials, n_channels, n_samples).
    passbands: list(list)
        A list of lists with passbands defined as [lower, higher] cutoff in Hz.
    fs: int
        The sampling frequency of the EEG data in Hz.
    tmin: float (default: None)
        The window before trial onset that can catch any filter artefacts and will be cut off after filtering. If None,
        no data will be cut away.
    ftype: str (default: "butterworth")
        The filter type: "butterworth" | "chebyshev1"
    N: int | list (default: None)
        The filter order. If a list is provided, it is the order for each passband. If None, the order is set given the
        stopbands, gpass and gstop.
    stopbands: list(list) (default: None)
        A list of lists with a stopband for each passband defined as [lower, higher] cutoff in Hz. If None, the
        stopbands default to [lower-2, higher+7] of the passbands. Only used if N=None.
    gpass: float | list (default: 3.0)
        The maximum loss in the passband (dB). If a list is provided, it is the gpass for each passband. Only used if
        N=None.
    gstop: float | list (default: 30.0)
        The minimum attenuation in the stopband (dB). If a list is provided, it is the gstop for each stopband. Only
        used if N=None.

    Returns
    -------
    X: np.ndarray
        The matrix of EEG data of shape (n_trials, n_channels, n_samples, n_passbands). If tmin is not None, n_samples
        will be reduced with tmin * fs number of samples.
    """
    for passband in passbands:
        assert isinstance(passband, list), "The passbands should be a list of lists."

    # Set default stopband to lower-2 and higher+7
    if N is None:
        if stopbands is None:
            stopbands = [[max([l - 2.0, 0.1]), min([h + 7.0, fs / 2])] for (l, h) in passbands]
        else:
            assert len(passbands) == len(stopbands), "Number of bands in passbands and stopbands should be equal."
        for passband, stopband in zip(passbands, stopbands):
            assert passband[0] > stopband[0], "Lower passband values should be higher than lower stopband values."
            assert passband[1] < stopband[1], "Higher passband values should be lower than higher stopband values."

    # Set default gpass and gstop
    if N is None:
        if not isinstance(gpass, list):
            gpass = len(passbands) * [gpass]
        else:
            assert len(passbands) == len(gpass), "The number of passbands should equal the number of gpass."
        if not isinstance(gstop, list):
            gstop = len(passbands) * [gstop]
        else:
            assert len(stopbands) == len(gstop), "The number of stopbands should equal the number of gstop."

    # Filter the data
    Xf = np.zeros((X.shape + (len(passbands),)), dtype=X.dtype)
    for i_band in range(len(passbands)):

        # Butterworth filter design
        if ftype == "butterworth":
            if N is None:
                N, Wn = buttord(wp=passbands[i_band], ws=stopbands[i_band], gpass=gpass[i_band], gstop=gstop[i_band],
                                fs=fs)
            else:
                Wn = passbands[i_band]
            b, a = butter(N=N, Wn=Wn, btype="bandpass", fs=fs)

        # Chebyshev Type I filter design
        elif ftype == "chebyshev1":
            if N is None:
                N, Wn = cheb1ord(wp=passbands[i_band], ws=stopbands[i_band], gpass=gpass[i_band], gstop=gstop[i_band],
                                 fs=fs)
            else:
                Wn = passbands[i_band]
            b, a = cheby1(N=N, rp=0.5, Wn=Wn, btype="bandpass", fs=fs)

        else:
            raise Exception("Unknown ftype:", ftype)

        # Filter the data
        Xf[:, :, :, i_band] = filtfilt(b, a, X, axis=2)

    # Cut away initial window that can capture filter artefacts
    if tmin is not None:
        Xf = Xf[:, :, int(abs(tmin * fs)):, :]

    return Xf


def itr(n, p, t):
    """Compute the information-transfer rate (ITR).

    Parameters
    ----------
    n: int | np.ndarray:
        The number of classes.
    p: float | np.ndarray:
        The decoding accuracy between 0 and 1.
    t: float | np.ndarray:
        The decoding time in seconds (including inter-trial time).

    Returns
    -------
    itr: float | np.ndarray
        The ITR in bits per minute.
    """
    p = np.atleast_1d(p)
    p[p >= 1] = 1.0 - np.finfo(p.dtype).eps
    p[p <= 0] = np.finfo(p.dtype).eps
    b = np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))
    return b * (60 / t)


def trials_to_epochs(X, y, codes, epoch_size, step_size):
    """Slice trials to epochs.
    
    Parameters
    ----------
    X: np.ndarray
        The EEG data of shape (n_trials, n_channels, n_samples).
    y: np.ndarray
        Label vector of shape (n_trials).
    codes: np.ndarray
        Codes matrix of shape (n_codes, n_samples).
    epoch_size: int
        The the width of an epoch starting at the onset of an epoch in samples.
    step_size: int
        The distance between consecutive epochs in samples.

    Returns
    -------
    np.ndarray
        The sliced EEG data of shape (n_trials, n_epochs, n_channels, epoch_size).
    np.ndarray
        The sliced label information of shape (n_trials, n_epochs).
    """
    n_trials, n_channels, n_samples = X.shape
    n_epochs = int((n_samples - epoch_size) / step_size)

    X_sliced = np.zeros((n_trials, n_epochs, n_channels, epoch_size), dtype="float32")
    y_sliced = np.zeros((n_trials, n_epochs), dtype="uint8")
    for i_epoch in range(n_epochs):
        start = i_epoch * step_size
        X_sliced[:, i_epoch, :, :] = X[:, :, start:start + epoch_size]
        y_sliced[:, i_epoch] = codes[y, start % codes.shape[1]]

    return X_sliced, y_sliced
