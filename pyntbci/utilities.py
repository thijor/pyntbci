from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from scipy.signal import butter, buttord, cheby1, cheb1ord, filtfilt


EVENTS = ("id", "on", "off", "onoff", "dur", "re", "fe", "refe")


def correct_latency(
        X: NDArray,
        y: NDArray,
        latency: NDArray,
        fs: int,
        axis: int = -1,
) -> NDArray:
    """Correct for a latency in data. This is done by shifting data according to the class-specific latencies.

    Parameters
    ----------
    X: NDArray
        The EEG data of shape (n_trials, ...)
    y: NDArray
        The labels of the EEG data of shape (n_trials) that indicate which latency to use for which trials.
    latency: NDArray
        The list with latencies in seconds of shape (n_classes).
    fs: int
        The sampling frequency of the EEG data in Hz.
    axis: int (default: -1)
        The axis in X to perform the correction to.
    
    Returns
    -------
    X: NDArray
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


def correlation(
        A: NDArray,
        B: NDArray,
) -> NDArray:
    """Compute the correlation coefficient. Computed between two sets of variables.

    Parameters
    ----------
    A: NDArray
        The first set of variables of shape (n_A, n_samples).
    B: NDArray
        The second set of variables of shape (n_B, n_samples).

    Returns
    -------
    scores: NDArray
        The correlation matrix of shape (n_A, n_B).
    """
    if not np.issubdtype(A.dtype, np.floating):
        A = A.astype("float")
    if not np.issubdtype(B.dtype, np.floating):
        B = B.astype("float")
    if A.ndim == 1:
        A = A[np.newaxis, :]
    if B.ndim == 1:
        B = B[np.newaxis, :]
    assert A.shape[1] == B.shape[1], f"Number of samples in A ({A.shape[1]}) does not equal B ({B.shape[1]})"

    A -= A.mean(axis=1, keepdims=True)
    B -= B.mean(axis=1, keepdims=True)

    ssA = (A ** 2).sum(axis=1, keepdims=True)
    ssB = (B ** 2).sum(axis=1, keepdims=True)
    scores = A @ B.T / np.sqrt(ssA * ssB.T)

    return scores


def covariance(
        data: NDArray,
        n_old: int = 0,
        avg_old: NDArray = None,
        cov_old: NDArray = None,
        estimator: BaseEstimator = None,
        running: bool = False,
) -> tuple[int, NDArray, NDArray]:
    """Compute the covariance matrix.

    Parameters
    ----------
    data: NDArray
        Data matrix of shape (n_samples, n_features).
    n_old: int (default: 0)
        Number of already observed samples.
    avg_old: NDArray (default: None)
        Already observed average of shape (n_features).
    cov_old: NDArray (default: None)
        Already observed covariance of shape (n_features, n_features).
    estimator: BaseEstimator (default: None)
        An object that estimates a covariance matrix using a fit method. If None, a custom implementation of the
        empirical covariance is used.
    running: bool (default: False)
        Whether to use a running covariance. If False, the covariance matrix is computed instantaneously using
        only X, such that n_old, avg_old, and cov_old are not used.

    Returns
    -------
    n_new: int
        Number of samples.
    avg_new: NDArray
        The average of shape (1, n_features).
    cov_new: NDArray
        The covariance of shape (n_features, n_features).
    """
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype("float")
    n_obs = data.shape[0]
    avg_obs = data.mean(axis=0, keepdims=True)
    if n_old == 0 or not running:
        n_new = n_obs
        avg_new = avg_obs
        data1 = data - avg_obs
        if estimator is None:
            cov_obs = data1.T @ data1 / (n_new - 1)
        else:
            cov_obs = estimator.fit(data1).covariance_
        cov_new = cov_obs
    else:
        n_new = n_old + n_obs
        data1 = data - avg_old
        avg_new = avg_old + (avg_obs - avg_old) * (n_obs / n_new)
        data2 = data - avg_new
        if estimator is None:
            cov_obs = data1.T @ data2 / (n_new - 1)
        else:
            # TODO: Compute the cumulative cross-covariance using estimator
            raise NotImplementedError
        cov_new = cov_obs + cov_old * ((n_new - n_obs - 1) / (n_new - 1))
    return n_new, avg_new, cov_new


def decoding_matrix(
        data: NDArray,
        length: int,
        stride: int = 1,
) -> NDArray:
    """Make a Hankel-like decoding matrix. Used to phase-shift the data (i.e., backward / decoding model), to learn a
    spatio-spectral filter (i.e., a spectral filter per channel).

    Parameters
    ----------
    data: NDArray
        Data matrix of shape (n_trials, n_channels, n_samples).
    length: int
        The length in samples of the spectral filter, i.e., the number of phase-shifted data per channel.
    stride: int (default: 1)
        The step size in samples over the length of the spectral filter.

    Returns
    -------
    dmatrix: NDArray
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


def encoding_matrix(
        stimulus: np.array,
        length: Union[int, list[int], tuple[int,...], NDArray],
        stride: Union[int, list[int], tuple[int,...], NDArray] = 1,
        amplitude: NDArray = None,
        tmin: float = 0,
) -> NDArray:
    """Make a Toeplitz-like encoding matrix. Used to phase-shift the stimulus (forward / encoding model), per event to
    learn a (or several) temporal filter(s). Also called a "structure matrix" or "design matrix".

    Parameters
    ----------
    stimulus: NDArray
        Stimulus matrix of shape (n_classes, n_events, n_samples).
    length: int | list[int] | tuple[int] | NDArray
        The length in samples of the temporal filter, i.e., the number of phase-shifted stimulus per event. If an array
        is provided, it denotes the length per event. If one value is provided, it is assumed all event responses are of
        the same length.
    stride: int | list[int] | tuple[int] | NDArray (default: 1)
        The step size in samples over the length of the temporal filter. If an array is provided, it denotes the stride
        per event. If one value is provided, it is assumed all event responses have the same stride.
    amplitude: NDArray (default: None)
        Amplitude information to embed in the encoding matrix of shape (n_classes, n_samples). If None, it is ignored.
    tmin: float (default: 0)
        The start of stimulation in samples. Can be used if there was a delay in the marker.
    Returns
    -------
    ematrix: NDArray
        Encoding matrix of shape (n_classes, n_events * length / stride, n_samples).
    """
    n_classes, n_events, n_samples = stimulus.shape

    assert (isinstance(length, int) or isinstance(length, list) or isinstance(length, tuple) or
            isinstance(length, np.ndarray)), "length must be int, list[int], tuple[int], or np.ndarray()."
    if isinstance(length, int):
        length = np.array(n_events * [length])
    elif isinstance(length, list) or isinstance(length, tuple):
        if len(length) == 1:
            length *= n_events
        length = np.array(length)
    elif isinstance(length, np.ndarray):
        if length.size == 1:
            length = np.repeat(length, n_events)
    assert length.size == n_events, "the number of events in length must match those in stimulus."
    assert np.issubdtype(length.dtype, np.integer), "length must contain integer values."

    assert (isinstance(stride, int) or isinstance(stride, list) or isinstance(stride, tuple) or
            isinstance(stride, np.ndarray)), "stride must be int, list[int], tuple[int], or np.ndarray()."
    if isinstance(stride, int):
        stride = np.array(n_events * [stride])
    elif isinstance(stride, list) or isinstance(stride, tuple):
        if len(stride) == 1:
            stride *= n_events
        stride = np.array(stride)
    elif isinstance(stride, np.ndarray):
        if stride.size == 1:
            stride = np.repeat(stride, n_events)
    assert stride.size == n_events, "the number of events in stride must match those in stimulus."
    assert np.issubdtype(stride.dtype, np.integer), "stride must contain integer values."

    # Create encoding window per event
    ematrix = []
    for i_event in range(n_events):
        stride_ = int(stride[i_event])

        # Add amplitude information
        if amplitude is not None:
            stimulus[:, i_event, :] *= amplitude

        # Create Toeplitz structure
        n_windows = int(length[i_event] / stride[i_event])
        tmp = np.zeros((n_classes, n_windows, n_samples), dtype=stimulus.dtype)
        tmp[:, 0, :] = stimulus[:, i_event, :]
        for i_window in range(1, n_windows):
            tmp[:, i_window, :] = np.roll(tmp[:, i_window - 1, :], stride_, axis=1)
            tmp[:, i_window, :stride_] = 0
        ematrix.append(tmp)

    # Concatenate matrices per event
    ematrix = np.concatenate(ematrix, axis=1)

    # Add delay
    if tmin < 0:
        # Stimulation started earlier, pad with zeros at the end
        ematrix = np.concatenate((
            ematrix[:, :, np.abs(tmin):],
            np.zeros((ematrix.shape[0], ematrix.shape[1], np.abs(tmin)))
        ), axis=2)
    elif tmin > 0:
        # Stimulation started later, pad with zeros at the start
        ematrix = np.concatenate((
            np.zeros((ematrix.shape[0], ematrix.shape[1], tmin)),
            ematrix[:, :, :-tmin]
        ), axis=2)

    return ematrix


def euclidean(
        A: NDArray,
        B: NDArray,
) -> NDArray:
    """Compute the Euclidean distance. Computed between two sets of variables.

    Parameters
    ----------
    A: NDArray
        The first set of variables of shape (n_A, n_samples).
    B: NDArray
        The second set of variables of shape (n_B, n_samples).

    Returns
    -------
    scores: NDArray
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


def event_matrix(
        stimulus: NDArray,
        event: str,
        onset_event: bool = False,
) -> tuple[NDArray, tuple[str]]:
    """Make an event matrix. The event matrix describes the onset of events in a stimulus sequence, given a particular
    event definition.

    Parameters
    ----------
    stimulus: NDArray
        The stimulus used for stimulation of shape (n_stimuli, n_samples). Note, this can be any continuous time-series
        per stimulus, it is not limited to binary sequences.
    event: str
        The event type to perform the transformation of codes to events with:
            "id" | "identity" | "stim" | "stimulus": the events are continuous, the stimulus itself.
            "on": whenever the stimulus value is larger than 0.
            "off": whenever the stimulus value is 0.
            "onoff": one event for on, one event for off.
            "dur" | "duration": each run-length of the same value is an event (e.g., 1, 11, 111, etc.).
            "re" | "rise" | "risingedge": each transition of a lower to a higher value is an event (e.g., 01).
            "fe" | "fall" | "fallingedge": each transition of a higher to a lower value is an event (i.e., 10).
            "refe" | "risefall" | "risingedgefallingedge" | "contrast": one event for rise and one event for fall
    onset_event: bool (default: False)
        Whether to model the onset of stimulation. This "onset" event is added as last event.

    Returns
    -------
    events: NDArray
        An event matrix of zeros and ones denoting the onsets of events of shape (n_stimuli, n_events, n_samples).
    labels: tuple
        A tuple of event labels of shape (n_events).
    """
    if stimulus.ndim == 1:
        stimulus = stimulus[np.newaxis, :]
    n_stims, n_samples = stimulus.shape

    if event == "id" or event == "identity" or event == "stim" or event == "stimulus":
        events = stimulus[:, np.newaxis, :]
        labels = (event,)

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

        # Convert dictionary to event matrix with sorted labels
        labels = sorted(events.keys())
        events = np.array([events[duration] for duration in labels]).transpose((1, 0, 2))
        labels = tuple([str(label) for label in labels])

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


def filterbank(
        X: NDArray,
        passbands: list[tuple[float, float]],
        fs: float,
        tmin: float = None,
        ftype: str = "butterworth",
        N: Union[int, list[int]] = None,
        stopbands: list[tuple[float, float]] = None,
        gpass: Union[float, list[float]] = 3.0,
        gstop: Union[float, list[float]] = 40.0,
) -> NDArray:
    """Apply a filterbank. Note, the order of the filter is set according to the maximum loss in the passband and the
    minimum loss in the stopband.

    Parameters
    ----------
    X: NDArray
        The matrix of EEG data of shape (n_trials, n_channels, n_samples).
    passbands: list[tuple[float, float]]
        A list of tuples with passbands defined as (lower, higher) cutoff in Hz.
    fs: int
        The sampling frequency of the EEG data in Hz.
    tmin: float (default: None)
        The window before trial onset that can catch any filter artefacts and will be cut off after filtering. If None,
        no data will be cut away.
    ftype: str (default: "butterworth")
        The filter type: "butterworth" | "chebyshev1"
    N: int | list[int] (default: None)
        The filter order. If a list is provided, it is the order for each passband. If None, the order is set given the
        stopbands, gpass and gstop.
    stopbands: list[tuple[float, float]] (default: None)
        A tuple of tuples with a stopband for each passband defined as (lower, higher) cutoff in Hz. If None, the
        stopbands default to (lower-2, higher+7) of the passbands. Only used if N=None.
    gpass: float | list[float] (default: 3.0)
        The maximum loss in the passband (dB). If a list is provided, it is the gpass for each passband. Only used if
        N=None.
    gstop: float | list[float] (default: 30.0)
        The minimum attenuation in the stopband (dB). If a list is provided, it is the gstop for each stopband. Only
        used if N=None.

    Returns
    -------
    X: NDArray
        The matrix of EEG data of shape (n_trials, n_channels, n_samples, n_passbands). If tmin is not None, n_samples
        will be reduced with tmin * fs number of samples.
    """
    assert isinstance(passbands, list), "The passbands should be a list of tuples."
    for passband in passbands:
        assert isinstance(passband, tuple), "The passbands should be a list of tuples."

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


def find_neighbours(
        layout: NDArray,
        border_value: int = -1
) -> NDArray:
    """
    Find the neighbour pairs (horizontal, vertical, diagonal) in a rectangular layout.

    Parameters
    ----------
    layout: NDArray
        A matrix of identities of shape (rows, columns).
    border_value: int (default: -1)
        A value not existing in the layout to represent a border to prevent wrapping around edges.

    Returns
    -------
    neighbours: NDArray
        A matrix of neighbour pairs of shape (n_neighbours, 2).
    """
    assert border_value not in layout, "The border_value must not be in the layout."

    # Add a border around the layout
    layout = np.concatenate((
        np.full((1, layout.shape[1]), border_value),
        layout,
        np.full((1, layout.shape[1]), border_value)), axis=0)
    layout = np.concatenate((
        np.full((layout.shape[0], 1), border_value),
        layout,
        np.full((layout.shape[0], 1), border_value)), axis=1)

    # Find all neighbours
    neighbours = np.stack((
        np.roll(layout, -1, axis=1).flatten(order="F"),
        np.roll(layout, -1, axis=0).flatten(order="F"),
        np.roll(np.roll(layout, -1, axis=0), -1, axis=1).flatten(order="F"),
        np.roll(np.roll(layout, -1, axis=0), 1, axis=1).flatten(order="F"),
    ), axis=1)

    # Find all neighbour pairs
    neighbours = np.stack((
        np.tile(layout.flatten(order="F"), (1, neighbours.shape[1])).flatten(order="F"),
        neighbours.flatten(order="F"),
    ), axis=1)

    # Remove the border
    neighbours = neighbours[~np.any(neighbours == border_value, axis=1), :]

    # Sort
    neighbours = neighbours[neighbours[:, 0].argsort(), :]

    return neighbours


def find_worst_neighbour(
        score: NDArray,
        neighbours: NDArray,
        layout: NDArray
) -> tuple[tuple[int, int], float]:
    """
    Find the neighbouring pair with maximum score.

    Parameters
    ----------
    score: NDArray
        The matrix of scores of shape (n_codes, n_codes).
    neighbours: NDArray
        The matrix of neighbouring pairs of shape (n_pairs, 2).
    layout: NDArray
        The vector mapping positions to identities of shape (n_codes).

    Returns
    -------
    idx: tuple[int, int]
        The two indexes of the neighbouring codes in the layout that have a maximum score.
    val: float
        The maximum score.
    """
    idx = neighbours[np.argmax(score[layout[neighbours[:, 0]], layout[neighbours[:, 1]]]), :]
    val = score[layout[idx[0]], layout[idx[1]]]
    return idx, val


def pinv(
        A: NDArray,
        alpha: float = None
) -> NDArray:
    """
    Compute the pseudo-inverse of a matrix.

    Parameters
    ----------
    A: NDArray
        Matrix of shape p x q to compute pseudo-inverse for.
    alpha: float (default: None)
        The amount of variance the retain.

    Returns
    -------
    iA: NDArray
        The pseudo-inverse of A of shape p x q.
    """
    assert A.ndim == 2, "A should be a matrix."
    assert not np.isnan(A).any(), "A should not contains NaNs."
    assert not np.isinf(A).any(), "A should not contains Infs."
    U, d, V = np.linalg.svd(A, full_matrices=False)
    if alpha is None:
        d = 1 / d
    else:
        for i in range(d.size):
            if (d[:d.size - i] / d.sum()).sum() < alpha:
                d = 1 / d
                d[d.size - i:] = 0
                break
    iA = np.dot(U * d, V)
    return iA


def itr(
        n: Union[int, NDArray],
        p: Union[float, NDArray],
        t: Union[float, NDArray],
) -> NDArray:
    """Compute the information-transfer rate (ITR).

    Parameters
    ----------
    n: int | NDArray:
        The number of classes.
    p: float | NDArray:
        The decoding accuracy between 0 and 1.
    t: float | NDArray:
        The decoding time in seconds (including inter-trial time).

    Returns
    -------
    itr: float | NDArray
        The ITR in bits per minute.
    """
    p = np.atleast_1d(p)
    p[p >= 1] = 1.0 - np.finfo(p.dtype).eps
    p[p <= 0] = np.finfo(p.dtype).eps
    b = np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))
    return b * (60 / t)


def trials_to_epochs(
        X: NDArray,
        y: NDArray,
        codes: NDArray,
        epoch_size: int,
        step_size: int,
) -> tuple[NDArray, NDArray]:
    """Slice trials to epochs.
    
    Parameters
    ----------
    X: NDArray
        The EEG data of shape (n_trials, n_channels, n_samples).
    y: NDArray
        Label vector of shape (n_trials).
    codes: NDArray
        Codes matrix of shape (n_codes, n_samples).
    epoch_size: int
        The width of an epoch starting at the onset of an epoch in samples.
    step_size: int
        The distance between consecutive epochs in samples.

    Returns
    -------
    NDArray
        The sliced EEG data of shape (n_trials, n_epochs, n_channels, epoch_size).
    NDArray
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
