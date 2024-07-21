import numpy as np
from numpy.typing import NDArray
from scipy import signal


def aud_space_bw(
        fmin: float,
        fmax: float,
        bw: float = 1.0,
        scale: str = "erb",
) -> NDArray:
    """Auditory scale points specified by bandwidth. It computes a vector containing values equidistantly scaled between
    frequencies fmin and fmax at the auditory scale. All frequencies are specified in Hz. The distance between two
    consecutive values is bw on the auditory scale, and the points will be centered on the auditory scale between fmin
    and fmax.

    Adapted from LTFAT: https://ltfat.org/doc/auditory/audspacebw audspacebw.m (Peter L. Søndergaard)

    Parameters
    ----------
    fmin: float
        Minimum center frequency.
    fmax: float
        Maximum center frequency.
    bw: float (default: 1.0)
        Bandwidth or spacing between center frequencies.
    scale: str (default: "erb")
        Auditory scale.

    Returns
    -------
    y: NDArray
        A vector containing the center frequencies.
    """
    # Convert the frequency limits to auditory scale
    aud_limits = freq_to_aud(np.array([fmin, fmax]), scale)
    aud_range = aud_limits[1] - aud_limits[0]

    # Calculate number of points, excluding final point
    n = np.floor(aud_range / bw)

    # The remainder is calculated in order to center the points
    # correctly between fmin and fmax.
    remainder = aud_range - n * bw

    # Compute the center points
    aud_points = aud_limits[0] + np.arange(n + 1) * bw + remainder / 2

    # Convert auditory scale to frequencies
    y = aud_to_freq(aud_points, scale)

    return y


def aud_to_freq(
        aud: NDArray,
        scale: str = "erb",
) -> NDArray:
    """Convert auditory units at the auditory scale to frequency (Hz).

    Adapted from LTFAT: https://ltfat.org/doc/auditory/audtofreq audtofreq.m (Peter L. Søndergaard)

    Parameters
    ----------
    aud: NDArray
        The values at the ERB auditory scale.
    scale: str (default: "erb")
        Auditory scale.

    Returns
    -------
    freq: NDArray
        The values in frequencies measured in Hz.
    """
    if scale == "erb":
        freq = (1 / 0.00437) * np.sign(aud) * (np.exp(np.abs(aud) / 9.2645) - 1)
    else:
        raise Exception("Unknown auditory scale:", scale)
    return freq


def gammatone(
        audio: NDArray,
        fs: int,
        fs_inter: int = 8000,
        fs_target: int = 32,
        power: float = 0.6,
        lowpass: float = 9.0,
        fmin: float = 150.0,
        fmax: float = 4000.0,
        spacing: float = 1.5,
) -> NDArray:
    """Compute the envelope of audio using a gammatone filterbank.

    Developed in collaboration with Hanneke Scheppink.
    Adapted from: https://zenodo.org/records/3377911 preprocess_data.m (Neetha Das)
    Deviations:
        - Use of gammatone filterbank from scipy
        - Use of a lowpass rather than a bandpass on the envelope
        - Use of a butterworth filter rather than an equiripple
        - No downsample before the lowpass, as it seems redundant

    Parameters
    ----------
    audio: NDArray
        A vector containing the raw audio sampled at fs.
    fs: int
        The sampling frequency in Hz of the raw audio.
    fs_inter: int (default: 8000)
        The sampling frequency in Hz to which the audio will be downsampled for computational efficiency.
    fs_target: int (default: 32)
        The sampling frequency in Hz to which the envelope will be downsampled.
    power: float (default: 0.6)
        The power law coefficient applied to the envelope.
    lowpass: float (default: 9.0)
        The lowpass cutoff frequency in Hz to lowpass filter the envelope.
    fmin: float (default: 150.0)
        The minimum center frequency.
    fmax: float (default: 4000.0)
        The maxmum center frequency.
    spacing: float (default: 1.5)
        The bandwidth or spacing between center frequencies.

    Returns
    -------
    envelope: NDArray
        The envelope for each of the subbands in the gammatone filterbank of shape (samples, subbands).
    """
    freqs = erb_space_bw(fmin, fmax, spacing)

    # Build bandpass filter
    N, Wn = signal.buttord(wp=lowpass - 0.45, ws=lowpass + 0.45, gpass=0.5, gstop=15, fs=fs_inter)
    bbutter, abutter = signal.butter(N=N, Wn=Wn, btype="low", fs=fs_inter)

    # Resample stimulus
    audio = signal.resample(audio, int(audio.size / fs) * fs_inter)

    # Apply gammatone filterbank
    envelope = []
    for freq in freqs:
        # Build gammatone filter subband
        bgamma, agamma = signal.gammatone(freq, 'fir', order=4, fs=fs_inter)

        # Compute envelope of gammatone filter subband
        env = np.real(signal.filtfilt(bgamma, agamma, audio))

        # Apply the powerlaw
        env = np.abs(env) ** power

        # Lowpass filter the envelope
        env = signal.filtfilt(bbutter, 1, env)

        # Downsample to ultimate frequency
        env = signal.resample(env, int(env.size / fs_inter) * fs_target)

        envelope.append(env)

    # Stack gammatone filterbank subbands
    envelope = np.stack(envelope, axis=1)

    return envelope


def rms(
        audio: NDArray,
        fs: int,
        fs_inter: int = 8000,
        fs_target: int = 32,
) -> NDArray:
    """Compute the envelope of the audio as the root mean square (RMS) of the signal.

    Parameters
    ----------
    audio: NDArray
        A vector containing the raw audio sampled at fs.
    fs: int
        The sampling frequency in Hz of the raw audio.
    fs_inter: int (default: 8000)
        The sampling frequency in Hz to which the audio will be downsampled for computational efficiency.
    fs_target: int (default: 32)
        The sampling frequency in Hz to which the envelope will be downsampled.

    Returns
    -------
    envelope: NDArray
        The envelope using the RMS of the audio.
    """
    # Resample stimulus
    audio = signal.resample(audio, int(audio.size / fs) * fs_inter)

    # Reshape to compute RMS per bin
    audio = audio.reshape((-1, int(fs_inter / fs_target)))

    # Compute envelope using RMS
    envelope = np.sqrt(np.mean(audio ** 2, axis=1))

    return envelope


def erb_space_bw(
        fmin: float,
        fmax: float,
        bw: float = 1.0,
) -> NDArray:
    """Auditory scale points specified by bandwidth. It computes a vector containing values equidistantly scaled between
    frequencies fmin and fmax at the ERB auditory scale. All frequencies are specified in Hz. The distance between two
    consecutive values is bw on ERB auditory scale, and the points will be centered on the scale between fmin and fmax.

    Adapted from LTFAT: https://ltfat.org/doc/auditory/erbspacebw erbspacebw.m (Peter L. Søndergaard)

    Parameters
    ----------
    fmin: float
        Minimum center frequency.
    fmax: float
        Maximum center frequency.
    bw: float (default: 1.0)
        Bandwidth or spacing between center frequencies.

    Returns
    -------
    y: NDArray
        A vector containing the center frequencies.
    """
    y = aud_space_bw(fmin, fmax, bw, "erb")
    return y


def freq_to_aud(
        freq: NDArray,
        scale: str = "erb",
) -> NDArray:
    """Convert frequencies (Hz) to auditory units at the auditory scale.

    Adapted from LTFAT: https://ltfat.org/doc/auditory/freqtoaud freqtoaud.m (Peter L. Søndergaard)

    Parameters
    ----------
    freq: NDArray
        The values in frequencies measured in Hz.
    scale: str (default: "erb")
        Auditory scale.

    Returns
    -------
    aud: NDArray
        The values at the ERB auditory scale.
    """
    if scale == "erb":
        aud = 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)
    else:
        raise Exception("Unknown auditory scale:", scale)
    return aud
