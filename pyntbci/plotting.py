from matplotlib.axes import Axes
from matplotlib.patches import Circle, Ellipse, Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
from numpy.typing import NDArray


def eventplot(
        S: NDArray,
        E: NDArray,
        fs: int,
        ax: Axes = None,
        upsample: int = 20,
        plotfs: bool = True,
        events: tuple[str] = None,
) -> None:
    """
    Plot the event time-series. Specifically, shows a figure with the original stimulus and the decomposed events across
    time.

    Parameters
    ----------
    S: NDArray
        The stimulus time-series of shape (n_samples)
    E: NDArray
        The event time-series of shape (n_events, n_samples)
    fs: int
        The sampling rate in Hz
    ax: Axes (default: None)
        The axis to plot in. If None, a new figure will be opened.
    upsample: int (default: 20)
        A scalar value to upsample the stimulus and event time-series with for improved visualization
    plotfs: bool (default: True)
        Whether to plot vertical gridlines at the original sampling rate fs
    events: tuple (default: None)
        A tuple with the names of each of the events
    """
    # Make figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    n_events, n_samples = E.shape
    S = S.astype("float")
    E = E.astype("float")

    # Upsample for better visibility
    S = S.repeat(20)
    E = E.repeat(20, axis=1)

    # Plot stimulus
    S -= S.min()
    if S.max() != 0:
        S /= S.max()
    ax.plot(np.arange(n_samples * upsample) / (upsample * fs), S, "k")

    # Plot events
    for i in range(n_events):
        E[i, :] -= E[i, :].min()
        if E[i, :].max() != 0:
            E[i, :] /= E[i, :].max()
        ax.plot(np.arange(n_samples * upsample) / (upsample * fs), -1.5 * (1 + i) + E[i, :])

    # Plot sampling rate
    if plotfs:
        for i in range(1 + int(n_samples)):
            ax.axvline(i / fs, c="k", alpha=0.1)

    # Markup
    ax.set_xlabel("time [s]")
    if events is None:
        ax.set_ylabel("events")
    else:
        ax.set_ylabel("")
        ax.set_yticks(-1.5 * np.arange(0, 1 + n_events) + 0.5, ("stimulus",) + events)


def stimplot(
        S: NDArray,
        fs: int,
        ax: Axes = None,
        upsample: int = 20,
        plotfs: bool = True,
        labels: list[str] = None,
) -> None:
    """
    Plot the stimulus time-series.

    Parameters
    ----------
    S: NDArray
        The stimulus time-series of shape (n_stimuli, n_samples)
    fs: int
        The sampling rate in Hz
    ax: Axes (default: None)
        The axis to plot in. If None, a new figure will be opened.
    upsample: int (default: 20)
        A scalar value to upsample the stimulus and event time-series with for improved visualization
    plotfs: bool (default: True)
        Whether to plot vertical gridlines at the original sampling rate fs
    labels: list[str] (default: None)
        A list of string labels for each of the stimuli, used as y ticks in the stimulus plot. If None, stimulus labels
        of 1 to the number of stimuli is used.
    """
    # Make figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # Up-sample to better visualize the sharp edges
    S = S.repeat(upsample, axis=1)
    n_stimuli, n_samples = S.shape

    # Visualize stimuli
    ax.plot(np.arange(n_samples) / (upsample * fs), 2 * np.arange(n_stimuli) + S[::-1, :].T)

    # Add sample rate
    if plotfs:
        for i in range(1 + int(n_samples / upsample)):
            ax.axvline(i / fs, c="k", alpha=0.1)

    # Label axes
    if labels is None:
        ax.set_yticks(2 * np.arange(n_stimuli), np.arange(1, 1 + n_stimuli)[::-1])
    else:
        assert len(labels) == n_stimuli, "The number of provided labels does not match the number of stimuli."
        ax.set_yticks(2 * np.arange(n_stimuli), labels[::-1])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("stimulus")


def topoplot(
        z: NDArray,
        locfile: str,
        cbar: bool = False,
        ax: Axes = None,
        iso: bool = False,
        chan: bool = True,
) -> None:
    """Plot a topoplot. The values at each electrode are interpolated on an outline of a head using an electrode
    position file (loc file).

    Parameters
    ----------
    z: NDArray
        A vector of electrode values, e.g. a spatial filter/patterns of shape (n_channels).
    locfile: str
        A .loc file with electrode position information.
    cbar: bool (default: False)
        Whether to add a colorbar.
    ax: Axes (default: None)
        Axes to plot in. A new one is made when None.
    iso: bool (default: False)
        Whether to add iso lines.
    chan: bool (default: True)
        Whether to plot the channel positions.
    """
    assert locfile[-4:] == ".loc", "The topoplot function accepts .loc files only."

    # Read electrode positions from .loc file
    with open(locfile) as fid:
        lines = fid.read().split("\n")
        xy = np.zeros((len(lines), 2))
        for i, line in enumerate(lines):
            __, t, r, __ = line.split("\t")
            t = (float(t) + 90) / 180 * np.pi
            r = float(r) * 2
            xy[i, :] = r * np.cos(t), r * np.sin(t)

    # Add additional points for interpolation to edge of head
    edge = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    xy = np.concatenate((xy, edge), axis=0)
    z = np.concatenate((z, np.zeros((4,))), axis=None).astype("float")

    # Make grid
    N = 300
    xi = np.linspace(-2, 2, N)
    yi = np.linspace(-2, 2, N)
    zi = griddata((xy[:, 0], xy[:, 1]), z, (xi[None, :], yi[:, None]), method="cubic", fill_value=np.nan)

    # Set points outside radius to nan, so they will not be plotted.
    d = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            if np.sqrt(xi[i] ** 2 + yi[j] ** 2) + d > 1:
                zi[j, i] = np.nan

    # Make figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect=1)

    # Add head
    circle = Circle(xy=(0., 0.), radius=1, edgecolor="k", facecolor="w", zorder=1)
    ax.add_patch(circle)

    # Add ears
    circle = Ellipse(xy=(-1., 0.), width=0.25, height=0.5, angle=0, edgecolor="k", facecolor="w", zorder=0)
    ax.add_patch(circle)
    circle = Ellipse(xy=(1., 0.), width=0.25, height=0.5, angle=0, edgecolor="k", facecolor="w", zorder=0)
    ax.add_patch(circle)

    # Add a nose
    polygon = Polygon(xy=[(-0.1, 0.9), (0, 1.25), (0.1, 0.9)], edgecolor="k", facecolor="w", zorder=0)
    ax.add_patch(polygon)

    # Add the interpolated data
    cs = ax.contourf(xi, yi, zi, 60, cmap="RdYlBu_r", zorder=2)

    # Add iso-lines
    if iso:
        ax.contour(xi, yi, zi, 15, colors="grey", zorder=3)

    # Add data points
    if chan:
        ax.scatter(xy[:-4, 0], xy[:-4, 1], marker="o", c="k", s=15, zorder=4)

    # Add color bar
    if cbar:
        ax.get_figure().colorbar(cs, ax=ax)

    # Make the axis invisible
    ax.axis("off")

    # set axes limits
    ax.set_xlim(1.25, -1.25)
    ax.set_ylim(-1.05, 1.4)
    ax.set_aspect("equal")
