from matplotlib.patches import Circle, Ellipse, Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np


def topoplot(z, locfile, cbar=False, ax=None, iso=False):
    """Plot a topoplot. The values at each electrode are interpolated on an outline of a head using an electrode
    position file (loc file).

    Parameters
    ----------
    z: np.ndarray
        A vector of electrode values, e.g. a spatial filter/patterns.
    locfile: str
        A .loc file with electrode position information.
    cbar: bool (default: False)
        Whether or not to add a colorbar.
    ax: matplotlib.axes.Axes (default: None)
        Axes to plot in. A new one is made when None.
    iso: bool (default: False)
        Whether or not to add iso lines.
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
    z = np.concatenate((z, np.zeros((4,))), axis=None)

    # Make grid
    N = 300
    xi = np.linspace(-2, 2, N)
    yi = np.linspace(-2, 2, N)
    zi = griddata((xy[:, 0], xy[:, 1]), z, (xi[None, :], yi[:, None]), method="cubic", fill_value=np.nan)

    # Set points outside radius to nan so they will not be plotted.
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
