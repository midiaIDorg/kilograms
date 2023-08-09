import itertools
import pathlib

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from .histogramming import scatterplot_matrix


def simple_kilogram_plot(
    data,
    show: bool = False,
    path: pathlib.Path | str = "",
    y_hist_bottom_lim: float = 0,
    title: str = "",
    width: int = 10,
    height: int = 10,
    dpi: int = 100,
    style: str = "dark_background",
    **kwargs,
) -> tuple[matplotlib.figure.Figure, npt.NDArray]:
    kwargs["y_hist_bottom_lim"] = y_hist_bottom_lim
    with plt.style.context(style):
        fig, axes = scatterplot_matrix(data, **kwargs, show=False)
        if title:
            plt.suptitle(title)
        if show:
            plt.show()
        if path != "":
            fig.set_size_inches(width, height)
            plt.savefig(path, dpi=dpi)
        if path != "" or show:
            plt.close()
        return fig, axes


def plot_histograms(
    bins,
    path: pathlib.Path | str = "",
    title: str = "",
    y_hist_bottom_lim: float = 0,
    width: int = 10,
    height: int = 10,
    dpi: int = 100,
    style: str = "dark_background",
    **data,
):
    mids = (bins[1:] + bins[:-1]) / 2
    for name, xx in data.items():
        counts, _ = np.histogram(xx, bins=bins)
        plt.plot(mids, counts / counts.sum(), label=name)
    plt.legend()
    if title:
        plt.title(title)
    plt.ylim(bottom=y_hist_bottom_lim)
    if path == "":
        plt.show()
    else:
        plt.gcf().set_size_inches(width, height)
        plt.savefig(path, dpi=dpi)
        plt.close()


def overplot_grid_on_half_of_kilogram_scatterplot(
    grid: pd.DataFrame,
    fig,
    axes,
    show: bool = True,
    path: pathlib.Path | str = "",
    width: int = 10,
    height: int = 10,
    dpi: int = 100,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, npt.NDArray]:
    for (i, dim_i), (j, dim_j) in itertools.combinations(
        enumerate(grid.columns),
        r=2,
    ):
        axes[i, j].scatter(grid[dim_j], grid[dim_i], **kwargs)
    if show:
        fig.show()
    if path != "":
        fig.set_size_inches(width, height)
        plt.savefig(path, dpi=dpi)
    if path != "" or show:
        plt.close()
    return fig, axes
