import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .histogramming import scatterplot_matrix


def simple_kilogram_plot(
    data,
    show=False,
    path="",
    y_hist_bottom_lim=0,
    title="",
    width=10,
    height=10,
    dpi=100,
    style="dark_background",
    **kwargs,
):
    kwargs["y_hist_bottom_lim"] = y_hist_bottom_lim
    with plt.style.context(style):
        fig, axes = scatterplot_matrix(data, **kwargs, show=False)
        if title:
            plt.suptitle(title)
        fig.set_size_inches(width, height)
        if path != "":
            plt.savefig(path, dpi=dpi)
        if show:
            plt.show()
        if path != "" or show:
            plt.close()
        return fig, axes


def plot_histograms(
    bins,
    path="",
    title="",
    y_hist_bottom_lim=0,
    width=10,
    height=10,
    dpi=100,
    style="dark_background",
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
    grid: pd.DataFrame, fig, axes, show: bool = True, **kwargs
) -> None:
    for (i, dim_i), (j, dim_j) in itertools.combinations(
        enumerate(grid.columns),
        r=2,
    ):
        axes[i, j].scatter(grid[dim_j], grid[dim_i], **kwargs)
    if show:
        fig.show()
