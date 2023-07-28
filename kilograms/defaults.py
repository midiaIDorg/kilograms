import matplotlib.pyplot as plt
import numpy as np

from .histogramming import scatterplot_matrix


def simple_kilogram_plot(
    data,
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
        fig, axes = scatterplot_matrix(data, **kwargs)
        if title:
            plt.suptitle(title)
        fig.set_size_inches(width, height)
        if path == "":
            plt.show()
        else:
            plt.savefig(path, dpi=dpi)
            plt.close()


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
