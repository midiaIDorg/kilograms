import itertools
import numpy as np
import numpy.typing as npt
import numba
import matplotlib.pyplot as plt
import pandas as pd


# ToDo: the float can be a generic numeric type...

@numba.jit(nopython=True)
def min_max(xx: npt.NDArray[float]) -> tuple[float]:
    """Establish the extent of data (minimal and maximal values).

    Arguments:
        xx (np.array): Values that are searched for minimum and maximum.

    Returns:
        tuple: The minimal and the maximal value.
    """
    xx_min = np.inf 
    xx_max =-np.inf
    for x in xx:
        xx_min = min(xx_min, x)
        xx_max = max(xx_max, x)
    return (xx_min,xx_max)


@numba.jit(nopython=True)
def histogram1D(
    xx: npt.NDArray[float],
    extent: tuple[float],
    bins: int
) -> npt.NDArray[np.uint32]:
    """Make a 1D histogram.

    Arguments:
        xx (np.array): Data to bin.
        extent (tuple)

    Returns:
        np.array: The counts of data in the evenly spaced bins.
    """
    xx_min, xx_max = extent
    mult = bins / (xx_max - xx_min)
    result = np.zeros(bins+1, dtype=np.uint32)# +1 for as max guardians
    for x in xx:
        idx = (x-xx_min)*mult
        result[int(idx)] += 1
    result[bins-1] += result[bins]
    return result[:-1]


@numba.jit(nopython=True, cache=True)
def histogram2D(
    xx: npt.NDArray[float],
    yy: npt.NDArray[float], 
    extent: tuple[tuple[float]],
    bins: tuple[int],
):
    """Make a 2D histogram.

    Arguments:
        xx (np.array): First coordinates of points to bin.
        yy (np.array): Second coordinates of points to bin.
        extent (tuple of tuples): A tuple of tuples with the extent of data, first in xx, then in yy. 
        bins (tuple of ints): The number of bins in xx and in yy.
    Returns:
        np.array: The counts of data in the evenly spaced bins.
    """
    (xx_min, xx_max), (yy_min,yy_max) = extent
    xx_bins, yy_bins = bins
    xx_mult = xx_bins / (xx_max - xx_min)
    yy_mult = yy_bins / (yy_max - yy_min)
    result = np.zeros((xx_bins+1, yy_bins+1), dtype=np.uint32)# +1 for as max guardians
    for x,y in zip(xx,yy):
        result[int((x-xx_min)*xx_mult), int((y-yy_min)*yy_mult)] += 1
    result[ -2, -2] = result[ -1, -1]
    result[:-1, -2] = result[:-1, -1]
    result[ -2,:-1] = result[ -1,:-1]
    return result[:-1,:-1]


def get_1D_marginals(
    df: pd.DataFrame,
    bins: dict[str,int],
    extents: dict[str,tuple[float]],
) -> dict[tuple[str], npt.NDArray[np.uint32]]:
    """Get all 1D marginal histograms for a data frame.

    Arguments:
        df (pd.DataFrame): A data-frame with unique column names.
        bins (dict of tuple of ints): Maps column name to number of bins.
        extents (dict of tuples of floats): Maps column name to dimension extent (min, max).

    Returns:
        dict: The counts of data in the evenly spaced bins.
    """
    return {
        _col: histogram1D(
            xx=df[_col].to_numpy(),
            extent=extents[_col], 
            bins=bins[_col]
        )
        for _col in df
    }


def get_2D_marginals(
    df: pd.DataFrame,
    bins: dict[str,int],
    extents: dict[str,tuple[float]],
) -> dict[tuple[str], npt.NDArray[np.uint32]]:
    """Get all 2D marginal 2D histograms for a data frame.

    Arguments:
        df (pd.DataFrame): A data-frame with unique column names.
        bins (dict of tuple of ints): Maps column name to number of bins.
        extents (dict of tuples of floats): Maps column name to dimension extent (min, max).

    Returns:
        dict: The counts of data in the evenly spaced bins.
    """
    return {
        (c0, c1): histogram2D(
            df[c0].to_numpy(),
            df[c1].to_numpy(),
            extent=(extents[c0], extents[c1]),
            bins=(bins[c0], bins[c1]),
        )
        for c0, c1 in itertools.combinations(df, r=2)
    }


def scatterplot_matrix(
    df: pd.DataFrame,
    bins: dict|int=100,
    imshow_kwargs: dict={"cmap":"inferno"},
    plot_kwargs: dict={},
    show: bool=True,
    y_labels_offset: float=-0.02,
    **kwargs,
) -> None:
    """
    Make a scatterplot matrix with 1D histograms on the diagonal and 2D histograms off the diagonal to quickly summarize a data-frame with numeric values.

    Arguments:
        df (pd.DataFrame): A data-frame with values to summarize.
        bins (dict or int): Either a number of bins in each dimensions, or a mapping between the column name and the number of bins.
        imshow_kwargs (dict): Keyword arguments to the off-diagonal plots.
        plot_kwargs (dict): Keyword arguments to the diagonal plots.
        show (bool): Show the canvas immediately.
        y_labels_offset (float): A distance of the y labels from the axis.
        **kwargs: Other keyword arguments to the plt.subplots function.
    """
    if isinstance(bins, int):
        bins = {col: bins for col in df}    
    assert set(bins) == set(df.columns), "The keys of bins have to be the same as the columns in the data frame."

    extents = {col: min_max(df[col].to_numpy()) for col in df}
    marginals1D = get_1D_marginals(df, bins=bins, extents=extents)
    marginals2D = get_2D_marginals(df, bins=bins, extents=extents)

    bin_borders = {
        col: np.linspace( extents[col][0], extents[col][1], bins[col]+1 )
        for col in df
    }
    bin_centers = {
        col: (bin_border[:-1] + bin_border[1:]) / 2.0
        for col, bin_border in bin_borders.items()      
    }

    N = len(df.columns)
    fig, axs = plt.subplots(nrows=N, ncols=N, squeeze=True, **kwargs)
    limits = {}
    for i, c0 in enumerate(df):
        for j, c1 in enumerate(df):
            ax = axs[i,j]
            if c0 != c1:
                try:
                    counts2D = marginals2D[(c0,c1)]
                except KeyError:
                    counts2D = marginals2D[(c1,c0)].T
                ax.imshow(
                    counts2D,
                    extent=extents[c1]+extents[c0],
                    origin="lower",
                    aspect="auto",
                    **imshow_kwargs
                )  
                limits[c1] = ax.get_xlim()
            if j==0:
                ax.set_ylabel(c0)
            if i==0:
                ax.set_title(c1)
    for i,col in enumerate(df):
        ax = axs[i,i]
        ax.plot(
            bin_centers[col],
            marginals1D[col],
            **plot_kwargs,
        )
        left, right = limits[col]
        ax.set_xlim(left, right, auto=True)

    labelx = y_labels_offset  # axes coords
    for j in range(N):
        axs[j,0].yaxis.set_label_coords(labelx, 0.5)

    if show:
        fig.show()

