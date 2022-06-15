from __future__ import annotations
import itertools
import numpy as np
import numpy.typing as npt
import numba
import matplotlib.pyplot as plt
import pandas as pd


# ToDo:
# 

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


def get_max_extent(*extents):
    """Given a sequence of (min,max) tuples, find the min of mins and the max of maxes."""
    mins, maxes = zip(*extents)
    return (min(mins), max(maxes))


def histogram1D_slow(
    xx: npt.NDArray[float],
    extent: tuple[float],
    bins: int
) -> npt.NDArray[np.uint32]:
    xx_min, xx_max = extent
    mult = bins / (xx_max - xx_min)
    result = np.zeros(bins+1, dtype=np.uint32)# +1 for as max guardians
    for x in xx:
        idx = (x-xx_min)*mult
        result[int(idx)] += 1
    result[bins-1] += result[bins]
    return result[:-1]

histogram1D_modes = {
    "slow": histogram1D_slow,
    "fast": numba.jit(nopython=True, cache=True)(histogram1D_slow),
    "safe": numba.jit(nopython=True, cache=True, boundscheck=True)(histogram1D_slow),
}

def histogram1D(
    xx: npt.NDArray[float],
    extent: tuple[float],
    bins: int,
    mode: str="fast",
) -> npt.NDArray[np.uint32]:
    """Make a 1D histogram.

    Arguments:
        xx (np.array): Data to bin.
        extent (tuple): Min and max values of the data.
        bins (int): The number of bins.
        mode (str): The mode of carrying out the comptutatoins: 'fast', 'safe', or 'slow'.

    Returns:
        np.array: The counts of data in the evenly spaced bins.
    """
    return histogram1D_modes[mode](xx, extent, bins)


def weighted_histogram1D_slow(
    xx: npt.NDArray[float],
    weights: npt.NDArray[float],
    extent: tuple[float],
    bins: int
) -> npt.NDArray[np.uint32]:
    xx_min, xx_max = extent
    mult = bins / (xx_max - xx_min)
    result = np.zeros(bins+1, dtype=np.uint32)# +1 for as max guardians
    for x,w in zip(xx,weights):
        idx = (x-xx_min)*mult
        result[int(idx)] += w
    result[bins-1] += result[bins]
    return result[:-1]


weighted_histogram1D_modes = {
    "slow": weighted_histogram1D_slow,
    "fast": numba.jit(nopython=True, cache=True)(weighted_histogram1D_slow),
    "safe": numba.jit(nopython=True, cache=True, boundscheck=True)(weighted_histogram1D_slow),
}


def weighted_histogram1D(
    xx: npt.NDArray[float],
    weights: npt.NDArray[float],
    extent: tuple[float],
    bins: int,
    mode: str="fast",
) -> npt.NDArray[np.uint32]:
    """Make a 1D histogram weighted by weights.

    Arguments:
        xx (np.array): Data to bin.
        weights (np.array): Typically nonnnegative weights.
        extent (tuple): Min and max values of the data.
        bins (int): The number of bins.
        mode (str): The mode of carrying out the comptutatoins: 'fast', 'safe', or 'slow'.

    Returns:
        np.array: The counts of data in the evenly spaced bins.
    """
    return weighted_histogram1D_modes[mode](xx, weights, extent, bins)



def histogram2D_slow(
    xx: npt.NDArray[float],
    yy: npt.NDArray[float], 
    extent: tuple[tuple[float]],
    bins: tuple[int],
) -> npt.NDArray[float]:
    (xx_min, xx_max), (yy_min,yy_max) = extent
    xx_bins, yy_bins = bins
    xx_mult = xx_bins / (xx_max - xx_min)
    yy_mult = yy_bins / (yy_max - yy_min)
    result = np.zeros((xx_bins+1, yy_bins+1), dtype=np.uint32)# +1 for as max guardians
    for x,y in zip(xx,yy):
        result[int((x-xx_min)*xx_mult), int((y-yy_min)*yy_mult)] += 1
    result[ -2, -2] += result[ -1, -1]
    result[:-1, -2] += result[:-1, -1]
    result[ -2,:-1] += result[ -1,:-1]
    return result[:-1,:-1]


histogram2D_modes = {
    "slow": histogram2D_slow,
    "fast": numba.jit(nopython=True, cache=True)(histogram2D_slow),
    "safe": numba.jit(nopython=True, cache=True, boundscheck=True)(histogram2D_slow),
}


def histogram2D(
    xx: npt.NDArray[float],
    yy: npt.NDArray[float], 
    extent: tuple[tuple[float]],
    bins: tuple[int],
    mode: str="fast",
) -> npt.NDArray[float]:
    """Make a 2D histogram.

    Arguments:
        xx (np.array): First coordinates of points to bin.
        yy (np.array): Second coordinates of points to bin.
        extent (tuple of tuples): A tuple of tuples with the extent of data, first in xx, then in yy. 
        bins (tuple of ints): The number of bins in xx and in yy.
        mode (str): The mode of carrying out the comptutatoins: 'fast', 'safe', or 'slow'.

    Returns:
        np.array: The counts of data in the evenly spaced bins.
    """
    return histogram2D_modes[mode](xx, yy, extent, bins)



def weighted_histogram2D_slow(
    xx: npt.NDArray[float],
    yy: npt.NDArray[float],
    weights: npt.NDArray[float],
    extent: tuple[tuple[float]],
    bins: tuple[int],
) -> npt.NDArray[float]:
    (xx_min, xx_max), (yy_min,yy_max) = extent
    xx_bins, yy_bins = bins
    xx_mult = xx_bins / (xx_max - xx_min)
    yy_mult = yy_bins / (yy_max - yy_min)
    result = np.zeros((xx_bins+1, yy_bins+1), dtype=np.double)# +1 for as max guardians
    for x,y,w in zip(xx,yy,weights):
        result[int((x-xx_min)*xx_mult), int((y-yy_min)*yy_mult)] += w
    result[ -2, -2] += result[ -1, -1]
    result[:-1, -2] += result[:-1, -1]
    result[ -2,:-1] += result[ -1,:-1]
    return result[:-1,:-1]


weighted_histogram2D_modes = {
    "slow": weighted_histogram2D_slow,
    "fast": numba.jit(nopython=True, cache=True)(weighted_histogram2D_slow),
    "safe": numba.jit(nopython=True, cache=True, boundscheck=True)(weighted_histogram2D_slow),
}


def weighted_histogram2D(
    xx: npt.NDArray[float],
    yy: npt.NDArray[float],
    weights: npt.NDArray[float],
    extent: tuple[tuple[float]],
    bins: tuple[int],
    mode: str="fast",
) -> npt.NDArray[float]:
    """Make a 2D histogram.

    Arguments:
        xx (np.array): First coordinates of points to bin.
        yy (np.array): Second coordinates of points to bin.
        weights (np.array): Typically nonnnegative weights.
        extent (tuple of tuples): A tuple of tuples with the extent of data, first in xx, then in yy. 
        bins (tuple of ints): The number of bins in xx and in yy.
        mode (str): The mode of carrying out the comptutatoins: 'fast', 'safe', or 'slow'.

    Returns:
        np.array: The counts of data in the evenly spaced bins.
    """
    return weighted_histogram2D_modes[mode](xx, yy, weights, extent, bins)



def kilogram2D(
    xx: npt.NDArray[float],
    yy: npt.NDArray[float], 
    weights: npt.NDArray[float]|None=None,
    bins: tuple[int]=100,
    extent: tuple[tuple[float]]|None=None,
    mode: str="fast",
) -> npt.NDArray[float]:
    """Make a 2D kilogram-histogram.

    A convenience wrapper around histogram2D.

    Arguments:
        xx (np.array): First coordinates of points to bin.
        yy (np.array): Second coordinates of points to bin.
        weights (np.array): Typically nonnnegative weights.
        extent (tuple of tuples): A tuple of tuples with the extent of data, first in xx, then in yy. 
        bins (tuple of ints): The number of bins in xx and in yy.
        mode (str): The mode of carrying out the comptutatoins: 'fast', 'safe', or 'slow'.

    Returns:
        np.array: The counts of data in the evenly spaced bins.
    """
    assert len(xx) == len(yy), "xx and yy must have the same length."
    xx = np.array(xx)# this makes a copy
    yy = np.array(yy)# this makes a copy
    if extent is None:
        extent = (min_max(xx), min_max(yy))
    if isinstance(bins, int):
        bins = (bins, bins)
    if weights is None:
        return histogram2D(xx, yy, extent=extent, bins=bins, mode=mode)
    assert len(weights) == len(xx), "xx, yy, and weights must have the same length."
    weights = np.array(weights)
    return weighted_histogram2D(xx, yy, weights, extent=extent, bins=bins, mode=mode)


def get_1D_marginals(
    df: pd.DataFrame,
    bins: dict[str,int],
    extents: dict[str,tuple[float]],
    weights: pd.Series|None = None,
    mode: str="fast",
) -> dict[tuple[str], npt.NDArray[np.uint32]]:
    """Get all 1D marginal histograms for a data frame.

    Arguments:
        df (pd.DataFrame): A data-frame with unique column names.
        bins (dict of tuple of ints): Maps column name to number of bins.
        extents (dict of tuples of floats): Maps column name to dimension extent (min, max).
        weights (pd.Series or None): Typically nonnnegative weights.
        mode (str): The mode of carrying out the comptutatoins: 'fast', 'safe', or 'slow'.

    Returns:
        dict: The counts of data in the evenly spaced bins.
    """
    if weights is None:
        return {
            _col: histogram1D(
                xx=df[_col].to_numpy(),
                extent=extents[_col], 
                bins=bins[_col],
                mode=mode,
            )
            for _col in df
        }
    else:
        assert len(df)==len(weights), "There must be the same number of weights as rows in the 'df'."
        return {
            _col: weighted_histogram1D(
                xx=df[_col].to_numpy(),
                weights=weights.to_numpy(),
                extent=extents[_col], 
                bins=bins[_col],
                mode=mode,
            )
            for _col in df
        }


def get_2D_marginals(
    df: pd.DataFrame,
    bins: dict[str,int],
    extents: dict[str,tuple[float]],
    weights: pd.Series|None = None,
    mode: str="fast",
) -> dict[tuple[str], npt.NDArray[np.uint32]]:
    """Get all 2D marginal 2D histograms for a data frame.

    Arguments:
        df (pd.DataFrame): A data-frame with unique column names.
        bins (dict of tuple of ints): Maps column name to number of bins.
        extents (dict of tuples of floats): Maps column name to dimension extent (min, max).
        weights (pd.Series or None): Typically nonnnegative weights.
        mode (str): The mode of carrying out the comptutatoins: 'fast', 'safe', or 'slow'.

    Returns:
        dict: The counts of data in the evenly spaced bins.
    """
    if weights is None:
        return {
            (c0, c1): histogram2D(
                df[c0].to_numpy(),
                df[c1].to_numpy(),
                extent=(extents[c0], extents[c1]),
                bins=(bins[c0], bins[c1]),
                mode=mode,
            )
            for c0, c1 in itertools.combinations(df, r=2)
        }
    else:
        assert len(df)==len(weights), "There must be the same number of weights as rows in the 'df'."
        return {
            (c0, c1): weighted_histogram2D(
                df[c0].to_numpy(),
                df[c1].to_numpy(),
                extent=(extents[c0], extents[c1]),
                bins=(bins[c0], bins[c1]),
                weights=weights.to_numpy(),
                mode=mode,
            )
            for c0, c1 in itertools.combinations(df, r=2)
        }


def scatterplot_matrix(
    df: pd.DataFrame,
    weights: pd.Series|None=None,
    bins: dict|int=100,
    extents: dict|None=None,
    imshow_kwargs: dict={"cmap":"inferno"},
    plot_kwargs: dict={},
    show: bool=True,
    mode: str="fast",
    # y_labels_offset: float=-0.02,
    **kwargs,
) -> None:
    """
    Make a scatterplot matrix with 1D histograms on the diagonal and 2D histograms off the diagonal to quickly summarize a data-frame with numeric values.

    Arguments:
        df (pd.DataFrame): A data-frame with values to summarize.
        weights (pd.Series or None): Typically nonnnegative weights.
        bins (dict or int): Either a number of bins in each dimensions, or a mapping between the column name and the number of bins.
        extents (dict of tuples of floats or None): Maps column name to dimension extent (min, max). If not provided (None), will be calculated.
        imshow_kwargs (dict): Keyword arguments to the off-diagonal plots.
        plot_kwargs (dict): Keyword arguments to the diagonal plots.
        show (bool): Show the canvas immediately.
        mode (str): The mode of carrying out the comptutatoins: 'fast', 'safe', or 'slow'.

        **kwargs: Other keyword arguments to the plt.subplots function.
    """
    if isinstance(bins, int):
        bins = {col: bins for col in df}    
    assert set(bins) == set(df.columns), "The keys of bins have to be the same as the columns in the data frame."
    
    if extents is None:
        extents = {col: min_max(df[col].to_numpy()) for col in df}
    assert set(extents) == set(df.columns), "The keys of extents have to be the same as the columns in the data frame."

    bin_borders = {
        col: np.linspace( extents[col][0], extents[col][1], bins[col]+1 )
        for col in df
    }
    bin_centers = {
        col: (bin_border[:-1] + bin_border[1:]) / 2.0
        for col, bin_border in bin_borders.items()      
    }
    marginals1D = get_1D_marginals(df, bins=bins, extents=extents, weights=weights, mode=mode)
    if len(df.columns) == 1:
        col = df.columns[0]
        plt.plot(
            bin_centers[col],
            marginals1D[col],
            **plot_kwargs,
        )
        if show:
            plt.show()
    else:
        marginals2D = get_2D_marginals(df, bins=bins, extents=extents, weights=weights, mode=mode)
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
                    ax.set_ylabel(c0, loc="center")
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

        # labelx = y_labels_offset  # axes coords
        # for j in range(N):
        #     pass
        #     # axs[j,0].yaxis.set_label_coords(labelx, 0.5)
        #     # axs[j,0].yaxis.set_ylabel()
        if show:
            fig.show()
    return fig, axs


def crossdata(
    df: pd.DataFrame,
    yy: pd.Series,
    bins: dict[str,int],
    extents: dict[str,tuple[float]],
    weights: pd.Series|None=None,
    mode: str="fast",
) -> dict[npt.NDArray[np.uint32]]:
    """
    Get 2D histograms summarizing the distributions of columns of 'df' and the 'yy' series. 
    
    Warning: all of the names should be unique, so that columns of df and the name of the yy series should not repeat.

    Arguments:
        df (pd.DataFrame): A data-frame with values to summarize.
        yy (pd.Series): A series with data to cross-histogramize with columns of 'df'.
        bins (dict): A dictionary of bin sizes per each dimension. Must also contain a value for the 'yy.name'.
        extents (dict of tuples of floats): Maps column name to dimension extent (min, max).  Must also contain a value for the 'yy.name'.
        weights (pd.Series or None): An optional series of weights.
        mode (str): The mode of carrying out the comptutatoins: 'fast', 'safe', or 'slow'.

    Returns:
        dict: Mapping between tuples of `df` column name and `yy.name` and the calculated binnings.
    """
    if weights is None:
        return {
            (col, yy.name): histogram2D(
                xx=df[col].to_numpy(),
                yy=yy.to_numpy(),
                extent=(extents[col], extents[yy.name]),
                bins=(bins[col], bins[yy.name]),
                mode=mode,
            )
            for col in df
        }
    else:
        return {
            (col, yy.name): weighted_histogram2D(
                xx=df[col].to_numpy(),
                yy=yy.to_numpy(),
                extent=(extents[col], extents[yy.name]),
                bins=(bins[col], bins[yy.name]),
                weights=weights.to_numpy(),
                mode=mode,
            ) 
            for col in df
        }



def crossplot(
    df: pd.DataFrame,
    yy: pd.Series,
    weights: pd.Series|None=None,
    bins: dict|int=100,
    extents: dict|None=None,
    imshow_kwargs: dict={"cmap":"inferno"},
    show: bool=True,
    nrows: int=1,
    ncols: int|None=None,
    mode: str="fast",
    **kwargs,
) -> None:
    """
    Get a plot with a series of 2D histograms summarizing the distributions of columns of 'df' and the 'yy' series. 
    
    Warning: all of the names should be unique, so that columns of df and the name of the yy series should not repeat.

    Arguments:
        df (pd.DataFrame): A data-frame with values to summarize.
        yy (pd.Series): A series with data to cross-histogramize with columns of 'df'.
        weights (pd.Series or None): Typically nonnnegative weights.
        bins (dict): A dictionary of bin sizes per each dimension. Must also contain a value for the 'yy.name'.
        extents (dict of tuples of floats): Maps column name to dimension extent (min, max).  Must also contain a value for the 'yy.name'.
        imshow_kwargs (dict): Keyword arguments to the off-diagonal plots.
        show (bool): Show the canvas immediately.
        nrows (int): Number of rows in the plot.
        ncols (int): Number of columns in the plot.
        mode (str): The mode of carrying out the comptutatoins: 'fast', 'safe', or 'slow'.
        **kwargs: Other keyword arguments to the plt.subplots function.

    Returns:
        dict: Mapping between tuples of `df` column name and `yy.name` and the calculated binnings.
    """
    assert len(set([*df.columns, yy.name])) == len(df.columns) + 1, "Name of columns of 'df' and that of 'yy' must be unique."

    if isinstance(bins, int):
        _bins = {col: bins for col in df}
        _bins[yy.name] = bins
        bins = _bins
    assert set(bins) == (set(df.columns) | set([yy.name])), "The keys of bins have to be the same as the columns in the data frame."

    if ncols is None:
        ncols = len(df.columns)
    assert nrows * ncols >= len(df.columns), "nrows*ncols must be greater or equal to the number of columns in 'df'."

    if extents is None:
        extents = {col: min_max(df[col].to_numpy()) for col in df}
        extents[yy.name] = min_max(yy.to_numpy())

    cd = crossdata(
        df=df,
        yy=yy,
        bins=bins,
        extents=extents,
        weights=weights,
        mode=mode,
    )

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True, **kwargs)
    axes = iter(axs.flatten())
    for ax, col in zip(axes, df):
        ax.imshow(
            cd[(col, yy.name)],
            extent=extents[col] + extents[yy.name],
            origin="lower",
            aspect="auto",
            **imshow_kwargs
        )
        ax.set_title(col, loc="center")
        ax.set_ylabel(yy.name, loc="center")
    if show:
        fig.show()

    return fig, axs


