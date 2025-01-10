from __future__ import annotations

import collections
import itertools
import math
import multiprocessing
import typing
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit

# TODO:
# get a script that accepts calculated data and plots those
# likely a class that consumes inputs and makes histograms on the flight: map-reduce???
# also, there is quite a lot of spaghetti code below
# and missing usages of types


@njit
def min_max(xx: npt.NDArray) -> tuple[float, float]:
    """Establish the extent of data (minimal and maximal values).

    Arguments:
        xx (np.array): Values that are searched for minimum and maximum.

    Returns:
        tuple: The minimal and the maximal value.
    """
    xx_min = np.inf
    xx_max = -np.inf
    for x in xx:
        xx_min = min(xx_min, x)
        xx_max = max(xx_max, x)
    return (xx_min, xx_max)


def get_max_extent(*extents):
    """Given a sequence of (min,max) tuples, find the min of mins and the max of maxes."""
    mins, maxes = zip(*extents)
    return (min(mins), max(maxes))


def histogram1D_slow(
    xx: npt.NDArray, extent: tuple[float, float], bins: int
) -> npt.NDArray[np.uint32]:
    xx_min, xx_max = extent
    mult = bins / (xx_max - xx_min)
    result = np.zeros(bins + 1, dtype=np.uint32)  # +1 for as max guardians
    for x in xx:
        idx = (x - xx_min) * mult
        result[int(idx)] += 1
    result[bins - 1] += result[bins]
    return result[:-1]


# todo: there is some copy-paste code to execute with another loop
histogram1D_modes = {
    "slow": histogram1D_slow,
    "fast": njit(cache=True)(histogram1D_slow),
    "safe": njit(cache=True, boundscheck=True)(histogram1D_slow),
}


def histogram1D(
    xx: npt.NDArray,
    extent: tuple[float],
    bins: int,
    mode: str = "safe",
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
    xx: npt.NDArray,
    weights: npt.NDArray,
    extent: tuple[float],
    bins: int,
) -> npt.NDArray[np.uint32]:
    xx_min, xx_max = extent
    mult = bins / (xx_max - xx_min)
    result = np.zeros(bins + 1, dtype=np.uint32)  # +1 for as max guardians
    for x, w in zip(xx, weights):
        idx = (x - xx_min) * mult
        result[int(idx)] += w
    result[bins - 1] += result[bins]
    return result[:-1]


weighted_histogram1D_modes = {
    "slow": weighted_histogram1D_slow,
    "fast": njit(cache=True)(weighted_histogram1D_slow),
    "safe": njit(cache=True, boundscheck=True)(weighted_histogram1D_slow),
}


def weighted_histogram1D(
    xx: npt.NDArray,
    weights: npt.NDArray,
    extent: tuple[float],
    bins: int,
    mode: str = "safe",
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


def discrete_histogram1D_slow(xx):
    _min, _max = min_max(xx)
    _min = np.intp(_min)
    _max = np.intp(_max)
    hist = np.zeros(dtype=np.uintp, shape=_max - _min + 1)
    for i, x in enumerate(xx):
        hist[x - _min] += 1
    return np.arange(_min, _max + 1), hist


discrete_histogram1D = njit(discrete_histogram1D_slow)


def histogram2D_slow(
    xx: npt.NDArray,
    yy: npt.NDArray,
    extent: tuple[tuple[float, float], tuple[float, float]],
    bins: tuple[int, int],
) -> npt.NDArray:
    (xx_min, xx_max), (yy_min, yy_max) = extent
    xx_bins, yy_bins = bins
    xx_mult = xx_bins / (xx_max - xx_min)
    yy_mult = yy_bins / (yy_max - yy_min)
    result = np.zeros(
        (xx_bins + 1, yy_bins + 1), dtype=np.uint32
    )  # +1 for as max guardians
    for x, y in zip(xx, yy):
        result[int((x - xx_min) * xx_mult), int((y - yy_min) * yy_mult)] += 1
    result[-2, -2] += result[-1, -1]
    result[:-1, -2] += result[:-1, -1]
    result[-2, :-1] += result[-1, :-1]
    return result[:-1, :-1]


histogram2D_modes = {
    "slow": histogram2D_slow,
    "fast": njit(cache=True)(histogram2D_slow),
    "safe": njit(cache=True, boundscheck=True)(histogram2D_slow),
}


def histogram2D(
    xx: npt.NDArray,
    yy: npt.NDArray,
    extent: tuple[tuple[float, float], tuple[float, float]],
    bins: tuple[int, int],
    mode: str = "fast",
) -> npt.NDArray:
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
    xx: npt.NDArray,
    yy: npt.NDArray,
    weights: npt.NDArray,
    extent: tuple[tuple[float, float], tuple[float, float]],
    bins: tuple[int, int],
) -> npt.NDArray:
    (xx_min, xx_max), (yy_min, yy_max) = extent
    xx_bins, yy_bins = bins
    xx_mult = xx_bins / (xx_max - xx_min)
    yy_mult = yy_bins / (yy_max - yy_min)
    result = np.zeros(
        (xx_bins + 1, yy_bins + 1), dtype=np.double
    )  # +1 for as max guardians
    for x, y, w in zip(xx, yy, weights):
        result[int((x - xx_min) * xx_mult), int((y - yy_min) * yy_mult)] += w
    result[-2, -2] += result[-1, -1]
    result[:-1, -2] += result[:-1, -1]
    result[-2, :-1] += result[-1, :-1]
    return result[:-1, :-1]


weighted_histogram2D_modes = {
    "slow": weighted_histogram2D_slow,
    "fast": njit(cache=True)(weighted_histogram2D_slow),
    "safe": njit(cache=True, boundscheck=True)(weighted_histogram2D_slow),
}


def weighted_histogram2D(
    xx: npt.NDArray,
    yy: npt.NDArray,
    weights: npt.NDArray,
    extent: tuple[tuple[float, float], tuple[float, float]],
    bins: tuple[int, int],
    mode: str = "fast",
) -> npt.NDArray:
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
    xx: npt.NDArray,
    yy: npt.NDArray,
    weights: npt.NDArray | None = None,
    bins: int | tuple[int, int] = 100,
    extent: tuple[tuple[float, float], tuple[float, float]] | None = None,
    mode: str = "fast",
) -> npt.NDArray:
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
    xx = np.array(xx)  # this makes a copy
    yy = np.array(yy)  # this makes a copy
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
    bins: dict[str, int],
    extents: dict[str, tuple[float]],
    weights: pd.Series | None = None,
    mode: str = "fast",
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
        assert len(df) == len(
            weights
        ), "There must be the same number of weights as rows in the 'df'."
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
    bins: dict[str, int],
    extents: dict[str, tuple[float]],
    weights: pd.Series | None = None,
    mode: str = "fast",
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
        assert len(df) == len(
            weights
        ), "There must be the same number of weights as rows in the 'df'."
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


def unique_columns(columns: typing.Iterable[str]) -> bool:
    return all(v == 1 for v in collections.Counter(columns).values())


def scatterplot_matrix(
    df: pd.DataFrame,
    weights: pd.Series | None = None,
    bins: dict | int = 100,
    extents: dict | None = None,
    imshow_kwargs: dict = {"cmap": "inferno"},
    plot_kwargs: dict = {},
    show: bool = True,
    mode: str = "fast",
    lims: dict | None = None,
    y_hist_bottom_lim: float | None = None,
    common_vlim: bool = False,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, npt.NDArray]:
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
    assert unique_columns(df.columns), "Column names must be unique."
    if isinstance(bins, int):
        bins = {col: bins for col in df}
    assert set(bins) == set(
        df.columns
    ), "The keys of bins have to be the same as the columns in the data frame."

    if extents is None:
        extents = {col: min_max(df[col].to_numpy()) for col in df}
    assert set(extents) == set(
        df.columns
    ), "The keys of extents have to be the same as the columns in the data frame."

    if lims is not None:
        for col in lims:
            assert (
                col in df.columns
            ), "The keys of lims have to be a subset of columns of the dataframe."

    bin_borders = {
        col: np.linspace(extents[col][0], extents[col][1], bins[col] + 1) for col in df
    }
    bin_centers = {
        col: (bin_border[:-1] + bin_border[1:]) / 2.0
        for col, bin_border in bin_borders.items()
    }
    marginals1D = get_1D_marginals(
        df, bins=bins, extents=extents, weights=weights, mode=mode
    )
    vmin = None
    vmax = None
    if common_vlim:
        vmin = 0
        vmax = np.sum(marginals1D[df.columns[0]])

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
        marginals2D = get_2D_marginals(
            df, bins=bins, extents=extents, weights=weights, mode=mode
        )
        N = len(df.columns)
        fig, axs = plt.subplots(nrows=N, ncols=N, squeeze=True, **kwargs)
        limits = {}
        for i, c0 in enumerate(df):
            for j, c1 in enumerate(df):
                ax = axs[i, j]
                if c0 != c1:
                    try:
                        counts2D = marginals2D[(c0, c1)]
                    except KeyError:
                        counts2D = marginals2D[(c1, c0)].T
                    ax.imshow(
                        counts2D,
                        extent=extents[c1] + extents[c0],
                        origin="lower",
                        aspect="auto",
                        vmin=vmin,
                        vmax=vmax,
                        **imshow_kwargs,
                    )
                    limits[c1] = ax.get_xlim()
                    if lims is not None:
                        try:
                            ax.set_ylim(lims[c0])
                        except KeyError:
                            pass
                        try:
                            ax.set_xlim(lims[c1])
                        except KeyError:
                            pass
                if j == 0:
                    ax.set_ylabel(c0, loc="center")
                if i == 0:
                    ax.set_title(c1)
        for i, col in enumerate(df):
            ax = axs[i, i]
            ax.plot(
                bin_centers[col],
                marginals1D[col],
                **plot_kwargs,
            )
            left, right = limits[col]
            if lims is not None:
                try:
                    left, right = lims[col]
                except KeyError:
                    pass
            ax.set_xlim(left, right, auto=True)
            if y_hist_bottom_lim is not None:
                ax.set_ylim(bottom=y_hist_bottom_lim)

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
    bins: dict[str, int],
    extents: dict[str, tuple[float]],
    weights: pd.Series | None = None,
    mode: str = "fast",
) -> dict[str, npt.NDArray[np.uint32]]:
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
    assert unique_columns(df.columns), "Column names must be unique."
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
    weights: pd.Series | None = None,
    bins: dict | int = 100,
    extents: dict | None = None,
    imshow_kwargs: dict = {"cmap": "inferno"},
    show: bool = True,
    nrows: int = 1,
    ncols: int | None = None,
    mode: str = "fast",
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
    assert unique_columns(df.columns), "Column names must be unique."
    assert (
        len(set([*df.columns, yy.name])) == len(df.columns) + 1
    ), "Name of columns of 'df' and that of 'yy' must be unique."

    if isinstance(bins, int):
        _bins = {col: bins for col in df}
        _bins[yy.name] = bins
        bins = _bins
    assert set(bins) == (
        set(df.columns) | set([yy.name])
    ), "The keys of bins have to be the same as the columns in the data frame."

    if ncols is None:
        ncols = len(df.columns)
    assert nrows * ncols >= len(
        df.columns
    ), "nrows*ncols must be greater or equal to the number of columns in 'df'."

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
            **imshow_kwargs,
        )
        ax.set_title(col, loc="center")
        ax.set_ylabel(yy.name, loc="center")
    if show:
        fig.show()

    return fig, axs


def get_scatterplot_data(
    df: pd.DataFrame,
    bin_cnt: int = 100,
    _extent_mult: float = 0.01,
) -> dict:
    data = {c: df[c].to_numpy() for c in df.columns}
    get_bin_centers = lambda xx: (xx[1:] + xx[:-1]) / 2.0

    dim_bins = {}
    bin_centers = {}
    counts1D = {}
    for column in data:
        try:
            _min, _max = min_max(data[column])
            _extent = _max - _min
            _min -= _extent * _extent_mult
            _max += _extent * _extent_mult
            dim_bins[column] = np.linspace(_min, _max, bin_cnt + 1)
            bin_centers[column] = get_bin_centers(dim_bins[column])
            counts1D[column] = histogram1D(
                xx=data[column],
                extent=(_min, _max),
                bins=bin_cnt,
            )
        except IndexError as exc:
            print(column)
            raise exc
    counts2D = {}
    for column_hor, column_ver in itertools.combinations(data, r=2):
        counts2D[(column_hor, column_ver)] = histogram2D(
            xx=data[column_hor],
            yy=data[column_ver],
            extent=(dim_bins[column_hor][[0, -1]], dim_bins[column_ver][[0, -1]]),
            bins=(bin_cnt, bin_cnt),
        )

    return dict(
        counts1D=counts1D,
        counts2D=counts2D,
        dim_bins=dim_bins,
        bin_centers=bin_centers,
    )


def get_grouped_scatterplot_data(
    df: pd.DataFrame,
    groups: list,
    bin_cnt: int = 40,
    _extent_mult: float = 0.01,
    _full_matrix: bool = False,
) -> dict:
    _group_tags = set(groups)

    dim_bins = {}
    counts1D = {g: {} for g in _group_tags}
    bin_centers = {}
    get_bin_centers = lambda xx: (xx[1:] + xx[:-1]) / 2.0

    for column in df.columns:
        _min, _max = min_max(df[column].to_numpy())
        _extent = _max - _min
        _min -= _extent * _extent_mult
        _max += _extent * _extent_mult
        dim_bins[column] = np.linspace(_min, _max, bin_cnt + 1)
        bin_centers[column] = get_bin_centers(dim_bins[column])
        for _group, _xx in df[column].groupby(groups):
            counts1D[_group][column] = histogram1D(
                xx=_xx.to_numpy(),
                extent=(_min, _max),
                bins=bin_cnt,
            )

    counts2D = {g: {} for g in _group_tags}
    for _group, _df in df.groupby(groups):
        for column_hor, column_ver in itertools.combinations(df.columns, r=2):
            counts2D[_group][(column_hor, column_ver)] = histogram2D(
                xx=_df[column_hor].to_numpy(),
                yy=_df[column_ver].to_numpy(),
                extent=(
                    dim_bins[column_hor][[0, -1]],
                    dim_bins[column_ver][[0, -1]],
                ),
                bins=(bin_cnt, bin_cnt),
            )
            if _full_matrix:
                counts2D[_group][(column_ver, column_hor)] = counts2D[_group][
                    (column_hor, column_ver)
                ].T

    return dict(
        counts1D=counts1D,
        counts2D=counts2D,
        dim_bins=dim_bins,
        bin_centers=bin_centers,
    )


# scatterplot_data = get_scatterplot_data(df = edge_stats, bin_cnt = 100)


def save_1D_histograms(
    scatterplot_data: dict,
    folder: str | Path | None = None,
    show: bool = False,
    silent: bool = False,
    prefix: str = "",
    suffix: str = "",
    title: str = "",
    **kwargs,
):
    for column in scatterplot_data["counts1D"]:
        if not silent:
            print(f"Saving `{prefix}{column}{suffix}.pdf`.")
        plt.plot(
            scatterplot_data["bin_centers"][column],
            scatterplot_data["counts1D"][column],
        )
        plt.xlabel(column)
        plt.ylabel("count")
        plt.ylim(bottom=0)
        if title:
            plt.title(title)
        if show:
            plt.show()
        else:
            if folder is not None:
                folder = Path(folder)
                plt.savefig(folder / f"{prefix}{column}{suffix}.pdf", **kwargs)
        plt.close()


# def plot2D_histogram(column_hor, column_ver, )


# speed up su
def save_2D_histograms(
    scatterplot_data: dict,
    folder: str | Path | None = None,
    show: bool = False,
    silent: bool = False,
    prefix: str = "",
    suffix: str = "",
    imshow_kwargs: dict = {},
    both_diagonals: bool = True,
    title: str = "",
    # process_cnt: int = multiprocessing.cpu_count(),
    values_to_nans: float | None = None,
    **kwargs,
):
    # scatterplot_data = _edge_features_aggregates
    # column_hor = "frame_correlation"
    # column_ver = "scan_wasserstein"
    for column_hor, column_ver in scatterplot_data["counts2D"]:
        if not silent:
            print(f"Saving `{prefix}{column_hor}_{column_ver}{suffix}.pdf`.")

        dim_bins = scatterplot_data["dim_bins"]

        X = scatterplot_data["counts2D"][(column_hor, column_ver)].T

        if values_to_nans is not None:
            X = X.astype(float)
            X[X == values_to_nans] = np.nan
        plt.imshow(
            X,
            extent=list(dim_bins[column_hor][[0, -1]])
            + list(dim_bins[column_ver][[0, -1]]),
            origin="lower",
            aspect="auto",
            # **imshow_kwargs,
        )
        plt.xlabel(column_hor)
        plt.ylabel(column_ver)
        if title:
            plt.title(title)
        if show:
            plt.show()
        else:
            if folder is not None:
                folder = Path(folder)
                plt.savefig(
                    folder / f"{prefix}{column_hor}_{column_ver}{suffix}.pdf", **kwargs
                )
        plt.close()

        if both_diagonals:
            plt.imshow(
                X.T,
                extent=list(dim_bins[column_ver][[0, -1]])
                + list(dim_bins[column_hor][[0, -1]]),
                origin="lower",
                aspect="auto",
                **imshow_kwargs,
            )
            plt.xlabel(column_ver)
            plt.ylabel(column_hor)
            if title:
                plt.title(title)
            if show:
                plt.show()
            else:
                if folder is not None:
                    folder = Path(folder)
                    plt.savefig(
                        folder / f"{prefix}{column_ver}_{column_hor}{suffix}.pdf",
                        **kwargs,
                    )
            plt.close()


def two_groups_hist2D(
    intensities_A,
    intensities_B,
    bins_A,
    bins_B,
    labels_A="A",
    labels_B="B",
    title="",
    path=None,
    show=False,
    levels=3,
    vmin=None,
    vmax=None,
    **kwargs,
) -> None:
    if vmin is None:
        vmin = min(intensities_A.min(), intensities_B.min())
    if vmax is None:
        vmax = max(intensities_A.max(), intensities_B.max())
    plt.contourf(
        bins_A,
        bins_B,
        intensities_A.T,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
    )
    plt.contour(
        bins_A,
        bins_B,
        intensities_B.T,
        levels=levels,
        colors="white",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel(labels_A)
    plt.ylabel(labels_B)
    if title:
        plt.title(title)
    if show:
        plt.show()
    if path is not None:
        plt.savefig(path, **kwargs)


def grouped_hist2D(
    intensity_2D_marginals: list[npt.NDArray],
    x_bin_centers: npt.NDArray,
    y_bin_centers: npt.NDArray,
    xlabel: str,
    ylabel: str,
    colors: list[str],
    title: str = "",
    path: str = "",
    show: bool = False,
    levels: int = 3,
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs,
) -> None:
    assert len(colors) + 1 == len(
        intensity_2D_marginals
    ), "All but first set of marginal intensities needs a color to map to."

    if vmin is None:
        vmin = math.inf
        for intensities in intensity_2D_marginals:
            vmin = min(vmin, intensities.min())

    if vmax is None:
        vmax = -math.inf
        for intensities in intensity_2D_marginals:
            vmax = max(vmax, intensities.max())

    intensity_2D_marginals = iter(intensity_2D_marginals)

    plt.contourf(
        x_bin_centers,
        y_bin_centers,
        next(intensity_2D_marginals).T,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
    )

    for intensities, color in zip(intensity_2D_marginals, colors):
        plt.contour(
            x_bin_centers,
            y_bin_centers,
            intensities.T,
            levels=levels,
            colors=color,
            vmin=vmin,
            vmax=vmax,
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if show:
        plt.show()
    if path != "":
        plt.savefig(path, **kwargs)
    plt.close()


def freqpolyplot(
    bin_centers: npt.NDArray,
    group_to_intensities: dict[str, npt.NDArray],
    xlabel: str = "",
    title: str = "",
    path: str = "",
    show: bool = False,
    close: bool = True,
    **kwargs,
) -> None:
    for group, intensities in group_to_intensities.items():
        plt.plot(bin_centers, intensities, label=group)
    plt.legend()
    if xlabel != "":
        plt.xlabel(xlabel)
    plt.ylabel("count")
    if show:
        plt.show()
    if path != "":
        plt.savefig(path, **kwargs)
    if close:
        plt.close()


def hist2D(
    intensities: npt.NDArray,
    extent: tuple[tuple[int]],
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    show: bool = False,
    path: str | None = None,
    value_below_which_nan: float | None = None,
    imshow_kwargs: dict = {},
    **kwargs,
) -> None:
    if value_below_which_nan is not None:
        intensities = intensities.astype(float)
        intensities[intensities <= value_below_which_nan] = np.nan
    plt.imshow(
        intensities.T,
        extent=extent,
        origin="lower",
        aspect="auto",
        **imshow_kwargs,
    )
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if show:
        plt.show()
    if path is not None:
        plt.savefig(path, **kwargs)
    plt.close()
