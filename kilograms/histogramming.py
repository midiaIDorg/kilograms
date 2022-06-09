import itertools
import numpy as np
import numba
import matplotlib.pyplot as plt
import pandas as pd


@numba.jit(nopython=True)
def min_max(xx):
    xx_min = np.inf 
    xx_max =-np.inf
    for x in xx:
        xx_min = min(xx_min, x)
        xx_max = max(xx_max, x)
    return (xx_min,xx_max)


@numba.jit(nopython=True)
def histogram1D(xx, extent, bins):
    xx_min, xx_max = extent
    mult = bins / (xx_max - xx_min)
    result = np.zeros(bins+1, dtype=np.uint32)# +1 for as max guardians
    for x in xx:
        idx = (x-xx_min)*mult
        result[int(idx)] += 1
    result[bins-1] += result[bins]
    return result[:-1]


@numba.jit(nopython=True, cache=True)
def histogram2D(xx, yy, extent, bins):
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


def get_1D_marginals(df, bins, extents):
    return {
        _col: histogram1D(
            xx=df[_col].to_numpy(),
            extent=extents[_col], 
            bins=bins[_col]
        )
        for _col in df
    }


def get_2D_marginals(df, bins, extents):
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
    if isinstance(bins, int):
        bins = {col: bins for col in df}
    
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

