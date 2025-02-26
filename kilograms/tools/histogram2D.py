import click
import math
import numba
import numpy as np

from itertools import combinations
from opentimspy import OpenTIMS
from pandas_ops.io import read_df
from pathlib import Path
from tqdm import tqdm

from kilograms.histogramming import histogram2D
from kilograms.histogramming import minmax
from kilograms.histogramming import weighted_histogram2D


def floor_round(x, digits):
    return math.floor(x * 10**digits) / 10**digits


def ceil_round(x, digits):
    return math.ceil(x * 10**digits) / 10**digits


@numba.njit
def sorted_isin(to_check, checked_against):
    res = np.empty(shape=len(to_check), dtype=np.bool_)
    i = 0
    j = 0
    while i < len(to_check) and j < len(checked_against):
        curr = checked_against[j]
        x = to_check[i]
        if x < curr:
            res[i] = False
            i += 1
        elif x > curr:
            j += 1
        else:
            res[i] = True
            i += 1
    return res


@click.command(context_settings={"show_default": True})
@click.argument("output_path", type=Path)
@click.argument("dataset_path", type=Path)
@click.argument("memmapped_dataset_path", type=Path)
def raw_data_2D_histograms(
    output_path: Path,
    dataset_path: Path,
    memmapped_dataset_path: Path,
) -> None:
    """Dump histogrammed data to numpy savez format."""
    op = OpenTIMS(dataset_path)
    data = read_df(memmapped_dataset_path)
    weights = data.intensity.to_numpy()

    ms1_frames = sorted_isin(data.frame.to_numpy(), np.sort(op.ms1_frames))
    data_dct = {
        c: data[c].to_numpy() for c in ("retention_time", "inv_ion_mobility", "mz")
    }

    digits = {
        "retention_time": 0,
        "inv_ion_mobility": 1,
        "mz": 0,
    }

    bins = {
        "retention_time": 100,
        "inv_ion_mobility": 100,
        "mz": 100,
    }

    min_maxs = {c: minmax(v) for c, v in data_dct.items()}

    borders = {}
    for c, (_min, _max) in min_maxs.items():
        _min = floor_round(_min, digits[c])
        _max = ceil_round(_max, digits[c])
        borders[c] = np.linspace(_min, _max, bins[c] + 1)

    N = len(data_dct)
    K = N * (N - 1) // 2
    histogrammed_data = {}
    for (c0, xx), (c1, yy) in tqdm(
        combinations(data_dct.items(), r=2),
        total=K,
        desc="Going fru all disctinct pairs of columns.",
    ):
        extent = ((borders[c0][0], borders[c0][-1]), (borders[c1][0], borders[c1][-1]))
        bins_x = bins[c0]
        bins_y = bins[c1]

        xx_ms1 = xx[ms1_frames]
        yy_ms1 = yy[ms1_frames]
        xx_ms2 = xx[~ms1_frames]
        yy_ms2 = yy[~ms1_frames]

        histogrammed_data[f"counts/ms1/{c0}/{c1}"] = histogram2D(
            xx_ms1,
            yy_ms1,
            extent=extent,
            bins=(bins_x, bins_y),
            mode="safe",
        )
        histogrammed_data[f"counts/ms2/{c0}/{c1}"] = histogram2D(
            xx_ms2,
            yy_ms2,
            extent=extent,
            bins=(bins_x, bins_y),
            mode="safe",
        )

        weights_ms1 = weights[ms1_frames]
        weights_ms2 = weights[~ms1_frames]

        histogrammed_data[f"intensity/ms1/{c0}/{c1}"] = weighted_histogram2D(
            xx_ms1,
            yy_ms1,
            weights_ms1,
            extent=extent,
            bins=(bins_x, bins_y),
            mode="safe",
        )
        histogrammed_data[f"intensity/ms2/{c0}/{c1}"] = weighted_histogram2D(
            xx_ms2,
            yy_ms2,
            weights_ms1,
            extent=extent,
            bins=(bins_x, bins_y),
            mode="safe",
        )

    np.savez_compressed(
        output_path,
        **{f"border/{c}": v for c, v in borders.items()},
        **histogrammed_data,
    )
