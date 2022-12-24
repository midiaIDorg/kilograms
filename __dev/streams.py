%load_ext autoreload
%autoreload 2
from dataclasses import dataclass
import numpy as np
import functools
import numpy.typing as npt
import numba
import os
import opentimspy
import pandas as pd
import typing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 4)
os.chdir("/home/matteo/Projects/MIDIA/SNAKEMAKE/midia_pipe/")
import collections
from tqdm import tqdm
from dia_common import DiaRun
op = opentimspy.OpenTIMS("data/6533.d")
import matplotlib.gridspec as grid_spec


X = op.query(frames=1, columns=("scan","mz","intensity"))
xx = X["mz"]
yy = X["scan"]
weights = X["intensity"]

x_min = op.min_mz - 10
x_max = op.max_mz + 10
x_bins = 1000
y_bins = op.max_scan+1

results = np.zeros((x_bins+1, y_bins), dtype=weights.dtype)
T = typing.TypeVar("T", npt.NDArray[float], npt.NDArray[int])



FooMode = collections.namedtuple("ModedFunction", "slow fast safe")
def get_modes(foo):
    return FooMode(
        slow=foo,
        fast=numba.jit(nopython=True, cache=True)(foo),
        safe=numba.jit(nopython=True, cache=True, boundscheck=True)(foo),
    )

@get_modes
def weight2D(
    results: T,
    xx: npt.NDArray[int],
    yy: npt.NDArray[int],
    weights: T,
):
    for x, y, w in zip(xx, yy, weights):
        results[x, y] += w

@get_modes
def count2D(
    results: T,
    xx: npt.NDArray[int],
    yy: npt.NDArray[int],
):
    for x, y, w in zip(xx, yy, weights):
        results[x, y] += 1



# preprocessings: float binning, int binning (summarize consecutive ints by one, like integer division)

# each variable should be investigated and preprocessed
# this is float preprocessing
def get_bin_indices(x, x_min, x_max, x_bins):
    return ((x - x_min) * x_bins / (x_max - x_min)).astype(np.uint32)

preprocess_xx = functools.partial(get_bin_indices, x_min=x_min, x_max=x_max, x_bins=x_bins)
preprocess_yy = lambda yy: yy

statistic = weight2D
# statistic = count2D

# would be nicer to pass in grouping.

def fill(op, frames, x_bins, y_bins):
    results = np.zeros((x_bins+1, y_bins), dtype=np.float64)
    for frame in tqdm(frames):
        X = op.query(frames=frame, columns=("scan","mz","intensity"))
        weight2D.fast(
            results,
            preprocess_xx(X["mz"]),
            preprocess_yy(X["scan"]), 
            X["intensity"],
        )
    return results

MS1 = fill(op, op.ms1_frames, x_bins, y_bins)

dia_run = DiaRun(op, preload_data=False, cache_size=None)
results = {}
for step, frames in dia_run.DiaFrameMsMsInfo.groupby("step").frame:
    results[step] = fill(op, frames, x_bins, y_bins)



import matplotlib.pyplot as plt
with plt.style.context('dark_background'):
    plt.imshow(results[3], cmap='inferno', aspect="auto", origin="lower")
    plt.show()



cols = 5
with plt.style.context('dark_background'):
    gs = grid_spec.GridSpec(5, 4)
    fig = plt.figure(figsize=(16, 9))
    for step, stat in results.items():
        ax = fig.add_subplot(gs[step % cols, step // cols])
        ax.imshow(stat, cmap='inferno', aspect="auto", origin="lower")
        ax.set_title(f"Step {step}")
    plt.show()


title = "sqrt(TIC) per MIDIA diagonal"
title_fontsize = 20
panel_title_fontsize = 15
label_fontsize = 15
cols = 5
rows = 4
assert cols*rows >= len(results)
trans = np.sqrt
vmax = trans(max(r.max() for r in results.values()))
extents = (x_min, x_max, 0, op.max_scan+1)
with plt.style.context('dark_background'):
    gs = grid_spec.GridSpec(cols, rows)
    fig = plt.figure(figsize=(16, 9))
    for step, stat in results.items():
        ax = fig.add_subplot(gs[step % cols, step // cols])
        ax.imshow(
            trans(stat),
            cmap='inferno',
            aspect="auto",
            origin="lower",
            vmin=0,
            vmax=vmax,
            extent=extents)
        ax.set_title(f"Step {step}", fontsize=panel_title_fontsize)
        ax.set_xlabel("M/Z", fontsize=label_fontsize)
        ax.set_ylabel("Scan", fontsize=label_fontsize)
    fig.suptitle(title, fontsize=title_fontsize)
    plt.tight_layout()
    plt.show()


