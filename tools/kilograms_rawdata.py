#!/usr/bin/env python3
import argparse
import itertools
from functools import partial
from math import ceil, floor, inf
from pathlib import Path

import numba
import numpy as np
import opentimspy
from matplotlib import pyplot as plt
from tqdm import tqdm

# from scipy.interpolate import RegularGridInterpolator

# from IPython import get_ipython

# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")


# class args:
#     rawdata_path = "spectra/G8027.d"
#     ms_level = 1
#     output = None
#     progressbar_message = "Iterating for test"
#     tof_rounding = 100
#     marginals1D = True
#     verbose = True
#     width = 30
#     height = 30
#     dpi = 100


parser = argparse.ArgumentParser(description="Produce kilograms plot.")
parser.add_argument(
    "rawdata_path", help="File containing data to be plotted", type=Path
)
parser.add_argument(
    "ms_level",
    help="Which MS level to plot?",
    type=int,
    choices=[1, 2],
)
parser.add_argument(
    "-o",
    "--output",
    help="Output folder path. Will display onscreen if omitted.",
    type=Path,
    default=None,
)
parser.add_argument(
    "--progressbar_message",
    help="Message to use with tqdm.",
    default=None,
)
parser.add_argument(
    "--tof_rounding",
    help="Order of numbers to round up to.",
    default=100,
    type=int,
)
parser.add_argument(
    "--verbose",
    help="Print info to stdout: logging is for people with too much time on their hands.",
    action="store_true",
)
parser.add_argument(
    "--width",
    help="Plot width.",
    type=int,
    default=10,
)
parser.add_argument(
    "--height",
    help="Plot height.",
    type=int,
    default=10,
)
parser.add_argument(
    "--dpi",
    help="Plot dpi.",
    type=int,
    default=100,
)
parser.add_argument("--transparent", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
    op = opentimspy.OpenTIMS(args.rawdata_path)

    frames = op.ms1_frames
    if args.ms_level == 2:
        frames = op.ms2_frames

    def round_up(x: float | int, rounding: int) -> int:
        return ceil(x // rounding) * rounding

    max_scan = op.max_scan + 1
    max_tof = round_up(op.mz_to_tof(op.max_mz, frames[[0]])[0], args.tof_rounding) + 1
    sizes = {
        "frame": len(frames),
        "scan": max_scan,
        "tof": max_tof // args.tof_rounding + 1,
    }

    marginals2D = {
        (c0, c1): np.zeros(shape=(s0, s1), dtype=np.uint32)
        for (c0, s0), (c1, s1) in itertools.combinations(sizes.items(), 2)
    }

    @numba.njit(boundscheck=True)
    def add_sparse_array_to_full(arr, xx, yy, values):
        for x, y, v in zip(xx, yy, values):
            arr[x, y] += v

    if args.verbose:
        print("Calculating 2D marginal distributions of raw events.")

    # frame_idx, frame = 0,1
    for frame_idx, frame in enumerate(tqdm(frames)):
        tofs, scans, intensities = op.query(
            frame, columns=("tof", "scan", "intensity")
        ).values()
        frame_idxs = np.full(shape=tofs.shape, fill_value=frame_idx)
        tof_idxs = tofs // args.tof_rounding
        add_sparse_array_to_full(
            marginals2D[("frame", "scan")],
            xx=frame_idxs,
            yy=scans,
            values=intensities,
        )
        add_sparse_array_to_full(
            marginals2D[("frame", "tof")],
            xx=frame_idxs,
            yy=tof_idxs,
            values=intensities,
        )
        add_sparse_array_to_full(
            marginals2D[("scan", "tof")],
            xx=scans,
            yy=tof_idxs,
            values=intensities,
        )

    intensity_transformations = {
        ("frame", "scan"): np.sqrt,
        ("frame", "tof"): np.log1p,
        ("scan", "tof"): np.log1p,
    }

    if args.verbose:
        print("Making marginals1D.")

    marginals1D = {
        "frame": marginals2D[("frame", "scan")].sum(axis=1),
        "scan": marginals2D[("frame", "scan")].sum(axis=0),
        "tof": marginals2D[("scan", "tof")].sum(axis=0),
    }

    axes = {
        "frame": frames,
        "scan": np.arange(0, max_scan, 1),
        "tof": np.arange(0, max_tof, args.tof_rounding),
    }

    axes["retention_time"] = op.frame2retention_time(axes["frame"])
    axes["inv_ion_mobility"] = op.scan_to_inv_ion_mobility(axes["scan"], frames[[0]])
    axes["mz"] = op.tof_to_mz(axes["tof"], frames[[0]])

    physical_to_index = {
        "retention_time": "frame",
        "inv_ion_mobility": "scan",
        "mz": "tof",
    }
    index_to_physical = {i: p for p, i in physical_to_index.items()}

    interpolator_kwargs = {
        "bounds_error": False,
        "fill_value": 0,
    }

    ##TODO: finish off non-index dimensions.

    # interpolator = partial(RegularGridInterpolator, **interpolator_kwargs)
    # interpolations = {
    #     (c0, c1): interpolator(
    #         (axes[c0], axes[c1]),
    #         intensities,
    #     )
    #     for (c0, c1), intensities in marginals2D.items()
    # }
    # for i0, i1 in list(interpolations):
    #     c0 = index_to_physical[i0]
    #     c1 = index_to_physical[i1]
    #     interpolations[(c0, c1)] = interpolator(
    #         (
    #             axes[c0],
    #             axes[c1],
    #         ),
    #         marginals2D[(i0, i1)],
    #     )

    # for i0, intensities in marginals1D.items():
    #     interpolations[i0] = interpolator((axes[i0],), intensities)
    #     c0 = index_to_physical[i0]
    #     interpolations[c0] = interpolator((axes[c0],), intensities)

    extents = {c: [ax.min().round(4), ax.max().round(4)] for c, ax in axes.items()}

    imshow_kwargs = dict(
        origin="lower",
        aspect="auto",
        vmin=0,
        cmap="inferno",
    )

    max_intensity = max(map(np.max, marginals2D.values()))

    for (i0, i1), intensities in marginals2D.items():
        transform = intensity_transformations[(i0, i1)]
        plt.imshow(
            transform(intensities),
            extent=extents[i0] + extents[i1],
            **imshow_kwargs,
        )
        plt.xlabel(i0)
        plt.ylabel(i1)
        plt.title(f"{args.rawdata_path.name}: {transform.__name__}(intensity)")
        if args.output is None:
            plt.show()
        else:
            plt.savefig(
                args.output / f"{i0}_{i1}.pdf",
                dpi=args.dpi,
                transparent=args.transparent,
            )
        plt.close()
        ##TODO: finish off non-index dimensions.

        # c0 = index_to_physical[i0]
        # c1 = index_to_physical[i1]
        # interpolation = interpolations[(c0, c1)]
        # interpolated_intensities = interpolation(
        #     tuple(np.meshgrid(axes[c0], axes[c1], indexing="ij", sparse=True))
        # )

        # plt.imshow(
        #     transform(interpolated_intensities),
        #     extent=extents[c0] + extents[c1],
        #     **imshow_kwargs,
        # )
        # plt.xlabel(c0)
        # plt.ylabel(c1)
        # plt.title(f"{transform.__name__}(intensity)")
        # plt.show()
