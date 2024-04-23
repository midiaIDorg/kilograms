#!/usr/bin/env python3
import argparse
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomllib
from kilograms.histogramming import histogram1D, histogram2D, min_max

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 5)

if True:
    args = SimpleNamespace(
        tables=[
            Path(
                "P/cluster_stats/raw/GyMCIGRxW2wfpplKv6G-YesO2L__h1dbUALNs7Z3i6AFpnGG6TYRwcOj12wOPSMKvOyCMEOqu6nCca2FNOrnK8a0K5H0imag4x07fsnzXBlhwTlPYkvA8WhrukrIAsPyWSPhKl_WSABXx60HGlcEs2tsvw1NN7TgcD_u074u2ycFhGLKaDl2tbPkwA_EdEsK9W-smPFKWrAg-4OFdt4tSQ==/precursor_stats.parquet"
            ),
            Path(
                "P/cluster_stats/raw/GyMCIGRxW2wfpplKv6G-YesO2L__h1dbUALNs7Z3i6AFpnGG6TYRwcOj12wOPSMKvOyCMEOqu6nCca2FNOrnK8a0K5H0imag4x07fsnzXBlhwTlPYkvA8WhrukrIAsPyWSPhKl_WSABXx60HGlcEs2tsvw1NN7TgcD_u074u2ycFhGLKaDl2tbPkwA_EdEsK9W-smPFKWrAg-4OFdt4tSQ==/precursor_stats.parquet"
            ),
        ],
        config=Path("configs/plots/histograms_config.toml"),
        output=Path("/tmp/testplots"),
    )
else:
    parser = argparse.ArgumentParser(
        description="Compare two sets of clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="Path to a config file specifying in SQL how to extract data from each file.",
        type=Path,
    )
    parser.add_argument(
        "tables",
        help="Path to ms1 level statistics in the parquet format.",
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--output",
        help="Path to where to save the histograms.",
        default=None,
        type=Path,
    )
    args = SimpleNamespace(**parser.parse_args().__dict__)

get_bin_centers = lambda xx: (xx[1:] + xx[:-1]) / 2.0

if __name__ == "__main__":
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    conn = duckdb.connect()

    df2dict = lambda df: {c: df[c].to_numpy() for c in df.columns}

    dfs = [
        df2dict(conn.execute(config["sql"].format(table=table_path)).df())
        for table_path in args.tables
    ]

    columns = set(dfs[0])
    for df in dfs:
        columns &= set(df)

    assert len(columns) > 0, "No common columns to plot."

    try:
        groups = config.groups
    except AttributeError:
        groups = list(map(str, range(len(dfs))))

    dim_bins = {}
    bin_centers = {}
    counts = {}
    for column in columns:
        if not config["silent"]:
            print(f"Plotting `{column}`.")

        _min = math.inf
        _max = -math.inf
        for df in dfs:
            _df_min, _df_max = min_max(df[column])
            _min = min(_min, _df_min)
            _max = max(_max, _df_max)
        _extent = _max - _min
        _min -= _extent * 0.01
        _max += _extent * 0.01
        dim_bins[column] = np.linspace(_min, _max, config["bins"] + 1)
        bin_centers[column] = get_bin_centers(dim_bins[column])

        for group, df in zip(groups, dfs):
            counts[column, group] = histogram1D(
                xx=df[column],
                extent=dim_bins[column][[0, -1]],
                bins=config["bins"],
            )
            plt.plot(
                bin_centers[column],
                counts[column, group],
                label=group,
            )
        plt.legend()
        plt.title(column)

        if args.output is None:
            plt.show()
        else:
            plt.savefig(args.output / f"{column}.pdf", dpi=config["dpi"])
        plt.close()


# # column_hor, column_ver = "frame_wmean", "tof_wmean"
# # group, df = next( zip(groups, dfs))
# for column_hor, column_ver in itertools.combinations(columns, r=2):
#     for group, df in zip(groups, dfs):
#         intensities = histogram2D(
#             xx=df[column_hor],
#             yy=df[column_ver],
#             extent=(dim_bins[column_hor][[0, -1]], dim_bins[column_ver][[0, -1]]),
#             bins=(config["bins"], config["bins"]),
#             mode="safe",
#         )
#         w = np.quantile(intensities, np.linspace(0.1,1,3))
#         w = np.unique(w)

#         plt.contour(intensities, color="red", levels=w, label=group)
#     plt.legend()
#     plt.title(f"column")
#     plt.show()
