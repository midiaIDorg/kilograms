#!/usr/bin/env python3
import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from midia_search_engines.io import open_config

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 5)


class args:
    tables = [
        "cluster_stats/G8027/b5e24bd5ca2/4D/0/ms1.parquet",
        "cluster_stats/G8027/bcc7976cc60/4D/0/ms1.parquet",
    ]
    config = "configs/cluster_comparison/ms1.toml"
    output = Path("/tmp/plots")
    width = 20
    height = 20
    dpi = 400
    style = "default"
    alpha = 0.5
    point_size = 1
    verbose = True
    show = True


parser = argparse.ArgumentParser(
    description="Compare two sets of clusters.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "tables",
    help="Path to ms1 level statistics in the parquet format.",
    nargs="+",
    type=Path,
)
parser.add_argument(
    "config",
    help="Path to a config file specifying in SQL how to extract data from each file.",
    type=Path,
)
parser.add_argument(
    "--output", help="Path to where to save the histograms.", default=None, type=Path
)
parser.add_argument(
    "--plots_folder",
    help="Folder to save QC plots.",
    default=None,
    type=Path,
)
parser.add_argument(
    "-width",
    help="Plot width.",
    type=int,
    default=10,
)
parser.add_argument(
    "-height",
    help="Plot height.",
    type=int,
    default=10,
)
parser.add_argument(
    "-alpha",
    help="Histogram alpha.",
    type=float,
    default=0.5,
)
parser.add_argument(
    "-dpi",
    help="Plot dpi.",
    type=int,
    default=100,
)
parser.add_argument(
    "--show",
    help="Show plots one after another.",
    action="store_true",
)
parser.add_argument(
    "--verbose",
    help="Show plots one after another.",
    action="store_true",
)
args = parser.parse_args()


if __name__ == "__main__":
    config = open_config(args.config)
    conn = duckdb.connect()

    dfs = [
        conn.execute(dataset["sql"].format(clusters=path)).df()
        for dataset, path in zip(config["dataset_definitions"], args.tables)
    ]
    common_columns = dfs[0].columns
    for df in dfs:
        assert set(df.columns) == set(common_columns)

    # column_name = "peak_count"
    save = args.output is not None
    if args.show or save:
        for column_name in common_columns:
            if args.verbose:
                print(f"Plotting `{column_name}`.")

            _min = min(df[column_name].min() for df in dfs)
            _max = max(df[column_name].max() for df in dfs)
            _extent = _max - _min
            _min -= _extent * 0.01
            _max += _extent * 0.01
            bins = np.linspace(_min, _max, config["bins"])

            for path, df in zip(args.tables, dfs):
                plt.hist(
                    df[column_name],
                    bins=bins,
                    alpha=args.alpha,
                    label=str(path),
                )
            plt.legend()
            plt.title(column_name)
            if column_name in config["log10_x_axis"]:
                plt.xscale("log")
            if column_name in config["log10_y_axis"]:
                plt.yscale("log")
            if args.show:
                plt.show()
            if save:
                plt.savefig(args.output / f"{column_name}.pdf", dpi=args.dpi)
            plt.close()
