#!/usr/bin/env python3
import argparse
import kilograms

from matplotlib import pyplot as plt
from pathlib import Path

# from dia_common.misc import pandas_read_table
from kilograms.io import pandas_read_table

parser = argparse.ArgumentParser(description="Produce kilograms plot.")
parser.add_argument("data_path", help="File containing data to be plotted", type=Path)
parser.add_argument(
    "-c",
    "--columns",
    help="Columns to include in plot, in comma-separated list. Will use all if omitted.",
    type=str,
    default=None,
)
parser.add_argument(
    "-o",
    "--output",
    help="Output file path. Will display onscreen if omitted.",
    type=Path,
    default=None,
)
parser.add_argument(
    "--force",
    help="force overwriting of the target path if it exists",
    action="store_true",
)
args = parser.parse_args()

if (not args.force) and (not (args.output is None)):
    assert not args.output.exists()


data = pandas_read_table(args.data_path)

if not (args.columns is None):
    cols = args.columns.split(",")
    try:
        data = data[cols]
    except KeyError:
        print("Available columns are:", data.columns)
        raise


with plt.style.context("dark_background"):
    kilograms.scatterplot_matrix(data)
    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output)
