#!/usr/bin/env python3
import argparse
import kilograms
import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path


parser = argparse.ArgumentParser(description="Produce kilograms plot.")
parser.add_argument(
    "data_path", help="Feather file containing data to be plotted", type=Path
)
parser.add_argument(
    "-c",
    "--columns",
    help="Columns to include in plot",
    nargs="*",
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
    "-n",
    help="Number of rows to show.",
    type=int,
    default=5,
)


args = parser.parse_args()


if __name__ == "__main__":
    data = pd.read_feather(
        path=args.data_path,
        columns=args.columns,
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", args.n)
    print(data)
