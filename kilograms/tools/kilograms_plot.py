#!/usr/bin/env python3
import argparse
from pathlib import Path

import kilograms
import pandas as pd
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Produce kilograms plot.")
    parser.add_argument(
        "data_path", help="Feather file containing data to be plotted", type=Path
    )
    parser.add_argument(
        "column",
        help="Column to plot",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path. Will display onscreen if omitted.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-dpi",
        help="Plot dpi.",
        type=int,
        default=100,
    )

    args = parser.parse_args()
    df = pd.read_feather(args.data_path)
    plt.plot(df[args.column])
    plt.savefig(str(args.output), dpi=args.dpi, transparent=False)
    plt.close()


if __name__ == "__main__":
    main()
