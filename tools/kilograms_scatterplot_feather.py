#!/home/matteo/Projects/MIDIA/veMIDIA/bin/python
import argparse
from pathlib import Path

import kilograms
import pandas as pd
from matplotlib import pyplot as plt

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
    "-w",
    "--weights_column_name",
    help="Column that will be used as weights.",
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
    "-title",
    help="Plot title string.",
    type=str,
    default=None,
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
    "-dpi",
    help="Plot dpi.",
    type=int,
    default=50,
)
parser.add_argument(
    "--style",
    help="Style of the plot.",
    default="dark_background",
)
parser.add_argument(
    "--cmap",
    help="Color map used for the 2D marginals.",
    default="inferno",
)


args = parser.parse_args()


if __name__ == "__main__":

    cols = set(args.columns)
    if args.weights_column_name is not None:
        cols.add(args.weights_column_name)

    data = pd.read_feather(
        path=args.data_path,
        columns=list(cols),
    )
    with plt.style.context(args.style):
        if args.weights_column_name is not None:
            cols.remove(args.weights_column_name)
            fig, axes = kilograms.scatterplot_matrix(
                data[args.columns],
                weights=data[args.weights_column_name],
                imshow_kwargs={"cmap": args.cmap},
                show=False,
            )
        else:
            fig, axes = kilograms.scatterplot_matrix(
                data[args.columns],
                imshow_kwargs={"cmap": args.cmap},
                show=False,
            )

        if args.title is not None:
            plt.suptitle(args.title)

        if args.output is None:
            plt.show()
        else:
            fig.set_size_inches(args.width, args.height)
            plt.savefig(args.output, dpi=args.dpi, transparent=False)
            plt.close()
