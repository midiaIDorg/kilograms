#!/usr/bin/env python3
import argparse
from pprint import pprint

import duckdb
import kilograms
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(
    description="Make a scatterplot_matrix based on a duckb sql: amazing, isn't it doc? You don't have a PhD? What a pity! Work harder. Or better simply find yourself a decent work not in academia, cause the are crazy people here."
)
parser.add_argument(
    "sql",
    help="An SQL string to be executed with `duckdb` to get the data for plotting.",
)
parser.add_argument(
    "-o",
    "--output",
    help="Output file path. Will display onscreen if omitted.",
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
    default=100,
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
parser.add_argument(
    "--lims",
    nargs=3,
    action="append",
    help="Limits in form <name> <left> <right>",
    default=None,
)
parser.add_argument(
    "--debug",
    action="store_true",
)
parser.add_argument("--transparent", action="store_true")
parser.add_argument(
    "--y_hist_bottom_lim",
    help="Set the lower level for the diagonal histograms.",
    type=float,
    default=0,
)


args = parser.parse_args()
if args.debug:
    pprint(args.__dict__)

if __name__ == "__main__":
    df = duckdb.query(args.sql).df()

    with plt.style.context(args.style):
        fig, axes = kilograms.scatterplot_matrix(
            df,
            imshow_kwargs={"cmap": args.cmap},
            show=False,
            lims=args.lims,
            constrained_layout=True,
        )
        if args.title is not None:
            plt.suptitle(args.title)

        if args.output is None:
            plt.show()
        else:
            fig.set_size_inches(args.width, args.height)
            plt.savefig(args.output, dpi=args.dpi, transparent=args.transparent)
            plt.close()
