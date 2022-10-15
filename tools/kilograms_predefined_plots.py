#!/usr/bin/env python3
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description='Produce a set of predefined kilograms plots.')
parser.add_argument("data_path", help="File containing data to be plotted", type=Path)
parser.add_argument("-c", "--config", help="Config file, containing plot type definitions.", type=Path, required=True)
parser.add_argument("-n", "--names", help="Names of plots to be produced (comma-separated)", type=str, default=None)
parser.add_argument("-o", "--output", help="Output file path. Will display onscreen if omitted. \"?\" will be replaced by plot name (remember to bash-escape it).", type=str, default="plot-?.pdf")
parser.add_argument("-f", "--force", help="force overwriting of the target path if it exists", action='store_true')
args=parser.parse_args()

import math
import numpy as np
import toml
from matplotlib import pyplot as plt
import pandas
from dia_common.misc import pandas_read_table
import kilograms

config = toml.load(args.config)

if args.names is None:
    args.names = ','.join(config.keys())

data = pandas_read_table(args.data_path)
data_dct = {colname: data[colname] for colname in data}


for plot_name in args.names.split(","):
    plot = config[plot_name]
    out_p = Path(args.output.replace("?", plot_name))
    
    if not args.force:
        assert not out_p.exists()

    df = pandas.DataFrame()
    for colname, coldef in plot.items():
        df[colname] = eval(coldef, globals() | data_dct)

    with plt.style.context('dark_background'):
        fig, axes = kilograms.scatterplot_matrix(df, show=False)
        fig.set_size_inches(30, 30)
        plt.savefig(out_p, dpi=100, transparent=False)
        plt.close()
