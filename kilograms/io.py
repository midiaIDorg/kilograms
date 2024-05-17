import os
import pandas as pd
import pickle
import tables
import random
import time

from pathlib import Path

try:
    from pandas_ops.io import read_df as pandas_read_table
except ModuleNotFoundError:

    def pandas_read_table(filename: str | os.PathLike):
        filename = Path(filename)
        extension = filename.suffix
        if extension in [".csv", "tsv"]:
            return pd.read_csv(filename)
        if extension == ".feather":
            return pd.read_feather(filename)
        if extension == ".parquet":
            return pd.read_parquet(filename)
        if extension in [".pickle", ".peakpkl", ".psmpkl"]:
            with open(filename, "rb") as f:
                DT = pickle.load(f)
            assert isinstance(DT, pd.DataFrame)
            return DT
        if extension == ".hdf":
            retries = 100
            for _ in range(retries):
                try:
                    return pd.read_hdf(filename)
                except tables.exceptions.HDF5ExtError:
                    time.sleep(random.uniform(0.0, 0.1))
            return pd.read_hdf(filename)

    raise RuntimeError(
        f"Don't know how to open file with {extension} extension. Path: {str(filename)}"
    )
