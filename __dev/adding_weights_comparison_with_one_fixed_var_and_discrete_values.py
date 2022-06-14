%load_ext autoreload
%autoreload 2
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 43)
import matplotlib.pyplot as plt
import pathlib

from MSclusterparser.parser import read_4DFF_to_df_physical_dims_only
from MSclusterparser.boxes_ops import get_extents_vol_centers, cut_df, column_quantiles
from kilograms.histogramming import *

project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
source = unfragHeLa = _get_d("*3342.d")

# MS1 = read_4DFF_to_df_physical_dims_only(source/f"{source.name}.features")
# MS1.to_hdf(source/"results.hdf", mode="a", key="ms1_features")
MS1 = pd.read_hdf(source/"results.hdf", "ms1_features")
MS1 = get_extents_vol_centers(MS1)

extent_cols = [c for c in MS1.columns if "_extent" in c]
MS1_cut = cut_df(MS1, **column_quantiles(MS1[extent_cols])).copy()

# scatterplot_matrix(MS1_cut[extent_cols], bins=50)
# 
df = MS1_cut[extent_cols]

# bins = {c: 100 for c in df}
# extents = {c: min_max(df[c].to_numpy()) for c in df}

# z = get_1D_marginals(df, bins=bins, extents=extents)
# z = get_1D_marginals(df, bins=bins, extents=extents, weights=MS1_cut.intensity)
# z = get_2D_marginals(df, bins=bins, extents=extents, weights=MS1_cut.intensity)
# z = get_2D_marginals(df, bins=bins, extents=extents)

# with plt.style.context('dark_background'):
#     scatterplot_matrix(df, weights=MS1_cut.intensity)

# with plt.style.context('dark_background'):
#     fig, axs = scatterplot_matrix(df, weights=MS1_cut.peak_count, show=False)
#     fig.suptitle("Peak Count Weighted Features")
#     fig.show()

# df.mz_extent.to_numpy()

yy = MS1_cut.vol
# bins = {c:100 for c in df}
# bins[yy.name]=100
# extents = None
# weights = None
# extents = {c:min_max(df[c].to_numpy()) for c in df}
# extents[yy.name] = min_max(yy.to_numpy())
# imshow_kwargs={"cmap":"inferno"}

with plt.style.context('dark_background'):
    fig, axs = crossplot(df, np.log10(MS1_cut.intensity), show=False)
    fig.suptitle("Cross Plot")
    fig.show()

with plt.style.context('dark_background'):
    fig, axs = crossplot(df, np.log10(MS1_cut.intensity), show=False)
    fig.suptitle("Cross Plot")
    fig.show()

# now 
