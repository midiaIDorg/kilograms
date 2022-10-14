import MSclusterparser.raw_peaks_4DFF_parser as parsers
import pathlib

from kilograms import scatterplot_matrix

small5p = pathlib.Path("/home/matteo/Projects/MIDIA/rawdata/small5P.d")

raw_peaks_sqlite_physical = parsers.Clusters_4DFF_SQLITE(
    small5p / "small5P_index.d.features"
)
output_4DFF_index = raw_peaks_sqlite_physical.range_query(
    1, 100, columns=("ClusterID", "RT", "Mobility", "MZ", "Intensity")
)

df0 = output_4DFF_physical.as_frame()
df1 = df0.copy()
df1.columns = [f"{col}_1" for col in df0.columns]
# df2 = df0.copy()
# df2.columns = [f"{col}_2" for col in df0.columns]
df = pd.concat([df0, df1], axis=1)

import matplotlib.pyplot as plt

with plt.style.context("seaborn-white"):
    fig, axes = scatterplot_matrix(df, show=False)
    fig.set_size_inches(30, 30)
    plt.savefig("/tmp/test.png", dpi=100, transparent=False)
