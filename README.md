A simple package for making histograms and simple summaries of rather large data-frames.
For me, it is faster than numpy and fast-histogram.
But your mileage might vary.

# Usage

So, we have some bigger data:
```{python3}
          mz_extent  inv_ion_mobility_extent  retention_time_extent  \
1          0.036465                 0.111729              17.646362
2          0.036499                 0.102713              15.293213
3          0.029226                 0.079561              12.940674
4          0.037246                 0.120728              16.469666
5          0.033551                 0.118480              16.469666
...             ...                      ...                    ...
45695322   0.020608                 0.011571              12.937378
45695325   0.015745                 0.009255               5.882202
45695326   0.015708                 0.013889               4.704956
45695333   0.020197                 0.009259               3.529663
45695336   0.019840                 0.013889               7.056519

          log10_intensity  log10_vol
1                4.270305  -1.143297
2                3.582283  -1.241600
3                2.874973  -1.521569
4                4.767440  -1.130434
5                4.284236  -1.183966
...                   ...        ...
45695322         2.671344  -2.510728
45695325         2.550970  -3.066955
45695326         2.549371  -2.988647
45695333         2.187869  -3.180395
45695336         2.092367  -2.711186

[34_739_252 rows x 5 columns]
```
like ~35M points.

Say we want to bin `mz_extent` between min and max value:
```
xx = df.mz_extent.to_numpy()
extent = min_max(xx)
counts1D = histogram1D(xx, extent=extent, bins=100)
len(counts1D)# 100
sum(counts1D)# 34_739_252
```

You cannot have smaller extents: that will not work (for speed, no border checks).

Say we want to bin both `mz_extent` and `inv_ion_mobility_extent`
```{python}
xx = df.mz_extent.to_numpy()
yy = df.inv_ion_mobility_extent.to_numpy()
counts2D = histogram2D(xx, yy, extent=(min_max(xx), min_max(yy)), bins=(100,100))
counts2D.shape
np.all(np.sum(counts2D, axis=1) == counts1D) # True
np.sum(counts2D) == len(df) # True
```

To make a plot:
```{python3}
import matplotlib.pyplot as plt
from kilograms import scatterplot_matrix

# df is your data-frame with uniquely named columns
# each dimension has a numerical value.

with plt.style.context('dark_background'):
    scatterplot_matrix(df, y_labels_offset=-.2)
```

![](https://github.com/MatteoLacki/kilograms/blob/main/scatterplot_matrix.png "Scatterplot Matrix")