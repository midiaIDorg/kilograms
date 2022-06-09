A simple package for making histograms and simple summaries of rather large data-frames.

# Usage

To make a plot:
```{python3}
import matplotlib.pyplot as plt
from kilograms import scatterplot_matrix

with plt.style.context('dark_background'):
    scatterplot_matrix(
        df2[extent_cols+["log10_intensity","log10_vol"]],
        show=False,
        y_labels_offset=-.2,
    )
    plt.suptitle("MS1 features")
    plt.show()
```