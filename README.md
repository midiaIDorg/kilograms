A simple package for making histograms and simple summaries of rather large data-frames.

# Usage

To make a plot:
```{python3}
import matplotlib.pyplot as plt
from kilograms import scatterplot_matrix

# df is your data-frame with uniquely named columns
# each dimension has a numerical value.

with plt.style.context('dark_background'):
    scatterplot_matrix(
        df,
        show=False,
        y_labels_offset=-.2,
    )
    plt.suptitle("MS1 features")
    plt.show()
```