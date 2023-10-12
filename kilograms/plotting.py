import matplotlib.pyplot as plt
import numpy.typing as npt


def plot_overlayed_histograms(
    bins: int | npt.NDArray | list | str = "auto",
    alpha: float = 0.5,
    show: bool = True,
    plot_args: list = [],
    plot_kwargs: dict = {},
    **tag_to_data,
):
    _iter = iter(tag_to_data.items())
    tag, x = next(_iter)
    _, bins, _ = plt.hist(
        x, alpha=alpha, bins=bins, label=tag, *plot_args, **plot_kwargs
    )
    for tag, x in _iter:
        plt.hist(x, alpha=alpha, bins=bins, label=tag, *plot_args, **plot_kwargs)
    plt.legend()
    if show:
        plt.show()
