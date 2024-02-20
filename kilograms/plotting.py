import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd


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


def get3DclusterVis(cl: pd.DataFrame):
    import plotly.express as px

    fig = px.scatter_3d(
        cl,
        z="frameIdx",
        y="scan",
        x="tof",
        opacity=0.5,
        size="intensity",
        size_max=50,
        hover_data=cl.columns,
    )
    fig = fig.update_layout(
        scene=dict(
            # xaxis=dict(backgroundcolor="white"),
            # yaxis=dict(backgroundcolor="white"),
            # zaxis=dict(backgroundcolor="white"),
            bgcolor="white",
        ),
        showlegend=True,
    )
    return fig
