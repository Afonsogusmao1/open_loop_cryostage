from __future__ import annotations


DEFAULT_PLOT_RC = {
    "figure.figsize": (8.5, 4.5),
    "savefig.dpi": 250,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
}


def configure_matplotlib(pyplot=None, *, figure_size: tuple[float, float] | None = None) -> None:
    if pyplot is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

    rc = dict(DEFAULT_PLOT_RC)
    if figure_size is not None:
        rc["figure.figsize"] = figure_size
    pyplot.rcParams.update(rc)


__all__ = ["DEFAULT_PLOT_RC", "configure_matplotlib"]
