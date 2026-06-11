"""loss_plotter.py — grouped, class-based training curve visualiser."""
 
from __future__ import annotations
 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from dataclasses import dataclass
 
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
 
@dataclass
class PlotConfig:
    title:     str        = "Plot Title"
    xlabel:    str        = "X label"
    ylabel:    str        = "Y label"
    figsize:   tuple      = (9, 5)
    log_scale: bool       = False
    smoothing: float      = 0.0        # EMA factor in [0, 1)
    save_path: str | None = None
 
 
@dataclass(frozen=True)
class _Theme:
    colors:     tuple = ("#378ADD", "#D85A30", "#2DBD8F", "#A06CC8")
    annotation: str   = "#888680"
    text:       str   = "#2C2C2A"
    subtext:    str   = "#5F5E5A"
    grid:       str   = "#D3D1C7"
    background: str   = "#FAFAF9"
 
THEME = _Theme()
 
 
# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
 
class DataProcessor:
 
    @staticmethod
    def to_array(data: list | np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float)
 
    @staticmethod
    def ema(values: np.ndarray, alpha: float) -> np.ndarray:
        """Exponential moving average. alpha=0 returns values unchanged."""
        if alpha == 0:
            return values
        out = np.empty_like(values)
        out[0] = values[0]
        for i in range(1, len(values)):
            out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
        return out
 
 
 
 
# ---------------------------------------------------------------------------
# Axes styling
# ---------------------------------------------------------------------------
 
class AxesStyler:
 
    @staticmethod
    def apply_base(ax: plt.Axes, cfg: PlotConfig) -> None:
        ax.set_facecolor(THEME.background)
        ax.grid(True, color=THEME.grid, linewidth=0.6, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(THEME.subtext)
            spine.set_linewidth(1.4)
        ax.tick_params(colors=THEME.subtext, labelsize=9)
        ax.set_xlabel(cfg.xlabel, fontsize=11, color=THEME.text , labelpad=8)
        ax.set_ylabel(cfg.ylabel, fontsize=11, color=THEME.text , labelpad=8)
        if cfg.title:
            ax.set_title(cfg.title, fontsize=18, fontweight="medium",
                         color=THEME.text, pad=14)
 
    @staticmethod
    def apply_log_scale(ax: plt.Axes) -> None:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
 
    @staticmethod
    def apply_legend(ax: plt.Axes) -> None:
        legend = ax.legend(
            fontsize=10, frameon=True, framealpha=0.9,
            edgecolor=THEME.grid, facecolor=THEME.background,
            loc="upper right", handlelength=2.0,
        )
        for text in legend.get_texts():
            text.set_color("#444441")
 
 
# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
 
class SeriesRenderer:
 
    @staticmethod
    def draw(
        ax: plt.Axes,
        epochs: np.ndarray,
        values: np.ndarray,
        color: str,
        label: str,
        dashed: bool,
    ) -> None:
        line_kwargs: dict = dict(color=color, linewidth=2.0, label=label, zorder=3)
        if dashed:
            line_kwargs |= dict(linestyle="--", dashes=(6, 4))
        ax.plot(epochs, values, **line_kwargs)
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def plot_subplots(
    *series: list | np.ndarray,
    labels: list[str] | None = None,
    cfg:    PlotConfig       = PlotConfig(),
    show:   bool             = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot each series in its own subplot, stacked vertically.
 
    X axis is shared and fixed to the longest series length.
    Y axis is independent per subplot, scaled to each curve.
 
    Parameters
    ----------
    *series : one or more array-like sequences of values
    labels  : display name for each series (default: "Series 1", "Series 2", ...)
    cfg     : PlotConfig — all visual and behaviour options
 
    Returns
    -------
    fig, list[ax]
 
    Examples
    --------
    plot_subplots(train_loss, val_loss,
                  labels=["Train loss", "Val loss"],
                  cfg=PlotConfig(title="Transformer"))
    """
    if not series:
        raise ValueError("At least one series is required.")
 
    arrays  = [DataProcessor.to_array(s) for s in series]
    labels  = labels or [f"Series {i + 1}" for i in range(len(arrays))]
    smooth  = [DataProcessor.ema(a, cfg.smoothing) for a in arrays]
    x_max   = max(len(a) for a in arrays)
    n       = len(arrays)
 
    fig_h   = cfg.figsize[1] * n
    fig, axes = plt.subplots(n, 1, figsize=(cfg.figsize[0], fig_h), sharex=True)
    fig.patch.set_facecolor(THEME.background)
 
    if n == 1:
        axes = [axes]
 
    for i, (ax, sm, label) in enumerate(zip(axes, smooth, labels)):
        epochs = np.arange(1, len(arrays[i]) + 1)
        color  = THEME.colors[i % len(THEME.colors)]
 
        _cfg = PlotConfig(
            title="",
            xlabel=cfg.xlabel if i == n - 1 else "",
            ylabel=cfg.ylabel,
            log_scale=cfg.log_scale,
        )
        AxesStyler.apply_base(ax, _cfg)
        ax.set_xlim(1, x_max)
 
        SeriesRenderer.draw(ax, epochs, sm, color, label, dashed=False)
 
        if cfg.log_scale:
            AxesStyler.apply_log_scale(ax)
 
        AxesStyler.apply_legend(ax)
 
    fig.suptitle(cfg.title, fontsize=18, fontweight="medium",
                 color=THEME.text, y=1.01)
    plt.tight_layout()
 
    if cfg.save_path:
        fig.savefig(cfg.save_path, dpi=150, bbox_inches="tight")
        #print(f"Saved → {cfg.save_path}")
    elif show:
        plt.show()
    plt.close(fig)
 
    return fig, axes
 
 
def plot_timeseries(
    *series: list | np.ndarray,
    labels: list[str] | None = None,
    cfg:    PlotConfig       = PlotConfig(),
    show:   bool             = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one or more time series on a single axes.
 
    Parameters
    ----------
    *series : one or more array-like sequences of values
    labels  : display name for each series (default: "Series 1", "Series 2", ...)
    cfg     : PlotConfig — all visual and behaviour options
 
    Returns
    -------
    fig, ax
 
    Examples
    --------
    # loss curves
    plot_timeseries(train_loss, val_loss,
                    labels=["Train loss", "Val loss"],
                    cfg=PlotConfig(title="Transformer", smoothing=0.6))
 
    # single series
    plot_timeseries(lr_schedule, labels=["Learning rate"], cfg=PlotConfig(ylabel="LR"))
    """
    if not series:
        raise ValueError("At least one series is required.")
 
    arrays  = [DataProcessor.to_array(s) for s in series]
    labels  = labels or [f"Series {i + 1}" for i in range(len(arrays))]
    epochs  = np.arange(1, len(arrays[0]) + 1)
    smooth  = [DataProcessor.ema(a, cfg.smoothing) for a in arrays]
 
    fig, ax = plt.subplots(figsize=cfg.figsize)
    fig.patch.set_facecolor(THEME.background)
 
    AxesStyler.apply_base(ax, cfg)
 
    for i, (raw, sm, label) in enumerate(zip(arrays, smooth, labels)):
        color  = THEME.colors[i % len(THEME.colors)]
        dashed = (i > 0)
        SeriesRenderer.draw(ax, epochs, sm, color, label, dashed)
 
    if cfg.log_scale:
        AxesStyler.apply_log_scale(ax)
 
    AxesStyler.apply_legend(ax)
    plt.tight_layout()
 
    if cfg.save_path:
        fig.savefig(cfg.save_path, dpi=150, bbox_inches="tight")
        #print(f"Saved → {cfg.save_path}")
    elif show:
        plt.show()
    plt.close(fig)
 
    return fig, ax


def save_figures_to_pdf(figs: list[plt.Figure], path: str) -> None:
    """Save a list of figures as pages in a single PDF."""
    with PdfPages(path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _generate_demo(n: int = 60, seed: int = 0) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    train, val = [], []
    t = 2.3
    for i in range(n):
        t *= 0.955 + rng.standard_normal() * 0.005
        v  = t * (1.04 + rng.standard_normal() * 0.008 + max(0, (i - 40) * 0.002))
        train.append(round(t, 4))
        val.append(round(v, 4))
    return train, val


if __name__ == "__main__":
    train_losses, val_losses = _generate_demo()

    plot_timeseries(
        train_losses,
        val_losses,
        labels=["Train loss", "Val loss"],
        cfg=PlotConfig(
            title="Transformer — time series anomaly detection",
            figsize = (9, 5),
            # smoothing=0.6,
            # log_scale=True,
            # save_path="loss_curve.png",
        ),
    )
    
    plot_subplots(
        train_losses,
        val_losses,
        train_losses,
        val_losses,
        labels=["Train loss", "Val loss", "a", "b"],
        cfg=PlotConfig(
            #title="Transformer — time series anomaly detection",
            title = None,
            figsize = (9, 2),
            xlabel = "Time Step",
            ylabel = None,
            # smoothing=0.6,
            # log_scale=True,
            # save_path="loss_curve.png",
        ),
    )