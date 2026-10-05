"""Training-curve plots (matplotlib), styled per the reference dataviz palette:
categorical slots 1-2 (pre-validated as a set), recessive grid and axes, text
in ink tokens, legend + direct labels, one y-axis."""

from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"train": "#2a78d6", "dev": "#eb6834"}


def plot_history(result, title: str, ax=None):
    """Train vs dev per-token cross-entropy over optimizer steps (log y).

    ``result`` is a :class:`rag_etl.lora.TrainResult`; the best-dev step
    (the saved checkpoint) is marked with a vertical hairline."""
    import matplotlib.pyplot as plt

    h = result.history
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)
    fig = ax.figure
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for name, col in (("train", "train_loss"), ("dev", "dev_loss")):
        ax.plot(h["step"], h[col], color=SERIES[name], linewidth=2, marker="o",
                markersize=3.5, label=f"{name} error", zorder=3)
        ax.annotate(f"{name} {h[col].iloc[-1]:.3f}", (h["step"].iloc[-1], h[col].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK_SECONDARY, fontsize=9)

    ax.axvline(result.best_step, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=2)
    ax.annotate(f"saved: best dev {result.best_dev_loss:.3f} @ step {result.best_step}",
                (result.best_step, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -4), textcoords="offset points", va="top",
                color=INK_SECONDARY, fontsize=9)

    ax.set_yscale("log")
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel("optimizer step", color=MUTED)
    ax.set_ylabel("per-token cross-entropy (log scale)", color=MUTED)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.margins(x=0.08)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return fig
