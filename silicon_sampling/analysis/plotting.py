"""Chart helpers, on a validated palette.

Colours are the reference data-viz palette, used by *job*: one hue (blue,
light→dark) wherever the quantity is a magnitude, a blue↔red diverging pair with
a neutral grey midpoint wherever it is signed, and the fixed categorical order
only where series identity matters.  The categorical trio used here validates on
every check; aqua sits below 3:1 on the light surface, so every chart that uses
it ships the numbers as a table beside it.

These render to PNG for markdown reports, so they commit to the light surface —
there is no viewer theme to follow.
"""

from __future__ import annotations

import os
from pathlib import Path

# ``/home/claude/.cache`` is root-owned in this container, so matplotlib falls back
# to a fresh temp dir on every import and says so, loudly.
from ..pfander.paths import CACHE  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#7a7973"
GRID = "#e5e4e0"

#: Fixed categorical order; never cycled, never re-ordered per chart.
CATEGORICAL = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)

#: Single-hue ramp for magnitude.
SEQUENTIAL = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#184f95",
    "#0d366b",
)

BLUE = "#2a78d6"
RED = "#d03b3b"
NEUTRAL = "#f0efec"

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red",
    ["#184f95", "#3987e5", "#9ec5f4", NEUTRAL, "#f0a3a3", "#e34948", "#a52a2a"],
)
MAGNITUDE = LinearSegmentedColormap.from_list("blue_seq", list(SEQUENTIAL))


def style() -> None:
    """Recessive axes, thin marks, no chartjunk."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def forest(
    labels,
    estimates,
    lows,
    highs,
    *,
    title: str,
    xlabel: str,
    path: Path,
    reference: float = 0.0,
    highlight=None,
    figsize=(7.6, 6.2),
) -> Path:
    """Point estimate + confidence interval per row, sorted by the caller."""
    style()
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))
    ax.axvline(reference, color=TEXT_MUTED, linewidth=1.0, zorder=1)
    for index in y:
        ax.plot(
            [lows[index], highs[index]],
            [index, index],
            color=BLUE,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=2,
            alpha=0.55,
        )
    colors = [
        RED if highlight and labels[i] in highlight else BLUE
        for i in range(len(labels))
    ]
    ax.scatter(
        estimates, y, s=42, color=colors, zorder=3, edgecolor=SURFACE, linewidth=1.4
    )
    for index in y:
        offset = (max(highs) - min(lows)) * 0.015
        ax.annotate(
            f"{estimates[index]:+.2f}",
            (highs[index] + offset, index),
            va="center",
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.8, len(labels) - 0.2)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.14)
    return save(fig, path)


def heatmap(
    matrix,
    row_labels,
    col_labels,
    *,
    title: str,
    cbar_label: str,
    path: Path,
    diverging: bool = True,
    annotate: bool = True,
    figsize=(11.5, 7.2),
    fmt: str = "{:+.1f}",
) -> Path:
    """Signed values on a blue↔red diverging scale; magnitudes on one hue."""
    style()
    data = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    if diverging:
        limit = float(np.nanmax(np.abs(data))) or 1.0
        image = ax.imshow(data, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    else:
        image = ax.imshow(data, cmap=MAGNITUDE, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=38, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.grid(visible=False)
    ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    # A 2px surface gap between cells, per the mark spec.
    ax.grid(which="minor", color=SURFACE, linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    if annotate:
        span = float(np.nanmax(np.abs(data))) or 1.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if np.isnan(value):
                    continue
                ax.annotate(
                    fmt.format(value),
                    (j, i),
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="#ffffff" if abs(value) > 0.62 * span else TEXT_PRIMARY,
                )
    bar = fig.colorbar(image, ax=ax, shrink=0.72, pad=0.015)
    bar.set_label(cbar_label, color=TEXT_SECONDARY, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, color=GRID)
    ax.set_title(title)
    return save(fig, path)


def hist_grid(
    frame,
    columns,
    *,
    title: str,
    path: Path,
    bins: int = 26,
    ncols: int = 4,
    ranges=None,
) -> Path:
    """One histogram per variable, shared styling, small multiples."""
    style()
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.35 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, columns):
        values = frame[column].dropna().to_numpy(dtype=float)
        span = (ranges or {}).get(column)
        ax.hist(
            values, bins=bins, range=span, color=BLUE, edgecolor=SURFACE, linewidth=0.5
        )
        ax.set_title(column, fontsize=9.5)
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=8)
        ax.annotate(
            f"M={values.mean():.1f}  SD={values.std(ddof=1):.1f}",
            (0.97, 0.92),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7.8,
            color=TEXT_SECONDARY,
        )
    for ax in axes[len(columns) :]:
        ax.set_visible(False)
    fig.suptitle(
        title,
        x=0.005,
        ha="left",
        fontsize=12,
        fontweight="semibold",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save(fig, path)


def bars(
    labels,
    values,
    *,
    title: str,
    xlabel: str,
    path: Path,
    figsize=(7.4, 4.6),
    fmt: str = "{:.3f}",
    color: str = BLUE,
) -> Path:
    """Horizontal magnitude bars with data-end labels."""
    style()
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    span = max(values) if len(values) else 1.0
    for index, value in enumerate(values):
        ax.annotate(
            fmt.format(value),
            (value + span * 0.012, index),
            va="center",
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.margins(x=0.12)
    return save(fig, path)


def grouped_lines(
    frame,
    x_labels,
    series: dict,
    *,
    title: str,
    ylabel: str,
    path: Path,
    figsize=(7.8, 4.8),
) -> Path:
    """Few series, fixed categorical hues, direct-labelled at the right edge."""
    style()
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(x_labels))
    for index, (name, values) in enumerate(series.items()):
        color = CATEGORICAL[index % len(CATEGORICAL)]
        ax.plot(
            x,
            values,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=name,
        )
        ax.annotate(
            name, (x[-1] + 0.08, values[-1]), va="center", fontsize=8.8, color=color
        )
    ax.set_xticks(x, x_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.margins(x=0.18)
    ax.legend(loc="best")
    return save(fig, path)
