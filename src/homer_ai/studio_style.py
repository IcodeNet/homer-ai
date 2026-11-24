"""One shared look for every graph in the course notebooks.

The look: a dark navy glow, a small orange "eyebrow" heading, a bold white
title, green for good and pink-red for bad.

Use it like this in a notebook:

    from homer_ai.studio_style import GOOD, studio_axes, studio_figure

    fig = studio_figure("Films as arrows", "Same direction, high score")
    ax = studio_axes(fig)
    ...

For boxes-and-arrows pictures call `diagram_axes(fig)` instead of
`studio_axes(fig)`, then draw with `box`, `chips`, `arrow`, `plus`, `tag`,
`bracket`, `stack_of_floors` and `ring` (see the end of this file).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ACCENT = "#fdba74"  # orange: eyebrows, and the main thing in a picture
GOOD = "#86efac"  # green: good, similar, positive
BAD = "#fca5a5"  # pink-red: bad, opposite, negative
MUTED = "#94a3b8"  # slate grey: labels and less important text
INK = "#f8fafc"  # near-white: titles and numbers
GLOW = "#1e3a8a"  # navy: the bright part of the background
BACKGROUND = "#0f172a"  # dark navy: the rest of the background
GRID = "#1e293b"  # faint grid lines
EDGE = "#334155"  # axes edges

# Red → dark → green, for numbers that can be negative or positive
GOOD_BAD = LinearSegmentedColormap.from_list("good_bad", [BAD, GRID, GOOD])

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "savefig.facecolor": BACKGROUND,
    }
)


def studio_figure(eyebrow, title, size=(9, 6)):
    """A new figure with the navy glow, the orange eyebrow and the white title."""
    fig = plt.figure(figsize=size, dpi=120)
    background = fig.add_axes((0, 0, 1, 1), zorder=-10)
    background.axis("off")
    y, x = np.mgrid[0:1:300j, 0:1:300j]
    distance = np.clip(np.hypot(x - 0.3, (y - 0.2) / 0.6), 0, 1)  # glow at top-left
    glow = LinearSegmentedColormap.from_list("glow", [GLOW, BACKGROUND])
    background.imshow(glow(distance), extent=(0, 1, 0, 1), aspect="auto")
    fig.text(0.07, 0.93, eyebrow.upper(), color=ACCENT, fontsize=10, fontweight="bold")
    fig.text(0.07, 0.87, title, color=INK, fontsize=20, fontweight="bold")
    return fig


def studio_axes(fig, rect=(0.1, 0.1, 0.8, 0.7)):
    """Axes for a figure from studio_figure: see-through, with faint grid and edges."""
    ax = fig.add_axes(rect)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.grid(color=GRID, linewidth=0.8)
    return ax


def text_on(value, limit):
    """Dark text on a bright square, white text on a dark one (for heatmaps)."""
    return BACKGROUND if abs(value) > 0.55 * limit else INK


# ------------------------------------------------------------------ drawings
# Small helpers for boxes-and-arrows pictures. They draw on axes made by
# `diagram_axes`, where x and y are plain numbers (x from 0 to 10), so a layout
# is just a few coordinates. Use orange (ACCENT) only for the one thing the
# reader should look at.


def diagram_axes(fig, rect=(0.04, 0.03, 0.92, 0.80), xlim=(0, 10), ylim=None):
    """Axes with no frame and no grid, for drawings. x runs over `xlim`;
    y runs from 0 up, as high as the figure shape allows (equal-sized units)."""
    ax = fig.add_axes(rect)
    width_units = xlim[1] - xlim[0]
    if ylim is None:
        shape = (rect[3] * fig.get_figheight()) / (rect[2] * fig.get_figwidth())
        ylim = (0, width_units * shape)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax


def _text_style(mono, fontsize, color, weight="normal"):
    style = {"color": color, "fontsize": fontsize, "fontweight": weight}
    if mono:
        style["family"] = "monospace"
    return style


def box(
    ax,
    x,
    y,
    w,
    h,
    text="",
    color=EDGE,
    text_color=INK,
    mono=False,
    fontsize=11,
    fill=GRID,
    alpha=0.9,
    lw=1.5,
    weight="normal",
    dashed=False,
):
    """Rounded rectangle centred on (x, y), with centred text. Returns the patch."""
    rounding = min(w, h) * 0.18
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=fill,
        edgecolor=color,
        linewidth=lw,
        alpha=alpha,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)
    if text:
        style = _text_style(mono, fontsize, text_color, weight)
        ax.text(
            x, y, text, ha="center", va="center", zorder=3, linespacing=1.35, **style
        )
    return patch


def chips(
    ax, x, y, labels, size=0.8, gap=0.1, colors=None, mono=True, fontsize=12, fill=GRID
):
    """A row of small equal squares (letters, IDs, tokens). `x` is the left edge,
    `y` the middle. `colors` gives each square's border. Returns the centres."""
    centres = []
    for i, label in enumerate(labels):
        cx = x + size / 2 + i * (size + gap)
        color = colors[i] if colors else EDGE
        box(
            ax,
            cx,
            y,
            size,
            size,
            label,
            color=color,
            mono=mono,
            fontsize=fontsize,
            fill=fill,
        )
        centres.append((cx, y))
    return centres


def arrow(
    ax,
    start,
    end,
    label=None,
    color=MUTED,
    width=1.5,
    curved=0.0,
    dashed=False,
    label_offset=(0, 0.15),
    fontsize=9,
    head=12,
    label_color=None,
    mono=False,
    zorder=1,
):
    """An arrow from `start` to `end` (each an (x, y) pair). `curved` bends it
    (shortcut loops); `label` is written beside its middle."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=head,
        linewidth=width,
        color=color,
        connectionstyle=f"arc3,rad={curved}",
        linestyle="--" if dashed else "-",
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        style = _text_style(mono, fontsize, label_color or color)
        ax.text(mx, my, label, ha="center", va="center", zorder=4, **style)
    return patch


def plus(ax, x, y, r=0.25, color=ACCENT):
    """A small circle with a plus: where a shortcut joins, or token + position."""
    ax.add_patch(
        Circle(
            (x, y), r, facecolor=BACKGROUND, edgecolor=color, linewidth=1.8, zorder=3
        )
    )
    ax.text(
        x,
        y,
        "+",
        ha="center",
        va="center",
        color=color,
        fontsize=13,
        fontweight="bold",
        zorder=4,
    )


def tag(
    ax, x, y, text, color=MUTED, fontsize=8, ha="center", mono=False, weight="normal"
):
    """A small caption, like a notebook name or a shape such as (1, 5, 32)."""
    style = _text_style(mono, fontsize, color, weight)
    ax.text(x, y, text, ha=ha, va="center", zorder=4, **style)


def bracket(ax, x0, x1, y, text, color=ACCENT, below=False, fontsize=10, mono=False):
    """A horizontal bracket from x0 to x1 at height y, with a label.
    The little ends point down (or up, when `below` is True)."""
    end = 0.15 if below else -0.15
    ax.plot(
        [x0, x0, x1, x1], [y + end, y, y, y + end], color=color, linewidth=2, zorder=3
    )
    shift = -0.3 if below else 0.3
    tag(ax, (x0 + x1) / 2, y + shift, text, color=color, fontsize=fontsize, mono=mono)


def stack_of_floors(ax, x, y, w, floors, floor_h=1.0, gap=0.15, highlight=None):
    """Floors drawn from the bottom up, starting at height `y`.
    `floors` is a list of (title, subtitle, color); `highlight` is the index of
    one floor to give an orange border. Returns each floor's middle height."""
    middles = []
    for i, (title, subtitle, color) in enumerate(floors):
        cy = y + floor_h / 2 + i * (floor_h + gap)
        edge = ACCENT if i == highlight else color
        box(ax, x, cy, w, floor_h, color=edge, lw=2.4 if i == highlight else 1.5)
        if subtitle:
            tag(
                ax, x, cy + floor_h * 0.17, title, color=INK, fontsize=11, weight="bold"
            )
            tag(ax, x, cy - floor_h * 0.22, subtitle, fontsize=8.5)
        else:
            tag(ax, x, cy, title, color=INK, fontsize=11, weight="bold")
        middles.append(cy)
    return middles


def ring(
    ax,
    cx,
    cy,
    rx,
    ry,
    labels,
    colors=None,
    box_w=2.6,
    box_h=0.9,
    fontsize=10,
    mono=False,
):
    """Boxes placed round an ellipse, first at the top, going clockwise, joined by
    curved arrows (a loop that repeats). Returns the boxes' centres."""
    count = len(labels)
    centres = []
    patches = []
    for i, label in enumerate(labels):
        angle = np.pi / 2 - 2 * np.pi * i / count
        px, py = cx + rx * np.cos(angle), cy + ry * np.sin(angle)
        color = colors[i] if colors else EDGE
        patches.append(
            box(
                ax,
                px,
                py,
                box_w,
                box_h,
                label,
                color=color,
                fontsize=fontsize,
                mono=mono,
            )
        )
        centres.append((px, py))
    for i in range(count):
        link = FancyArrowPatch(
            posA=centres[i],
            posB=centres[(i + 1) % count],
            patchA=patches[i],
            patchB=patches[(i + 1) % count],
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=MUTED,
            connectionstyle="arc3,rad=-0.25",
            shrinkA=3,
            shrinkB=3,
            zorder=1,
        )
        ax.add_patch(link)
    return centres


def dot(ax, x, y, text="", r=0.4, color=EDGE, fontsize=11, text_color=INK, mono=True):
    """A round dot (a node of a network) with a short text inside."""
    ax.add_patch(
        Circle((x, y), r, facecolor=GRID, edgecolor=color, linewidth=1.8, zorder=3)
    )
    if text:
        tag(ax, x, y, text, color=text_color, fontsize=fontsize, mono=mono)
