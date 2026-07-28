"""One visual language for every figure of the training-standardization paper.

WHY THIS EXISTS
---------------
Before this module the paper mixed three unrelated styles: the R panels used
cividis navy/yellow, the matplotlib panels used the default blue/orange cycle,
and the between-group SPM figure came from a 2020 script with its own defaults.
Reviewers read a figure set as one object, so the palette, the type scale and
the panel lettering are defined once, here, and every figure imports them.

THE PALETTE
-----------
Group is the paper's one categorical dimension outside the Figure 2 methods
panel, so there are exactly two hues:

    NH  #A62B36  burgundy
    IK  #008B85  teal

The pair was chosen against four alternatives (Okabe-Ito vermillion/blue, NEJM
brick/blue, a deepened Lancet wine/petrol, and Brewer purple/green) by drawing
Figures 3 and 6 in each and testing all five the same way. All five clear the
categorical colour checks in colour, so the decision fell to the test a colour
validator does not run: journals still print some figures in black and white,
and a pair separated only by hue collapses there. Burgundy/teal has the widest
greyscale separation of the five (relative-luminance gap 0.101, against 0.069
for Okabe-Ito and 0.013 for the NEJM pair, which merges outright in B&W).

In colour it clears every check against a white surface -- CVD deltaE 11.8
deutan / 31.3 tritan, normal-vision 27.2, both >= 3:1 contrast. One deliberate
exception: the teal's chroma is 0.099 against a 0.10 floor. That floor is
calibrated for screen dashboards; on a printed page the slight desaturation is
what makes it read as ink rather than as a UI colour.

Figure 2 is the one panel that colours something other than group -- it draws
three signal channels from a single repetition -- and it therefore uses a third
set of hues, kept clear of both group hues on purpose so that no reader carries
"teal = IK" from Figure 3 into "teal = velocity" in Figure 2. Those live in
``plot/figure_2_methods_signals.py``, next to the figure that owns them.

Training session is a *magnitude*, not an identity, so it is encoded with a
single-hue sequential ramp per group -- light early sessions to dark late ones,
anchored on that group's own hue. This keeps Figure 6 tied to the group
identity established in Figure 3 instead of introducing a third colour scheme.

Everything that is not data -- grid, spines, tick labels, annotation text --
is a neutral grey, so the ink that carries meaning is the only saturated ink on
the page.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# --------------------------------------------------------------------------- #
# colour
# --------------------------------------------------------------------------- #
#: Group identity. Assigned in fixed order and never cycled.
NH = "#A62B36"          # burgundy
IK = "#008B85"          # teal
GROUP_COLOR = {"NH": NH, "IK": IK}

#: Plain-English name of each group's hue. Captions that tell the reader which
#: colour means what must read it from here rather than spell it out, so that
#: changing the palette above cannot leave a caption describing the old one --
#: which is exactly what happened to Figure 5 when this palette was adopted.
GROUP_COLOR_NAME = {"NH": "burgundy", "IK": "teal"}

#: Long-form group labels, used wherever a legend or an axis names a group.
GROUP_LABEL = {
    "NH": "Nordic hamstring (NH)",
    "IK": "Isokinetic (IK)",
}
GROUP_LABEL_SHORT = {"NH": "NH", "IK": "IK"}

#: Neutrals. Text tokens are never a series colour.
INK = "#1A1A1A"          # primary text
INK_SOFT = "#52514E"     # secondary text, annotations
GRID = "#DCDCDA"         # grid lines
SPINE = "#8A8A86"        # axis spines and ticks
BAND = "#B9B9B4"         # supra-threshold shading, reference bands
SURFACE = "#FFFFFF"

#: Zero / identity / target reference lines. One neutral, one weight.
REFERENCE = "#3D3D3A"


def session_ramp(group: str, n: int = 15) -> list[str]:
    """A light-to-dark sequential ramp in one group's hue.

    Sessions are ordered magnitudes, so the encoding is lightness, not hue. The
    ramp starts well short of white (L ~ 0.88) so session 1 is still visible on
    a white surface, and ends short of black so the darkest curve keeps its hue.
    """
    base = np.array(to_rgb(GROUP_COLOR[group]))
    light = base + (1.0 - base) * 0.80          # near-white tint of the hue
    dark = base * 0.42                          # deep shade of the same hue
    cmap = LinearSegmentedColormap.from_list(f"{group}_sessions", [light, base, dark])
    return [mpl.colors.to_hex(cmap(x)) for x in np.linspace(0.0, 1.0, n)]


# --------------------------------------------------------------------------- #
# type and geometry
# --------------------------------------------------------------------------- #
#: Journal column widths in inches (single / 1.5 / double column).
COL_SINGLE = 3.35
COL_ONE_HALF = 5.0
COL_DOUBLE = 7.0

RC = {
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 400,
    "figure.dpi": 110,
    "axes.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
    "xtick.color": SPINE,
    "ytick.color": SPINE,
    "xtick.labelcolor": INK_SOFT,
    "ytick.labelcolor": INK_SOFT,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "legend.labelcolor": INK,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6,
    "legend.columnspacing": 1.4,
    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
    "patch.linewidth": 0.6,
    "pdf.fonttype": 42,       # embed TrueType so the PDF stays editable
    "ps.fonttype": 42,
}


def use_paper_style() -> None:
    """Install the paper's rcParams. Call once at the top of every figure script."""
    mpl.rcParams.update(RC)


def panel_letter(ax, letter: str, *, x: float = -0.16, y: float = 1.06, size: float = 10.5):
    """Put a bold panel letter outside the axes, in the same place on every panel."""
    return ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=size, fontweight="bold", color=INK,
        ha="left", va="bottom",
    )


def tidy(ax, *, grid_axis: str = "both") -> None:
    """Recessive grid on the requested axis only, and nowhere else."""
    ax.grid(False)
    if grid_axis in ("both", "x"):
        ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    if grid_axis in ("both", "y"):
        ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def group_legend(ax, groups: Sequence[str] = ("NH", "IK"), **kwargs):
    """A legend that names the groups in full, in the fixed NH-then-IK order."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=GROUP_COLOR[g], lw=2.2, label=GROUP_LABEL[g])
        for g in groups
    ]
    defaults = dict(loc="best", frameon=False)
    defaults.update(kwargs)
    return ax.legend(handles=handles, **defaults)


def save(fig, out_dir, stem: str, *, pdf: bool = True) -> list:
    """Write ``stem.png`` (and ``stem.pdf``) into ``out_dir`` and report the paths."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    png = out_dir / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight", facecolor=SURFACE)
    written.append(png)
    if pdf:
        pdf_path = out_dir / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=SURFACE)
        written.append(pdf_path)
    return written
