"""Figure 3 -- how the two modalities load the knee across the repetition.

This is the mechanistic core of the paper: at an equated dose, isokinetic
loading is front-loaded and Nordic loading is back-loaded, and the velocity
profiles mirror that.

WHAT THE TEST IS RUN ON
-----------------------
One curve per participant (13 NH vs 14 IK), obtained by averaging each
participant's repetitions across all sessions. The shipped analysis fed
``ttest2`` one curve per participant x session, treating ~15 curves from the
same person as independent and inflating the residual degrees of freedom
roughly fifteen-fold. The manuscript now reports the participant-averaged
contrast (``spm_*_paravg_*`` in ``training_stats.json``), so this figure is
drawn from the same contrast -- otherwise the figure and the text disagree
about where the supra-threshold clusters are.

The computation is imported from ``extract_manuscript_stats`` rather than
reimplemented, so there is exactly one definition of the test in the project.

WHAT THE CURVE PANELS DRAW, AND WHY IT IS A DIFFERENT UNIT
----------------------------------------------------------
Panels A and B draw one curve per participant per session -- 185 NH and 208 IK
-- not the 27 curves the test uses and not the 15 883 individual repetitions.
That middle unit is a deliberate choice about what the figure has to convey:

    27 curves      readable, but a mean +- SD band shows none of the work behind
                   the study and reads like any two-group comparison.
    15 883 curves  a texture rather than a set of curves; at the alpha needed to
                   keep it legible no single repetition is visible, and two
                   overlaid clouds of different hue mix into a third colour.
    393 curves     few enough that one line stays visible at alpha 0.14, many
                   enough that the panel obviously rests on far more than 27
                   observations.

The caveat this carries is not cosmetic. Participant x session is *exactly* the
unit whose misuse as an inferential unit the section above describes. Here it is
a drawing unit only: the SPM{t} in panels C and D is unchanged, still 13 vs 14
participant-averaged curves. So the panels must never label these curves "n",
and the caption states what one line is -- otherwise a reader reasonably infers
that the test had 393 observations.

THREE DRAWING DECISIONS WORTH KNOWING ABOUT
-------------------------------------------
*Fair layering.* Drawing all of one group and then all of the other puts one
group permanently on top; at alpha 0.14 that is visible as extra saturation, and
it would be read as the groups differing in consistency. The curves are
interleaved so neither group is systematically occluded.

*Robust limits.* A handful of session curves peak far above the mass (one NH
velocity curve reaches 270 deg/s against a 99.4th percentile near 200). Letting
them set the axis wastes a third of the panel, so the limits come from a high
percentile of the curve extrema and the caption counts what that crops. Nothing
is dropped silently.

*A direction-coded significance strip.* A full-height shaded span behind the
curves competes with the data, and where two clusters leave a gap the
*unshaded* strip reads as the highlighted one. A slim bar along the top edge,
coloured by which group is higher, ties the curve panel to the SPM panel below
and carries direction as well as extent.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from .style import (
    COL_DOUBLE,
    GROUP_COLOR,
    GROUP_LABEL,
    INK,
    INK_SOFT,
    REFERENCE,
    SURFACE,
    panel_letter,
    save,
    use_paper_style,
)

#: (variable in the curve frame, axis label, panel titles).
VARIABLES = (
    ("torque", "Knee torque (N·m)", "Torque"),
    ("knee_v", "Knee angular velocity (°·s$^{-1}$)", "Angular velocity"),
)

GROUPS = ("NH", "IK")

#: Node grid of the normalised curves, and the x vector every panel shares.
X = np.arange(101)

#: Alpha and width of a single participant x session curve. At ~200 curves per
#: group this alpha is an order of magnitude above what a 6 000-curve
#: repetition cloud tolerates, which is the point of the unit.
CURVE_ALPHA = 0.14
CURVE_LW = 0.55

#: Percentile of the curve extrema the y-axis is cut at, per side.
CLIP_Q = 99.4

#: Where the significance strip sits, in axes fraction. It is above the legend,
#: not behind it, so neither has to give way.
STRIP_BOTTOM = 0.968


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _import_analysis():
    """Import the manuscript's own SPM helpers from the repository root."""
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from extract_manuscript_stats import _clusters, _curve_matrix  # noqa: N813

    return _curve_matrix, _clusters


def compute(training_df: pd.DataFrame) -> Dict[str, dict]:
    """Run the participant-averaged two-sample SPM{t} for both variables.

    Returns, per variable: the participant matrices the test is run on
    (``nh``/``ik``, 13 and 14 rows -- ``make_paper_figures.py --verify`` checks
    those row counts against ``training_stats.json``), the participant x session
    matrices the curve panels draw (``nh_sessions``/``ik_sessions``), the number
    of repetitions behind them, and the inference.
    """
    import spm1d

    curve_matrix, clusters = _import_analysis()
    out: Dict[str, dict] = {}
    for var, _label, _title in VARIABLES:
        df = training_df[training_df["var"] == var]
        y_nh = curve_matrix(df, "NH", by_participant=True)
        y_ik = curve_matrix(df, "IK", by_participant=True)
        t = spm1d.stats.ttest2(y_nh, y_ik, equal_var=False)
        ti = t.inference(0.05, two_tailed=True)
        out[var] = {
            "nh": y_nh,
            "ik": y_ik,
            # Same helper, ``by_participant=False``: one row per participant x
            # session. Reusing it keeps a single definition of "a curve" in the
            # project, and it raises rather than silently averaging over a NaN.
            "nh_sessions": curve_matrix(df, "NH", by_participant=False),
            "ik_sessions": curve_matrix(df, "IK", by_participant=False),
            "n_reps": int(
                df.groupby(["par", "trses", "set", "rep"], observed=True).ngroups
            ),
            "z": np.asarray(ti.z, dtype=float),
            "zstar": float(ti.zstar),
            "df": [float(v) for v in np.ravel(ti.df)],
            "clusters": clusters(ti),
        }
    return out


# --------------------------------------------------------------------------- #
# curve panels
# --------------------------------------------------------------------------- #
def _limits(res: dict) -> Tuple[float, float, int]:
    """Robust y-limits, plus how many curves they crop."""
    pooled = np.concatenate([res[f"{g.lower()}_sessions"] for g in GROUPS], axis=0)
    tops, bottoms = pooled.max(axis=1), pooled.min(axis=1)
    hi = float(np.percentile(tops, CLIP_Q))
    lo = float(np.percentile(bottoms, 100 - CLIP_Q))
    span = hi - lo
    # The headroom carries the legend and the significance strip above it, so it
    # is larger than a plain data margin would need to be.
    lo, hi = lo - 0.05 * span, hi + 0.23 * span
    outside = int(((tops > hi) | (bottoms < lo)).sum())
    return lo, hi, outside


def _interleaved(res: dict) -> LineCollection:
    """One rasterized LineCollection holding both groups, alternating by row.

    Rasterizing keeps the vector PDF at a few hundred kB instead of tens of MB
    while the axes, text and mean curves stay vector and therefore editable.
    """
    ys = {g: res[f"{g.lower()}_sessions"] for g in GROUPS}
    segs, colours = [], []
    for i in range(max(y.shape[0] for y in ys.values())):
        for g in GROUPS:
            if i < ys[g].shape[0]:
                segs.append(np.column_stack([X, ys[g][i]]))
                colours.append(GROUP_COLOR[g])
    return LineCollection(segs, colors=colours, linewidths=CURVE_LW,
                          alpha=CURVE_ALPHA, zorder=2, rasterized=True,
                          capstyle="round")


def _significance_strip(ax, res: dict) -> None:
    """A slim bar along the top edge, coloured by which group is higher."""
    for c in res["clusters"]:
        colour = GROUP_COLOR["NH"] if c["direction"] > 0 else GROUP_COLOR["IK"]
        ax.axvspan(c["start_pct"], c["end_pct"], ymin=STRIP_BOTTOM, ymax=1.0,
                   color=colour, lw=0, zorder=7)


def _mean_line(ax, y: np.ndarray, group: str) -> None:
    """The group mean, with a surface halo so it survives the cloud.

    The means are not decoration here. NH torque is right-skewed, so its mean
    runs above the visual centre of its own band; without the line a reader
    takes the dense core for the average and puts NH some 20 N·m too low.
    """
    m = y.mean(axis=0)
    ax.plot(X, m, color=SURFACE, lw=2.9, zorder=5, solid_capstyle="round")
    ax.plot(X, m, color=GROUP_COLOR[group], lw=1.7, zorder=6, solid_capstyle="round")


def _plot_curves(ax, res: dict, ylabel: str) -> int:
    lo, hi, outside = _limits(res)
    ax.add_collection(_interleaved(res))
    for group in GROUPS:
        _mean_line(ax, res[f"{group.lower()}_sessions"], group)
    _significance_strip(ax, res)

    ax.set_ylim(lo, hi)
    ax.set_xlim(0, 100)
    ax.set_ylabel(ylabel)

    # The strip is explained in the caption rather than the legend: one neutral
    # swatch would misreport it (the strip is two-coloured) and two swatches
    # would double the legend's height for one line of information.
    handles = [
        Line2D([0], [0], color=GROUP_COLOR[g], lw=2.2,
               label=f"{GROUP_LABEL[g]} — {res[f'{g.lower()}_sessions'].shape[0]}"
                     " session curves")
        for g in GROUPS
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.7,
              handlelength=1.5, labelspacing=0.34, borderpad=0.1,
              bbox_to_anchor=(0.0, 0.955))
    return outside


# --------------------------------------------------------------------------- #
# SPM panels
# --------------------------------------------------------------------------- #
#: A cluster narrower than this carries no interpretable extent, so it gets a
#: marker and a footnote rather than a label that would crowd the panel. The
#: 0.0-0.7% velocity cluster is one sample point wide and the Results text
#: explicitly declines to interpret it.
MIN_LABELLED_EXTENT_PCT = 2.0


def _plot_spm(ax, res: dict, ylabel: str) -> None:
    z, zstar = res["z"], res["zstar"]

    ax.axhline(0, color=REFERENCE, lw=0.7, zorder=1)
    for sign in (+1, -1):
        ax.axhline(sign * zstar, color=REFERENCE, lw=0.9, ls=(0, (4, 3)), zorder=2)

    ax.fill_between(X, z, zstar, where=z >= zstar, color=GROUP_COLOR["NH"],
                    alpha=0.35, lw=0, interpolate=True, zorder=2)
    ax.fill_between(X, z, -zstar, where=z <= -zstar, color=GROUP_COLOR["IK"],
                    alpha=0.35, lw=0, interpolate=True, zorder=2)
    ax.plot(X, z, color=INK, lw=1.4, zorder=4)

    # Reserve a clear lane above and below the curve for the cluster labels, so
    # a label never has to be placed on top of the statistic it describes.
    lo, hi = float(np.min(z)), float(np.max(z))
    span = hi - lo
    ax.set_ylim(lo - 0.30 * span, hi + 0.30 * span)

    # Thresholds are named outside the axes; inside there is no room that is
    # reliably free in both panels.
    for value, name in ((zstar, "$t^*$ = " + f"{zstar:.2f}"), (-zstar, "$-t^*$")):
        ax.annotate(
            name, xy=(1.005, value), xycoords=("axes fraction", "data"),
            color=INK_SOFT, fontsize=6.8, va="center", ha="left",
            annotation_clip=False, zorder=5,
        )

    tiny = []
    for c in res["clusters"]:
        if c["extent_pct"] < MIN_LABELLED_EXTENT_PCT:
            tiny.append(c)
            continue
        # Centre on the cluster but keep the text inside the panel.
        mid = float(np.clip(0.5 * (c["start_pct"] + c["end_pct"]), 14, 86))
        who = "NH > IK" if c["direction"] > 0 else "IK > NH"
        p = "p < 0.001" if c["p"] < 0.001 else f"p = {c['p']:.3f}"
        above = c["direction"] > 0
        ax.annotate(
            f"{who}, {c['start_pct']:.1f}–{c['end_pct']:.1f}%\n{p}",
            xy=(mid, 0.985 if above else 0.015),
            xycoords=("data", "axes fraction"),
            color=INK, fontsize=6.6, ha="center",
            va="top" if above else "bottom", zorder=6,
        )

    if tiny:
        # The top-left corner is the one lane no cluster label can occupy: the
        # labels are centred on their clusters and clamped away from the edges.
        note = "\n".join(
            f"{c['start_pct']:.1f}–{c['end_pct']:.1f}%, p = {c['p']:.3f}\n"
            "single sample point, not interpreted"
            for c in tiny
        )
        ax.text(
            0.015, 0.985, note,
            transform=ax.transAxes, color=INK_SOFT, fontsize=6.2,
            ha="left", va="top", zorder=6,
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Repetition (%)")
    ax.set_xlim(0, 100)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def _caption(fig, res: Dict[str, dict], cropped: Dict[str, int]) -> None:
    """State the drawing unit, the inferential unit, and anything cropped.

    On the figure rather than only in the manuscript, because the two units
    differ and a figure that does not say so invites the reader to assume the
    test had 393 observations.
    """
    # Both variables come from the same repetitions, so either count is the
    # figure's total; taking the max rather than assuming they agree means a
    # future divergence understates nothing.
    n_reps = max(res[var]["n_reps"] for var, _l, _t in VARIABLES)
    blocks = [
        "A, B  one curve per participant per session, averaged over that session's "
        "repetitions (" + f"{n_reps:,}".replace(",", " ") + " repetitions in total). "
        "The bar along the top marks the supra-threshold regions of C and D, "
        "coloured by the group with the higher value.",
        "C, D  SPM{$t$} on participant-averaged curves — "
        f"{res[VARIABLES[0][0]]['nh'].shape[0]} NH vs "
        f"{res[VARIABLES[0][0]]['ik'].shape[0]} IK, unequal variances, two-tailed, "
        "$α$ = 0.05.",
    ]
    notes = []
    for var, _label, title in VARIABLES:
        n = cropped[var]
        if n:
            notes.append(f"{n} {title.lower()} curve{'s extend' if n > 1 else ' extends'}")
    if notes:
        blocks.append("; ".join(notes).capitalize() + " beyond the plotted range.")

    # Each sentence is wrapped on its own so "A, B" and "C, D" always begin a
    # line. Wrapping at all is not cosmetic: a run of text wider than the axes
    # makes ``bbox_inches="tight"`` widen the whole canvas, which silently
    # changes the figure's aspect ratio and shrinks every panel.
    lines = [line for block in blocks for line in textwrap.wrap(block, width=112)]
    fig.text(
        0.5, -0.02, "\n".join(lines),
        color=INK_SOFT, fontsize=6.3, ha="center", va="top", linespacing=1.55,
    )


def make_figure(training_df: pd.DataFrame, out_dir: Path | None = None,
                *, stem: str = "Figure_3"):
    use_paper_style()
    res = compute(training_df)

    fig, axes = plt.subplots(
        2, 2, figsize=(COL_DOUBLE, 5.0), sharex="col",
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.18, "wspace": 0.36},
    )

    letters = [["A", "B"], ["C", "D"]]
    cropped: Dict[str, int] = {}
    for col, (var, ylabel, title) in enumerate(VARIABLES):
        cropped[var] = _plot_curves(axes[0][col], res[var], ylabel)
        axes[0][col].set_title(title, color=INK, pad=8)
        _plot_spm(axes[1][col], res[var], "SPM{$t$}")
        for row in (0, 1):
            panel_letter(axes[row][col], letters[row][col], x=-0.17, y=1.02)

    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    _caption(fig, res, cropped)

    if out_dir is None:
        return fig, res, None
    written = save(fig, out_dir, stem)
    plt.close(fig)
    return None, res, written


def describe(res: Dict[str, dict]) -> str:
    """One line per cluster, in the same shape the stats JSON stores them."""
    lines: List[str] = []
    for var, _label, _title in VARIABLES:
        r = res[var]
        lines.append(
            f"{var:>8}  n = {r['nh'].shape[0]} vs {r['ik'].shape[0]}  "
            f"t* = {r['zstar']:.3f}  df = {r['df']}"
        )
        lines.append(
            f"          drawn: {r['nh_sessions'].shape[0]} + "
            f"{r['ik_sessions'].shape[0]} participant x session curves "
            f"from {r['n_reps']:,} repetitions"
        )
        for i, c in enumerate(r["clusters"], start=1):
            who = "NH>IK" if c["direction"] > 0 else "IK>NH"
            lines.append(
                f"            c{i} {who} {c['start_pct']:.1f}-{c['end_pct']:.1f}%  p = {c['p']:.4f}"
            )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    staged = Path(sys.argv[1])
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports")
    df = pd.read_parquet(staged / "training_df.parquet")
    _, result, paths = make_figure(df, target)
    print(describe(result))
    for p in paths:
        print(f"wrote {p}")
