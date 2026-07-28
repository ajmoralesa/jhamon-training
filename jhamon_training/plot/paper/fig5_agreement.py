"""Figure 5 -- did the standardization deliver what it prescribed?

The paper's first aim. Two quantities were equated by design and each gets one
Bland-Altman panel:

    A   mechanical work per set (J)
    B   mean knee angular velocity (deg/s), Nordic whole-range mean against the
        isokinetic constant-velocity phase

UNIT OF ANALYSIS
----------------
NH and IK were *between-participant* groups, so there is no participant-to-
participant correspondence and a conventional paired Bland-Altman on individual
data would be meaningless. What the procedure paired was the **prescription**:
the target work for a given session x set cell came from the NH group mean for
that same cell, and the dynamometer velocity for that cell was programmed to
the NH group mean for that cell. The unit is therefore the session x set cell
(73 complete pairs), NH group mean minus IK group mean.

DATASET
-------
All repetitions performed -- not the analysed subset. The cumulative-work
criterion that terminated each isokinetic set applied to every repetition
performed, so that is the population the standardization actually targeted.
The analysed subset additionally passes a 10-35 deg/s velocity window, which
filters on one of the two quantities under test and moves the group means
towards each other; using it here would manufacture part of the agreement.

WHAT IS DRAWN
-------------
Bias with its 95% CI, the 95% limits of agreement with their CIs, and -- where
the test for proportional bias is significant -- the regression-based limits
alongside the fixed ones, because fixed limits are wrong at both ends of the
range when the difference varies with the magnitude.

There is deliberately no concordance-correlation panel: the CCC is reported in
the text, and an identity scatter beside a Bland-Altman plot of the same 73
pairs shows the reader nothing the difference plot has not already shown.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (
    COL_DOUBLE,
    GRID,
    GROUP_COLOR_NAME,
    IK,
    INK,
    INK_SOFT,
    NH,
    REFERENCE,
    panel_letter,
    save,
    use_paper_style,
)

#: (column, rep aggregation, panel title, axis unit, decimals, stats prefix).
QUANTITIES = (
    ("work", "sum", "Mechanical work per set", "J", 0, "bav_all_work"),
    ("knee_v_iso", "pooled_mean", "Mean knee angular velocity",
     "°·s$^{-1}$", 2, "bav_all_viso"),
)

#: Proportional bias is called significant, and regression limits drawn, below
#: this p. It is the same alpha the Methods use everywhere else.
ALPHA = 0.05


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def compute(disc_all: pd.DataFrame) -> Dict[str, dict]:
    """Cell-level agreement for both standardised quantities, from the same code
    path the manuscript statistics use."""
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from extract_validity_stats import regression_limits

    from jhamon_training.stats.agreement import compute_agreement

    out: Dict[str, dict] = {}
    for var, agg, _title, _unit, _nd, _prefix in QUANTITIES:
        res = compute_agreement(disc_all, var, rep_aggregation=agg)
        pairs = res["pairs"]
        res["reg"] = regression_limits(
            pairs["mean"].to_numpy(float), pairs["diff"].to_numpy(float)
        )
        out[var] = res
    return out


def _panel(ax, res: dict, *, title: str, unit: str, nd: int) -> None:
    pairs = res["pairs"]
    m = pairs["mean"].to_numpy(float)
    d = pairs["diff"].to_numpy(float)
    cell = res  # compute_agreement returns a flat summary dict

    x_lo, x_hi = float(m.min()), float(m.max())
    pad = 0.06 * (x_hi - x_lo)
    x0, x1 = x_lo - pad, x_hi + pad
    grid = np.linspace(x0, x1, 200)

    # Zero first: the reference the whole panel is read against.
    ax.axhline(0, color=REFERENCE, lw=0.8, zorder=2)

    # Confidence bands for the bias and for each limit of agreement.
    for lo, hi, colour in (
        (cell["bias_ci_low"], cell["bias_ci_high"], INK),
        (cell["loa_low_ci_low"], cell["loa_low_ci_high"], INK_SOFT),
        (cell["loa_high_ci_low"], cell["loa_high_ci_high"], INK_SOFT),
    ):
        ax.axhspan(lo, hi, color=colour, alpha=0.10, lw=0, zorder=1)

    ax.axhline(cell["bias"], color=INK, lw=1.5, zorder=4)
    for limit in (cell["loa_low"], cell["loa_high"]):
        ax.axhline(limit, color=INK_SOFT, lw=1.1, ls=(0, (5, 3)), zorder=4)

    # Regression-based limits, drawn only when the proportional bias is real.
    # A shaded ribbon, not a third and fourth dashed line: it has to be
    # distinguishable at a glance from the fixed limits it is being compared to,
    # and it must not borrow a group hue, because nothing in this panel is a
    # group -- every mark is a session x set cell.
    prop_p = float(cell["prop_bias_p"])
    if prop_p < ALPHA:
        reg = res["reg"]
        fit = reg["reg_intercept"] + reg["reg_slope"] * grid
        ax.fill_between(
            grid, fit - reg["reg_half_width"], fit + reg["reg_half_width"],
            color=INK_SOFT, alpha=0.10, lw=0, zorder=1,
        )
        ax.plot(grid, fit, color=INK_SOFT, lw=1.0, ls=(0, (1, 1.6)), zorder=5)

    # The difference is signed and the sign is the whole point: above the zero
    # line the Nordic cell mean exceeded the isokinetic one, below it the
    # reverse. Encoding that with the paper's two group hues about a neutral
    # zero is a diverging scale whose poles the reader already knows.
    above = d >= 0
    for mask, colour in ((above, NH), (~above, IK)):
        if mask.any():
            ax.scatter(
                m[mask], d[mask], s=22, facecolor=colour, edgecolor="white",
                linewidth=0.6, alpha=0.9, zorder=6,
            )

    ax.set_xlim(x0, x1)
    ax.set_title(title, color=INK, pad=6)
    ax.set_xlabel(f"Mean of NH and IK ({unit})")
    ax.set_ylabel(f"Difference, NH − IK ({unit})")

    # Right-hand labels for the three horizontal references, outside the axes so
    # they can never sit on a data point.
    for value, text in (
        (cell["loa_high"], f"+1.96 SD  {cell['loa_high']:+.{nd}f}"),
        (cell["bias"], f"bias  {cell['bias']:+.{nd}f}"),
        (cell["loa_low"], f"−1.96 SD  {cell['loa_low']:+.{nd}f}"),
    ):
        ax.annotate(
            text, xy=(1.01, value), xycoords=("axes fraction", "data"),
            color=INK_SOFT, fontsize=6.6, va="center", ha="left",
            annotation_clip=False, zorder=7,
        )

    p_bias = cell["bias_p"]
    p_txt = "p < 0.001" if p_bias < 0.001 else f"p = {p_bias:.3f}"
    prop_txt = "p < 0.001" if prop_p < 0.001 else f"p = {prop_p:.3f}"
    summary = (
        f"{int(cell['n_pairs'])} session × set cells\n"
        f"bias {cell['bias']:+.{nd}f} [{cell['bias_ci_low']:+.{nd}f}, "
        f"{cell['bias_ci_high']:+.{nd}f}], {p_txt}\n"
        f"{cell['pct_bias']:+.2f}% of the {cell['grand_mean']:.{nd}f} {unit} grand mean\n"
        f"CCC {cell['ccc']:.3f} [{cell['ccc_ci_low']:.3f}, {cell['ccc_ci_high']:.3f}]\n"
        f"proportional bias: slope {cell['prop_bias_slope']:.3f}, {prop_txt}"
    )
    # Reserve a clear band at the top of the panel for the summary, so it never
    # has to be placed over the cells or over a limit line.
    y_lo = min(float(d.min()), cell["loa_low_ci_low"])
    y_hi = max(float(d.max()), cell["loa_high_ci_high"])
    span = y_hi - y_lo
    ax.set_ylim(y_lo - 0.06 * span, y_hi + 0.46 * span)
    ax.text(
        0.02, 0.98, summary, transform=ax.transAxes, color=INK,
        fontsize=6.6, ha="left", va="top", linespacing=1.45, zorder=7,
    )


def make_figure(disc_all: pd.DataFrame, out_dir: Path | None = None,
                *, stem: str = "Figure_5"):
    use_paper_style()
    res = compute(disc_all)

    fig, axes = plt.subplots(
        1, 2, figsize=(COL_DOUBLE, 3.2),
        gridspec_kw={"wspace": 0.55},
    )
    for ax, (var, _agg, title, unit, nd, _prefix) in zip(axes, QUANTITIES):
        _panel(ax, res[var], title=title, unit=unit, nd=nd)
    panel_letter(axes[0], "A", x=-0.20, y=1.03)
    panel_letter(axes[1], "B", x=-0.20, y=1.03)

    fig.text(
        0.5, -0.09,
        "Each point is one session × set cell (NH group mean − IK group mean); "
        f"{GROUP_COLOR_NAME['NH']} where NH was greater, {GROUP_COLOR_NAME['IK']} where IK was.\n"
        "Solid line, mean difference; dashed lines, 95% limits of agreement; grey shading, 95% CI of each.\n"
        "Dotted line and ribbon in A, regression-based limits, drawn only where the proportional bias is significant.",
        ha="center", va="top", color=INK_SOFT, fontsize=6.8, linespacing=1.4,
    )

    if out_dir is None:
        return fig, res, None
    written = save(fig, out_dir, stem)
    plt.close(fig)
    return None, res, written


def describe(res: Dict[str, dict]) -> str:
    lines = []
    for var, _agg, _title, unit, nd, prefix in QUANTITIES:
        c = res[var]
        lines.append(
            f"{prefix:>16}  n = {int(c['n_pairs'])}  "
            f"bias {c['bias']:+.{nd}f} [{c['bias_ci_low']:+.{nd}f}, {c['bias_ci_high']:+.{nd}f}] "
            f"({c['pct_bias']:+.2f}%)  LoA [{c['loa_low']:.{nd}f}, {c['loa_high']:+.{nd}f}]  "
            f"CCC {c['ccc']:.3f}  prop slope {c['prop_bias_slope']:.4f} p={c['prop_bias_p']:.4f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    staged = Path(sys.argv[1])
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports")
    disc = pd.read_parquet(staged / "disc_all.parquet")
    _, result, paths = make_figure(disc, target)
    print(describe(result))
    for p in paths:
        print(f"wrote {p}")
