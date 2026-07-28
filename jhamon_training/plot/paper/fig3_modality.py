"""Figure 3 -- how the two modalities load the knee across the repetition.

This is the mechanistic core of the paper: at an equated dose, isokinetic
loading is front-loaded and Nordic loading is back-loaded, and the velocity
profiles mirror that.

UNIT OF ANALYSIS
----------------
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
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (
    BAND,
    COL_DOUBLE,
    GROUP_COLOR,
    GROUP_LABEL,
    INK,
    INK_SOFT,
    REFERENCE,
    panel_letter,
    save,
    use_paper_style,
)

#: (variable in the curve frame, axis label, panel titles).
VARIABLES = (
    ("torque", "Knee torque (N·m)", "Torque"),
    ("knee_v", "Knee angular velocity (°·s$^{-1}$)", "Angular velocity"),
)


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
    """Run the participant-averaged two-sample SPM{t} for both variables."""
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
            "z": np.asarray(ti.z, dtype=float),
            "zstar": float(ti.zstar),
            "df": [float(v) for v in np.ravel(ti.df)],
            "clusters": clusters(ti),
        }
    return out


def _plot_curves(ax, res: dict, ylabel: str) -> None:
    x = np.arange(101)

    # Supra-threshold regions first, as a recessive band behind the data.
    for c in res["clusters"]:
        ax.axvspan(c["start_pct"], c["end_pct"], color=BAND, alpha=0.28, lw=0, zorder=0)

    lo, hi = np.inf, -np.inf
    for group, key in (("NH", "nh"), ("IK", "ik")):
        y = res[key]
        mean, sd = y.mean(axis=0), y.std(axis=0, ddof=1)
        colour = GROUP_COLOR[group]
        ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.16, lw=0, zorder=1)
        ax.plot(x, mean, color=colour, lw=1.9, zorder=3,
                label=f"{GROUP_LABEL[group]}, n = {y.shape[0]}")
        lo, hi = min(lo, float((mean - sd).min())), max(hi, float((mean + sd).max()))

    # The +-1 SD envelope must fit, and the legend needs a clear band above it.
    span = hi - lo
    ax.set_ylim(lo - 0.04 * span, hi + 0.20 * span)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", frameon=False, fontsize=7.2)


#: A cluster narrower than this carries no interpretable extent, so it gets a
#: marker and a footnote rather than a label that would crowd the panel. The
#: 0.0-0.7% velocity cluster is one sample point wide and the Results text
#: explicitly declines to interpret it.
MIN_LABELLED_EXTENT_PCT = 2.0


def _plot_spm(ax, res: dict, ylabel: str) -> None:
    x = np.arange(101)
    z, zstar = res["z"], res["zstar"]

    ax.axhline(0, color=REFERENCE, lw=0.7, zorder=1)
    for sign in (+1, -1):
        ax.axhline(sign * zstar, color=REFERENCE, lw=0.9, ls=(0, (4, 3)), zorder=2)

    ax.fill_between(x, z, zstar, where=z >= zstar, color=GROUP_COLOR["NH"],
                    alpha=0.35, lw=0, interpolate=True, zorder=2)
    ax.fill_between(x, z, -zstar, where=z <= -zstar, color=GROUP_COLOR["IK"],
                    alpha=0.35, lw=0, interpolate=True, zorder=2)
    ax.plot(x, z, color=INK, lw=1.4, zorder=4)

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


def make_figure(training_df: pd.DataFrame, out_dir: Path | None = None,
                *, stem: str = "Figure_3"):
    use_paper_style()
    res = compute(training_df)

    fig, axes = plt.subplots(
        2, 2, figsize=(COL_DOUBLE, 5.0), sharex="col",
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.16, "wspace": 0.36},
    )

    letters = [["A", "B"], ["C", "D"]]
    for col, (var, ylabel, title) in enumerate(VARIABLES):
        _plot_curves(axes[0][col], res[var], ylabel)
        axes[0][col].set_title(title, color=INK, pad=6)
        _plot_spm(axes[1][col], res[var], "SPM{$t$}")
        for row in (0, 1):
            panel_letter(axes[row][col], letters[row][col], x=-0.17, y=1.02)

    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    if out_dir is None:
        return fig, res, None
    written = save(fig, out_dir, stem)
    plt.close(fig)
    return None, res, written


def describe(res: Dict[str, dict]) -> str:
    """One line per cluster, in the same shape the stats JSON stores them."""
    lines = []
    for var, _label, _title in VARIABLES:
        r = res[var]
        lines.append(
            f"{var:>8}  n = {r['nh'].shape[0]} vs {r['ik'].shape[0]}  "
            f"t* = {r['zstar']:.3f}  df = {r['df']}"
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
