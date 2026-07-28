"""Figure 6 -- what changed across the fifteen training sessions.

Two questions in one figure, because the Results text cites Figure 6 for both:

    A, B    the torque curves themselves, session by session, per group
    C, D    where each session differed from session 1 (paired SPM{t})
    E-I     the five discrete outcomes across the programme

WHY C AND D ARE A RASTER AND NOT FOURTEEN OVERLAID SPM CURVES
-------------------------------------------------------------
Each of the 14 session-1-versus-session-k contrasts is a *separate* paired test
with its own sample size (9 to 13 participants, because a participant must have
both sessions recorded) and therefore its own critical threshold t*. Overlaying
14 SPM{t} curves would need 14 different threshold lines, and the reader could
not tell which curve belonged to which threshold.

The raster answers the question the text actually asks -- which part of the
movement changed, and from which session onwards -- by showing, for every
comparison, the supra-threshold extent itself. Two tones separate the two alphas
the Results report: the dark bar is significant after the Bonferroni correction
across the 14 comparisons (alpha = 0.05/14 = 0.0036), the pale bar is
significant only at the uncorrected alpha = 0.05. That distinction carries the
paper's own caveat -- no cluster survives correction in the NH group at the
final session -- as a visible feature rather than a sentence.

PROVENANCE
----------
The cluster extents are read from ``manuscript_stats_detail.json``, which is
written by ``extract_manuscript_stats.py``; the discrete outcomes are the same
cells the mixed models are fitted to (top three repetitions by peak torque per
participant x session, from the analysed subset). Nothing here is recomputed
with a second definition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import (
    BAND,
    COL_DOUBLE,
    GRID,
    GROUP_COLOR,
    GROUP_LABEL,
    INK,
    INK_SOFT,
    REFERENCE,
    panel_letter,
    save,
    session_ramp,
    use_paper_style,
)

N_SESSIONS = 15

#: The five discrete outcomes, in the order the Results report them.
#: (column, panel label, unit, decimals, stats prefix in training_stats.json).
OUTCOMES = (
    ("angle_at_peak_torque", "Knee angle at peak torque", "°", 1, "lmm_angle_at_peak_torque"),
    ("peak_torque", "Peak torque", "N·m", 1, "lmm_peak_torque"),
    ("mean_torque", "Mean torque", "N·m", 1, "lmm_mean_torque"),
    ("knee_ROM", "Knee range of motion", "°", 1, "lmm_knee_ROM"),
    ("knee_v_iso", "Mean knee angular velocity", "°·s$^{-1}$", 1, "lmmiso_knee_v"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------- #
def session_curves(torque_curves: pd.DataFrame, group: str) -> np.ndarray:
    """Mean torque curve per session for one group: a (15 x 101) array.

    Uses the same top-three-repetitions selection as the progression SPM, then
    averages within participant and finally across participants, so a
    participant with more recorded sets does not weigh more heavily.
    """
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from extract_manuscript_stats import _select_top_curves

    sub = torque_curves[torque_curves["tr_group"] == group]
    top = _select_top_curves(sub)
    per_par = (
        top.groupby(["trses", "par", "timepoint"], observed=True)["value"]
        .mean()
        .groupby(["trses", "timepoint"], observed=True)
        .mean()
        .unstack("timepoint")
        .sort_index(axis=1)
    )
    out = np.full((N_SESSIONS, 101), np.nan)
    for k in range(1, N_SESSIONS + 1):
        key = f"tr_{k}"
        if key in per_par.index:
            out[k - 1] = per_par.loc[key].to_numpy(float)
    return out


def outcome_by_session(cells: pd.DataFrame, column: str) -> Dict[str, pd.DataFrame]:
    """Per-session mean and SD across participants, one table per group."""
    out = {}
    for group in ("NH", "IK"):
        sub = cells[cells["tr_group"] == group]
        agg = (
            sub.groupby("trses_num")[column]
            .agg(["mean", "std", "count"])
            .reindex(range(1, N_SESSIONS + 1))
        )
        out[group] = agg
    return out


def lmm_cells(disc_analysed: pd.DataFrame) -> pd.DataFrame:
    """The exact cells the mixed models are fitted to, plus the phase velocity."""
    columns = ["mean_torque", "peak_torque", "knee_ROM", "angle_at_peak_torque",
               "knee_v_mean", "knee_v_iso"]
    top3 = (
        disc_analysed.groupby(["par", "trses_num"], group_keys=False)
        .apply(lambda x: x.nlargest(3, "peak_torque"))
        .reset_index(drop=True)
    )
    return top3.groupby(["par", "tr_group", "trses_num"])[columns].mean().reset_index()


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def _plot_session_curves(ax, curves: np.ndarray, group: str) -> None:
    x = np.arange(101)
    ramp = session_ramp(group, N_SESSIONS)
    for k in range(N_SESSIONS):
        if np.isnan(curves[k]).all():
            continue
        ax.plot(x, curves[k], color=ramp[k], lw=1.15, zorder=2 + k * 0.01)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Repetition (%)")
    ax.set_ylabel("Knee torque (N·m)")
    ax.set_title(GROUP_LABEL[group], color=INK, pad=5)

    # A colour bar would need its own axes; a two-ended key is smaller and says
    # the same thing, because the only ordering that matters is early -> late.
    ax.plot([], [], color=ramp[0], lw=2.2, label="session 1")
    ax.plot([], [], color=ramp[-1], lw=2.2, label="session 15")
    ax.legend(loc="upper left", frameon=False, fontsize=7)


def _plot_raster(ax, detail: Dict[str, Any], group: str, alpha_bonf: float) -> None:
    key_unc = f"prog_{group.lower()}_unc"
    key_bonf = f"prog_{group.lower()}_bonf"
    colour = GROUP_COLOR[group]
    pale = session_ramp(group, 5)[1]

    n_labels: List[str] = []
    for k in range(2, N_SESSIONS + 1):
        row = k
        for source, tone, height in ((key_unc, pale, 0.72), (key_bonf, colour, 0.72)):
            entry = detail[source].get(f"tr_{k}", {})
            alpha = entry.get("alpha", 0.05)
            for c in entry.get("clusters_t", []):
                if c["p"] >= alpha:
                    continue
                ax.barh(
                    row, c["end_pct"] - c["start_pct"], left=c["start_pct"],
                    height=height, color=tone, edgecolor="none",
                    zorder=3 if source == key_bonf else 2,
                )
        entry = detail[key_bonf].get(f"tr_{k}", {})
        n_labels.append(str(entry.get("n", "")))

    # The number of paired participants varies from 9 to 13 across comparisons
    # and the Results say so, so it is on the figure. It goes *inside* the panel
    # at the left, where no cluster ever reaches, rather than in the margin,
    # where it would collide with the neighbouring panel's axis label.
    for k, label in zip(range(2, N_SESSIONS + 1), n_labels):
        ax.annotate(
            label, xy=(1.02, k), xycoords=("axes fraction", "data"),
            color=INK_SOFT, fontsize=6.0, va="center", ha="left",
            annotation_clip=False, zorder=5,
        )
    ax.annotate(
        "n =", xy=(1.02, N_SESSIONS + 0.85), xycoords=("axes fraction", "data"),
        color=INK_SOFT, fontsize=6.0, va="center", ha="left",
        annotation_clip=False, zorder=5,
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(1.3, N_SESSIONS + 1.3)
    ax.set_yticks([2, 5, 10, 15])
    ax.set_xlabel("Repetition (%)")
    ax.set_ylabel("Session vs session 1")
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.legend(
        handles=[
            Patch(facecolor=colour, label=f"$p$ < {alpha_bonf:.4f} (Bonferroni)"),
            Patch(facecolor=pale, label="$p$ < 0.05 (uncorrected)"),
        ],
        loc="lower left", frameon=False, fontsize=6.2, handlelength=1.2,
        handleheight=0.9, borderpad=0.1, bbox_to_anchor=(0.10, -0.02),
    )


def _plot_outcome(ax, cells: pd.DataFrame, column: str, label: str, unit: str,
                  nd: int, stats: Dict[str, Any], prefix: str) -> None:
    tables = outcome_by_session(cells, column)
    for group in ("NH", "IK"):
        agg = tables[group]
        x = agg.index.to_numpy(float)
        ax.errorbar(
            x, agg["mean"], yerr=agg["std"],
            color=GROUP_COLOR[group], lw=1.4, marker="o", ms=3.2,
            elinewidth=0.7, capsize=0, alpha=0.95, zorder=3,
        )
    ax.set_xlim(0.3, 15.7)
    ax.set_xticks([1, 5, 10, 15])
    ax.set_xlabel("Training session")
    ax.set_ylabel(f"{label} ({unit})")

    def p_of(effect: str) -> str:
        value = stats.get(f"{prefix}_{effect}_p")
        if value is None:
            return "—"
        return "< 0.001" if float(value) < 0.001 else f"= {float(value):.3f}"

    # These panels are a fifth of the page wide, so the omnibus tests go inside
    # on three short lines. As a title they overflow the axes and collide with
    # the neighbouring panel's letter.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.34 * (hi - lo))
    ax.text(
        0.03, 0.97,
        f"group $p$ {p_of('group')}\ntime $p$ {p_of('time')}\n"
        f"G × T $p$ {p_of('interaction')}",
        transform=ax.transAxes, color=INK_SOFT, fontsize=6.2,
        ha="left", va="top", linespacing=1.4, zorder=5,
    )


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def make_figure(torque_curves: pd.DataFrame, disc_analysed: pd.DataFrame,
                detail: Dict[str, Any], stats: Dict[str, Any],
                out_dir: Path | None = None, *, stem: str = "Figure_6"):
    use_paper_style()
    cells = lmm_cells(disc_analysed)
    alpha_bonf = float(stats.get("prog_alpha_bonferroni", 0.05 / 14))

    fig = plt.figure(figsize=(COL_DOUBLE, 9.4))
    gs = fig.add_gridspec(
        4, 6, height_ratios=[1.15, 1.05, 1.0, 1.0],
        hspace=0.60, wspace=1.25,
    )

    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    nh_curves = session_curves(torque_curves, "NH")
    ik_curves = session_curves(torque_curves, "IK")
    _plot_session_curves(ax_a, nh_curves, "NH")
    _plot_session_curves(ax_b, ik_curves, "IK")
    # Same quantity, side by side: a shared y-axis, or the reader reads the two
    # groups as reaching similar torques.
    top = float(np.nanmax([nh_curves, ik_curves])) * 1.14
    for ax in (ax_a, ax_b):
        ax.set_ylim(0, top)

    ax_c = fig.add_subplot(gs[1, 0:3])
    ax_d = fig.add_subplot(gs[1, 3:6])
    _plot_raster(ax_c, detail, "NH", alpha_bonf)
    _plot_raster(ax_d, detail, "IK", alpha_bonf)

    outcome_axes = [
        fig.add_subplot(gs[2, 0:2]),
        fig.add_subplot(gs[2, 2:4]),
        fig.add_subplot(gs[2, 4:6]),
        fig.add_subplot(gs[3, 0:2]),
        fig.add_subplot(gs[3, 2:4]),
    ]
    for ax, (column, label, unit, nd, prefix) in zip(outcome_axes, OUTCOMES):
        _plot_outcome(ax, cells, column, label, unit, nd, stats, prefix)

    # The spare cell carries the group key and the two notes the panels cannot.
    ax_key = fig.add_subplot(gs[3, 4:6])
    ax_key.axis("off")
    ax_key.legend(
        handles=[
            Line2D([0], [0], color=GROUP_COLOR[g], lw=2.4, marker="o", ms=4,
                   label=GROUP_LABEL[g])
            for g in ("NH", "IK")
        ],
        loc="upper left", frameon=False, fontsize=7.2, bbox_to_anchor=(-0.02, 1.02),
    )
    ax_key.text(
        -0.02, 0.62,
        "E–I: mean ± SD across participants of the\n"
        "three highest-peak-torque repetitions per\n"
        "participant × session. Velocity is the\n"
        "constant-velocity phase. The $p$ values in\n"
        "each panel are the mixed-model omnibus\n"
        "Wald tests reported in the Results.",
        transform=ax_key.transAxes, color=INK_SOFT, fontsize=6.4,
        ha="left", va="top", linespacing=1.5,
    )

    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], ["A", "B", "C", "D"]):
        panel_letter(ax, letter, x=-0.16, y=1.04)
    for ax, letter in zip(outcome_axes, ["E", "F", "G", "H", "I"]):
        panel_letter(ax, letter, x=-0.34, y=1.04)

    if out_dir is None:
        return fig, cells, None
    written = save(fig, out_dir, stem)
    plt.close(fig)
    return None, cells, written


def describe(detail: Dict[str, Any]) -> str:
    lines = []
    for group in ("nh", "ik"):
        for tag in ("bonf", "unc"):
            key = f"prog_{group}_{tag}"
            sig = [
                k for k in range(2, N_SESSIONS + 1)
                if any(
                    c["p"] < detail[key][f"tr_{k}"].get("alpha", 0.05)
                    for c in detail[key].get(f"tr_{k}", {}).get("clusters_t", [])
                )
            ]
            lines.append(f"{key:>14}  {len(sig)} of 14 significant: {sig}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    staged = Path(sys.argv[1])
    results = Path(sys.argv[2])
    target = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("reports")

    curves = pd.read_parquet(staged / "torque_curves_excl.parquet")
    disc = pd.read_parquet(staged / "disc_analysed.parquet")
    with open(results / "manuscript_stats_detail.json") as f:
        detail_json = json.load(f)
    with open(results / "training_stats.json") as f:
        stats_json = json.load(f)

    _, _cells, paths = make_figure(curves, disc, detail_json, stats_json, target)
    print(describe(detail_json))
    for p in paths:
        print(f"wrote {p}")
