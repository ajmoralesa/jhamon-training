"""Figure 4 -- what it cost each modality to deliver the same dose.

    A   repetitions performed per set, session by session, NH and IK
    B   mechanical work per repetition
    C   mechanical work per set

Read left to right the figure is the argument itself: the isokinetic group needs
more repetitions (A) because each of its repetitions delivers less work (B), and
the two therefore arrive at the same work per set (C).

ACHIEVED AGAINST ACHIEVED
-------------------------
Panel A plots the repetitions each group actually performed. The earlier version
of this figure showed the isokinetic counts against the *prescribed* Nordic
count, which is not a like-for-like comparison: the NH group completed 92.1% of
its prescription, so setting achieved against prescribed understates the gap by
0.15-0.3 repetitions per set. The Results now report achieved versus achieved
(+3.24 repetitions per set), and the figure has to show the same thing. The
prescription is still marked, as a thin reference tick, because it explains why
both groups' counts step upward across the programme -- but it is not what the
isokinetic distribution is being compared against.

POPULATION
----------
All repetitions performed (15,883), which is the population the repetition and
work descriptives in the Results are computed on. The cumulative-work criterion
that ended each isokinetic set applied to every repetition performed, so this is
also the population the standardization targeted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from .style import (
    COL_DOUBLE,
    GRID,
    GROUP_COLOR,
    GROUP_LABEL,
    INK,
    INK_SOFT,
    REFERENCE,
    panel_letter,
    save,
    use_paper_style,
)

N_SESSIONS = 15
GROUP_ORDER = ("NH", "IK")

#: Prescribed repetitions per set by session (see fig1_programme.PRESCRIPTION).
PRESCRIBED_REPS = {s: (5 if s <= 5 else 6 if s <= 8 else 8) for s in range(1, 16)}

#: How many rows tall the tallest ridge in a panel is allowed to be. This is
#: ggridges' ``scale=``: one global factor per panel, so relative heights within
#: the panel stay comparable.
RIDGE_SCALE = 1.5
RIDGE_MIN_HEIGHT = 0.04

#: Panel A's ridges are normalised per ridge rather than per panel, so they all
#: reach full height; they need to be shorter than the KDE ridges of B and C or
#: every session would overlap its neighbour.
DISCRETE_RIDGE_SCALE = 0.85


def _session_num(value) -> int:
    text = str(value)
    return int(text.split("_")[-1]) if "_" in text else int(text)


def _tidy(disc: pd.DataFrame) -> pd.DataFrame:
    """Per-repetition frame with a numeric session column."""
    out = disc[["par", "trses", "set", "rep", "tr_group", "work"]].dropna(subset=["work"]).copy()
    out["sesnum"] = out["trses"].map(_session_num)
    return out


def _ridgeline(ax, panel: pd.DataFrame, value_col: str, x_grid: np.ndarray) -> None:
    """Stacked per-session KDE ridges, one filled band per group."""
    sessions = list(range(1, N_SESSIONS + 1))

    densities: Dict[tuple, np.ndarray] = {}
    gmax = 0.0
    for ses in sessions:
        for group in GROUP_ORDER:
            v = panel.loc[
                (panel["sesnum"] == ses) & (panel["tr_group"] == group), value_col
            ].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size < 2 or np.ptp(v) == 0:
                continue
            try:
                d = gaussian_kde(v)(x_grid)
            except np.linalg.LinAlgError:
                continue
            densities[(ses, group)] = d
            gmax = max(gmax, float(d.max()))
    if gmax == 0:
        return
    norm = RIDGE_SCALE / gmax

    # Bottom (session 1) to top, so later sessions overlap earlier ones.
    for i, ses in enumerate(sessions):
        for group in GROUP_ORDER:
            d = densities.get((ses, group))
            if d is None:
                continue
            y = d * norm
            y = np.where(y < RIDGE_MIN_HEIGHT * RIDGE_SCALE, np.nan, y)
            colour = GROUP_COLOR[group]
            ax.fill_between(x_grid, i, i + y, color=colour, alpha=0.55, lw=0, zorder=2 + i)
            ax.plot(x_grid, i + y, color=colour, lw=0.7, alpha=0.95, zorder=2 + i)


def _discrete_ridgeline(ax, per_set: pd.DataFrame) -> None:
    """Panel A: the distribution of an integer count, per session and group.

    Repetitions per set are counts, and the Nordic group is very nearly a point
    mass on the prescription (SD 0.17 sets across participants). A kernel
    density of that is a spike that, under the single global scaling a ridgeline
    normally uses, flattens the isokinetic distribution to an invisible line.
    So panel A draws the empirical distribution over the integers themselves,
    each ridge scaled to its own maximum: the question here is where each
    group's counts sit and how wide they spread, not how one density compares
    with another in absolute height.
    """
    for i, ses in enumerate(range(1, N_SESSIONS + 1)):
        for group in GROUP_ORDER:
            v = per_set.loc[
                (per_set["sesnum"] == ses) & (per_set["tr_group"] == group), "n"
            ].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            counts = np.bincount(v.astype(int), minlength=21)[:21]
            share = counts / counts.sum()
            if share.max() == 0:
                continue
            y = share / share.max() * DISCRETE_RIDGE_SCALE
            edges = np.arange(-0.5, 21.5)
            colour = GROUP_COLOR[group]
            ax.stairs(
                i + y, edges, baseline=i, fill=True, color=colour,
                alpha=0.55, lw=0, zorder=2 + i,
            )
            ax.stairs(i + y, edges, baseline=i, color=colour, lw=0.7,
                      alpha=0.95, zorder=2 + i)


def _finish(ax, *, xlabel: str, show_sessions: bool) -> None:
    ax.set_yticks(range(N_SESSIONS))
    if show_sessions:
        ax.set_yticklabels([str(s) for s in range(1, N_SESSIONS + 1)])
        ax.set_ylabel("Training session")
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-0.35, N_SESSIONS - 1 + RIDGE_SCALE + 0.25)
    ax.set_xlabel(xlabel)
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def make_figure(disc_all: pd.DataFrame, out_dir: Path | None = None,
                *, stem: str = "Figure_4"):
    use_paper_style()
    work = _tidy(disc_all)

    # A: repetitions actually performed in each (participant, session, set).
    per_set = (
        work.groupby(["par", "sesnum", "set", "tr_group"], observed=True)
        .agg(n=("rep", "nunique"), work=("work", "sum"))
        .reset_index()
    )

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(COL_DOUBLE, 5.4),
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.0], "wspace": 0.14},
    )

    _discrete_ridgeline(ax_a, per_set)
    # The prescription sits *below* the baseline as a caret, not as a line
    # through the bars: inside a filled ridge it is invisible, and it is a
    # reference rather than a comparator.
    for i, ses in enumerate(range(1, N_SESSIONS + 1)):
        ax_a.plot(
            [PRESCRIBED_REPS[ses]], [i - 0.13], marker="^", ms=3.2,
            color=REFERENCE, zorder=100, clip_on=False,
        )
    ax_a.set_xlim(2.5, 18.5)
    ax_a.set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
    _finish(ax_a, xlabel="Repetitions per set", show_sessions=True)

    x_b = np.linspace(0, float(work["work"].quantile(0.999)), 400)
    _ridgeline(ax_b, work, "work", x_b)
    ax_b.set_xlim(0, x_b[-1])
    _finish(ax_b, xlabel="Work per repetition (J)", show_sessions=False)

    x_c = np.linspace(0, float(per_set["work"].quantile(0.999)), 400)
    _ridgeline(ax_c, per_set, "work", x_c)
    ax_c.set_xlim(0, x_c[-1])
    _finish(ax_c, xlabel="Work per set (J)", show_sessions=False)

    # A figure-level key: the legend applies to all three panels, and every
    # corner of panel A is occupied by one session's ridge or another.
    fig.legend(
        handles=[
            Patch(facecolor=GROUP_COLOR[g], alpha=0.55, label=GROUP_LABEL[g])
            for g in GROUP_ORDER
        ]
        + [Line2D([0], [0], color=REFERENCE, lw=0, marker="^", ms=3.6,
                  label="NH prescription")],
        loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3,
        frameon=False, fontsize=7.2, handlelength=1.1, handleheight=0.9,
        columnspacing=1.8,
    )

    for ax, letter in zip((ax_a, ax_b, ax_c), "ABC"):
        panel_letter(ax, letter, x=-0.10 if ax is ax_a else -0.04, y=1.01)

    fig.text(
        0.5, 0.015,
        "One ridge per training session, session 1 at the bottom. A: distribution over integer counts, each ridge\n"
        "scaled to its own maximum. B, C: kernel density of every repetition performed. Repetition counts are\n"
        "compared achieved against achieved; the prescription is marked for reference only.",
        ha="center", va="top", color=INK_SOFT, fontsize=6.8, linespacing=1.4,
    )

    if out_dir is None:
        return fig, per_set, None
    written = save(fig, out_dir, stem)
    plt.close(fig)
    return None, per_set, written


def describe(disc_all: pd.DataFrame) -> str:
    """The three numbers the Results quote, recomputed from what the figure draws."""
    work = _tidy(disc_all)
    per_set = (
        work.groupby(["par", "sesnum", "set", "tr_group"], observed=True)
        .agg(n=("rep", "nunique"), work=("work", "sum"))
        .reset_index()
    )
    lines = []
    for group in GROUP_ORDER:
        reps = per_set.loc[per_set["tr_group"] == group].groupby("par")["n"].mean()
        wrep = work.loc[work["tr_group"] == group].groupby("par")["work"].mean()
        wset = per_set.loc[per_set["tr_group"] == group].groupby("par")["work"].mean()
        lines.append(
            f"{group}  reps/set {reps.mean():.2f} ± {reps.std(ddof=1):.2f}   "
            f"work/rep {wrep.mean():.1f} ± {wrep.std(ddof=1):.1f} J   "
            f"work/set {wset.mean():.1f} ± {wset.std(ddof=1):.1f} J"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    staged = Path(sys.argv[1])
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports")
    disc = pd.read_parquet(staged / "disc_all.parquet")
    _, _per_set, paths = make_figure(disc, target)
    print(describe(disc))
    for p in paths:
        print(f"wrote {p}")
