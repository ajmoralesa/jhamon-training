"""Figure 4 -- what it cost each modality to deliver the same dose.

    A   repetitions per set in the isokinetic group, set by set
    B   mechanical work per repetition, by group
    C   mechanical work per set, by group

Read left to right the figure is the argument itself: the isokinetic group needs
more repetitions than the fixed Nordic prescription (A) because each of its
repetitions delivers less work (B), and the two therefore arrive at the same
work per set (C).

WHY PANEL A DRAWS ONLY ONE GROUP
--------------------------------
The Nordic repetition count was *prescribed*, not measured: five repetitions per
set to session 5, six to session 8, eight thereafter. Of the sets they actually
performed, participants hit the prescribed count exactly in 83.8% and were
within one repetition of it in 95.5%, delivering 98.0% of the repetitions those
sets called for. So the Nordic "distribution" is a point mass sitting on the
prescription. Drawing it as a density is drawing the protocol, not a result --

(The Results separately report that the group completed 92.1% of the
prescription. That is the same numerator over a different denominator -- 5,951
repetitions against the 6,461 the full programme prescribed, so it also charges
the sets and sessions that were missed outright. Both numbers are right; this
panel is about the sets that happened, which is why it quotes the 98.0%.)

and under a shared vertical scale that spike flattens the isokinetic densities
to an invisible line, which is what forced the previous version of this panel
into hard-edged bars.

So panel A shows the quantity that actually varied: how many repetitions the
isokinetic group needed, set by set, to reach the same cumulative work. The
Nordic prescription is the dashed reference it is measured against. This
restores the design of the v1/v3 figure, which the bar version had lost.

WHY COLOUR IS SET NUMBER
------------------------
Within a session, the isokinetic set-ending criterion was cumulative work, so
repetitions per set is a fatigue read-out: if torque falls as the session wears
on, it takes more repetitions to accumulate the same work. Splitting each ridge
by set is what makes that visible, and what it shows is that the effect is not
constant across the programme. Averaged over the whole programme the rise from
the first set to the last is only +0.32 repetitions, and in sessions 2 to 6 it
is flat or slightly negative. It appears in the second half and grows with
volume: +0.97 repetitions across sessions 12 to 15, reaching +1.5 in session 15
(11.2 repetitions in set 1, 12.8 in set 6).

Two cautions the panel's geometry makes easy to get wrong. Sets do NOT compare
across sessions -- the programme ran three sets in session 1 and six from
session 12 -- so set colour is only interpretable *within* a ridge; pooling sets
over the programme confounds set with session and inflates that first-to-last
rise from 0.32 to 2.3 repetitions, because sets 5 and 6 exist only in the later
sessions, which prescribe more repetitions anyway. And because later ridges
simply contain more sets, the colour of a ridge drifts up the panel for a
reason that is programme design, not a result.

Set is an ordered magnitude, so it takes a perceptually uniform sequential ramp
(viridis, truncated short of its palest yellow so the last set still reads on
white). Its luminance is monotonic, so the set order survives greyscale
printing, which is the test ``style.py`` applies to the group palette.

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
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from .style import (
    COL_DOUBLE,
    GRID,
    GROUP_COLOR,
    GROUP_LABEL,
    INK_SOFT,
    NH as NH_COLOR,
    panel_letter,
    save,
    use_paper_style,
)

N_SESSIONS = 15
GROUP_ORDER = ("NH", "IK")

#: Prescribed repetitions per set by session (see fig1_programme.PRESCRIPTION).
PRESCRIBED_REPS = {s: (5 if s <= 5 else 6 if s <= 8 else 8) for s in range(1, 16)}

#: Sets the programme actually reached often enough to draw. A seventh set
#: exists for a single participant-session and is excluded by MIN_CELL below.
MAX_SETS = 6

#: A density needs a sample. Cells thinner than this are dropped rather than
#: smoothed into a shape that looks like evidence -- session 1 sets 4 and 5 have
#: one participant each, because the programme had not yet settled.
MIN_CELL = 3

#: How many rows tall the tallest ridge in a panel is allowed to be. This is
#: ggridges' ``scale=``: one global factor per panel, so relative heights within
#: the panel stay comparable.
RIDGE_SCALE = 1.5
RIDGE_SCALE_A = 1.6
RIDGE_MIN_HEIGHT = 0.04

#: Kernel width floor for panel A, in repetitions. Repetitions are counts, so a
#: density narrower than half a repetition is claiming resolution the
#: measurement does not have; it also stops a tight set from spiking and
#: crushing every other ridge under the shared vertical scale.
BW_FLOOR = 0.55


def _session_num(value) -> int:
    text = str(value)
    return int(text.split("_")[-1]) if "_" in text else int(text)


def _set_num(value) -> int:
    text = str(value)
    return int(text.split("_")[-1]) if "_" in text else int(text)


def _tidy(disc: pd.DataFrame) -> pd.DataFrame:
    """Per-repetition frame with a numeric session column."""
    out = disc[["par", "trses", "set", "rep", "tr_group", "work"]].dropna(subset=["work"]).copy()
    out["sesnum"] = out["trses"].map(_session_num)
    return out


def set_palette(n: int = MAX_SETS) -> list:
    """Sequential colours for set 1..n, dark early to light late.

    Truncated short of viridis' palest yellow, which has too little contrast
    against a white page to carry a filled shape.
    """
    cmap = plt.get_cmap("viridis")
    return [cmap(x) for x in np.linspace(0.04, 0.88, n)]


def _shade(rgba, factor: float = 0.72):
    """A darker version of a fill colour, for that fill's outline."""
    r, g, b = rgba[:3]
    return (r * factor, g * factor, b * factor)


def _count_density(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gaussian kernel density over integer counts, with a bandwidth floor.

    Written out rather than delegated to ``gaussian_kde`` for two reasons: the
    floor has to apply to the absolute kernel width (scipy's ``bw_method`` is a
    multiple of the sample SD, which is the wrong handle when the SD is what is
    small), and cells where every participant did the same number of
    repetitions have zero variance, which makes ``gaussian_kde`` raise rather
    than return the narrow bump those cells should draw.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return np.zeros_like(x)
    sigma = float(v.std(ddof=1)) if n > 1 else 0.0
    h = 1.06 * sigma * n ** (-0.2) if sigma > 0 else 0.0
    h = max(h, BW_FLOOR)
    z = (x[:, None] - v[None, :]) / h
    return np.exp(-0.5 * z**2).sum(axis=1) / (n * h * np.sqrt(2.0 * np.pi))


def _set_ridgeline(ax, per_set: pd.DataFrame, x_grid: np.ndarray) -> int:
    """Panel A: per-session ridges, one density per set, isokinetic only.

    Returns the number of (session, set) cells dropped for being too thin, so
    the caller can report it rather than let it pass silently.
    """
    colours = set_palette()

    densities: Dict[tuple, np.ndarray] = {}
    dropped = 0
    gmax = 0.0
    for ses in range(1, N_SESSIONS + 1):
        for s in range(1, MAX_SETS + 1):
            v = per_set.loc[
                (per_set["sesnum"] == ses) & (per_set["setnum"] == s), "n"
            ].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            if v.size < MIN_CELL:
                dropped += 1
                continue
            d = _count_density(v, x_grid)
            densities[(ses, s)] = d
            gmax = max(gmax, float(d.max()))
    if gmax == 0:
        return dropped
    norm = RIDGE_SCALE_A / gmax

    # Bottom (session 1) to top, so later sessions overlap earlier ones; within
    # a session, later sets sit over earlier ones.
    for i, ses in enumerate(range(1, N_SESSIONS + 1)):
        for s in range(1, MAX_SETS + 1):
            d = densities.get((ses, s))
            if d is None:
                continue
            y = d * norm
            y = np.where(y < RIDGE_MIN_HEIGHT * RIDGE_SCALE_A, np.nan, y)
            colour = colours[s - 1]
            z = 2 + i * 10 + s
            # Light wash, crisp outline. Six opaque fills per ridge turn into a
            # blob whose apparent colour is just whichever set was drawn last,
            # which reads as a session trend that is not in the data; the
            # outline is what makes an individual set traceable through the
            # overlap.
            ax.fill_between(x_grid, i, i + y, color=colour, alpha=0.22, lw=0, zorder=z)
            ax.plot(x_grid, i + y, color=_shade(colour, 0.88), lw=1.1,
                    alpha=0.98, zorder=z, solid_joinstyle="round")
    return dropped


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
            ax.fill_between(x_grid, i, i + y, color=colour, alpha=0.5, lw=0, zorder=2 + i)
            ax.plot(x_grid, i + y, color=_shade(to_rgb(colour), 0.85), lw=0.95,
                    alpha=0.98, zorder=2 + i)


def _finish(ax, *, xlabel: str, show_sessions: bool, scale: float = RIDGE_SCALE) -> None:
    ax.set_yticks(range(N_SESSIONS))
    if show_sessions:
        ax.set_yticklabels([str(s) for s in range(1, N_SESSIONS + 1)])
        ax.set_ylabel("Training session")
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-0.35, N_SESSIONS - 1 + scale + 0.25)
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

    # Repetitions actually performed in each (participant, session, set).
    per_set = (
        work.groupby(["par", "sesnum", "set", "tr_group"], observed=True)
        .agg(n=("rep", "nunique"), work=("work", "sum"))
        .reset_index()
    )
    per_set["setnum"] = per_set["set"].map(_set_num)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(COL_DOUBLE, 6.0),
        gridspec_kw={"width_ratios": [1.22, 1.0, 1.0], "wspace": 0.14},
    )

    # ----- A: isokinetic repetitions per set ------------------------------- #
    x_a = np.linspace(2.0, 20.0, 500)
    dropped = _set_ridgeline(ax_a, per_set[per_set["tr_group"] == "IK"], x_a)

    # The Nordic prescription: a dashed rule through each ridge, in the group's
    # own hue, because it is what the isokinetic counts are being read against.
    for i, ses in enumerate(range(1, N_SESSIONS + 1)):
        ax_a.plot(
            [PRESCRIBED_REPS[ses]] * 2, [i - 0.06, i + 1.02],
            color=NH_COLOR, lw=1.1, ls=(0, (2.6, 1.6)),
            solid_capstyle="butt", zorder=500 + i * 10,
        )

    ax_a.set_xlim(2.5, 19.5)
    ax_a.set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
    _finish(ax_a, xlabel="Repetitions per set", show_sessions=True, scale=RIDGE_SCALE_A)

    # Set key, in the corner panel A leaves empty: early sessions never reach
    # the high repetition counts the axis has to span for the late ones.
    set_colours = set_palette()
    ax_a.legend(
        handles=[
            Patch(facecolor=(*set_colours[s - 1][:3], 0.22),
                  edgecolor=_shade(set_colours[s - 1], 0.88), lw=1.1, label=f"Set {s}")
            for s in range(1, MAX_SETS + 1)
        ],
        loc="lower right", bbox_to_anchor=(1.02, -0.015),
        frameon=False, fontsize=6.2, handlelength=1.1, handleheight=0.75,
        labelspacing=0.22, borderpad=0.15,
    )

    # ----- B, C: work, by group -------------------------------------------- #
    x_b = np.linspace(0, float(work["work"].quantile(0.999)), 400)
    _ridgeline(ax_b, work, "work", x_b)
    ax_b.set_xlim(0, x_b[-1])
    _finish(ax_b, xlabel="Work per repetition (J)", show_sessions=False)

    x_c = np.linspace(0, float(per_set["work"].quantile(0.999)), 400)
    _ridgeline(ax_c, per_set, "work", x_c)
    ax_c.set_xlim(0, x_c[-1])
    _finish(ax_c, xlabel="Work per set (J)", show_sessions=False)

    # A figure-level key for the group colours of B and C, plus the reference
    # rule of A. The set key lives inside panel A, next to what it names.
    fig.legend(
        handles=[
            Patch(facecolor=GROUP_COLOR[g], alpha=0.55, label=GROUP_LABEL[g])
            for g in GROUP_ORDER
        ]
        + [Line2D([0], [0], color=NH_COLOR, lw=1.1, ls=(0, (2.6, 1.6)),
                  label="NH prescription (fixed)")],
        loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3,
        frameon=False, fontsize=7.2, handlelength=1.4, handleheight=0.9,
        columnspacing=1.8,
    )

    for ax, letter in zip((ax_a, ax_b, ax_c), "ABC"):
        panel_letter(ax, letter, x=-0.10 if ax is ax_a else -0.04, y=1.01)

    fig.text(
        0.5, 0.015,
        "One ridge per training session, session 1 at the bottom. A: repetitions per set in the isokinetic\n"
        "group, one kernel-smoothed density per set. The programme added sets as it progressed (three in\n"
        "session 1, six from session 12), so later ridges carry more colours; sets are comparable within a\n"
        "session, not across them. The Nordic count was prescribed and met in 83.8% of sets, so it is drawn\n"
        "as a reference rule rather than a distribution. B, C: kernel density of every repetition performed.",
        ha="center", va="top", color=INK_SOFT, fontsize=6.8, linespacing=1.4,
    )

    if out_dir is None:
        return fig, per_set, None
    written = save(fig, out_dir, stem)
    plt.close(fig)
    if dropped:
        print(f"      panel A: {dropped} session-set cells below n={MIN_CELL}, not drawn")
    return None, per_set, written


def describe(disc_all: pd.DataFrame) -> str:
    """The numbers the Results quote, recomputed from what the figure draws."""
    work = _tidy(disc_all)
    per_set = (
        work.groupby(["par", "sesnum", "set", "tr_group"], observed=True)
        .agg(n=("rep", "nunique"), work=("work", "sum"))
        .reset_index()
    )
    per_set["setnum"] = per_set["set"].map(_set_num)
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

    # Panel A's own claim: the Nordic count is the prescription, and the
    # isokinetic count climbs across the sets of a session.
    nh = per_set.loc[per_set["tr_group"] == "NH"].copy()
    nh["pres"] = nh["sesnum"].map(PRESCRIBED_REPS)
    lines.append(
        f"NH  sets exactly at prescription {100 * (nh['n'] == nh['pres']).mean():.1f}%   "
        f"prescribed repetitions performed {100 * nh['n'].sum() / nh['pres'].sum():.1f}%"
    )
    # The set effect has to be read WITHIN a session. Pooling sets across the
    # programme confounds set with session, because sets 5 and 6 exist only in
    # the later sessions, which prescribe more repetitions anyway: pooled, that
    # inflates the apparent first-to-last rise from 1.1 to 2.3 repetitions.
    ik = per_set.loc[(per_set["tr_group"] == "IK") & (per_set["setnum"] <= MAX_SETS)]
    cells = ik.groupby(["sesnum", "setnum"])["n"].agg(["mean", "size"])
    cells = cells[cells["size"] >= MIN_CELL]["mean"]
    deltas = []
    for ses, block in cells.groupby(level="sesnum"):
        sets = block.index.get_level_values("setnum")
        if len(sets) >= 2:
            deltas.append(block.loc[(ses, sets.max())] - block.loc[(ses, sets.min())])
    lines.append(
        f"IK  first-to-last set, within session  +{np.mean(deltas):.2f} repetitions "
        f"(mean over {len(deltas)} sessions)"
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
