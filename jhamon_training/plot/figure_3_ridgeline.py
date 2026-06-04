"""Figure 3 — repetitions and mechanical-work distributions (ridgeline plot).

Python port of the original `geom_density_ridges` figure that lived in the
legacy `jhamon_stats` repo (`jHamon_training.Rmd`, panels A/B/C combined with
`plot_grid` -> `reports/figure2.tiff`). That R code was never carried into this
repository; this module reproduces it from `training_disc`.

Three panels, sessions tr_1 (bottom) -> tr_15 (top) on the y-axis:
  A) Number of IK repetitions per set, one density ridge per session, coloured by
     set, with red ticks marking the prescribed (fixed) NH repetition count.
  B) Mechanical work per repetition (J), NH vs IK.
  C) Mechanical work per set (J, summed within each set), NH vs IK.

The function accepts `training_disc` in either layout:
  * current pipeline (wide): one row per repetition with a numeric ``work`` column;
  * legacy (long): a ``var``/``value`` pair, where ``var == "knee_work"`` holds the
    per-repetition work in Joules.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from jhamon_training.pathutils import RESULTS_TRAINING_PATH

# Prescribed (fixed) NH repetitions per set, by training session.
# Protocol: tr1 3x5, tr2-5 4x5, tr6-8 5x6, tr9-11 5x8, tr12-15 6x8.
PRESCRIBED_REPS = {s: (5 if s <= 5 else 6 if s <= 8 else 8) for s in range(1, 16)}

# NH/IK palette — matches the rest of the Python pipeline (Figs 4/6/7).
GROUP_COLORS = {"NH": "#ff7f0e", "IK": "#1f77b4"}
GROUP_ORDER = ["NH", "IK"]

_ACADEMIC_RC = {
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _session_num(s) -> int:
    """tr_7 -> 7 (accepts ints or 'tr_7' strings)."""
    s = str(s)
    return int(s.split("_")[-1]) if "_" in s else int(s)


def _extract_work(training_disc: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy per-repetition frame: par, sesnum, set, rep, tr_group, work."""
    df = training_disc.copy()
    if "work" in df.columns:  # current wide layout (one row per rep)
        work = df[["par", "trses", "set", "rep", "tr_group", "work"]].copy()
    elif "var" in df.columns and "value" in df.columns:  # legacy long layout
        work = df[df["var"] == "knee_work"][
            ["par", "trses", "set", "rep", "tr_group", "value"]
        ].rename(columns={"value": "work"})
    else:
        raise ValueError(
            "training_disc must contain either a 'work' column or 'var'/'value' "
            f"columns; got {df.columns.tolist()}"
        )
    work = work.dropna(subset=["work"])
    work["sesnum"] = work["trses"].map(_session_num)
    return work


def _ridgeline(ax, panel, value_col, color_col, color_map, x_grid, *, scale=1.4,
               rel_min_height=0.05, draw_order=None):
    """Draw stacked KDE ridges (one row per session) on `ax`.

    `panel` columns: sesnum, `value_col`, `color_col`. Densities are scaled by a
    single global factor so the tallest ridge in the panel spans `scale` rows
    (this is what ggridges' `scale=` does), preserving relative heights.
    """
    sessions = sorted(panel["sesnum"].unique())
    groups = draw_order or sorted(panel[color_col].unique())

    # First pass: KDEs + global max density for consistent scaling.
    densities, gmax = {}, 0.0
    for ses in sessions:
        for grp in groups:
            v = panel[(panel["sesnum"] == ses) & (panel[color_col] == grp)][value_col]
            v = v.to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size < 2 or np.ptp(v) == 0:
                continue
            try:
                d = gaussian_kde(v)(x_grid)
            except np.linalg.LinAlgError:
                continue
            densities[(ses, grp)] = d
            gmax = max(gmax, d.max())
    if gmax == 0:
        return sessions
    norm = scale / gmax

    # Second pass: draw bottom (tr_1) to top.
    for i, ses in enumerate(sessions):
        for grp in groups:
            d = densities.get((ses, grp))
            if d is None:
                continue
            y = d * norm
            y = np.where(y < rel_min_height * scale, np.nan, y)  # trim tails
            c = color_map[grp]
            ax.fill_between(x_grid, i, i + y, color=c, alpha=0.7, lw=0, zorder=i)
            ax.plot(x_grid, i + y, color=c, lw=0.6, alpha=0.9, zorder=i)
    return sessions


def plot_figure_3_ridgeline(
    training_disc: pd.DataFrame,
    output_path: Optional[Path] = None,
    dpi: int = 300,
):
    """Build the three-panel reps/work ridgeline figure (manuscript Figure 3)."""
    work = _extract_work(training_disc)

    # ---- Panel data ---------------------------------------------------------
    # A: IK repetitions per (par, session, set)
    ik = work[work["tr_group"] == "IK"]
    repcount = (
        ik.groupby(["par", "sesnum", "set"])["rep"].nunique().reset_index(name="n")
    )
    sets_present = sorted(repcount["set"].unique())
    set_colors = {
        s: c for s, c in zip(sets_present, plt.cm.viridis(np.linspace(0, 1, len(sets_present))))
    }

    # B: work per repetition ; C: work per set
    wset = (
        work.groupby(["par", "sesnum", "set", "tr_group"])["work"].sum().reset_index()
    )

    with plt.rc_context(_ACADEMIC_RC):
        fig, (axA, axB, axC) = plt.subplots(
            1, 3, figsize=(11, 6), gridspec_kw={"width_ratios": [1.25, 1, 1]}
        )

        # Panel A — repetitions, coloured by set
        xA = np.linspace(2, 19, 300)
        sessions = _ridgeline(
            axA, repcount.rename(columns={"set": "grp"}), "n", "grp", set_colors, xA,
            draw_order=sets_present,
        )
        # red ticks at the prescribed NH rep count for each session
        for i, ses in enumerate(sessions):
            xr = PRESCRIBED_REPS.get(int(ses))
            if xr is not None:
                axA.plot([xr, xr], [i, i + 0.35], color="red", lw=1.6, zorder=100)
        axA.set_xlabel("Number of repetitions")
        axA.set_ylabel("Training session")
        axA.set_xlim(2, 19)
        set_handles = [
            Line2D([0], [0], color=set_colors[s], lw=6, alpha=0.7,
                   label=s.replace("_", " "))
            for s in sets_present
        ]
        axA.legend(handles=set_handles, loc="upper right", frameon=False,
                   handlelength=1.0, labelspacing=0.25)

        # Panel B — work per repetition, NH vs IK
        xB = np.linspace(0, float(work["work"].quantile(0.999)), 300)
        _ridgeline(axB, work, "work", "tr_group", GROUP_COLORS, xB,
                   draw_order=GROUP_ORDER)
        axB.set_xlabel("Work (J)")
        axB.set_xlim(0, xB[-1])

        # Panel C — work per set, NH vs IK
        xC = np.linspace(0, float(wset["work"].quantile(0.999)), 300)
        _ridgeline(axC, wset, "work", "tr_group", GROUP_COLORS, xC,
                   draw_order=GROUP_ORDER)
        axC.set_xlabel("Work (J)")
        axC.set_xlim(0, xC[-1])
        group_handles = [
            Line2D([0], [0], color=GROUP_COLORS[g], lw=6, alpha=0.7, label=g)
            for g in GROUP_ORDER
        ]
        axC.legend(handles=group_handles, loc="upper right", frameon=False,
                   handlelength=1.0)

        # Shared y ticks = session numbers; hide redundant labels on B/C
        yticks = list(range(len(sessions)))
        for ax in (axA, axB, axC):
            ax.set_yticks(yticks)
            ax.set_ylim(-0.3, len(sessions) + scale_pad())
        axA.set_yticklabels([str(int(s)) for s in sessions])
        for ax in (axB, axC):
            ax.set_yticklabels([])

        # Panel letters
        for ax, letter in zip((axA, axB, axC), "ABC"):
            ax.annotate(letter, xy=(0.02, 0.98), xycoords="axes fraction",
                        fontsize=12, fontweight="bold", va="top", ha="left")

        fig.tight_layout(w_pad=1.0)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            if output_path.suffix.lower() != ".pdf":
                fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
            print(f"Figure 3 saved to {output_path}")
        else:
            plt.show()
    return None


def scale_pad():
    """Top headroom so the tallest ridge isn't clipped (matches `scale` in _ridgeline)."""
    return 1.4


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate manuscript Figure 3 (ridgeline).")
    p.add_argument(
        "--training-disc",
        default=str(RESULTS_TRAINING_PATH / "oldrafts" / "training_disc"),
        help="Path to a training_disc feather (wide 'work' or legacy long layout).",
    )
    p.add_argument("--out", default="reports/Figure_3_ridgeline.png")
    args = p.parse_args()

    disc = pd.read_feather(args.training_disc)
    plot_figure_3_ridgeline(disc, output_path=Path(args.out))
