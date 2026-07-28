"""Figure 1 -- the training programme: sessions, sets and repetitions.

A Methods figure. It carries one fact: what was prescribed, session by session,
and how the dose progressed. The prescription is the Nordic one; the isokinetic
group's dose was derived from it (Equation 1), which is why the isokinetic
repetition counts are an *outcome* (Figure 4) rather than a prescription.

    session  1        3 sets x 5 reps      15
    sessions 2-5      4 sets x 5 reps      20
    sessions 6-8      5 sets x 6 reps      30
    sessions 9-11     5 sets x 8 reps      40
    sessions 12-15    6 sets x 8 reps      48
                                          ---
                                          497 repetitions

The 497 total is the denominator the Results use for "the NH group completed
92% of the prescription", so it is asserted here and checked at draw time.

The encoding is a stacked bar per session with one segment per set, separated
by a 2 px surface gap. That shows sets and repetitions at once: segment count
is the number of sets, segment width the repetitions in it, total width the
session dose. The R original (``R/Figure1.R``) drew a single bar plus a text
label, which left the set count to be read from prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import to_hex, to_rgb

from .style import (
    COL_ONE_HALF,
    GRID,
    INK,
    INK_SOFT,
    NH,
    save,
    use_paper_style,
)

#: Large solid areas read heavier than thin marks at the same hex, so the bars
#: use a tint of the NH hue rather than the hue itself. Identity is unambiguous
#: here -- Figure 1 shows only one thing -- so the tint costs nothing.
FILL = to_hex(tuple(c + (1.0 - c) * 0.22 for c in to_rgb(NH)))

#: (first session, last session, sets, repetitions per set), inclusive.
PRESCRIPTION: Tuple[Tuple[int, int, int, int], ...] = (
    (1, 1, 3, 5),
    (2, 5, 4, 5),
    (6, 8, 5, 6),
    (9, 11, 5, 8),
    (12, 15, 6, 8),
)

#: Two sessions a week for seven weeks, then a single fifteenth session.
WEEK_OF_SESSION = {s: (s + 1) // 2 for s in range(1, 16)}

TOTAL_PRESCRIBED = 497


def programme() -> List[dict]:
    """Expand the prescription table into one record per training session."""
    out: List[dict] = []
    for first, last, sets, reps in PRESCRIPTION:
        for session in range(first, last + 1):
            out.append(
                {
                    "session": session,
                    "sets": sets,
                    "reps": reps,
                    "total": sets * reps,
                    "week": WEEK_OF_SESSION[session],
                }
            )
    out.sort(key=lambda r: r["session"])
    total = sum(r["total"] for r in out)
    if total != TOTAL_PRESCRIBED:
        raise ValueError(f"prescription sums to {total}, expected {TOTAL_PRESCRIBED}")
    return out


def make_figure(out_dir: Path | None = None, *, stem: str = "Figure_1"):
    use_paper_style()
    rows = programme()

    fig, ax = plt.subplots(figsize=(COL_ONE_HALF, 4.5))
    trans = ax.get_yaxis_transform()  # x in axes fraction, y in data units

    # Alternating week bands, drawn first so every bar sits on top of them.
    for week in sorted({r["week"] for r in rows}):
        sessions = [r["session"] for r in rows if r["week"] == week]
        if week % 2:
            ax.axhspan(min(sessions) - 0.5, max(sessions) + 0.5,
                       color=GRID, alpha=0.40, lw=0, zorder=0)
        ax.text(
            1.02, np.mean(sessions), f"Week {week}",
            transform=trans, clip_on=False,
            color=INK_SOFT, fontsize=7, va="center", ha="left", zorder=3,
        )

    gap = 0.35  # repetitions-worth of white space between two set segments
    for row in rows:
        left = 0.0
        for _ in range(row["sets"]):
            ax.barh(
                row["session"], row["reps"] - gap, left=left,
                height=0.62, color=FILL, edgecolor="none", zorder=2,
            )
            left += row["reps"]
        ax.text(
            left + 1.0, row["session"],
            f"{row['sets']} sets × {row['reps']} reps",
            color=INK, fontsize=7, va="center", ha="left", zorder=3,
        )

    ax.set_yticks(range(1, 16))
    ax.set_yticklabels([str(s) for s in range(1, 16)])
    ax.set_ylim(0.4, 15.6)
    ax.set_xlim(0, 75)
    ax.set_xticks(range(0, 51, 10))
    ax.set_xlabel("Repetitions prescribed per session")
    ax.set_ylabel("Training session")
    ax.grid(False)
    for x in range(0, 51, 10):
        ax.axvline(x, color=GRID, linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 50)
    ax.tick_params(axis="y", length=0)

    ax.annotate(
        f"Each segment is one set.  {TOTAL_PRESCRIBED} repetitions prescribed in total.",
        xy=(0.0, -0.13), xycoords="axes fraction",
        color=INK_SOFT, fontsize=7, ha="left", va="top",
    )

    fig.tight_layout()
    if out_dir is None:
        return fig, None
    written = save(fig, out_dir, stem)
    plt.close(fig)
    return None, written


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports")
    _, paths = make_figure(target)
    for p in paths:
        print(f"wrote {p}")
