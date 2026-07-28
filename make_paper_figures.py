#!/usr/bin/env python3
"""Draw every figure of the training-standardization paper, in one command.

    source venv/bin/activate
    python make_paper_figures.py --out ~/…/PAPER_1_training_standardization/v4

WHY THIS EXISTS
---------------
Until now no single command regenerated the manuscript's figures. Some panels
came out of ``jhamon_training.py``, some out of ``R/``, one out of an orphan
2020 script, and they disagreed about palette, type size and — worse — about
which analysis they were drawing. This script is the one entry point, and every
figure it draws is computed from the same cached signal-processing results the
manuscript statistics come from.

    Figure 1  training programme: sessions, sets, repetitions      Methods
    Figure 2  measurement set-up and representative signals        Methods
    Figure 3  modality comparison: torque and velocity + SPM{t}    Results
    Figure 4  repetitions and mechanical work                      Results
    Figure 5  standardization validity: Bland-Altman               Results
    Figure 6  training progression and discrete outcomes           Results

The numbering is the manuscript's, and it is what ``jhamon_results.qmd`` calls
out in the Results text. Figures 1 and 2 need no analysis; 3 to 6 do.

WHAT IT NEEDS
-------------
``nht_results.pkl`` and ``ikt_results.pkl`` in the results folder, plus
``training_stats.json`` and ``manuscript_stats_detail.json`` — the latter two so
that every cluster extent, bias and p value drawn on a figure is read from the
same file the prose reads it from, rather than recomputed with a second
definition that could silently drift. Run ``reproduce.py`` first if either JSON
is stale.

The raw-data drive ``/Volumes/AJMA`` is NOT needed.

VERIFICATION
------------
``--verify`` re-derives the quantities the figures depend on and diffs them
against ``training_stats.json`` key by key, so a figure can never quietly
disagree with the text. It is the figure-side equivalent of
``reproduce.py --check``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Keys whose value a figure draws, and how to recompute them from the figure's
#: own code path. Checked by ``--verify``; a mismatch means the figure and the
#: manuscript text disagree.
TOLERANCE = 5e-3


def _rule(char: str = "=") -> str:
    return char * 76


def _stage(number: int, total: int, title: str) -> None:
    print(f"\n[{number}/{total}] {title}")
    print("-" * 76)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_everything(results_path: Path) -> Dict[str, Any]:
    """Rebuild every frame the figures need, from the two cached pickles."""
    from extract_manuscript_stats import _install_legacy_pickle_shim, _pseudonymise
    from extract_validity_stats import protocol_subset

    from jhamon_training.data import frames
    from jhamon_training.data.utils import dame_manual_exclusions, filter_training_data
    from jhamon_training.saveload import load_obj
    from jhamon_training.stats.analyses_discrete_vars import analyze_discrete_variables
    from jhamon_training.stats.velocity_phase import attach_isokinetic_velocity

    _install_legacy_pickle_shim()
    print("      loading nht_results.pkl / ikt_results.pkl")
    nordict = _pseudonymise(load_obj(name="nht_results.pkl", path=results_path))
    ikdict = _pseudonymise(load_obj(name="ikt_results.pkl", path=results_path))

    print("      building 101-point curve frames")
    nordf = frames.nht_todf(my_dict=nordict)
    ikdf = frames.ikt_todf(my_dict=ikdict)
    training_df = pd.concat([nordf, ikdf])

    print("      computing discrete outcomes")
    with tempfile.TemporaryDirectory(prefix="jhamon_figures_") as tmp:
        disc = analyze_discrete_variables(
            nordict=nordict, ikdf=ikdf, training_df=training_df,
            results_output_path=Path(tmp),
        )["training_disc"]
    disc = attach_isokinetic_velocity(disc, ikdf)

    analysed = filter_training_data(protocol_subset(disc).copy())
    torque_curves = dame_manual_exclusions(training_df).pipe(
        lambda d: d[d["var"] == "torque"]
    )

    print(
        f"      all repetitions {len(disc):,} · analysed {len(analysed):,} · "
        f"curve rows {len(training_df):,}"
    )
    return {
        "training_df": training_df,
        "disc_all": disc,
        "disc_analysed": analysed,
        "torque_curves": torque_curves,
    }


def load_stats(results_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(results_path / "training_stats.json") as f:
        stats = json.load(f)
    with open(results_path / "manuscript_stats_detail.json") as f:
        detail = json.load(f)
    return stats, detail


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify(data: Dict[str, Any], stats: Dict[str, Any]) -> List[str]:
    """Diff what the figures draw against what the manuscript reports.

    Only quantities a figure actually renders are checked; there is no point
    re-verifying statistics no figure shows, ``reproduce.py --check`` does that.
    """
    from jhamon_training.plot.paper import fig3_modality, fig5_agreement

    problems: List[str] = []

    def check(key: str, value: float, label: str) -> None:
        if key not in stats:
            problems.append(f"{label}: key {key!r} absent from training_stats.json")
            return
        expected = float(stats[key])
        if abs(float(value) - expected) > max(TOLERANCE, abs(expected) * TOLERANCE):
            problems.append(
                f"{label}: figure has {float(value):.6g}, training_stats.json has {expected:.6g}"
            )

    print("      Figure 3 — participant-averaged SPM{t} clusters")
    spm = fig3_modality.compute(data["training_df"])
    for var, tag in (("torque", "torque"), ("knee_v", "velocity")):
        res = spm[var]
        check(f"spm_{tag}_paravg_tstar", res["zstar"], f"F3 {tag} t*")
        check(f"spm_{tag}_paravg_n_nh", res["nh"].shape[0], f"F3 {tag} n NH")
        check(f"spm_{tag}_paravg_n_ik", res["ik"].shape[0], f"F3 {tag} n IK")
        for i, c in enumerate(res["clusters"], start=1):
            check(f"spm_{tag}_paravg_c{i}_start", c["start_pct"], f"F3 {tag} c{i} start")
            check(f"spm_{tag}_paravg_c{i}_end", c["end_pct"], f"F3 {tag} c{i} end")

    print("      Figure 5 — Bland-Altman bias, limits and concordance")
    agree = fig5_agreement.compute(data["disc_all"])
    for var, _agg, _title, _unit, _nd, prefix in fig5_agreement.QUANTITIES:
        c = agree[var]
        for suffix, value in (
            ("bias", c["bias"]), ("loa_low", c["loa_low"]), ("loa_high", c["loa_high"]),
            ("ccc", c["ccc"]), ("n_pairs", c["n_pairs"]), ("grand_mean", c["grand_mean"]),
        ):
            check(f"{prefix}_{suffix}", value, f"F5 {prefix} {suffix}")

    print("      Figure 4 — repetitions and work per repetition")
    from jhamon_training.plot.paper import fig4_reps_work

    work = fig4_reps_work._tidy(data["disc_all"])
    per_set = (
        work.groupby(["par", "sesnum", "set", "tr_group"], observed=True)
        .agg(n=("rep", "nunique"), work=("work", "sum"))
        .reset_index()
    )
    for group, tag in (("NH", "nh"), ("IK", "ik")):
        reps = per_set.loc[per_set["tr_group"] == group].groupby("par")["n"].mean()
        wrep = work.loc[work["tr_group"] == group].groupby("par")["work"].mean()
        check(f"reps_all_perset_{tag}", reps.mean(), f"F4 {tag} reps/set")
        check(f"workrep_{tag}_mean", wrep.mean(), f"F4 {tag} work/rep")

    print("      Figure 1 — prescribed repetition total")
    from jhamon_training.plot.paper import fig1_programme

    total = sum(r["total"] for r in fig1_programme.programme())
    if total != fig1_programme.TOTAL_PRESCRIBED:
        problems.append(f"F1: prescription sums to {total}")
    # ``prescribed_total_nh`` is the group total the Results divide by to get
    # "the NH group completed 92% of the prescription", so the per-participant
    # figure this panel draws has to be that number over the NH participants.
    if {"prescribed_total_nh", "lmm_n_nh"} <= set(stats):
        expected = float(stats["prescribed_total_nh"]) / float(stats["lmm_n_nh"])
        if abs(total - expected) > 1e-6:
            problems.append(
                f"F1 prescribed per participant: figure has {total}, "
                f"training_stats.json implies {expected:.1f} "
                f"({stats['prescribed_total_nh']} / {stats['lmm_n_nh']})"
            )

    return problems


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def draw_all(data: Dict[str, Any], stats: Dict[str, Any], detail: Dict[str, Any],
             out_dir: Path, *, only: Optional[Sequence[int]] = None) -> List[Path]:
    from jhamon_training.plot import figure_2_methods_signals
    from jhamon_training.plot.paper import (
        fig1_programme,
        fig3_modality,
        fig4_reps_work,
        fig5_agreement,
        fig6_progression,
    )

    written: List[Path] = []
    wanted = set(only) if only else {1, 2, 3, 4, 5, 6}

    if 1 in wanted:
        _stage(1, 6, "Figure 1 — training programme")
        _, paths = fig1_programme.make_figure(out_dir)
        written += paths

    if 2 in wanted:
        _stage(2, 6, "Figure 2 — measurement set-up and representative signals")
        figure_2_methods_signals.make_figure(save_dir=out_dir, with_photos=True)
        # That module names its own files; give them the manuscript's numbering.
        for suffix in (".png", ".pdf"):
            src = out_dir / f"Figure_2_methods_signals{suffix}"
            if src.exists():
                dst = out_dir / f"Figure_2{suffix}"
                src.replace(dst)
                written.append(dst)

    if 3 in wanted:
        _stage(3, 6, "Figure 3 — modality comparison (participant-averaged SPM{t})")
        _, res, paths = fig3_modality.make_figure(data["training_df"], out_dir)
        print(fig3_modality.describe(res))
        written += paths

    if 4 in wanted:
        _stage(4, 6, "Figure 4 — repetitions and mechanical work")
        _, _per_set, paths = fig4_reps_work.make_figure(data["disc_all"], out_dir)
        print(fig4_reps_work.describe(data["disc_all"]))
        written += paths

    if 5 in wanted:
        _stage(5, 6, "Figure 5 — standardization validity (Bland-Altman)")
        _, res, paths = fig5_agreement.make_figure(data["disc_all"], out_dir)
        print(fig5_agreement.describe(res))
        written += paths

    if 6 in wanted:
        _stage(6, 6, "Figure 6 — training progression and discrete outcomes")
        _, _cells, paths = fig6_progression.make_figure(
            data["torque_curves"], data["disc_analysed"], detail, stats, out_dir
        )
        print(fig6_progression.describe(detail))
        written += paths

    return written


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default="reports",
                   help="directory to write the figures into (default: ./reports)")
    p.add_argument("--results-path", default=None,
                   help="override the results folder holding the caches and stats JSON")
    p.add_argument("--only", default=None,
                   help="comma-separated figure numbers to draw, e.g. --only 3,5")
    p.add_argument("--verify", action="store_true",
                   help="also diff what the figures draw against training_stats.json")
    p.add_argument("--verify-only", action="store_true",
                   help="run the verification and draw nothing")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    if args.results_path:
        os.environ["JHAMON_RESULTS"] = str(Path(args.results_path).expanduser().resolve())
    from jhamon_training.pathutils import RESULTS_TRAINING_PATH as results_path

    only = None
    if args.only:
        only = [int(v) for v in args.only.replace(" ", "").split(",") if v]

    out_dir = Path(args.out).expanduser()
    print(_rule())
    print("jHamON paper 1 — figures")
    print(_rule())
    print(f"      results  {results_path}")
    print(f"      output   {out_dir}")

    for name in ("nht_results.pkl", "ikt_results.pkl",
                 "training_stats.json", "manuscript_stats_detail.json"):
        if not (results_path / name).exists():
            print(f"\nERROR: {name} not found in {results_path}")
            print("Run `python reproduce.py` first, or pass --results-path.")
            return 2

    stats, detail = load_stats(results_path)

    # Figures 1 and 2 draw nothing derived from the caches, so a request for
    # only those must not pay the one-minute rebuild.
    needs_data = args.verify or args.verify_only or (only is None) or bool(
        set(only) & {3, 4, 5, 6}
    )
    data: Dict[str, Any] = {}
    if needs_data:
        _stage(0, 6, "Rebuilding the analysis frames from cache")
        data = load_everything(results_path)

    problems: List[str] = []
    if args.verify or args.verify_only:
        print("\n" + _rule("-"))
        print("Verifying every drawn quantity against training_stats.json")
        print(_rule("-"))
        problems = verify(data, stats)
        if problems:
            print(f"\n{len(problems)} MISMATCH(ES):")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("\n      all drawn quantities match training_stats.json")
        if args.verify_only:
            return 1 if problems else 0

    out_dir.mkdir(parents=True, exist_ok=True)
    written = draw_all(data, stats, detail, out_dir, only=only)

    print("\n" + _rule())
    print(f"wrote {len(written)} file(s) to {out_dir}")
    for path in written:
        print(f"      {path.name}")
    print(_rule())
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
