#!/usr/bin/env python3
"""Run the manuscript's SPM analyses on three abscissae and report the difference.

The project's SPM results are computed on a linearly time-normalised axis. Time
is not a shared reference between the two modalities: a Nordic descent
accelerates throughout while an isokinetic repetition does not, so the same node
denotes a different knee angle in each group. This script re-runs both SPM
families on the axis that IS shared -- knee angle -- and on an intermediate
axis, and writes everything needed to judge whether the conclusions hold.

Axes
    time         0-100 % of duration, 101 nodes. The shipped analysis.
    rom_percent  0-100 % of each repetition's own excursion, 101 nodes. Same
                 node count, nothing truncated, no repetition dropped -- so a
                 difference against `time` isolates the abscissa from every
                 other change.
    knee_angle   absolute degrees on a common grid. Anatomically interpretable;
                 costs the repetitions that do not span the grid.

Every axis is run on the SAME repetitions (the intersection of those registrable
on all three), so a moved cluster cannot be a changed sample. Both exclusion
regimes are reported -- none at all, and the corrected repetition-level ones --
because a conclusion that depends on the exclusion rule is not a conclusion.

Outputs, into the results folder unless --output-dir is given:
    axis_comparison.json          every statistic, machine-readable
    figures/axis_comparison.png   group means and clusters, time versus angle
    figures/axis_mapping.png      the time-to-angle mapping that causes it all

Usage:
    source venv/bin/activate && python compare_axes.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

CURVE_VARS = (("torque", "torque", "Knee flexor torque (N·m)"),
              ("knee_v", "velocity", "Knee angular velocity (deg·s⁻¹)"))
GRID_VARIANTS = ((80.0, 30.0), (85.0, 25.0), (88.0, 20.0), (88.0, 15.0))
PRIMARY_GRID = (80.0, 30.0)
NH_COLOUR, IK_COLOUR = "#C2410C", "#1D4ED8"
POS_COLOUR, NEG_COLOUR = "#15803D", "#7E22CE"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_curves(results_path: Path) -> pd.DataFrame:
    """Rebuild the long curve frame from the pickled caches."""
    from reproduce import _install_legacy_pickle_shim, _pseudonymise

    from jhamon_training.data import frames
    from jhamon_training.saveload import load_obj

    _install_legacy_pickle_shim()
    nordict = _pseudonymise(load_obj(name="nht_results.pkl", path=results_path))
    ikdict = _pseudonymise(load_obj(name="ikt_results.pkl", path=results_path))
    nordf = frames.nht_todf(my_dict=nordict)
    ikdf = frames.ikt_todf(my_dict=ikdict)
    print(f"      curve rows: NH {len(nordf):,}  IK {len(ikdf):,}")
    return pd.concat([nordf, ikdf])


# --------------------------------------------------------------------------- #
# 1. the mapping that causes the problem
# --------------------------------------------------------------------------- #
def axis_mapping(training_df: pd.DataFrame) -> Dict[str, Any]:
    """How far each group's angle-versus-time trajectory departs from linear.

    This is the whole argument in one table. If both modalities advanced through
    their range of motion at a constant rate, the time axis would be an affine
    rescaling of the angle axis and the choice would not matter. They do not.
    """
    out: Dict[str, Any] = {}
    means: Dict[str, np.ndarray] = {}
    for group in ("NH", "IK"):
        sub = training_df[
            (training_df["var"] == "knee_ROM") & (training_df["tr_group"] == group)
        ]
        per_participant = (
            sub.groupby(["par", "timepoint"], observed=True)["value"]
            .mean()
            .unstack("timepoint")
            .sort_index(axis=1)
        )
        mean = per_participant.to_numpy(float).mean(axis=0)
        means[group] = mean
        linear = np.linspace(mean[0], mean[-1], mean.size)
        deviation = mean - linear
        rom = abs(mean[-1] - mean[0])
        out[group] = {
            "angle_start": round(float(mean[0]), 1),
            "angle_end": round(float(mean[-1]), 1),
            "rom": round(float(rom), 1),
            "max_deviation_deg": round(float(np.abs(deviation).max()), 1),
            "max_deviation_pct_rom": round(float(100 * np.abs(deviation).max() / rom), 1),
            "angle_at_node": {int(k): round(float(mean[k]), 1) for k in range(0, 101, 10)},
        }
        print(f"      {group}: {mean[0]:.1f} -> {mean[-1]:.1f} deg, "
              f"max departure from linear {np.abs(deviation).max():.1f} deg "
              f"({100 * np.abs(deviation).max() / rom:.1f}% of ROM)")

    gap = means["NH"] - means["IK"]
    out["angle_gap_at_same_node"] = {int(k): round(float(gap[k]), 1) for k in range(0, 101, 10)}
    out["max_angle_gap_deg"] = round(float(np.abs(gap).max()), 1)
    out["max_angle_gap_node"] = int(np.argmax(np.abs(gap)))
    print(f"      the same node denotes knee angles up to {np.abs(gap).max():.1f} deg apart "
          f"(at node {int(np.argmax(np.abs(gap)))})")
    out["_mean_curves"] = {g: means[g].tolist() for g in means}
    return out


def monotonicity_audit(training_df: pd.DataFrame) -> Dict[str, Any]:
    """Evidence for the invertibility assumption angle registration makes."""
    from jhamon_training.curves import backtrack_fraction, repetition_matrices

    out: Dict[str, Any] = {}
    for group in ("NH", "IK"):
        sub = training_df[training_df["tr_group"] == group]
        _, _, angles = repetition_matrices(sub, "torque")
        frac = backtrack_fraction(angles)
        out[group] = {
            "median": round(float(np.nanmedian(frac)), 4),
            "p99": round(float(np.nanpercentile(frac, 99)), 4),
            "pct_above_10pct": round(float(100 * np.nanmean(frac > 0.10)), 2),
        }
        print(f"      {group}: median backtrack {out[group]['median']:.4f} of ROM, "
              f"{out[group]['pct_above_10pct']:.2f}% of repetitions exceed 10%")
    return out


# --------------------------------------------------------------------------- #
# 2 & 3. the SPM families
# --------------------------------------------------------------------------- #
def axes_for(grid=PRIMARY_GRID):
    from jhamon_training.curves import knee_angle_axis, rom_percent_axis, time_axis

    return {
        "time": time_axis(),
        "rom_percent": rom_percent_axis(),
        "knee_angle": knee_angle_axis(hi=grid[0], lo=grid[1]),
    }


def between_group_all_axes(training_df: pd.DataFrame, var: str, label: str) -> Dict[str, Any]:
    """The NH-versus-IK contrast on each axis, holding the repetitions fixed."""
    from jhamon_training.curves import build_curveset, common_repetitions
    from jhamon_training.stats.spm_curves import between_groups

    axes = axes_for()
    shared = common_repetitions(training_df, var, list(axes.values()))
    results: Dict[str, Any] = {}
    for name, axis in axes.items():
        curves = build_curveset(training_df, var, axis, restrict_to=shared)
        res = between_groups(curves)
        results[name] = res
        print(f"      {label:>8} [{name:>11}] n={res['n_nh']}v{res['n_ik']} "
              f"t*={res['tstar']:.3f} df={res['df']} FWHM={res['fwhm']}  {res['summary']}")
    results["_n_shared_repetitions"] = int(len(shared))
    return results


def grid_sensitivity(training_df: pd.DataFrame, var: str, label: str) -> Dict[str, Any]:
    """Is the angle-axis conclusion an artefact of where the grid is cut?"""
    from jhamon_training.curves import build_curveset, knee_angle_axis
    from jhamon_training.stats.spm_curves import between_groups

    out: Dict[str, Any] = {}
    for hi, lo in GRID_VARIANTS:
        axis = knee_angle_axis(hi=hi, lo=lo)
        curves = build_curveset(training_df, var, axis)
        res = between_groups(curves)
        cov = res["coverage"]
        out[f"{hi:g}-{lo:g}"] = {**res, "coverage_nh": cov.get("NH"), "coverage_ik": cov.get("IK")}
        print(f"      {label:>8} [{hi:g}-{lo:g} deg] coverage NH {cov.get('NH', 0):.0%} / "
              f"IK {cov.get('IK', 0):.0%}  t*={res['tstar']:.3f}  {res['summary']}")
    return out


def progression_all_axes(training_df: pd.DataFrame) -> Dict[str, Any]:
    """The session-1-versus-session-k contrasts on each axis, per group."""
    from jhamon_training.curves import (build_curveset, common_repetitions,
                                        select_top_repetitions, time_axis)
    from jhamon_training.stats.spm_curves import progression, progression_summary

    axes = axes_for()
    shared = common_repetitions(training_df, "torque", list(axes.values()))

    # Select the top-3 repetitions ONCE, on the time axis, and reuse that
    # selection for every axis. Selecting per axis would let the angle grid's
    # truncation change which repetitions are compared, confounding the axis
    # with the sample.
    seed = select_top_repetitions(
        build_curveset(training_df, "torque", time_axis(), restrict_to=shared), n_reps=3
    )
    chosen = pd.MultiIndex.from_frame(seed.meta[["par", "trses", "set", "rep"]])
    print(f"      top-3 selection: {len(chosen):,} repetitions, applied to every axis")

    alphas = {"unc": 0.05, "bonf": 0.05 / 14}
    out: Dict[str, Any] = {"alpha_bonferroni": round(alphas["bonf"], 5), "n_comparisons": 14}
    for group in ("NH", "IK"):
        for atag, alpha in alphas.items():
            for name, axis in axes.items():
                curves = build_curveset(training_df, "torque", axis, restrict_to=chosen)
                per = progression(curves, group=group, alpha=alpha)
                summary = progression_summary(per)
                out[f"{group}_{atag}_{name}"] = {"per_session": per, **summary}
                print(f"      {group} [{atag:>4}] [{name:>11}] "
                      f"{summary['n_sig']}/{summary['n_tested']} significant, "
                      f"first sustained {summary['first_sustained'] or 'none'}")
    return out


def matched_angle_values(training_df: pd.DataFrame) -> Dict[str, Any]:
    """Group-mean torque and velocity at matched joint angles.

    An SPM cluster says where the curves differ, not by how much. On a shared
    angle axis the magnitude is finally quotable without qualification.
    """
    from jhamon_training.curves import build_curveset, knee_angle_axis

    axis = knee_angle_axis(hi=PRIMARY_GRID[0], lo=PRIMARY_GRID[1])
    out: Dict[str, Any] = {"angles": axis.values.tolist()}
    for var, label, _ in CURVE_VARS:
        curves = build_curveset(training_df, var, axis)
        entry = {}
        for group in ("NH", "IK"):
            m = curves.participant_means(group)
            entry[group] = {"mean": m.mean(axis=0).tolist(), "sd": m.std(axis=0, ddof=1).tolist()}
        entry["difference"] = (
            np.asarray(entry["NH"]["mean"]) - np.asarray(entry["IK"]["mean"])
        ).tolist()
        out[label] = entry
        idx = [int(np.argmin(np.abs(axis.values - a))) for a in (80, 70, 60, 50, 40, 30)]
        print(f"      {label}: NH-IK at 80/70/60/50/40/30 deg = "
              + ", ".join(f"{entry['difference'][i]:+.1f}" for i in idx))
    return out


def exclusion_census(training_df: pd.DataFrame) -> Dict[str, Any]:
    """What the corrected exclusions remove that the legacy ones did not."""
    from jhamon_training.data.utils import dame_manual_exclusions

    keys = ["par", "trses", "set", "rep"]
    out: Dict[str, Any] = {}
    base = training_df.drop_duplicates(keys).groupby("tr_group").size()
    for tag, kwargs in (("legacy", {"legacy": True}), ("corrected", {})):
        filtered = dame_manual_exclusions(training_df, **kwargs)
        reps = filtered.drop_duplicates(keys).groupby("tr_group").size()
        per_var = filtered.groupby(["tr_group", "var"], observed=True).size().unstack(fill_value=0)
        out[tag] = {
            "repetitions": {g: int(reps.get(g, 0)) for g in ("NH", "IK")},
            "rows_per_var": {g: {v: int(per_var.loc[g, v]) for v in per_var.columns}
                             for g in per_var.index if g in ("NH", "IK")},
        }
        print(f"      {tag:>9}: repetitions surviving NH {int(reps.get('NH', 0)):,} / "
              f"IK {int(reps.get('IK', 0)):,}")
    out["baseline_repetitions"] = {g: int(base.get(g, 0)) for g in ("NH", "IK")}
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def draw_mapping(mapping: Dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    nodes = np.arange(101)
    for group, colour in (("NH", NH_COLOUR), ("IK", IK_COLOUR)):
        mean = np.asarray(mapping["_mean_curves"][group])
        ax1.plot(nodes, mean, color=colour, lw=2.2, label=group)
        ax1.plot(nodes, np.linspace(mean[0], mean[-1], 101), color=colour, lw=1,
                 ls="--", alpha=0.55)
    ax1.set_xlabel("Repetition duration (%)")
    ax1.set_ylabel("Knee flexion angle (deg)")
    ax1.set_title("Angle advances with time — dashed = constant rate", fontsize=10, loc="left")
    ax1.legend(frameon=False, fontsize=9)

    gap = np.asarray(mapping["_mean_curves"]["NH"]) - np.asarray(mapping["_mean_curves"]["IK"])
    ax2.fill_between(nodes, 0, gap, color="#B91C1C", alpha=0.22, lw=0)
    ax2.plot(nodes, gap, color="#B91C1C", lw=2)
    ax2.axhline(0, color="#525252", lw=0.8)
    ax2.set_xlabel("Repetition duration (%)")
    ax2.set_ylabel("NH − IK knee angle (deg)")
    ax2.set_title("Angle mismatch at the same node — the confound", fontsize=10, loc="left")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"      wrote {path}")


def draw_comparison(training_df: pd.DataFrame, between: Dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jhamon_training.curves import build_curveset, common_repetitions

    axes_spec = axes_for()
    fig, grid = plt.subplots(2, 2, figsize=(12.5, 7.8))
    for row, (var, label, ylabel) in enumerate(CURVE_VARS):
        shared = common_repetitions(training_df, var, list(axes_spec.values()))
        for col, name in enumerate(("time", "knee_angle")):
            ax = grid[row, col]
            axis = axes_spec[name]
            curves = build_curveset(training_df, var, axis, restrict_to=shared)
            for group, colour in (("NH", NH_COLOUR), ("IK", IK_COLOUR)):
                m = curves.participant_means(group)
                mu, sd = m.mean(axis=0), m.std(axis=0, ddof=1)
                ax.plot(axis.values, mu, color=colour, lw=2.2, label=group, zorder=3)
                ax.fill_between(axis.values, mu - sd, mu + sd, color=colour, alpha=0.16, lw=0)
            for c in between[label][name]["clusters"]:
                ax.axvspan(min(c["start"], c["end"]), max(c["start"], c["end"]),
                           color=POS_COLOUR if c["direction"] > 0 else NEG_COLOUR,
                           alpha=0.13, lw=0, zorder=1)
            if axis.kind == "knee_angle":
                ax.invert_xaxis()
            ax.set_xlabel(axis.label)
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.set_title(f"{label} — {'shipped time axis' if col == 0 else 'knee-angle axis'}",
                         fontsize=10, loc="left")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Identical repetitions, two abscissae.  Shaded = supra-threshold SPM{t} "
                 "cluster (green NH>IK, purple IK>NH)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"      wrote {path}")


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    ap.add_argument("--results-path", default=None, help="override the results folder")
    ap.add_argument("--output-dir", default=None, help="where to write JSON and figures")
    args = ap.parse_args(argv)

    if args.results_path:
        os.environ["JHAMON_RESULTS"] = str(Path(args.results_path).expanduser().resolve())
    from jhamon_training.pathutils import RESULTS_TRAINING_PATH as results_path

    out_dir = Path(args.output_dir).expanduser() if args.output_dir else results_path
    fig_dir = out_dir / "figures"

    print("=" * 78)
    print("jHamON -- SPM abscissa comparison: time versus knee angle")
    print("=" * 78)
    print(f"      results path {results_path}")

    print("\n[1/7] Loading caches")
    print("-" * 78)
    training_df = load_curves(results_path)

    print("\n[2/7] The time-to-angle mapping")
    print("-" * 78)
    mapping = axis_mapping(training_df)
    print("      invertibility audit (angle traces must be monotone to register):")
    monotonicity = monotonicity_audit(training_df)

    print("\n[3/7] Manual exclusions: legacy versus corrected")
    print("-" * 78)
    census = exclusion_census(training_df)

    print("\n[4/7] Between-modality SPM{t} on each axis")
    print("-" * 78)
    from jhamon_training.data.utils import dame_manual_exclusions
    regimes = {"none": training_df, "corrected": dame_manual_exclusions(training_df)}

    between: Dict[str, Any] = {}
    for regime, data in regimes.items():
        print(f"    -- exclusions: {regime} --")
        between[regime] = {}
        for var, label, _ in CURVE_VARS:
            between[regime][label] = between_group_all_axes(data, var, label)

    print("\n[5/7] Angle-grid sensitivity")
    print("-" * 78)
    grids = {label: grid_sensitivity(training_df, var, label) for var, label, _ in CURVE_VARS}

    print("\n[6/7] Progression SPM{t} on each axis")
    print("-" * 78)
    prog = progression_all_axes(regimes["corrected"])

    print("\n      magnitudes at matched joint angles:")
    matched = matched_angle_values(training_df)

    print("\n[7/7] Writing")
    print("-" * 78)
    payload = {
        "primary_grid": {"hi": PRIMARY_GRID[0], "lo": PRIMARY_GRID[1]},
        "mapping": mapping,
        "monotonicity": monotonicity,
        "exclusions": census,
        "between_groups": between,
        "grid_sensitivity": grids,
        "progression": prog,
        "matched_angle_values": matched,
    }
    if args.dry_run:
        print("      --dry-run: nothing written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    draw_mapping(mapping, fig_dir / "axis_mapping.png")
    draw_comparison(training_df, between["none"], fig_dir / "axis_comparison.png")

    target = out_dir / "axis_comparison.json"
    fd, tmp = tempfile.mkstemp(dir=str(out_dir), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    print(f"      wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
