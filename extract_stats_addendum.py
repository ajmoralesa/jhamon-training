#!/usr/bin/env python3
"""Flatten the last few quantities the manuscript prose quotes directly.

Two gaps remain after ``reproduce.py``, ``extract_manuscript_stats.py`` and
``extract_validity_stats.py``:

1. **Welch degrees of freedom for the repetition-count and repetition-difference
   tests.** ``extract_manuscript_stats.py`` stores the t statistic and the p
   value but not the df, and a t reported without its df is not reportable.

2. **The terminal progression contrasts.** ``manuscript_stats_detail.json``
   holds every session-by-session SPM cluster, but the Results sentence needs
   the session-1-versus-session-15 contrast for each group as scalars: the
   number of paired participants, the cluster extent and the cluster-level p.
   Collapsing 14 comparisons into ``cluster_start_min``/``cluster_start_max``
   loses exactly the contrast a reader wants named.

Both are pure re-expressions of quantities already computed; nothing new is
estimated here. Run after the other two scripts.

Usage:
    source venv/bin/activate && python extract_stats_addendum.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from reproduce import _install_legacy_pickle_shim, _pseudonymise


def _welch_df(a: np.ndarray, b: np.ndarray) -> float:
    """Welch-Satterthwaite degrees of freedom for two independent samples."""
    va, vb = np.var(a, ddof=1) / a.size, np.var(b, ddof=1) / b.size
    return float((va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1)))


def repetition_test_df(results_path: Path) -> Dict[str, Any]:
    """Welch df and CI for the actual-versus-actual repetitions-per-set contrast.

    The published "+3.1 +- 0.5 repetitions" compares isokinetic ACHIEVED counts
    against Nordic PRESCRIBED counts, and its "+- 0.5" is the spread across
    session x set cells rather than across participants. This recomputes the
    like-for-like contrast with the participant as the unit, and returns the
    pieces a reportable t-test needs: difference, 95 % CI, t, df, p.
    """
    from jhamon_training.data import frames
    from jhamon_training.data.utils import filter_training_data
    from jhamon_training.saveload import load_obj
    from jhamon_training.stats.analyses_discrete_vars import analyze_discrete_variables

    _install_legacy_pickle_shim()
    nordict = _pseudonymise(load_obj(name="nht_results.pkl", path=results_path))
    ikdict = _pseudonymise(load_obj(name="ikt_results.pkl", path=results_path))
    nordf = frames.nht_todf(my_dict=nordict)
    ikdf = frames.ikt_todf(my_dict=ikdict)

    with tempfile.TemporaryDirectory(prefix="jhamon_addendum_") as tmpdir:
        disc = analyze_discrete_variables(
            nordict=nordict,
            ikdf=ikdf,
            training_df=pd.concat([nordf, ikdf]),
            results_output_path=Path(tmpdir),
        )["training_disc"]

    out: Dict[str, Any] = {}
    for tag, data in (("all", disc), ("analysed", filter_training_data(disc.copy()))):
        per_participant = (
            data.groupby(["par", "tr_group", "trses", "set"], observed=True)
            .size()
            .reset_index(name="n")
            .groupby(["par", "tr_group"], observed=True)["n"]
            .mean()
            .reset_index()
        )
        nh = per_participant.loc[per_participant["tr_group"] == "NH", "n"].to_numpy(float)
        ik = per_participant.loc[per_participant["tr_group"] == "IK", "n"].to_numpy(float)
        res = sps.ttest_ind(ik, nh, equal_var=False)
        ci = res.confidence_interval()
        out[f"reps_{tag}_perset_df"] = round(_welch_df(ik, nh), 1)
        out[f"reps_{tag}_perset_ci_low"] = round(float(ci.low), 2)
        out[f"reps_{tag}_perset_ci_high"] = round(float(ci.high), 2)
        print(
            f"      reps/set [{tag:>8}] IK-NH {ik.mean() - nh.mean():+.2f} "
            f"[{ci.low:.2f}, {ci.high:.2f}], t({_welch_df(ik, nh):.1f}) = {res.statistic:.2f}, "
            f"p = {res.pvalue:.2e}"
        )

    # Welch df for the work-per-repetition test (participant level).
    per_participant = (
        disc.groupby(["par", "tr_group"], observed=True)["work"].mean().reset_index()
    )
    nh = per_participant.loc[per_participant["tr_group"] == "NH", "work"].to_numpy(float)
    ik = per_participant.loc[per_participant["tr_group"] == "IK", "work"].to_numpy(float)
    ci = sps.ttest_ind(nh, ik, equal_var=False).confidence_interval()
    out["workrep_ci_low"] = round(float(ci.low), 1)
    out["workrep_ci_high"] = round(float(ci.high), 1)
    print(f"      work/rep NH-IK {nh.mean() - ik.mean():+.1f} J [{ci.low:.1f}, {ci.high:.1f}]")
    return out


def terminal_progression(results_path: Path, stats_path: Path) -> Dict[str, Any]:
    """Session-1-versus-session-15 SPM contrast, per group, as scalar keys.

    Reads the per-session detail written by ``extract_manuscript_stats.py`` and
    lifts out the final contrast, which is the one the Results sentence names.
    Both alpha levels are carried through so the corrected and uncorrected
    versions can be quoted side by side.
    """
    detail = json.loads((stats_path.parent / "manuscript_stats_detail.json").read_text())
    out: Dict[str, Any] = {}
    for group in ("nh", "ik"):
        for tag in ("unc", "bonf"):
            entry = detail.get(f"prog_{group}_{tag}", {}).get("tr_15", {})
            clusters = entry.get("clusters_t", [])
            significant = [c for c in clusters if c["p"] < entry.get("alpha", 0.05)]
            prefix = f"prog_{group}_{tag}_tr15"
            out[f"{prefix}_n"] = entry.get("n")
            out[f"{prefix}_tstar"] = entry.get("tstar")
            out[f"{prefix}_n_clusters"] = len(significant)
            if significant:
                widest = max(significant, key=lambda c: c["extent_pct"])
                out[f"{prefix}_start"] = widest["start_pct"]
                out[f"{prefix}_end"] = widest["end_pct"]
                out[f"{prefix}_p"] = round(float(widest["p"]), 4)
            print(f"      {prefix}: n={out[f'{prefix}_n']} clusters={len(significant)}")
    return out


def cluster_angles_and_magnitudes(results_path: Path, stats_path: Path) -> Dict[str, Any]:
    """Knee-angle range and NH-vs-IK magnitude inside each SPM cluster.

    An SPM cluster is reported as a percentage of the repetition, which says
    *where* the two modalities differ but not *by how much* nor *at what joint
    angle* -- and the joint angle is what a reader of a knee-flexor paper needs,
    because the two modalities traverse the same total range of motion at
    different instantaneous rates, so equal percentages are not equal angles.

    For every supra-threshold cluster of the participant-averaged between-group
    contrasts this computes, from the group-mean normalised curves:

    * the knee angle at the start and end of the cluster, in each group;
    * the mean value of the outcome inside the cluster, in each group;
    * the absolute and relative NH - IK difference inside the cluster.

    Angles follow the convention used throughout (0 deg = full extension).
    """
    from jhamon_training.data import frames
    from jhamon_training.saveload import load_obj

    _install_legacy_pickle_shim()
    nordict = _pseudonymise(load_obj(name="nht_results.pkl", path=results_path))
    ikdict = _pseudonymise(load_obj(name="ikt_results.pkl", path=results_path))
    training_df = pd.concat(
        [frames.nht_todf(my_dict=nordict), frames.ikt_todf(my_dict=ikdict)]
    )

    def group_mean_curve(var: str, group: str) -> np.ndarray:
        """Participant-averaged group-mean curve, matching the SPM unit."""
        sub = training_df[(training_df["var"] == var) & (training_df["tr_group"] == group)]
        per_participant = (
            sub.groupby(["par", "timepoint"], observed=True)["value"]
            .mean()
            .unstack("timepoint")
            .sort_index(axis=1)
        )
        return per_participant.to_numpy(dtype=float).mean(axis=0)

    curves = {
        (var, group): group_mean_curve(var, group)
        for var in ("torque", "knee_v", "knee_ROM")
        for group in ("NH", "IK")
    }

    stats = json.loads(stats_path.read_text())
    out: Dict[str, Any] = {}
    for tag, var in (("torque", "torque"), ("velocity", "knee_v")):
        prefix = f"spm_{tag}_paravg"
        for n in range(1, int(stats.get(f"{prefix}_n_clusters", 0)) + 1):
            start = int(round(stats[f"{prefix}_c{n}_start"]))
            end = int(round(stats[f"{prefix}_c{n}_end"]))
            lo, hi = min(start, end), max(start, end)
            key = f"{prefix}_c{n}"
            for group in ("NH", "IK"):
                angle = curves[("knee_ROM", group)]
                out[f"{key}_{group.lower()}_angle_start"] = round(float(angle[lo]), 1)
                out[f"{key}_{group.lower()}_angle_end"] = round(float(angle[min(hi, 100)]), 1)
                value = curves[(var, group)][lo : min(hi, 100) + 1]
                out[f"{key}_{group.lower()}_value"] = round(float(np.abs(value).mean()), 1)
            nh_v = out[f"{key}_nh_value"]
            ik_v = out[f"{key}_ik_value"]
            out[f"{key}_diff"] = round(nh_v - ik_v, 1)
            out[f"{key}_pct_diff"] = (
                round(100.0 * (nh_v - ik_v) / ik_v, 1) if ik_v else None
            )
            print(
                f"      {key}: {lo}-{hi}% | NH {out[f'{key}_nh_angle_start']:.1f}->"
                f"{out[f'{key}_nh_angle_end']:.1f} deg, IK {out[f'{key}_ik_angle_start']:.1f}->"
                f"{out[f'{key}_ik_angle_end']:.1f} deg | NH {nh_v} vs IK {ik_v} "
                f"({out[f'{key}_pct_diff']:+.1f}%)"
            )
    return out


def qc_filter_census(results_path: Path) -> Dict[str, Any]:
    """Exactly which repetitions the 10-35 deg/s window removes, and which way it pushes.

    The window filters on a variable that is also a reported outcome, so its
    effect has to be quantified rather than asserted. It is NOT symmetric: the
    Nordic distribution is truncated mainly at the fast end (segmentation
    failures reaching implausible speeds for a Nordic descent) while the
    isokinetic distribution can only lose repetitions at the slow end, because
    an isokinetic movement cannot exceed its programmed speed.

    That asymmetry moves the two group means TOWARDS each other, which is
    anti-conservative for a claim that the two modalities were velocity-matched.
    Reported so the manuscript can say so plainly and use the unfiltered
    population as primary for the agreement analyses.
    """
    import tempfile as _tf

    from jhamon_training.data import frames
    from jhamon_training.saveload import load_obj
    from jhamon_training.stats.analyses_discrete_vars import analyze_discrete_variables

    from extract_validity_stats import protocol_subset

    _install_legacy_pickle_shim()
    nordict = _pseudonymise(load_obj(name="nht_results.pkl", path=results_path))
    ikdict = _pseudonymise(load_obj(name="ikt_results.pkl", path=results_path))
    with _tf.TemporaryDirectory(prefix="jhamon_qc_") as tmpdir:
        disc = analyze_discrete_variables(
            nordict=nordict,
            ikdf=frames.ikt_todf(my_dict=ikdict),
            training_df=pd.concat(
                [frames.nht_todf(my_dict=nordict), frames.ikt_todf(my_dict=ikdict)]
            ),
            results_output_path=Path(tmpdir),
        )["training_disc"]

    prot = protocol_subset(disc)
    out: Dict[str, Any] = {}
    for group in ("NH", "IK"):
        v = prot.loc[prot["tr_group"] == group, "knee_v_mean"].to_numpy(float)
        fast, slow = int((v >= 35).sum()), int((v <= 10).sum())
        kept = v[(v > 10) & (v < 35)]
        tag = group.lower()
        out[f"qc_{tag}_n"] = int(v.size)
        out[f"qc_{tag}_fast"] = fast
        out[f"qc_{tag}_slow"] = slow
        out[f"qc_{tag}_removed_pct"] = round(100.0 * (fast + slow) / v.size, 2)
        out[f"qc_{tag}_mean_before"] = round(float(v.mean()), 2)
        out[f"qc_{tag}_mean_after"] = round(float(kept.mean()), 2)
        out[f"qc_{tag}_mean_shift"] = round(float(kept.mean() - v.mean()), 2)
        print(
            f"      {group}: n={v.size} removed fast={fast} slow={slow} "
            f"({out[f'qc_{tag}_removed_pct']}%), mean {v.mean():.2f} -> {kept.mean():.2f}"
        )
    out["qc_gap_before"] = round(out["qc_nh_mean_before"] - out["qc_ik_mean_before"], 2)
    out["qc_gap_after"] = round(out["qc_nh_mean_after"] - out["qc_ik_mean_after"], 2)
    print(f"      NH-IK velocity gap {out['qc_gap_before']} -> {out['qc_gap_after']}")
    return out


def nordic_has_no_plateau(results_path: Path) -> Dict[str, Any]:
    """Evidence that the plateau rule is meaningful for IK and meaningless for NH.

    The velocity comparison is deliberately asymmetric -- the Nordic value is a
    whole-range mean, the isokinetic value a constant-velocity-phase mean -- and
    the justification has to be shown, not asserted. Applying the same
    95 %-of-peak rule to the Nordic curves is the direct test: if a Nordic
    descent had a constant-velocity phase, the rule would find one.

    It does not. The Nordic profile accelerates monotonically to the end of its
    window, so the rule selects essentially the single fastest sample, and a
    symmetric fixed mid-window rule measures a different part of an accelerating
    ramp in each modality. Both numbers are stored so the manuscript can state
    why no equivalent trimming was applied to the Nordic side.
    """
    from jhamon_training.data import frames
    from jhamon_training.saveload import load_obj

    _install_legacy_pickle_shim()
    frames_by_group = {
        "nh": frames.nht_todf(
            my_dict=_pseudonymise(load_obj(name="nht_results.pkl", path=results_path))
        ),
        "ik": frames.ikt_todf(
            my_dict=_pseudonymise(load_obj(name="ikt_results.pkl", path=results_path))
        ),
    }

    out: Dict[str, Any] = {}
    for tag, frame in frames_by_group.items():
        sub = frame[frame["var"] == "knee_v"]
        wide = (
            sub.groupby(["par", "trses", "set", "rep", "timepoint"], observed=True)["value"]
            .mean()
            .unstack("timepoint")
            .sort_index(axis=1)
        )
        Y = np.abs(wide.to_numpy(dtype=float))
        mask = Y >= 0.95 * np.nanmax(Y, axis=1, keepdims=True)
        out[f"velphase_{tag}_p95_mean"] = round(float(np.nanmean(np.where(mask, Y, np.nan))), 2)
        out[f"velphase_{tag}_p95_width_median"] = int(np.median(mask.sum(axis=1)))
        out[f"velphase_{tag}_mid2575_mean"] = round(float(np.nanmean(Y[:, 25:76])), 2)
        out[f"velphase_{tag}_whole_mean"] = round(float(np.nanmean(Y)), 2)
        print(
            f"      {tag.upper()}: 95%-of-peak mean {out[f'velphase_{tag}_p95_mean']} deg/s "
            f"over a median {out[f'velphase_{tag}_p95_width_median']} node(s); "
            f"mid 25-75% {out[f'velphase_{tag}_mid2575_mean']}; "
            f"whole window {out[f'velphase_{tag}_whole_mean']}"
        )
    return out


def spm_error_df(results_path: Path) -> Dict[str, Any]:
    """Residual degrees of freedom of the participant-averaged between-group SPM.

    A t statistic reported without its degrees of freedom is not reportable, and
    for these contrasts the df is not simply n1 + n2 - 2: ``spm1d``'s
    ``equal_var=False`` estimates a non-sphericity component per group and
    returns a single ReML effective df applied to every node. That is why the
    velocity threshold (t* = 3.60) is so much higher than the torque one
    (t* = 2.97) on the same 13-versus-14 design -- the velocity residuals are far
    more heterogeneous between groups.
    """
    from jhamon_training.data import frames
    from jhamon_training.saveload import load_obj

    from extract_manuscript_stats import spm_between_groups

    _install_legacy_pickle_shim()
    training_df = pd.concat(
        [
            frames.nht_todf(
                my_dict=_pseudonymise(load_obj(name="nht_results.pkl", path=results_path))
            ),
            frames.ikt_todf(
                my_dict=_pseudonymise(load_obj(name="ikt_results.pkl", path=results_path))
            ),
        ]
    )
    out: Dict[str, Any] = {}
    for tag, var in (("torque", "torque"), ("velocity", "knee_v")):
        res = spm_between_groups(training_df, var, by_participant=True)
        out[f"spm_{tag}_paravg_df"] = round(float(np.ravel(res["df"])[-1]), 1)
        print(f"      {tag}: t* = {res['tstar']}, error df = {out[f'spm_{tag}_paravg_df']}")
    return out


def hedges_g_for_counts(stats_path: Path) -> Dict[str, Any]:
    """Hedges' g for the repetitions-per-set contrast (stored d is uncorrected)."""
    stats = json.loads(stats_path.read_text())
    n_nh, n_ik = int(stats["lmm_n_nh"]), int(stats["lmm_n_ik"])
    j = 1.0 - 3.0 / (4.0 * (n_nh + n_ik) - 9.0)
    out = {
        f"reps_{tag}_perset_g": round(stats[f"reps_{tag}_perset_d"] * j, 2)
        for tag in ("all", "analysed")
    }
    # The MVIF effect size is stored under a "_d" name but is quoted as g.
    out["mvif_max_g"] = round(stats["mvif_max_d"] * (1.0 - 3.0 / (4.0 * 26 - 9.0)), 3)
    print(f"      {out}")
    return out


def merge_into_json(flat: Dict[str, Any], json_path: Path) -> None:
    with open(json_path) as f:
        existing = json.load(f)
    n_before = len(existing)
    existing.update(flat)
    fd, tmp = tempfile.mkstemp(dir=str(json_path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(existing, f, indent=4)
        os.replace(tmp, json_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    print(f"\nwrote {json_path}: {n_before} keys before, {len(existing)} after.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--results-path", default=None, help="override the results folder")
    ap.add_argument(
        "--stats-path",
        default=None,
        help=(
            "training_stats.json to read and update. Defaults to the one in the results "
            "folder; reproduce.py passes its staging copy so the real file is only replaced "
            "at the end of a successful run."
        ),
    )
    args = ap.parse_args(argv)

    if args.results_path:
        os.environ["JHAMON_RESULTS"] = str(Path(args.results_path).expanduser().resolve())
    from jhamon_training.pathutils import RESULTS_TRAINING_PATH as results_path

    stats_path = Path(args.stats_path) if args.stats_path else results_path / "training_stats.json"

    print("[1/7] Welch df and confidence intervals for the count/work contrasts")
    print("-" * 72)
    flat = repetition_test_df(results_path)

    print("\n[2/7] Terminal progression contrasts (session 1 vs session 15)")
    print("-" * 72)
    flat.update(terminal_progression(results_path, stats_path))

    print("\n[3/7] Knee-angle range and magnitude inside each SPM cluster")
    print("-" * 72)
    flat.update(cluster_angles_and_magnitudes(results_path, stats_path))

    print("\n[4/7] Census of the 10-35 deg/s quality-control window")
    print("-" * 72)
    flat.update(qc_filter_census(results_path))

    print("\n[5/7] Is there a constant-velocity phase in a Nordic descent?")
    print("-" * 72)
    flat.update(nordic_has_no_plateau(results_path))

    print("\n[6/7] Residual degrees of freedom of the between-group SPM")
    print("-" * 72)
    flat.update(spm_error_df(results_path))

    print("\n[7/7] Hedges' g for the count contrasts")
    print("-" * 72)
    flat.update(hedges_g_for_counts(stats_path))

    if args.dry_run:
        print(json.dumps(flat, indent=2))
        return 0
    merge_into_json(flat, stats_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
