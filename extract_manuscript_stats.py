#!/usr/bin/env python3
"""Compute every manuscript statistic that ``reproduce.py`` does not produce.

Why this exists
---------------
``reproduce.py`` rebuilds the descriptive block, the linear mixed models and the
Bland-Altman agreement analysis, and writes them to ``training_stats.json``.
Four families of numbers that the Methods section promises are still missing
from that file, so the manuscript either quotes them from a 2020 draft, quotes
them with no test behind them, or does not quote them at all:

1. **Between-group SPM{t}** on the torque-angle and velocity-angle curves.
   ``stats/training.py:spm_group_comparison`` computes it and the figure code
   draws it, but no cluster extent or cluster-level p value is ever persisted.
   The manuscript's "0-37%", "48% to 100%" and "CT = 2.729" are copied from an
   old draft and have never been re-derived. Both units of analysis are
   computed here -- the shipped one (one curve per participant x session, which
   is pseudoreplicated) and the participant-averaged one (13 vs 14 curves) --
   so the choice is made on evidence rather than inherited.

2. **Within-group progression SPM** across the 15 sessions. The code runs
   ``spm1d.stats.anova1rm`` (a one-way repeated-measures SPM{F}) while the
   Methods and the figure caption describe a paired SPM{t}. With only two
   levels these are the *same test* (F = t^2), which this script verifies
   numerically; the t form is computed as well because it carries direction and
   the F form does not. Bonferroni-corrected inference for the 14 comparisons
   is computed alongside the uncorrected inference, because the Methods promise
   a correction the code never applies.

3. **Baseline equivalence in isometric knee-flexion torque (MVIF)**, the
   quantity Eq. 1 scales the isokinetic prescription by. Reported for all three
   defensible reduction rules over each participant's familiarisation sessions,
   so the conclusion can be shown to be rule-invariant.

4. **The work-per-repetition comparison.** ``jhamon_results.qmd`` states
   "p < 0.001" as literal text with no test anywhere in the codebase. The
   participant-level Welch test is run here so the claim has a statistic.

Plus two harmonisation quantities the audit flagged: actual-versus-actual
repetition counts per set (the published "+3.1" compares IK *achieved* against
NH *prescribed*), and the post-filter work and velocity descriptives (the
published ones are pre-filter while every inferential statistic is post-filter).

Everything is merged into ``training_stats.json`` under new prefixes
(``spm_``, ``prog_``, ``mvif_``, ``workrep_``, ``reps_``, ``desc_``) without
touching any existing key. Run it AFTER ``reproduce.py``.

Usage:
    source venv/bin/activate && python extract_manuscript_stats.py [--dry-run]
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
from scipy import stats as sps

# The 2019 caches were pickled under pandas < 1.3; reuse reproduce.py's shim.
from reproduce import _install_legacy_pickle_shim, _pseudonymise

CURVE_NODES = 101  # every repetition is time-normalised to 101 points => node == %


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _r(value: Any, nd: int) -> Any:
    """Round for JSON; non-finite becomes None so the file stays valid."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return None if not np.isfinite(f) else round(f, nd)


def _clusters(spmi) -> List[Dict[str, Any]]:
    """Extract supra-threshold clusters from an inferred spm1d object.

    Node indices are returned directly as percentages of the movement, which is
    exact because the curves carry 101 nodes (0-100). ``direction`` is the sign
    of the test statistic inside the cluster; for a two-sample SPM{t} with NH
    first, +1 means NH > IK.
    """
    out: List[Dict[str, Any]] = []
    for c in getattr(spmi, "clusters", []) or []:
        start, end = (float(v) for v in c.endpoints)
        z_in = np.asarray(spmi.z)[int(np.floor(start)) : int(np.ceil(end)) + 1]
        out.append(
            {
                "start_pct": round(start, 1),
                "end_pct": round(end, 1),
                "p": float(c.P),
                "extent_pct": round(end - start, 1),
                "z_peak": round(float(np.nanmax(np.abs(z_in))), 3) if z_in.size else None,
                "direction": int(np.sign(np.nanmean(z_in))) if z_in.size else 0,
            }
        )
    return sorted(out, key=lambda d: d["start_pct"])


def _fmt_clusters(clusters: Sequence[Dict[str, Any]], pos: str, neg: str) -> str:
    """One-line human-readable summary of a cluster list, for the console."""
    if not clusters:
        return "no supra-threshold clusters"
    parts = []
    for c in clusters:
        who = pos if c["direction"] > 0 else neg
        p = "<0.001" if c["p"] < 0.001 else f"={c['p']:.3f}"
        parts.append(f"{who} {c['start_pct']}-{c['end_pct']}% (p{p})")
    return " | ".join(parts)


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with the pooled SD (Hedges' correction NOT applied)."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("nan")


# --------------------------------------------------------------------------- #
# 1. between-group SPM on the whole curves
# --------------------------------------------------------------------------- #
def _curve_matrix(df: pd.DataFrame, group: str, by_participant: bool) -> np.ndarray:
    """Stack one group's normalised curves into an (observations x 101) array.

    Args:
        df: long curve frame for a single ``var`` (torque or knee_v).
        group: ``"NH"`` or ``"IK"``.
        by_participant: if True, average each participant's sessions into a
            single curve, so the rows are independent participants. If False,
            reproduce the shipped behaviour of one row per participant x
            session, which treats ~15 curves from one person as independent.
    """
    sub = df[df["tr_group"] == group]
    keys = ["par"] if by_participant else ["par", "trses"]
    wide = (
        sub.groupby(keys + ["timepoint"], observed=True)["value"]
        .mean()
        .unstack("timepoint")
        .sort_index(axis=1)
    )
    Y = wide.to_numpy(dtype=float)
    if Y.shape[1] != CURVE_NODES:
        raise ValueError(f"expected {CURVE_NODES} nodes, got {Y.shape[1]}")
    if np.isnan(Y).any():
        raise ValueError(f"{group}: NaN in the curve matrix ({by_participant=})")
    return Y


def spm_between_groups(training_df: pd.DataFrame, var: str, by_participant: bool) -> Dict[str, Any]:
    """Two-sample SPM{t}, NH vs IK, unequal variances, two-tailed, alpha 0.05.

    The sign convention follows ``spm_group_comparison``: NH is the first
    sample, so a positive supra-threshold cluster means NH > IK.
    """
    import spm1d

    df = training_df[training_df["var"] == var]
    Y_nh = _curve_matrix(df, "NH", by_participant)
    Y_ik = _curve_matrix(df, "IK", by_participant)
    t = spm1d.stats.ttest2(Y_nh, Y_ik, equal_var=False)
    ti = t.inference(0.05, two_tailed=True)
    return {
        "var": var,
        "unit": "participant" if by_participant else "participant x session",
        "n_nh": int(Y_nh.shape[0]),
        "n_ik": int(Y_ik.shape[0]),
        "tstar": round(float(ti.zstar), 3),
        "df": [round(float(v), 1) for v in np.ravel(ti.df)],
        "clusters": _clusters(ti),
    }


# --------------------------------------------------------------------------- #
# 2. within-group progression SPM across sessions
# --------------------------------------------------------------------------- #
def _select_top_curves(df: pd.DataFrame, n_reps: int = 3) -> pd.DataFrame:
    """Reproduce ``jhamon_training.select_top_repetitions`` exactly.

    Selects, per participant and session, the ``n_reps`` repetitions with the
    highest peak of the *normalised torque curve* (note: the LMM instead selects
    on the discrete ``peak_torque`` field -- two rules that usually agree).
    """
    peaks = df.groupby(["par", "trses", "set", "rep"])["value"].max().reset_index()
    top = peaks.sort_values(["par", "trses", "value"], ascending=[True, True, False])
    top = top.groupby(["par", "trses"]).head(n_reps)
    idx_top = pd.MultiIndex.from_frame(top[["par", "trses", "set", "rep"]])
    idx_all = pd.MultiIndex.from_frame(df[["par", "trses", "set", "rep"]])
    return df[idx_all.isin(idx_top)]


def spm_progression(df_group: pd.DataFrame, alpha: float, label: str) -> Dict[str, Any]:
    """Session 1 vs session k for k = 2..15, as SPM{F} and as paired SPM{t}.

    ``stats/spm.py:spm_nh_kinetics`` runs ``spm1d.stats.anova1rm`` on the two
    sessions. With two levels a one-way repeated-measures ANOVA *is* a paired
    t-test (F = t^2), so both are computed and the identity is checked
    numerically: if it holds, the Methods' "paired SPM{t}" and the code's
    SPM{F} are the same inference and only the reported statistic differs. The
    t form is preferred for reporting because it carries direction.

    Args:
        df_group: long torque-curve frame for one training group, already
            reduced to the top repetitions.
        alpha: per-comparison alpha. Pass ``0.05 / 14`` for the Bonferroni
            correction the Methods promise.
        label: group name, for the console log.
    """
    import spm1d

    out: Dict[str, Any] = {}
    for k in range(2, 16):
        pair = df_group[df_group["trses"].isin(["tr_1", f"tr_{k}"])]

        # Balanced design: a participant must have both sessions.
        have_both = pair.groupby("par")["trses"].nunique()
        keep = have_both[have_both == 2].index
        pair = pair[pair["par"].isin(keep)]
        if len(keep) < 3:
            out[f"tr_{k}"] = {"n": int(len(keep)), "skipped": "fewer than 3 paired participants"}
            continue

        # Average across the selected repetitions -> one curve per (par, session).
        wide = (
            pair.groupby(["par", "trses", "timepoint"], observed=True)["value"]
            .mean()
            .unstack("timepoint")
            .sort_index(axis=1)
        )
        Y = wide.to_numpy(dtype=float)
        subj = pd.factorize(wide.index.get_level_values("par"))[0]
        time = pd.factorize(wide.index.get_level_values("trses"))[0]

        F = spm1d.stats.anova1rm(Y=Y, A=time, SUBJ=subj, equal_var=True)
        Fi = F.inference(alpha)

        # Paired t on the same data, ordered session 1 -> session k.
        order = np.argsort(wide.index.get_level_values("par").to_numpy(), kind="stable")
        first = Y[[i for i in order if wide.index.get_level_values("trses")[i] == "tr_1"]]
        later = Y[[i for i in order if wide.index.get_level_values("trses")[i] == f"tr_{k}"]]
        t = spm1d.stats.ttest_paired(later, first)  # positive => session k > session 1
        ti = t.inference(alpha, two_tailed=True)

        f_eq_t2 = bool(np.allclose(np.asarray(Fi.z), np.asarray(ti.z) ** 2, rtol=1e-6, atol=1e-6))
        out[f"tr_{k}"] = {
            "n": int(len(keep)),
            "alpha": alpha,
            "fstar": round(float(Fi.zstar), 3),
            "tstar": round(float(ti.zstar), 3),
            "f_equals_t_squared": f_eq_t2,
            "clusters_t": _clusters(ti),
            "n_clusters_f": len(_clusters(Fi)),
        }
        print(
            f"      [{label} tr_1 vs tr_{k}] n={len(keep)} t*={ti.zstar:.3f} "
            f"F=t^2:{f_eq_t2}  {_fmt_clusters(_clusters(ti), f'tr_{k}>tr_1', f'tr_1>tr_{k}')}"
        )
    return out


def _progression_summary(per_session: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Collapse 14 session comparisons into the few numbers prose can carry.

    Reports how many comparisons reached significance, from which session
    onwards they did so continuously, and the union of the significant cluster
    extents -- i.e. which part of the movement changed.
    """
    sig_sessions, starts, ends = [], [], []
    for k in range(2, 16):
        entry = per_session.get(f"tr_{k}", {})
        clusters = entry.get("clusters_t", [])
        sig = [c for c in clusters if c["p"] < entry.get("alpha", 0.05)]
        if sig:
            sig_sessions.append(k)
            starts.append(min(c["start_pct"] for c in sig))
            ends.append(max(c["end_pct"] for c in sig))

    # First session from which every later session is also significant.
    onset = None
    for k in sig_sessions:
        if all(j in sig_sessions for j in range(k, 16)):
            onset = k
            break
    return {
        f"{prefix}_n_sig": len(sig_sessions),
        f"{prefix}_sig_sessions": sig_sessions,
        f"{prefix}_first_sustained": onset,
        f"{prefix}_cluster_start_min": _r(min(starts), 1) if starts else None,
        f"{prefix}_cluster_start_max": _r(max(starts), 1) if starts else None,
        f"{prefix}_cluster_end_min": _r(min(ends), 1) if ends else None,
        f"{prefix}_cluster_end_max": _r(max(ends), 1) if ends else None,
    }


# --------------------------------------------------------------------------- #
# 3. baseline MVIF equivalence
# --------------------------------------------------------------------------- #
def mvif_baseline(results_path: Path) -> Dict[str, Any]:
    """Welch t-test and Cohen's d on pre-intervention MVIF, three reduction rules.

    Participants completed 1-3 familiarisation sessions, each yielding one MVIF
    value, so a reduction rule is required before a between-group test. All
    three defensible rules are reported; the conclusion has to be shown not to
    depend on the choice, because none of them is obviously correct.

    ``jhamon02`` has no usable passive trial for gravity correction and is
    absent from the file, so NH is n = 12 here while the training analysis has
    n = 13. That must be stated in the manuscript.
    """
    csv = results_path / "mvif_pre_results.csv"
    df = pd.read_csv(csv)
    ses_order = {f"fam_{i}": i for i in range(1, 5)}
    df["ses_num"] = df["ntest"].map(ses_order)

    out: Dict[str, Any] = {
        "mvif_source": csv.name,
        "mvif_n_sessions": int(len(df)),
        "mvif_n_participants": int(df["par"].nunique()),
    }
    for rule in ("max", "mean", "last"):
        if rule == "last":
            red = (
                df.sort_values("ses_num")
                .groupby(["par", "tr_group"], as_index=False)
                .last()[["par", "tr_group", "value"]]
            )
        else:
            red = df.groupby(["par", "tr_group"], as_index=False)["value"].agg(rule)
        nh = red.loc[red["tr_group"] == "NH", "value"].to_numpy(float)
        ik = red.loc[red["tr_group"] == "IK", "value"].to_numpy(float)
        t, p = sps.ttest_ind(nh, ik, equal_var=False)
        out.update(
            {
                f"mvif_{rule}_nh_n": int(nh.size),
                f"mvif_{rule}_nh_mean": _r(nh.mean(), 1),
                f"mvif_{rule}_nh_sd": _r(nh.std(ddof=1), 1),
                f"mvif_{rule}_ik_n": int(ik.size),
                f"mvif_{rule}_ik_mean": _r(ik.mean(), 1),
                f"mvif_{rule}_ik_sd": _r(ik.std(ddof=1), 1),
                f"mvif_{rule}_t": _r(t, 3),
                f"mvif_{rule}_p": _r(p, 3),
                f"mvif_{rule}_d": _r(_cohen_d(nh, ik), 2),
            }
        )
        print(
            f"      MVIF [{rule:>4}] NH {nh.mean():6.1f} +- {nh.std(ddof=1):4.1f} (n={nh.size})  "
            f"IK {ik.mean():6.1f} +- {ik.std(ddof=1):4.1f} (n={ik.size})  "
            f"t={t:5.3f} p={p:.3f} d={_cohen_d(nh, ik):+.2f}"
        )
    return out


# --------------------------------------------------------------------------- #
# 4. work per repetition, repetition counts, post-filter descriptives
# --------------------------------------------------------------------------- #
def work_per_repetition(training_disc: pd.DataFrame) -> Dict[str, Any]:
    """Welch test on mean work per repetition, at the participant level.

    The unit of analysis matters enormously here. A repetition-level test on
    15 883 repetitions treats every repetition of the same person as an
    independent observation and returns an absurd t; the participant is the
    randomised unit, so the test is on 13 vs 14 participant means. Both are
    computed so the difference is on record.
    """
    per_par = (
        training_disc.groupby(["par", "tr_group"], observed=True)["work"].mean().reset_index()
    )
    nh = per_par.loc[per_par["tr_group"] == "NH", "work"].to_numpy(float)
    ik = per_par.loc[per_par["tr_group"] == "IK", "work"].to_numpy(float)
    t, p = sps.ttest_ind(nh, ik, equal_var=False)

    nh_rep = training_disc.loc[training_disc["tr_group"] == "NH", "work"].to_numpy(float)
    ik_rep = training_disc.loc[training_disc["tr_group"] == "IK", "work"].to_numpy(float)
    t_rep, p_rep = sps.ttest_ind(nh_rep, ik_rep, equal_var=False)

    print(
        f"      work/rep participant-level: NH {nh.mean():.1f} +- {nh.std(ddof=1):.1f} (n={nh.size}) "
        f"vs IK {ik.mean():.1f} +- {ik.std(ddof=1):.1f} (n={ik.size}); "
        f"t={t:.2f} p={p:.2e} d={_cohen_d(nh, ik):.2f}"
    )
    return {
        "workrep_nh_n": int(nh.size),
        "workrep_nh_mean": _r(nh.mean(), 1),
        "workrep_nh_sd": _r(nh.std(ddof=1), 1),
        "workrep_ik_n": int(ik.size),
        "workrep_ik_mean": _r(ik.mean(), 1),
        "workrep_ik_sd": _r(ik.std(ddof=1), 1),
        "workrep_diff": _r(nh.mean() - ik.mean(), 1),
        "workrep_t": _r(t, 2),
        "workrep_df": _r(sps.ttest_ind(nh, ik, equal_var=False).df, 1),
        "workrep_p": float(p),
        "workrep_d": _r(_cohen_d(nh, ik), 2),
        "workrep_replevel_t": _r(t_rep, 1),
        "workrep_replevel_p": float(p_rep),
        "workrep_replevel_n": int(nh_rep.size + ik_rep.size),
    }


def repetition_counts(training_disc: pd.DataFrame, filtered: pd.DataFrame) -> Dict[str, Any]:
    """Actual repetitions per set in each group, before and after filtering.

    The published "+3.1 +- 0.5 repetitions" compares IK's *achieved* count with
    NH's *prescribed* count. Since NH completed 92.1 % of the prescription, that
    comparison understates the real gap; the like-for-like actual-versus-actual
    figure is computed here for both datasets.
    """
    out: Dict[str, Any] = {}
    for tag, data in (("all", training_disc), ("analysed", filtered)):
        per_set = (
            data.groupby(["par", "tr_group", "trses", "set"], observed=True)
            .size()
            .reset_index(name="n_reps")
        )
        per_par = (
            per_set.groupby(["par", "tr_group"], observed=True)["n_reps"].mean().reset_index()
        )
        nh = per_par.loc[per_par["tr_group"] == "NH", "n_reps"].to_numpy(float)
        ik = per_par.loc[per_par["tr_group"] == "IK", "n_reps"].to_numpy(float)
        t, p = sps.ttest_ind(ik, nh, equal_var=False)
        out.update(
            {
                f"reps_{tag}_total": int(len(data)),
                f"reps_{tag}_nh": int((data["tr_group"] == "NH").sum()),
                f"reps_{tag}_ik": int((data["tr_group"] == "IK").sum()),
                f"reps_{tag}_nh_sets": int((per_set["tr_group"] == "NH").sum()),
                f"reps_{tag}_ik_sets": int((per_set["tr_group"] == "IK").sum()),
                f"reps_{tag}_perset_nh": _r(nh.mean(), 2),
                f"reps_{tag}_perset_nh_sd": _r(nh.std(ddof=1), 2),
                f"reps_{tag}_perset_ik": _r(ik.mean(), 2),
                f"reps_{tag}_perset_ik_sd": _r(ik.std(ddof=1), 2),
                f"reps_{tag}_perset_diff": _r(ik.mean() - nh.mean(), 2),
                f"reps_{tag}_perset_t": _r(t, 2),
                f"reps_{tag}_perset_p": float(p),
                f"reps_{tag}_perset_d": _r(_cohen_d(ik, nh), 2),
            }
        )
        print(
            f"      reps/set [{tag:>8}] NH {nh.mean():.2f} vs IK {ik.mean():.2f} "
            f"(IK-NH = {ik.mean() - nh.mean():+.2f}, p={p:.2e})"
        )

    # Session 1 vs session 15, actual versus actual, for the "the gap widened" claim.
    for tag, data in (("all", training_disc), ("analysed", filtered)):
        for ses in (1, 15):
            sub = data[data["trses"] == f"tr_{ses}"]
            per_set = (
                sub.groupby(["par", "tr_group", "set"], observed=True)
                .size()
                .reset_index(name="n_reps")
            )
            per_par = (
                per_set.groupby(["par", "tr_group"], observed=True)["n_reps"].mean().reset_index()
            )
            nh = per_par.loc[per_par["tr_group"] == "NH", "n_reps"].to_numpy(float)
            ik = per_par.loc[per_par["tr_group"] == "IK", "n_reps"].to_numpy(float)
            out[f"reps_{tag}_tr{ses}_perset_nh"] = _r(nh.mean(), 2)
            out[f"reps_{tag}_tr{ses}_perset_ik"] = _r(ik.mean(), 2)
            out[f"reps_{tag}_tr{ses}_perset_ik_sd"] = _r(ik.std(ddof=1), 2)
            out[f"reps_{tag}_tr{ses}_perset_diff"] = _r(ik.mean() - nh.mean(), 2)
    return out


def post_filter_descriptives(filtered: pd.DataFrame) -> Dict[str, Any]:
    """Work and velocity per repetition on the analysed (post-filter) subset.

    The published ``nh_work_mean`` / ``nh_vel_mean`` are computed *before*
    ``filter_training_data`` while every model is fitted after it, so the same
    quantity currently has two values in the project. Both are kept, with a
    distinct prefix, so the manuscript can say which one it is quoting.
    """
    out: Dict[str, Any] = {}
    for var, nd in (("work", 1), ("knee_v_mean", 1)):
        g = filtered.groupby("tr_group", observed=True)[var].agg(["mean", "std"])
        for grp in ("NH", "IK"):
            out[f"desc_filt_{grp.lower()}_{var}_mean"] = _r(g.loc[grp, "mean"], nd)
            out[f"desc_filt_{grp.lower()}_{var}_sd"] = _r(g.loc[grp, "std"], nd)
    return out


# --------------------------------------------------------------------------- #
# JSON merge
# --------------------------------------------------------------------------- #
def merge_into_json(flat: Dict[str, Any], json_path: Path) -> None:
    """Read-update-atomic-rewrite, never truncating what is already there."""
    with open(json_path) as f:
        existing = json.load(f)
    if not isinstance(existing, dict):
        raise RuntimeError(f"{json_path} does not hold a JSON object")

    overwritten = sorted(k for k in flat if k in existing and existing[k] != flat[k])
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
    if overwritten:
        print(f"  overwrote {len(overwritten)} existing key(s): {', '.join(overwritten)}")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
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

    from jhamon_training.data import frames
    from jhamon_training.data.utils import dame_manual_exclusions, filter_training_data
    from jhamon_training.saveload import load_obj
    from jhamon_training.stats.analyses_discrete_vars import analyze_discrete_variables

    print("=" * 72)
    print("jHamON -- manuscript statistics not covered by reproduce.py")
    print("=" * 72)
    print(f"      results path {results_path}")

    print("\n[1/6] Loading caches and rebuilding curve frames")
    print("-" * 72)
    _install_legacy_pickle_shim()
    nordict = _pseudonymise(load_obj(name="nht_results.pkl", path=results_path))
    ikdict = _pseudonymise(load_obj(name="ikt_results.pkl", path=results_path))
    nordf = frames.nht_todf(my_dict=nordict)
    ikdf = frames.ikt_todf(my_dict=ikdict)
    training_df = pd.concat([nordf, ikdf])
    print(f"      curve rows: NH {len(nordf):,}  IK {len(ikdf):,}")

    print("\n[2/6] Between-group SPM{t} on torque and velocity curves")
    print("-" * 72)
    flat: Dict[str, Any] = {}
    spm_detail: Dict[str, Any] = {}
    for var, tag in (("torque", "torque"), ("knee_v", "velocity")):
        for by_par, unit in ((False, "shipped"), (True, "paravg")):
            res = spm_between_groups(training_df, var, by_participant=by_par)
            key = f"spm_{tag}_{unit}"
            spm_detail[key] = res
            flat[f"{key}_n_nh"] = res["n_nh"]
            flat[f"{key}_n_ik"] = res["n_ik"]
            flat[f"{key}_tstar"] = res["tstar"]
            flat[f"{key}_n_clusters"] = len(res["clusters"])
            for i, c in enumerate(res["clusters"], start=1):
                flat[f"{key}_c{i}_start"] = c["start_pct"]
                flat[f"{key}_c{i}_end"] = c["end_pct"]
                flat[f"{key}_c{i}_p"] = _r(c["p"], 4)
                flat[f"{key}_c{i}_dir"] = "NH>IK" if c["direction"] > 0 else "IK>NH"
                flat[f"{key}_c{i}_zpeak"] = c["z_peak"]
            print(
                f"      {tag:>8} [{unit:>7}] n={res['n_nh']} vs {res['n_ik']}  "
                f"t*={res['tstar']}  {_fmt_clusters(res['clusters'], 'NH>IK', 'IK>NH')}"
            )

    print("\n[3/6] Within-group progression SPM across the 15 sessions")
    print("-" * 72)
    torque_curves = _select_top_curves(
        dame_manual_exclusions(training_df).pipe(lambda d: d[d["var"] == "torque"])
    )
    alpha_bonf = 0.05 / 14
    for grp in ("NH", "IK"):
        sub = torque_curves[torque_curves["tr_group"] == grp]
        for alpha, tag in ((0.05, "unc"), (alpha_bonf, "bonf")):
            print(f"    -- {grp}, alpha = {alpha:.5f} ({tag})")
            per_ses = spm_progression(sub, alpha=alpha, label=f"{grp}/{tag}")
            spm_detail[f"prog_{grp.lower()}_{tag}"] = per_ses
            flat.update(_progression_summary(per_ses, f"prog_{grp.lower()}_{tag}"))
    flat["prog_alpha_bonferroni"] = round(alpha_bonf, 5)
    flat["prog_n_comparisons"] = 14

    print("\n[4/6] Baseline isometric knee-flexion torque (MVIF)")
    print("-" * 72)
    flat.update(mvif_baseline(results_path))

    print("\n[5/6] Discrete outcomes: work per repetition and repetition counts")
    print("-" * 72)
    with tempfile.TemporaryDirectory(prefix="jhamon_extract_") as tmpdir:
        disc = analyze_discrete_variables(
            nordict=nordict,
            ikdf=ikdf,
            training_df=training_df,
            results_output_path=Path(tmpdir),
        )["training_disc"]
    filtered = filter_training_data(disc.copy())
    flat.update(work_per_repetition(disc))
    flat.update(repetition_counts(disc, filtered))
    flat.update(post_filter_descriptives(filtered))

    print("\n[6/6] Writing")
    print("-" * 72)
    stats_path = Path(args.stats_path) if args.stats_path else results_path / "training_stats.json"
    detail_path = stats_path.parent / "manuscript_stats_detail.json"
    if args.dry_run:
        print(f"      --dry-run: {len(flat)} keys computed, nothing written")
        print(json.dumps(flat, indent=2, default=str))
        return 0

    merge_into_json(flat, stats_path)
    with open(detail_path, "w") as f:
        json.dump(spm_detail, f, indent=2, default=str)
    print(f"wrote {detail_path} (full per-cluster and per-session SPM detail)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
