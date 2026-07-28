#!/usr/bin/env python3
"""The standardisation-validity statistics: agreement in work and in velocity.

Why this exists
---------------
The paper's first aim is the validity of a procedure that equated external
mechanical work per set and mean angular velocity between the Nordic hamstring
(NH) and isokinetic (IK) modalities. ``reproduce.py`` writes one version of that
result (``ba_*``): Bland-Altman on the ANALYSED subset, with isokinetic velocity
averaged over the whole repetition window. Three things are wrong with using
that version alone as the paper's primary validity claim.

1. **The dataset.** The standardisation targeted every repetition the
   participants actually performed, not the subset that survives the analysis
   filters -- one of which (``10 < knee_v_mean < 35``) filters on the very
   outcome being validated. Three nested populations are therefore computed
   here, and the primary one is stated rather than inherited:

       all        every recorded repetition                         15 883
       protocol   after the session/set and participant/set
                  protocol exclusions, but with NO filtering on a
                  measured outcome                                  (computed)
       analysed   after everything, including the velocity window   15 632

2. **The velocity that was prescribed.** The dynamometer was programmed to a
   CONSTANT target velocity per set. Averaging isokinetic velocity over the
   whole window folds in the machine's acceleration and deceleration ramps,
   which nobody prescribed, and drags the isokinetic mean from 23.6 to 20.3
   deg/s. That 3.3 deg/s gap is the entire apparent "velocity standardisation
   bias". ``stats/velocity_phase.py`` computes the plateau velocity instead;
   both are reported here so the contrast is explicit and auditable.

3. **What the confidence interval means.** All 73 session x set differences come
   from the same 13 + 14 people, so a t-interval over 73 cells describes
   cell-to-cell reproducibility, not the uncertainty of a between-group
   difference, and is roughly an order of magnitude too narrow if read as the
   latter. A participant-level contrast is computed alongside every cell-level
   one, and both are labelled.

Two further reporting corrections are made here because they are cheap and a
reviewer will otherwise raise them:

* **Regression-based limits of agreement.** Proportional bias is significant for
  both outcomes, so a single pair of fixed +/-1.96 SD limits misdescribes the
  disagreement at the extremes of the range. The regression-based limits are
  computed so they can be reported alongside.
* **t-based confidence intervals for the mixed-model contrasts.**
  ``mixed_lmm_analysis.py`` builds them on a normal quantile (``+/- 1.96 se``),
  which is 5 % too narrow at 25 between-participant degrees of freedom.

Everything is merged into ``training_stats.json`` under ``velphase_*``,
``bav_*``, ``lmmt_*`` and ``workrep_g`` keys, without touching any existing key.
Run AFTER ``reproduce.py`` and ``extract_manuscript_stats.py``.

Usage:
    source venv/bin/activate && python extract_validity_stats.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from reproduce import _install_legacy_pickle_shim, _pseudonymise

#: Which agreement analyses to run: (dataset tier, velocity column, label).
#: "work" is run on every tier; velocity is run under both definitions so the
#: difference between them is a reported quantity rather than a footnote.
TIERS = ("all", "protocol", "analysed")
VELOCITY_COLUMNS = (("knee_v_mean", "vwin"), ("knee_v_iso", "viso"))


def _r(value: Any, nd: int) -> Any:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return None if not np.isfinite(f) else round(f, nd)


# --------------------------------------------------------------------------- #
# dataset tiers
# --------------------------------------------------------------------------- #
def protocol_subset(data: pd.DataFrame) -> pd.DataFrame:
    """Apply only the protocol exclusions from ``filter_training_data``.

    Steps 1 and 2 of that function remove (session, set) cells and
    (participant, session, set) cells that were not part of the matched
    prescription -- protocol facts, known independently of any measurement.
    Step 3, the ``10 < knee_v_mean < 35`` window, filters on a measured outcome
    and is deliberately NOT applied here, so this tier can validate the velocity
    standardisation without the circularity of having pre-selected on velocity.

    The predicates are copied verbatim from ``data/utils.py:198-241``; they are
    duplicated rather than imported because that function performs all three
    steps in one pass and offers no way to stop after the second.
    """
    session_set = ~(
        ((data["trses"] == "tr_1") & data["set"].isin(["set_4", "set_5"]))
        | ((data["trses"] == "tr_4") & (data["set"] == "set_5"))
        | ((data["trses"] == "tr_5") & (data["set"] == "set_5"))
        | ((data["trses"] == "tr_10") & data["set"].isin(["set_6", "set_7"]))
        | ((data["trses"] == "tr_15") & (data["set"] == "set_7"))
    )
    data = data[session_set]

    participant = ~(
        ((data["par"] == "jhamon02") & (data["trses"] == "tr_9") & (data["set"] == "set_5"))
        | ((data["par"] == "jhamon03") & (data["trses"] == "tr_11") & (data["set"] == "set_4"))
        | (
            (data["par"] == "jhamon03")
            & (data["trses"] == "tr_10")
            & (data["set"] == "set_5")
            & (data["rep"] == "rep_1")
        )
        | ((data["par"] == "jhamon20") & (data["trses"] == "tr_14") & (data["set"] == "set_1"))
        | ((data["par"] == "jhamon20") & (data["trses"] == "tr_13") & (data["set"] == "set_5"))
        | ((data["par"] == "jhamon33") & (data["trses"] == "tr_15") & (data["set"] == "set_1"))
    )
    return data[participant].copy()


# --------------------------------------------------------------------------- #
# agreement, cell level and participant level
# --------------------------------------------------------------------------- #
def regression_limits(mean: np.ndarray, diff: np.ndarray) -> Dict[str, float]:
    """Bland & Altman's regression-based limits for a proportional bias.

    When the difference varies systematically with the magnitude, a single pair
    of fixed limits is wrong at both ends of the range. The regression form
    models the bias as a line in the mean and puts the limits a constant
    distance either side of it (valid when the residuals are homoscedastic,
    which is checked here and reported).

        bias(M)  = a + b*M
        LoA(M)   = bias(M) +/- 1.96 * sqrt(pi/2) * mean|residual|

    Returns the intercept, slope, the half-width of the limits, and a
    Breusch-Pagan-style check on the residuals (regression of |residual| on the
    mean) so the constant-width assumption is not merely assumed.
    """
    mean = np.asarray(mean, dtype=float)
    diff = np.asarray(diff, dtype=float)
    fit = sps.linregress(mean, diff)
    resid = diff - (fit.intercept + fit.slope * mean)
    half = 1.96 * np.sqrt(np.pi / 2.0) * float(np.mean(np.abs(resid)))
    het = sps.linregress(mean, np.abs(resid))
    return {
        "reg_intercept": float(fit.intercept),
        "reg_slope": float(fit.slope),
        "reg_slope_p": float(fit.pvalue),
        "reg_half_width": half,
        "reg_resid_het_p": float(het.pvalue),
    }


def participant_level_contrast(
    data: pd.DataFrame, var: str, per_set: bool
) -> Dict[str, float]:
    """NH - IK difference with the PARTICIPANT as the unit of analysis.

    The Bland-Altman pairing is by prescription cell, which is the right unit
    for asking "how reproducibly did the standardisation hit its target". It is
    the wrong unit for a confidence interval on the between-group difference,
    because all 73 cell differences are built from the same 27 people. This
    computes the honest between-group contrast: one number per participant,
    then Welch.

    Args:
        data: per-repetition frame.
        var: outcome column.
        per_set: if True, sum the outcome within each (session, set) for the
            participant and then average those set totals -- the "per set"
            quantity that work is standardised on. If False, average over
            repetitions -- the "per repetition" quantity velocity is
            standardised on.
    """
    if per_set:
        per_participant = (
            data.groupby(["par", "tr_group", "trses", "set"], observed=True)[var]
            .sum()
            .groupby(["par", "tr_group"], observed=True)
            .mean()
            .reset_index()
        )
    else:
        per_participant = (
            data.groupby(["par", "tr_group"], observed=True)[var].mean().reset_index()
        )

    nh = per_participant.loc[per_participant["tr_group"] == "NH", var].to_numpy(float)
    ik = per_participant.loc[per_participant["tr_group"] == "IK", var].to_numpy(float)
    res = sps.ttest_ind(nh, ik, equal_var=False)
    ci = res.confidence_interval()
    pooled = np.sqrt(
        ((nh.size - 1) * np.var(nh, ddof=1) + (ik.size - 1) * np.var(ik, ddof=1))
        / (nh.size + ik.size - 2)
    )
    d = float((nh.mean() - ik.mean()) / pooled) if pooled > 0 else float("nan")
    j = 1.0 - 3.0 / (4.0 * (nh.size + ik.size) - 9.0)  # Hedges' small-sample correction
    return {
        "par_nh_n": int(nh.size),
        "par_nh_mean": float(nh.mean()),
        "par_nh_sd": float(nh.std(ddof=1)),
        "par_ik_n": int(ik.size),
        "par_ik_mean": float(ik.mean()),
        "par_ik_sd": float(ik.std(ddof=1)),
        "par_bias": float(nh.mean() - ik.mean()),
        "par_ci_low": float(ci.low),
        "par_ci_high": float(ci.high),
        "par_t": float(res.statistic),
        "par_df": float(res.df),
        "par_p": float(res.pvalue),
        "par_g": d * j,
    }


def agreement_block(
    data: pd.DataFrame, var: str, rep_aggregation: str, per_set: bool, prefix: str
) -> Dict[str, Any]:
    """Cell-level Bland-Altman + regression limits + participant-level contrast."""
    from jhamon_training.stats.agreement import compute_agreement

    res = compute_agreement(data, var, rep_aggregation=rep_aggregation)
    pairs = res["pairs"]
    reg = regression_limits(pairs["mean"].to_numpy(), pairs["diff"].to_numpy())
    par = participant_level_contrast(data, var, per_set=per_set)

    nd = 1 if rep_aggregation == "sum" else 2
    flat = {
        f"{prefix}_n_pairs": int(res["n_pairs"]),
        f"{prefix}_nh_mean": _r(res["test_mean"], nd),
        f"{prefix}_nh_sd": _r(res["test_sd"], nd),
        f"{prefix}_ik_mean": _r(res["reference_mean"], nd),
        f"{prefix}_ik_sd": _r(res["reference_sd"], nd),
        f"{prefix}_grand_mean": _r(res["grand_mean"], nd),
        f"{prefix}_bias": _r(res["bias"], nd),
        f"{prefix}_bias_ci_low": _r(res["bias_ci_low"], nd),
        f"{prefix}_bias_ci_high": _r(res["bias_ci_high"], nd),
        f"{prefix}_bias_p": _r(res["bias_p"], 4),
        f"{prefix}_pct_bias": _r(res["pct_bias"], 2),
        f"{prefix}_sd_diff": _r(res["sd_diff"], nd),
        f"{prefix}_loa_low": _r(res["loa_low"], nd),
        f"{prefix}_loa_high": _r(res["loa_high"], nd),
        f"{prefix}_loa_low_ci_low": _r(res["loa_low_ci_low"], nd),
        f"{prefix}_loa_low_ci_high": _r(res["loa_low_ci_high"], nd),
        f"{prefix}_loa_high_ci_low": _r(res["loa_high_ci_low"], nd),
        f"{prefix}_loa_high_ci_high": _r(res["loa_high_ci_high"], nd),
        f"{prefix}_pct_loa_low": _r(res["pct_loa_low"], 2),
        f"{prefix}_pct_loa_high": _r(res["pct_loa_high"], 2),
        f"{prefix}_ccc": _r(res["ccc"], 3),
        f"{prefix}_ccc_ci_low": _r(res["ccc_ci_low"], 3),
        f"{prefix}_ccc_ci_high": _r(res["ccc_ci_high"], 3),
        f"{prefix}_pearson_r": _r(res["pearson_r"], 3),
        f"{prefix}_prop_slope": _r(res["prop_bias_slope"], 4),
        f"{prefix}_prop_p": _r(res["prop_bias_p"], 4),
        f"{prefix}_shapiro_p": _r(res["shapiro_p"], 4),
        f"{prefix}_reg_intercept": _r(reg["reg_intercept"], nd),
        f"{prefix}_reg_slope": _r(reg["reg_slope"], 4),
        f"{prefix}_reg_half_width": _r(reg["reg_half_width"], nd),
        f"{prefix}_reg_resid_het_p": _r(reg["reg_resid_het_p"], 3),
    }
    for key, value in par.items():
        nd_key = 0 if key.endswith("_n") else (4 if key.endswith("_p") else nd)
        flat[f"{prefix}_{key}"] = int(value) if key.endswith("_n") else _r(value, nd_key)

    print(
        f"      {prefix:<24} n={res['n_pairs']:>3} pairs | "
        f"NH {res['test_mean']:>8.2f} vs IK {res['reference_mean']:>8.2f} | "
        f"bias {res['bias']:+7.2f} ({res['pct_bias']:+6.2f}%) | "
        f"LoA [{res['loa_low']:>8.2f},{res['loa_high']:>8.2f}] | CCC {res['ccc']:.3f} | "
        f"participant-level {par['par_bias']:+7.2f} "
        f"[{par['par_ci_low']:+.2f},{par['par_ci_high']:+.2f}] p={par['par_p']:.3f}"
    )
    return flat


# --------------------------------------------------------------------------- #
# the velocity mixed model, refitted on the like-for-like quantity
# --------------------------------------------------------------------------- #
def lmm_isokinetic_velocity(filtered: pd.DataFrame) -> Dict[str, Any]:
    """Refit the velocity mixed model on the isokinetic-phase velocity.

    The published ``knee_v_mean`` model inherits exactly the artefact that
    ``velocity_phase`` removes: the isokinetic value is dragged down by the
    dynamometer's ramps, so the model reports a group difference (NH > IK,
    +1.76 deg/s) that contradicts the paper's own claim that velocity was
    equated. That contradiction is an artefact of the averaging window, not a
    finding, and the audit showed the +1.76 estimate is the one conclusion in
    the paper that does not survive any change of analysis.

    This refits the identical model -- same cells, same top-3 selection, same
    sum-to-zero coding, same random intercept -- on ``knee_v_iso``, which is the
    velocity each group actually trained at. Reported alongside the original so
    the change is visible rather than substituted silently.
    """
    import patsy
    import statsmodels.formula.api as smf

    cell = (
        filtered.groupby(["par", "trses_num"], group_keys=False)
        .apply(lambda x: x.nlargest(3, "peak_torque"))
        .reset_index(drop=True)
        .groupby(["par", "tr_group", "trses_num"])[["knee_v_iso"]]
        .mean()
        .reset_index()
    )
    model = smf.mixedlm(
        "knee_v_iso ~ C(tr_group, Sum) * C(trses_num, Sum)", data=cell, groups=cell["par"]
    )
    res = model.fit(reml=True, method="lbfgs", maxiter=200)

    def omnibus(keep) -> Tuple[float, int, float]:
        names = res.model.exog_names
        cols = [i for i, n in enumerate(names) if keep(n)]
        L = np.zeros((len(cols), len(res.params)))
        for r, c in enumerate(cols):
            L[r, c] = 1.0
        wt = res.wald_test(L, scalar=True)
        return float(np.ravel(wt.statistic)[0]), len(cols), float(np.ravel(wt.pvalue)[0])

    sessions = sorted(cell["trses_num"].unique())
    combos = pd.DataFrame(
        [(g, s) for g in ["NH", "IK"] for s in sessions], columns=["tr_group", "trses_num"]
    )
    X = patsy.dmatrix(res.model.data.design_info, combos, return_type="dataframe")
    fe = res.fe_params
    X = X[fe.index]
    is_nh = combos["tr_group"].values == "NH"
    contrast = X[is_nh].mean(axis=0).values - X[~is_nh].mean(axis=0).values
    est = float(contrast @ fe.values)
    se = float(np.sqrt(contrast @ res.cov_params().loc[fe.index, fe.index].values @ contrast))
    df = int(cell["par"].nunique() - 2)
    t_crit = float(sps.t.ppf(0.975, df))
    desc = cell.groupby("tr_group")["knee_v_iso"].agg(["mean", "std"])

    grp, grp_df, grp_p = omnibus(lambda n: "C(tr_group, Sum)" in n and ":" not in n)
    tim, tim_df, tim_p = omnibus(lambda n: "C(trses_num, Sum)" in n and ":" not in n)
    itr, itr_df, itr_p = omnibus(lambda n: ":" in n)

    print(
        f"      knee_v_iso: NH {desc.loc['NH', 'mean']:.1f} +- {desc.loc['NH', 'std']:.1f} vs "
        f"IK {desc.loc['IK', 'mean']:.1f} +- {desc.loc['IK', 'std']:.1f}; "
        f"NH-IK {est:+.2f} [{est - t_crit * se:+.2f}, {est + t_crit * se:+.2f}] "
        f"chi2({grp_df})={grp:.1f} p={grp_p:.3f}"
    )
    return {
        "lmmiso_knee_v_desc_nh_mean": _r(desc.loc["NH", "mean"], 1),
        "lmmiso_knee_v_desc_nh_sd": _r(desc.loc["NH", "std"], 1),
        "lmmiso_knee_v_desc_ik_mean": _r(desc.loc["IK", "mean"], 1),
        "lmmiso_knee_v_desc_ik_sd": _r(desc.loc["IK", "std"], 1),
        "lmmiso_knee_v_nhmik_est": _r(est, 2),
        "lmmiso_knee_v_ci_low": _r(est - t_crit * se, 2),
        "lmmiso_knee_v_ci_high": _r(est + t_crit * se, 2),
        "lmmiso_knee_v_group_chi2": _r(grp, 1),
        "lmmiso_knee_v_group_df": grp_df,
        "lmmiso_knee_v_group_p": float(grp_p),
        "lmmiso_knee_v_time_chi2": _r(tim, 1),
        "lmmiso_knee_v_time_df": tim_df,
        "lmmiso_knee_v_time_p": float(tim_p),
        "lmmiso_knee_v_interaction_chi2": _r(itr, 1),
        "lmmiso_knee_v_interaction_df": itr_df,
        "lmmiso_knee_v_interaction_p": float(itr_p),
    }


# --------------------------------------------------------------------------- #
# t-based confidence intervals for the mixed-model contrasts
# --------------------------------------------------------------------------- #
def lmm_t_intervals(stats: Dict[str, Any], n_participants: int) -> Dict[str, Any]:
    """Rebuild the NH-minus-IK 95 % CIs on a t quantile instead of 1.96.

    ``mixed_lmm_analysis._nh_minus_ik_emm`` hard-codes ``est +/- 1.96*se``. The
    between-group contrast carries roughly ``n_participants - 2`` degrees of
    freedom, not infinity, so the stored intervals are about 5 % too narrow. The
    standard error is recovered from the stored interval (``se = (hi-lo)/3.92``)
    rather than refitting, which keeps this a pure re-expression of the
    published estimate and cannot introduce a new number.
    """
    df = n_participants - 2
    t_crit = float(sps.t.ppf(0.975, df))
    out: Dict[str, Any] = {"lmmt_df": df, "lmmt_tcrit": round(t_crit, 4)}
    for dv in (
        "mean_torque",
        "peak_torque",
        "knee_ROM",
        "angle_at_peak_torque",
        "knee_v_mean",
    ):
        est = stats[f"lmm_{dv}_nhmik_est"]
        lo, hi = stats[f"lmm_{dv}_nhmik_ci_low"], stats[f"lmm_{dv}_nhmik_ci_high"]
        se = (hi - lo) / (2 * 1.96)
        out[f"lmmt_{dv}_se"] = _r(se, 2)
        out[f"lmmt_{dv}_ci_low"] = _r(est - t_crit * se, 2)
        out[f"lmmt_{dv}_ci_high"] = _r(est + t_crit * se, 2)
        out[f"lmmt_{dv}_t"] = _r(est / se, 2)
        out[f"lmmt_{dv}_p"] = float(2 * sps.t.sf(abs(est / se), df))
        print(
            f"      {dv:<22} {est:+8.2f}  normal [{lo:+7.2f},{hi:+7.2f}]  "
            f"t({df}) [{est - t_crit * se:+7.2f},{est + t_crit * se:+7.2f}]  "
            f"p={2 * sps.t.sf(abs(est / se), df):.4f}"
        )
    return out


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def merge_into_json(flat: Dict[str, Any], json_path: Path) -> None:
    with open(json_path) as f:
        existing = json.load(f)
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
        print(f"  overwrote: {', '.join(overwritten)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--results-path", default=None)
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
    from jhamon_training.data.utils import filter_training_data
    from jhamon_training.saveload import load_obj
    from jhamon_training.stats.analyses_discrete_vars import analyze_discrete_variables
    from jhamon_training.stats.velocity_phase import (
        attach_isokinetic_velocity,
        describe_phases,
        isokinetic_phase_velocity,
    )

    print("=" * 100)
    print("jHamON -- standardisation validity: agreement in work and in angular velocity")
    print("=" * 100)

    print("\n[1/6] Loading caches")
    print("-" * 100)
    _install_legacy_pickle_shim()
    nordict = _pseudonymise(load_obj(name="nht_results.pkl", path=results_path))
    ikdict = _pseudonymise(load_obj(name="ikt_results.pkl", path=results_path))
    nordf = frames.nht_todf(my_dict=nordict)
    ikdf = frames.ikt_todf(my_dict=ikdict)

    print("\n[2/6] Isokinetic velocity: plateau vs whole window")
    print("-" * 100)
    iso = isokinetic_phase_velocity(ikdf)
    flat: Dict[str, Any] = describe_phases(iso)
    for key, value in flat.items():
        print(f"      {key:<36}{value}")

    print("\n[3/6] Building the three dataset tiers")
    print("-" * 100)
    with tempfile.TemporaryDirectory(prefix="jhamon_validity_") as tmpdir:
        disc = analyze_discrete_variables(
            nordict=nordict,
            ikdf=ikdf,
            training_df=pd.concat([nordf, ikdf]),
            results_output_path=Path(tmpdir),
        )["training_disc"]
    disc = attach_isokinetic_velocity(disc, ikdf)

    datasets = {
        "all": disc,
        "protocol": protocol_subset(disc),
        "analysed": filter_training_data(protocol_subset(disc).copy()),
    }
    for tier, data in datasets.items():
        counts = data.groupby("tr_group", observed=True).size().to_dict()
        flat[f"bav_{tier}_n_reps"] = int(len(data))
        flat[f"bav_{tier}_n_nh"] = int(counts.get("NH", 0))
        flat[f"bav_{tier}_n_ik"] = int(counts.get("IK", 0))
        print(
            f"      {tier:<10} {len(data):>6,} repetitions "
            f"(NH {counts.get('NH', 0):,} / IK {counts.get('IK', 0):,})"
        )

    print("\n[4/6] Agreement analyses")
    print("-" * 100)
    for tier, data in datasets.items():
        flat.update(
            agreement_block(data, "work", "sum", per_set=True, prefix=f"bav_{tier}_work")
        )
        for column, tag in VELOCITY_COLUMNS:
            flat.update(
                agreement_block(
                    data, column, "pooled_mean", per_set=False, prefix=f"bav_{tier}_{tag}"
                )
            )

    print("\n[5/6] Velocity mixed model on the isokinetic-phase velocity")
    print("-" * 100)
    flat.update(lmm_isokinetic_velocity(datasets["analysed"]))

    print("\n[6/6] t-based confidence intervals for the mixed-model contrasts")
    print("-" * 100)
    stats_path = Path(args.stats_path) if args.stats_path else results_path / "training_stats.json"
    with open(stats_path) as f:
        stored = json.load(f)
    flat.update(lmm_t_intervals(stored, int(stored["lmm_n_participants"])))

    # Hedges' g for the work-per-repetition comparison (the stored d is uncorrected).
    n_nh, n_ik = int(stored["workrep_nh_n"]), int(stored["workrep_ik_n"])
    j = 1.0 - 3.0 / (4.0 * (n_nh + n_ik) - 9.0)
    flat["workrep_g"] = _r(stored["workrep_d"] * j, 2)

    if args.dry_run:
        print(f"\n--dry-run: {len(flat)} keys computed, nothing written")
        return 0
    merge_into_json(flat, stats_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
