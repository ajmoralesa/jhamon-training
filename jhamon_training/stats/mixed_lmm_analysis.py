"""Linear mixed-model analysis of the discrete training outcomes.

Why this exists
---------------
The two-way mixed ANOVA (``mixed_anova_analysis.run_mixed_anova``) requires a
fully balanced design and so listwise-drops every participant with even one
missing (participant x session) cell. That collapsed the ANOVA to 6 NH + 12 IK
participants: it discarded 143 good cell observations to accommodate 12 missing
ones, and took more than half the NH group with it.

This module replaces that with a linear mixed model (random intercept per
participant) that uses every available (participant x session) cell, so all
13 NH + 14 IK participants contribute. It was verified to reproduce the ANOVA
conclusions with the full sample (peak torque, angle-at-peak and velocity all
NH > IK; mean torque and ROM n.s.).

Missing data -- what these 12 cells actually are (corrected 2026-07-27)
----------------------------------------------------------------------
NOT absences. All 27 participants completed all 15 training sessions. In 12 of
the 405 (participant x session) combinations the session was trained but the
mechanical data were not captured or retained -- a technical failure during
data acquisition.

An earlier version of this docstring called them "genuine absences", which is
wrong and had begun to propagate into the manuscript. In prose use "technical
failure during data acquisition"; never "absence", "missed session", "dropout"
or "attrition", all of which describe participants rather than instruments.
Incomplete measurement is a much weaker threat to validity than incomplete
participation, and the Methods wording depends on the distinction.

Affected: 7 NH (jhamon02, 03, 04, 06, 09, 10, 12) and 2 IK (jhamon20, 32).
Losses are concentrated in NH -- 10 of 195 NH cells vs 2 of 210 IK cells
(Fisher exact p = 0.017) -- consistent with the heavier NH instrumentation
(external force transducer + kinematics vs one integrated dynamometer).

Verified 2026-07-27 by rebuilding the (par x session) grid from the 2023
caches: the count is 393 cells at every stage from the raw dictionary keys
onward, so all 12 were already missing before any filtering.
``filter_training_data`` removes 251 repetitions but empties no cell, so
missingness is not a function of the recorded signal. This supports treating
the data as missing at random (conditional on group, which is a fixed effect).

Design (agreed 2026-06-05)
--------------------------
- Outcome aggregation: the three highest-peak-torque reps per (participant,
  session) are averaged to one value per cell (matches the Methods wording).
- Fixed effects: training group x training session, with SESSION AS A
  15-LEVEL CATEGORICAL factor (assumption-free, like the original ANOVA).
- Random effects: random intercept per participant.
- Coding: sum-to-zero (deviation) contrasts, so the group and time Wald tests
  are genuine *main effects* averaged over the other factor's levels.
- Reporting: omnibus Wald chi-square (group / time / group x time) + the
  NH - IK estimated marginal difference with 95% CI (estimates+CI style).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf

from jhamon_training.pathutils import RESULTS_TRAINING_PATH

DVS: List[str] = [
    "mean_torque",
    "peak_torque",
    "knee_ROM",
    "angle_at_peak_torque",
    "knee_v_mean",
]


def _cell_means(training_disc: pd.DataFrame) -> pd.DataFrame:
    """Top-3 reps (by peak torque) per (par, session) -> one mean row per cell.

    ``training_disc`` must already be filtered (``filter_training_data``) so it
    carries the numeric ``trses_num`` column and the velocity QC filter.
    """
    df3 = (
        training_disc.groupby(["par", "trses_num"], group_keys=False)
        .apply(lambda x: x.nlargest(3, "peak_torque"))
        .reset_index(drop=True)
    )
    cell = (
        df3.groupby(["par", "tr_group", "trses_num"])[DVS].mean().reset_index()
    )
    return cell


def _omnibus_wald(res, keep) -> Dict[str, float]:
    """Joint Wald test of all fixed-effect columns selected by ``keep(name)``."""
    names = res.model.exog_names  # fixed effects = first block of res.params
    n_total = len(res.params)  # fixed effects + variance components
    cols = [i for i, n in enumerate(names) if keep(n)]
    L = np.zeros((len(cols), n_total))
    for r, c in enumerate(cols):
        L[r, c] = 1.0
    wt = res.wald_test(L, scalar=True)
    return {
        "chi2": round(float(np.ravel(wt.statistic)[0]), 2),
        "df": int(L.shape[0]),
        "p": float(np.ravel(wt.pvalue)[0]),
    }


def _nh_minus_ik_emm(res, cell: pd.DataFrame) -> Dict[str, float]:
    """NH - IK estimated marginal difference (averaged equally over sessions).

    Built as a linear contrast c'b on the fixed effects, where c is the mean
    design row for NH minus the mean design row for IK over all 15 sessions.
    """
    sessions = sorted(cell["trses_num"].unique())
    combos = pd.DataFrame(
        [(g, s) for g in ["NH", "IK"] for s in sessions],
        columns=["tr_group", "trses_num"],
    )
    X = patsy.dmatrix(
        res.model.data.design_info, combos, return_type="dataframe"
    )
    fe = res.fe_params
    X = X[fe.index]  # align column order with fixed-effect params
    c = X[combos["tr_group"].values == "NH"].mean(axis=0).values - X[
        combos["tr_group"].values == "IK"
    ].mean(axis=0).values
    cov = res.cov_params().loc[fe.index, fe.index].values
    beta = fe.values
    est = float(c @ beta)
    se = float(np.sqrt(c @ cov @ c))
    z = est / se
    # marginal (model-estimated) group means averaged over sessions
    emm_nh = float(X[combos["tr_group"].values == "NH"].mean(axis=0).values @ beta)
    emm_ik = float(X[combos["tr_group"].values == "IK"].mean(axis=0).values @ beta)
    from scipy import stats as _st

    return {
        "estimate": round(est, 2),
        "se": round(se, 2),
        "ci_low": round(est - 1.96 * se, 2),
        "ci_high": round(est + 1.96 * se, 2),
        "z": round(z, 2),
        "p": float(2 * _st.norm.sf(abs(z))),
        "emm_nh": round(emm_nh, 1),
        "emm_ik": round(emm_ik, 1),
    }


def run_mixed_lmm(
    training_disc: pd.DataFrame,
    variables: Optional[List[str]] = None,
    json_path: Optional[Path] = None,
    write_json: bool = False,
) -> Dict[str, Any]:
    """Fit the per-outcome LMM and (optionally) append results to a JSON file.

    Args:
        training_disc: filtered discrete data (has ``trses_num``).
        variables: outcomes to model (defaults to the five discrete outcomes).
        json_path: where to read/append the stats JSON (defaults to the NAS
            ``training_stats.json``). Only used when ``write_json`` is True.
        write_json: if True, merge the LMM keys into the JSON file.

    Returns:
        Nested dict of results per outcome.
    """
    variables = variables or DVS
    cell = _cell_means(training_disc)
    n_par = cell["par"].nunique()
    n_nh = cell[cell["tr_group"] == "NH"]["par"].nunique()
    n_ik = cell[cell["tr_group"] == "IK"]["par"].nunique()
    print(
        f"LMM data: {n_par} participants (NH {n_nh}, IK {n_ik}); "
        f"{len(cell)} of {n_par * 15} possible (par x session) cells."
    )

    results: Dict[str, Any] = {}
    flat: Dict[str, Any] = {
        "lmm_n_participants": int(n_par),
        "lmm_n_nh": int(n_nh),
        "lmm_n_ik": int(n_ik),
        "lmm_n_cells": int(len(cell)),
    }

    for dv in variables:
        m = smf.mixedlm(
            f"{dv} ~ C(tr_group, Sum) * C(trses_num, Sum)",
            data=cell,
            groups=cell["par"],
        )
        res = m.fit(reml=True, method="lbfgs", maxiter=200)

        is_grp = lambda n: ("C(tr_group, Sum)" in n) and (":" not in n)
        is_time = lambda n: ("C(trses_num, Sum)" in n) and (":" not in n)
        is_int = lambda n: ":" in n

        group = _omnibus_wald(res, is_grp)
        time = _omnibus_wald(res, is_time)
        inter = _omnibus_wald(res, is_int)
        contrast = _nh_minus_ik_emm(res, cell)

        # observed descriptive group means (for continuity with the prose)
        gm = cell.groupby("tr_group")[dv].agg(["mean", "std"])
        desc = {
            "NH_mean": round(float(gm.loc["NH", "mean"]), 1),
            "NH_sd": round(float(gm.loc["NH", "std"]), 1),
            "IK_mean": round(float(gm.loc["IK", "mean"]), 1),
            "IK_sd": round(float(gm.loc["IK", "std"]), 1),
        }

        results[dv] = {
            "group": group,
            "time": time,
            "interaction": inter,
            "nh_minus_ik": contrast,
            "descriptive": desc,
        }

        # flatten for JSON / qmd
        flat[f"lmm_{dv}_group_chi2"] = group["chi2"]
        flat[f"lmm_{dv}_group_df"] = group["df"]
        flat[f"lmm_{dv}_group_p"] = round(group["p"], 3)
        flat[f"lmm_{dv}_time_chi2"] = time["chi2"]
        flat[f"lmm_{dv}_time_df"] = time["df"]
        flat[f"lmm_{dv}_time_p"] = round(time["p"], 3)
        flat[f"lmm_{dv}_interaction_chi2"] = inter["chi2"]
        flat[f"lmm_{dv}_interaction_df"] = inter["df"]
        flat[f"lmm_{dv}_interaction_p"] = round(inter["p"], 3)
        flat[f"lmm_{dv}_nhmik_est"] = contrast["estimate"]
        flat[f"lmm_{dv}_nhmik_ci_low"] = contrast["ci_low"]
        flat[f"lmm_{dv}_nhmik_ci_high"] = contrast["ci_high"]
        flat[f"lmm_{dv}_nhmik_p"] = round(contrast["p"], 3)
        flat[f"lmm_{dv}_emm_nh"] = contrast["emm_nh"]
        flat[f"lmm_{dv}_emm_ik"] = contrast["emm_ik"]
        flat[f"lmm_{dv}_desc_nh_mean"] = desc["NH_mean"]
        flat[f"lmm_{dv}_desc_nh_sd"] = desc["NH_sd"]
        flat[f"lmm_{dv}_desc_ik_mean"] = desc["IK_mean"]
        flat[f"lmm_{dv}_desc_ik_sd"] = desc["IK_sd"]

    results["_flat"] = flat

    if write_json:
        json_path = Path(json_path or (RESULTS_TRAINING_PATH / "training_stats.json"))
        try:
            with open(json_path) as f:
                existing = json.load(f)
        except FileNotFoundError:
            existing = {}
        existing.update(flat)
        with open(json_path, "w") as f:
            json.dump(existing, f, indent=4)
        print(f"LMM stats written to {json_path}")

    return results
