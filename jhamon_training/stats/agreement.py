"""Bland-Altman agreement analysis of the work-/velocity-standardisation.

Why this exists
---------------
The Methods of the manuscript state that "the validity of the work- and
velocity-standardisation procedure was evaluated using Bland-Altman analyses
comparing the NH and IK groups for (i) total mechanical work per set and (ii)
mean angular velocity per set", yet ``training_stats.json`` contains no
agreement statistics at all: the two functions that draw the Bland-Altman
panels (``plot/plot_work_comparison.py``, ``plot/plot_knee_velocity_comparison.py``)
compute the bias and limits of agreement *inside the plotting code*, print them
onto the figure, and throw them away. Nothing is persisted, so the numbers
cannot be cited in the text, cannot be checked, and cannot be regenerated.

This module recomputes exactly the same quantities from exactly the same data
preparation as those two plotting functions, adds the inferential pieces a
Bland-Altman analysis is normally expected to report (confidence intervals for
the bias and for both limits of agreement, a test for proportional bias, Lin's
concordance correlation coefficient, and the bias expressed as a percentage),
and merges them into ``training_stats.json`` under ``ba_*`` keys.

WHAT IS BEING PAIRED, AND WHY (read this before quoting any number)
-------------------------------------------------------------------
NH and IK are BETWEEN-SUBJECT groups: 13 people did the Nordic hamstring
exercise, 14 different people did isokinetic eccentric knee flexion. There is
therefore NO participant-to-participant correspondence, and a classical paired
Bland-Altman analysis on raw participant data would be meaningless — any
"pairing" of an NH participant with an IK participant would be arbitrary, and
the resulting limits of agreement would be an artefact of the sort order.

What *is* paired is the PRESCRIPTION. The standardisation was applied cell by
cell: the target work for set k of session j in IK participant i was

    W_target(i, j, k) = mean_NH_work(j, k) * MVIF_i / mean_MVIF_NH

(Methods, Eq. 1, adapted from Guilhem et al. 2012), and the dynamometer
velocity for that same set was programmed to the NH group's mean velocity for
that cell. The NH group mean of a (session x set) cell is thus the *reference*
value that the IK group was instructed to reproduce, and the IK group mean of
that same cell is the *achieved* value.

=> THE UNIT OF ANALYSIS IS THE (session x set) CELL, NOT THE PARTICIPANT.
   Each Bland-Altman point is one pair of GROUP MEANS: the NH group mean for
   cell (j, k) against the IK group mean for the same cell. With the standard
   session/set exclusions applied this gives 73 complete pairs from 15
   sessions. This is precisely what the two published figures already plot, so
   the numbers written here reproduce the figure annotations.

Consequences that MUST be stated wherever these numbers are reported:

1. These limits of agreement describe how reproducibly the standardisation
   worked AT THE GROUP LEVEL, session-set by session-set. They are NOT the
   agreement an individual participant would experience. Between-participant
   variability is averaged out before the difference is taken, so these limits
   are much narrower than participant-level limits would be and must never be
   presented as individual-level agreement.
2. The bias is a group-mean contrast. Because the IK targets were individually
   scaled by MVIF_i / mean_MVIF_NH, the IK group's *expected* mean target equals
   the NH cell mean multiplied by mean_MVIF_IK / mean_MVIF_NH. The group-level
   comparison is therefore only clean to the extent that the two groups' mean
   MVIF are equal (the Methods report a baseline between-group MVIF test for
   exactly this reason). Any residual group difference in mean MVIF maps
   directly into an apparent proportional bias here.
3. The 73 differences are not independent replicates: cells from the same
   session share participants and share a prescription. The limits of agreement
   and their confidence intervals use the classical Bland & Altman (1986, 1999)
   formulae, which assume independence, so the confidence intervals are
   somewhat optimistic. A session-level sensitivity analysis (the 73 cell pairs
   collapsed to 15 session pairs by averaging within session, which removes the
   within-session clustering) is reported alongside under ``ba_*_ses_*`` keys.
   The Methods sentence "computed across the 15 training sessions" is ambiguous
   between these two units; both are therefore reported, and the primary
   (cell-level, n = 73) analysis is the one the figures show.
4. Part of the spread of the differences is pure sampling error in the cell
   means themselves (each is a mean of only 13 or 14 people), not real
   mismatch. The limits of agreement are therefore an UPPER bound on the
   systematic-mismatch component of the standardisation error.

What this module deliberately does NOT do
-----------------------------------------
The ideal analysis would be a within-participant "achieved vs prescribed"
Bland-Altman for each IK participant and set, using the individually scaled
target W_target(i, j, k) as the reference. That is not attempted here because
the per-participant prescribed targets are not stored in the cached results:
reconstructing them would require re-deriving each participant's MVIF from the
familiarisation trials with a selection rule that is not documented anywhere in
the code, and a fabricated reference is worse than an honestly-labelled
group-level one. If those targets are ever recovered, ``compute_agreement`` can
be pointed at them unchanged, since it only needs two paired columns.

No Bayesian methods and no bootstrapping are used. Everything is the closed-form
Bland & Altman (1986; 1999) treatment plus Lin (1989) for the concordance
correlation coefficient.

Data preparation (copied from the plotting code so the numbers match)
---------------------------------------------------------------------
- Input is ``training_disc`` AFTER ``filter_training_data`` (this is what
  ``jhamon_training.py`` passes to both plotting functions).
- WORK, per ``plot_work_comparison.py``: work is SUMMED across repetitions
  within each (session, set, group, participant) — i.e. total work per set for
  that participant — and then AVERAGED across the participants of the group.
  This is participant-weighted.
- MEAN KNEE VELOCITY, per ``plot_knee_velocity_comparison.py``: a flat
  ``pivot_table`` mean of the per-repetition ``knee_v_mean`` over every
  repetition of every participant in the cell. This is REPETITION-weighted, so
  participants who performed more repetitions in a set count more. That is a
  minor inconsistency with the work preparation, but it is what the published
  figure shows, so it is the default here; the participant-weighted value is
  also computed and written as ``ba_knee_v_mean_pw_bias`` /
  ``ba_knee_v_mean_pw_sd_diff`` so the discrepancy is visible rather than
  hidden.
- Sign convention: difference = NH - IK, as in both figures. NH is the
  reference (it defines the target), IK is the achieved value, so a POSITIVE
  bias means IK delivered less than the NH-derived target.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from jhamon_training.pathutils import RESULTS_TRAINING_PATH

# Repetition-aggregation modes accepted by ``compute_agreement``.
#   "sum"          per participant, SUM the repetitions inside the cell, then
#                  average across participants  -> total work per set
#   "mean"         per participant, MEAN the repetitions inside the cell, then
#                  average across participants  -> participant-weighted mean
#   "pooled_mean"  flat mean over every repetition in the cell (no participant
#                  step)                        -> repetition-weighted mean
REP_AGGREGATIONS = ("sum", "mean", "pooled_mean")

# How each outcome is prepared, chosen to reproduce the published figures.
VARIABLE_SPECS: Dict[str, Dict[str, str]] = {
    "work": {
        "rep_aggregation": "sum",
        "units": "J per set",
        "source": "plot_work_comparison.py",
    },
    "knee_v_mean": {
        "rep_aggregation": "pooled_mean",
        "units": "deg/s",
        "source": "plot_knee_velocity_comparison.py",
    },
}

DEFAULT_PAIR_BY: Tuple[str, ...] = ("trses", "set")


# --------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------- #
def cell_means(
    df: pd.DataFrame,
    var: str,
    rep_aggregation: str = "mean",
    pair_by: Sequence[str] = DEFAULT_PAIR_BY,
    group_col: str = "tr_group",
    participant_col: str = "par",
) -> pd.DataFrame:
    """Collapse per-repetition rows to one value per (cell, group).

    Args:
        df: per-repetition discrete outcomes (already filtered).
        var: outcome column (e.g. ``"work"``, ``"knee_v_mean"``).
        rep_aggregation: one of :data:`REP_AGGREGATIONS`. See the module
            docstring for what each one means and which figure uses it.
        pair_by: columns defining the cell (default session x set).
        group_col: column holding ``"NH"`` / ``"IK"``.
        participant_col: participant identifier column.

    Returns:
        Long DataFrame with ``pair_by + [group_col, var]``.
    """
    if rep_aggregation not in REP_AGGREGATIONS:
        raise ValueError(
            f"rep_aggregation must be one of {REP_AGGREGATIONS}, got {rep_aggregation!r}"
        )
    missing = [c for c in (*pair_by, group_col, var) if c not in df.columns]
    if missing:
        raise KeyError(f"missing column(s) in df: {sorted(set(missing))}")

    keys = list(pair_by) + [group_col]

    if rep_aggregation == "pooled_mean":
        # Flat mean over every repetition in the cell (repetition-weighted).
        out = df.groupby(keys, observed=True)[var].mean().reset_index()
        return out

    if participant_col not in df.columns:
        raise KeyError(f"missing column(s) in df: ['{participant_col}']")

    per_participant = (
        df.groupby(keys + [participant_col], observed=True)[var]
        .agg("sum" if rep_aggregation == "sum" else "mean")
        .reset_index()
    )
    return per_participant.groupby(keys, observed=True)[var].mean().reset_index()


def build_pairs(
    df: pd.DataFrame,
    var: str,
    rep_aggregation: str = "mean",
    pair_by: Sequence[str] = DEFAULT_PAIR_BY,
    test_group: str = "NH",
    reference_group: str = "IK",
    group_col: str = "tr_group",
    participant_col: str = "par",
) -> Tuple[pd.DataFrame, int]:
    """Build the wide, one-row-per-cell table the Bland-Altman analysis needs.

    Returns:
        ``(pairs, n_incomplete)`` where ``pairs`` has columns
        ``pair_by + [test_group, reference_group, "mean", "diff"]`` restricted
        to cells in which BOTH groups are present, and ``n_incomplete`` counts
        the cells dropped because one group was missing. ``n_incomplete`` must
        be reported: silently dropping half-empty cells would bias the bias.
    """
    long = cell_means(
        df,
        var=var,
        rep_aggregation=rep_aggregation,
        pair_by=pair_by,
        group_col=group_col,
        participant_col=participant_col,
    )
    wide = long.pivot_table(
        index=list(pair_by), columns=group_col, values=var, aggfunc="mean"
    ).reset_index()

    for grp in (test_group, reference_group):
        if grp not in wide.columns:
            raise KeyError(f"group {grp!r} not present in column {group_col!r}")

    n_all = len(wide)
    pairs = wide.dropna(subset=[test_group, reference_group]).copy()
    n_incomplete = n_all - len(pairs)

    pairs["mean"] = (pairs[test_group] + pairs[reference_group]) / 2.0
    pairs["diff"] = pairs[test_group] - pairs[reference_group]
    return pairs, n_incomplete


def collapse_pairs(
    pairs: pd.DataFrame,
    by: Sequence[str],
    test_group: str = "NH",
    reference_group: str = "IK",
) -> pd.DataFrame:
    """Average cell-level pairs within a coarser unit (e.g. within session).

    Used for the clustering sensitivity analysis: averaging the sets of a
    session gives one pair per session, so the resulting differences are
    independent across sessions.
    """
    out = (
        pairs.groupby(list(by), observed=True)[[test_group, reference_group]]
        .mean()
        .reset_index()
    )
    out["mean"] = (out[test_group] + out[reference_group]) / 2.0
    out["diff"] = out[test_group] - out[reference_group]
    return out


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def concordance_correlation(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> Dict[str, float]:
    """Lin's concordance correlation coefficient with a Fisher-z 95% CI.

    Two point estimates are returned because two conventions are in circulation:

    ``ccc``     uses the sample (``ddof=1``) variances and covariance. This is
                the formula already implemented in ``calculate_ccc`` inside the
                two plotting modules, so it is the value printed on the
                published figures and is the headline value here.
    ``ccc_lin`` uses the population (``ddof=0``) moments of Lin (1989), which is
                what R's ``epiR::epi.ccc`` and most reference implementations
                return. The confidence interval belongs to this estimator,
                because Lin's asymptotic variance is derived from it.

    The two differ by a factor (n-1)/n on the variances and are numerically
    almost identical for the sample sizes used here; both are reported so the
    figure annotation and any external cross-check both reconcile.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3 or np.isnan(x).any() or np.isnan(y).any():
        nan = float("nan")
        return {
            "ccc": nan,
            "ccc_lin": nan,
            "ccc_ci_low": nan,
            "ccc_ci_high": nan,
            "pearson_r": nan,
            "cb": nan,
        }

    mx, my = float(np.mean(x)), float(np.mean(y))
    r = float(np.corrcoef(x, y)[0, 1])

    # figure-matching (sample moments, ddof=1)
    vx1, vy1 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    cov1 = float(np.cov(x, y, ddof=1)[0, 1])
    ccc_sample = 2.0 * cov1 / (vx1 + vy1 + (mx - my) ** 2)

    # Lin (1989) population moments
    vx0, vy0 = vx1 * (n - 1) / n, vy1 * (n - 1) / n
    cov0 = cov1 * (n - 1) / n
    ccc_lin = 2.0 * cov0 / (vx0 + vy0 + (mx - my) ** 2)

    # Lin's asymptotic SE of the estimate, then the z-transformed CI.
    u = (my - mx) / ((vx0 * vy0) ** 0.25)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_ccc = (
            (1.0 - r**2) * ccc_lin**2 * (1.0 - ccc_lin**2) / r**2
            + 2.0 * ccc_lin**3 * (1.0 - ccc_lin) * u**2 / r
            - 0.5 * ccc_lin**4 * u**4 / r**2
        ) / (n - 2)
        se_ccc = float(np.sqrt(var_ccc)) if var_ccc > 0 else float("nan")
        se_z = se_ccc / (1.0 - ccc_lin**2)
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z = np.arctanh(ccc_lin)
    ci_low = float(np.tanh(z - z_crit * se_z))
    ci_high = float(np.tanh(z + z_crit * se_z))

    # Bias-correction factor: how far the best-fit line sits from the identity
    # line (ccc = r * Cb, so Cb isolates the accuracy component).
    cb = ccc_lin / r if r != 0 else float("nan")

    return {
        "ccc": float(ccc_sample),
        "ccc_lin": float(ccc_lin),
        "ccc_ci_low": ci_low,
        "ccc_ci_high": ci_high,
        "pearson_r": r,
        "cb": float(cb),
    }


def _bland_altman_core(
    test: np.ndarray, reference: np.ndarray, alpha: float = 0.05, loa_z: float = 1.96
) -> Dict[str, float]:
    """Bias, SD of differences, limits of agreement and their CIs.

    Formulae are Bland & Altman (1986, Lancet; 1999, Stat Methods Med Res):

        bias        d_bar = mean(test - reference)
        SD(diff)    s     (ddof = 1)
        LoA         d_bar +/- 1.96 s
        SE(bias)    s / sqrt(n)                       -> CI with t(n-1)
        SE(LoA)     s * sqrt(1/n + 1.96^2 / (2(n-1))) -> CI with t(n-1)

    No bootstrapping: the limits-of-agreement CI is the standard closed-form
    normal approximation.
    """
    d = np.asarray(test, dtype=float) - np.asarray(reference, dtype=float)
    n = d.size
    if n < 2:
        raise ValueError(f"need at least 2 pairs for a Bland-Altman analysis, got {n}")

    bias = float(np.mean(d))
    sd_diff = float(np.std(d, ddof=1))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))

    se_bias = sd_diff / np.sqrt(n)
    se_loa = sd_diff * np.sqrt(1.0 / n + loa_z**2 / (2.0 * (n - 1)))

    loa_low = bias - loa_z * sd_diff
    loa_high = bias + loa_z * sd_diff

    # one-sample t-test of "is the systematic bias zero?"
    if sd_diff > 0:
        t_stat = bias / se_bias
        p_bias = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
    else:
        t_stat, p_bias = float("nan"), float("nan")

    return {
        "n_pairs": int(n),
        "bias": bias,
        "sd_diff": sd_diff,
        "bias_se": float(se_bias),
        "bias_ci_low": bias - t_crit * se_bias,
        "bias_ci_high": bias + t_crit * se_bias,
        "bias_t": float(t_stat),
        "bias_p": p_bias,
        "loa_low": loa_low,
        "loa_low_ci_low": loa_low - t_crit * se_loa,
        "loa_low_ci_high": loa_low + t_crit * se_loa,
        "loa_high": loa_high,
        "loa_high_ci_low": loa_high - t_crit * se_loa,
        "loa_high_ci_high": loa_high + t_crit * se_loa,
        "loa_se": float(se_loa),
        "loa_z": float(loa_z),
    }


def _proportional_bias(mean: np.ndarray, diff: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """OLS regression of the difference on the mean (Bland & Altman's check).

    A slope significantly different from zero means the disagreement grows (or
    shrinks) with the magnitude of the measurement, i.e. a single pair of
    limits of agreement is not appropriate across the whole range.
    """
    mean = np.asarray(mean, dtype=float)
    diff = np.asarray(diff, dtype=float)
    n = mean.size
    if n < 3 or np.allclose(mean, mean[0]):
        nan = float("nan")
        return {
            "prop_bias_slope": nan,
            "prop_bias_intercept": nan,
            "prop_bias_slope_se": nan,
            "prop_bias_slope_ci_low": nan,
            "prop_bias_slope_ci_high": nan,
            "prop_bias_p": nan,
            "prop_bias_r2": nan,
        }
    reg = stats.linregress(mean, diff)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 2))
    return {
        "prop_bias_slope": float(reg.slope),
        "prop_bias_intercept": float(reg.intercept),
        "prop_bias_slope_se": float(reg.stderr),
        "prop_bias_slope_ci_low": float(reg.slope - t_crit * reg.stderr),
        "prop_bias_slope_ci_high": float(reg.slope + t_crit * reg.stderr),
        "prop_bias_p": float(reg.pvalue),
        "prop_bias_r2": float(reg.rvalue**2),
    }


def agreement_from_pairs(
    test: np.ndarray,
    reference: np.ndarray,
    alpha: float = 0.05,
    loa_z: float = 1.96,
) -> Dict[str, Any]:
    """Full agreement summary for two already-paired arrays.

    Kept separate from :func:`compute_agreement` so it can be reused on any
    pairing (cell-level, session-level, or a future achieved-vs-prescribed
    pairing) without re-deriving the cells.
    """
    test = np.asarray(test, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if test.shape != reference.shape:
        raise ValueError("test and reference must have the same length")

    pair_mean = (test + reference) / 2.0
    diff = test - reference

    out: Dict[str, Any] = {}
    out.update(_bland_altman_core(test, reference, alpha=alpha, loa_z=loa_z))
    out.update(_proportional_bias(pair_mean, diff, alpha=alpha))
    out.update(concordance_correlation(test, reference, alpha=alpha))

    grand_mean = float(np.mean(pair_mean))
    out["test_mean"] = float(np.mean(test))
    out["test_sd"] = float(np.std(test, ddof=1))
    out["reference_mean"] = float(np.mean(reference))
    out["reference_sd"] = float(np.std(reference, ddof=1))
    out["grand_mean"] = grand_mean

    # Percentage difference, expressed against the grand mean of the two
    # groups (the usual "bias as % of the measured quantity").
    if grand_mean != 0:
        out["pct_bias"] = 100.0 * out["bias"] / grand_mean
        out["pct_loa_low"] = 100.0 * out["loa_low"] / grand_mean
        out["pct_loa_high"] = 100.0 * out["loa_high"] / grand_mean
    else:
        out["pct_bias"] = out["pct_loa_low"] = out["pct_loa_high"] = float("nan")

    # ... and as the mean of the per-pair percentage differences, which is what
    # some authors mean by "percentage difference". Reported separately because
    # it is NOT the same number as pct_bias whenever the magnitude varies.
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_pairwise = np.where(pair_mean != 0, 100.0 * diff / pair_mean, np.nan)
    if np.isfinite(pct_pairwise).sum() >= 2:
        out["pct_diff_mean"] = float(np.nanmean(pct_pairwise))
        out["pct_diff_sd"] = float(np.nanstd(pct_pairwise, ddof=1))
    else:
        out["pct_diff_mean"] = out["pct_diff_sd"] = float("nan")

    # Normality of the differences underpins the 1.96*SD limits.
    if diff.size >= 3:
        out["shapiro_w"], out["shapiro_p"] = (
            float(v) for v in stats.shapiro(diff)
        )
    else:
        out["shapiro_w"] = out["shapiro_p"] = float("nan")

    return out


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #
def compute_agreement(
    df: pd.DataFrame,
    var: str,
    rep_aggregation: Optional[str] = None,
    pair_by: Sequence[str] = DEFAULT_PAIR_BY,
    test_group: str = "NH",
    reference_group: str = "IK",
    group_col: str = "tr_group",
    participant_col: str = "par",
    alpha: float = 0.05,
    loa_z: float = 1.96,
    session_col: str = "trses",
) -> Dict[str, Any]:
    """Bland-Altman agreement between the NH and IK groups for one outcome.

    The pairing is at the (session x set) CELL level: one point per cell, the
    NH group mean against the IK group mean. See the module docstring for why
    that — and not participant-to-participant — is the defensible unit, and for
    the four caveats that must accompany any reported number.

    Args:
        df: per-repetition discrete outcomes, already passed through
            ``filter_training_data`` (the plotting functions receive the same
            frame, which is why the numbers match the published figures).
        var: outcome column. ``"work"`` and ``"knee_v_mean"`` get their
            repetition aggregation from :data:`VARIABLE_SPECS` automatically.
        rep_aggregation: override the aggregation; one of
            :data:`REP_AGGREGATIONS`. Defaults to the spec for ``var``, or
            ``"mean"`` for an unlisted outcome.
        pair_by: columns defining a cell. Default ``("trses", "set")``.
        test_group: group put FIRST in the difference (default ``"NH"``).
        reference_group: group SUBTRACTED (default ``"IK"``), so
            ``diff = NH - IK`` exactly as in both figures. NH is conceptually
            the reference because it defined the target; a positive bias means
            IK delivered less than prescribed.
        group_col, participant_col: column names.
        alpha: two-sided alpha for every confidence interval (default 0.05).
        loa_z: multiplier for the limits of agreement (default 1.96).
        session_col: column used for the session-level sensitivity analysis.

    Returns:
        dict with, in plain terms: ``n_pairs``; the mean and SD of each group
        (``test_mean``/``reference_mean``); ``bias`` (NH minus IK) with its
        standard error, 95% CI and t-test p-value; ``sd_diff``; ``loa_low`` and
        ``loa_high`` (bias +/- 1.96 SD) each with their own 95% CI;
        ``prop_bias_slope``/``prop_bias_p`` (OLS of difference on mean);
        ``ccc`` (Lin's concordance correlation, with ``ccc_lin`` and its CI) and
        ``pearson_r``; the percentage difference (``pct_bias``, ``pct_loa_*``,
        ``pct_diff_mean``); a Shapiro-Wilk check on the differences; and a
        ``session`` sub-dict holding the same core statistics computed on the
        15 session-level pairs as a clustering sensitivity analysis.
    """
    spec = VARIABLE_SPECS.get(var, {})
    rep_aggregation = rep_aggregation or spec.get("rep_aggregation", "mean")

    pairs, n_incomplete = build_pairs(
        df,
        var=var,
        rep_aggregation=rep_aggregation,
        pair_by=pair_by,
        test_group=test_group,
        reference_group=reference_group,
        group_col=group_col,
        participant_col=participant_col,
    )

    result = agreement_from_pairs(
        pairs[test_group].to_numpy(),
        pairs[reference_group].to_numpy(),
        alpha=alpha,
        loa_z=loa_z,
    )

    n_test = int(df.loc[df[group_col] == test_group, participant_col].nunique())
    n_ref = int(df.loc[df[group_col] == reference_group, participant_col].nunique())

    result.update(
        {
            "var": var,
            "units": spec.get("units", ""),
            "test_group": test_group,
            "reference_group": reference_group,
            "rep_aggregation": rep_aggregation,
            "pair_by": list(pair_by),
            "unit_of_analysis": (
                f"{' x '.join(pair_by)} cell; each value is a GROUP MEAN "
                f"(between-subject design, so pairing is by prescription cell, "
                f"never participant to participant)"
            ),
            "n_pairs_incomplete_dropped": int(n_incomplete),
            "n_participants_test": n_test,
            "n_participants_reference": n_ref,
        }
    )

    # Sensitivity: collapse the sets of a session so the differences are
    # independent across sessions (the cell-level differences are clustered).
    if session_col in pairs.columns:
        ses = collapse_pairs(
            pairs, by=[session_col], test_group=test_group, reference_group=reference_group
        )
        if len(ses) >= 2:
            result["session"] = agreement_from_pairs(
                ses[test_group].to_numpy(),
                ses[reference_group].to_numpy(),
                alpha=alpha,
                loa_z=loa_z,
            )

    result["pairs"] = pairs
    return result


def _r(value: Any, nd: int) -> Any:
    """Round for JSON, passing NaN/inf through as None so the file stays valid."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(f):
        return None
    return round(f, nd)


def _flatten(res: Dict[str, Any], var: str, mag_nd: int) -> Dict[str, Any]:
    """Flatten one outcome's results into ``ba_{var}_*`` scalar keys.

    ``mag_nd`` is the number of decimals used for quantities carrying the
    outcome's own units (joules, degrees per second); unitless quantities
    (coefficients, p-values, percentages) get 3-4.
    """
    p = f"ba_{var}_"
    flat: Dict[str, Any] = {
        p + "n_pairs": int(res["n_pairs"]),
        p + "n_dropped_cells": int(res["n_pairs_incomplete_dropped"]),
        p + "n_nh": int(res["n_participants_test"]),
        p + "n_ik": int(res["n_participants_reference"]),
        p + "units": res["units"],
        p + "unit_of_analysis": res["unit_of_analysis"],
        p + "rep_aggregation": res["rep_aggregation"],
        p + "nh_mean": _r(res["test_mean"], mag_nd),
        p + "nh_sd": _r(res["test_sd"], mag_nd),
        p + "ik_mean": _r(res["reference_mean"], mag_nd),
        p + "ik_sd": _r(res["reference_sd"], mag_nd),
        p + "grand_mean": _r(res["grand_mean"], mag_nd),
        p + "bias": _r(res["bias"], mag_nd),
        p + "bias_ci_low": _r(res["bias_ci_low"], mag_nd),
        p + "bias_ci_high": _r(res["bias_ci_high"], mag_nd),
        p + "bias_p": _r(res["bias_p"], 4),
        p + "sd_diff": _r(res["sd_diff"], mag_nd),
        p + "loa_low": _r(res["loa_low"], mag_nd),
        p + "loa_low_ci_low": _r(res["loa_low_ci_low"], mag_nd),
        p + "loa_low_ci_high": _r(res["loa_low_ci_high"], mag_nd),
        p + "loa_high": _r(res["loa_high"], mag_nd),
        p + "loa_high_ci_low": _r(res["loa_high_ci_low"], mag_nd),
        p + "loa_high_ci_high": _r(res["loa_high_ci_high"], mag_nd),
        p + "prop_bias_slope": _r(res["prop_bias_slope"], 4),
        p + "prop_bias_slope_ci_low": _r(res["prop_bias_slope_ci_low"], 4),
        p + "prop_bias_slope_ci_high": _r(res["prop_bias_slope_ci_high"], 4),
        p + "prop_bias_p": _r(res["prop_bias_p"], 4),
        p + "ccc": _r(res["ccc"], 3),
        p + "ccc_lin": _r(res["ccc_lin"], 3),
        p + "ccc_ci_low": _r(res["ccc_ci_low"], 3),
        p + "ccc_ci_high": _r(res["ccc_ci_high"], 3),
        p + "pearson_r": _r(res["pearson_r"], 3),
        p + "pct_bias": _r(res["pct_bias"], 2),
        p + "pct_loa_low": _r(res["pct_loa_low"], 2),
        p + "pct_loa_high": _r(res["pct_loa_high"], 2),
        p + "pct_diff_mean": _r(res["pct_diff_mean"], 2),
        p + "pct_diff_sd": _r(res["pct_diff_sd"], 2),
        p + "shapiro_p": _r(res["shapiro_p"], 4),
    }

    ses = res.get("session")
    if ses is not None:
        flat.update(
            {
                p + "ses_n_pairs": int(ses["n_pairs"]),
                p + "ses_bias": _r(ses["bias"], mag_nd),
                p + "ses_bias_ci_low": _r(ses["bias_ci_low"], mag_nd),
                p + "ses_bias_ci_high": _r(ses["bias_ci_high"], mag_nd),
                p + "ses_sd_diff": _r(ses["sd_diff"], mag_nd),
                p + "ses_loa_low": _r(ses["loa_low"], mag_nd),
                p + "ses_loa_high": _r(ses["loa_high"], mag_nd),
                p + "ses_pct_bias": _r(ses["pct_bias"], 2),
            }
        )
    return flat


def _merge_into_json(flat: Dict[str, Any], json_path: Path) -> int:
    """Read, update, and atomically rewrite the stats JSON.

    The file is the single source of truth for the manuscript, so this must
    never truncate it. A missing file is fine (start empty); a corrupt one is
    an error rather than an excuse to overwrite, and the new content is written
    to a temporary file in the same directory and then atomically renamed, so
    an interrupted write cannot leave a half-file behind.
    """
    try:
        with open(json_path) as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {}
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"{json_path} is not valid JSON ({exc}); refusing to overwrite it. "
            "Fix or restore the file before rerunning."
        ) from exc

    if not isinstance(existing, dict):  # pragma: no cover - defensive
        raise RuntimeError(f"{json_path} does not contain a JSON object; refusing to overwrite.")

    n_before = len(existing)
    n_new = len([k for k in flat if k not in existing])
    existing.update(flat)

    fd, tmp = tempfile.mkstemp(dir=str(json_path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(existing, f, indent=4)
        os.replace(tmp, json_path)
    except BaseException:  # pragma: no cover - defensive
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    print(
        f"Agreement stats written to {json_path}: "
        f"{n_before} keys before, {len(existing)} after ({n_new} new)."
    )
    return len(existing)


def run_agreement(
    training_disc: pd.DataFrame,
    write_json: bool = True,
    json_path: Optional[Path] = None,
    variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute NH-vs-IK agreement for work and mean knee velocity.

    Mirrors ``mixed_lmm_analysis.run_mixed_lmm``: returns a nested dict of
    results keyed by outcome, with a ``_flat`` entry holding the scalar
    ``ba_*`` keys, and optionally merges those keys into ``training_stats.json``
    without disturbing anything already in it.

    Args:
        training_disc: filtered per-repetition discrete data (the frame
            ``jhamon_training.py`` builds and hands to the plotting functions).
        write_json: merge the ``ba_*`` keys into the stats JSON.
        json_path: target JSON (defaults to
            ``RESULTS_TRAINING_PATH / "training_stats.json"``).
        variables: outcomes to analyse (defaults to work and knee velocity).

    Returns:
        ``{var: <result dict>, "_flat": {...}}``. Each result dict also carries
        the cell-level ``pairs`` DataFrame, which the Bland-Altman figures could
        be redrawn from.
    """
    variables = variables or ["work", "knee_v_mean"]

    results: Dict[str, Any] = {}
    flat: Dict[str, Any] = {
        "ba_pairing": (
            "session x set cell; NH group mean vs IK group mean (between-subject "
            "groups, so pairing is by prescription cell, not by participant)"
        ),
        "ba_reference": "NH (defines the target); difference = NH - IK",
    }

    for var in variables:
        # Magnitudes in joules deserve 1 decimal; degrees per second, 2.
        mag_nd = 1 if var == "work" else 2
        res = compute_agreement(training_disc, var)
        results[var] = res
        flat.update(_flatten(res, var, mag_nd))

        ses = res.get("session", {})
        print(
            f"[agreement] {var}: n={res['n_pairs']} cell pairs "
            f"(dropped {res['n_pairs_incomplete_dropped']}), "
            f"NH {res['test_mean']:.2f} vs IK {res['reference_mean']:.2f}, "
            f"bias {res['bias']:+.2f} ({res['pct_bias']:+.2f}%), "
            f"LoA [{res['loa_low']:.2f}, {res['loa_high']:.2f}], "
            f"CCC {res['ccc']:.3f}"
            + (f", session-level bias {ses['bias']:+.2f}" if ses else "")
        )

    # The velocity figure averages over repetitions rather than participants,
    # so participants with more repetitions carry more weight. Recompute it the
    # participant-weighted way and store it, so the choice is auditable instead
    # of buried in the plotting code.
    if "knee_v_mean" in variables:
        pw = compute_agreement(
            training_disc, "knee_v_mean", rep_aggregation="mean"
        )
        flat["ba_knee_v_mean_pw_bias"] = _r(pw["bias"], 2)
        flat["ba_knee_v_mean_pw_sd_diff"] = _r(pw["sd_diff"], 2)
        flat["ba_knee_v_mean_pw_loa_low"] = _r(pw["loa_low"], 2)
        flat["ba_knee_v_mean_pw_loa_high"] = _r(pw["loa_high"], 2)
        results["knee_v_mean_participant_weighted"] = pw

    results["_flat"] = flat

    if write_json:
        target = Path(json_path or (RESULTS_TRAINING_PATH / "training_stats.json"))
        _merge_into_json(flat, target)

    return results
