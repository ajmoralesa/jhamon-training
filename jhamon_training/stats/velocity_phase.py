"""Isokinetic-phase angular velocity: the like-for-like velocity comparison.

Why this exists
---------------
The standardisation procedure programmed the dynamometer to a CONSTANT target
angular velocity for each set, equal to the Nordic group's mean angular velocity
for the corresponding set. Asking whether that standardisation worked therefore
means asking whether the velocity the isokinetic participants actually trained
at equals the Nordic mean.

The velocity stored per repetition does not answer that question, because the
two modalities' repetition windows end on different kinematic events and the
isokinetic window contains two phases that are instrumental rather than
biological:

    Nordic      onset (force) -> the instant of PEAK angular velocity
                (``data/nordic.py:651``, ``REP = D[force_onset:vmin_idx]``).
                A free, gravity-driven descent: velocity rises monotonically
                throughout and there is no deceleration inside the window.

    Isokinetic  velocity zero-crossing -> velocity zero-crossing
                (``data/ik.py:75-81``). The dynamometer must ACCELERATE the limb
                from rest up to the preset velocity and DECELERATE it back to
                rest at the end of the range of motion. Both ramps are properties
                of the machine, not of the prescribed stimulus, and both sit
                inside the averaging window.

Measured on the 9 932 isokinetic repetitions of this dataset, the group-mean
velocity profile is::

    % of repetition    0     5    10    15    25    50    75    85    90    95   100
    |omega| (deg/s)  0.06 16.74 21.63 23.05 23.58 23.60 23.58 23.58 11.42  3.11  0.06
                     |<-- ramp up -->|<------- isokinetic plateau ------->|<-ramp->|

Averaging across the whole window gives 20.27 deg/s; averaging across the
plateau gives 23.58 deg/s. The 3.3 deg/s difference between them is entirely the
two ramps, and it is the same 3.3 deg/s that appears as the apparent NH-minus-IK
"velocity standardisation bias" when the full-window mean is used.

So the full-window mean answers "how fast did the limb move on average, including
the time the machine spent starting and stopping it", which no one prescribed.
The plateau mean answers "how fast did the participant train", which is exactly
what was prescribed. This module computes the second one.

Definition
----------
The isokinetic phase of a repetition is the set of samples whose absolute
angular velocity is at least ``threshold`` times that repetition's peak absolute
angular velocity, with ``threshold = 0.95`` by default. This is the conventional
"load range" / constant-velocity window of the isokinetic literature and it is
defined per repetition, so it adapts to the different target velocities
prescribed for different sets without needing the target to be known.

The definition is deliberately insensitive to the exact threshold: 0.90, 0.95
and 0.98 give plateau means of 23.51, 23.58 and 23.61 deg/s respectively, and
the fixed mid-window rule (25-75 % of the repetition) gives 23.60. Any of them
supports the same conclusion; :func:`isokinetic_phase_velocity` returns all of
them so the sensitivity is reported rather than asserted.

What is NOT done here
---------------------
No equivalent trimming is applied to the Nordic repetitions. Their window
contains no instrumental ramp -- the acceleration of a Nordic descent IS the
stimulus, not an artefact of the apparatus -- so trimming it would remove signal
rather than noise. The asymmetry of the treatment is therefore deliberate and
must be stated wherever these numbers are reported: the Nordic value is the mean
over the whole analysed window, the isokinetic value is the mean over the
constant-velocity plateau, and the comparison is justified because the plateau
is what the dynamometer was programmed to deliver.

This module also does not address the separate question of whether the Nordic
window should extend past peak velocity to the velocity zero-crossing, as the
Methods text states. That would change Nordic work and range of motion as well
as velocity and requires re-segmenting the raw signals.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

#: Fraction of a repetition's peak absolute velocity above which a sample counts
#: as part of the constant-velocity plateau. 0.95 is the primary definition; the
#: others are computed alongside it as a sensitivity analysis.
DEFAULT_THRESHOLD = 0.95
SENSITIVITY_THRESHOLDS: Sequence[float] = (0.90, 0.95, 0.98)

#: Fixed-window alternative: the middle portion of the time-normalised
#: repetition, as a threshold-free check that the plateau value is not an
#: artefact of the peak-relative rule.
MID_WINDOW = (25, 75)


def _curve_matrix(ikdf: pd.DataFrame, var: str = "knee_v") -> pd.DataFrame:
    """One row per repetition, 101 columns of absolute angular velocity.

    Args:
        ikdf: long isokinetic curve frame from ``frames.ikt_todf``.
        var: curve variable to extract.

    Returns:
        DataFrame indexed by (par, trses, set, rep) with the 101 time-normalised
        nodes as columns, values as ``|omega|`` so the sign convention of the
        cache cannot change the result.
    """
    sub = ikdf[ikdf["var"] == var]
    wide = (
        sub.groupby(["par", "trses", "set", "rep", "timepoint"], observed=True)["value"]
        .mean()
        .unstack("timepoint")
        .sort_index(axis=1)
    )
    if wide.shape[1] != 101:
        raise ValueError(f"expected 101 time-normalised nodes, got {wide.shape[1]}")
    return wide.abs()


def isokinetic_phase_velocity(
    ikdf: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    sensitivity: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Mean angular velocity over the constant-velocity plateau of each repetition.

    Args:
        ikdf: long isokinetic curve frame (``frames.ikt_todf`` output).
        threshold: primary plateau definition, as a fraction of the
            repetition's peak absolute velocity.
        sensitivity: additional thresholds to compute alongside. Defaults to
            :data:`SENSITIVITY_THRESHOLDS`.

    Returns:
        One row per repetition with columns:

        ``knee_v_iso``          plateau mean at ``threshold`` (deg/s) -- primary
        ``knee_v_iso_p{n}``     plateau mean at each sensitivity threshold
        ``knee_v_mid``          mean over :data:`MID_WINDOW` of the repetition
        ``knee_v_window``       mean over the whole window (reproduces the
                                pipeline's ``knee_v_mean``, as a cross-check)
        ``iso_start_pct`` /
        ``iso_end_pct`` /
        ``iso_width_pct``       extent of the plateau, so the ramps can be
                                described rather than merely excluded
    """
    sensitivity = list(sensitivity or SENSITIVITY_THRESHOLDS)
    if threshold not in sensitivity:
        sensitivity = sorted({*sensitivity, threshold})

    wide = _curve_matrix(ikdf)
    Y = wide.to_numpy(dtype=float)
    peak = np.nanmax(Y, axis=1, keepdims=True)

    out = pd.DataFrame(index=wide.index)
    out["knee_v_window"] = np.nanmean(Y, axis=1)
    lo, hi = MID_WINDOW
    out["knee_v_mid"] = np.nanmean(Y[:, lo : hi + 1], axis=1)

    for thr in sensitivity:
        mask = Y >= thr * peak
        plateau = np.where(mask, Y, np.nan)
        column = "knee_v_iso" if thr == threshold else f"knee_v_iso_p{int(thr * 100)}"
        out[column] = np.nanmean(plateau, axis=1)
        if thr == threshold:
            # Node indices are percentages because the curves carry 101 nodes.
            first = np.argmax(mask, axis=1).astype(float)
            last = (100 - np.argmax(mask[:, ::-1], axis=1)).astype(float)
            out["iso_start_pct"] = first
            out["iso_end_pct"] = last
            out["iso_width_pct"] = mask.sum(axis=1).astype(float)

    return out.reset_index()


def attach_isokinetic_velocity(
    training_disc: pd.DataFrame,
    ikdf: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Add the plateau velocity to the per-repetition discrete frame.

    Isokinetic repetitions receive their plateau mean; Nordic repetitions
    receive their existing window mean unchanged, because the Nordic window
    contains no instrumental ramp to remove (see the module docstring). The
    resulting ``knee_v_iso`` column is therefore "the velocity each participant
    actually trained at" in both groups, and is the column the velocity
    standardisation should be judged on.

    Args:
        training_disc: per-repetition discrete outcomes for both groups.
        ikdf: long isokinetic curve frame.
        threshold: plateau definition, as a fraction of peak velocity.

    Returns:
        A copy of ``training_disc`` with ``knee_v_iso`` and the sensitivity
        columns merged in.

    Raises:
        RuntimeError: if any isokinetic repetition fails to receive a plateau
            value, which would silently bias the group mean.
    """
    iso = isokinetic_phase_velocity(ikdf, threshold=threshold)
    keys = ["par", "trses", "set", "rep"]
    merged = training_disc.merge(iso, on=keys, how="left", validate="one_to_one")

    is_ik = merged["tr_group"] == "IK"
    missing = int(is_ik.sum() - merged.loc[is_ik, "knee_v_iso"].notna().sum())
    if missing:
        raise RuntimeError(
            f"{missing} isokinetic repetitions did not match a plateau velocity; "
            "the merge keys are inconsistent and the group mean would be biased."
        )

    # Nordic keeps its window mean under every velocity column, so downstream
    # code can treat the columns uniformly without special-casing the group.
    nordic_columns = [c for c in iso.columns if c.startswith("knee_v")]
    for column in nordic_columns:
        merged.loc[~is_ik, column] = merged.loc[~is_ik, "knee_v_mean"]

    return merged


def describe_phases(iso: pd.DataFrame) -> Dict[str, float]:
    """Summary of the plateau and the two ramps, for the Methods and Results.

    Args:
        iso: output of :func:`isokinetic_phase_velocity`.

    Returns:
        Flat dict of ``velphase_*`` keys: the plateau extent, the plateau mean
        under every threshold, the mid-window mean, the full-window mean, and
        the difference between plateau and full-window means -- which is the
        quantity that would otherwise be misread as a standardisation failure.
    """
    plateau = float(iso["knee_v_iso"].mean())
    window = float(iso["knee_v_window"].mean())
    out: Dict[str, float] = {
        "velphase_threshold": DEFAULT_THRESHOLD,
        "velphase_n_reps": int(len(iso)),
        "velphase_ik_plateau_mean": round(plateau, 2),
        "velphase_ik_plateau_sd": round(float(iso["knee_v_iso"].std()), 2),
        "velphase_ik_window_mean": round(window, 2),
        "velphase_ik_window_sd": round(float(iso["knee_v_window"].std()), 2),
        "velphase_ik_mid_mean": round(float(iso["knee_v_mid"].mean()), 2),
        "velphase_ramp_cost": round(plateau - window, 2),
        "velphase_ramp_cost_pct": round(100.0 * (plateau - window) / plateau, 1),
        "velphase_start_pct_median": round(float(iso["iso_start_pct"].median()), 1),
        "velphase_end_pct_median": round(float(iso["iso_end_pct"].median()), 1),
        "velphase_width_pct_median": round(float(iso["iso_width_pct"].median()), 1),
    }
    for column in iso.columns:
        if column.startswith("knee_v_iso_p"):
            out[f"velphase_{column.replace('knee_v_iso_', '')}_mean"] = round(
                float(iso[column].mean()), 2
            )
    return out
