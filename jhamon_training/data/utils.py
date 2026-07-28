REP_KEYS = ["par", "trses", "set", "rep"]


def dame_manual_exclusions(df, legacy=False):
    """Drop repetitions whose curves are physically implausible.

    A repetition is rejected on the evidence of ONE of its variables -- an
    impossible starting knee angle, a force trace that never rises -- but the
    rejection applies to the whole repetition, because a segmentation failure
    corrupts every signal recorded over that window, not just the one that
    happened to trip a threshold.

    Two defects in the original implementation are corrected here.

    1. *The exclusion was scoped to a variable, not to a repetition.* Filters
       collected the offending rows' ``all_labels`` and dropped rows matching
       them, but ``frames.py`` builds ``all_labels`` from
       ``par trses set rep var tr_group`` -- the variable name is part of the
       label. Only the flagged variable's 101 rows were therefore removed, so a
       Nordic repetition rejected for an impossible starting angle went on
       contributing its torque curve to every torque analysis. Measured on the
       current caches the mismatch is large: 12 827 Nordic ``knee_ROM`` rows
       were removed against 2 727 ``torque`` rows and 12 423 ``force`` rows,
       i.e. three different subsets of repetitions depending on which variable
       an analysis happened to read.

    2. *The isokinetic angle threshold predated the sign convention.*
       ``IKfilt_knee2`` tested ``knee_ROM`` at node 100 for ``> -70``, written
       when isokinetic angles were stored negative. ``frames.ikt_todf`` now
       converts them to the Nordic convention (``90 - abs(value)``, so about
       +11.7 deg at the end of a repetition), and every value is greater than
       -70: the filter matched 100 % of isokinetic repetitions (9 932 of 9 932)
       and deleted the entire isokinetic angle dataset. Defect 1 hid it, since
       isokinetic torque and velocity survived untouched, so nothing downstream
       failed -- it only surfaced when an analysis needed isokinetic
       ``knee_ROM``, which angle normalisation does. The threshold is restated
       below in the current convention: reject a repetition that finished more
       than 20 deg short of full extension, which is what ``> -70`` meant when
       angles ran the other way.

    Two further observations, recorded rather than silently perpetuated:

    * The three ``hip_ROM`` filters are inert. ``analizame_curvas`` stopped
      exporting ``hip_ROM`` as a normalised curve (it is commented out in
      ``data/nordic.py``), so no row ever matches them. They are kept, disabled
      and labelled, so that restoring the hip curves also restores the filter.
    * ``NHfilt_hip3`` was assigned twice, the second assignment overwriting the
      first, so the ``hip_ROM`` at node 99 ``> 30`` criterion never applied.
      Both halves are expressed as one two-sided criterion below.

    Args:
        df: long curve frame with ``par``, ``trses``, ``set``, ``rep``, ``var``,
            ``timepoint``, ``value`` and ``tr_group``.
        legacy: reproduce the original variable-scoped behaviour, including the
            broken isokinetic threshold. Only for demonstrating what the
            published numbers were computed on; never for new analyses.

    Returns:
        The frame with the rejected repetitions removed.
    """
    import pandas as pd

    if legacy:
        return _dame_manual_exclusions_legacy(df)

    def flagged(group, timepoint, var, op, threshold):
        """Repetition keys whose ``var`` at ``timepoint`` fails the criterion."""
        rows = df[
            (df["tr_group"] == group)
            & (df["timepoint"] == timepoint)
            & (df["var"] == var)
        ]
        hit = rows["value"] > threshold if op == ">" else rows["value"] < threshold
        return pd.MultiIndex.from_frame(rows.loc[hit, REP_KEYS])

    criteria = [
        # Nordic: knee angle must start near 90 deg of flexion and finish near
        # extension; a repetition outside that window was mis-segmented.
        ("NH", 0, "knee_ROM", "<", 80),
        ("NH", 0, "knee_ROM", ">", 108),
        ("NH", 99, "knee_ROM", ">", 45),
        # Nordic: force must start from rest and be established by mid-descent.
        ("NH", 1, "force", ">", 100),
        ("NH", 60, "force", "<", 50),
        ("NH", 0, "torque", ">", 50),
        # Isokinetic: the dynamometer's own range, in the positive convention
        # frames.ikt_todf now produces. Mirrors the Nordic start/end checks.
        ("IK", 0, "knee_ROM", "<", 80),
        ("IK", 100, "knee_ROM", ">", 20),
        ("IK", 80, "torque", "<", 50),
    ]
    # Deliberately NOT carried over: ("IK", 50, "knee_ROM", ">", 50).
    #
    # Like IKfilt_knee2 this was written for the old negative convention, under
    # which no isokinetic value ever exceeded 50 and the criterion was inert --
    # it has never excluded a repetition in any published run. Reinstating it
    # verbatim under the current positive convention would not restore an
    # intended behaviour, it would invent one, and a harmful one: a linear sweep
    # from 88.9 to 11.7 deg passes 50.3 deg at its midpoint, so "reject a
    # repetition whose mid-descent angle exceeds 50 deg" sits at the centre of
    # the normal distribution and discards 9.1 % of perfectly ordinary
    # repetitions. The start-of-range check above expresses the plausibility
    # constraint that the mid-descent one was presumably reaching for.
    # Disabled: hip_ROM is no longer exported as a normalised curve, so these
    # match nothing. Re-enable together with the curve in data/nordic.py.
    _hip_criteria_pending_curve_restoration = [
        ("NH", 1, "hip_ROM", ">", 10),
        ("NH", 1, "hip_ROM", "<", -20),
        ("NH", 99, "hip_ROM", ">", 30),
        ("NH", 99, "hip_ROM", "<", -30),
    ]
    available = set(df["var"].unique())
    active = [c for c in criteria if c[2] in available]

    rejected = pd.MultiIndex.from_tuples([], names=REP_KEYS)
    for group, timepoint, var, op, threshold in active:
        rejected = rejected.append(flagged(group, timepoint, var, op, threshold))
    rejected = rejected.unique()

    keep = ~pd.MultiIndex.from_frame(df[REP_KEYS]).isin(rejected)
    return df[keep]


def _dame_manual_exclusions_legacy(df):
    """The original variable-scoped implementation, for reproduction only.

    Retained verbatim so that the published numbers can be regenerated and the
    effect of the correction quantified. See ``dame_manual_exclusions`` for what
    is wrong with it.
    """

    import pandas as pd
    import numpy as np

    # Nordic filters
    NHfilt_knee1 = df[
        (
            (df["tr_group"] == "NH")
            & (df["timepoint"] == 0)
            & (df["var"] == "knee_ROM")
            & (df["value"] < 80)
        )
    ]["all_labels"]
    NHfilt_knee2 = df[
        (
            (df["tr_group"] == "NH")
            & (df["timepoint"] == 0)
            & (df["var"] == "knee_ROM")
            & (df["value"] > 108)
        )
    ]["all_labels"]
    NHfilt_knee3 = df[
        (
            (df["tr_group"] == "NH")
            & (df["timepoint"] == 99)
            & (df["var"] == "knee_ROM")
            & (df["value"] > 45)
        )
    ]["all_labels"]

    NHfilt_hip1 = df[
        (
            (df["tr_group"] == "NH")
            & (df["timepoint"] == 1)
            & (df["var"] == "hip_ROM")
            & (df["value"] > 10)
        )
    ]["all_labels"]
    NHfilt_hip2 = df[
        (
            (df["tr_group"] == "NH")
            & (df["timepoint"] == 1)
            & (df["var"] == "hip_ROM")
            & (df["value"] < -20)
        )
    ]["all_labels"]
    NHfilt_hip3 = df[
        (
            (df["tr_group"] == "NH")
            & (df["timepoint"] == 99)
            & (df["var"] == "hip_ROM")
            & (df["value"] > 30)
        )
    ]["all_labels"]
    NHfilt_hip3 = df[
        (
            (df["tr_group"] == "NH")
            & (df["timepoint"] == 99)
            & (df["var"] == "hip_ROM")
            & (df["value"] < -30)
        )
    ]["all_labels"]

    NHfilt_fonset = df[
        (
            (df["tr_group"] == "NH")
            & (df["timepoint"] == 1)
            & (df["var"] == "force")
            & (df["value"] > 100)
        )
    ]["all_labels"]
    NHfilt_f = df[
        (
            (df["tr_group"] == "NH")
            & (df["timepoint"] == 60)
            & (df["var"] == "force")
            & (df["value"] < 50)
        )
    ]["all_labels"]

    NHfilt_tor = df[
        (
            (df["tr_group"] == "NH")
            & (df["timepoint"] == 0)
            & (df["var"] == "torque")
            & (df["value"] > 50)
        )
    ]["all_labels"]

    # Isokinetic filters
    IKfilt_knee1 = df[
        (
            (df["tr_group"] == "IK")
            & (df["timepoint"] == 50)
            & (df["var"] == "knee_ROM")
            & (df["value"] > 50)
        )
    ]["all_labels"]
    IKfilt_knee2 = df[
        (
            (df["tr_group"] == "IK")
            & (df["timepoint"] == 100)
            & (df["var"] == "knee_ROM")
            & (df["value"] > -70)
        )
    ]["all_labels"]
    IKfilt_tor = df[
        (
            (df["tr_group"] == "IK")
            & (df["timepoint"] == 80)
            & (df["var"] == "torque")
            & (df["value"] < 50)
        )
    ]["all_labels"]

    # filter out
    filts = pd.concat(
        [
            NHfilt_hip1,
            NHfilt_hip2,
            NHfilt_hip3,
            NHfilt_knee1,
            NHfilt_knee2,
            NHfilt_knee3,
            NHfilt_fonset,
            NHfilt_f,
            NHfilt_tor,
            IKfilt_knee1,
            IKfilt_knee2,
            IKfilt_tor,
        ]
    )

    dfilt = df[~df["all_labels"].isin(filts)]

    # # set to NaN the filtered
    # df.loc[(df.all_labels == filts), 'value'] = np.nan
    # dfilt = df

    # # PRINT print stats of filtered reps
    # print('Total percentage amount of data kept: ', round((len(dfilt)*100)/len(df),2))

    # ff = pd.DataFrame(pd.concat([NHfilt_hip1, NHfilt_hip2, NHfilt_hip3, NHfilt_knee1, NHfilt_knee2, NHfilt_knee3, NHfilt_fonset]).unique())

    # splff = ff[0].str.split(expand=True)
    # splff.columns = ['par', 'tr', 'set', 'rep', 'var', 'tr_group']

    # for index, row in splff.iterrows():
    #     print(index, row)

    # for row in splff.itertuples():
    #     print(row.par)

    return dfilt


def excludeNH():
    """Set as NAs

    Returns:
        [type] -- [description]
    """

    return NHnas


import pandas as pd
import re
from typing import Dict, Any


def filter_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter training data according to specific criteria.

    This function applies several filters to the training data:
    1. Filters out specific combinations of training sessions and sets
    2. Filters out specific participant-session-set combinations
    3. Filters repetitions with knee velocity between 10 and 35 degrees/s
    4. Extracts numeric values from training session and set identifiers

    Args:
        data: DataFrame containing the training data with columns:
            - par: participant ID
            - trses: training session
            - set: set number
            - rep: repetition number
            - knee_v_mean: knee velocity

    Returns:
        Filtered DataFrame with additional columns:
            - trses_num: numeric training session number
            - set_num: numeric set number
    """
    print("Original training_disc shape:", data.shape)

    # 1. Filter out specific combinations of training sessions and sets
    session_set_filter = ~(
        ((data["trses"] == "tr_1") & data["set"].isin(["set_4", "set_5"]))
        | ((data["trses"] == "tr_4") & (data["set"] == "set_5"))
        | ((data["trses"] == "tr_5") & (data["set"] == "set_5"))
        | ((data["trses"] == "tr_10") & data["set"].isin(["set_6", "set_7"]))
        | ((data["trses"] == "tr_15") & (data["set"] == "set_7"))
    )
    data = data[session_set_filter]
    print("After filtering specific session-set combinations:", data.shape)

    # 2. Filter out specific participant-session-set combinations
    participant_filter = ~(
        (
            (data["par"] == "jhamon02")
            & (data["trses"] == "tr_9")
            & (data["set"] == "set_5")
        )
        | (
            (data["par"] == "jhamon03")
            & (data["trses"] == "tr_11")
            & (data["set"] == "set_4")
        )
        | (
            (data["par"] == "jhamon03")
            & (data["trses"] == "tr_10")
            & (data["set"] == "set_5")
            & (data["rep"] == "rep_1")
        )
        | (
            (data["par"] == "jhamon20")
            & (data["trses"] == "tr_14")
            & (data["set"] == "set_1")
        )
        | (
            (data["par"] == "jhamon20")
            & (data["trses"] == "tr_13")
            & (data["set"] == "set_5")
        )
        | (
            (data["par"] == "jhamon33")
            & (data["trses"] == "tr_15")
            & (data["set"] == "set_1")
        )
    )
    data = data[participant_filter]
    print(
        "After filtering specific participant-session-set combinations:",
        data.shape,
    )

    # 3. Filter to keep repetitions with knee velocity between 10 and 35
    data = data[(data["knee_v_mean"] > 10) & (data["knee_v_mean"] < 35)]

    # 4. Extract numeric values from identifiers
    data["trses_num"] = data["trses"].str.extract(r"(\d+)").astype(int)
    data["set_num"] = (
        data["set"].str.extract(r"set[_\s]*(\d+)", flags=re.IGNORECASE).astype(float)
    )

    # Print information about the filtered data
    print("Training sessions available:", sorted(data["trses_num"].unique()))
    print("Set numbers available:", sorted(data["set_num"].unique()))

    return data
