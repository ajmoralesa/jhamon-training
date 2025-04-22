def dame_manual_exclusions(df):

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
