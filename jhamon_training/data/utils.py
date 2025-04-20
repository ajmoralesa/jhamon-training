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
