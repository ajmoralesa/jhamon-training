# make multiple comparisons between tr_1 and the rest of sessions


def spm_aov_trevolution(df):

    import spm1d
    import numpy as np
    import pandas as pd

    trses = [
        "tr_2",
        "tr_3",
        "tr_4",
        "tr_5",
        "tr_6",
        "tr_7",
        "tr_8",
        "tr_9",
        "tr_10",
        "tr_11",
        "tr_12",
        "tr_13",
        "tr_14",
        "tr_15",
    ]

    spmdict = dict()

    tr_compar = []
    for tr in trses:
        print(tr)

        tr_compar = (df["trses"] == "tr_1") | (df["trses"] == tr)
        datadf = df[tr_compar]

        # ANOVA
        # create subject, training group and time (PRE, POST) condition from column names
        datadf_wide = datadf[["timepoint", "all_labels", "value"]].pivot(
            index="timepoint", columns="all_labels", values="value"
        )
        splt_cols = [i.split(" ", 6) for i in datadf_wide.columns]

        parnames = []
        trgroup = []
        testime = []
        for v in np.arange(len(splt_cols)):
            parnames.append(splt_cols[v][0])
            trgroup.append(splt_cols[v][5])
            testime.append(splt_cols[v][1])

        Subjects = pd.factorize(pd.DataFrame(parnames)[0])[0]
        TestingTime = pd.factorize(pd.DataFrame(testime)[0])[0]
        GroupTraining = pd.factorize(pd.DataFrame(trgroup)[0])[0]

        # array of arrays with all signals
        Y = np.array(datadf_wide).transpose()

        # Two-way repeated-measures ANOVA with repeated-measures on one factor
        F = spm1d.stats.anova2onerm(Y=Y, A=GroupTraining, B=TestingTime, SUBJ=Subjects)
        F_tgroup = F[0].inference()
        F_time = F[1].inference()
        F_interaction = F[2].inference()
        anova = {"F_tgroup": F_tgroup, "F_time": F_time, "F_interaction": F_interaction}

        spmdict[str(tr)] = anova

    # get spmdict to dataframe
    spmdict_ok = dict()
    for t in spmdict.keys():
        spmdict_ok[t] = {
            "F_time": {
                "z": spmdict[t]["F_time"].z,
                "zstar": spmdict[t]["F_time"].zstar,
            },
            "F_group": {
                "z": spmdict[t]["F_tgroup"].z,
                "zstar": spmdict[t]["F_tgroup"].zstar,
            },
            "F_interaction": {
                "z": spmdict[t]["F_interaction"].z,
                "zstar": spmdict[t]["F_interaction"].zstar,
            },
        }

    return spmdict_ok


def spm_group_comparison(df):

    import spm1d
    import numpy as np

    mdt = (
        df.groupby(["timepoint", "par", "trses", "tr_group"])["value"]
        .mean()
        .reset_index()
    )

    mdt["all_labels"] = mdt.apply(lambda x: x.par + " " + x.trses, axis=1)

    # NH group
    datadf_wideNH = mdt[(mdt["tr_group"] == "NH")][
        ["timepoint", "all_labels", "value"]
    ].pivot(index="timepoint", columns="all_labels", values="value")
    Y_NH = np.array(datadf_wideNH).transpose()

    # IK group
    datadf_wideIK = mdt[(mdt["tr_group"] == "IK")][
        ["timepoint", "all_labels", "value"]
    ].pivot(index="timepoint", columns="all_labels", values="value")
    Y_IK = np.array(datadf_wideIK).transpose()

    alpha = 0.05
    t = spm1d.stats.ttest2(Y_NH, Y_IK, equal_var=False)
    ti = t.inference(alpha, two_tailed=True)

    spmdict = {"t": t, "ti": ti}

    return spmdict


def spm_nh_kinetics(df):

    import spm1d
    import numpy as np
    import pandas as pd

    trses = [
        "tr_2",
        "tr_3",
        "tr_4",
        "tr_5",
        "tr_6",
        "tr_7",
        "tr_8",
        "tr_9",
        "tr_10",
        "tr_11",
        "tr_12",
        "tr_13",
        "tr_14",
        "tr_15",
    ]

    spmdict = dict()

    tr_compar = []
    for tr in trses:
        print(tr)

        tr_compar = (df["trses"] == "tr_1") | (df["trses"] == tr)
        datadf = df[tr_compar]

        parlist = datadf.groupby(["par"])["trses"].unique().to_frame()
        parlist["subjects"] = parlist.index
        parlist.reset_index(drop=True)
        exclude_parts = parlist[parlist["trses"].map(len) < 2].index.tolist()

        for p in exclude_parts:
            print("PRE or POST measurement missing, removing: ", p, tr)

        df_filt = datadf[~datadf["par"].isin(exclude_parts)]

        # make sure you have a BALANCED DESING: average across repetitions
        grpvars = ["timepoint", "par", "trses"]
        dfavg = df_filt.groupby(grpvars).mean().reset_index()

        colu = ["par", "trses"]
        dfavg["reps_labels"] = dfavg[colu].apply(lambda x: " ".join(x), axis=1)

        # ANOVA
        datadf_wide = dfavg[["timepoint", "reps_labels", "value"]].pivot(
            index="timepoint", columns="reps_labels", values="value"
        )
        splt_cols = [i.split(" ", 2) for i in datadf_wide.columns]

        parnames = []
        testime = []
        for v in np.arange(len(splt_cols)):
            parnames.append(splt_cols[v][0])
            testime.append(splt_cols[v][1])

        Subjects = pd.factorize(pd.DataFrame(parnames)[0])[0]
        TestingTime = pd.factorize(pd.DataFrame(testime)[0])[0]
        Y = np.array(datadf_wide).transpose()

        # Check balanced design and remove subject if needed
        ybin = np.bincount(Subjects)
        ii = np.nonzero(ybin)[0]
        count_check = list(zip(ii, ybin[ii]))

        if np.argwhere(ybin[ii] < 2):
            print("One participants has ", min(ybin), " repetitions")

        # One-way repeated-measures ANOVA
        F = spm1d.stats.anova1rm(Y=Y, A=TestingTime, SUBJ=Subjects, equal_var=True)
        Finf = F.inference()
        anova = {"aov": Finf}

        spmdict[str(tr)] = anova

    # get spmdict to dataframe
    spmdict_ok = dict()
    for t in spmdict.keys():
        spmdict_ok[t] = {
            "z": spmdict[t]["aov"].z,
            "zstar": spmdict[t]["aov"].zstar,
            "ti": spmdict[t]["aov"].ti,
        }

    return spmdict_ok
