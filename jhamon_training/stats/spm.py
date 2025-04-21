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
        dfavg = df_filt.groupby(grpvars)["value"].mean().reset_index()

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
        spmdict_ok[t] = {"z": spmdict[t]["aov"].z, "zstar": spmdict[t]["aov"].zstar}

    return spmdict_ok
