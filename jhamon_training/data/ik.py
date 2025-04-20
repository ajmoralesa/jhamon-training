def dame_ik_data(training_sessions):

    import os
    import numpy as np

    resultados_IK = dict()
    for participant in training_sessions.keys():
        # for participant in ['jhamon18', 'jhamon22']:

        results_session = dict()
        for tr_session in training_sessions[participant].keys():
            data_path = training_sessions[participant][tr_session]
            files = os.listdir(data_path)

            paths = _damegauche(data_path, files)

            sets = dict()
            for ii in np.arange(len(paths)):
                serie = "set_" + str(ii + 1)
                print(
                    "Amos allá con las sesiones de: "
                    + participant
                    + " "
                    + tr_session
                    + " "
                    + serie
                )

                rep = _segmentaIK(paths[ii], participant, tr_session, serie)
                sets[serie] = rep

            results_session[tr_session] = sets
        resultados_IK[participant] = results_session

    return resultados_IK


def _damegauche(data_path, files):

    filespaths = [data_path / str(files[ii]) for ii in range(len(files))]
    paths_droite = []
    for ii in range(len(filespaths)):
        file = open(filespaths[ii], "r", encoding="latin-1")
        file.readline()
        info = file.readline()
        file.close()
        if "Gauche" in info:
            paths_droite.append(filespaths[ii])

    return paths_droite


def _segmentaIK(ikpath, participant, tr_session, serie, tidx=0, vidx=1, pidx=2):

    import numpy as np
    from jhamon_training.signal.mech import _detect_onset
    from jhamon_training.utils import linear_tnorm

    data = gravitycorrect(ikpath)
    torque = data[:, tidx]
    velocity = data[:, vidx]

    onsets_1 = _detect_onset(
        torque, max(torque) * 0.25, n_above=80, n_below=0, show=False
    )[:, 0]

    # Skip first rep if not close to max
    if not np.max(torque[onsets_1[0] : onsets_1[0] + 1000]) > max(torque) * 0.8:
        onsets_1 = onsets_1[1:]

    onsets_1 = set_manual_onsets(participant, tr_session, serie, onsets_1)

    vonset_precise = []
    for jj in range(len(onsets_1)):
        idx = velocity[: onsets_1[jj] + 100] > 0
        vonset_precise.append(np.where(idx)[0][-1])

    voffset_precise = []
    for jj in range(len(onsets_1)):
        idx = velocity[vonset_precise[jj] : vonset_precise[jj] + 2000] < 0
        voffset_precise.append(np.where(idx)[0][-1] + vonset_precise[jj])

    # # Plot onsets
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(velocity)
    # ax.plot(torque)
    # for onset in vonset_precise:
    #     ax.axvline(x=onset, color='red', linestyle='--')
    # for offset in voffset_precise:
    #     ax.axvline(x=offset, color='green', linestyle='--')
    # plt.show()

    rep = dict()
    for ii in range(len(vonset_precise)):
        torq = data[vonset_precise[ii] : voffset_precise[ii], tidx]
        veloc = data[vonset_precise[ii] : voffset_precise[ii], vidx]
        angle = data[vonset_precise[ii] : voffset_precise[ii], pidx]

        rep["rep_" + str(ii + 1)] = {
            "torque": linear_tnorm(torq)[0],
            "knee_v": linear_tnorm(veloc)[0] * -1,
            "knee_ROM": linear_tnorm(angle)[0],
            "knee_work": linear_tnorm((torq * np.radians(angle)) * -1)[0],
        }

    return rep


def set_manual_onsets(participant, tr_session, serie, onsets_1):

    if participant in ["jhamon29"] and tr_session in ["tr_2"] and serie in ["set_1"]:
        onsets_1 = onsets_1[1:]

    if participant in ["jhamon18"] and tr_session in ["tr_12"] and serie in ["set_2"]:
        onsets_1 = onsets_1[2:]

    if participant in ["jhamon18"] and tr_session in ["tr_13"] and serie in ["set_3"]:
        onsets_1 = onsets_1[1:]

    if participant in ["jhamon20"] and tr_session in ["tr_12"] and serie in ["set_5"]:
        onsets_1 = onsets_1[4:]

    if participant in ["jhamon20"] and tr_session in ["tr_15"] and serie in ["set_1"]:
        onsets_1 = onsets_1[2:]

    if participant in ["jhamon23"] and tr_session in ["tr_1"] and serie in ["set_1"]:
        onsets_1 = onsets_1[2:]

    if participant in ["jhamon23"] and tr_session in ["tr_10"] and serie in ["set_1"]:
        onsets_1 = onsets_1[2:]

    if participant in ["jhamon23"] and tr_session in ["tr_7"] and serie in ["set_1"]:
        onsets_1 = onsets_1[2:]

    if participant in ["jhamon26"] and tr_session in ["tr_9"] and serie in ["set_4"]:
        onsets_1 = onsets_1[2:]

    if participant in ["jhamon28"] and tr_session in ["tr_1"] and serie in ["set_1"]:
        onsets_1 = onsets_1[3:]

    if participant in ["jhamon28"] and tr_session in ["tr_2"] and serie in ["set_4"]:
        onsets_1 = onsets_1[2:4]

    if participant in ["jhamon28"] and tr_session in ["tr_3"] and serie in ["set_2"]:
        onsets_1 = onsets_1[2:]

    if participant in ["jhamon28"] and tr_session in ["tr_5"] and serie in ["set_5"]:
        onsets_1 = onsets_1[2:]

    if participant in ["jhamon28"] and tr_session in ["tr_8"] and serie in ["set_5"]:
        onsets_1 = onsets_1[2:]

    if participant in ["jhamon28"] and tr_session in ["tr_9"] and serie in ["set_2"]:
        onsets_1 = onsets_1[2:]

    if participant in ["jhamon31"] and tr_session in ["tr_4"] and serie in ["set_5"]:
        onsets_1 = onsets_1[2:]

    if participant in ["jhamon32"] and tr_session in ["tr_1"] and serie in ["set_1"]:
        onsets_1 = onsets_1[2:]

    if participant in ["jhamon32"] and tr_session in ["tr_2"] and serie in ["set_2"]:
        onsets_1 = onsets_1[2:9]

    if participant in ["jhamon32"] and tr_session in ["tr_3"] and serie in ["set_1"]:
        onsets_1 = onsets_1[2:10]

    if participant in ["jhamon33"] and tr_session in ["tr_2"] and serie in ["set_2"]:
        onsets_1 = onsets_1[3:]

    if participant in ["jhamon33"] and tr_session in ["tr_15"] and serie in ["set_1"]:
        onsets_1 = onsets_1[3:]

    return onsets_1


def gravitycorrect(ikpath):
    """This function performs torque gravity correction. It takes two possible options:
    1) When anatomical 0 has been set at 0º and gravity correction points are located at the firts part of the 360º
    2) When anatomical 0 has been set at 360, and hence gravity correction info is located at the end.
    Depending on the case, gravity correction is added or substracted from the original torque signal. In jHamON project, eccentric contractions have been performed in a prone position and
    therefore passive torque should be ADDED (i.e. corrected torque signals should display greater peak values).


    Parameters
    ----------
    ikpath : type
        `ikpath` should be a path to a .CTM file containing torque data and gravity correction information.

    Returns
    -------
    type
        it returns `tcorrected`, torque gravity-corrected

    """
    import numpy as np
    from jhamon_training.signal.filters import butter_low_filter

    data = np.genfromtxt(ikpath, skip_header=100, delimiter="", encoding="latin-1")
    gdata = np.genfromtxt(
        ikpath, skip_header=79, max_rows=18, delimiter="", encoding="latin-1"
    )
    gdat = gdata[:, 2:].flatten()

    # condition to identify if data has been recorded at the beggining of gdat
    if abs(gdat[40]) > 0:
        # create new pos variable
        mypos = np.arange(len(np.argwhere(gdat))) * -1

        # leave out 10 points at each side to avoid extremes
        posk = mypos[15:-15]
        torok = (gdat[np.argwhere(gdat)] / 10)[15:-15].flatten()

        # Build polynomial
        gc = np.poly1d(np.polyfit(posk, torok, 2))
        tcorrected = data[:, 0] - gc(data[:, 2])

        # plt.plot(data[:, 0], c='r')
        # plt.plot(tcorrected, c='g')
        # plt.legend(('Raw', 'Gravity corrected'))
        # plt.show()

        data[:, 0] = tcorrected

    else:
        # create new pos variable
        mypos = np.arange(len(np.argwhere(gdat))) * -1

        # leave out 10 points at each side to avoid extremes
        posk = mypos[15:-15]
        torok = np.flip((gdat[np.argwhere(gdat)] / 10) * -1, axis=0)[15:-15].flatten()

        # Build polynomial
        gc = np.poly1d(np.polyfit(posk, torok, 2))
        tcorrected = data[:, 0] - gc(data[:, 2])

        # plt.plot(data[:, 0], c='r')
        # plt.plot(tcorrected, c='g')
        # plt.legend(('Raw', 'Gravity corrected'))
        # plt.show()

        data[:, 0] = tcorrected

    fs = 256
    cut_hz = 20
    order = 2
    data[:, 0] = butter_low_filter(data[:, 0], fs, cut_hz, order)
    data[:, 1] = butter_low_filter(data[:, 1], fs, cut_hz, order)
    data[:, 2] = butter_low_filter(data[:, 2], fs, cut_hz, order)

    return data
