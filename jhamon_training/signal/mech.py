import numpy as np


def _corrigegrav(position, torque, path):

    import fnmatch
    import os

    import numpy as np

    from .filters import butter_low_filter

    pass_file = fnmatch.filter(os.listdir(path), "passive_*")

    # import passive data
    passive_path = path / str(pass_file[0])
    passive_raw = np.genfromtxt(passive_path, delimiter=";", skip_header=2)
    passive_torque = butter_low_filter(passive_raw[:, 4], fs=2000, cut_hz=10, order=2)
    passive_position = butter_low_filter(passive_raw[:, 6], fs=2000, cut_hz=10, order=2)
    tor_pas = 69.7489 * (passive_torque) - 8.32223
    pos_pas = 36.10566 * (passive_position) - 5.57087 - 140
    pos_rad = pos_pas * 0.0174533

    # Perform actual gravity correction
    gc = np.poly1d(np.polyfit(pos_rad, tor_pas, 3))
    # plt.plot(pos_rad, tor_pas, '-', pos_rad, gc(pos_rad), '--')

    correction = gc(position * 0.0174533)
    # ax = plt.subplot(111)
    # ax.plot(torque, '--')
    # ax.plot(torque - correction)
    # plt.show()

    torque_corrected = torque - correction

    return torque_corrected


def _filcalibmech(force, pos):
    """Short summary.
    Parameters
    ----------
    force : type
        Description of parameter `force`.
    pos : type
        Description of parameter `pos`.
    Returns
    -------
    type
        Description of returned object.
    """
    import numpy as np
    from scipy.signal import butter, filtfilt

    freq = 2000
    C = 0.802
    cut_hz = 20
    b, a = butter(2, (cut_hz / (freq / 2) / C), btype="low")
    force_filt = np.transpose(filtfilt(b, a, force))
    pos_filt = np.transpose(filtfilt(b, a, pos))
    torque = 69.7489 * (force_filt) - 8.32223
    position = 36.10566 * (pos_filt) - 5.57087 - 140

    return torque, position


def _detect_peaks(
    x,
    mph=None,
    mpd=1,
    threshold=0,
    edge="rising",
    kpsh=False,
    valley=False,
    show=False,
    ax=None,
):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype("float64")
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ["rising", "both"]:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ["falling", "both"]:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[
            np.in1d(
                ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True
            )
        ]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (
                    x[ind[i]] > x[ind] if kpsh else True
                )
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plotpeaks(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plotpeaks(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available.")
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, "b", lw=1)
        if ind.size:
            label = "valley" if valley else "peak"
            label = label + "s" if ind.size > 1 else label
            ax.plot(
                ind,
                x[ind],
                "+",
                mfc=None,
                mec="r",
                mew=2,
                ms=8,
                label="%d %s" % (ind.size, label),
            )
            ax.legend(loc="best", framealpha=0.5, numpoints=1)
        ax.set_xlim(-0.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel("Data #", fontsize=14)
        ax.set_ylabel("Amplitude", fontsize=14)
        mode = "Valley detection" if valley else "Peak detection"
        ax.set_title(
            "%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
            % (mode, str(mph), mpd, str(threshold), edge)
        )
        # plt.grid()
        plt.show()


def _detect_onset(
    x,
    threshold=0,
    n_above=1,
    n_below=0,
    threshold2=None,
    n_above2=1,
    show=False,
    ax=None,
):
    """Detects onset in data based on amplitude threshold.
    Parameters
    ----------
    x : 1D array_like
        data.
    threshold : number, optional (default = 0)
        minimum amplitude of `x` to detect.
    n_above : number, optional (default = 1)
        minimum number of continuous samples >= `threshold`
        to detect (but see the parameter `n_below`).
    n_below : number, optional (default = 0)
        minimum number of continuous samples below `threshold` that
        will be ignored in the detection of `x` >= `threshold`.
    threshold2 : number or None, optional (default = None)
        minimum amplitude of `n_above2` values in `x` to detect.
    n_above2 : number, optional (default = 1)
        minimum number of samples >= `threshold2` to detect.
    show  : bool, optional (default = False)
        True (1) plots data in matplotlib figure, False (0) don't plot.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    inds : 2D array_like [indi, indf]
        initial and final indeces of the onset events.
    Notes
    -----
    You might have to tune the parameters according to the signal-to-noise
    characteristic of the data.
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectOnset.ipynb
    Examples
    --------
    >>> from detect_onset import detect_onset
    >>> x = np.random.randn(200)/10
    >>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
    >>> detect_onset(x, np.std(x[:50]), n_above=10, n_below=0, show=True)
    >>> x = np.random.randn(200)/10
    >>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
    >>> x[80:140:20] = 0
    >>> detect_onset(x, np.std(x[:50]), n_above=10, n_below=0, show=True)
    >>> x = np.random.randn(200)/10
    >>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
    >>> x[80:140:20] = 0
    >>> detect_onset(x, np.std(x[:50]), n_above=10, n_below=2, show=True)
    >>> x = [0, 0, 2, 0, np.nan, 0, 2, 3, 3, 0, 1, 1, 0]
    >>> detect_onset(x, threshold=1, n_above=1, n_below=0, show=True)
    >>> x = np.random.randn(200)/10
    >>> x[11:41] = np.ones(30)*.3
    >>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
    >>> x[80:140:20] = 0
    >>> detect_onset(x, .1, n_above=10, n_below=1, show=True)
    >>> x = np.random.randn(200)/10
    >>> x[11:41] = np.ones(30)*.3
    >>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
    >>> x[80:140:20] = 0
    >>> detect_onset(x, .4, n_above=10, n_below=1, show=True)
    >>> x = np.random.randn(200)/10
    >>> x[11:41] = np.ones(30)*.3
    >>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
    >>> x[80:140:20] = 0
    >>> detect_onset(x, .1, n_above=10, n_below=1,
                     threshold2=.4, n_above2=5, show=True)
    Version history
    ---------------
    '1.0.6':
        Deleted 'from __future__ import'
        added parameters `threshold2` and `n_above2`
    """

    x = np.atleast_1d(x).astype("float64")
    # deal with NaN's (by definition, NaN's are not greater than threshold)
    x[np.isnan(x)] = -np.inf
    # indices of data greater than or equal to threshold
    inds = np.nonzero(x >= threshold)[0]
    if inds.size:
        # initial and final indexes of almost continuous data
        inds = np.vstack(
            (
                inds[np.diff(np.hstack((-np.inf, inds))) > n_below + 1],
                inds[np.diff(np.hstack((inds, np.inf))) > n_below + 1],
            )
        ).T
        # indexes of almost continuous data longer than or equal to n_above
        inds = inds[inds[:, 1] - inds[:, 0] >= n_above - 1, :]
        # minimum amplitude of n_above2 values in x to detect
        if threshold2 is not None and inds.size:
            idel = np.ones(inds.shape[0], dtype=bool)
            for i in range(inds.shape[0]):
                if (
                    np.count_nonzero(x[inds[i, 0] : inds[i, 1] + 1] >= threshold2)
                    < n_above2
                ):
                    idel[i] = False
            inds = inds[idel, :]
    if not inds.size:
        inds = np.array([])  # standardize inds shape for output
    if show and x.size > 1:
        _plotonsets(x, threshold, n_above, n_below, threshold2, n_above2, inds, ax)

    return inds


def _plotonsets(x, threshold, n_above, n_below, threshold2, n_above2, inds, ax):
    """Plot results of the detect_onset function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available.")
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        if inds.size:
            for indi, indf in inds:
                if indi == indf:
                    ax.plot(indf, x[indf], "ro", mec="r", ms=6)
                else:
                    ax.plot(range(indi, indf + 1), x[indi : indf + 1], "r", lw=1)
                    ax.axvline(x=indi, color="b", lw=1, ls="--")
                ax.axvline(x=indf, color="b", lw=1, ls="--")
            inds = np.vstack(
                (np.hstack((0, inds[:, 1])), np.hstack((inds[:, 0], x.size - 1)))
            ).T
            for indi, indf in inds:
                ax.plot(range(indi, indf + 1), x[indi : indf + 1], "k", lw=1)
        else:
            ax.plot(x, "k", lw=1)
            ax.axhline(y=threshold, color="r", lw=1, ls="-")

        ax.set_xlim(-0.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel("Data #", fontsize=14)
        ax.set_ylabel("Amplitude", fontsize=14)
        if threshold2 is not None:
            text = (
                "threshold=%.3g, n_above=%d, n_below=%d, threshold2=%.3g, n_above2=%d"
            )
        else:
            text = "threshold=%.3g, n_above=%d, n_below=%d, threshold2=%r, n_above2=%d"
        ax.set_title(text % (threshold, n_above, n_below, threshold2, n_above2))
        # plt.grid()
        plt.show()


def _segmentame(signal, show=True, ax=None):

    peaks = _detect_peaks(signal, mph=-45, mpd=1000, show=False)
    onsts = _detect_onset(signal, threshold=-102, n_above=1000, n_below=0, show=False)
    onsets = onsts[:, 0]

    # # make sure that no other onsets than the real reps are included
    # if len(onsets) != len(peaks):
    #     onsets = onsets[1:]

    # Take onsets and peaks to identify precise ONSETS and OFFSETS
    OKnsets = np.zeros(len(onsets), dtype="int64")
    for ii in range(len(onsets)):
        cacho = np.argwhere(np.diff(signal[onsets[ii] - 1000 : onsets[ii] + 1000]) < 0)
        if len(cacho) > 0:
            OKnsets[ii] = cacho[-1] + (onsets[ii] - 1000)

    OFsets = np.zeros(len(OKnsets), dtype="int64")
    for ii in range(len(OKnsets)):
        cacho = np.argwhere(np.diff(signal[OKnsets[ii] + 1 : OKnsets[ii] + 3500]) < 0)
        OFsets[ii] = cacho[0] + OKnsets[ii]

    if OKnsets[0] == 0:
        OKnsets = OKnsets[1:]
        OFsets = OFsets[1:]

    # don't waste my time ploting one datum
    if show and signal.size > 1:
        _plotsegments(signal, OKnsets, OFsets, ax)

    return (OKnsets, OFsets)


def _plotsegments(signal, onsets, ofsets, ax):
    """Plot signal with onsets and offsets."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available.")
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

            ax.plot(signal)
            for xc in onsets:
                ax.axvline(x=xc)
            for xc in ofsets:
                ax.axvline(x=xc, c="r")
        plt.show()


def getorque(pth, fil, freq=2000):

    import numpy as np
    from jhamon.signal.tnorm import tnorm

    mech_path = pth + fil

    raw_mech = np.genfromtxt(mech_path, delimiter=";", skip_header=2)
    f = raw_mech[:, 4]
    pos = raw_mech[:, 6]

    torque, position = _filcalibmech(f, pos)
    torque_gcorrected = _corrigegrav(position, torque, pth)

    # find start and end of contraction
    OKnsets, OFsets = _segmentame(signal=position, show=False)

    # use trigger to sync data
    trig = np.argwhere(raw_mech[:, -1] > 0.9)[0][0]

    # Index at half a sencond back from OKnsets
    onset_back = int(OKnsets[0] - (freq * 0.5))

    # in seconds
    onset_backt = int((OKnsets[0] - (freq * 0.5)) / freq)

    # time difference between the onset back and the trigger
    on_trig_dift = (onset_back - trig) / freq

    mech = dict()
    mech["torqueback"] = torque_gcorrected[onset_back : OFsets[0]]
    mech["torque"] = torque_gcorrected[onset_back : OFsets[0]]
    mech["position"] = position[onset_back : OFsets[0]]
    mech["trigger"] = raw_mech[:, -1]

    # normalize torque
    mech["ntorque"] = tnorm(mech["torque"], show=False)[0]

    # !! jhamon27 POST onsets are not OK, manual exception should be done
    idxs = {
        "OKnset": OKnsets[0],
        "OFset": OFsets[0],
        "trigger": trig,
        "onset_back": onset_back,
        "onset_backt": onset_backt,
        "on_trig_dift": on_trig_dift,
    }

    return mech, idxs


def getMVC(mvc_path, pth, freq=2000):

    import numpy as np
    from jhamon.signal.filters import butter_low_filter, filtemg

    from jhamon.signal.mech import _filcalibmech, _corrigegrav
    from jhamon.signal.emg import getEMG, _window_rms

    mvc_raw = np.genfromtxt(mvc_path, delimiter=";", skip_header=2)

    # avoid analysing tests that did not include EMG recordings
    if mvc_raw.shape[1] == 7 or mvc_raw.shape[1] == 8:

        MVIC = dict()
        # mechanical data
        f = mvc_raw[:, 4]
        pos = mvc_raw[:, 6]
        torque, position = _filcalibmech(f, pos)
        torque_gcorrected = _corrigegrav(position, torque, pth)
        MVIC["mvf"] = np.array([max(torque_gcorrected)])
        mvf_idx = np.argmax(torque_gcorrected)

        # EMG data
        EMG = mvc_raw[:, 1:4]
        EMG_filt = filtemg(EMG)

        EMG_RMS = []
        for muscle in np.arange(EMG_filt.shape[1]):
            EMG_RMS.append(
                _window_rms(EMG_filt[:, muscle], window_size=int(freq * 0.1))
            )
        EMG_RMS = np.array(EMG_RMS)

        idx1 = int(mvf_idx - (freq * 0.1))
        idx2 = int(mvf_idx + (freq * 0.1))

        MVIC["SM_rms"] = np.array([np.mean(EMG_RMS[0, idx1:idx2])])
        MVIC["ST_rms"] = np.array([np.mean(EMG_RMS[1, idx1:idx2])])
        MVIC["BF_rms"] = np.array([np.mean(EMG_RMS[2, idx1:idx2])])

    return MVIC
