from numpy.ma import max


def dame_nht_data(training_sessions):
    """
    Import mechanical and motion capture data. Apply calibration (for force data)
    and filter data.

    Parameters:
        session (dict): A dictionary containing information about mechanical and
                        motion capture data files for a specific session. It should
                        be structured as follows:
                        {
                            'set_1': [['path_to_mech_data_file_1', 'path_to_motion_capture_data_file_1']],
                            'set_2': [['path_to_mech_data_file_2', 'path_to_motion_capture_data_file_2']],
                            ...
                        }

    Returns:
        dict: A dictionary containing all data synchronized and indexes segmenting
              repetitions for each set in the session. The structure of the
              returned dictionary is as follows:
              {
                  'set_1': [[repetition_indexes], [synchronized_data]],
                  'set_2': [[repetition_indexes], [synchronized_data]],
                  ...
              }

    Dependencies:
        This function requires the following modules to be imported:
        - numpy (np)
        - scipy (signal)
        - jhamon.signal.filters (butter_low_filter)
        - jhamon.signal.mech (_detect_onset)
        - The _llenahuecos function defined below.

    Notes:
        - This function is designed to process mechanical and motion capture data
          for a given session. It synchronizes the data and segmentates repetitions
          based on the mechanical force data.
        - The mechanical data should be in a specific format (txt file) and undergo
          calibration and filtering processes to convert the raw values into
          corresponding units (e.g., from volts to newtons).
        - The motion capture data should be in a specific CSV format.
        - The function internally uses the scipy.signal.butter function to design a
          low-pass Butterworth filter and jhamon.signal.filters.butter_low_filter to
          apply the filter to motion capture marker trajectories.
        - The function uses jhamon.signal.mech._detect_onset to segmentate repetitions
          based on a force threshold in the mechanical data.

    Example:
        >>> session_info = {
                'set_1': [['path_to_mech_data_file_1', 'path_to_motion_capture_data_file_1']],
                'set_2': [['path_to_mech_data_file_2', 'path_to_motion_capture_data_file_2']]
            }
        >>> session_data = damedata(session_info)
        >>> print(session_data['set_1'])  # Access the processed data for set 1
    """

    import os
    import fnmatch

    from jhamon_training.data.nordic import damedata, arregla_errores, dame_pesocorporal, analizame_curvas

    resultados_nordics = dict()
    for participant in training_sessions.keys():
        print('Vamos allá con las sesiones de: ' + participant)

        results_session = dict()
        for tr_session in training_sessions[participant].keys():
            print(str('Sesión de entrenamiento: ') + tr_session)

            data_path = training_sessions[participant][tr_session]
            files = os.listdir(data_path)

            mech_files = [x for x in fnmatch.filter(
                files, "*.txt") if 'readme' not in x]
            mech_paths = [data_path + "\\" + mech_files[ii]
                          for ii in range(len(mech_files))]

            kin_files = fnmatch.filter(files, '*Take *' and '*.csv')
            kin_paths = [data_path + "\\" + kin_files[ii]
                         for ii in range(len(kin_files))]

            # Create dict with session{'set_n': [ [mech], [optitrack] ]}
            session = dict()
            for ii in range(len(mech_paths)):
                session[str('set_' + str(ii+1))
                        ] = [[mech_paths[ii], kin_paths[ii]]]

            # Get data and indexes for each set of the current session
            session_data = damedata(session)

            # Evaluate known exceptions
            session_data = arregla_errores(
                session_data, participant, tr_session)

            # get body weight
            pesocorporal = dame_pesocorporal(
                participant, datapath=os.path.split(data_path)[0])

            # Get results
            results_session[tr_session] = analizame_curvas(session_data,
                                                           pesocorporal,
                                                           participant,
                                                           tr_session)

            resultados_nordics[participant] = results_session

    return resultados_nordics


def damedata(session):
    """
    Import mechanical and motion capture data.
    Apply calibration (for force data) and filter data.

    Parameters:
    --------------
    session:

    Returns:
    -------------
    Dict with all data synchronised and indexes segmenting repetitions.
    """

    import numpy as np
    from scipy import signal

    from jhamon.signal.filters import butter_low_filter

    session_data = dict()
    for jj in range(len(session)):
        serie = 'set_' + str(jj + 1)

        print('Toma data de la serie: ' + serie)

        # Function to convert ',' to '.' in the text file
        def conv(x):
            return x.replace(',', '.').encode()
        mech_raw = np.genfromtxt((conv(x) for x in open(
            session[serie][0][0])), delimiter=';', skip_header=2)

        # From volts to newtons
        mech_raw[:, 1] = ((mech_raw[:, 1])*-1 -
                          0.10632009202953419) / 0.005860458908676682
        mech_raw[:, 2] = ((mech_raw[:, 2])*-1 -
                          0.0476401060416749) / 0.006073002639830023

        # Use trigger chanel to cut mechanical data
        trigger_threshold = 1

        # get trigger indexes
        idx = np.argwhere(mech_raw[:, 3] > trigger_threshold)
        mech_data = mech_raw[idx[0, 0]:idx[-1, 0], [0, 1, 2]]

        # Get motion capture data
        mo_data = np.genfromtxt(session[serie][0][1], usecols=np.arange(
            20), skip_header=7, delimiter=',')

        # array with only x,y,z axes for each marker in CM
        markers = np.asarray(mo_data[:, 2:])

        # Fill gaps with quadratic interpolation
        time = mo_data[:, 1]
        markers_filled = np.zeros((len(markers), markers.shape[1]))
        for ii in range(markers.shape[1]):
            markers_filled[:, ii] = _llenahuecos(time, markers[:, ii])

        # Low pass filter marker trajectories
        from scipy.signal import butter, filtfilt
        freq = 100
        C = 0.802
        cut_hz = 6
        b, a = butter(2, (cut_hz / (freq / 2) / C), btype='low')
        markers_filt = np.transpose(np.asarray(
            [filtfilt(b, a, markers_filled[:, jj]) for jj in range(markers_filled.shape[1])]))

        # Downsample force data to markers
        mech_resampled = np.asarray(signal.resample(
            mech_data[:, 1:], len(markers_filt)))
        time = np.transpose(np.linspace(
            0, len(mech_resampled) / 100, num=len(mech_resampled)))
        DATA = np.column_stack((time, mech_resampled, markers_filt))

        # Segmentate repetitions
        from jhamon.signal.mech import _detect_onset
        force_threshold = DATA[:, 2].max() * 0.4
        dins = _detect_onset(DATA[:, 2], threshold=force_threshold,
                             n_above=100, n_below=0, threshold2=1, show=False)
        # thresholds not precise
        indexes = np.array([dins[:, 0] - 350, dins[:, 1] + 300])

        # If repetition 1 started without enough baseline time...
        if indexes[0, 0] < 0:
            indexes[0, 0] = 1

        # # PLOT segmented repetitions
        # fig, ax = plt.subplots(1)
        # ax.plot(DATA[:,1:3])
        # [ax.axvline(_x, linewidth=1, color='g', ls = '--') for _x in indexes[0,:]]
        # [ax.axvline(_x, linewidth=1, color='r', ls = '--') for _x in indexes[1,:]]

        session_data[serie] = [[indexes], [DATA]]

    return(session_data)


def _llenahuecos(time, y):
    """
    Fill gaps in a time series by performing linear interpolation.

    This function fills the gaps in the input time series `y` by using linear
    interpolation based on the available data points at corresponding `time`
    values. Any missing or invalid data (NaNs) in `y` will be replaced with
    interpolated values using linear extrapolation.

    Parameters:
        time (array_like): A 1-D array containing time values corresponding to
                           the data points in `y`.
        y (array_like): A 1-D array containing the time series data with gaps.

    Returns:
        array_like: A 1-D array containing the time series data with gaps
                    filled using linear interpolation.

    Dependencies:
        This function requires the following modules to be imported:
        - scipy
        - numpy

    Notes:
        - The input `time` and `y` arrays should have the same length.
        - The function internally uses scipy.interpolate.interp1d to perform
          the linear interpolation. The 'fill_value' parameter is set to
          'extrapolate' to handle data outside the range of the available
          points by performing linear extrapolation.

    Example:
        >>> time = [0, 1, 3, 6, 10]
        >>> y = [1, 2, np.nan, 4, np.nan]
        >>> filled_y = _llenahuecos(time, y)
        >>> print(filled_y)
        [1.  2.  3.  4.  4.]
    """
    import scipy
    import numpy as np

    idx = np.isfinite(y)

    # Create interpolation function without the
    f_interp = scipy.interpolate.interp1d(
        time[idx], y[idx], fill_value='extrapolate', kind='linear')
    ynew = f_interp(time)

    return ynew


def arregla_errores(session_data, participant, tr_session):
    """These are manual exeptions in some isolated repetitions where for some reason the
    criteria to segmentate signal does not apply. These have been manually checked and processed.
    """

    ###########################################################################
    if participant in ['jhamon05'] and tr_session in ['tr_1']:
        session_data['set_1'][0][0][0][0] = 100
        session_data['set_3'][0][0][0][0] = 100

    if participant in ['jhamon16'] and tr_session in ['tr_1']:
        session_data['set_1'][0][0][0][3] = 5150
        session_data['set_1'][0][0][0][4] = 6320

    ###########################################################################

    if participant in ['jhamon06'] and tr_session in ['tr_3']:
        session_data['set_1'][0][0][0][0] = 100
        session_data['set_4'][0][0][0][0] = 100

    # Exclude repetitions 4 and 5 5 where markers are not available
    if participant in ['jhamon09'] and tr_session in ['tr_3']:
        session_data['set_4'][0][0] = session_data['set_4'][0][0][:, (0, 1, 2)]

    ###########################################################################

    if participant in ['jhamon14'] and tr_session in ['tr_4']:
        session_data['set_1'][0][0][0][2] = 2730

    ###########################################################################

    if participant in ['jhamon06'] and tr_session in ['tr_5']:
        session_data['set_3'][0][0][0][4] = 6600
        session_data['set_4'][0][0][0][4] = 6500

    # Exclude repetition 5 where markers are not available
    if participant in ['jhamon15'] and tr_session in ['tr_5']:
        session_data['set_4'][0][0] = session_data['set_4'][0][0][:,
                                                                  (0, 1, 2, 3, 5)]

    ###########################################################################

    if participant in ['jhamon04'] and tr_session in ['tr_6']:
        session_data['set_1'][0][0] = session_data['set_1'][0][0][:, 1:]

    ###########################################################################
    if participant in ['jhamon04'] and tr_session in ['tr_7']:
        session_data['set_2'][0][0] = session_data['set_1'][0][0][:,
                                                                  (0, 2, 3, 4, 5)]

    ###########################################################################
    if participant in ['jhamon03'] and tr_session in ['tr_15']:
        session_data['set_5'][0][0] = session_data['set_5'][0][0][:,
                                                                  (0, 1, 2, 3, 5, 6, 7)]

    return(session_data)


def dame_pesocorporal(participant, datapath):
    """
    Get  weight value from 'antro.xlsx' file stored in each participant's path.

    Parameters:
    --------------
    participant: string of the name and code of participant (eg: 'jhamon01')

    Returns:
    -------------
    Peso: float64 value corresponding to the first weight available of the
    participant in 'antro.xlsx'

    """

    from pathlib import Path
    import pandas as pd
    # Create dictionary with all paths to training sessions
    path_antro = str(Path(datapath)) + '\\' + str('antro.xlsx')
    peso = pd.read_excel(path_antro).iloc[0, 1]

    return(peso)


def linear_tnorm(y, num_points=101, plot=False):

    import numpy as np
    import matplotlib.pyplot as plt

    """
    Linear time normalization from 0 to 100%.

    Parameters:
        y (array_like): 1-D array of data points to be interpolated.
        num_points (int, optional): Number of points at the interpolation (default is 101).
        plot (bool, optional): If True, plot the original data and the interpolated data (default is False).

    Returns:
        yn (ndarray): Interpolated data points.
        tn (ndarray): New x values (from 0 to 100) for the interpolated data.
    """
    y = np.asarray(y)

    # Original time points
    t = np.linspace(0, 100, y.size)

    # New time points
    tn = np.linspace(0, 100, num_points)

    # Perform linear interpolation
    yn = np.interp(tn, t, y)

    if plot:
        plt.plot(t, y, 'o-', label='Original Data')
        plt.plot(tn, yn, '.-', label='Interpolated Data')
        plt.xlabel('[%]')
        plt.legend()
        plt.grid(True)
        plt.show()

    return yn, tn


def analizame_curvas(session_data, pesocorporal, participant, tr_session):

    import numpy as np
    import pandas as pd
    from jhamon.signal.nordic import dame_inercia
    from jhamon.signal.filters import butter_low_filter
    from jhamon.signal.mech import _detect_onset
    from jhamon_training.kinematics import calculate_knee_velocity, calculate_hip_velocity

    results_session = dict()
    for serie in session_data.keys():
        print(serie)
        indexes = session_data[serie][0][0]
        DATA = session_data[serie][1][0]

        results = dict()
        for repetition in range(indexes.shape[1]):
            rep_num = 'rep_' + str(repetition + 1)
            print(rep_num)
            repe = DATA[indexes[0, repetition]:indexes[1, repetition], :]

            # Correct manually some different setup exceptions
            if participant in ['jhamon06'] and tr_session in ['tr_1', 'tr_2']:
                marker1 = np.array(repe[:, [4, 3]])
                marker2 = np.array(repe[:, [7, 6]])
                marker3 = np.array(repe[:, [10, 9]])
                marker4 = np.array(repe[:, [13, 12]])
                marker5 = np.array(repe[:, [16, 15]])
                marker6 = np.array(repe[:, [19, 18]])

            if participant in ['jhamon10'] and tr_session in ['tr_1']:
                marker1 = np.array(repe[:, [4, 3]])
                marker2 = np.array(repe[:, [7, 6]])
                marker3 = np.array(repe[:, [10, 9]])
                marker4 = np.array(repe[:, [13, 12]])
                marker5 = np.array(repe[:, [16, 15]])
                marker6 = np.array(repe[:, [19, 18]])

            else:
                marker1 = np.array(repe[:, [4, 5]])
                marker2 = np.array(repe[:, [7, 8]])
                marker3 = np.array(repe[:, [10, 11]])
                marker4 = np.array(repe[:, [13, 14]])
                marker5 = np.array(repe[:, [16, 17]])
                marker6 = np.array(repe[:, [19, 20]])

            knee_rad, knee_v_rad, knee_acc = calculate_knee_velocity(
                marker3, marker4, marker5, marker6, freq=100)

            hip_rad, hip_v_rad, hip_acc = calculate_hip_velocity(
                marker1, marker2, marker3, marker4, freq=100)

            # All data rep
            D = np.column_stack((repe[:-2, :3], knee_rad[:-2], hip_rad[:-2],
                                 knee_v_rad[:-1], hip_v_rad[:-1],
                                 knee_acc, hip_acc,
                                 marker1[:-2], marker2[:-2], marker3[:-2],
                                 marker4[:-2], marker5[:-2], marker6[:-2]))

            D[:, 0] = np.transpose(np.linspace(0, len(D) / 100, num=len(D)))

            # Flipped calibrations
            if participant in ['jhamon06'] and tr_session in ['tr_1', 'tr_2']:
                D[:, 5] = D[:, 5] * -1

            if participant in ['jhamon10'] and tr_session in ['tr_1']:
                D[:, 5] = D[:, 5] * -1

            # Find start of repetition from torque (filtered)
            force = D[:, 2]  # torque

            # Filter
            force_filt = butter_low_filter(force, fs=100, cut_hz=3, order=2)

            dif_force = np.diff(force_filt)
            onset_1 = _detect_onset(force_filt, max(
                force_filt[:]) * 0.25, n_above=80, n_below=0, show=False)[0, 0]
            onset_precise = dif_force[:onset_1 + 5] < 0

            if np.where(onset_precise)[0].shape[0] == 0:
                force_onset = 0
            else:
                force_onset = np.where(onset_precise)[0][-1]

            if force[force_onset] > max(force_filt[:]) * 0.25:
                force_onset = np.where(onset_precise)[0][-2]

            # Set a maximum index at which velocity is evaluated (i.e.
            # when force < max(force)*0.2)
            force_offset = _detect_onset(force_filt, max(force_filt[:]) * 0.3,
                                         n_above=80, n_below=0, show=False)[0, 1]
            vmax_idx = D[force_onset:force_offset + 25, 5].argmax() + \
                force_onset

            # manual fix
            if participant in ['jhamon09'] and tr_session in ['tr_3']:
                vmax_idx = D[force_onset:force_offset +
                             250, 5].argmax() + force_onset

            # Final data of segmented repetitions
            REP = D[force_onset:vmax_idx, :]
            knee_ROM = np.abs(REP[0, 3] - REP[-1, 3]) * 180 / np.pi
            hip_ROM = np.abs(REP[1, 4] - REP[-1, 4]) * 180 / np.pi

            # Distances between markers for torque calculations
            x1, y1, x2, y2 = (marker4[0, 0], marker4[0, 1],
                              marker6[0, 0], marker6[0, 1])

            shank_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            lever_arm = shank_length - 0.05

            if participant in ['jhamon12'] and tr_session in ['tr_5']:
                lever_arm = 0.32231839872197643

            # Calculate variables of interest and store them
            knee_v_mean = np.abs(REP[:, 5]).mean() * 180 / np.pi
            knee_v_peak = np.abs(REP[:, 5]).max() * 180 / np.pi
            hip_v_mean = np.abs(REP[:, 6]).mean() * 180 / np.pi
            hip_v_peak = np.abs(REP[:, 6]).max() * 180 / np.pi

            # Find angle at which velocity sudenly increases
            vmax_REP_idx = REP[:, 5].argmax()
            angDWA_idx = np.diff(REP[:vmax_REP_idx, 5]) < 0
            angDWA_idx2 = np.where(angDWA_idx)[0][-1]

            # angle at which velocity increases
            knee_angDWA = np.abs(REP[angDWA_idx2, 3] * 180 / np.pi)

            # % of ROM at which angDWA occurs
            knee_ROMDWA = (angDWA_idx2 * 100) / len(REP)

            # Angle at which peak force occurs
            fpeak_REP_idx = REP[:, 2].argmax()

            # angle at peak force
            knee_fpeak = np.abs(REP[fpeak_REP_idx, 3] * 180 / np.pi)
            knee_ROMfpeak = (fpeak_REP_idx * 100) / len(REP)

            # Torque, time, work
            knee_tor_mean = REP[:, 2].mean() * lever_arm
            knee_tor_peak = REP[:, 2].max() * lever_arm

            REP_time = len(REP) / freq  # in seconds
            knee_impulse = knee_tor_mean * REP_time
            knee_work = np.abs(
                np.trapz(REP[:, 2] * lever_arm, REP[:, 3]))  # Jules

            # Torque inertia corrected
            bm = pesocorporal
            ex_weight = 0
            I_knee = dame_inercia(REP, bm, ex_weight)

            # Apply correction factor to inertia, based on the force developed
            torque_corrected = np.zeros(len(REP))
            for ii in range(len(REP)):
                torque_measured = REP[ii, 2] * lever_arm
                torque_corrected[ii] = -torque_measured + \
                    ((I_knee[ii] * REP[ii, 7]) / 2)

            normcurves = dict()
            normcurves = {'force': linear_tnorm(REP[:, 2])[0],
                          'torque': linear_tnorm(REP[:, 2]*lever_arm)[0],
                          'knee_ROM': linear_tnorm((REP[:, 3] * 180 / np.pi))[0],
                          'hip_ROM': linear_tnorm((REP[:, 4] * 180 / np.pi))[0],
                          'knee_v': linear_tnorm((REP[:, 5] * 180 / np.pi))[0],
                          'hip_v': linear_tnorm((REP[:, 6] * 180 / np.pi))[0],
                          'knee_work': linear_tnorm(((REP[:, 2] * lever_arm)*REP[:, 3]))[0]}
            normcurvesdf = pd.DataFrame(data=normcurves)

            curves = dict()
            curves = {'force': REP[:, 2],
                      'torque': REP[:, 2] * lever_arm,
                      'knee_ROM': REP[:, 3] * 180 / np.pi,
                      'hip_ROM': REP[:, 4] * 180 / np.pi,
                      'knee_v': REP[:, 5] * 180 / np.pi,
                      'hip_v': REP[:, 6] * 180 / np.pi}
            curvesdf = pd.DataFrame(data=curves)

            # save all discrete results in a dict (vdict)
            vars_nam = ['REP_time', 'hip_ROM', 'hip_v_mean', 'hip_v_peak', 'knee_ROM', 'knee_v_mean',
                        'knee_v_peak', 'knee_angDWA', 'knee_ROMDWA', 'knee_fpeak',
                        'knee_ROMfpeak', 'knee_tor_mean', 'knee_tor_peak', 'knee_impulse', 'knee_work']

            vars_discrete = np.array((REP_time, hip_ROM, hip_v_mean, hip_v_peak,
                                      knee_ROM, knee_v_mean, knee_v_peak,
                                      knee_angDWA, knee_ROMDWA,
                                      knee_fpeak, knee_ROMfpeak,
                                      knee_tor_mean, knee_tor_peak,
                                      knee_impulse, knee_work))

            vdict = dict(zip(vars_nam, vars_discrete))

            # list with both discrete results and time normalized curves
            results[rep_num] = [vdict, normcurves]

            resultsdf = pd.DataFrame(results)

        #    # Plot to check onsets and vmx
        #    plotea_nordic(D, force_onset, vmax_idx, angDWA_idx2)

        results_session[serie] = resultsdf

    return(results_session)
