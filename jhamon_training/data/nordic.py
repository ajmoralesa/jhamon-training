from numpy.ma import max
import logging
from typing import Optional, Union, List
from joblib import Parallel, delayed  # Import joblib
import os
import fnmatch

from jhamon_training.data import frames


# Define the worker function to process a single participant
def _process_participant_nht(participant, participant_sessions):
    """
    Processes NHT data for a single participant.

    Args:
        participant (str): The participant ID.
        participant_sessions (dict): Dictionary of training sessions for this participant.
                                     Example: {'tr_1': PosixPath('/path/to/tr_1'), ...}

    Returns:
        tuple: A tuple containing (participant_id, participant_results_dict)
               or None if processing fails for this participant.
               participant_results_dict structure: {session_name: {set_name: [[indexes], [DATA]]}}
    """
    results_session = dict()
    base_data_path = None  # To store the base path for dame_pesocorporal

    for tr_session, data_path in participant_sessions.items():
        # Store the base path (assuming all sessions share the same parent for antro.xlsx)
        if base_data_path is None:
            # Get the parent directory of the session data path
            try:
                # Ensure data_path is converted to string if it's a Path object
                base_data_path = os.path.split(str(data_path))[0]
            except Exception as e:
                logging.error(
                    f"Could not determine base path for {participant} / {tr_session}: {e}"
                )
                continue  # Skip session if base path cannot be determined

        try:
            # print(str(participant + '; Sesión de entrenamiento: ') + tr_session) # Optional: keep for debugging if needed
            files = os.listdir(data_path)
            visible_files = [f for f in files if not f.startswith(".")]

            mech_files = [
                x for x in fnmatch.filter(visible_files, "*.txt") if "readme" not in x
            ]
            mech_paths = [data_path / mech_files[ii] for ii in range(len(mech_files))]

            kin_files = fnmatch.filter(visible_files, "*Take *" and "*.csv")
            kin_paths = [data_path / kin_files[ii] for ii in range(len(kin_files))]

            num_mech = len(mech_paths)
            num_kin = len(kin_paths)
            num_pairs = min(num_mech, num_kin)

            if num_mech != num_kin:
                logging.warning(
                    f"Mismatch in file counts for {participant} / {tr_session}: "
                    f"{num_mech} mechanical files vs {num_kin} kinematic files. "
                    f"Processing {num_pairs} pairs."
                )

            session = dict()
            for ii in range(num_pairs):
                set_name = f"set_{ii + 1}"
                session[set_name] = [[mech_paths[ii], kin_paths[ii]]]

            if not session:
                logging.warning(
                    f"No valid mech/kin pairs found for {participant} / {tr_session}. Skipping session analysis."
                )
                continue

            session_data = damedata(session)
            session_data = arregla_errores(session_data, participant, tr_session)

            # Make sure base_data_path was successfully determined
            if base_data_path is None:
                logging.error(
                    f"Cannot get peso corporal for {participant} / {tr_session} due to missing base path."
                )
                continue  # Skip session if we don't have the path for antro.xlsx

            pesocorporal = dame_pesocorporal(participant, datapath=base_data_path)

            results_session[tr_session] = analizame_curvas(
                session_data, pesocorporal, participant, tr_session
            )
        except Exception as e:
            logging.error(
                f"Error processing {participant} / {tr_session}: {e}", exc_info=True
            )  # Log traceback
            # Optionally continue to next session or return None for participant
            continue  # Continue to the next session for this participant

    if (
        not results_session
    ):  # If no sessions were successfully processed for this participant
        return None

    return participant, results_session


def dame_nht_data(
    training_sessions, participant_id: Optional[Union[str, List[str]]] = None
):
    """
    Import mechanical and motion capture data. Apply calibration (for force data)
    and filter data. Processes participants in parallel.

    Parameters:
        training_sessions (dict): A dictionary where keys are participant IDs and
                                  values are dictionaries mapping training session
                                  names (e.g., 'tr_1') to data paths.
        participant_id (Optional[Union[str, List[str]]]): The specific participant ID(s)
                                                        to process. Can be a single
                                                        string or a list of strings.
                                                        If None, process all participants.
                                                        Defaults to None.

    Returns:
        dict: A dictionary containing processed data for the specified participant(s).
              Structure: {participant_id: {session_name: {set_name: [[indexes], [DATA]]}}}

    Dependencies:
        - joblib (for parallel processing)
        - Requires helper functions: damedata, arregla_errores, dame_pesocorporal, analizame_curvas
        - numpy, scipy, jhamon modules (as used by helper functions)
        - os, fnmatch, logging, typing
    """

    resultados_nordics = dict()

    if participant_id is None:
        # Ensure all keys in training_sessions are strings if needed, or handle non-string keys
        participants_to_process = list(training_sessions.keys())
    elif isinstance(participant_id, str):
        # Ensure the participant exists in training_sessions
        if participant_id not in training_sessions:
            logging.warning(
                f"Participant ID {participant_id} not found in training_sessions. Skipping."
            )
            return {}  # Return empty dict if the single participant is not found
        participants_to_process = [participant_id]
    elif isinstance(participant_id, list):
        # Filter the list to include only participants present in training_sessions
        original_count = len(participant_id)
        participants_to_process = [p for p in participant_id if p in training_sessions]
        if len(participants_to_process) < original_count:
            missing_participants = set(participant_id) - set(participants_to_process)
            logging.warning(
                f"Participants not found in training_sessions and will be skipped: {missing_participants}"
            )
        if not participants_to_process:
            logging.warning(
                "None of the specified participants were found in training_sessions."
            )
            return {}  # Return empty if no specified participants are found
    else:
        raise TypeError("participant_id must be None, a string, or a list of strings")

    if not participants_to_process:
        logging.warning("No participants selected for processing.")
        return {}

    # Use joblib to parallelize the loop over participants
    # n_jobs=-1 uses all available CPU cores. Adjust if needed.
    # backend="loky" is generally robust. Consider "multiprocessing" if loky causes issues.
    # Set prefer='processes' for CPU-bound tasks like this.
    print(f"Processing {len(participants_to_process)} participants in parallel...")
    try:
        results_list = Parallel(n_jobs=-1, backend="loky", prefer="processes")(
            delayed(_process_participant_nht)(
                participant, training_sessions[participant]
            )
            for participant in participants_to_process
        )
    except Exception as e:
        logging.error(f"Parallel processing failed: {e}", exc_info=True)
        # Optionally, implement a fallback to sequential processing here
        print("Parallel processing failed. Falling back to sequential processing...")
        results_list = []
        for participant in participants_to_process:
            try:
                result = _process_participant_nht(
                    participant, training_sessions[participant]
                )
                results_list.append(result)
            except Exception as inner_e:
                logging.error(
                    f"Error processing participant {participant} sequentially: {inner_e}",
                    exc_info=True,
                )
                results_list.append(
                    None
                )  # Add None if sequential processing for one participant fails

    # Combine results from the parallel (or sequential fallback) processes
    for result in results_list:
        if result:  # Check if the result is not None (i.e., processing was successful)
            p_id, p_data = result
            if p_data:  # Ensure participant data is not empty
                resultados_nordics[p_id] = p_data
            else:
                logging.warning(f"No data processed for participant {p_id}, skipping.")

    # Remove empty repetitions or sessions
    if resultados_nordics:  # Only run if there are results
        try:
            # Make sure frames.remove_empty handles the structure correctly
            resultados_nordics = frames.remove_empty(resultados_nordics)
        except Exception as e:
            logging.error(f"Error during frames.remove_empty: {e}", exc_info=True)
            # Decide how to proceed: return potentially unclean data or raise error

    print("Finished processing participants.")
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
        serie = "set_" + str(jj + 1)

        print("Toma data de la serie: " + serie)

        # Function to convert ',' to '.' in the text file
        def conv(x):
            return x.replace(",", ".").encode()

        mech_raw = np.genfromtxt(
            (conv(x) for x in open(session[serie][0][0], encoding="latin-1")),
            delimiter=";",
            skip_header=2,
        )

        # From volts to newtons
        mech_raw[:, 1] = (
            (mech_raw[:, 1]) * -1 - 0.10632009202953419
        ) / 0.005860458908676682
        mech_raw[:, 2] = (
            (mech_raw[:, 2]) * -1 - 0.0476401060416749
        ) / 0.006073002639830023

        # Use trigger chanel to cut mechanical data
        trigger_threshold = 1

        # get trigger indexes
        idx = np.argwhere(mech_raw[:, 3] > trigger_threshold)
        mech_data = mech_raw[idx[0, 0] : idx[-1, 0], [0, 1, 2]]

        # Get motion capture data
        mo_data = np.genfromtxt(
            session[serie][0][1], usecols=np.arange(20), skip_header=7, delimiter=","
        )

        # array with only x,y,z axes for each marker in CM
        markers = np.asarray(mo_data[:, 2:])

        # Fill gaps with linear interpolation
        time = mo_data[:, 1]
        markers_filled = np.zeros((len(markers), markers.shape[1]))
        for ii in range(markers.shape[1]):
            markers_filled[:, ii] = _llenahuecos(time, markers[:, ii])

        # Low pass filter marker trajectories
        from scipy.signal import butter, filtfilt

        freq = 100
        C = 0.802
        cut_hz = 6
        b, a = butter(2, (cut_hz / (freq / 2) / C), btype="low")
        markers_filt = np.transpose(
            np.asarray(
                [
                    filtfilt(b, a, markers_filled[:, jj])
                    for jj in range(markers_filled.shape[1])
                ]
            )
        )

        # Downsample force data to markers
        mech_resampled = np.asarray(
            signal.resample(mech_data[:, 1:], len(markers_filt))
        )
        time = np.transpose(
            np.linspace(0, len(mech_resampled) / 100, num=len(mech_resampled))
        )
        DATA = np.column_stack((time, mech_resampled, markers_filt))

        # Segmentate repetitions
        from jhamon.signal.mech import _detect_onset

        force_threshold = DATA[:, 2].max() * 0.4
        dins = _detect_onset(
            DATA[:, 2],
            threshold=force_threshold,
            n_above=100,
            n_below=0,
            threshold2=1,
            show=False,
        )
        # thresholds not precise
        indexes = np.array([dins[:, 0] - 350, dins[:, 1] + 300])

        # If repetition 1 started without enough baseline time set index as 1
        if indexes[0, 0] < 0:
            indexes[0, 0] = 1

        # # PLOT segmented repetitions
        # fig, ax = plt.subplots(1)
        # ax.plot(DATA[:,1:3])
        # [ax.axvline(_x, linewidth=1, color='g', ls = '--') for _x in indexes[0,:]]
        # [ax.axvline(_x, linewidth=1, color='r', ls = '--') for _x in indexes[1,:]]

        session_data[serie] = [[indexes], [DATA]]

    return session_data


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
        time[idx], y[idx], fill_value="extrapolate", kind="linear"
    )
    ynew = f_interp(time)

    return ynew


def arregla_errores(session_data, participant, tr_session):
    """These are manual exeptions in some isolated repetitions where for some reason the
    criteria to segmentate signal does not apply. These have been manually checked and processed.
    """

    ###########################################################################
    if participant in ["jhamon05"] and tr_session in ["tr_1"]:
        session_data["set_1"][0][0][0][0] = 100
        session_data["set_3"][0][0][0][0] = 100

    if participant in ["jhamon16"] and tr_session in ["tr_1"]:
        session_data["set_1"][0][0][0][3] = 5150
        session_data["set_1"][0][0][0][4] = 6320

    ###########################################################################

    if participant in ["jhamon06"] and tr_session in ["tr_3"]:
        session_data["set_1"][0][0][0][0] = 100
        session_data["set_4"][0][0][0][0] = 100

    # Exclude repetitions 4 and 5 5 where markers are not available
    if participant in ["jhamon09"] and tr_session in ["tr_3"]:
        session_data["set_4"][0][0] = session_data["set_4"][0][0][:, (0, 1, 2)]

    ###########################################################################

    if participant in ["jhamon14"] and tr_session in ["tr_4"]:
        session_data["set_1"][0][0][0][2] = 2730

    ###########################################################################

    if participant in ["jhamon06"] and tr_session in ["tr_5"]:
        session_data["set_3"][0][0][0][4] = 6600
        session_data["set_4"][0][0][0][4] = 6500

    # Exclude repetition 5 where markers are not available
    if participant in ["jhamon15"] and tr_session in ["tr_5"]:
        session_data["set_4"][0][0] = session_data["set_4"][0][0][:, (0, 1, 2, 3, 5)]

    ###########################################################################

    if participant in ["jhamon04"] and tr_session in ["tr_6"]:
        session_data["set_1"][0][0] = session_data["set_1"][0][0][:, 1:]

    ###########################################################################
    if participant in ["jhamon04"] and tr_session in ["tr_7"]:
        session_data["set_2"][0][0] = session_data["set_1"][0][0][:, (0, 2, 3, 4, 5)]

    ###########################################################################
    if participant in ["jhamon03"] and tr_session in ["tr_15"]:
        session_data["set_5"][0][0] = session_data["set_5"][0][0][
            :, (0, 1, 2, 3, 5, 6, 7)
        ]

    return session_data


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

    # Create path to antro.xlsx using pathlib for cross-platform compatibility
    path_antro = Path(datapath) / "antro.xlsx"
    peso = pd.read_excel(path_antro).iloc[0, 1]

    return peso


def analizame_curvas(session_data, pesocorporal, participant, tr_session):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler()],
    )

    import numpy as np
    import pandas as pd
    from jhamon.signal.nordic import dame_inercia
    from jhamon.signal.filters import butter_low_filter
    from jhamon.signal.mech import _detect_onset
    from jhamon_training.kinematics import (
        calculate_knee_velocity,
        calculate_hip_velocity,
    )
    from jhamon_training.utils import linear_tnorm

    results_session = dict()
    for serie in session_data.keys():
        indexes = session_data[serie][0][0]
        DATA = session_data[serie][1][0]

        results = dict()
        for repetition in range(indexes.shape[1]):
            rep_num = "rep_" + str(repetition + 1)
            print(participant + " " + tr_session + " " + serie + " " + rep_num)
            repe = DATA[indexes[0, repetition] : indexes[1, repetition], :]

            # Create empty dictionary with all variables of interest
            vdict = {
                "REP_time": 0,
                "hip_ROM": 0,
                "hip_v_mean": 0,
                "hip_v_peak": 0,
                "knee_ROM": 0,
                "knee_v_mean": 0,
                "knee_v_peak": 0,
                "knee_angDWA": 0,
                "knee_ROMDWA": 0,
                "knee_fpeak": 0,
                "knee_ROMfpeak": 0,
                "knee_tor_mean": 0,
                "knee_tor_peak": 0,
                "knee_impulse": 0,
                "knee_work": 0,
            }

            normcurves = {
                "force": [],
                "torque": [],
                "knee_ROM": [],
                "hip_ROM": [],
                "knee_v": [],
                "hip_v": [],
                "knee_work": [],
            }

            try:
                # Correct manually some different setup exceptions
                if participant in ["jhamon06"] and tr_session in ["tr_1", "tr_2"]:
                    marker1 = np.array(repe[:, [4, 3]])
                    marker2 = np.array(repe[:, [7, 6]])
                    marker3 = np.array(repe[:, [10, 9]])
                    marker4 = np.array(repe[:, [13, 12]])
                    marker5 = np.array(repe[:, [16, 15]])
                    marker6 = np.array(repe[:, [19, 18]])

                if participant in ["jhamon10"] and tr_session in ["tr_1"]:
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
                    marker3, marker4, marker5, marker6, freq=100
                )

                hip_rad, hip_v_rad, hip_acc = calculate_hip_velocity(
                    marker1, marker2, marker3, marker4, freq=100
                )

                # All data rep
                D = np.column_stack(
                    (
                        repe[:-2, :3],
                        knee_rad[:-2],
                        hip_rad[:-2],
                        knee_v_rad[:-1],
                        hip_v_rad[:-1],
                        knee_acc,
                        hip_acc,
                        marker1[:-2],
                        marker2[:-2],
                        marker3[:-2],
                        marker4[:-2],
                        marker5[:-2],
                        marker6[:-2],
                    )
                )

                D[:, 0] = np.transpose(np.linspace(0, len(D) / 100, num=len(D)))

                # Flipped calibrations
                if participant in ["jhamon06"] and tr_session in ["tr_1", "tr_2"]:
                    D[:, 5] = D[:, 5] * -1

                if participant in ["jhamon10"] and tr_session in ["tr_1"]:
                    D[:, 5] = D[:, 5] * -1

                # Find start of repetition from torque (filtered)
                force = D[:, 2]  # torque

                # Filter
                force_filt = butter_low_filter(force, fs=100, cut_hz=3, order=2)

                dif_force = np.diff(force_filt)
                onset_1 = _detect_onset(
                    force_filt,
                    max(force_filt[:]) * 0.25,
                    n_above=80,
                    n_below=0,
                    show=False,
                )[0, 0]
                onset_precise = dif_force[: onset_1 + 5] < 0

                if np.where(onset_precise)[0].shape[0] == 0:
                    force_onset = 0
                else:
                    force_onset = np.where(onset_precise)[0][-1]

                if force[force_onset] > max(force_filt[:]) * 0.25:
                    force_onset = np.where(onset_precise)[0][-2]

                # Set a maximum index at which velocity is evaluated
                force_offset = _detect_onset(
                    force_filt,
                    max(force_filt[:]) * 0.3,
                    n_above=80,
                    n_below=0,
                    show=False,
                )[0, 1]
                force_offset = force_offset + 25

                # Find the index at which the velocity is minimum
                vmin_idx = D[force_onset:, 5].argmin() + force_onset

                # Handle the case where force_offset happens before vmin_idx
                if force_offset < vmin_idx:
                    error_message = f"""Error processing in participant = {participant},
                        tr_session = {tr_session},
                        serie = {serie},
                        rep = {rep_num}: force_offset < vmin_idx (force offset detected too early)"""
                    logging.error(error_message)
                    continue

                vmax_idx = D[vmin_idx:force_offset, 5].argmax() + vmin_idx

                # manual fix
                if participant in ["jhamon09"] and tr_session in ["tr_3"]:
                    vmax_idx = (
                        D[force_onset : force_offset + 250, 5].argmax() + force_onset
                    )

                # Final data of segmented repetitions
                REP = D[force_onset:vmin_idx, :]

                # Check if REP is empty
                if REP.size == 0:
                    # Handle the case where REP is empty
                    error_message = f"""Error processing in participant = {participant},
                        tr_session = {tr_session},
                        serie = {serie},
                        rep = {rep_num}: REP is empty"""
                    logging.error(error_message)
                    continue

                knee_ROM = np.abs(REP[0, 3] - REP[-1, 3]) * 180 / np.pi
                hip_ROM = np.abs(REP[1, 4] - REP[-1, 4]) * 180 / np.pi

                # Distances between markers for torque calculations
                x1, y1, x2, y2 = (
                    marker4[0, 0],
                    marker4[0, 1],
                    marker6[0, 0],
                    marker6[0, 1],
                )

                shank_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                lever_arm = shank_length - 0.05

                if participant in ["jhamon12"] and tr_session in ["tr_5"]:
                    lever_arm = 0.32231839872197643

                # Calculate variables of interest and store them
                knee_v_mean = np.abs(REP[:, 5]).mean() * 180 / np.pi
                knee_v_peak = np.abs(REP[:, 5]).max() * 180 / np.pi
                hip_v_mean = np.abs(REP[:, 6]).mean() * 180 / np.pi
                hip_v_peak = np.abs(REP[:, 6]).max() * 180 / np.pi

                # Find angle at which velocity sudenly increases
                vmax_REP_idx = REP[:, 5].argmin()
                angDWA_idx = np.diff(REP[:vmax_REP_idx, 5]) < 0
                angDWA_idx2 = np.where(angDWA_idx)[0][-1]

                # angle at which velocity increases
                knee_angDWA = np.abs(REP[angDWA_idx2, 3] * 180 / np.pi)

                # % of ROM at which angDWA occurs
                knee_ROMDWA = (angDWA_idx2 * 100) / len(REP)

                # Angle at which peak force occurs
                fpeak_REP_idx = REP[:, 2].argmax()
                # Knee angle at which peak force occurs
                knee_fpeak = np.abs(REP[fpeak_REP_idx, 3] * 180 / np.pi)
                # Percentage of ROM at which peak force occurs
                knee_ROMfpeak = (fpeak_REP_idx * 100) / len(REP)

                # Torque, time, work
                knee_tor_mean = REP[:, 2].mean() * lever_arm
                knee_tor_peak = REP[:, 2].max() * lever_arm

                freq = 100
                REP_time = len(REP) / freq  # in seconds
                knee_impulse = knee_tor_mean * REP_time
                knee_work = np.abs(np.trapz(REP[:, 2] * lever_arm, REP[:, 3]))  # Jules

                # Torque inertia corrected
                bm = pesocorporal
                ex_weight = 0
                I_knee = dame_inercia(REP, bm, ex_weight)

                # Apply correction factor to inertia, based on the force developed
                torque_corrected = np.zeros(len(REP))
                for ii in range(len(REP)):
                    torque_measured = REP[ii, 2] * lever_arm
                    torque_corrected[ii] = -torque_measured + (
                        (I_knee[ii] * REP[ii, 7]) / 2
                    )

                normcurves = dict()
                normcurves = {
                    "force": linear_tnorm(REP[:, 2])[0],
                    "torque": linear_tnorm(REP[:, 2] * lever_arm)[0],
                    "knee_ROM": linear_tnorm((np.abs(REP[:, 3]) * 180 / np.pi))[0],
                    "knee_v": linear_tnorm(np.abs(REP[:, 5] * 180 / np.pi))[0],
                    #   'hip_ROM': linear_tnorm((REP[:, 4] * 180 / np.pi))[0],
                    #   'hip_v': linear_tnorm((REP[:, 6] * 180 / np.pi))[0],
                    "knee_work": linear_tnorm(((REP[:, 2] * lever_arm) * REP[:, 3]))[0],
                }

                # save all discrete results in a dict (vdict)
                vars_nam = [
                    "REP_time",
                    "hip_ROM",
                    "hip_v_mean",
                    "hip_v_peak",
                    "knee_ROM",
                    "knee_v_mean",
                    "knee_v_peak",
                    "knee_angDWA",
                    "knee_ROMDWA",
                    "knee_fpeak",
                    "knee_ROMfpeak",
                    "knee_tor_mean",
                    "knee_tor_peak",
                    "knee_impulse",
                    "knee_work",
                ]

                vars_discrete = np.array(
                    (
                        REP_time,
                        hip_ROM,
                        hip_v_mean,
                        hip_v_peak,
                        knee_ROM,
                        knee_v_mean,
                        knee_v_peak,
                        knee_angDWA,
                        knee_ROMDWA,
                        knee_fpeak,
                        knee_ROMfpeak,
                        knee_tor_mean,
                        knee_tor_peak,
                        knee_impulse,
                        knee_work,
                    )
                )

                vdict = dict(zip(vars_nam, vars_discrete))

                # Store the vdict for each repetition in the results dictionary
                results[rep_num] = [vdict, normcurves]

            except Exception as e:
                error_message = f"""Error processing in participant = {participant},
                    tr_session = {tr_session},
                    serie = {serie},
                    rep = {rep_num}: {str(e)}"""
                logging.error(error_message)
                continue

        # # Plot to check onsets and vmx
        # plotea_nordic(D, force_onset, vmax_idx, angDWA_idx2)

        # After collecting all repetitions, create a DataFrame for the results
        resultsdf = pd.DataFrame(results)
        results_session[serie] = resultsdf

    return results_session


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plotea_nordic(data, force_onset, vmax_idx, angDWA_idx2):

    import matplotlib.pyplot as plt
    import numpy as np

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.2))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the
    # patch and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)

    (p1,) = host.plot(data[:, 0], data[:, 2], label="Torque (Nm)")
    (p2,) = par1.plot(data[:, 0], data[:, 3] * 180 / np.pi, "r-", label="Knee angle")
    (p3,) = par2.plot(data[:, 0], data[:, 5] * 180 / np.pi, "g-", label="Knee Velocity")

    host.set_xlabel("Time (s)")
    host.set_ylabel("Force (N)")
    par1.set_ylabel("Knee angle (deg)")
    par2.set_ylabel("Knee velocity (deg/s)")

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis="y", colors=p1.get_color(), **tkw)
    par1.tick_params(axis="y", colors=p2.get_color(), **tkw)
    par2.tick_params(axis="y", colors=p3.get_color(), **tkw)
    host.tick_params(axis="x", **tkw)

    lines = [p1, p2, p3]
    host.legend(lines, [l.get_label() for l in lines])

    [
        host.axvline(_x, linewidth=1, color="k", ls="--")
        for _x in np.array(
            (
                data[force_onset, 0],
                data[vmax_idx, 0],
                data[angDWA_idx2 + force_onset, 0],
            )
        )
    ]

    labels = ["force onset", "peak velocity", "angDWA"]
    for i, x in enumerate(
        np.array(
            (
                data[force_onset, 0],
                data[vmax_idx, 0],
                data[angDWA_idx2 + force_onset, 0],
            )
        )
        + 0.02
    ):
        host.text(x, 50, labels[i], rotation=90, verticalalignment="center")

    host.axvspan(data[force_onset, 0], data[vmax_idx, 0], color="red", alpha=0.1)
    host.grid()
    plt.show()
