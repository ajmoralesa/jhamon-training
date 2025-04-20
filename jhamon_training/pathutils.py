"""Main module."""

from typing import Optional, Union, List
import os
import fnmatch


def dame_participants():

    part = [
        "jhamon01",
        "jhamon02",
        "jhamon03",
        "jhamon04",
        "jhamon05",
        "jhamon06",
        "jhamon09",
        "jhamon10",
        "jhamon11",
        "jhamon12",
        "jhamon14",
        "jhamon15",
        "jhamon16",
        "jhamon18",
        "jhamon20",
        "jhamon22",
        "jhamon23",
        "jhamon24",
        "jhamon25",
        "jhamon26",
        "jhamon28",
        "jhamon29",
        "jhamon30",
        "jhamon31",
        "jhamon41",
        "jhamon33",
        "jhamon34",
        "jhamon07",
        "jhamon08",
        "jhamon17",
        "jhamon19",
        "jhamon21",
        "jhamon27",
        "jhamon35",
        "jhamon36",
        "jhamon37",
        "jhamon38",
        "jhamon40",
        "jhamon42",
    ]

    return part


def dame_nht_paths(
    path_to_folders, participant_id: Optional[Union[str, List[str]]] = None
):
    """
    Create dict with all paths to each participant and training
    session available for the specified participant(s).

    Args:
        path_to_folders: Path object pointing to the main data directory.
        participant_id: Optional string or list of strings specifying which
                        participants to include. If None, uses a default list.

    Returns:
        dict: Dictionary mapping participant IDs to their training session paths.
    """
    # Default list if no specific participant is requested
    default_participants = [
        "jhamon01",
        "jhamon02",
        "jhamon03",
        "jhamon04",
        "jhamon05",
        "jhamon06",
        "jhamon09",
        "jhamon10",
        "jhamon11",
        "jhamon12",
        "jhamon14",
        "jhamon15",
        "jhamon16",
    ]

    if participant_id is None:
        participants_to_process = default_participants
    elif isinstance(participant_id, str):
        participants_to_process = [participant_id]
    elif isinstance(participant_id, list):
        participants_to_process = participant_id
    else:
        raise TypeError("participant_id must be None, a string, or a list of strings")

    training_sessions = dict()
    for participant in participants_to_process:
        participant_path = path_to_folders / participant
        try:
            tr_folders = fnmatch.filter(os.listdir(participant_path), "*tr_*")
            training_sessions[participant] = {}
            for train_folder in tr_folders:
                training_sessions[participant][train_folder] = (
                    participant_path / train_folder
                )
        except FileNotFoundError:
            print(
                f"Warning: Directory not found for participant {participant} at {participant_path}. Skipping."
            )
            continue  # Skip this participant if their directory doesn't exist

    return training_sessions


def dame_ik_paths(
    path_to_folders, participant_id: Optional[Union[str, List[str]]] = None
):
    """
    Create dict with all paths to each participant and training
    session available for the specified participant(s).

    Args:
        path_to_folders: Path object pointing to the main data directory.
        participant_id: Optional string or list of strings specifying which
                        participants to include. If None, uses a default list.

    Returns:
        dict: Dictionary mapping participant IDs to their training session paths.
    """
    default_participants = [
        "jhamon18",
        "jhamon20",
        "jhamon22",
        "jhamon23",
        "jhamon24",
        "jhamon25",
        "jhamon26",
        "jhamon28",
        "jhamon29",
        "jhamon30",
        "jhamon31",
        "jhamon32",
        "jhamon33",
        "jhamon34",
    ]

    if participant_id is None:
        participants_to_process = default_participants
    elif isinstance(participant_id, str):
        participants_to_process = [participant_id]
    elif isinstance(participant_id, list):
        participants_to_process = participant_id
    else:
        raise TypeError("participant_id must be None, a string, or a list of strings")

    training_sessions = dict()
    for participant in participants_to_process:
        participant_path = path_to_folders / participant
        try:
            tr_folders = fnmatch.filter(os.listdir(participant_path), "*tr_*")
            training_sessions[participant] = {}
            for train_folder in tr_folders:
                training_sessions[participant][train_folder] = (
                    participant_path / train_folder
                )
        except FileNotFoundError:
            print(
                f"Warning: Directory not found for participant {participant} at {participant_path}. Skipping."
            )
            continue  # Skip this participant if their directory doesn't exist

    return training_sessions
