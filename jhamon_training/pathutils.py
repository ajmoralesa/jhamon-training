"""Main module."""

from typing import Optional, Union, List
import os
import fnmatch
from pathlib import Path


# Single source of truth for where analysis results, caches, figures, and stats
# are written/read. The folder has moved several times (Desktop -> SynologyDrive
# -> external drive), so instead of hardcoding one location we probe an ordered
# list of known candidates and take the first one that is actually there. To use
# a location that is not on the list, export JHAMON_RESULTS=/path/to/folder.
#
# Order matters: the OneDrive copy is the live, canonical location because it is
# stored locally and therefore always reachable; the "Espace Privé" share is the
# older canonical copy but lives behind a VPN and drops without warning;
# SynologyDrive is the 2026-05 location; ~/Desktop/2019_jHamON/_RESULTS is the
# original, now-legacy copy kept as a last resort. Add new locations at the top.
_ENV_RESULTS_VAR = "JHAMON_RESULTS"

# Canonical location (2026-07): the locally-synced OneDrive copy. Preferred over
# every network location because it survives a dropped VPN connection.
_RESULTS_PATH_ONEDRIVE = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "OneDrive-insep"
    / "RECHERCHE"
    / "Projects"
    / "2019_jHamON"
    / "_RESULTS_TRAINING"
)

# Previous canonical location: an institutional share reachable only over VPN.
# The connection is unstable and the mount disappears mid-session, which is
# exactly why the OneDrive copy above now takes precedence.
_RESULTS_PATH_EXTERNAL = Path(
    "/Volumes/Espace Privé/MORALES Antonio/Antonio J Morales"
    "/perso/RECHERCHE/Projects/2019_jHamON/_RESULTS_TRAINING"
)

# Previous location (2026-05): the paper's project folder on SynologyDrive. The
# Synology is retired and no longer mounts; kept only so an old machine that
# still has it can fall back to it.
_RESULTS_PATH_SYNOLOGY = (
    Path.home()
    / "SynologyDrive"
    / "perso"
    / "RECHERCHE"
    / "Projects"
    / "2019_jHamON"
    / "_RESULTS_TRAINING"
)

# Legacy location: the original folder on the Desktop (named _RESULTS, not
# _RESULTS_TRAINING). Still holds usable pickle/feather caches.
_RESULTS_PATH_LEGACY_DESKTOP = (
    Path.home() / "Desktop" / "2019_jHamON" / "_RESULTS"
)

_RESULTS_PATH_CANDIDATES = (
    _RESULTS_PATH_ONEDRIVE,
    _RESULTS_PATH_EXTERNAL,
    _RESULTS_PATH_SYNOLOGY,
    _RESULTS_PATH_LEGACY_DESKTOP,
)


def _results_path_candidates() -> List[Path]:
    """Ordered list of folders to probe for the results directory.

    An explicit ``JHAMON_RESULTS`` environment variable short-circuits the
    search: if it is set it becomes the one and only candidate, so the user
    always gets the folder they asked for (even if it does not exist yet, e.g.
    a fresh output directory the pipeline is about to create).
    """
    override = os.environ.get(_ENV_RESULTS_VAR)
    if override:
        # expanduser() so JHAMON_RESULTS=~/somewhere works from a shell that
        # did not expand the tilde itself.
        return [Path(override).expanduser()]

    return list(_RESULTS_PATH_CANDIDATES)


def _is_readable_dir(path: Path) -> bool:
    """True if ``path`` is an existing directory, False for anything else.

    ``Path.is_dir()`` is not safe to call directly on ``/Volumes/...``: it only
    swallows ENOENT/ENOTDIR, so a half-mounted or stale external drive makes it
    raise (observed: ``OSError: [Errno 57] Socket is not connected``). That would
    blow up at import time, which is precisely what this module must never do —
    so every OSError is treated as "not available".
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def resolve_results_path(require_exists: bool = False) -> Path:
    """Return the folder holding analysis results, caches, figures, and stats.

    Resolution order:
      1. ``$JHAMON_RESULTS`` if set (always wins).
      2. Otherwise the first candidate in ``_RESULTS_PATH_CANDIDATES`` that
         exists on disk.
      3. Otherwise the first candidate, so that downstream error messages name
         the *intended* location rather than something arbitrary.

    Args:
        require_exists: When True, raise instead of falling back to a
                        non-existent path. Use this in scripts that are about to
                        read the caches, so the failure is reported up front with
                        a message that lists everywhere we looked.

    Returns:
        Path: the results folder.

    Raises:
        FileNotFoundError: only when ``require_exists`` is True and none of the
                           candidates exist.
    """
    candidates = _results_path_candidates()

    for candidate in candidates:
        if _is_readable_dir(candidate):
            return candidate

    if require_exists:
        tried = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "Could not locate the jHamON results folder. Tried:\n"
            f"{tried}\n"
            "If the external drive is unmounted, mount it; otherwise set "
            f"{_ENV_RESULTS_VAR} to the folder holding nht_results.pkl / "
            "ikt_results.pkl / training_stats.json."
        )

    # Deliberately do NOT raise at import time: a missing drive must not stop
    # editors, linters, or `--help` from importing this package.
    return candidates[0]


# Kept as a module-level constant for backwards compatibility — many modules do
# `from jhamon_training.pathutils import RESULTS_TRAINING_PATH`. Resolved once at
# import time; call resolve_results_path() directly if the drive may be mounted
# after import, or resolve_results_path(require_exists=True) to fail loudly.
RESULTS_TRAINING_PATH = resolve_results_path()


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
