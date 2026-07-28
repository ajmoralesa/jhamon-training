"""PRE-intervention MVIF (maximal voluntary isometric flexion torque).

Computes the peak gravity-corrected isometric knee-flexion torque for the
familiarization (PRE) sessions of the NH and IK training groups. This mirrors
the MVC pipeline from the base ``jhamon`` repository (``data/emg.py`` +
``signal/mech.getMVC``) but is restricted to:

* PRE timepoints only -- the ``fam_*`` familiarization sessions; POST sessions
  are ignored;
* the mechanical MVIF only -- no EMG is computed; and
* the NH and IK intervention groups -- controls (CO) are excluded.

Every PRE session that contains an MVC test is kept as its own row (no
per-participant aggregation), so a participant with several familiarization
sessions contributes one MVIF value per session.
"""

import os
import fnmatch
import warnings

import pandas as pd

from jhamon_training.data.frames import alocate_training_group
from jhamon_training.signal.mech import getMVIF


# Familiarization sessions that constitute the PRE (baseline) timepoint.
PRE_SESSIONS = ("fam_1", "fam_2", "fam_3", "fam_4")


def dame_pre_mvif_paths(path_to_folders, participants):
    """Discover PRE MVC test folders for the given participants.

    Walks each participant's ``fam_*/mtu`` directories and records those that
    contain an ``mvc_*`` file.

    Arguments:
        path_to_folders: Path to the data root (one sub-folder per participant).
        participants: Iterable of participant ids to include.

    Returns:
        list of dict -- one entry per PRE MVC test with keys ``par``, ``ntest``,
        ``mtu_path``, ``mvc_file`` and ``has_passive`` (whether a ``passive_*``
        file needed for gravity correction is present).
    """
    records = []
    for par in participants:
        par_path = path_to_folders / par
        if not par_path.is_dir():
            warnings.warn(f"{par}: participant folder not found at {par_path}, skipping")
            continue

        for ntest in PRE_SESSIONS:
            mtu = par_path / ntest / "mtu"
            if not mtu.is_dir():
                continue

            files = os.listdir(mtu)
            mvc_files = fnmatch.filter(files, "mvc_*")
            if not mvc_files:
                continue
            if len(mvc_files) > 1:
                warnings.warn(
                    f"{par}/{ntest}: {len(mvc_files)} mvc files, using {mvc_files[0]}"
                )

            records.append(
                {
                    "par": par,
                    "ntest": ntest,
                    "mtu_path": mtu,
                    "mvc_file": mvc_files[0],
                    "has_passive": bool(fnmatch.filter(files, "passive_*")),
                }
            )

    return records


def compute_pre_mvif(path_to_folders):
    """Compute PRE MVIF for all NH and IK participants.

    Arguments:
        path_to_folders: Path to the data root (one sub-folder per participant).

    Returns:
        pandas.DataFrame -- long format, one row per (participant, fam session)
        with columns ``par``, ``ntest``, ``tr_group``, ``timepoint`` (always
        ``"PRE"``), ``var`` (always ``"mvif"``) and ``value`` (N.m).
    """
    NH, IK, _CO = alocate_training_group()
    group_of = {par: "NH" for par in NH}
    group_of.update({par: "IK" for par in IK})
    participants = NH + IK

    records = dame_pre_mvif_paths(path_to_folders, participants)

    rows = []
    skipped = []
    for rec in records:
        par, ntest, mtu = rec["par"], rec["ntest"], rec["mtu_path"]

        # Gravity correction needs a passive_ trial in the same folder.
        if not rec["has_passive"]:
            skipped.append((par, ntest, "no passive_ file for gravity correction"))
            continue

        try:
            mvif = getMVIF(mvc_path=mtu / rec["mvc_file"], pth=mtu)
        except Exception as exc:  # report and continue rather than abort the run
            skipped.append((par, ntest, f"MVIF computation failed: {exc}"))
            continue

        rows.append(
            {
                "par": par,
                "ntest": ntest,
                "tr_group": group_of[par],
                "timepoint": "PRE",
                "var": "mvif",
                "value": mvif,
            }
        )

    df = pd.DataFrame(
        rows, columns=["par", "ntest", "tr_group", "timepoint", "var", "value"]
    )

    if skipped:
        print(f"\nMVIF: skipped {len(skipped)} PRE session(s):")
        for par, ntest, reason in skipped:
            print(f"  - {par}/{ntest}: {reason}")

    return df
