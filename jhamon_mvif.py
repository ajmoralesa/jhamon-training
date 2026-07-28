"""Compute PRE-intervention MVIF (maximal voluntary isometric flexion torque)
for the NH and IK training groups and save the long-format table.

MVIF is the peak gravity-corrected isometric knee-flexion torque measured during
the familiarization (PRE) sessions. One row is produced per participant per
familiarization session that contains an MVC test (no per-participant
aggregation); see ``jhamon_training/data/mvif.py``.

Usage:
    python jhamon_mvif.py [DATA_ROOT]

DATA_ROOT defaults to the mounted raw-data volume (``/Volumes/jHamON``, falling
back to ``/Volumes/AJMA``). Results are written to ``RESULTS_TRAINING_PATH``
(``~/SynologyDrive/perso/RECHERCHE/Projects/2019_jHamON/_RESULTS_TRAINING``).
"""

import sys
from pathlib import Path

from jhamon_training.data.mvif import compute_pre_mvif
from jhamon_training.pathutils import RESULTS_TRAINING_PATH


def resolve_data_root(argv):
    """Pick the data root from argv[1], else the first mounted candidate."""
    if len(argv) > 1:
        root = Path(argv[1])
        if not root.is_dir():
            raise SystemExit(f"Data root not found: {root}")
        return root

    for candidate in (Path("/Volumes/jHamON"), Path("/Volumes/AJMA")):
        if candidate.is_dir():
            return candidate

    raise SystemExit(
        "No data volume mounted (looked for /Volumes/jHamON and /Volumes/AJMA). "
        "Pass the data root explicitly: python jhamon_mvif.py DATA_ROOT"
    )


def main(data_root):
    results_output_path = RESULTS_TRAINING_PATH
    results_output_path.mkdir(parents=True, exist_ok=True)

    print(f"Computing PRE MVIF from {data_root} ...")
    df = compute_pre_mvif(data_root)

    if df.empty:
        raise SystemExit("No PRE MVIF values were computed (see warnings above).")

    n_par = df["par"].nunique()
    by_group = df.groupby("tr_group")["par"].nunique().to_dict()
    print(
        f"\nComputed {len(df)} PRE MVIF rows across {n_par} participants "
        f"(participants per group: {by_group})."
    )
    print(df.to_string(index=False))

    feather_path = results_output_path / "mvif_pre_results.feather"
    csv_path = results_output_path / "mvif_pre_results.csv"
    df.to_feather(feather_path)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved:\n  {feather_path}\n  {csv_path}")


if __name__ == "__main__":
    main(resolve_data_root(sys.argv))
