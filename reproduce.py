#!/usr/bin/env python3
"""Regenerate every published number of the jHamON training paper from cache.

WHAT THIS DOES
--------------
Rebuilds the whole discrete-outcome analysis of the NH-vs-IK training paper,
starting from the two cached signal-processing results and ending at
``training_stats.json`` -- the file that ``jhamon_results.qmd`` and the
manuscript read every reported number from. The stages are exactly the ones
``jhamon_training.py`` runs, in the same order, with the same functions:

    nht_results.pkl / ikt_results.pkl        (cached per-repetition signals)
        -> frames.nht_todf / frames.ikt_todf (101-point normalised curves)
        -> analyze_discrete_variables()      (work, torques, ROM, velocity per rep)
        -> filter_training_data()            (session/set/participant exclusions)
        -> run_mixed_lmm()                   (linear mixed model, lmm_* keys)
        -> run_agreement()                   (Bland-Altman, ba_* keys)
        -> extract_manuscript_stats.py       (whole-curve SPM, session progression,
                                              baseline torque, repetition counts:
                                              spm_*, prog_*, mvif_*, workrep_*,
                                              reps_*, desc_* keys)
        -> extract_validity_stats.py         (constant-velocity phase and the
                                              agreement analysis by dataset tier:
                                              velphase_*, bav_*, lmmt_*, lmmiso_*)
        -> extract_stats_addendum.py         (Welch df, terminal progression
                                              contrasts, per-cluster knee angles)
        -> training_stats.json               (921 keys)
        -> manuscript_stats_detail.json      (per-cluster, per-session SPM detail)

Those last three stages are separate scripts rather than functions because each
is independently runnable during analysis; this script calls their ``main()``
directly, pointing them at the staging file, so a single command regenerates
every number the manuscript quotes. ``jhamon_results.qmd`` reads both JSON files
and hard-codes no value, so ``--check`` passing is what makes the manuscript's
reproducibility statement true.

WHAT IT NEEDS
-------------
1. ``nht_results.pkl`` and ``ikt_results.pkl`` in the results folder
   (``RESULTS_TRAINING_PATH``, or ``--results-path``, or ``$JHAMON_RESULTS``).
   Together they are ~60 MB and hold every processed repetition of every
   participant; they are the real archive of this study.
2. The project virtualenv (``source venv/bin/activate``). Nothing outside
   ``requirements.txt`` plus statsmodels/patsy/scipy, all already installed.

WHAT IT DOES *NOT* NEED
-----------------------
* The raw-data drive ``/Volumes/AJMA/``. Not mounted, not looked at, not
  needed. Everything upstream of the caches (C3D parsing, filtering, onset
  detection, repetition segmentation) is already baked into the two pickles.
* Any network share, VPN mount, or SynologyDrive folder.
* MATLAB, R, or Quarto. Those consume ``training_stats.json``; they do not
  produce it.

So a colleague with a copy of the two pickles and this repository can
regenerate the paper's statistics, and nothing else is required.

USAGE
-----
    source venv/bin/activate

    python reproduce.py                  # rebuild and install training_stats.json
    python reproduce.py --check          # rebuild to a temp dir and DIFF instead
    python reproduce.py --figures        # also redraw the discrete-outcome figures
    python reproduce.py --results-path /some/copy/of/_RESULTS_TRAINING

``--check`` is the regression test this project otherwise lacks: it re-runs the
entire chain into a throwaway directory and compares the result key by key with
the ``training_stats.json`` already on disk, exiting non-zero if any published
number moved. Run it after touching anything under ``jhamon_training/`` --
if a refactor changes a number, this is what tells you.

SAFETY
------
The pipeline always writes into a temporary staging directory first. The real
``training_stats.json`` is replaced only at the very end, atomically, and only
when ``--check`` is absent. A crash halfway through can therefore never leave a
half-written stats file behind.

NOTES FOR FUTURE READERS
------------------------
* The caches were pickled in 2019 under pandas < 1.3, whose block layout modern
  pandas cannot unpickle (``TypeError: Argument 'placement' has incorrect
  type``). ``_install_legacy_pickle_shim()`` below patches that at load time;
  without it ``nht_results.pkl`` will not open at all under pandas 2.x.
* Some archived copies of the caches are keyed by real first names
  (``01_Kevin``) rather than pseudonyms (``jhamon01``). ``_pseudonymise()``
  normalises those; on the canonical caches it is a no-op.
* The velocity QC filter keeps ``10 < knee_v_mean < 35``. One archived IK cache
  stores that velocity as negative, which silently deletes the entire IK group.
  The run therefore reports the sign explicitly and refuses to fit the model if
  a whole group disappears, rather than producing an NH-only "reproduction".
* Figures that come from the SPM curve analyses (manuscript Figure 2 and the
  training-progression panel) are NOT redrawn by ``--figures``; they need the
  ``spm1d`` runs that live in ``jhamon_training.py``. ``--figures`` covers every
  figure derived from the per-repetition discrete outcomes.
* The between-group SPM figures still drawn by ``jhamon_training.py`` use one
  curve per participant x session, whereas the manuscript now reports the
  participant-averaged contrast (``spm_*_paravg_*``). Those figures must be
  regenerated from the participant-averaged analysis before submission, or the
  figure and the text will disagree about the cluster extents.
* Stage 6 takes several minutes: the session-by-session progression SPM fits
  4 x 14 random-field models. Everything else runs in well under a minute.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
NH_CACHE = "nht_results.pkl"
IK_CACHE = "ikt_results.pkl"
STATS_FILENAME = "training_stats.json"

#: Absolute *and* relative tolerance used when diffing two stats files. The
#: values in the JSON are already rounded (2-4 decimals), so a genuine
#: reproduction matches exactly; this only absorbs float round-tripping.
DEFAULT_TOLERANCE = 1e-9

#: Figures that need nothing but the filtered per-repetition outcomes.
#: ``(module, function, output filename)``.
DISCRETE_FIGURES: Tuple[Tuple[str, str, str], ...] = (
    ("jhamon_training.plot.figure_3_ridgeline", "plot_figure_3_ridgeline",
     "Figure_3_ridgeline.png"),
    ("jhamon_training.plot.plot_peak_torque_evolution", "plot_peak_torque_evolution",
     "peak_torque_evolution.png"),
    ("jhamon_training.plot.plot_work_comparison", "plot_work_comparison",
     "work_comparison.png"),
    ("jhamon_training.plot.plot_knee_velocity_comparison",
     "plot_knee_velocity_comparison", "knee_velocity_comparison.png"),
    ("jhamon_training.plot.plot_repetition_count", "plot_repetition_count",
     "repetition_count.png"),
)

_N_STAGES = 7


# --------------------------------------------------------------------------- #
# console output
# --------------------------------------------------------------------------- #
def _rule(char: str = "=") -> str:
    return char * 72


def _stage(number: int, title: str) -> None:
    """Print a stage header so the log reads as a numbered pipeline."""
    print(f"\n[{number}/{_N_STAGES}] {title}")
    print(_rule("-"))


def _item(label: str, value: Any) -> None:
    print(f"      {label:<34}{value}")


# --------------------------------------------------------------------------- #
# environment / compatibility
# --------------------------------------------------------------------------- #
def _install_legacy_pickle_shim() -> None:
    """Let pandas 2.x unpickle the 2019 caches (blocks with ``placement=slice``).

    pandas < 1.3 stored a block's column placement as a plain ``slice``; pandas
    2.x requires a ``BlockPlacement``. Its ``new_block`` no longer converts, so
    ``pickle.load`` on ``nht_results.pkl`` dies with ``TypeError: Argument
    'placement' has incorrect type``. We wrap ``new_block`` to coerce the
    argument, in both modules that expose it, and register the index classes
    that older pickles may reference by name.

    Called before any cache is opened. Harmless when the pickles are modern.
    """
    import types

    import pandas as pd
    from pandas._libs.internals import BlockPlacement
    import pandas.core.internals.blocks as _blocks
    import pandas.core.internals.managers as _managers

    if getattr(_blocks.new_block, "_jhamon_shim", False):
        return

    original_new_block = _blocks.new_block

    def new_block(values, placement, *, ndim, refs=None):
        if not isinstance(placement, BlockPlacement):
            placement = BlockPlacement(placement)
        try:
            return original_new_block(values, placement, ndim=ndim, refs=refs)
        except TypeError:
            # pandas versions whose new_block has no ``refs`` parameter.
            return original_new_block(values, placement, ndim=ndim)

    new_block._jhamon_shim = True  # type: ignore[attr-defined]
    _blocks.new_block = new_block
    # managers.py did ``from .blocks import new_block``, so it holds its own
    # reference and has to be patched separately.
    _managers.new_block = new_block

    # Index classes that were removed in pandas 2.x but are named inside old
    # pickles. Mapping them all to pd.Index is what pandas itself does on load.
    legacy_index_module = "pandas.core.indexes.numeric"
    if legacy_index_module not in sys.modules:
        module = types.ModuleType(legacy_index_module)
        for attribute in ("Int64Index", "Float64Index", "UInt64Index", "NumericIndex"):
            setattr(module, attribute, pd.Index)
        sys.modules[legacy_index_module] = module


def resolve_results_path(requested: Optional[str]) -> Path:
    """Decide which results folder to use and make the whole package agree.

    ``jhamon_training.pathutils`` resolves ``RESULTS_TRAINING_PATH`` once, at
    import time, honouring ``$JHAMON_RESULTS`` above everything else. Setting
    that variable here -- before a single ``jhamon_training`` module is
    imported -- is therefore what makes ``--results-path`` apply everywhere,
    including inside modules that read the constant at import.

    Args:
        requested: value of ``--results-path``, or None to use the default
            resolution order in ``pathutils``.

    Returns:
        The results folder as an absolute path.
    """
    if requested is not None:
        path = Path(requested).expanduser().resolve()
        os.environ["JHAMON_RESULTS"] = str(path)
        return path

    from jhamon_training.pathutils import resolve_results_path as _resolve

    path = _resolve()
    # Pin it, so a module imported later cannot resolve to a different folder
    # (e.g. if a network share mounts halfway through the run).
    os.environ["JHAMON_RESULTS"] = str(path)
    return path


def require_caches(results_path: Path) -> Tuple[Path, Path]:
    """Verify both caches exist, or explain precisely what is missing.

    Raises:
        SystemExit: with a message naming every expected path and the two ways
            to point the script somewhere else. This is the failure a colleague
            is most likely to hit, so it must be self-explanatory.
    """
    nh_path = results_path / NH_CACHE
    ik_path = results_path / IK_CACHE
    missing = [p for p in (nh_path, ik_path) if not p.exists()]
    if not missing:
        return nh_path, ik_path

    lines = [
        "",
        _rule(),
        "CANNOT REPRODUCE: cached results are missing.",
        _rule(),
        f"Results folder searched: {results_path}"
        + ("" if results_path.is_dir() else "   <-- this folder does not exist"),
        "",
        "Expected both of:",
    ]
    for path in (nh_path, ik_path):
        lines.append(f"  {'MISSING ' if path in missing else 'found   '}{path}")
    lines += [
        "",
        "These two pickles are the whole input of this script; without them the",
        "analysis can only be rebuilt from the raw drive /Volumes/AJMA/ by",
        "running jhamon_training.py, which takes hours.",
        "",
        "Point the script at a copy with either:",
        "    python reproduce.py --results-path /path/to/_RESULTS_TRAINING",
        "    export JHAMON_RESULTS=/path/to/_RESULTS_TRAINING",
        _rule(),
    ]
    raise SystemExit("\n".join(lines))


# --------------------------------------------------------------------------- #
# pipeline stages
# --------------------------------------------------------------------------- #
def _pseudonymise(cache: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite ``01_Kevin`` -> ``jhamon01``; leave pseudonymous keys untouched.

    Some archived copies of the caches predate pseudonymisation. Every
    downstream function (group allocation, manual exclusions, the participant
    filters in ``filter_training_data``) matches on ``jhamonNN``, so an
    unmapped key silently loses its training group and crashes ``nht_todf``.
    """
    remapped: Dict[str, Any] = {}
    n_renamed = 0
    for key, value in cache.items():
        match = re.match(r"^(\d+)_", key)
        if match:
            remapped[f"jhamon{match.group(1)}"] = value
            n_renamed += 1
        else:
            remapped[key] = value
    if n_renamed:
        print(f"      note: remapped {n_renamed} real-name keys to jhamonNN pseudonyms")
    return remapped


def load_caches(results_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load and normalise the two cached signal-processing results."""
    nh_path, ik_path = require_caches(results_path)
    _item("NH cache", f"{nh_path.name}  ({nh_path.stat().st_size / 1e6:.1f} MB)")
    _item("IK cache", f"{ik_path.name}  ({ik_path.stat().st_size / 1e6:.1f} MB)")

    _install_legacy_pickle_shim()
    from jhamon_training.saveload import load_obj

    nordict = _pseudonymise(load_obj(name=NH_CACHE, path=results_path))
    ikdict = _pseudonymise(load_obj(name=IK_CACHE, path=results_path))
    _item("NH participants loaded", len(nordict))
    _item("IK participants loaded", len(ikdict))
    return nordict, ikdict


def build_curve_frames(nordict: Dict[str, Any], ikdict: Dict[str, Any]):
    """Convert the nested cache dicts to the long-format curve DataFrames.

    These are the 101-point time-normalised traces (torque, knee angle, knee
    velocity, force, hip angle). ``analyze_discrete_variables`` needs the IK
    frame to derive IK discrete outcomes, and the combined frame to count
    repetitions per set.
    """
    import pandas as pd
    from jhamon_training.data import frames

    nordf = frames.nht_todf(my_dict=nordict)
    ikdf = frames.ikt_todf(my_dict=ikdict)
    _item("NH curve rows", f"{len(nordf):,}")
    _item("IK curve rows", f"{len(ikdf):,}")
    return nordf, ikdf, pd.concat([nordf, ikdf])


def compute_discrete_outcomes(nordict, ikdf, training_df, staging_dir: Path):
    """Per-repetition work, peak/mean torque, ROM and mean knee velocity.

    Also writes the descriptive block of ``training_stats.json`` (rep counts,
    work and velocity means, NH-vs-IK repetition difference) into
    ``staging_dir`` -- ``analyze_discrete_variables`` does that internally, and
    it *truncates* the file, which is precisely why the real stats file is only
    touched at the end of a successful run.
    """
    from jhamon_training.stats.analyses_discrete_vars import analyze_discrete_variables

    results = analyze_discrete_variables(
        nordict=nordict,
        ikdf=ikdf,
        training_df=training_df,
        results_output_path=staging_dir,
    )
    training_disc = results["training_disc"]

    counts = training_disc.groupby("tr_group").size().to_dict()
    _item("repetitions loaded (total)", f"{len(training_disc):,}")
    _item("  NH repetitions", f"{counts.get('NH', 0):,}")
    _item("  IK repetitions", f"{counts.get('IK', 0):,}")
    _item("participants", training_disc["par"].nunique())
    return training_disc


def apply_filters(training_disc):
    """Session/set/participant exclusions plus the 10-35 deg/s velocity window.

    Reports the mean knee velocity per group *before* filtering, because the
    velocity window is signed: a cache storing IK velocity as negative loses
    the entire IK group here, and a silently NH-only "reproduction" would be
    far worse than a crash.
    """
    from jhamon_training.data.utils import filter_training_data

    for group, mean in training_disc.groupby("tr_group")["knee_v_mean"].mean().items():
        n_negative = int(
            (training_disc.loc[training_disc["tr_group"] == group, "knee_v_mean"] < 0).sum()
        )
        flag = f"   <-- {n_negative:,} NEGATIVE values" if n_negative else ""
        _item(f"mean knee velocity {group} (deg/s)", f"{mean:6.2f}{flag}")

    filtered = filter_training_data(training_disc)

    counts = filtered.groupby("tr_group").size().to_dict()
    _item("repetitions after filtering", f"{len(filtered):,}")
    _item("  NH repetitions", f"{counts.get('NH', 0):,}")
    _item("  IK repetitions", f"{counts.get('IK', 0):,}")
    _item("participants after filtering", filtered["par"].nunique())

    for group in ("NH", "IK"):
        if counts.get(group, 0) == 0:
            raise SystemExit(
                f"\nABORTING: the {group} group is empty after filtering.\n"
                "filter_training_data() keeps only repetitions with\n"
                "10 < knee_v_mean < 35, so a cache that stores this velocity with\n"
                "the opposite sign deletes a whole group. Check the mean knee\n"
                "velocities printed above: if one is negative, this cache uses the\n"
                "other sign convention and must be corrected in\n"
                "calculate_ik_discrete_variables() before the model can be fitted.\n"
                "Refusing to fit a model on one group and call it a reproduction."
            )
    return filtered


def run_statistics(training_disc, stats_path: Path) -> Dict[str, Any]:
    """Fit the linear mixed model and the Bland-Altman agreement analysis.

    Both merge their keys into ``stats_path`` on top of the descriptive block
    already written there:

    * ``run_mixed_lmm``  -> ``lmm_*``: group / time / interaction Wald tests and
      the NH-minus-IK estimated marginal difference for the five outcomes, on
      the top-3-repetition cell means.
    * ``run_agreement``  -> ``ba_*``: bias, limits of agreement and concordance
      for the work- and velocity-standardisation, paired by (session x set)
      cell.

    Returns:
        The flat LMM key dict, whose ``lmm_n_cells`` / ``lmm_n_participants``
        are reported on the console.
    """
    from jhamon_training.stats.mixed_lmm_analysis import run_mixed_lmm
    from jhamon_training.stats.agreement import run_agreement

    print(
        "      note: statsmodels prints 'Random effects covariance is singular'\n"
        "      and numpy emits matmul RuntimeWarnings during this stage. They are\n"
        "      raised inside the L-BFGS line search, for candidate parameter values\n"
        "      the optimiser probes and discards, not for the converged fit\n"
        "      (statsmodels/regression/mixed_linear_model.py:1634). They are\n"
        "      expected on this dataset; --check is what proves they are harmless,\n"
        "      because any effect on an estimate would move a key."
    )
    lmm = run_mixed_lmm(training_disc, json_path=stats_path, write_json=True)
    run_agreement(training_disc, json_path=stats_path, write_json=True)
    return lmm["_flat"]


def run_manuscript_statistics(results_path: Path, stats_path: Path) -> None:
    """Add the statistics the manuscript reports that the stages above do not.

    ``run_statistics`` covers the descriptives, the mixed models and the
    agreement analysis. The manuscript also reports whole-curve SPM contrasts,
    the session-by-session progression analysis, baseline equivalence, the
    work-per-repetition and repetition-count tests, and the constant-velocity
    agreement analysis. Those live in three sibling scripts, and until they ran
    here the claim that one command regenerates every published number was false
    -- ``--check`` would report several hundred keys as missing simply because
    the entry point never produced them.

    Each script reads the caches from ``results_path`` but reads and writes the
    stats file at ``stats_path``, so during ``--check`` they operate entirely
    inside the staging directory and the real file is untouched.

    Raises:
        SystemExit: if any stage fails. A partial stats file is worse than none,
            and the caller only installs the file after every stage succeeds.
    """
    import extract_manuscript_stats
    import extract_stats_addendum
    import extract_validity_stats

    stages = (
        ("whole-curve SPM, progression, baseline, counts", extract_manuscript_stats),
        ("standardisation validity and velocity phase", extract_validity_stats),
        ("Welch df, terminal contrasts, cluster angles", extract_stats_addendum),
    )
    argv = ["--results-path", str(results_path), "--stats-path", str(stats_path)]
    for title, module in stages:
        print(f"\n      -> {title}")
        code = module.main(argv)
        if code:
            raise SystemExit(
                f"\nABORTING: '{title}' exited with status {code}. The stats file was "
                "not installed."
            )


def make_figures(training_disc, figures_dir: Path) -> List[str]:
    """Redraw every figure derived from the discrete outcomes.

    A figure that fails to draw must not invalidate statistics that are already
    computed, so each one is attempted independently and its failure recorded.

    Returns:
        List of human-readable failure messages (empty when all figures drew).
    """
    import importlib

    figures_dir.mkdir(parents=True, exist_ok=True)
    _item("output directory", figures_dir)
    failures: List[str] = []

    for module_name, function_name, filename in DISCRETE_FIGURES:
        target = figures_dir / filename
        try:
            module = importlib.import_module(module_name)
            getattr(module, function_name)(training_disc, output_path=target)
            _item(filename, "ok")
        except Exception as exc:  # noqa: BLE001 - report, never abort the run
            failures.append(f"{filename}: {type(exc).__name__}: {exc}")
            _item(filename, f"FAILED ({type(exc).__name__}: {exc})")

    print("      (manuscript Figure 2 and the training-progression panel are not")
    print("       redrawn here: they need the spm1d curve analyses in jhamon_training.py)")
    return failures


# --------------------------------------------------------------------------- #
# --check: diff two stats files
# --------------------------------------------------------------------------- #
def _values_match(new: Any, old: Any, tolerance: float) -> bool:
    """True if two JSON values agree, numbers within ``tolerance``."""
    if isinstance(new, bool) or isinstance(old, bool):
        return new == old
    if isinstance(new, (int, float)) and isinstance(old, (int, float)):
        return math.isclose(float(new), float(old), rel_tol=tolerance, abs_tol=tolerance)
    return new == old


def _format(value: Any) -> str:
    """Render a JSON value for the comparison table, truncating long strings."""
    text = json.dumps(value)
    return text if len(text) <= 40 else text[:37] + '..."'


def compare_stats(
    new_path: Path, reference_path: Path, tolerance: float, strict: bool
) -> bool:
    """Print a key-by-key comparison of two stats files.

    Keys are sorted into four buckets, because they do not mean the same thing:

    ``identical``  same value (numbers within ``tolerance``) -- the good case.
    ``changed``    present in both, different value. Always a failure: a number
                   the manuscript cites has moved.
    ``missing``    in the reference but not in the fresh run. Always a failure:
                   a number the manuscript cites can no longer be produced.
    ``added``      produced now but absent from the reference. Not a failure by
                   default -- this is what a newly-enabled analysis looks like
                   (the ``ba_*`` agreement keys were never written before) --
                   but ``--strict`` treats it as one too.

    Returns:
        True when the reproduction passes.
    """
    if not reference_path.exists():
        print(f"\nNo reference stats file at {reference_path}.")
        print("Nothing to compare against; run without --check to create it.")
        return False

    reference = json.loads(reference_path.read_text())
    fresh = json.loads(new_path.read_text())

    identical: List[str] = []
    changed: List[Tuple[str, Any, Any]] = []
    missing: List[str] = []
    added: List[str] = []

    for key in sorted(set(reference) | set(fresh)):
        in_ref, in_new = key in reference, key in fresh
        if in_ref and in_new:
            if _values_match(fresh[key], reference[key], tolerance):
                identical.append(key)
            else:
                changed.append((key, reference[key], fresh[key]))
        elif in_ref:
            missing.append(key)
        else:
            added.append(key)

    print(f"      reference : {reference_path}   ({len(reference)} keys)")
    print(f"      reproduced: {new_path}   ({len(fresh)} keys)")
    print(f"      tolerance : {tolerance:g} (relative and absolute)")
    print()
    _item("identical", len(identical))
    _item("CHANGED", len(changed))
    _item("MISSING from this run", len(missing))
    _item("added by this run", len(added))

    if changed:
        print("\n  CHANGED -- these published numbers moved:")
        print(f"    {'key':<44}{'reference':>14}{'reproduced':>14}")
        for key, old, new in changed:
            print(f"    {key:<44}{_format(old):>14}{_format(new):>14}")

    if missing:
        print("\n  MISSING -- in the reference file but not produced by this run:")
        for key in missing:
            print(f"    {key:<44}{_format(reference[key])}")

    if added:
        print("\n  ADDED -- produced by this run, absent from the reference file:")
        for key in added:
            print(f"    {key:<44}{_format(fresh[key])}")
        if not strict:
            print("    (not counted as a failure; pass --strict to fail on these too)")

    passed = not changed and not missing and not (strict and added)
    print()
    if passed:
        print(_rule())
        print(
            f"PASS -- {len(identical)} of {len(reference)} reference keys reproduce "
            f"within {tolerance:g}."
        )
        if added:
            print(f"        {len(added)} new key(s) produced (see above).")
        print(_rule())
    else:
        print(_rule())
        print(
            f"FAIL -- {len(changed)} changed, {len(missing)} missing"
            + (f", {len(added)} added (strict)" if strict and added else "")
            + "."
        )
        if changed or missing:
            print("        The cached data no longer produce the published numbers.")
        else:
            print(
                f"        Every reference key reproduced; --strict also rejects the "
                f"{len(added)} new key(s)."
            )
        print(_rule())
    return passed


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reproduce.py",
        description=(
            "Rebuild the jHamON training paper's statistics from the cached "
            "results (nht_results.pkl / ikt_results.pkl). Does not need the raw "
            "data drive."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python reproduce.py                 rebuild and install training_stats.json\n"
            "  python reproduce.py --check         rebuild to a temp dir and diff (exit 1 on drift)\n"
            "  python reproduce.py --figures       also redraw the discrete-outcome figures\n"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "regression test: write the rebuilt stats to a temporary directory "
            "and compare them key by key with the existing training_stats.json "
            "instead of replacing it. Exits non-zero if any number moved."
        ),
    )
    parser.add_argument(
        "--results-path",
        metavar="PATH",
        default=None,
        help=(
            "folder holding the caches and training_stats.json. Defaults to "
            "RESULTS_TRAINING_PATH as resolved by jhamon_training.pathutils "
            "(overridable with $JHAMON_RESULTS)."
        ),
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="also regenerate the figures that derive from the discrete outcomes.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        metavar="EPS",
        help=(
            "relative and absolute tolerance when comparing numbers in --check "
            f"(default {DEFAULT_TOLERANCE:g}; the JSON values are pre-rounded, so "
            "a true reproduction matches exactly)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "in --check, also fail when the run produces keys the reference file "
            "does not have (by default extra keys are reported but tolerated)."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # The libraries below write to stderr; keep our stdout line-buffered so the
    # two streams stay interleaved when the run is piped to a file or to `tee`.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    results_path = resolve_results_path(args.results_path)
    stats_reference = results_path / STATS_FILENAME
    mode = "CHECK (nothing will be overwritten)" if args.check else "REBUILD"

    print(_rule())
    print("jHamON training -- reproduction from cached results")
    print(_rule())
    _item("mode", mode)
    _item("results path", results_path)
    _item("stats file", stats_reference)
    _item("raw data drive", "NOT REQUIRED -- nothing is read from /Volumes/AJMA/")

    staging_dir = Path(tempfile.mkdtemp(prefix="jhamon_reproduce_"))
    staging_stats = staging_dir / STATS_FILENAME
    figure_failures: List[str] = []
    keep_staging = False

    try:
        _stage(1, "Locating and loading caches")
        nordict, ikdict = load_caches(results_path)

        _stage(2, "Rebuilding time-normalised curve dataframes")
        _, ikdf, training_df = build_curve_frames(nordict, ikdict)

        _stage(3, "Computing per-repetition discrete outcomes")
        training_disc = compute_discrete_outcomes(
            nordict, ikdf, training_df, staging_dir
        )

        _stage(4, "Applying exclusions and the velocity filter")
        training_disc = apply_filters(training_disc)

        _stage(5, "Fitting the linear mixed model and the agreement analysis")
        lmm_flat = run_statistics(training_disc, staging_stats)
        _item("LMM participants", lmm_flat["lmm_n_participants"])
        _item("  NH / IK", f"{lmm_flat['lmm_n_nh']} / {lmm_flat['lmm_n_ik']}")
        _item(
            "LMM cells (participant x session)",
            f"{lmm_flat['lmm_n_cells']} of "
            f"{lmm_flat['lmm_n_participants'] * 15} possible",
        )

        _stage(6, "Computing the remaining manuscript statistics")
        run_manuscript_statistics(results_path, staging_stats)
        _item("total keys", len(json.loads(staging_stats.read_text())))

        if args.figures:
            figure_failures = make_figures(training_disc, staging_dir / "figures")
            # In --check nothing is installed, so the only copy of the figures
            # is the staging one; keep it rather than deleting what was asked for.
            keep_staging = args.check

        if args.check:
            _stage(7, "Comparing with the stats file on disk")
            passed = compare_stats(
                staging_stats, stats_reference, args.tolerance, args.strict
            )
            if not passed:
                keep_staging = True
                print(f"\nRebuilt stats kept for inspection: {staging_stats}")
            exit_code = 0 if passed else 1
        else:
            _stage(7, "Installing results")
            results_path.mkdir(parents=True, exist_ok=True)
            # Same-filesystem atomic replace: copy next to the target, then
            # rename over it, so the stats file is never partially written.
            temporary_target = stats_reference.with_suffix(".json.tmp")
            shutil.copyfile(staging_stats, temporary_target)
            os.replace(temporary_target, stats_reference)
            n_keys = len(json.loads(stats_reference.read_text()))
            _item("wrote", f"{stats_reference}  ({n_keys} keys)")

            # The per-cluster SPM detail is written next to the staging stats
            # file and is what the manuscript reads for cluster extents.
            detail = staging_dir / "manuscript_stats_detail.json"
            if detail.exists():
                shutil.copyfile(detail, results_path / detail.name)
                _item("wrote", results_path / detail.name)

            if args.figures:
                figures_target = results_path / "figures"
                figures_target.mkdir(parents=True, exist_ok=True)
                for figure in sorted((staging_dir / "figures").glob("*.png")):
                    shutil.copyfile(figure, figures_target / figure.name)
                    _item("wrote", figures_target / figure.name)
            exit_code = 0

        if figure_failures:
            print("\nFIGURE FAILURES (statistics above are unaffected):")
            for failure in figure_failures:
                print(f"  - {failure}")
            exit_code = exit_code or 1

        return exit_code
    finally:
        if keep_staging:
            print(f"Staging directory retained: {staging_dir}")
        else:
            shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
