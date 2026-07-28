# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Biomechanics research analysis pipeline comparing Nordic Hamstring (NH) vs Isokinetic (IK) leg training across 15 sessions and 27 participants. Hybrid Python/R project with Quarto-based reporting for the results of the scienitfic manuscript.

## Running the Project

```bash
# Activate Python virtual environment
source venv/bin/activate

# Run the main analysis pipeline
python jhamon_training.py

# Extract anthropometric data from Excel files
python run_anthropometrics.py

# Generate Quarto report (DOCX output)
quarto render jhamon_results.qmd
```

There are no tests, linters, or CI/CD pipelines configured.

## Key Paths

- **Raw data**: `/Volumes/AJMA/` (external drive, must be mounted)
- **Results output**: `~/Library/CloudStorage/OneDrive-insep/RECHERCHE/Projects/2019_jHamON/_RESULTS_TRAINING/` (figures, feather caches, stats JSON). The SynologyDrive and VPN-mounted "Espace Privé" locations are retired; OneDrive is canonical because it syncs locally and stays reachable. Resolved by `resolve_results_path()` in `jhamon_training/pathutils.py`, which probes an ordered candidate list and exposes the winner as `RESULTS_TRAINING_PATH`. Set `JHAMON_RESULTS=/path/to/folder` to override, or add a new candidate at the top of `_RESULTS_PATH_CANDIDATES` if the folder moves again.
- **R scripts**: `R/` directory, managed with `renv` (auto-activated via `.Rprofile`)

## Architecture

The main orchestrator is `jhamon_training.py`. It follows this pipeline:

1. **Data loading** — `jhamon_training/__init__.py:check_result_file()` loads raw mechanical/kinematic data via `data/nordic.py` (NH) and `data/ik.py` (IK), caching results as pickle files. Nordic processing uses joblib for parallelism.
2. **DataFrame conversion** — `data/frames.py` converts nested dicts to long-format pandas DataFrames, cached as feather files.
3. **Filtering** — `data/utils.py` applies manual exclusions (out-of-range ROM, force, torque) and session/set filters.
4. **Discrete variables** — `stats/analyses_discrete_vars.py:analyze_discrete_variables()` computes work (trapezoidal integration), peak/mean torque, knee angular velocity, and ROM per repetition.
5. **Statistical analysis** — SPM via `spm1d` (group comparisons, repeated measures ANOVA) and mixed ANOVA via `pingouin`.
6. **Visualization** — `plot/` subpackage generates PNG figures.
7. **Reporting** — `jhamon_results.qmd` reads `training_stats.json` for dynamic DOCX reports.

## Data Structures

Nordic data dict (`nordict`): nested `{participant: {session: {set: [[indexes], [values]]}}}`.

Training DataFrames use long format with columns: `par`, `trses`, `set`, `rep`, `var`, `value`, `tr_group`, `timepoint`, `all_labels`.

## Dependencies

Python deps in `requirements.txt`. Additional unlisted but required: `spm1d`, `pingouin`, `joblib`, `openpyxl`. The package installs itself as an editable local dependency (`-e .`).

This project no longer depends on the separate `jhamon` base library: the only functions used from it (`butter_low_filter`, `_detect_onset`, `dame_inercia`/`_segmentCMaxes`) are now vendored into `jhamon_training/signal/` (see `signal/nordic.py` header for provenance), making the repo self-contained. One exception: `signal/mech.py` still imports base `jhamon` inside unused EMG/MVC helpers, which the NH/IK training pipeline never calls.

R deps managed by `renv` (lock file: `renv.lock`).
