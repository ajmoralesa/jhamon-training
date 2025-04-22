"""Module for handling repetition statistics in training data.

This module provides functions to analyze and save repetition statistics
from training data to JSON files for reporting purposes.
"""

import json
from pathlib import Path
import pandas as pd
from jhamon_training.plot.plot_repetition_count import plot_repetition_count


def save_repetition_stats(
    training_disc: pd.DataFrame, results_output_path: Path
) -> None:
    """Save repetition statistics to JSON file.

    Args:
        training_disc: DataFrame containing training data
        results_output_path: Path to save the statistics file
    """
    # Count repetitions
    nh_reps = len(training_disc[training_disc["tr_group"] == "NH"])
    ik_reps = len(training_disc[training_disc["tr_group"] == "IK"])

    # Save to statistics file
    stats_path = results_output_path / "training_stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {}

    stats.update(
        {"nh_reps": nh_reps, "ik_reps": ik_reps, "total_reps": nh_reps + ik_reps}
    )

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    # Plot repetition count
    figures_path = results_output_path / "figures"
    plot_repetition_count(
        training_disc, output_path=figures_path / "repetition_count.png"
    )
