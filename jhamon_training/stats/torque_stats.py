import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any


def calculate_torque_stats(training_disc: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate mean and peak torque statistics for each training group.

    Args:
        training_disc: DataFrame containing the training data with torque measurements

    Returns:
        Dictionary containing the calculated statistics
    """
    # Calculate mean torque statistics
    mean_torque_stats = training_disc.groupby("tr_group")["mean_torque"].agg(
        ["mean", "std"]
    )

    # Calculate peak torque statistics
    peak_torque_stats = training_disc.groupby("tr_group")["peak_torque"].agg(
        ["mean", "std"]
    )

    # Format statistics for reporting
    torque_stats = {
        "nh_mean_torque": np.round(np.array(mean_torque_stats.loc["NH", "mean"]), 1),
        "nh_mean_torque_sd": np.round(np.array(mean_torque_stats.loc["NH", "std"]), 1),
        "ik_mean_torque": np.round(np.array(mean_torque_stats.loc["IK", "mean"]), 1),
        "ik_mean_torque_sd": np.round(np.array(mean_torque_stats.loc["IK", "std"]), 1),
        "nh_peak_torque": np.round(np.array(peak_torque_stats.loc["NH", "mean"]), 1),
        "nh_peak_torque_sd": np.round(np.array(peak_torque_stats.loc["NH", "std"]), 1),
        "ik_peak_torque": np.round(np.array(peak_torque_stats.loc["IK", "mean"]), 1),
        "ik_peak_torque_sd": np.round(np.array(peak_torque_stats.loc["IK", "std"]), 1),
    }

    return torque_stats


def report_torque_stats(training_disc: pd.DataFrame, output_path: Path) -> None:
    """
    Calculate and report torque statistics, saving them to the training_stats.json file.

    Args:
        training_disc: DataFrame containing the training data with torque measurements
        output_path: Path where to save the statistics file
    """
    # Calculate statistics
    torque_stats = calculate_torque_stats(training_disc)

    # Print statistics to console
    print("\nAverage Torque Values by Training Group:")
    print("----------------------------------------")

    print("\nMean Torque (Nm):")
    mean_torque_df = pd.DataFrame(
        {
            "NH": [
                torque_stats["nh_mean_torque"],
                torque_stats["nh_mean_torque_sd"],
            ],
            "IK": [
                torque_stats["ik_mean_torque"],
                torque_stats["ik_mean_torque_sd"],
            ],
        },
        index=["mean", "std"],
    )
    print(mean_torque_df)

    print("\nPeak Torque (Nm):")
    peak_torque_df = pd.DataFrame(
        {
            "NH": [
                torque_stats["nh_peak_torque"],
                torque_stats["nh_peak_torque_sd"],
            ],
            "IK": [
                torque_stats["ik_peak_torque"],
                torque_stats["ik_peak_torque_sd"],
            ],
        },
        index=["mean", "std"],
    )
    print(peak_torque_df)

    # Load existing stats or create new dict
    stats_path = output_path / "training_stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {}

    # Update stats with torque data
    stats.update(torque_stats)

    # Save updated stats
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nTorque statistics saved to {stats_path}")
