import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from matplotlib.axes import Axes


def plot_velocity_comparison(
    training_df: pd.DataFrame,
    ax: Axes,
):
    """Plots filtered, participant-averaged knee velocity comparison curves onto a given Axes object.

    Filters out extreme values (velocity > 300 deg/s or near 0 during mid-cycle)
    before averaging across participants for each unique session/set/rep.

    Args:
        training_df: DataFrame containing the training data (potentially pre-filtered).
        ax: The Matplotlib Axes object to plot on.
    """
    # Create x-axis points (0-100%)
    x_points = np.linspace(0, 100, 101)

    # --- Filter data ---
    # Filter for knee velocity
    training_df_velocity = training_df[training_df["var"] == "knee_v"].copy()

    # Filter out extreme high values
    training_df_velocity = training_df_velocity[training_df_velocity["value"] <= 300]

    # Filter out near-zero values in the central part (e.g., 25% to 75% cycle)
    central_part_mask = (training_df_velocity["timepoint"] >= 25) & (
        training_df_velocity["timepoint"] <= 75
    )
    near_zero_mask = np.abs(training_df_velocity["value"]) < 5
    training_df_velocity = training_df_velocity[~(central_part_mask & near_zero_mask)]

    # --- Group and Plot Individual Lines ---
    nh_data = training_df_velocity[training_df_velocity["tr_group"] == "NH"]
    ik_data = training_df_velocity[training_df_velocity["tr_group"] == "IK"]

    # Get colors from colormap
    cmap = plt.cm.get_cmap("cividis")
    nh_color = cmap(0.2)
    ik_color = cmap(0.8)

    # --- Average across participants and plot ---

    # Average NH data
    nh_avg = (
        nh_data.groupby(["timepoint", "trses", "set", "rep"])["value"]
        .mean()
        .reset_index()
    )
    nh_avg["curve_id"] = nh_avg.apply(
        lambda row: f"{row['trses']}_{row['set']}_{row['rep']}", axis=1
    )

    # Plot averaged NH curves
    for curve_id, group in nh_avg.groupby("curve_id"):
        sorted_group = group.sort_values("timepoint")
        if len(sorted_group["value"]) == len(x_points):
            y_values = np.array(sorted_group["value"].values)
            ax.plot(x_points, y_values, color=nh_color, alpha=0.05)

    # Average IK data
    ik_avg = (
        ik_data.groupby(["timepoint", "trses", "set", "rep"])["value"]
        .mean()
        .reset_index()
    )
    ik_avg["curve_id"] = ik_avg.apply(
        lambda row: f"{row['trses']}_{row['set']}_{row['rep']}", axis=1
    )

    # Plot averaged IK curves
    for curve_id, group in ik_avg.groupby("curve_id"):
        sorted_group = group.sort_values("timepoint")
        if len(sorted_group["value"]) == len(x_points):
            y_values = np.array(sorted_group["value"].values)
            ax.plot(x_points, y_values, color=ik_color, alpha=0.05)

    # --- Add Legend ---
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=nh_color, lw=2, label="NH"),
        Line2D([0], [0], color=ik_color, lw=2, label="IK"),
    ]
    ax.legend(handles=legend_elements, title="tr_group")

    ax.set_xlabel("Time (%)")
    ax.set_ylabel("Velocity (deg/s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)
