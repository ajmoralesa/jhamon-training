import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from matplotlib.axes import Axes


def plot_torque_comparison(
    training_df: pd.DataFrame,
    ax: Axes,
):
    """Plots participant-averaged torque comparison curves onto a given Axes object.

    Averages are calculated for each unique combination of session, set, and rep
    across participants before plotting.

    Args:
        training_df: DataFrame containing the training data.
        ax: The Matplotlib Axes object to plot on.
    """
    # Create x-axis points (0-100%)
    x_points = np.linspace(0, 100, 101)

    # Filter data for torque and groups
    training_df_torque = training_df[training_df["var"] == "torque"].copy()
    # Ensure required columns are numeric if needed for grouping/averaging
    # training_df_torque[[\'trses\', \'set\', \'rep\']] = training_df_torque[[\'trses\', \'set\', \'rep\']].apply(pd.to_numeric, errors=\'coerce\')

    nh_data = training_df_torque[training_df_torque["tr_group"] == "NH"]
    ik_data = training_df_torque[training_df_torque["tr_group"] == "IK"]

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
    # Create a unique identifier for each averaged curve
    nh_avg["curve_id"] = nh_avg.apply(
        lambda row: f"{row['trses']}_{row['set']}_{row['rep']}", axis=1
    )

    # Plot averaged NH curves
    for curve_id, group in nh_avg.groupby("curve_id"):
        sorted_group = group.sort_values("timepoint")
        if len(sorted_group["value"]) == len(x_points):
            y_values = np.array(sorted_group["value"].values)
            ax.plot(x_points, y_values, color=nh_color, alpha=0.05)
        # else: # Optional: Add warning/logging if needed
        #     print(
        #         f"Warning: Skipping averaged NH curve {curve_id} due to length mismatch..."
        #     )

    # Average IK data
    ik_avg = (
        ik_data.groupby(["timepoint", "trses", "set", "rep"])["value"]
        .mean()
        .reset_index()
    )
    # Create a unique identifier for each averaged curve
    ik_avg["curve_id"] = ik_avg.apply(
        lambda row: f"{row['trses']}_{row['set']}_{row['rep']}", axis=1
    )

    # Plot averaged IK curves
    for curve_id, group in ik_avg.groupby("curve_id"):
        sorted_group = group.sort_values("timepoint")
        if len(sorted_group["value"]) == len(x_points):
            y_values = np.array(sorted_group["value"].values)
            ax.plot(x_points, y_values, color=ik_color, alpha=0.05)
        # else: # Optional: Add warning/logging if needed
        #     print(
        #         f"Warning: Skipping averaged IK curve {curve_id} due to length mismatch..."
        #     )

    # --- Add Legend ---
    # Create dummy lines for the legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=nh_color, lw=2, label="NH"),
        Line2D([0], [0], color=ik_color, lw=2, label="IK"),
    ]
    ax.legend(handles=legend_elements, title="tr_group")

    ax.set_xlabel("Time (%)")
    ax.set_ylabel("Torque (Nm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)
