import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib import colormaps
from typing import Dict, Any
import pandas as pd


def plot_training_progression(
    training_df: pd.DataFrame,
    tr_progression: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Generates a figure showing training progression with torque curves and SPM statistics.

    Args:
        training_df: DataFrame containing the training data
        tr_progression: Dictionary containing SPM analysis results for NH and IK groups
        output_path: Path where to save the figure
    """
    # Set up the figure with publication-quality settings
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.figsize": (12, 10),
            "axes.linewidth": 0.5,
        }
    )

    # Create figure with subplots and adjust spacing
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.05, wspace=0.3
    )  # Minimal vertical spacing

    ax_nh = fig.add_subplot(gs[0, 0])
    ax_ik = fig.add_subplot(gs[0, 1])
    ax_nh_spm = fig.add_subplot(gs[1, 0], sharex=ax_nh)  # Share x-axis with top plot
    ax_ik_spm = fig.add_subplot(gs[1, 1], sharex=ax_ik)  # Share x-axis with top plot

    fig.set_facecolor("white")

    # Create x-axis points (0-100%)
    x_points = np.linspace(0, 100, 101)

    # Plot NH torque curves (top left)
    nh_data = training_df[
        (training_df["var"] == "torque") & (training_df["tr_group"] == "NH")
    ]
    plot_session_curves(nh_data, ax_nh, "NH")
    ax_nh.set_xticks([])  # Remove x-axis ticks for top plots

    # Plot IK torque curves (top right)
    ik_data = training_df[
        (training_df["var"] == "torque") & (training_df["tr_group"] == "IK")
    ]
    plot_session_curves(ik_data, ax_ik, "IK")
    ax_ik.set_xticks([])  # Remove x-axis ticks for top plots

    # Plot SPM statistics (bottom row)
    if "torqueNH" in tr_progression:
        plot_spm_results(tr_progression["torqueNH"], ax_nh_spm)
        ax_nh_spm.set_xlabel("Time (%)")  # Add x-axis label for bottom plot
        ax_nh_spm.set_xticks([0, 20, 40, 60, 80, 100])  # Add x-axis ticks
    if "torqueIK" in tr_progression:
        plot_spm_results(tr_progression["torqueIK"], ax_ik_spm)
        ax_ik_spm.set_xlabel("Time (%)")  # Add x-axis label for bottom plot
        ax_ik_spm.set_xticks([0, 20, 40, 60, 80, 100])  # Add x-axis ticks

    # Add panel labels
    for i, ax in enumerate([ax_nh, ax_ik, ax_nh_spm, ax_ik_spm]):
        label = chr(65 + i)  # A, B, C, D
        ax.text(
            -0.1,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    # Ensure consistent y-axis limits for top and bottom subplots
    y_min_top = min(ax_nh.get_ylim()[0], ax_ik.get_ylim()[0])
    y_max_top = max(ax_nh.get_ylim()[1], ax_ik.get_ylim()[1])
    ax_nh.set_ylim(y_min_top, y_max_top)
    ax_ik.set_ylim(y_min_top, y_max_top)

    y_min_bottom = min(ax_nh_spm.get_ylim()[0], ax_ik_spm.get_ylim()[0])
    y_max_bottom = max(ax_nh_spm.get_ylim()[1], ax_ik_spm.get_ylim()[1])
    ax_nh_spm.set_ylim(y_min_bottom, y_max_bottom)
    ax_ik_spm.set_ylim(y_min_bottom, y_max_bottom)

    # Adjust layout and save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_session_curves(data: pd.DataFrame, ax: Axes, group: str) -> None:
    """Plot torque curves for each session with color gradient."""
    # Create x-axis points (0-100%)
    x_points = np.linspace(0, 100, 101)

    # Get unique sessions and sort them
    sessions = sorted(data["trses"].unique(), key=lambda x: int(x.split("_")[1]))

    # Create colormap
    cmap = colormaps["viridis"]
    colors = [cmap(i / len(sessions)) for i in range(len(sessions))]

    # First plot all individual curves
    for session, color in zip(sessions, colors):
        session_data = data[data["trses"] == session]
        for _, rep_data in session_data.groupby(["par", "set", "rep"]):
            if len(rep_data) == len(x_points):
                ax.plot(
                    x_points, rep_data["value"].values, color=color, alpha=0.1, zorder=1
                )

    # Then plot all average curves on top
    for session, color in zip(sessions, colors):
        session_data = data[data["trses"] == session]
        avg_curve = np.zeros(len(x_points), dtype=float)
        count = 0
        for _, rep_data in session_data.groupby(["par", "set", "rep"]):
            if len(rep_data) == len(x_points):
                avg_curve += rep_data["value"].values
                count += 1
        if count > 0:
            avg_curve /= count
            ax.plot(x_points, avg_curve, color=color, linewidth=2, alpha=0.8, zorder=2)

    # Add labels and grid
    ax.set_ylabel("Torque (N·m)")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Only add colorbar to the NH plot (top left)
    if group == "NH":
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(1, len(sessions)))
        cax = ax.inset_axes(
            (0.05, 0.7, 0.02, 0.2)
        )  # (left, bottom, width, height) in axes coordinates
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Session number", fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_frame_on(True)
        cbar.ax.patch.set_alpha(0.8)  # Semi-transparent background


def plot_spm_results(spm_data: Dict[str, Any], ax: Axes) -> None:
    """Plot SPM statistics."""
    # Get all training sessions
    sessions = sorted(
        [k for k in spm_data.keys() if k.startswith("tr_")],
        key=lambda x: int(x.split("_")[1]),
    )

    if not sessions:
        return

    # Create x-axis points (0-100%)
    x_axis = np.linspace(0, 100, len(spm_data[sessions[0]]["z"]))

    # Get the last session's threshold as reference
    reference_threshold = spm_data[sessions[-1]]["zstar"]

    # Create colormap
    cmap = colormaps["viridis"]
    colors = [cmap(i / len(sessions)) for i in range(len(sessions))]

    # Plot SPM{F} curves for each session
    for session, color in zip(sessions, colors):
        z_values = spm_data[session]["z"]
        z_threshold = spm_data[session]["zstar"]

        # Plot SPM{F} curve
        ax.plot(x_axis, z_values, color=color, lw=0.5, alpha=0.3)

        # Fill areas above threshold
        ax.fill_between(
            x_axis,
            z_values,
            z_threshold,
            where=z_values > z_threshold,
            facecolor=color,
            alpha=0.1,
        )

    # Plot threshold line using reference threshold
    ax.axhline(reference_threshold, color="k", linestyle="--", lw=1)
    ax.axhline(0, color="k", linestyle="--", lw=0.5)

    # Add labels
    ax.set_xlabel("Time (%)")
    ax.set_ylabel("SPM{F}")
    ax.grid(False)
