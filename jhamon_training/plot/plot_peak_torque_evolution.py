import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from matplotlib.axes import Axes
import re


def plot_peak_torque_evolution(
    training_disc: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plots the evolution of peak torque and corresponding knee angles across training sessions
    for each participant's 3 highest repetitions, with NH and IK groups plotted in the same subplots.
    Optimized for scientific publication.

    Args:
        training_disc: DataFrame containing the training data with peak torque values.
        output_path: Path where the plot will be saved.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = training_disc.copy()

    # Extract numeric part from trses column if trses_num doesn't exist
    if "trses_num" not in df.columns:
        df["trses_num"] = df["trses"].str.extract(r"tr_(\d+)").astype(int)

    # Debug prints for angle values
    print("\nIK angle values:")
    print(df[df["tr_group"] == "IK"]["angle_at_peak_torque"].describe())

    print("\nNH angle values:")
    print(df[df["tr_group"] == "NH"]["angle_at_peak_torque"].describe())

    # Calculate ROM for each group
    # For IK: Use the pre-calculated knee_ROM
    df.loc[df["tr_group"] == "IK", "rom"] = df.loc[df["tr_group"] == "IK", "knee_ROM"]

    # For NH: Also use the pre-calculated knee_ROM
    df.loc[df["tr_group"] == "NH", "rom"] = df.loc[df["tr_group"] == "NH", "knee_ROM"]

    # Set up the figure with publication-quality settings
    plt.rcParams.update(
        {
            "font.size": 10,  # Increased from 8
            "axes.labelsize": 10,  # Increased from 8
            "axes.titlesize": 10,  # Increased from 8
            "xtick.labelsize": 10,  # Increased from 8
            "ytick.labelsize": 10,  # Increased from 8
            "legend.fontsize": 10,  # Increased from 8
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.figsize": (10.8, 7.2),  # Wider figure for 2x2 subplots
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "lines.linewidth": 1,
            "lines.markersize": 4,
        }
    )

    # Create figure with four subplots (2x2)
    fig, ((ax_mean_torque, ax_peak_torque), (ax_rom, ax_angle)) = plt.subplots(
        2,
        2,
        figsize=(10.8, 7.2),
        sharex=True,  # Share x-axis across all subplots
        gridspec_kw={"wspace": 0.3, "hspace": 0.1},  # Further reduced vertical spacing
    )

    # Define colors for the plots
    ik_color = "#1f77b4"  # Blue
    nh_color = "#ff7f0e"  # Orange

    # Process data for both groups
    for group, color in [("IK", ik_color), ("NH", nh_color)]:
        group_data = df[df["tr_group"] == group]
        sessions = sorted(group_data["trses_num"].unique())

        # Calculate mean and SD for each session (torque, angle, and ROM)
        means_torque = []
        sds_torque = []
        means_angle = []
        sds_angle = []
        means_rom = []
        sds_rom = []
        means_peak_torque = []
        sds_peak_torque = []

        for session in sessions:
            session_data = group_data[group_data["trses_num"] == session]
            # Get top 3 repetitions based on peak torque
            top_3_reps = (
                session_data.groupby("par")
                .apply(lambda x: x.nlargest(3, "peak_torque"))
                .reset_index(drop=True)
            )

            # Calculate statistics for mean torque (using the same top 3 reps)
            means_torque.append(top_3_reps["mean_torque"].mean())
            sds_torque.append(top_3_reps["mean_torque"].std())

            # Calculate statistics for peak torque
            means_peak_torque.append(top_3_reps["peak_torque"].mean())
            sds_peak_torque.append(top_3_reps["peak_torque"].std())

            # Calculate statistics for angle
            means_angle.append(top_3_reps["angle_at_peak_torque"].mean())
            sds_angle.append(top_3_reps["angle_at_peak_torque"].std())

            # Calculate statistics for ROM
            means_rom.append(top_3_reps["rom"].mean())
            sds_rom.append(top_3_reps["rom"].std())

            # Add jittered points for mean torque
            jitter = np.random.normal(0, 0.1, size=len(top_3_reps))
            ax_mean_torque.scatter(
                [session + j for j in jitter],
                top_3_reps["mean_torque"],
                color=color,
                alpha=0.3,
                s=10,
                edgecolors="none",
            )

            # Add jittered points for peak torque
            ax_peak_torque.scatter(
                [session + j for j in jitter],
                top_3_reps["peak_torque"],
                color=color,
                alpha=0.3,
                s=10,
                edgecolors="none",
            )

            # Add jittered points for angle
            ax_angle.scatter(
                [session + j for j in jitter],
                top_3_reps["angle_at_peak_torque"],
                color=color,
                alpha=0.3,
                s=10,
                edgecolors="none",
            )

            # Add jittered points for ROM
            ax_rom.scatter(
                [session + j for j in jitter],
                top_3_reps["rom"],
                color=color,
                alpha=0.3,
                s=10,
                edgecolors="none",
            )

        # Plot mean and SD for mean torque
        ax_mean_torque.errorbar(
            sessions,
            means_torque,
            yerr=sds_torque,
            fmt="o-",
            color=color,
            markersize=4,
            capsize=2,
            capthick=0.5,
            elinewidth=0.5,
            label=f"{group} Mean ± SD",
        )

        # Plot mean and SD for peak torque
        ax_peak_torque.errorbar(
            sessions,
            means_peak_torque,
            yerr=sds_peak_torque,
            fmt="o-",
            color=color,
            markersize=4,
            capsize=2,
            capthick=0.5,
            elinewidth=0.5,
            label=f"{group} Mean ± SD",
        )

        # Plot mean and SD for angle
        ax_angle.errorbar(
            sessions,
            means_angle,
            yerr=sds_angle,
            fmt="o-",
            color=color,
            markersize=4,
            capsize=2,
            capthick=0.5,
            elinewidth=0.5,
            label=f"{group} Mean ± SD",
        )

        # Plot mean and SD for ROM
        ax_rom.errorbar(
            sessions,
            means_rom,
            yerr=sds_rom,
            fmt="o-",
            color=color,
            markersize=4,
            capsize=2,
            capthick=0.5,
            elinewidth=0.5,
            label=f"{group} Mean ± SD",
        )

    # Customize subplots
    for ax in [ax_mean_torque, ax_peak_torque, ax_rom, ax_angle]:
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend(frameon=False)
        # Set x-axis ticks to show all training sessions
        ax.set_xticks(range(1, 16))  # Show all sessions from 1 to 15

    # Set x-labels only for bottom plots
    ax_rom.set_xlabel("Training Session")
    ax_angle.set_xlabel("Training Session")

    # Set y-labels
    ax_mean_torque.set_ylabel("Mean Torque (N·m)")
    ax_peak_torque.set_ylabel("Peak Torque (N·m)")
    ax_rom.set_ylabel("Range of Motion (°)")
    ax_angle.set_ylabel("Knee Angle at Peak Torque (°)")

    # Add panel labels with adjusted vertical position
    for i, ax in enumerate([ax_mean_torque, ax_peak_torque, ax_rom, ax_angle]):
        label = chr(65 + i)  # A, B, C, D
        y_pos = 1.02 if i < 2 else 1.05  # Lower position for bottom panels
        ax.text(
            -0.1,
            y_pos,
            label,
            transform=ax.transAxes,
            fontsize=12,  # Increased from 10
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    # Share y-axis limits between mean and peak torque plots
    y_min_torque = min(
        ax_mean_torque.get_ylim()[0],
        ax_peak_torque.get_ylim()[0],
    )
    y_max_torque = max(
        ax_mean_torque.get_ylim()[1],
        ax_peak_torque.get_ylim()[1],
    )
    ax_mean_torque.set_ylim(y_min_torque, y_max_torque)
    ax_peak_torque.set_ylim(y_min_torque, y_max_torque)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
