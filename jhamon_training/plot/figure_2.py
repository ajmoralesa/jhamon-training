import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np  # Assuming np might be used within the imported plot functions

# Assuming these plotting functions are importable from the project structure
# Adjust imports if these files are located elsewhere (e.g., in submodules)
try:
    from plot_torque_comparison import plot_torque_comparison
    from plot_velocity_comparison import plot_velocity_comparison

    # Assuming spm_plot is in a submodule as per the original import
    from jhamon_training.plot.spm_plot import plot_spm_comparison
except ImportError as e:
    print(
        f"Warning: Could not import plotting functions. Ensure they are accessible: {e}"
    )

    # Define dummy functions if imports fail to avoid crashing later
    def plot_torque_comparison(*args, **kwargs):
        print("Warning: plot_torque_comparison not found.")

    def plot_velocity_comparison(*args, **kwargs):
        print("Warning: plot_velocity_comparison not found.")

    def plot_spm_comparison(*args, **kwargs):
        print("Warning: plot_spm_comparison not found.")


def generate_figure_2(
    training_dfilt: pd.DataFrame, torqcomp: dict, kneevcom: dict, figures_path: Path
) -> None:
    """
    Generates and saves the combined 2x2 figure comparing Torque and Velocity
    between Nordic and IK training, including SPM analysis.

    Args:
        training_dfilt: Filtered DataFrame containing training data.
        torqcomp: Dictionary containing torque SPM comparison results.
        kneevcom: Dictionary containing knee velocity SPM comparison results.
        figures_path: Path object pointing to the directory where the figure will be saved.
    """
    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.set_facecolor("white")  # Set background color to white

    # Panel A: Torque Comparison
    plot_torque_comparison(training_dfilt, ax=axes[0, 0])
    axes[0, 0].set_title("")

    # Panel B: Torque SPM
    # Ensure 'ti' key exists before accessing
    if torqcomp and "ti" in torqcomp:
        plot_spm_comparison(torqcomp["ti"], variable_name="torque", ax=axes[0, 1])
    else:
        print("Warning: torqcomp data or 'ti' key missing for Panel B.")
        axes[0, 1].text(
            0.5,
            0.5,
            "Torque SPM data missing",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
    axes[0, 1].set_title("")

    # Panel C: Velocity Comparison
    plot_velocity_comparison(training_dfilt, ax=axes[1, 0])
    axes[1, 0].set_title("")

    # Panel D: Velocity SPM
    # Ensure 'ti' key exists before accessing
    if kneevcom and "ti" in kneevcom:
        plot_spm_comparison(kneevcom["ti"], variable_name="knee_v", ax=axes[1, 1])
    else:
        print("Warning: kneevcom data or 'ti' key missing for Panel D.")
        axes[1, 1].text(
            0.5,
            0.5,
            "Velocity SPM data missing",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
    axes[1, 1].set_title("")

    # Add panel labels (A, B, C, D)
    panel_labels = ["A", "B", "C", "D"]
    for i, ax in enumerate(axes.flat):
        ax.text(
            -0.1,
            1.05,
            panel_labels[i],
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="right",
        )
        # Remove individual x-axis labels for top row
        if i < 2:
            ax.set_xlabel("")
        # Remove individual y-axis labels for right column
        if i % 2 != 0:
            ax.set_ylabel("")

    # Adjust layout to prevent overlap
    plt.tight_layout(pad=2.0)  # Add some padding

    # Save the combined figure
    combined_figure_path = figures_path / "Figure_2_training_comparison.png"
    try:
        plt.savefig(
            combined_figure_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        print(f"Figure 2 saved to: {combined_figure_path}")
    except Exception as e:
        print(f"Error saving Figure 2: {e}")
    finally:
        plt.close(fig)  # Close the figure to free memory
