import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.axes import Axes


def plot_spm_comparison(
    spm_inference,
    variable_name: str,
    ax: Axes,
):
    """
    Generates a plot for SPM comparison results (e.g., t-test) onto a given Axes object.

    Args:
        spm_inference: The inference object from spm1d containing SPM results (e.g., t-test).
        variable_name: The name of the variable being compared (e.g., 'torque', 'knee_v').
        ax: The Matplotlib Axes object to plot on.
    """
    # Assuming inference object provides .z, .zstar, .h0reject like TTestInference
    z_values = spm_inference.z
    z_threshold = spm_inference.zstar
    alpha = spm_inference.alpha
    # h0_rejected = spm_inference.h0reject # Not used in the plot anymore
    n_points = len(z_values)
    x_axis = np.linspace(0, 100, n_points)  # Create x-axis (0-100%)

    # fig, ax = plt.subplots(figsize=(8, 4)) # Removed figure creation

    # Plot the SPM{t} statistic curve (Assuming t-statistic)
    ax.plot(
        x_axis,
        z_values,
        "k-",
        lw=1,  # label="SPM{t}" # Removed label
    )

    # Plot the critical threshold lines
    ax.axhline(
        z_threshold, color="k", linestyle="--", lw=1
    )  # Changed color, removed label
    ax.axhline(-z_threshold, color="k", linestyle="--", lw=1)  # Changed color
    ax.axhline(0, color="k", linestyle="--", lw=0.5)  # Add line at y=0

    # Fill areas where the curve exceeds the threshold
    fill_color = "grey"
    fill_alpha = 0.4
    ax.fill_between(
        x_axis,
        z_values,
        z_threshold,
        where=z_values > z_threshold,
        facecolor=fill_color,  # Changed color
        alpha=fill_alpha,  # Changed alpha
    )
    ax.fill_between(
        x_axis,
        z_values,
        -z_threshold,
        where=z_values < -z_threshold,
        facecolor=fill_color,  # Changed color
        alpha=fill_alpha,  # Changed alpha
    )

    # Add labels
    ax.set_xlabel("Time (%)")
    ax.set_ylabel("SPM{t}")  # Assuming t-statistic
    # title_str = f"SPM Comparison: {variable_name} - H0 Rejected: {h0_rejected}" # Title removed
    # ax.set_title(title_str)
    # ax.legend() # Removed legend
    ax.grid(False)  # Remove grid for similarity to example

    # Add alpha level text
    ax.text(0.95, 0.95, f"α={alpha:.2f}", transform=ax.transAxes, ha="right", va="top")

    # Save and close plot with dynamic filename - Removed
    # plot_filename = f"spm_direct_{variable_name}_comparison.png"
    # plot_path = figures_path / plot_filename
    # fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    # plt.close(fig)
    # print(f"SPM {variable_name} comparison plot saved to: {plot_path}")

    # No fig.show() here, as the main script might handle showing plots or run headless.
