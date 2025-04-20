import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from jhamon_training.data import frames
from jhamon_training import check_result_file


def plot_fpeak_data(nordict, save_path=None):
    """
    Create a scientific plot showing knee_ROM, knee_fpeak and knee_ROMfpeak across training sessions.

    Parameters:
    -----------
    nordict : dict
        Dictionary containing Nordic hamstring exercise data
    save_path : Path, optional
        Path to save the figure. If None, the figure will be displayed

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    # Extract data into lists for plotting
    rom_data = []
    fpeak_data = []
    romfpeak_data = []

    for participant in nordict.keys():
        for tr_session in nordict[participant].keys():
            tr_num = int(tr_session.split("_")[1])  # Extract session number
            for set_name in nordict[participant][tr_session].keys():
                for rep_name in nordict[participant][tr_session][set_name].keys():
                    # Get the discrete variables dictionary
                    discrete_vars = nordict[participant][tr_session][set_name][
                        rep_name
                    ][0]

                    # Append data with session number
                    rom_data.append(
                        {
                            "tr_session": tr_num,
                            "value": discrete_vars["knee_ROM"],
                            "participant": participant,
                        }
                    )
                    fpeak_data.append(
                        {
                            "tr_session": tr_num,
                            "value": discrete_vars["knee_fpeak"],
                            "participant": participant,
                        }
                    )
                    romfpeak_data.append(
                        {
                            "tr_session": tr_num,
                            "value": discrete_vars["knee_ROMfpeak"],
                            "participant": participant,
                        }
                    )

    # Convert to DataFrames
    rom_df = pd.DataFrame(rom_data)
    fpeak_df = pd.DataFrame(fpeak_data)
    romfpeak_df = pd.DataFrame(romfpeak_data)

    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot settings
    point_alpha = 0.4
    box_color = "#1f77b4"  # Professional blue color
    point_color = "#2c3e50"  # Dark blue-gray for points

    # Style settings for all plots
    plot_params = {
        "showfliers": False,  # Don't show outlier points (they'll be in the strip plot)
        "boxprops": {"facecolor": box_color, "alpha": 0.3},
        "medianprops": {"color": "black"},
        "whiskerprops": {"color": "black"},
        "capprops": {"color": "black"},
    }

    # Plot knee_ROM
    sns.boxplot(x="tr_session", y="value", data=rom_df, ax=ax1, **plot_params)
    sns.stripplot(
        x="tr_session",
        y="value",
        data=rom_df,
        ax=ax1,
        color=point_color,
        alpha=point_alpha,
        size=4,
        jitter=0.2,
    )

    ax1.set_xlabel("Training Session")
    ax1.set_ylabel("Knee ROM (degrees)")
    ax1.set_title("A", loc="left", fontweight="bold")

    # Plot knee_fpeak
    sns.boxplot(x="tr_session", y="value", data=fpeak_df, ax=ax2, **plot_params)
    sns.stripplot(
        x="tr_session",
        y="value",
        data=fpeak_df,
        ax=ax2,
        color=point_color,
        alpha=point_alpha,
        size=4,
        jitter=0.2,
    )

    ax2.set_xlabel("Training Session")
    ax2.set_ylabel("Knee Angle at Peak Force (degrees)")
    ax2.set_title("B", loc="left", fontweight="bold")

    # Plot knee_ROMfpeak
    sns.boxplot(x="tr_session", y="value", data=romfpeak_df, ax=ax3, **plot_params)
    sns.stripplot(
        x="tr_session",
        y="value",
        data=romfpeak_df,
        ax=ax3,
        color=point_color,
        alpha=point_alpha,
        size=4,
        jitter=0.2,
    )

    ax3.set_xlabel("Training Session")
    ax3.set_ylabel("Relative ROM at Peak Force (%)")
    ax3.set_title("C", loc="left", fontweight="bold")

    # Additional styling for all plots
    for ax in [ax1, ax2, ax3]:
        # Remove top and right spines
        sns.despine(ax=ax)

        # Add a light grid
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

        # Adjust tick parameters
        ax.tick_params(axis="both", which="major", labelsize=9)

        # Set background color
        ax.set_facecolor("#f8f9fa")

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    # Example usage
    results_output_path = Path.home() / "Desktop" / "_RESULTS_TRAINING"
    pathtodata = Path("/Volumes/AJMA/")

    # Load Nordic training data
    nordict = check_result_file(
        pathtodata,
        results_output_path,
        res_file="nht_results.pkl",
    )

    # Create and show the plot
    fig = plot_fpeak_data(nordict)
    plt.show()
