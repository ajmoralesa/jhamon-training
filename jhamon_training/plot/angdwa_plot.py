import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from jhamon_training.data import frames
from jhamon_training import check_result_file


def plot_angdwa_data(nordict, save_path=None):
    """
    Create a scientific plot showing knee_angDWA and knee_ROMDWA across training sessions.

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
    angdwa_data = []
    romdwa_data = []

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
                    angdwa_data.append(
                        {
                            "tr_session": tr_num,
                            "value": discrete_vars["knee_angDWA"],
                            "participant": participant,
                        }
                    )
                    romdwa_data.append(
                        {
                            "tr_session": tr_num,
                            "value": discrete_vars["knee_ROMDWA"],
                            "participant": participant,
                        }
                    )

    # Convert to DataFrames
    angdwa_df = pd.DataFrame(angdwa_data)
    romdwa_df = pd.DataFrame(romdwa_data)

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot settings
    point_alpha = 0.4
    box_color = "#1f77b4"  # Professional blue color
    point_color = "#2c3e50"  # Dark blue-gray for points

    # Style settings for both plots
    plot_params = {
        "showfliers": False,  # Don't show outlier points (they'll be in the strip plot)
        "boxprops": {"facecolor": box_color, "alpha": 0.3},
        "medianprops": {"color": "black"},
        "whiskerprops": {"color": "black"},
        "capprops": {"color": "black"},
    }

    # Plot knee_angDWA
    sns.boxplot(x="tr_session", y="value", data=angdwa_df, ax=ax1, **plot_params)
    sns.stripplot(
        x="tr_session",
        y="value",
        data=angdwa_df,
        ax=ax1,
        color=point_color,
        alpha=point_alpha,
        size=4,
        jitter=0.2,
    )

    ax1.set_xlabel("Training Session")
    ax1.set_ylabel("Knee Angle at DWA (degrees)")
    ax1.set_title("A", loc="left", fontweight="bold")

    # Plot knee_ROMDWA
    sns.boxplot(x="tr_session", y="value", data=romdwa_df, ax=ax2, **plot_params)
    sns.stripplot(
        x="tr_session",
        y="value",
        data=romdwa_df,
        ax=ax2,
        color=point_color,
        alpha=point_alpha,
        size=4,
        jitter=0.2,
    )

    ax2.set_xlabel("Training Session")
    ax2.set_ylabel("Relative ROM at DWA (%)")
    ax2.set_title("B", loc="left", fontweight="bold")

    # Additional styling for both plots
    for ax in [ax1, ax2]:
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
    fig = plot_angdwa_data(nordict)
    plt.show()
