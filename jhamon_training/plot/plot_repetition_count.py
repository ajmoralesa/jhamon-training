import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


def plot_repetition_count(
    training_disc: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> None:
    """
    Create a scientific plot showing the repetition count per participant and training session.

    Parameters
    ----------
    training_disc : pd.DataFrame
        DataFrame containing the training data with repetition counts
    output_path : Optional[Path]
        Path to save the figure. If None, the figure is displayed
    figsize : tuple
        Figure size in inches (width, height)
    dpi : int
        Dots per inch for the output figure
    """
    # Set the style for scientific publication
    plt.style.use("default")  # Reset to default style
    sns.set_style("whitegrid")  # Use seaborn's whitegrid style

    # Update rcParams for better typography and appearance
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        }
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    try:
        # Count repetitions per participant and training group
        rep_counts = (
            training_disc.groupby(["par", "tr_group"])["rep"].count().reset_index()
        )

        # Create the bar plot
        sns.barplot(
            data=rep_counts,
            x="par",
            y="rep",
            hue="tr_group",
            palette=["#1f77b4", "#ff7f0e"],  # Blue and orange for NH and IK
            ax=ax,
            errorbar=None,
        )

        # Customize the plot
        ax.set_xlabel("Participant", fontweight="bold")
        ax.set_ylabel("Number of Repetitions", fontweight="bold")
        ax.set_title("Repetition Count per Participant and Training Group", pad=20)

        # Customize legend
        ax.legend(title="Training Group", title_fontsize=10)

        # Adjust layout
        plt.tight_layout()

        # Save or show the figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    except Exception as e:
        plt.close()
        raise RuntimeError(f"Error creating repetition count plot: {str(e)}")
