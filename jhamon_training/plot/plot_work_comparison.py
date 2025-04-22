import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Union, Tuple
from matplotlib.axes import Axes
from pathlib import Path
from scipy import stats
from numpy.typing import NDArray


def calculate_ccc(x, y):
    """Calculate Concordance Correlation Coefficient (CCC) between two arrays."""
    # Check for NaN values
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        print("Warning: NaN values found in data")
        return np.nan, np.nan

    # Check for zero variance
    if np.var(x) == 0 or np.var(y) == 0:
        print("Warning: Zero variance in data")
        return np.nan, np.nan

    # Check for sufficient data points
    if len(x) < 2 or len(y) < 2:
        print("Warning: Insufficient data points for correlation")
        return np.nan, np.nan

    # Calculate means and variances
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x, ddof=1)
    y_var = np.var(y, ddof=1)
    xy_cov = np.cov(x, y, ddof=1)[0, 1]

    # Calculate CCC
    r = np.corrcoef(x, y)[0, 1]
    ccc = (2 * xy_cov) / (x_var + y_var + (x_mean - y_mean) ** 2)

    print(f"Data points: {len(x)}")
    print(f"Means - x: {x_mean:.2f}, y: {y_mean:.2f}")
    print(f"Variances - x: {x_var:.2f}, y: {y_var:.2f}")
    print(f"Covariance: {xy_cov:.2f}")
    print(f"CCC: {ccc:.3f}, Pearson's r: {r:.3f}")

    return ccc, r


def calculate_heteroscedasticity(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> Tuple[float, float]:
    """Calculate heteroscedasticity using Breusch-Pagan test.

    Args:
        x: Independent variable (mean values)
        y: Dependent variable (differences)

    Returns:
        Tuple containing:
        - F-statistic
        - p-value
    """
    # Calculate residuals
    mean_y = np.mean(y)
    residuals = y - mean_y

    # Calculate squared residuals
    squared_residuals = residuals**2

    # Perform linear regression of squared residuals on x
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, squared_residuals)

    # Calculate F-statistic
    n = len(x)
    k = 1  # number of predictors
    ssr = np.sum((squared_residuals - np.mean(squared_residuals)) ** 2)
    sse = np.sum((squared_residuals - (intercept + slope * x)) ** 2)
    f_stat = float((ssr / k) / (sse / (n - k - 1)))

    # Calculate p-value
    p_value = float(1 - stats.f.cdf(f_stat, k, n - k - 1))

    return f_stat, p_value


def plot_work_comparison(
    data: pd.DataFrame, output_path: Union[Path, str, None] = None
) -> None:
    """Generate a combined figure with work comparison plots.

    The figure contains:
    - Top: Work across training sessions
    - Bottom left: Bland-Altman plot
    - Bottom right: Scatter plot with CCC

    Args:
        data: DataFrame containing the training data with columns:
            - par: participant ID
            - trses: training session
            - tr_group: training group (NH or IK)
            - work: work in Joules
            - set: set number
        output_path: Path where to save the figure. If None, the figure will be displayed.
    """
    # Set the style for scientific publication
    plt.style.use("default")

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # Define colors from cividis colormap
    nh_color = plt.colormaps["cividis"](0.05)
    ik_color = plt.colormaps["cividis"](0.95)

    # Top plot (spans both columns)
    ax1 = fig.add_subplot(gs[0, :])

    # Get unique training sessions and sort them numerically
    training_sessions = sorted(
        data["trses"].unique(), key=lambda x: int(x.split("_")[1])
    )
    n_sessions = len(training_sessions)

    # Define scatter colors (slightly transparent)
    scatter_colors = {
        "NH": (*nh_color[:3], 0.5),
        "IK": (*ik_color[:3], 0.5),
    }

    # Define x-positions for groups
    x_positions = {
        "NH": -0.2,
        "IK": 0.2,
    }

    # Process data for each group
    for group_name in ["NH", "IK"]:
        all_work = []

        # Process each training session
        for trses in training_sessions:
            session_num = int(trses.split("_")[1])

            # Get unique sets for this session
            session_sets = sorted(
                data[data["trses"] == trses]["set"].unique(),
                key=lambda x: int(x.split("_")[1]),
            )

            # Process each set
            for set_name in session_sets:
                # Filter data for this group, session, and set
                set_data = data[
                    (data["tr_group"] == group_name)
                    & (data["trses"] == trses)
                    & (data["set"] == set_name)
                ]

                # Group by participant and sum work across repetitions
                participant_work = set_data.groupby("par")["work"].sum().reset_index()

                # Store total work for each participant
                all_work.extend([(session_num, w) for w in participant_work["work"]])

        # Plot individual points with jitter
        x_coords = [x[0] + x_positions[group_name] for x in all_work]
        y_coords = [x[1] for x in all_work]
        x_jitter = np.random.uniform(-0.1, 0.1, size=len(x_coords))
        y_jitter = np.random.uniform(-0.1, 0.1, size=len(y_coords))

        # Plot individual points
        ax1.scatter(
            np.array(x_coords) + x_jitter,
            np.array(y_coords) + y_jitter,
            color=scatter_colors[group_name],
            s=30,
            label=f"{group_name}",
        )

    # Customize top plot
    ax1.set_xlabel("Training Session", fontsize=14)
    ax1.set_ylabel("Work (J)", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_xticks(range(1, n_sessions + 1))
    ax1.legend(loc="upper right", fontsize=14)
    ax1.tick_params(axis="both", which="major", labelsize=14)

    # Add panel label
    ax1.text(
        -0.05,
        1.02,
        "A",
        transform=ax1.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    # Prepare data for Bland-Altman and scatter plots
    # Calculate total work per participant per set and group
    participant_set_work = (
        data.groupby(["trses", "set", "tr_group", "par"])["work"].sum().reset_index()
    )

    # Then average across participants for each set and group
    set_work = (
        participant_set_work.groupby(["trses", "set", "tr_group"])["work"]
        .mean()
        .reset_index()
    )

    # Pivot the data to get NH and IK values side by side
    pivoted_data = set_work.pivot_table(
        index=["trses", "set"], columns="tr_group", values="work", aggfunc="mean"
    ).reset_index()

    # Print data information
    print("\nData for CCC calculation:")
    print(f"Number of rows: {len(pivoted_data)}")
    print("NH values:", pivoted_data["NH"].describe())
    print("IK values:", pivoted_data["IK"].describe())
    print("\nCalculating CCC...")

    mean_work = (pivoted_data["NH"] + pivoted_data["IK"]) / 2
    diff = pivoted_data["NH"] - pivoted_data["IK"]

    # Calculate statistics for Bland-Altman
    mean_diff = diff.mean()
    std_diff = diff.std()
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    # Bland-Altman plot (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(mean_work, diff, alpha=0.5, color=nh_color)
    ax2.axhline(0, color="red", linestyle="--")
    ax2.axhline(
        mean_diff,
        color=nh_color,
        linestyle="--",
        label=f"Mean difference: {mean_diff:.2f} J",
    )
    ax2.axhline(
        upper_limit,
        color=nh_color,
        linestyle="--",
        label=f"Upper limit: {upper_limit:.2f} J",
    )
    ax2.axhline(
        lower_limit,
        color=nh_color,
        linestyle="--",
        label=f"Lower limit: {lower_limit:.2f} J",
    )

    ax2.set_xlabel("Mean Work (J)", fontsize=14)
    ax2.set_ylabel("Difference (NH - IK) (J)", fontsize=14)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis="both", which="major", labelsize=14)

    # Add panel label
    ax2.text(
        -0.05,
        1.02,
        "B",
        transform=ax2.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    # Scatter plot with CCC (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(pivoted_data["NH"], pivoted_data["IK"], alpha=0.5, color=nh_color)

    # Add identity line
    min_val = min(pivoted_data["NH"].min(), pivoted_data["IK"].min())
    max_val = max(pivoted_data["NH"].max(), pivoted_data["IK"].max())
    ax3.plot([min_val, max_val], [min_val, max_val], "r--", label="Identity line")

    # Calculate and add CCC
    ccc, r = calculate_ccc(pivoted_data["NH"], pivoted_data["IK"])
    ax3.set_xlabel("NH Work (J)", fontsize=14)
    ax3.set_ylabel("IK Work (J)", fontsize=14)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(fontsize=14)
    ax3.tick_params(axis="both", which="major", labelsize=14)

    # Add panel label
    ax3.text(
        -0.05,
        1.02,
        "C",
        transform=ax3.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    # Add CCC and r values as text
    ax3.text(
        0.05,
        0.95,
        f"CCC = {ccc:.3f}\nPearson's r = {r:.3f}",
        transform=ax3.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Adjust layout
    plt.tight_layout()

    # Save or show the figure
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
