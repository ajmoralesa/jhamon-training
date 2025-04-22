import pandas as pd
import numpy as np
from pathlib import Path
import pingouin as pg
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Union, List, Optional
import json


def calculate_descriptive_stats(df: pd.DataFrame, dv: str) -> Dict[str, Any]:
    """Calculate descriptive statistics for each group and training session.

    Args:
        df: DataFrame containing the data
        dv: Name of the dependent variable to analyze

    Returns:
        Dictionary containing mean and SD for each group and training session
    """
    # Calculate group means and SDs
    group_stats = df.groupby("tr_group")[dv].agg(["mean", "std"]).to_dict()

    # Calculate session means and SDs for each group
    session_stats = {}
    for group in df["tr_group"].unique():
        group_data = df[df["tr_group"] == group]
        session_stats[group] = (
            group_data.groupby("trses_num")[dv].agg(["mean", "std"]).to_dict()
        )

    return {"group_stats": group_stats, "session_stats": session_stats}


def check_assumptions(df: pd.DataFrame, dv: str) -> None:
    """Check assumptions for repeated measures ANOVA.

    Args:
        df: DataFrame containing the data
        dv: Name of the dependent variable to analyze
    """
    print(f"\nChecking ANOVA assumptions for {dv}:")

    # 1. Normality of residuals
    print("\n1. Normality of residuals:")
    for group in df["tr_group"].unique():
        for session in df["trses_num"].unique():
            data = df[(df["tr_group"] == group) & (df["trses_num"] == session)][dv]
            stat, p = stats.shapiro(data)
            print(f"Group {group}, Session {session}: W = {stat:.3f}, p = {p:.3f}")

    # 2. Homogeneity of variances
    print("\n2. Homogeneity of variances:")
    for session in df["trses_num"].unique():
        session_data = df[df["trses_num"] == session]
        stat, p = stats.levene(
            session_data[session_data["tr_group"] == "IK"][dv],
            session_data[session_data["tr_group"] == "NH"][dv],
        )
        print(f"Session {session}: F = {stat:.3f}, p = {p:.3f}")


def perform_anova(
    df: pd.DataFrame, dv: str, output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Perform Two-Way Mixed ANOVA using pingouin and save results.

    Args:
        df: DataFrame containing the data
        dv: Name of the dependent variable to analyze
        output_path: Optional path where the results will be saved. If None, results are only printed.

    Returns:
        Dictionary containing ANOVA results
    """
    print(f"\nPerforming Two-Way Mixed ANOVA for {dv}:")

    # Print study design information
    print("\nStudy Design:")
    print(
        "- Between-subjects factor: Training Group (IK vs NH, different participants)"
    )
    print("- Within-subjects factor: Training Session (repeated measures)")
    print(f"- Dependent variable: {dv}")
    print("\nMixed ANOVA is appropriate because:")
    print("1. Different participants in IK and NH groups (between-subjects)")
    print("2. Same participants measured across multiple sessions (within-subjects)")
    print("3. Allows testing of both group differences and session effects")

    # Perform the ANOVA
    aov = pg.mixed_anova(
        data=df,
        dv=dv,
        within="trses_num",
        between="tr_group",
        subject="par",
        correction="auto",
    )

    # Format the results table
    results_table = aov.copy()
    results_table["p-unc"] = results_table["p-unc"].map(lambda x: f"{x:.3f}")
    results_table["np2"] = results_table["np2"].map(lambda x: f"{x:.3f}")

    # Print results
    print("\nANOVA Results:")
    print(results_table)

    print("\nEffect sizes (η²):")
    for effect in aov.index:
        if effect != "Residual":
            print(f"{effect}: η² = {aov.loc[effect, 'np2']:.3f}")

    # Save results to file if path is provided
    if output_path:
        with open(output_path, "w") as f:
            f.write(f"Two-Way Mixed ANOVA Results for {dv}\n")
            f.write("==========================\n\n")

            f.write("Study Design:\n")
            f.write("------------\n")
            f.write(
                "- Between-subjects factor: Training Group (IK vs NH, different participants)\n"
            )
            f.write("- Within-subjects factor: Training Session (repeated measures)\n")
            f.write(f"- Dependent variable: {dv}\n\n")

            f.write("Rationale for Mixed ANOVA:\n")
            f.write("-------------------------\n")
            f.write(
                "1. Different participants in IK and NH groups (between-subjects)\n"
            )
            f.write(
                "2. Same participants measured across multiple sessions (within-subjects)\n"
            )
            f.write(
                "3. Allows testing of both group differences and session effects\n\n"
            )

            f.write("ANOVA Table:\n")
            f.write(results_table.to_string())
            f.write("\n\n")

            f.write("Effect Sizes:\n")
            for effect in aov.index:
                if effect != "Residual":
                    f.write(f"{effect}: η² = {aov.loc[effect, 'np2']:.3f}\n")

    # Calculate time effect statistics if significant
    time_stats = {}
    if aov.iloc[1]["p-unc"] < 0.05:  # If time effect is significant
        for session in range(1, 16):
            session_data = df[df["trses_num"] == session][dv]
            time_stats[f"session_{session}"] = {
                "mean": round(session_data.mean(), 1),
                "sd": round(session_data.std(), 1),
            }

    # Return ANOVA results in a structured format with appropriate rounding
    return {
        "group": {
            "f_value": (
                round(aov.iloc[0]["F"], 1)
                if dv in ["peak_torque", "knee_ROM"]
                else round(aov.iloc[0]["F"], 2)
            ),
            "p_value": round(
                aov.iloc[0]["p-unc"], 3
            ),  # Keep 3 decimal places for p-values
            "effect_size": round(
                aov.iloc[0]["np2"], 2
            ),  # Keep 2 decimal places for effect sizes
        },
        "time": {
            "f_value": round(aov.iloc[1]["F"], 1),
            "p_value": round(aov.iloc[1]["p-unc"], 3),
            "effect_size": round(aov.iloc[1]["np2"], 2),
            "session_stats": time_stats,
        },
        "interaction": {
            "f_value": round(aov.iloc[2]["F"], 1),
            "p_value": round(aov.iloc[2]["p-unc"], 3),
            "effect_size": round(aov.iloc[2]["np2"], 2),
        },
    }


def plot_results(df: pd.DataFrame, dv: str, output_path: Path) -> None:
    """Plot the results of the ANOVA.

    Args:
        df: DataFrame containing the data
        dv: Name of the dependent variable to plot
        output_path: Path where the plot will be saved
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Create line plot with error bars
    ax = sns.lineplot(
        data=df,
        x="trses_num",
        y=dv,
        hue="tr_group",
        style="tr_group",
        markers=True,
        dashes=False,
        errorbar="se",
        palette={"IK": "#1f77b4", "NH": "#ff7f0e"},
    )

    # Customize the plot
    plt.xlabel("Training Session", fontsize=12)
    plt.ylabel(f"{dv} ({get_units(dv)})", fontsize=12)
    plt.title(f"{dv} Evolution by Group", fontsize=14)
    plt.xticks(range(1, 16))
    plt.legend(title="Group", fontsize=10)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def get_units(dv: str) -> str:
    """Get the appropriate units for a dependent variable.

    Args:
        dv: Name of the dependent variable

    Returns:
        String containing the appropriate units
    """
    units = {
        "mean_torque": "N·m",
        "peak_torque": "N·m",
        "knee_ROM": "°",
        "angle_at_peak_torque": "°",
        "knee_v_mean": "°/s",
    }
    return units.get(dv, "")


def run_mixed_anova(
    training_disc: pd.DataFrame,
    output_path: Path,
    variables: List[str] = [
        "mean_torque",
        "peak_torque",
        "knee_ROM",
        "angle_at_peak_torque",
        "knee_v_mean",
    ],
    create_plots: bool = False,
    save_results: bool = False,
) -> Dict[str, Any]:
    """Run the mixed ANOVA analysis for multiple variables.

    Args:
        training_disc: DataFrame containing the training data
        output_path: Path where results will be saved (if save_results or create_plots is True)
        variables: List of variables to analyze
        create_plots: Whether to create evolution plots for each variable (default: False)
        save_results: Whether to save ANOVA results to text files (default: False)

    Returns:
        Dictionary containing ANOVA and descriptive statistics for each variable
    """
    # Create output directory if needed
    if save_results or create_plots:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter for top 3 repetitions
    df = (
        training_disc.groupby(["par", "trses_num"])
        .apply(lambda x: x.nlargest(3, "peak_torque"))
        .reset_index(drop=True)
    )

    # Dictionary to store all results
    all_results = {}

    # Run analysis for each variable
    for dv in variables:
        print(f"\nAnalyzing {dv}...")
        results = {}

        # Check assumptions
        check_assumptions(df, dv)

        # Calculate descriptive statistics
        desc_stats = calculate_descriptive_stats(df, dv)
        results["descriptive_stats"] = desc_stats

        # Perform ANOVA
        aov_results = perform_anova(
            df,
            dv,
            output_path.parent / f"{dv}_anova_results.txt" if save_results else None,
        )
        results["anova"] = aov_results

        # Create plots if requested
        if create_plots:
            plot_path = output_path.parent / f"{dv}_evolution.png"
            plot_results(df, dv, plot_path)

        # Store results for this variable
        all_results[dv] = results

    # Save results to training_stats.json
    stats_path = Path.home() / "Desktop" / "_RESULTS_TRAINING" / "training_stats.json"
    try:
        # Load existing stats
        with open(stats_path) as f:
            stats = json.load(f)
    except FileNotFoundError:
        stats = {}

    # Add ANOVA results to stats
    for dv, results in all_results.items():
        # Add descriptive statistics
        group_stats = results["descriptive_stats"]["group_stats"]
        stats[f"anova_{dv}_group_mean"] = {
            k: round(v, 1) for k, v in group_stats["mean"].items()
        }
        stats[f"anova_{dv}_group_sd"] = {
            k: round(v, 1) for k, v in group_stats["std"].items()
        }

        # Add ANOVA statistics
        for effect in ["group", "time", "interaction"]:
            stats[f"anova_{dv}_{effect}_f"] = results["anova"][effect]["f_value"]
            stats[f"anova_{dv}_{effect}_p"] = results["anova"][effect]["p_value"]
            stats[f"anova_{dv}_{effect}_eta"] = results["anova"][effect]["effect_size"]

            # Add time effect session statistics if significant
            if effect == "time" and "session_stats" in results["anova"][effect]:
                for session, session_stats in results["anova"][effect][
                    "session_stats"
                ].items():
                    stats[f"anova_{dv}_{session}_mean"] = session_stats["mean"]
                    stats[f"anova_{dv}_{session}_sd"] = session_stats["sd"]

    # Save updated stats
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    return all_results
