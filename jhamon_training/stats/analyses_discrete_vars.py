################### Discrete variables ##############################
import pandas as pd
import json
from jhamon_training.data import frames
from jhamon_training.data.calculate_ik_discrete import calculate_ik_discrete_variables
from jhamon_training.stats.repetition_analysis import compare_nh_ik_repetitions


def analyze_discrete_variables(nordict, ikdf, training_df, results_output_path):
    """
    Analyze discrete variables from Nordic and IK training sessions.

    Args:
        nordict: Dictionary with Nordic training data
        ikdf: DataFrame with IK training data
        training_df: Combined training DataFrame
        results_output_path: Path where to save results

    Returns:
        dict: Dictionary with calculated statistics
    """
    # Nordic discrete variables
    nor_disc_df = frames.nht_disc_todf(my_dict=nordict)
    print("Nordic dataframe columns:", nor_disc_df.columns.tolist())

    # IK discrete variables
    ik_discrete_df = calculate_ik_discrete_variables(ikdf)
    print("IK dataframe columns:", ik_discrete_df.columns.tolist())

    # Make sure both dataframes have the same columns
    common_columns = list(set(nor_disc_df.columns) & set(ik_discrete_df.columns))
    nor_disc_df = nor_disc_df[common_columns]
    ik_discrete_df = ik_discrete_df[common_columns]

    # Concatenate the dataframes
    training_disc = pd.concat([nor_disc_df, ik_discrete_df], ignore_index=True)
    print("Concatenated dataframe shape:", training_disc.shape)

    # calculate the average work per repetition per training group as save to a variable, average across all repetitions
    rep_work_mean = training_disc.groupby(["tr_group"])["work"].mean()
    rep_work_sd = training_disc.groupby(["tr_group"])["work"].std()

    # calculate average velocity per training group as save to a variable, average across all repetitions
    rep_vel_mean = training_disc.groupby(["tr_group"])["knee_v"].mean()
    rep_vel_sd = training_disc.groupby(["tr_group"])["knee_v"].std()

    # --- Calculate and Compare Repetitions ---
    reps_dif_df = compare_nh_ik_repetitions(training_df)

    # Calculate and save statistics for reporting
    report_stats = calculate_and_save_report_stats(
        training_disc, reps_dif_df, results_output_path
    )

    return {
        "training_disc": training_disc,
        "rep_work_mean": rep_work_mean,
        "rep_work_sd": rep_work_sd,
        "rep_vel_mean": rep_vel_mean,
        "rep_vel_sd": rep_vel_sd,
        "reps_dif_df": reps_dif_df,
        "report_stats": report_stats,
    }


def calculate_and_save_report_stats(training_disc, reps_dif_df, output_path):
    """
    Calculate training statistics and save them to a JSON file for use in reports.

    Args:
        training_disc: DataFrame with discrete training variables
        reps_dif_df: DataFrame with repetition differences between groups
        output_path: Path where to save the JSON file

    Returns:
        dict: Dictionary with calculated statistics
    """
    # Calculate mean work per repetition for each group (Joules)
    work_mean = training_disc.groupby(["tr_group"])["work"].mean()
    work_sd = training_disc.groupby(["tr_group"])["work"].std()

    # Calculate average repetition difference between groups
    rep_diff_mean = reps_dif_df["reps_diff"].mean()
    rep_diff_sd = reps_dif_df["reps_diff"].std()

    tr1_avg_diff = reps_dif_df[reps_dif_df["trses"] == 1]["reps_diff"].abs().mean()
    tr1_avg_diff_sd = reps_dif_df[reps_dif_df["trses"] == 1]["reps_diff"].abs().std()
    tr15_avg_diff = reps_dif_df[reps_dif_df["trses"] == 15]["reps_diff"].abs().mean()
    tr15_avg_diff_sd = reps_dif_df[reps_dif_df["trses"] == 15]["reps_diff"].abs().std()

    # Create stats dictionary with formatted values for reporting
    stats = {
        "nh_work_mean": round(work_mean["NH"], 1),
        "nh_work_sd": round(work_sd["NH"], 1),
        "ik_work_mean": round(work_mean["IK"], 1),
        "ik_work_sd": round(work_sd["IK"], 1),
        "rep_diff": abs(round(rep_diff_mean, 1)),
        "rep_diff_sd": abs(round(rep_diff_sd, 1)),
        "rep_diff_sign": "fewer" if rep_diff_mean > 0 else "more",
        "tr1_avg_diff": abs(round(tr1_avg_diff, 1)),
        "tr15_avg_diff": abs(round(tr15_avg_diff, 1)),
        "tr1_avg_diff_sd": abs(round(tr1_avg_diff_sd, 1)),
        "tr15_avg_diff_sd": abs(round(tr15_avg_diff_sd, 1)),
    }

    # Save stats to JSON file for use in Quarto reports
    json_path = output_path / "training_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Training statistics saved to {json_path}")
    return stats
