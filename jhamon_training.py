# This script generates results of the functional data recorded during the
# Nordic Hamstring and Isokinetic training sessions.

import matplotlib.pyplot as plt
from pathlib import Path
from jhamon_training import check_result_file
from jhamon_training.data import frames
import os
from jhamon_training.data.utils import dame_manual_exclusions, filter_training_data
from jhamon_training.saveload import save_obj
import jhamon_training.stats.training as spmtr
import jhamon_training.stats.spm as spm
from jhamon_training.stats.repetition_analysis import compare_nh_ik_repetitions
from jhamon_training.stats.rep_stats import save_repetition_stats
import pandas as pd
from jhamon_training.plot.figure_2 import generate_figure_2
from jhamon_training.stats.analyses_discrete_vars import analyze_discrete_variables
import numpy as np
import re
from scipy import stats
from jhamon_training.plot.plot_knee_velocity_comparison import (
    plot_knee_velocity_comparison,
)
from jhamon_training.plot.plot_work_comparison import plot_work_comparison
from jhamon_training.plot.plot_peak_torque_evolution import plot_peak_torque_evolution
from jhamon_training.plot.plot_repetition_count import plot_repetition_count
from jhamon_training.stats.mixed_anova_analysis import run_mixed_anova
import json
from jhamon_training.stats.torque_stats import report_torque_stats
from jhamon_training.plot.plot_training_progression import plot_training_progression


pathtodata = Path("/Volumes/AJMA/")
results_output_path = Path.home() / "Desktop" / "_RESULTS_TRAINING"
pathtosave = results_output_path
figures_path = pathtosave / "figures"

# Create figures directory if it doesn't exist
os.makedirs(figures_path, exist_ok=True)

# NORDIC training sessions
nordict = check_result_file(pathtodata, results_output_path, res_file="nht_results.pkl")

# Load or generate Nordic DataFrame
nordf_file = results_output_path / "nordf.feather"
if nordf_file.exists():
    print(f"Loading cached Nordic DataFrame from {nordf_file}")
    nordf = pd.read_feather(nordf_file)
else:
    print("Generating Nordic DataFrame...")
    nordf = frames.nht_todf(my_dict=nordict)
    print(f"Saving Nordic DataFrame to {nordf_file}")
    nordf.to_feather(nordf_file)

# IK training sessions
ikdict = check_result_file(pathtodata, results_output_path, res_file="ikt_results.pkl")

# Load or generate IK DataFrame
ikdf_file = results_output_path / "ikdf.feather"
if ikdf_file.exists():
    print(f"Loading cached IK DataFrame from {ikdf_file}")
    ikdf = pd.read_feather(ikdf_file)
else:
    print("Generating IK DataFrame...")
    ikdf = frames.ikt_todf(my_dict=ikdict)
    print(f"Saving IK DataFrame to {ikdf_file}")
    ikdf.to_feather(ikdf_file)

# Merge both datasets
training_df = pd.concat([nordf, ikdf])
training_dfilt = dame_manual_exclusions(training_df)


# ##### DISCRETE VARIABLES #####
# discrete_analysis_results = analyze_discrete_variables(
#     nordict=nordict,
#     ikdf=ikdf,
#     training_df=training_df,
#     results_output_path=results_output_path,
# )
# training_disc = discrete_analysis_results["training_disc"]

# # Filter the training data
# training_disc = filter_training_data(training_disc)

# # Report torque statistics
# report_torque_stats(training_disc, results_output_path)

# # Repetition statistics
# save_repetition_stats(training_disc, results_output_path)

# # Knee work
# plot_work_comparison(training_disc, output_path=figures_path / "work_comparison.png")

# # Knee velocity
# plot_knee_velocity_comparison(
#     training_disc, output_path=figures_path / "knee_velocity_comparison.png"
# )

# Mechanical patterns.
# ################# SPM STATISTICS ##############
# # NH vs IK : TORQUE and Knee Velocity
# torqcomp = spmtr.spm_group_comparison(df=training_df[training_df["var"] == "torque"])
# kneevcom = spmtr.spm_group_comparison(df=training_df[training_df["var"] == "knee_v"])
# generate_figure_2(training_df, torqcomp, kneevcom, figures_path)


# # Plot peak torque evolution
# plot_peak_torque_evolution(
#     training_disc, output_path=figures_path / "peak_torque_evolution.png"
# )

# # Run mean torque ANOVA analysis
# anova_results = run_mixed_anova(
#     training_disc=training_disc,
#     output_path=figures_path / "mean_torque_anova.png",
#     save_results=True,
# )


# Create filtered dataset with top 3 repetitions per participant and session
def select_top_repetitions(df: pd.DataFrame, n_reps: int = 3) -> pd.DataFrame:
    """
    Selects the n highest peak torque repetitions per participant and session.

    Args:
        df: DataFrame containing the training data
        n_reps: Number of top repetitions to select (default: 3)

    Returns:
        DataFrame containing only the selected repetitions
    """
    # Calculate peak torque for each repetition
    peak_torques = (
        df.groupby(["par", "trses", "set", "rep"])["value"].max().reset_index()
    )

    # Sort by peak torque in descending order and select top n_reps per participant and session
    top_reps = peak_torques.sort_values(
        ["par", "trses", "value"], ascending=[True, True, False]
    )
    top_reps = top_reps.groupby(["par", "trses"]).head(n_reps)

    # Create a multi-index for the top repetitions
    top_reps_idx = pd.MultiIndex.from_frame(top_reps[["par", "trses", "set", "rep"]])

    # Create a multi-index for the original dataframe
    df_idx = pd.MultiIndex.from_frame(df[["par", "trses", "set", "rep"]])

    # Use isin with the multi-index for efficient filtering
    mask = df_idx.isin(top_reps_idx)

    return df[mask]


# Filter torque data and select top 3 repetitions
torque_data = training_dfilt[training_dfilt["var"] == "torque"]
filtered_torque_data = select_top_repetitions(torque_data, n_reps=3)

# Save filtered dataset
filtered_data_file = results_output_path / "filtered_torque_data.feather"
filtered_torque_data.to_feather(filtered_data_file)
print(f"Saved filtered torque data to {filtered_data_file}")

# Training progression : TORQUE
torque_NH = spm.spm_nh_kinetics(
    df=filtered_torque_data[filtered_torque_data["tr_group"] == "NH"]
)
torque_IK = spm.spm_nh_kinetics(
    df=filtered_torque_data[filtered_torque_data["tr_group"] == "IK"]
)
tr_progression = {"torqueNH": torque_NH, "torqueIK": torque_IK}

# Generate training progression plot
plot_training_progression(
    training_df=filtered_torque_data,
    tr_progression=tr_progression,
    output_path=figures_path / "training_progression.png",
)
