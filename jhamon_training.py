# This script generates results of the functional data recorded during the
# Nordic Hamstring and Isokinetic training sessions.

import matplotlib.pyplot as plt
from pathlib import Path
from jhamon_training import check_result_file
from jhamon_training.pathutils import RESULTS_TRAINING_PATH
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
from jhamon_training.stats.mixed_lmm_analysis import run_mixed_lmm
from jhamon_training.stats.agreement import run_agreement
import json
from jhamon_training.stats.torque_stats import report_torque_stats
from jhamon_training.plot.plot_training_progression import plot_training_progression
from jhamon_training.plot.figure_3_ridgeline import plot_figure_3_ridgeline


pathtodata = Path("/Volumes/AJMA/")
results_output_path = RESULTS_TRAINING_PATH
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


##### DISCRETE VARIABLES #####
discrete_analysis_results = analyze_discrete_variables(
    nordict=nordict,
    ikdf=ikdf,
    training_df=training_df,
    results_output_path=results_output_path,
)
training_disc = discrete_analysis_results["training_disc"]

# Filter the training data
training_disc = filter_training_data(training_disc)

# Figure 3: repetitions + work-per-rep + work-per-set distributions (ridgeline)
plot_figure_3_ridgeline(
    training_disc, output_path=figures_path / "Figure_3_ridgeline.png"
)

# Report torque statistics
report_torque_stats(training_disc, results_output_path)

# # Repetition statistics
# NOTE: left disabled on purpose. It runs without error, but it rewrites
# nh_reps/ik_reps/total_reps from the POST-filter frame, while
# analyze_discrete_variables above already wrote those three keys — plus the
# nh_percentage and prescribed_total_nh derived from them — from the PRE-filter
# frame. Enabling it changes the three counts without recomputing the
# percentage, so the Results sentence in jhamon_results.qmd would quote a
# repetition count and a percentage that no longer refer to the same
# denominator. Re-enable only together with a fix for that inconsistency.
# save_repetition_stats(training_disc, results_output_path)

# Knee work
plot_work_comparison(training_disc, output_path=figures_path / "work_comparison.png")

# Knee velocity
plot_knee_velocity_comparison(
    training_disc, output_path=figures_path / "knee_velocity_comparison.png"
)

# Mechanical patterns.
################# SPM STATISTICS ##############
# NH vs IK : TORQUE and Knee Velocity
torqcomp = spmtr.spm_group_comparison(df=training_df[training_df["var"] == "torque"])
kneevcom = spmtr.spm_group_comparison(df=training_df[training_df["var"] == "knee_v"])
generate_figure_2(training_df, torqcomp, kneevcom, figures_path)


# Plot peak torque evolution
plot_peak_torque_evolution(
    training_disc, output_path=figures_path / "peak_torque_evolution.png"
)

# Discrete-outcome analysis: linear mixed model (random intercept per
# participant) on the top-3 cell means. Replaces the complete-case two-way
# mixed ANOVA, which listwise-dropped the 9 participants with incomplete
# session records (collapsing to 6 NH + 12 IK); the LMM keeps all 13 NH + 14
# IK and reproduces the same conclusions with full power. Writes lmm_* keys to
# training_stats.json. (run_mixed_anova remains in the repo for reference.)
lmm_results = run_mixed_lmm(training_disc, write_json=True)

# Prescription-matching check: Bland-Altman agreement between NH and IK for the
# two outcomes the two modalities were matched on — work per set and mean knee
# angular velocity. One point per (session x set) cell, NH group mean vs IK
# group mean, difference = NH - IK; reports bias, 95% limits of agreement,
# proportional bias and the concordance correlation. This is the paper's primary
# validity result: the whole design rests on the claim that the two modalities
# delivered an equivalent mechanical dose, and until now the bias/LoA were
# computed inside the two figures above and thrown away, so the Methods promised
# a Bland-Altman analysis that no number in training_stats.json backed. Must run
# AFTER run_mixed_lmm, which writes the JSON; this one merges ba_* keys into it.
agreement_results = run_agreement(training_disc, write_json=True)


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
