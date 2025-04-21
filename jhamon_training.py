# This script generates results of the functional data recorded during the
# Nordic Hamstring and Isokinetic training sessions.

import matplotlib.pyplot as plt
from pathlib import Path
from jhamon_training import check_result_file
from jhamon_training.data import frames
import os
from jhamon_training.data.utils import dame_manual_exclusions
from jhamon_training.saveload import save_obj
import jhamon_training.stats.training as spmtr
import jhamon_training.stats.spm as spm
from jhamon_training.stats.repetition_analysis import compare_nh_ik_repetitions
import pandas as pd
from jhamon_training.plot.figure_2 import generate_figure_2
from jhamon_training.data.calculate_ik_discrete import calculate_ik_discrete_variables
import json

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


# Compute the average work per rep
training_df["knee_work"] = training_df["torque"] * training_df["knee_ROM"]
training_df["var"].unique()


training_df[training_df["var"] == "knee_work"]

training_df["knee_work"] = training_df["torque"] * training_df["knee_ROM"]

################# SPM STATISTICS ##############
# NH vs IK : TORQUE
torqcomp = spmtr.spm_group_comparison(df=training_df[training_df["var"] == "torque"])
# # Call the generalized function with the variable name - Original call removed
# plot_spm_comparison(torqcomp["ti"], variable_name="torque", figures_path=figures_path)

# NH vs IK : Knee Velocity
kneevcom = spmtr.spm_group_comparison(df=training_df[training_df["var"] == "knee_v"])
# # Call the generalized function with the variable name - Original call removed
# plot_spm_comparison(kneevcom["ti"], variable_name="knee_v", figures_path=figures_path)

# Call the function from Figure_2.py
generate_figure_2(training_df, torqcomp, kneevcom, figures_path)


# separate training progression
torque_NH = spm.spm_nh_kinetics(
    df=training_df[(training_df["var"] == "torque") & (training_df["tr_group"] == "NH")]
)
torque_IK = spm.spm_nh_kinetics(
    df=training_df[(training_df["var"] == "torque") & (training_df["tr_group"] == "IK")]
)
tr_progression = {"torqueNH": torque_NH, "torqueIK": torque_IK}
save_obj(
    obj=tr_progression, path=pathtodata / "_RESULTS_TRAINING" / "trprogression_spmdict"
)


v_NH = spm.spm_nh_kinetics(
    df=training_df[(training_df["var"] == "knee_v") & (training_df["tr_group"] == "NH")]
)
v_IK = spm.spm_nh_kinetics(
    df=training_df[(training_df["var"] == "knee_v") & (training_df["tr_group"] == "IK")]
)
v_progression = {"vNH": v_NH, "vIK": v_IK}
save_obj(
    obj=v_progression, path=pathtodata / "_RESULTS_TRAINING" / "vtrprogression_spmdict"
)


################### Discrete variables ##############################

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


# Calculate and save statistics for reporting
report_stats = calculate_and_save_report_stats(
    training_disc, reps_dif_df, results_output_path
)
