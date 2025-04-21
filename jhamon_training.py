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
from jhamon_training.stats.analyses_discrete_vars import analyze_discrete_variables
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

################# SPM STATISTICS ##############
# NH vs IK : TORQUE and Knee Velocity
torqcomp = spmtr.spm_group_comparison(df=training_df[training_df["var"] == "torque"])
kneevcom = spmtr.spm_group_comparison(df=training_df[training_df["var"] == "knee_v"])
generate_figure_2(training_df, torqcomp, kneevcom, figures_path)

# Training progression : TORQUE
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

# Training progression : Knee Velocity
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

# Analyze discrete variables
discrete_analysis_results = analyze_discrete_variables(
    nordict=nordict,
    ikdf=ikdf,
    training_df=training_df,
    results_output_path=results_output_path,
)

# You can access any values you need from the results dictionary
# For example:
# training_disc = discrete_analysis_results["training_disc"]
# report_stats = discrete_analysis_results["report_stats"]
