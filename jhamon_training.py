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
import pandas as pd
import numpy as np
from scipy import integrate
from jhamon_training.plot.fpeak_plot import (
    plot_fpeak_data,
)
from jhamon_training.plot.ik_torque_plot import plot_ik_torque_curves

pathtodata = Path("/Volumes/AJMA/")
results_output_path = Path.home() / "Desktop" / "_RESULTS_TRAINING"
pathtosave = results_output_path
figures_path = pathtosave / "figures"

# Create figures directory if it doesn't exist
os.makedirs(figures_path, exist_ok=True)

# NORDIC training sessions# Record start time
nordict = check_result_file(
    pathtodata,
    results_output_path,
    res_file="nht_results.pkl",
)

nordf = frames.nht_todf(my_dict=nordict)
# nordict["jhamon01"]["tr_1"]["set_1"]["rep_1"][0]["knee_ROMfpeak"] # Example access


# IK training sessions
ikdict = check_result_file(pathtodata, results_output_path, res_file="ikt_results.pkl")
ikdf = frames.ikt_todf(my_dict=ikdict)

# ikdict["jhamon18"]["tr_1"]["set_1"]["rep_1"]["torque"] # Example access

# --- Plotting IK Torque --- #
plot_ik_torque_curves(ikdict, output_path=figures_path / "ik_torque_curves.png")


# Merge both datasets
training_df = pd.concat([nordf, ikdf])

# # filter data
training_dfilt = dame_manual_exclusions(training_df)
training_df_torque = training_df[training_df["var"] == "torque"]
training_df_velocity = training_df[training_df["var"] == "knee_v"]


################# SPM STATISTICS ##############

# NH vs IK : TORQUE
torqcomp = spmtr.spm_group_comparison(df=training_df[training_df["var"] == "torque"])
save_obj(obj=torqcomp, path=pathtodata / "_RESULTS_TRAINING" / "tor_spmdict.pkl")

# NH vs IK : Knee Velocity
kneevcom = spmtr.spm_group_comparison(df=training_df[training_df["var"] == "knee_v"])
save_obj(obj=kneevcom, path=pathtodata / "_RESULTS_TRAINING" / "kneev_spmdict.pkl")


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
nor_disc_df["var"].unique()

# Generate and save the fpeak plot
fpeak_plot = plot_fpeak_data(nordict)
fpeak_plot_path = figures_path / "fpeak_analysis.png"
# Create figures directory if it doesn't exist - redundant now, handled earlier
# os.makedirs(pathtosave / "figures", exist_ok=True)
fpeak_plot.savefig(fpeak_plot_path, dpi=300, bbox_inches="tight")
plt.close(fpeak_plot)  # Close the figure to free memory


# IK discrete variables
iktor = ikdf[ikdf["var"] == "torque"]
tor_wide = iktor[["timepoint", "all_labels", "value"]].pivot(
    index="timepoint", columns="all_labels", values="value"
)

ikrom = ikdf[ikdf["var"] == "knee_ROM"]
rom_wide = ikrom[["timepoint", "all_labels", "value"]].pivot(
    index="timepoint", columns="all_labels", values="value"
)

# aa = np.abs(np.trapz(tor_wide.values, np.deg2rad(rom_wide.values), axis=0)) # Old numpy trapz
aa = np.abs(
    integrate.trapezoid(tor_wide.values, x=np.deg2rad(rom_wide.values), axis=0)
)  # Use scipy.integrate.trapezoid

# Explicitly create a MultiIndex for clarity and type checking
iktor_index_df = iktor[iktor["timepoint"] == 0][
    ["par", "trses", "set", "rep"]
].drop_duplicates()
ikwork_index = pd.MultiIndex.from_frame(iktor_index_df)

ikwork = pd.DataFrame(
    aa, index=ikwork_index  # Pass the created MultiIndex
).reset_index()

# The following loop might be unnecessary if the MultiIndex is named correctly,
# but let's keep it for now as it matches the original logic.
# new_col_list = ["par", "trses", "set", "rep"] # Keep original column names
# for n, col in enumerate(new_col_list):
#     ikwork[col] = ikwork["index"].apply(lambda index: index[n]) # This assumes reset_index() creates a column named 'index'
# ikwork = ikwork.drop("index", axis=1)

# Rename the columns after reset_index if needed.
# reset_index() will create columns named after the index levels.
# Let's rename the value column first.
ikwork = ikwork.rename(columns={0: "value"})

# Check if index columns are named as expected ('par', 'trses', etc.)
# If not, rename them:
# expected_cols = ['par', 'trses', 'set', 'rep']
# rename_dict = {old_col: new_col for old_col, new_col in zip(ikwork.columns[:len(expected_cols)], expected_cols)}
# ikwork = ikwork.rename(columns=rename_dict)

ikwork["var"] = "knee_work"
ikwork["tr_group"] = "IK"

training_disc = pd.concat(
    [
        nor_disc_df[["par", "trses", "set", "rep", "var", "value", "tr_group"]],
        ikwork[
            ["par", "trses", "set", "rep", "var", "value", "tr_group"]
        ],  # Ensure column names are correct here
    ]
)

# save to continue analyses in R
# feather.write_dataframe(training_disc, str(pathtosave / 'training_disc.feather')) # Old way
training_disc.to_feather(pathtosave / "training_disc.feather")  # Use pandas.to_feather


# Function to calculate peak torque and angle associated during IK
last_p, last_n, last_s, last_r = (
    None,
    None,
    None,
    None,
)  # Initialize variables to store last keys
if ikdict:  # Check if ikdict is not empty
    for p in ikdict.keys():
        for n in ikdict[p].keys():
            for s in ikdict[p][n].keys():
                for r in ikdict[p][n][s].keys():
                    # print(p, n, s, r) # Original print statement
                    fmax_idx = ikdict[p][n][s][r]["torque"].argmax()
                    fmax_ROM = ikdict[p][n][s][r]["knee_ROM"][fmax_idx]
                    # Store the last set of keys processed
                    last_p, last_n, last_s, last_r = p, n, s, r

    # Check if the loop actually ran and assigned keys before plotting
    if all(k is not None for k in [last_p, last_n, last_s, last_r]):
        plt.plot(ikdict[last_p][last_n][last_s][last_r]["knee_ROM"])
        plt.plot(ikdict[last_p][last_n][last_s][last_r]["torque"])
        plt.title(
            f"IK Data: Participant {last_p}, Session {last_n}, Set {last_s}, Rep {last_r}"
        )  # Add title
        plt.xlabel("Timepoint/Index")  # Add x-axis label
        plt.ylabel("Value")  # Add y-axis label
        plt.legend(["Knee ROM", "Torque"])  # Add legend
        plt.show()
else:
    print("Skipping IK plot generation as ikdict is empty.")

# ikdict[p][n][s][r].keys() # This line likely caused errors if keys were unbound
