# This script generates results of the functional data recorded during the
# Nordic Hamstring and Isokinetic training sessions.

import matplotlib.pyplot as plt
from pathlib import Path
from jhamon_training import check_result_file
from jhamon_training.data import frames
import os
from jhamon.data.training import dame_manual_exclusions
from jhamon.saveload import save_obj
import jhamon.stats.training as spmtr
import jhamon_training.stats.spm as spm
import pandas as pd
import numpy as np
from scipy import integrate  # Add import for scipy.integrate

pathtodata = Path("/Volumes/AJMA/")
results_output_path = Path.home() / "Desktop" / "_RESULTS_TRAINING"

# NORDIC training sessions
nordict = check_result_file(
    pathtodata,
    results_output_path,
    participant_id="jhamon01",
    res_file="nht_results.pkl",
)
nordf = frames.nht_todf(my_dict=nordict)

# IK training sessions
ikdict = check_result_file(pathtodata, results_output_path, res_file="ikt_results.pkl")
ikdf = frames.ikt_todf(my_dict=ikdict)

# Merge both datasets
training_df = pd.concat([nordf, ikdf])

# # filter data
# training_dfilt = dame_manual_exclusions(training_df)

training_df_torque = training_df[training_df["var"] == "torque"]
training_df_velocity = training_df[training_df["var"] == "knee_v"]

# save dataframe
pathtosave = pathtodata / "_RESULTS_TRAINING"

# feather.write_dataframe(training_df_torque, str(pathtosave / 'training_results_torque.feather')) # Old way
# feather.write_dataframe(training_df_velocity, str(pathtosave / 'training_results_velocity.feather')) # Old way
training_df_torque.to_feather(
    pathtosave / "training_results_torque.feather"
)  # Use pandas.to_feather
training_df_velocity.to_feather(
    pathtosave / "training_results_velocity.feather"
)  # Use pandas.to_feather

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


# Filter the DataFrame for rows where 'var' is equal to 'knee_ROMfpeak'
filtered_df = nor_disc_df[nor_disc_df["var"] == "knee_fpeak"]

# Create a histogram of the 'value' column in the filtered DataFrame
# You can adjust the number of bins as needed
plt.hist(filtered_df["value"], bins=50)

# Add labels and a title
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of knee_ROMfpeak")

# Show the histogram
plt.show()


# Assuming your DataFrame is named 'nor_disc_df'

# Filter the DataFrame for rows where 'var' is equal to 'knee_ROMfpeak'
filtered_df = nor_disc_df[nor_disc_df["var"] == "knee_ROM"]

# Extract the participants and their corresponding 'value' column
participants = filtered_df["par"]
values = filtered_df["value"]

# Create a scatterplot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
# You can adjust the marker size (s) and transparency (alpha)
plt.scatter(participants, values, s=50, alpha=0.5)

# Customize the plot labels and title
plt.xlabel("Participants")
plt.ylabel("Value")
plt.title("Scatterplot of Values for var == knee_ROMfpeak")

# Rotate x-axis labels for better visibility (optional)
plt.xticks(rotation=90)

# Show the plot
plt.tight_layout()  # Ensures that labels are not cut off
plt.show()


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
ikwork = pd.DataFrame(
    aa, iktor[iktor["timepoint"] == 0][["par", "trses", "set", "rep"]]
).reset_index()

new_col_list = ["par", "trses", "set", "rep"]
for n, col in enumerate(new_col_list):
    ikwork[col] = ikwork["index"].apply(lambda index: index[n])
ikwork = ikwork.drop("index", axis=1)
ikwork.columns = ["value", "par", "trses", "set", "rep"]
ikwork["var"] = "knee_work"
ikwork["tr_group"] = "IK"

training_disc = pd.concat(
    [
        nor_disc_df[["par", "trses", "set", "rep", "var", "value", "tr_group"]],
        ikwork[["par", "trses", "set", "rep", "var", "value", "tr_group"]],
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
