# This script generates results of the functional data recorded during the
# Nordic Hamstring and Isokinetic training sessions.

import matplotlib.pyplot as plt
from pathlib import Path
from jhamon_training import check_result_file
from jhamon_training.data import frames

from jhamon.data.training import dame_manual_exclusions
from jhamon.saveload import save_obj
import jhamon.stats.training as spmtr
import jhamon_training.stats.spm as spm
import pandas as pd
import feather
import numpy as np

pathtodata = Path(r'E:')
pathtodata = Path(r'C:/Users/amorales/Desktop/')

# NORDIC training sessions
nordict = check_result_file(pathtodata, res_file='nht_results.pkl')
nordf = frames.nht_todf(my_dict=nordict)

# IK training sessions
ikdict = check_result_file(pathtodata, res_file='ikt_results.pkl')
ikdf = frames.ikt_todf(my_dict=ikdict)

# Merge both datasets
training_df = pd.concat([nordf, ikdf])

# # filter data
# training_dfilt = dame_manual_exclusions(training_df)

training_df_torque = training_df[training_df['var'] == 'torque']
training_df_velocity = training_df[training_df['var'] == 'knee_v']

# save dataframe
pathtosave = pathtodata / '_RESULTS_TRAINING'

feather.write_dataframe(
    training_df_torque, pathtosave / 'training_results_torque')
feather.write_dataframe(
    training_df_velocity, pathtosave / 'training_results_velocity')

################# SPM STATISTICS ##############

# NH vs IK : TORQUE
torqcomp = spmtr.spm_group_comparison(
    df=training_df[training_df['var'] == 'torque'])
save_obj(obj=torqcomp, path=pathtodata /
         '_RESULTS_TRAINING' / 'tor_spmdict.pkl')

# NH vs IK : Knee Velocity
kneevcom = spmtr.spm_group_comparison(
    df=training_df[training_df['var'] == 'knee_v'])
save_obj(obj=kneevcom, path=pathtodata /
         '_RESULTS_TRAINING' / 'kneev_spmdict.pkl')


# separate training progression
torque_NH = spm.spm_nh_kinetics(df=training_df[(
    training_df['var'] == 'torque') & (training_df['tr_group'] == 'NH')])
torque_IK = spm.spm_nh_kinetics(df=training_df[(
    training_df['var'] == 'torque') & (training_df['tr_group'] == 'IK')])
tr_progression = {'torqueNH': torque_NH, 'torqueIK': torque_IK}
save_obj(obj=tr_progression, path=pathtodata /
         '_RESULTS_TRAINING' / 'trprogression_spmdict')


v_NH = spm.spm_nh_kinetics(df=training_df[(
    training_df['var'] == 'knee_v') & (training_df['tr_group'] == 'NH')])
v_IK = spm.spm_nh_kinetics(df=training_df[(
    training_df['var'] == 'knee_v') & (training_df['tr_group'] == 'IK')])
v_progression = {'vNH': v_NH, 'vIK': v_IK}
save_obj(obj=v_progression, path=pathtodata /
         '_RESULTS_TRAINING' / 'vtrprogression_spmdict')


################### Discrete variables ##############################

# Nordic discrete variables
nor_disc_df = frames.nht_disc_todf(my_dict=nordict)

nor_disc_df['var'].unique()


# Filter the DataFrame for rows where 'var' is equal to 'knee_ROMfpeak'
filtered_df = nor_disc_df[nor_disc_df['var'] == 'knee_fpeak']

# Create a histogram of the 'value' column in the filtered DataFrame
# You can adjust the number of bins as needed
plt.hist(filtered_df['value'], bins=50)

# Add labels and a title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of knee_ROMfpeak')

# Show the histogram
plt.show()


# Assuming your DataFrame is named 'nor_disc_df'

# Filter the DataFrame for rows where 'var' is equal to 'knee_ROMfpeak'
filtered_df = nor_disc_df[nor_disc_df['var'] == 'knee_ROM']

# Extract the participants and their corresponding 'value' column
participants = filtered_df['par']
values = filtered_df['value']

# Create a scatterplot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
# You can adjust the marker size (s) and transparency (alpha)
plt.scatter(participants, values, s=50, alpha=0.5)

# Customize the plot labels and title
plt.xlabel('Participants')
plt.ylabel('Value')
plt.title('Scatterplot of Values for var == knee_ROMfpeak')

# Rotate x-axis labels for better visibility (optional)
plt.xticks(rotation=90)

# Show the plot
plt.tight_layout()  # Ensures that labels are not cut off
plt.show()


# IK discrete variables
iktor = ikdf[ikdf["var"] == "torque"]
tor_wide = (iktor[['timepoint', 'all_labels', 'value']]
            .pivot(index='timepoint', columns='all_labels', values='value'))

ikrom = ikdf[ikdf["var"] == "knee_ROM"]
rom_wide = (ikrom[['timepoint', 'all_labels', 'value']]
            .pivot(index='timepoint', columns='all_labels', values='value'))

aa = np.abs(np.trapz(tor_wide.values, np.deg2rad(rom_wide.values), axis=0))
ikwork = pd.DataFrame(aa, iktor[iktor['timepoint'] == 0][[
                      'par', 'trses', 'set', 'rep']]).reset_index()

new_col_list = ['par', 'trses', 'set', 'rep']
for n, col in enumerate(new_col_list):
    ikwork[col] = ikwork['index'].apply(lambda index: index[n])
ikwork = ikwork.drop('index', axis=1)
ikwork.columns = ['value', 'par', 'trses', 'set', 'rep']
ikwork['var'] = 'knee_work'
ikwork['tr_group'] = 'IK'

training_disc = pd.concat([nor_disc_df[['par', 'trses', 'set', 'rep', 'var', 'value', 'tr_group']],
                           ikwork[['par', 'trses', 'set', 'rep', 'var', 'value', 'tr_group']]])

# save to continue analyses in R
feather.write_dataframe(training_disc, pathtosave + 'training_disc')


# Function to calculate peak torque and angle associated during IK

for p in ikdict.keys():
    for n in ikdict[p].keys():
        for s in ikdict[p][n].keys():
            for r in ikdict[p][n][s].keys():
                print(p, n, s, r)
                fmax_idx = ikdict[p][n][s][r]['torque'].argmax()
                fmax_ROM = ikdict[p][n][s][r]['knee_ROM'][fmax_idx]


plt.plot(ikdict[p][n][s][r]['knee_ROM'])
plt.plot(ikdict[p][n][s][r]['torque'])
plt.show()


ikdict[p][n][s][r].keys()
