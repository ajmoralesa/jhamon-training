# This script generates results of the functional data recorded during the
# Nordic Hamstring and Isokinetic training sessions.

from jhamon.signal.tnorm import tnorm
from pathlib import Path
from jhamon_training import check_result_file


from jhamon.data import frames, alocate_training_group
from jhamon.data.training import dame_manual_exclusions
from jhamon.saveload import save_obj
import jhamon.stats.training as spmtr
import numpy as np
import pandas as pd
import seaborn as sns
import feather
import matplotlib.pyplot as plt

pathtodata = Path(r'D:')

# NORDIC training sessions
nordict = check_result_file(pathtodata, res_file='nht_results.pkl')
nordf = frames.nht_todf(my_dict=nordict)

# IK training sessions
ikdict = check_result_file(pathtodata, res_file='ikt_results.pkl')
ikdf = frames.ikt_todf(my_dict=ikdict)

# both trainings
training_df = pd.concat([nordf, ikdf])

# filter data
training_dfilt = dame_manual_exclusions(training_df)

# save dataframe
pathtosave = str(pathtodata + '\\_RESULTS\\')
feather.write_dataframe(training_dfilt, pathtosave + 'training_results')
feather.write_dataframe(training_df, pathtosave + 'training_results_all')

for p in nordict.keys():
    for t in nordict[p].keys():
        if not nordict[p][t]:
            print("Missing ", str(t), "from ", str(p))


for p in ikdict.keys():
    print(p, len(ikdict[p].keys()))
    for t in ikdict[p].keys():
        if not ikdict[p][t]:
            print("Missing ", str(t), "from ", str(p))


# calculate work variable in IK group
dw = training_dfilt[training_dfilt['var'] == 'knee_work']

sns.lineplot(x='timepoint', y='value', hue='tr_group', data=dw)
plt.show()

# save discrete variables data frame
def nht_disc_todf(my_dict):

    from jhamon.data import alocate_training_group
    import pandas as pd

    datadf = pd.DataFrame({(p, n, s, r, v): my_dict[p][n][s][r][0][v]
                           for p in my_dict.keys()
                           for n in my_dict[p].keys()
                           for s in my_dict[p][n].keys()
                           for r in my_dict[p][n][s].keys()
                           for v in my_dict[p][n][s][r][0].keys()}, index=[0]).stack(level=[0, 1, 2, 3, 4]).to_frame()

    datadf.reset_index(inplace=True)
    datadf.columns = ['timepoint', 'par',
                      'trses', 'set', 'rep', 'var', 'value']

    # create 'tr_group' variable
    NH, IK, CO = alocate_training_group()
    datadf.loc[datadf['par'].isin(NH), 'tr_group'] = 'NH'
    datadf.loc[datadf['par'].isin(IK), 'tr_group'] = 'IK'
    datadf.loc[datadf['par'].isin(CO), 'tr_group'] = 'CO'

    colu = ['par', 'trses', 'set', 'rep', 'var', 'tr_group']
    datadf['all_labels'] = datadf[colu].apply(lambda x: ' '.join(x), axis=1)
    datadf.reset_index(inplace=True)

    return datadf


nor_disc_df = nht_disc_todf(my_dict=nordict)


# save discrete variables IK
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


########################################################## SPM STATISTICS ############################################################

# # Training modes: NH vs IK
# group_spmdict = spmtr.spm_group_comparison(df=training_dfilt[training_dfilt['var']=='torque'])
# save_obj(obj=group_spmdict, path=str(pathtodata + '\\_RESULTS\\' + 'group_spmdict'))


# compare between tr_group and tr_1: TORQUE
training_spmdict = spmtr.spm_aov_trevolution(
    df=training_dfilt[training_dfilt['var'] == 'torque'])
save_obj(obj=training_spmdict, path=str(
    pathtodata + '\\_RESULTS\\' + 'training_spmdict'))


# NH vs IK comparison
torqcomp = spmtr.spm_group_comparison(
    df=training_dfilt[training_dfilt['var'] == 'torque'])
save_obj(obj=torqcomp, path=str(pathtodata + '\\_RESULTS\\' + 'tor_spmdict'))

kneevcom = spmtr.spm_group_comparison(
    df=training_dfilt[training_dfilt['var'] == 'knee_v'])
save_obj(obj=kneevcom, path=str(pathtodata + '\\_RESULTS\\' + 'kneev_spmdict'))


kneevcom['ti'].clusters


# separate training progression


torque_NH = spmtr.spm_nh_kinetics(df=training_dfilt[(
    training_dfilt['var'] == 'torque') & (training_dfilt['tr_group'] == 'NH')])
torque_IK = spmtr.spm_nh_kinetics(df=training_dfilt[(
    training_dfilt['var'] == 'torque') & (training_dfilt['tr_group'] == 'IK')])
tr_progression = {'torqueNH': torque_NH, 'torqueIK': torque_IK}
save_obj(obj=tr_progression, path=str(
    pathtodata + '\\_RESULTS\\' + 'trprogression_spmdict'))


tr_progression.keys()

torque_NH['tr_2'].keys()

['ti'].clusters



v_NH = spmtr.spm_nh_kinetics(df=training_dfilt[(
    training_dfilt['var'] == 'knee_v') & (training_dfilt['tr_group'] == 'NH')])
v_IK = spmtr.spm_nh_kinetics(df=training_dfilt[(
    training_dfilt['var'] == 'knee_v') & (training_dfilt['tr_group'] == 'IK')])
v_progression = {'vNH': v_NH, 'vIK': v_IK}
save_obj(obj=v_progression, path=str(pathtodata +
         '\\_RESULTS\\' + 'vtrprogression_spmdict'))


# Hip kinetics during the NHT
hipROM_spmdict = spmtr.spm_nh_kinetics(
    df=training_dfilt[training_dfilt['var'] == 'hip_ROM'])
hipV_spmdict = spmtr.spm_nh_kinetics(
    df=training_dfilt[training_dfilt['var'] == 'hip_v'])

NH_kinetics_spmdict = {'hipROM': hipROM_spmdict, 'hipV': hipV_spmdict}
save_obj(obj=NH_kinetics_spmdict, path=str(
    pathtodata + '\\_RESULTS\\' + 'NHkinematics_spmdict'))


# Save file with anthropometric data


ant = get_anthro()

pd.DataFrame(ant)

feather.write_dataframe(pd.DataFrame(ant), str(
    pathtodata + '\\_RESULTS\\' + 'anthro'))
