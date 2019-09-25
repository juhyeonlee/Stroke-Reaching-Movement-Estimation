import os
import numpy as np
import pickle
from scipy.signal import resample
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_struct import CompData

sns.set()
sns.set_context('talk')

num_subjects = 20
exclude_sub_num = 13
direction = ["L", "R"]

# list subject ID
subject_id = np.arange(0, num_subjects) + 1
subject_id = np.delete(subject_id, np.argwhere(subject_id == exclude_sub_num))

save_data_file = open('sub_data_filt.p', 'rb')
sub_data = pickle.load(save_data_file)

vel_reach_norm = {key: [] for key in range(4)}
vel_retract_norm = {key: [] for key in range(4)}

resample_num = 100

mean_val = []
for sub_id in subject_id:
    vel_reach = sub_data[sub_id].vel_affect_fore_reach
    comp_reach = sub_data[sub_id].reach_comp_score
    vel_retract = sub_data[sub_id].vel_affect_fore_retract
    comp_retract = sub_data[sub_id].retract_comp_score
    assert len(vel_reach) == len(vel_retract)
    assert len(vel_reach) == len(comp_reach)



    for ii in range(len(vel_reach)):
        vel_reach_norm[comp_reach[ii]].append(resample(vel_reach[ii], resample_num, np.arange(len(vel_reach[ii])))[0])
        vel_retract_norm[comp_retract[ii]].append(resample(vel_retract[ii], resample_num, np.arange(len(vel_retract[ii])))[0])
        mean_val.append(len(vel_reach[ii]))
        mean_val.append(len(vel_retract[ii]))

print(np.mean(mean_val), np.std(mean_val))

for comp in range(4):
    reach_data_x = pd.DataFrame(columns=['t', 'vel'])
    reach_data_y = pd.DataFrame(columns=['t', 'vel'])
    reach_data_z = pd.DataFrame(columns=['t', 'vel'])
    retract_data_x = pd.DataFrame(columns=['t', 'vel'])
    retract_data_y = pd.DataFrame(columns=['t', 'vel'])
    retract_data_z = pd.DataFrame(columns=['t', 'vel'])
    for ii in range(len(vel_reach_norm[comp])):
        reach_data_x = reach_data_x.append(pd.DataFrame({"t": np.arange(resample_num), "vel": vel_reach_norm[comp][ii][:, 0]}))
        reach_data_y = reach_data_y.append(pd.DataFrame({"t": np.arange(resample_num), "vel": vel_reach_norm[comp][ii][:, 1]}))
        reach_data_z = reach_data_z.append(pd.DataFrame({"t": np.arange(resample_num), "vel": vel_reach_norm[comp][ii][:, 2]}))
    for ii in range(len(vel_retract_norm[comp])):
        retract_data_x = retract_data_x.append(pd.DataFrame({"t": np.arange(resample_num), "vel": vel_retract_norm[comp][ii][:, 0]}))
        retract_data_y = retract_data_y.append(pd.DataFrame({"t": np.arange(resample_num), "vel": vel_retract_norm[comp][ii][:, 1]}))
        retract_data_z = retract_data_z.append(pd.DataFrame({"t": np.arange(resample_num), "vel": vel_retract_norm[comp][ii][:, 2]}))

    # vel_reach_norm_mean = np.mean(np.asarray(vel_reach_norm[comp]), axis=0)
    # vel_retract_norm_mean = np.mean(np.asarray(vel_retract_norm[comp]), axis=0)
    # vel_reach_norm_std = np.std(np.asarray(vel_reach_norm[comp]), axis=0)
    # vel_retract_norm_std = np.std(np.asarray(vel_retract_norm[comp]), axis=0)

    fig_avg_vel = plt.figure(figsize=[20, 12])

    x_reach_vel = fig_avg_vel.add_subplot(2, 3, 1)
    x_reach_vel.set_title('Avg. vel X: reach comp level {}'.format(comp))
    sns.lineplot(x="t", y="vel", data=reach_data_x, ci="sd")

    y_reach_vel = fig_avg_vel.add_subplot(2, 3, 2)
    y_reach_vel.set_title('Avg. vel Y: reach comp level {}'.format(comp))
    sns.lineplot(x="t", y="vel", data=reach_data_y, ci="sd")
    # sns.lineplot(np.arange(resample_num), vel_reach_norm_mean[:, 1])

    z_reach_vel = fig_avg_vel.add_subplot(2, 3, 3)
    z_reach_vel.set_title('Avg. vel Z: reach comp level {}'.format(comp))
    sns.lineplot(x="t", y="vel", data=reach_data_z, ci="sd")
    # sns.lineplot(np.arange(resample_num), vel_reach_norm_mean[:, 2])

    x_retract_vel = fig_avg_vel.add_subplot(2, 3, 4)
    x_retract_vel.set_title('Avg. vel X: retract comp level {}'.format(comp))
    sns.lineplot(x="t", y="vel", data=retract_data_x, ci="sd")
    # sns.lineplot(np.arange(resample_num), vel_retract_norm_mean[:, 0])

    y_retract_vel = fig_avg_vel.add_subplot(2, 3, 5)
    y_retract_vel.set_title('Avg. vel Y: retract comp level {}'.format(comp))
    sns.lineplot(x="t", y="vel", data=retract_data_y, ci="sd")

    # sns.lineplot(np.arange(resample_num), vel_retract_norm_mean[:, 1])

    z_retract_vel = fig_avg_vel.add_subplot(2, 3, 6)
    z_retract_vel.set_title('Avg. vel Z: retract comp level {}'.format(comp))
    sns.lineplot(x="t", y="vel", data=retract_data_z, ci="sd")
    # sns.lineplot(np.arange(resample_num), vel_retract_norm_mean[:, 2])

    fig_avg_vel.savefig('avg_vel_comp_level_{}'.format(comp))

    plt.show()


print('all set')