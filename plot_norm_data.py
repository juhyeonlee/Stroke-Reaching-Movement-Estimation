import os
import numpy as np
import pickle
from scipy.signal import resample, butter, filtfilt, find_peaks
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_struct import CompData
from utils import magnitude, magnitude_xy

num_subjects = 20
exclude_sub_num = 13
direction = ["L", "R"]

# list subject ID
subject_id = np.arange(0, num_subjects) + 1
subject_id = np.delete(subject_id, np.argwhere(subject_id == exclude_sub_num))

save_data_file = open('sub_data.p', 'rb')
sub_data = pickle.load(save_data_file)

fs = 100
dx = 1 / fs
f_lp = 4
f_bp_l = 0.01
f_bp_h = 2
b_lp, a_lp = butter(4, (f_lp / (fs/2)), btype='lowpass')
b_bp, a_bp = butter(4, (f_bp_l /(fs/ 2), f_bp_h / (fs / 2)), btype='bandpass')
color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']

resample_num = 500
if not os.path.exists('whole_ts_{}_{}to{}_norm{}_whole'.format(f_lp, f_bp_l, f_bp_h, resample_num)):
    os.makedirs('whole_ts_{}_{}to{}_norm{}_whole'.format(f_lp, f_bp_l, f_bp_h, resample_num))

reach_len_comp = {key: [] for key in range(4)}
retract_len_comp = {key: [] for key in range(4)}
peaks_reach_comp = {key: [] for key in range(4)}
peaks_retract_comp = {key: [] for key in range(4)}

for sub_id in subject_id:
    acc_affect_fore_reach = sub_data[sub_id].free_acc_affect_fore_reach
    acc_affect_fore_retract = sub_data[sub_id].free_acc_affect_fore_retract
    comp_reach = sub_data[sub_id].reach_comp_score
    comp_retract = sub_data[sub_id].retract_comp_score
    mvel_xy_reach_comp = {key: [] for key in range(4)}
    mvel_xy_retract_comp = {key: [] for key in range(4)}
    mvel_xy_reach_norm = {key: [] for key in range(4)}
    mvel_xy_retract_norm = {key: [] for key in range(4)}
    reach_len = []
    retract_len = []
    reach_angle = {key: [] for key in range(4)}
    retract_angle = {key: [] for key in range(4)}
    for ii in range(len(acc_affect_fore_reach)):
        acc_reach_filt = filtfilt(b_lp, a_lp, acc_affect_fore_reach[ii], axis=0)
        acc_retract_filt = filtfilt(b_lp, a_lp, acc_affect_fore_retract[ii], axis=0)
        acc_reach_filt = acc_reach_filt - acc_reach_filt.mean(axis=0)
        acc_retract_filt = acc_retract_filt - acc_retract_filt.mean(axis=0)

        vel_reach = cumtrapz(acc_reach_filt, dx=dx, axis=0, initial=0)
        vel_retract = cumtrapz(acc_retract_filt, dx=dx, axis=0, initial=0)

        vel_reach_filt = filtfilt(b_bp, a_bp, vel_reach, axis=0)
        vel_retract_filt = filtfilt(b_bp, a_bp, vel_retract, axis=0)

        mvel_xy_reach = magnitude_xy(vel_reach_filt)
        mvel_xy_retract = magnitude_xy(vel_retract_filt)
        mvel_xy_reach_comp[comp_reach[ii]].append(mvel_xy_reach)
        mvel_xy_retract_comp[comp_retract[ii]].append(mvel_xy_retract)

        reach_len.append(len(mvel_xy_reach))
        reach_len_comp[comp_reach[ii]].append(len(mvel_xy_reach))
        retract_len.append(len(mvel_xy_retract))
        retract_len_comp[comp_retract[ii]].append(len(mvel_xy_retract))
        reach_angle[comp_reach[ii]].append(int(np.logical_xor(sub_data[sub_id].affect_side == 1,
                                                              sub_data[sub_id].target_angle[ii] >= 0)))
        retract_angle[comp_retract[ii]].append(int(np.logical_xor(sub_data[sub_id].affect_side == 1,
                                                              sub_data[sub_id].target_angle[ii] >= 0)))


        mvel_xy_reach_norm[comp_reach[ii]].append(
            resample(mvel_xy_reach / np.mean(mvel_xy_reach), resample_num, np.arange(len(mvel_xy_reach)))[0])

        mvel_xy_retract_norm[comp_retract[ii]].append(
            resample(mvel_xy_retract / np.mean(mvel_xy_retract), resample_num, np.arange(len(mvel_xy_retract)))[0])

        peaks_reach, _ = find_peaks(mvel_xy_reach)
        peaks_retract, _ = find_peaks(mvel_xy_retract)
        peaks_reach_comp[comp_reach[ii]].append(len(peaks_reach))
        peaks_retract_comp[comp_retract[ii]].append(len(peaks_retract))


    fig_vel_mag_reach = plt.figure(figsize=[15, 10])
    fig_vel_mag_retract = plt.figure(figsize=[15, 10])

    for comp in range(4):

        reach_mvel_xy_df = pd.DataFrame(columns=['t', 'vel'])
        retract_mvel_xy_df = pd.DataFrame(columns=['t', 'vel'])
        # reach_idx = np.random.choice(len(mvel_xy_reach_comp[comp]), 100, replace=False)
        # retract_idx = np.random.choice(len(mvel_xy_retract_comp[comp]), 64, replace=False)
        for ii in range(len(mvel_xy_reach_comp[comp])):
        # for ii in range(100):
              reach_mvel_xy_df = reach_mvel_xy_df.append(
                pd.DataFrame({"t": np.arange(len(mvel_xy_reach_norm[comp][ii])), "vel": mvel_xy_reach_norm[comp][ii], "color": color[reach_angle[comp][ii]], "index": ii}))

        for ii in range(len(mvel_xy_retract_comp[comp])):
        # for ii in range(64):
            retract_mvel_xy_df = retract_mvel_xy_df.append(
                pd.DataFrame(
                    {"t": np.arange(len(mvel_xy_retract_norm[comp][ii])), "vel": mvel_xy_retract_norm[comp][ii], "color": color[retract_angle[comp][ii]], "index": ii}))

        reach_mvel_xy_ax = fig_vel_mag_reach.add_subplot(2, 4, comp + 1)
        reach_mvel_xy_ax.set_title('reach comp level {}'.format(comp))
        # reach_mvel_xy_ax.set_xlim(right=max(reach_len))
        if len(mvel_xy_reach_comp[comp]) != 0:
            sns.lineplot(x="t", y="vel", data=reach_mvel_xy_df, estimator=None, units='index', hue='index', legend=False,
                             ax=reach_mvel_xy_ax, palette="ch:2.5,.25")
            # sns.lineplot(x='t', y='vel', data=reach_mvel_xy_df, ci='sd', ax=reach_mvel_xy_ax)

        retract_mvel_xy_ax = fig_vel_mag_reach.add_subplot(2, 4, comp + 5)
        retract_mvel_xy_ax.set_title('retract comp level {}'.format(comp))
        # retract_mvel_xy_ax.set_xlim(right=max(retract_len))
        if len(mvel_xy_retract_comp[comp]) != 0:
            sns.lineplot(x="t", y="vel", data=retract_mvel_xy_df, estimator=None, units='index', hue='index', legend=False,
                              ax=retract_mvel_xy_ax, palette="ch:2.5,.25")
            # sns.lineplot(x='t', y='vel', data=retract_mvel_xy_df, ci='sd', ax=retract_mvel_xy_ax)

        plt.tight_layout()

    # plt.show()
    fig_vel_mag_reach.savefig('whole_ts_{}_{}to{}_norm{}_whole/reach_per_sub_{}'.format(f_lp, f_bp_l, f_bp_h, resample_num, sub_id ))
    fig_vel_mag_retract.savefig('whole_ts_{}_{}to{}_norm{}_whole/retract_per_sub_{}'.format(f_lp, f_bp_l, f_bp_h, resample_num,  sub_id ))
    plt.close('all')

sns.set()
sns.set_context('talk')
color_palette = sns.color_palette('deep')

hist_len = plt.figure(figsize=(40, 18))
for comp in range(4):
    reach_len_ax = hist_len.add_subplot(2, 4, comp + 1)

    reach_len_ax.set_title('reach {}'.format(comp))
    sns.distplot(np.array(peaks_reach_comp[comp]), kde=False, bins=np.linspace(0, 20, 21))
    print(comp)
    print('mean reach', np.mean(peaks_reach_comp[comp]))
    print('std reach', np.std(peaks_reach_comp[comp]))

    retract_len_ax = hist_len.add_subplot(2, 4, comp + 5)

    retract_len_ax.set_title('retract {}'.format(comp))
    sns.distplot(np.array(peaks_retract_comp[comp]), kde=False, bins=np.linspace(0, 15, 16))
    print('mean rt', np.mean(peaks_retract_comp[comp]))
    print('std rt', np.std(peaks_retract_comp[comp]))

    plt.tight_layout()
hist_len.savefig('reach_retract_peaks_comp_all')

