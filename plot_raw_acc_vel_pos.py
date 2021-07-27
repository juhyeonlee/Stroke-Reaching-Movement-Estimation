import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import cumtrapz
from scipy.signal import filtfilt
from scipy.signal import butter
from utils import plot_save_xyz, magnitude_xy
from data_struct import CompData

num_subjects = 20
exclude_sub_num = 13
direction = ["L", "R"]

# list subject ID
subject_id = np.arange(0, num_subjects) + 1
subject_id = np.delete(subject_id, np.argwhere(subject_id == exclude_sub_num))

save_data_file = open('sub_data.p', 'rb')
sub_data = pickle.load(save_data_file)

# dd = []
# ll = []
# for ii in subject_id:
#     dd = list(sub_data[ii].reach_comp_score.astype(float))
#     ll = list(sub_data[ii].retract_comp_score.astype(float))
#
#     print(len(dd))
#     fig_reach_dist = plt.figure()
#     plt.title('Sub {} / reach / total # {}'.format(ii, len(dd)))
#     sns.countplot(dd, order=[0, 1, 2, 3])
#     plt.xlim(-0.5, 3.5)
#     fig_reach_dist.savefig(os.path.join('fig_distribution', 'sub{}_reach'.format(ii)))
#     # plt.show()
#
#     fig_retract_dist = plt.figure()
#     plt.title('Sub {} / retract / total # {}'.format(ii, len(ll)))
#     sns.countplot(ll, order=[0, 1, 2, 3])
#     plt.xlim(-0.5, 3.5)
#     fig_retract_dist.savefig(os.path.join('fig_distribution', 'sub{}_retract'.format(ii)))
#     plt.close(fig_reach_dist)
#     plt.close(fig_retract_dist)
#     plt.show()

fs = 100
dx = 1 / fs
f_lp = 4
f_bp_l = 0.01
f_bp_h = 1.5
b_lp, a_lp = butter(4, (f_lp / (fs/2)), btype='lowpass')
b_bp, a_bp = butter(4, (f_bp_l /(fs/ 2), f_bp_h / (fs / 2)), btype='bandpass')

# rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) # y --> -y --> x, x --> y

for sub_id in subject_id:
    acc_affect_fore_reach = sub_data[sub_id].free_acc_affect_fore_reach
    acc_affect_fore_retract = sub_data[sub_id].free_acc_affect_fore_retract
    print(sub_id)
    for ii in range(len(acc_affect_fore_reach)):
        acc_reach_filt = filtfilt(b_lp, a_lp, acc_affect_fore_reach[ii], axis=0)
        acc_retract_filt = filtfilt(b_lp, a_lp, acc_affect_fore_retract[ii], axis=0)
        acc_reach_filt = acc_reach_filt - acc_reach_filt.mean(axis=0)
        acc_retract_filt = acc_retract_filt - acc_retract_filt.mean(axis=0)

        vel_reach = cumtrapz(acc_reach_filt, dx=dx, axis=0, initial=0)
        vel_retract = cumtrapz(acc_retract_filt, dx=dx, axis=0, initial=0)

        vel_reach_filt = filtfilt(b_bp, a_bp, vel_reach, axis=0)
        vel_retract_filt = filtfilt(b_bp, a_bp, vel_retract, axis=0)

        sub_data[sub_id].vel_affect_fore_reach.append(vel_reach)
        sub_data[sub_id].vel_affect_fore_retract.append(vel_retract)

        sub_data[sub_id].velfilt_affect_fore_reach.append(vel_reach_filt)
        sub_data[sub_id].velfilt_affect_fore_retract.append(vel_retract_filt)

        sub_data[sub_id].accfilt_affect_fore_reach.append(acc_reach_filt)
        sub_data[sub_id].accfilt_affect_fore_retract.append(acc_retract_filt)

        # # plot acc
        # plot_save_xyz(sub_id, sub_data[sub_id].target_dist[ii], sub_data[sub_id].target_angle[ii],
        #               sub_data[sub_id].reach_comp_score[ii], sub_data[sub_id].retract_comp_score[ii],
        #               np.concatenate((acc_reach_filt, acc_retract_filt), axis=0),
        #               len(acc_reach_filt), 'fig_acc',
        #               np.concatenate((acc_affect_fore_reach[ii], acc_affect_fore_retract[ii]), axis=0))
        #
        # # plot velocity
        # plot_save_xyz(sub_id, sub_data[sub_id].target_dist[ii], sub_data[sub_id].target_angle[ii],
        #               sub_data[sub_id].reach_comp_score[ii], sub_data[sub_id].retract_comp_score[ii],
        #               np.concatenate((vel_reach_filt, vel_retract_filt), axis=0),
        #               len(vel_reach_filt), 'fig_vel',
        #               np.concatenate((vel_reach, vel_retract), axis=0))
        mvel_xy_reach = magnitude_xy(vel_reach_filt)
        mvel_xy_retract = magnitude_xy(vel_retract_filt)
        mag_xy_vel_z = np.zeros((len(mvel_xy_reach)+ len(mvel_xy_retract), 3))
        mag_xy_vel_z[:len(mvel_xy_reach), 0] = mvel_xy_reach
        mag_xy_vel_z[:len(mvel_xy_reach), 1] = mvel_xy_reach
        mag_xy_vel_z[len(mvel_xy_reach):, 0] = mvel_xy_retract
        mag_xy_vel_z[len(mvel_xy_reach):, 1] = mvel_xy_retract
        mag_xy_vel_z[:len(mvel_xy_reach), 2] = vel_reach_filt[:, 2]
        mag_xy_vel_z[len(mvel_xy_reach):, 2] = vel_retract_filt[:, 2]
        # plot_save_xyz(sub_id, sub_data[sub_id].target_dist[ii], sub_data[sub_id].target_angle[ii],
        #               sub_data[sub_id].reach_comp_score[ii], sub_data[sub_id].retract_comp_score[ii],
        #               mag_xy_vel_z, len(vel_reach_filt), 'fig_mag_xy_vel_z')


save_data_file = open('sub_data_filt_ok.p', 'wb')
pickle.dump(sub_data, save_data_file)

# # pose figure!
# fig_vel = plt.figure()
# plt.title('velocity: sub ID {} / pos {} cm / {} degree / affect side {} / comp label {}'.format(
#     sub_id, sub_data[sub_id].target_dist[ii], sub_data[sub_id].target_angle[ii],
#     direction[sub_data[sub_id].affect_side], sub_data[sub_id].reach_comp_score[ii]))
#
# ax_vel_x = fig_vel.add_subplot(311)
# ax_vel_x.plot(np.hstack((vel_filt[:, 0], vel_retract_filt[:, 0])), label='X')
# ax_vel_y = fig_vel.add_subplot(312)
# ax_vel_y.plot(np.hstack((vel_filt[:, 1], vel_retract_filt[:, 1])), label='Y')
# ax_vel_z = fig_vel.add_subplot(313)
# ax_vel_z.plot(np.hstack((vel_filt[:, 2], vel_retract_filt[:, 2])), label='Z')
#
# plt.axhline(0, color='k')
# plt.axvline(0, color='k')
# plt.legend()
# plt.show()
#
# hold = cumtrapz(vel_filt, dx=dx, axis=0, initial=0)
# fig_pos = plt.figure()
# plt.title('sub ID {} / pos {} cm / {} degree / affect side {} / comp label {}'.format(sub_id, sub_data[sub_id].target_dist[ii],
#                                                                           sub_data[sub_id].target_angle[ii],
#                                                                           direction[sub_data[sub_id].affect_side],
#                                                                           sub_data[sub_id].reach_comp_score[ii]))
# # plt.xlim(-5000, 5000)
# # plt.ylim(-5000, 5000)
# plt.axhline(0)
# plt.axvline(0)
# sns.scatterplot(hold[:, 0], hold[:, 1], zorder=10,
#             color='steelblue', alpha=0.5)
# # sns.lineplot(np.arange(len(acc)), hold)
# plt.show()