import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import cumtrapz
from scipy.signal import filtfilt
from scipy.signal import butter
import seaborn as sns
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
fc_lp = 6
fc_lp_2 = 3
b_lp, a_lp = butter(6, fc_lp / (fs/2), btype='lowpass')
b_hp, a_hp = butter(6, fc_lp_2 / (fs/2), btype='lowpass')
dx = 1 / fs

rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) # y --> -y --> x, x --> y

for sub_id in subject_id:
    acc_affect_fore_reach = sub_data[sub_id].free_acc_affect_fore_reach
    acc_affect_fore_retract = sub_data[sub_id].free_acc_affect_fore_retract
    for ii in range(len(acc_affect_fore_reach)):
        acc_reach = acc_affect_fore_reach[ii]
        acc_retract = acc_affect_fore_retract[ii]
        # acc_reach = (rotation_matrix @ acc_reach.T).T
        # acc_retract = (rotation_matrix @ acc_retract.T).T

        acc_reach_filt = filtfilt(b_lp, a_lp, acc_reach, axis=0)
        acc_retract_filt = filtfilt(b_lp, a_lp, acc_retract, axis=0)

        acc_reach_filt = acc_reach_filt - acc_reach_filt.mean(axis=0)
        acc_retract_filt = acc_retract_filt - acc_retract_filt.mean(axis=0)

        # fig_acc = plt.figure()
        #
        # ax_acc_x = fig_acc.add_subplot(3, 1, 1)
        # plt.title('acc: ID {} / pos {} cm / {} deg / affect {} / comp label {}'.format(
        #     sub_id, sub_data[sub_id].target_dist[ii], sub_data[sub_id].target_angle[ii],
        #     direction[sub_data[sub_id].affect_side], sub_data[sub_id].reach_comp_score[ii]))
        # ax_acc_x.plot(np.hstack((acc_reach[:, 0], acc_retract[:, 0])), label='X')
        # ax_acc_x.plot(np.hstack((acc_reach_filt[:, 0], acc_retract_filt[:, 0])), label='X filtered')
        # ax_acc_x.legend()
        # ax_acc_x.set_ylabel('acc')
        # ax_acc_x.axhline(0, color='k')
        # ax_acc_x.axvline(len(acc_reach), color='k', linestyle='--')
        #
        # ax_acc_y = fig_acc.add_subplot(3, 1, 2)
        # ax_acc_y.plot(np.hstack((acc_reach[:, 1], acc_retract[:, 1])), label='Y')
        # ax_acc_y.plot(np.hstack((acc_reach_filt[:, 1], acc_retract_filt[:, 1])), label='Y filtered')
        # ax_acc_y.legend()
        # ax_acc_y.set_ylabel('acc')
        # ax_acc_y.axhline(0, color='k')
        # ax_acc_y.axvline(len(acc_reach), color='k', linestyle='--')
        #
        #
        # ax_acc_z = fig_acc.add_subplot(3, 1, 3)
        # ax_acc_z.plot(np.hstack((acc_reach[:, 2], acc_retract[:, 2])), label='Z')
        # ax_acc_z.plot(np.hstack((acc_reach_filt[:, 2], acc_retract_filt[:, 2])), label='Z filtered')
        # ax_acc_z.legend()
        # ax_acc_z.set_ylabel('acc')
        # ax_acc_z.axhline(0, color='k')
        # ax_acc_z.axvline(len(acc_reach), color='k', linestyle='--')
        #
        #
        # fig_acc.tight_layout()
        # fig_acc.savefig(os.path.join('fig_acc', 'acc_i{}_{}cm_{}d_{}_s{}'.format(sub_id, sub_data[sub_id].target_dist[ii], sub_data[sub_id].target_angle[ii],
        #     direction[sub_data[sub_id].affect_side], sub_data[sub_id].reach_comp_score[ii])))

        vel_reach = cumtrapz(acc_reach, dx=dx, axis=0, initial=0)
        vel_retract = cumtrapz(acc_retract, dx=dx, axis=0, initial=0)
        vel_reach_from_accfilt = cumtrapz(acc_reach_filt, dx=dx, axis=0, initial=0)
        vel_retract_from_accfilt = cumtrapz(acc_retract_filt, dx=dx, axis=0, initial=0)

        # fig_vel = plt.figure()
        #
        # vel_acc_x = fig_vel.add_subplot(311)
        # plt.title('vel acc vs acc filt: sID {} / pos {} cm / {} deg / affect {} / comp label {}'.format(
        #         sub_id, sub_data[sub_id].target_dist[ii], sub_data[sub_id].target_angle[ii],
        #         direction[sub_data[sub_id].affect_side], sub_data[sub_id].reach_comp_score[ii]))
        # vel_acc_x.plot(np.hstack((vel_reach[:, 0], vel_retract[:, 0])), label='X')
        # vel_acc_x.plot(np.hstack((vel_reach_from_accfilt[:, 0], vel_retract_from_accfilt[:, 0])), label='X filtered')
        # vel_acc_x.legend()
        # vel_acc_x.set_ylabel('vel')
        # vel_acc_x.axhline(0, color='k')
        # vel_acc_x.axvline(len(vel_reach), color='k', linestyle='--')
        #
        #
        # vel_acc_y = fig_vel.add_subplot(312)
        # vel_acc_y.plot(np.hstack((vel_reach[:, 1], vel_retract[:, 1])), label='Y')
        # vel_acc_y.plot(np.hstack((vel_reach_from_accfilt[:, 1], vel_retract_from_accfilt[:, 1])), label='Y filtered')
        # vel_acc_y.legend()
        # vel_acc_y.set_ylabel('vel')
        # vel_acc_y.axhline(0, color='k')
        # vel_acc_y.axvline(len(vel_reach), color='k', linestyle='--')
        #
        #
        # vel_acc_z = fig_vel.add_subplot(313)
        # vel_acc_z.plot(np.hstack((vel_reach[:, 2], vel_retract[:, 2])), label='Z')
        # vel_acc_z.plot(np.hstack((vel_reach_from_accfilt[:, 2], vel_retract_from_accfilt[:, 2])), label='Z filtered')
        # vel_acc_z.legend()
        # vel_acc_z.set_ylabel('vel')
        # vel_acc_z.axhline(0, color='k')
        # vel_acc_z.axvline(len(vel_reach), color='k', linestyle='--')
        #
        #
        # plt.ylabel('vel')
        # fig_vel.tight_layout()
        # # fig_vel.savefig(os.path.join('fig_vel', '{}_{}_{}_{}_{}'.format(sub_id, sub_data[sub_id].target_dist[ii],
        # #                                                              sub_data[sub_id].target_angle[ii],
        # #                                                              direction[sub_data[sub_id].affect_side],
        # #                                                              sub_data[sub_id].reach_comp_score[ii])))
        # plt.show()


        # if vel_reach_from_accfilt.shape[0] >= 1000:
        vel_reach_filt = filtfilt(b_hp, a_hp, vel_reach_from_accfilt, axis=0)
        # else:
        #     vel_reach_filt = vel_reach_from_accfilt

        # if vel_retract_from_accfilt.shape[0] >= 1000:
        vel_retract_filt = filtfilt(b_hp, a_hp, vel_retract_from_accfilt, axis=0)
        # else:
        #     vel_retract_filt = vel_retract_from_accfilt

        sub_data[sub_id].vel_affect_fore_reach.append(vel_reach_filt)
        sub_data[sub_id].vel_affect_fore_retract.append(vel_retract_filt)

        # fig_vel_filt = plt.figure()
        #
        # vel_x = fig_vel_filt.add_subplot(311)
        # plt.title('vel filt or not: sID {} / pos {} cm / {} deg / affect {} / comp label {}'.format(
        #     sub_id, sub_data[sub_id].target_dist[ii], sub_data[sub_id].target_angle[ii],
        #     direction[sub_data[sub_id].affect_side], sub_data[sub_id].reach_comp_score[ii]))
        # vel_x.plot(np.hstack((vel_reach_filt[:, 0], vel_retract_filt[:, 0])), label='X v filter')
        # vel_x.plot(np.hstack((vel_reach_from_accfilt[:, 0], vel_retract_from_accfilt[:, 0])), label='X v')
        # vel_x.legend()
        # vel_x.set_ylabel('vel')
        # vel_x.axhline(0, color='k')
        # vel_x.axvline(len(vel_reach), color='k', linestyle='--')
        #
        # vel_y = fig_vel_filt.add_subplot(312)
        # vel_y.plot(np.hstack((vel_reach_filt[:, 1], vel_retract_filt[:, 1])), label='Y v filter')
        # vel_y.plot(np.hstack((vel_reach_from_accfilt[:, 1], vel_retract_from_accfilt[:, 1])), label='Y V')
        # vel_y.legend()
        # vel_y.set_ylabel('vel')
        # vel_y.axhline(0, color='k')
        # vel_y.axvline(len(vel_reach), color='k', linestyle='--')
        #
        # vel_z = fig_vel_filt.add_subplot(313)
        # vel_z.plot(np.hstack((vel_reach_filt[:, 2], vel_retract_filt[:, 2])), label='Z v filter')
        # vel_z.plot(np.hstack((vel_reach_from_accfilt[:, 2], vel_retract_from_accfilt[:, 2])), label='Z v')
        # vel_z.legend()
        # vel_z.set_ylabel('vel')
        # vel_z.axhline(0, color='k')
        # vel_z.axvline(len(vel_reach), color='k', linestyle='--')
        #
        # plt.ylabel('vel')
        # fig_vel_filt.tight_layout()
        # # fig_vel_filt.savefig(os.path.join('fig_vel_filt', '{}_{}_{}_{}_{}'.format(sub_id, sub_data[sub_id].target_dist[ii],
        # #                                                                 sub_data[sub_id].target_angle[ii],
        # #                                                                 direction[sub_data[sub_id].affect_side],
        # #                                                                 sub_data[sub_id].reach_comp_score[ii])))
        # plt.show()
        # plt.close()


save_data_file = open('sub_data_filt.p', 'wb')
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