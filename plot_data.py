import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import cumtrapz
from scipy.signal import filtfilt
from scipy.signal import butter
from data_struct import CompData

num_subjects = 20
exclude_sub_num = 13

# list subject ID
subject_id = np.arange(0, num_subjects) + 1
subject_id = np.delete(subject_id, np.argwhere(subject_id == exclude_sub_num))

save_data_file = open('sub_data.p', 'rb')
sub_data = pickle.load(save_data_file)

fs = 128
fc_lp = 8
b_bp, a_bp = butter(2, fc_lp/(fs/2), 'lowpass')
b_bp_2, a_bp_2 = butter(2, (0.01/(fs/2),fc_lp/(fs/2)), 'bandpass')


# rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

for sub_id in subject_id:
    acc_affect_fore_reach = sub_data[sub_id].free_acc_affect_fore_reach
    for ii in range(len(acc_affect_fore_reach)):
        acc = acc_affect_fore_reach[ii]
        # acc = (rotation_matrix @ acc_affect_fore_reach[ii].T).T

        hold = filtfilt(b_bp, a_bp, acc, axis=0)
        hold = cumtrapz(hold,
                        np.arange(len(acc)), axis=0, initial=0)

        hold = filtfilt(b_bp_2, a_bp_2, hold, axis=0)
        hold = cumtrapz(hold, np.arange(len(acc)), axis=0, initial=0)

        plt.title('pos {} cm / {} degree / affect side {} / comp label {}'.format(sub_data[sub_id].target_dist[ii],
                                                                  sub_data[sub_id].target_angle[ii],
                                                                  sub_data[sub_id].affect_side, sub_data[sub_id].reach_comp_score[ii]))
        plt.xlim(-5000, 5000)
        plt.ylim(-5000, 5000)
        plt.axhline(0)
        plt.axvline(0)
        sns.scatterplot(hold[:, 0], hold[:, 1], zorder=10,
                    color='steelblue', alpha=0.5)
        # sns.lineplot(np.arange(len(acc)), hold)
        plt.show()