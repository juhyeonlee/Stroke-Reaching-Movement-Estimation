

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from utils import magnitude_xy, plot_confusion_matrix
import pandas as pd

num_subjects = 20
exclude_sub_num = 13
direction = ["L", "R"]
color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']


# list subject ID
subject_id = np.arange(0, num_subjects) + 1
subject_id = np.delete(subject_id, np.argwhere(subject_id == exclude_sub_num))
# with mft
subject_id = np.delete(subject_id, np.argwhere(subject_id == 3))
subject_id = np.delete(subject_id, np.argwhere(subject_id == 15))
subject_id = np.delete(subject_id, np.argwhere(subject_id == 6))
print(subject_id)
# subject_id = np.array([1, 5, 8, 9, 10, 11, 12, 17])

save_data_file = open('sub_data_filt_ok.p', 'rb')
sub_data = pickle.load(save_data_file)

# results = np.load('relieff_addf2.npy')
results = np.load('retract_exhaustive_2class.npy')

compensation_labels = []
compensation_origin_labels = []
sub_id_labels = []
mag = []
for sub_id in subject_id:
    vel_affect_fore_reach = sub_data[sub_id].vel_affect_fore_reach
    vel_affect_fore_retract = sub_data[sub_id].vel_affect_fore_retract

    # for ii in range(len(vel_affect_fore_reach)):
    #     mag.append(magnitude_xy(vel_affect_fore_reach[ii]))
    #
    # for ll in range(len(sub_data[sub_id].reach_comp_score)):
    #     compensation_origin_labels.append(sub_data[sub_id].reach_comp_score[ll])
    #     if sub_data[sub_id].reach_comp_score[ll] == 3: # or sub_data[sub_id].reach_comp_score[ll] == 2:
    #         compensation_labels.append(1)
    #     elif sub_data[sub_id].reach_comp_score[ll] != 3:
    #         compensation_labels.append(0)
    #
    # # compensation_labels.extend(sub_data[sub_id].reach_comp_score)
    # sub_id_labels.extend([sub_id] * len(sub_data[sub_id].reach_comp_score))

    for ii in range(len(vel_affect_fore_retract)):
        mag.append(magnitude_xy(vel_affect_fore_retract[ii]))

    for ll in range(len(sub_data[sub_id].retract_comp_score)):
        compensation_origin_labels.append(sub_data[sub_id].retract_comp_score[ll])
        if sub_data[sub_id].retract_comp_score[ll] == 3: # or sub_data[sub_id].retract_comp_score[ll] == 2:
            compensation_labels.append(1)
        elif sub_data[sub_id].retract_comp_score[ll] != 3:
            compensation_labels.append(0)

    # compensation_labels.extend(sub_data[sub_id].retract_comp_score)
    sub_id_labels.extend([sub_id] * len(sub_data[sub_id].retract_comp_score))

fig_conf = plt.figure(figsize=[6, 6])
ax = fig_conf.add_subplot(111)
plot_confusion_matrix(ax, compensation_labels, results, ['Abnormal', 'Normal'], normalize=True)
fig_conf.tight_layout()
plt.show()