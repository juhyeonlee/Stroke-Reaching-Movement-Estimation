
import os
import numpy as np
import pickle
from data_struct import CompData
from copy import deepcopy


num_subjects = 20
exclude_sub_num = 13

# get file paths
dir_path = './reaching_movement_data'
csv_index_file = 'indices_labels.csv'
mft_score_file = os.path.join(dir_path, 'MFT_Scores.csv')
affect_side_file = os.path.join(dir_path, 'affected_side.csv')

# left forearm, left upperarm, trunk, right upperarm, right forearm
sensor_name = ['00B4249F', '00B424B0', '00B424B4', '00B424B5', '00B424B6']
file_tag = 'MT_012106C2-004-000_'

# list subject ID
subject_id = np.arange(0, num_subjects) + 1
subject_id = np.delete(subject_id, np.argwhere(subject_id == exclude_sub_num))

# read meta csv files
mft_score_data = np.genfromtxt(mft_score_file, delimiter=',', skip_header=1, usecols=(0, 1), dtype=int)
affect_side_data = np.genfromtxt(affect_side_file, delimiter=',', skip_header=1, usecols=(0, 1), dtype=int)

# load data
sub_data = {}
for sub_id in subject_id:
    sub_data[sub_id] = CompData()

    for idx, id in enumerate(mft_score_data[:, 0]):
        if sub_id == id:
            sub_data[sub_id].mft_score = mft_score_data[idx, 1]

    for idx, id in enumerate(affect_side_data[:, 0]):
        if sub_id == id:
            sub_data[sub_id].affect_side = affect_side_data[idx, 1]

    # read individual csv file and txt
    indices_data = np.genfromtxt(os.path.join(dir_path, str(sub_id), csv_index_file),
                                 delimiter=',', skip_header=1, dtype=int)

    sub_data[sub_id].reach_begin = indices_data[:, 0]
    sub_data[sub_id].reach_end = indices_data[:, 1]
    sub_data[sub_id].retract_begin = indices_data[:, 2]
    sub_data[sub_id].retract_end = indices_data[:, 3]

    sub_data[sub_id].reach_comp_score = indices_data[:, 4]
    sub_data[sub_id].retract_comp_score = indices_data[:, 5]

    sub_data[sub_id].target_dist = indices_data[:, 6]
    sub_data[sub_id].target_angle = indices_data[:, 7]

    # to binary label (1: no comp, 0: comp)
    sub_data[sub_id].reach_comp_label = (indices_data[:, 4] > 0).astype(int)
    sub_data[sub_id].retract_comp_label = (indices_data[:, 5] > 0).astype(int)

    # left forearm, left upperarm, trunk, right upperarm, right forearm
    for idx, s_name in enumerate(sensor_name):
        time_series = np.genfromtxt(os.path.join(dir_path, str(sub_id), file_tag + s_name + '.txt'),
                                    delimiter=',', skip_header=7, usecols=(4, 5, 6))
        for ii in range(len(sub_data[sub_id].reach_begin)):
            # sub 1 because of the diff. between matlab and python index
            reach_begin = sub_data[sub_id].reach_begin[ii] - 1
            reach_end = sub_data[sub_id].reach_end[ii] - 1
            retract_begin = sub_data[sub_id].retract_begin[ii] - 1
            retract_end = sub_data[sub_id].retract_end[ii] - 1
            if idx == 0:
                sub_data[sub_id].free_acc_L_fore_reach.append(time_series[reach_begin:reach_end])
                sub_data[sub_id].free_acc_L_fore_retract.append(time_series[retract_begin:retract_end])
            elif idx == 1:
                sub_data[sub_id].free_acc_L_upper_reach.append(time_series[reach_begin:reach_end])
                sub_data[sub_id].free_acc_L_upper_retract.append(time_series[retract_begin:retract_end])
            elif idx == 2:
                sub_data[sub_id].free_acc_trunk_reach.append(time_series[reach_begin:reach_end])
                sub_data[sub_id].free_acc_trunk_retract.append(time_series[retract_begin:retract_end])
            elif idx == 3:
                sub_data[sub_id].free_acc_R_upper_reach.append(time_series[reach_begin:reach_end])
                sub_data[sub_id].free_acc_R_upper_retract.append(time_series[retract_begin:retract_end])
            elif idx == 4:
                sub_data[sub_id].free_acc_R_fore_reach.append(time_series[reach_begin:reach_end])
                sub_data[sub_id].free_acc_R_fore_retract.append(time_series[retract_begin:retract_end])

        if sub_data[sub_id].affect_side == 0: # left
            sub_data[sub_id].free_acc_affect_fore_reach = deepcopy(sub_data[sub_id].free_acc_L_fore_reach)
            sub_data[sub_id].free_acc_affect_fore_retract = deepcopy(sub_data[sub_id].free_acc_L_fore_retract)
            sub_data[sub_id].free_acc_affect_upper_reach = deepcopy(sub_data[sub_id].free_acc_L_upper_reach)
            sub_data[sub_id].free_acc_affect_upper_retract = deepcopy(sub_data[sub_id].free_acc_L_upper_retract)
        elif sub_data[sub_id].affect_side == 1: # right
            sub_data[sub_id].free_acc_affect_fore_reach = deepcopy(sub_data[sub_id].free_acc_R_fore_reach)
            sub_data[sub_id].free_acc_affect_fore_retract = deepcopy(sub_data[sub_id].free_acc_R_fore_retract)
            sub_data[sub_id].free_acc_affect_upper_reach = deepcopy(sub_data[sub_id].free_acc_R_upper_reach)
            sub_data[sub_id].free_acc_affect_upper_retract = deepcopy(sub_data[sub_id].free_acc_R_upper_retract)

save_data_file = open('sub_data.p', 'wb')
pickle.dump(sub_data, save_data_file)