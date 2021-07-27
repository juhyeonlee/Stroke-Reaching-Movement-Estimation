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
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 3))
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 15))
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 6))
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 5))

print(subject_id)
# subject_id = np.array([1, 5, 8, 9, 10, 11, 12, 17])

save_data_file = open('sub_data_filt_ok.p', 'rb')
sub_data = pickle.load(save_data_file)

results = np.load('gp_5class_regress_test.npy')
results = np.round(results).astype(int)
# predicted_testscore = np.load('gaussian_2class_1dot5_score.npy')
# results = (predicted_testscore > 0.5631760683364604).astype(float)
# results_reach = np.load('reach_relieff_2class.npy')
# results_retract = np.load('retract_relieff_2class.npy')
# results = []

compensation_labels = []
compensation_origin_labels = []
sub_id_labels = []
reach_retract_labels = [] # reach 1, retract 0

target_dists = []
target_angles = []
mag = []
reach_idx = 0
retract_idx = 0
for sub_id in subject_id:
    velfilt_affect_fore_reach = sub_data[sub_id].velfilt_affect_fore_reach
    velfilt_affect_fore_retract = sub_data[sub_id].velfilt_affect_fore_retract
    # results.extend(results_reach[reach_idx: reach_idx+len(vel_affect_fore_reach)])
    # results.extend(results_retract[retract_idx: reach_idx + len(vel_affect_fore_retract)])
    reach_idx += len(velfilt_affect_fore_reach)
    retract_idx += len(velfilt_affect_fore_retract)

    for ii in range(len(velfilt_affect_fore_reach)):
        mag.append(magnitude_xy(velfilt_affect_fore_reach[ii]))

    for ll in range(len(sub_data[sub_id].reach_fas_score)):
        compensation_origin_labels.append(sub_data[sub_id].reach_comp_score[ll])
        target_dists.append(sub_data[sub_id].target_dist[ll])
        target_angles.append(sub_data[sub_id].target_angle[ll])
        # if sub_data[sub_id].reach_fas_score[ll] == 5: #or sub_data[sub_id].reach_fas_score[ll] == 4:
        #     compensation_labels.append(1)
        # else:  # elif sub_data[sub_id].reach_fas_score[ll] != 5: #or sub_data[sub_id].reach_comp_score[ll] == 2:
        #     compensation_labels.append(0)
        compensation_labels.append(sub_data[sub_id].reach_fas_score[ll])

    # compensation_labels.extend(sub_data[sub_id].reach_comp_score)
    sub_id_labels.extend([sub_id] * len(sub_data[sub_id].reach_fas_score))
    reach_retract_labels.extend([1] * len(sub_data[sub_id].reach_fas_score))

    # for ii in range(len(velfilt_affect_fore_retract)):
    #     mag.append(magnitude_xy(velfilt_affect_fore_retract[ii]))
    #
    # for ll in range(len(sub_data[sub_id].retract_comp_score)):
    #     compensation_origin_labels.append(sub_data[sub_id].retract_comp_score[ll])
    #     if sub_data[sub_id].retract_comp_score[ll] == 3: # or sub_data[sub_id].retract_comp_score[ll] == 2:
    #         compensation_labels.append(1)
    #     elif sub_data[sub_id].retract_comp_score[ll] != 3:
    #         compensation_labels.append(0)
    #
    # # compensation_labels.extend(sub_data[sub_id].retract_comp_score)
    # sub_id_labels.extend([sub_id] * len(sub_data[sub_id].retract_comp_score))
    # reach_retract_labels.extend([0] * len(sub_data[sub_id].retract_comp_score))



compensation_labels = np.asarray(compensation_labels).reshape(-1)
target_angles = np.asarray(target_angles)
target_dists = np.asarray(target_dists)

acc = accuracy_score(compensation_labels, results)
fig_conf = plt.figure(figsize=[6, 6])
ax = fig_conf.add_subplot(111)
plot_confusion_matrix(ax, compensation_labels, results, ['Abnormal', 'Normal'], normalize=True)
fig_conf.tight_layout()

print(acc)
print(f1_score(compensation_labels, results, average='macro'))

sns.set()
sns.set_context('talk')
color_palette = sns.color_palette('deep')

error_dist = {'01':[], '12': [], '20': [], '02':[], '10': [], '21': []}
label_dist = {'0':[], '1': [], '2': []}
mag_dist_right = {'0':pd.DataFrame(columns=['t', 'vel']), '1': pd.DataFrame(columns=['t', 'vel']), '2': pd.DataFrame(columns=['t', 'vel'])}
mag_dist_wrong = {'0':pd.DataFrame(columns=['t', 'vel']), '1': pd.DataFrame(columns=['t', 'vel']), '2': pd.DataFrame(columns=['t', 'vel'])}

ii_1 = 0
ii_2 = 0
reach_cnt = 0
retract_cnt = 0

for i in range(len(compensation_labels)):
    label_dist[str(compensation_labels[i])].append(sub_id_labels[i])
    # if sub_id_labels[i] == 16:
    if compensation_labels[i] != results[i]:
        error_dist[str(compensation_labels[i]) + str(int(results[i]))].append(sub_id_labels[i])
        mag_dist_wrong[str(compensation_labels[i])] = mag_dist_wrong[str(compensation_labels[i])].append(
                    pd.DataFrame({"t": np.arange(len(mag[i])), "vel": mag[i], "color": color[int(results[i])], "index": ii_1}))
        # print(sub_id_labels[i], compensation_labels[i], results[i], compensation_origin_labels[i])
        ii_1 += 1
        print(target_dists[i], target_angles[i])
        if reach_retract_labels[i] == 1:
            reach_cnt += 1
        else:
            retract_cnt += 1
    else:
        mag_dist_right[str(compensation_labels[i])] = mag_dist_right[str(compensation_labels[i])].append(
                    pd.DataFrame({"t": np.arange(len(mag[i])), "vel": mag[i], "color": color[int(results[i])], "index": ii_2}))
        ii_2 += 1

print('reach: ', reach_cnt, ' retract: ', retract_cnt)

# for i in range(len(compensation_labels)):
#     if sub_id_labels[i] == 5:
#         if compensation_labels[i] != results[i]:
#             plt.figure()
#             plt.title(compensation_labels[i])
#             plt.plot(mag[i])
#             plt.show()
for sub in subject_id:
    right_total = np.sum(np.logical_and(compensation_labels == results, sub_id_labels == sub).astype(int))
    total = np.sum((sub_id_labels == sub).astype(int))
    print(sub, '{:.2f} %'.format(right_total/total*100))


print(len(compensation_labels), len(compensation_labels) * (1-acc))
fig_r = plt.figure(figsize=[25, 6])
ax1 = fig_r.add_subplot(1, 3, 1)
ax1.hist(label_dist['0'], bins=np.linspace(1, 21, 21), color='grey')
ax1.hist(error_dist['01'], bins=np.linspace(1, 21, 21), color='blue', label='abnormal -> normal')
# ax1.hist(error_dist['02'], bins=np.linspace(1, 21, 21), color='red', label='comp -> nomal')
ax1.set_title('Comp level 0')
ax1.legend()
ax1.set_xticks(np.arange(1, 21, 2))

ax2 = fig_r.add_subplot(1, 3, 2)
ax2.hist(label_dist['1'], bins=np.linspace(1, 21, 21), color='grey')
ax2.hist(error_dist['10'], bins=np.linspace(1, 21, 21), color='red', label='normal -> abnormal')
# ax2.hist(error_dist['12'], bins=np.linspace(1, 21, 21), color='red', label='inacc -> normal')
ax2.set_title('Comp level 1')
ax2.legend()
ax2.set_xticks(np.arange(1, 21, 2))

ax3 = fig_r.add_subplot(1, 3, 3)
ax3.hist(label_dist['2'], bins=np.linspace(1, 21, 21), color='grey')
ax3.hist(error_dist['20'], bins=np.linspace(1, 21, 21), color='blue', label='normal -> comp')
ax3.hist(error_dist['21'], bins=np.linspace(1, 21, 21), color='red', label='normal -> inacc')
ax3.set_title('Comp level 2')
ax3.legend()
ax3.set_xticks(np.arange(1, 21, 2))

fig_vel = plt.figure(figsize=[20, 15])
vx1 = fig_vel.add_subplot(2, 3, 1)
sns.lineplot(x="t", y="vel", data=mag_dist_right['0'], estimator=None, units='index', hue='index', legend=False,
                              ax=vx1, palette="ch:2.5,.25")
vx1.set_title('Correct: Abnomal', {'fontsize': 15})
vx2 = fig_vel.add_subplot(2, 3, 2)
sns.lineplot(x="t", y="vel", data=mag_dist_right['1'], estimator=None, units='index', hue='index', legend=False,
                              ax=vx2, palette="ch:2.5,.25")
vx2.set_title('Correct: Normal', {'fontsize': 15})
# vx3 = fig_vel.add_subplot(2, 3, 3)
# sns.lineplot(x="t", y="vel", data=mag_dist_right['2'], estimator=None, units='index', hue='index', legend=False,
#                               ax=vx3, palette="ch:2.5,.25")
# vx3.set_title('True Comp level 2: Normal')

vx4 = fig_vel.add_subplot(2, 3, 4)
sns.lineplot(x="t", y="vel", data=mag_dist_wrong['0'], estimator=None, units='index', hue='index', legend=False, #'brief',
                              ax=vx4)
vx4.set_title('Misclassified: Abnormal --> Normal ', {'fontsize': 15})
vx5 = fig_vel.add_subplot(2, 3, 5)
sns.lineplot(x="t", y="vel", data=mag_dist_wrong['1'], estimator=None, units='index', hue='index', legend=False, #'brief',
                              ax=vx5)
vx5.set_title('Misclassified: Normal --> Abmormal', {'fontsize': 15})
# vx6 = fig_vel.add_subplot(2, 3, 6)
# sns.lineplot(x="t", y="vel", data=mag_dist_wrong['2'], estimator=None, units='index', hue='index', legend=False, #'brief',
#                               ax=vx6)
# vx6.set_title('False Comp level 2: Normal')
# fig_vel.savefig('dfdf.png')
plt.tight_layout()
# fig_vel.savefig('error_samples')
plt.show()
fpr, tpr, thresholds = roc_curve(compensation_labels, results)
print(auc(fpr, tpr))
sns.set()
sns.set_context('talk')
color_palette = sns.color_palette('deep')
plt.figure(figsize=[8, 8])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

