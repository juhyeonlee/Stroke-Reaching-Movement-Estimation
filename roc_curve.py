import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, classification_report
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
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 12))



print(subject_id)
# subject_id = np.array([1, 5, 8, 9, 10, 11, 12, 17])

save_data_file = open('sub_data_filt_ok.p', 'rb')
sub_data = pickle.load(save_data_file)

predicted = np.load('gaussian_2class_1dot5.npy')
predicted_testscore = np.load('gaussian_2class_1dot5_score.npy')

compensation_labels = []
for sub_id in subject_id:
    for ll in range(len(sub_data[sub_id].reach_comp_score)):
        if sub_data[sub_id].reach_comp_score[ll] == 3:
            compensation_labels.append(1)
        elif sub_data[sub_id].reach_comp_score[ll] != 3:
            compensation_labels.append(0)

    # for ll in range(len(sub_data[sub_id].retract_comp_score)):
    #     if sub_data[sub_id].retract_comp_score[ll] == 3:
    #         compensation_labels.append(1)
    #     elif sub_data[sub_id].retract_comp_score[ll] != 3:
    #         compensation_labels.append(0)

compensation_labels = np.asarray(compensation_labels).reshape(-1)

fpr, tpr, thresholds = roc_curve(compensation_labels, predicted_testscore)

best_index = 79
print(thresholds[best_index])
predicted_adjust = (predicted_testscore > thresholds[best_index]).astype(float)
pred_set = []
for th in thresholds:
    pred_th = (predicted_testscore > th).astype(float)
    tp = np.sum(np.logical_and(compensation_labels == 1, pred_th == 1).astype(int))
    tn = np.sum(np.logical_and(compensation_labels == 0, pred_th == 0).astype(int))
    pred_set.append(abs(tp/472 - tn/384))
    # pred_set.append(f1_score(compensation_labels, pred_th, average='macro'))
pred_set = np.asarray(pred_set)
print(pred_set.argmin(), thresholds[pred_set.argmin()])



fig_conf = plt.figure(figsize=[6, 6])
ax1 = fig_conf.add_subplot(1, 1, 1)
plot_confusion_matrix(ax1, compensation_labels, predicted_adjust, ['Abnormal', 'Normal'], normalize=True)

# print('#### original ####')
# print('Accuracy: {:.1f}%'.format(accuracy_score(compensation_labels, predicted) * 100))
# print(classification_report(compensation_labels, predicted))
# print('unweighted per class F1 Score: {:.3f}'.format(f1_score(compensation_labels, predicted, average='macro')))

print('#### adjusted ####')
print('Accuracy: {:.1f}%'.format(accuracy_score(compensation_labels, predicted_adjust) * 100))
print(classification_report(compensation_labels, predicted_adjust))
print('unweighted per class F1 Score: {:.3f}'.format(f1_score(compensation_labels, predicted_adjust, average='macro')))


roc_auc = auc(fpr, tpr)
roc_fig = plt.figure()
roc_ax = roc_fig.add_subplot(1, 1, 1)
roc_ax.plot(fpr, tpr,  color='b', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
roc_ax.scatter(fpr[best_index], tpr[best_index], facecolor='r', label='operating point')
print(1 - fpr[best_index], tpr[best_index])
plt.legend()
roc_fig.savefig('roc_selectkbest_2class_nofiltadd')
thrs_fig = plt.figure()
thrs_ax = thrs_fig.add_subplot(1, 1, 1)
thrs_ax.plot(fpr, thresholds, markeredgecolor='b',linestyle='dashed', color='b')
plt.show()