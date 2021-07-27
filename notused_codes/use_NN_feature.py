import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from scipy.signal import resample
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
from ReliefF import ReliefF
import seaborn as sns

from extract_features import extract_features
from data_struct import CompData
from utils import ExhaustiveForwardSelect, efs_score, plot_confusion_matrix, magnitude_xy, CFS
import time
from nn import Net, train_NN, test_NN
import torch
from nn import autoencoder

num_subjects = 20
exclude_sub_num = 13
direction = ["L", "R"]
n_classes = 2

# list subject ID
subject_id = np.arange(0, num_subjects) + 1
subject_id = np.delete(subject_id, np.argwhere(subject_id == exclude_sub_num))
# with mft
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 3))
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 15))
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 6))
# # exclude 5?
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 5))
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 12))


# subject_id = np.array([1, 5, 8, 9, 10, 11, 12, 17])

save_data_file = open('sub_data_filt_ok.p', 'rb')
sub_data = pickle.load(save_data_file)

fs = 100
dx = 1 / fs
resample_num = 300

torch.manual_seed(0)
np.random.seed(0)


# feature extraction
mvel_xy_reach_list = []
mvel_z_reach_list = []
length_list = []
compensation_labels = []
all_sub = []
reach_retract_mask = []
for sub_id in subject_id:
    velfilt_affect_fore_reach = sub_data[sub_id].velfilt_affect_fore_reach
    velfilt_affect_fore_retract = sub_data[sub_id].velfilt_affect_fore_retract
    vel_affect_fore_reach = sub_data[sub_id].vel_affect_fore_reach
    vel_affect_fore_retract = sub_data[sub_id].vel_affect_fore_retract
    free_acc_affect_fore_reach = sub_data[sub_id].free_acc_affect_fore_reach
    free_acc_affect_fore_retract = sub_data[sub_id].free_acc_affect_fore_retract
    for ii in range(len(velfilt_affect_fore_reach)):
        mvel_xy_reach = magnitude_xy(free_acc_affect_fore_reach[ii])
        mvel_z_reach = np.sqrt(np.square(free_acc_affect_fore_reach[ii][:, 2]))

        mvel_xy_reach_list.append(mvel_xy_reach)
        # mvel_z_reach_list.append(mvel_z_reach)# / mvel_z_reach.mean())
        length_list.append(len(mvel_xy_reach))
        all_sub.append(sub_id)
        reach_retract_mask.append(True)

    for ll in range(len(sub_data[sub_id].reach_fas_score)):
        if sub_data[sub_id].reach_fas_score[ll] == 5: #or sub_data[sub_id].reach_fas_score[ll] == 4:
            sub_data[sub_id].reach_fas_score[ll] = 1
        else: # elif sub_data[sub_id].reach_fas_score[ll] != 5: #or sub_data[sub_id].reach_comp_score[ll] == 2:
            sub_data[sub_id].reach_fas_score[ll] = 0

    compensation_labels.extend(sub_data[sub_id].reach_fas_score)




compensation_labels = np.asarray(compensation_labels).reshape(-1)
all_sub = np.asarray(all_sub)

max_length = np.max(length_list)
mvel_xy_padded = np.zeros((len(mvel_xy_reach_list), max_length))
for i, mvel_xy in enumerate(mvel_xy_reach_list):
    mvel_xy_padded[i, :len(mvel_xy)] = mvel_xy
    # mvel_xy_padded[i, max_length:max_length+ len(mvel_xy)] = mvel_z_reach_list[i]
print(mvel_xy_padded.shape)

lr = 1e-3
batch_size = 32
epoch = 100


classifiy_loss_fn = torch.nn.CrossEntropyLoss()
recon_loss_fn = torch.nn.MSELoss()


predicted = np.zeros(compensation_labels.shape)
predicted_testscore = np.zeros(compensation_labels.shape)

for s in subject_id:
    start_time = time.time()

    train_feats = mvel_xy_padded[all_sub != s, :]
    train_labels = compensation_labels[all_sub != s]
    test_labels = compensation_labels[all_sub == s]
    test_feats = mvel_xy_padded[all_sub == s, :]
    train_sub = all_sub[all_sub != s]

    model = autoencoder(max_length)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for t in range(epoch):
        # if epoch == 50:
        #     lr = lr/10
        #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_epoch = []
        rand_order = np.random.permutation(len(train_feats))
        train_feats = train_feats[rand_order, :]
        train_labels = train_labels[rand_order]
        for beg_i in range(0, train_feats.shape[0], batch_size):
            X_batch = train_feats[beg_i:beg_i + batch_size, :]
            y_batch = train_labels[beg_i:beg_i + batch_size]
            X_batch = torch.from_numpy(X_batch).type(torch.FloatTensor)
            y_batch = torch.from_numpy(y_batch)

            model.zero_grad()
            de_out, classify_out = model(X_batch)
            loss = classifiy_loss_fn(classify_out, y_batch) + recon_loss_fn(de_out, X_batch)
            loss_epoch.append(loss.item())

            loss.backward()
            optimizer.step()
        if t % 10 == 0:
            print(np.mean(loss_epoch))


    model.eval()
    _, y_pred_train_score = model(torch.from_numpy(train_feats).type(torch.FloatTensor))
    _, y_pred_score = model(torch.from_numpy(test_feats).type(torch.FloatTensor))
    _, y_pre_train =  torch.max(y_pred_train_score, 1)
    _, y_pred = torch.max(y_pred_score, 1)
    predicted[all_sub == s] = y_pred.detach().numpy()
    predicted_testscore[all_sub == s] = y_pred_score.detach().numpy()[:, 1]
    print(s, 'train acc: ', accuracy_score(train_labels, y_pre_train))
    print(s, 'acc: ', accuracy_score(test_labels, predicted[all_sub == s]))
    print('elapsed time:', time.time() - start_time)

print(compensation_labels)
print(predicted)
print(predicted_testscore)

print('Accuracy: {:.3f}%'.format(accuracy_score(compensation_labels, predicted) * 100))
print(classification_report(compensation_labels, predicted))
print('weighted per class F1 Score: {:.3f}'.format(f1_score(compensation_labels, predicted, average='weighted')))
print('unweighted per class F1 Score: {:.3f}'.format(f1_score(compensation_labels, predicted, average='macro')))
print('global F1 Score: {:.3f}'.format(f1_score(compensation_labels, predicted, average='micro')))

for i in range(6):
    print(i, accuracy_score(compensation_labels[compensation_labels== i], predicted[compensation_labels == i]) * 100)

fig_conf = plt.figure(figsize=[6, 6])
ax1 = fig_conf.add_subplot(1, 1, 1)
plot_confusion_matrix(ax1, compensation_labels, predicted, ['Abnormal', 'Normal'], normalize=True)
fig_conf.tight_layout()
fig_conf.savefig('confusion_matrix_pure_NN')
np.save('pure_NN_2class_with_recon', predicted)
np.save('pure_NN_2class_score_with_recon', predicted_testscore)




# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(compensation_labels, predicted_testscore)
roc_auc = auc(fpr, tpr)
roc_fig = plt.figure()
roc_ax = roc_fig.add_subplot(1, 1, 1)
roc_ax.plot(fpr, tpr, color='red', lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
roc_ax.plot(fpr, thresholds, markeredgecolor='b',linestyle='dashed', color='b')
plt.legend()
plt.ylim(0, 1)
roc_fig.savefig('roc_rf_2class_nofiltadd_test2')
plt.show()