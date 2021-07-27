import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from scipy.signal import resample
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.decomposition import PCA
from ReliefF import ReliefF
import seaborn as sns

from extract_features import extract_features
from data_struct import CompData
from utils import ExhaustiveForwardSelect, efs_score, plot_confusion_matrix, magnitude_xy, CFS, regression_performance
import time


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

# feature extraction
features = []
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
        mvel_xy_reach = magnitude_xy(velfilt_affect_fore_reach[ii])
        mvelnofilt_xy_reach = magnitude_xy(vel_affect_fore_reach[ii])
        macc_xy_reach = magnitude_xy(free_acc_affect_fore_reach[ii])
        mvel_z_reach = np.sqrt(np.square(velfilt_affect_fore_reach[ii][:, 2]))
        # residual_reach = abs(mvelnofilt_xy_reach - mvel_xy_reach)
        mvel_xy_reach_norm = resample(mvel_xy_reach / np.mean(mvel_xy_reach), resample_num, np.arange(len(mvel_xy_reach)))[0]

        feat, feature_names = extract_features(mvel_xy_reach, sub_data[sub_id].mft_score, fs=100, prefix='velfilt_')
        feat_z, feature_names_z = extract_features(mvel_z_reach, None, z=True, fs=100, prefix='velfilt_z_')
        # feat2, feature_names2 = extract_features(mvelnofilt_xy_reach, None, fs=100, prefix='vel_')
        feat_acc, feature_names_acc = extract_features(macc_xy_reach, None, z=True, fs=100, prefix='acc_')

        # feat3, feature_names3 = extract_features(residual_reach, None, fs=100, prefix='residual_')

        # feat.append(1.0)
        # feature_names.append('reachretract')
        #
        # feat.append(np.max(velfilt_affect_fore_reach[ii][:, 2]))
        # feature_names.append('zheight')
        #
        # feat.append(np.argmax(velfilt_affect_fore_reach[ii][:, 2]))
        # feature_names.append('zmaxdur')

        features.append(feat + feat_z + feat_acc)
        all_sub.append(sub_id)
        reach_retract_mask.append(True)

    # for ll in range(len(sub_data[sub_id].reach_fas_score)):
    #     if sub_data[sub_id].reach_fas_score[ll] == 5: #or sub_data[sub_id].reach_fas_score[ll] == 4:
    #         sub_data[sub_id].reach_fas_score[ll] = 1
    #     else: # elif sub_data[sub_id].reach_fas_score[ll] != 5: #or sub_data[sub_id].reach_comp_score[ll] == 2:
    #         sub_data[sub_id].reach_fas_score[ll] = 0

    compensation_labels.extend(sub_data[sub_id].reach_fas_score)

    # for ii in range(len(velfilt_affect_fore_retract)):
    #     mvel_xy_retract = magnitude_xy(velfilt_affect_fore_retract[ii])
    #     mvelnofilt_xy_retract = magnitude_xy(vel_affect_fore_retract[ii])
    #     macc_xy_retract = magnitude_xy(np.gradient(free_acc_affect_fore_retract[ii], dx, axis=0))
    #     mvel_z_retract = np.sqrt(np.square(velfilt_affect_fore_retract[ii][:, 2]))
    #
    #     # residual_retract = abs(mvelnofilt_xy_retract - mvel_xy_retract)
    #     mvel_xy_retract_norm = resample(mvel_xy_retract / np.mean(mvel_xy_retract), resample_num, np.arange(len(mvel_xy_retract)))[0]
    #
    #     feat, feature_names = extract_features(mvel_xy_retract, sub_data[sub_id].mft_score, fs=100, prefix='velfilt_')
    #     feat_z, feature_names_z = extract_features(mvel_z_retract, None, z=True, fs=100, prefix='velfilt_z_')
    #
    #     # feat2, feature_names2 = extract_features(mvelnofilt_xy_retract, None, fs=100, prefix='vel_')
    #     # feat3, feature_names3 = extract_features(macc_xy_retract, None, fs=100, prefix='acc_')
    #
    #     # feat3, feature_names3 = extract_features(residual_retract, None, fs=100, prefix='residual_')
    #
    #     # feat.append(0.)
    #     # feature_names.append('reachretract')
    #     #
    #     # feat.append(np.max(velfilt_affect_fore_retract[ii][:, 2]))
    #     # feature_names.append('zheight')
    #     #
    #     # feat.append(np.argmax(velfilt_affect_fore_retract[ii][:, 2]))
    #     # feature_names.append('zmaxdur')
    #
    #     features.append(feat + feat_z)
    #     all_sub.append(sub_id)
    #     reach_retract_mask.append(False)
    #
    # for ll in range(len(sub_data[sub_id].retract_comp_score)):
    #     if sub_data[sub_id].retract_comp_score[ll] == 3: #  or sub_data[sub_id].retract_comp_score[ll] == 2:
    #         sub_data[sub_id].retract_comp_score[ll] = 1
    #     elif sub_data[sub_id].retract_comp_score[ll] != 3: #or sub_data[sub_id].retract_comp_score[ll] == 2:
    #         sub_data[sub_id].retract_comp_score[ll] = 0
    #
    # compensation_labels.extend(sub_data[sub_id].retract_comp_score)

    # if sub_id == 12:
    #     kkk = np.asarray(features)
    #     dfd = np.asarray(compensation_labels).reshape(-1)
    #     print(np.mean(kkk[np.where(dfd == 0)[0]], axis=0))
    #     print(np.mean(kkk[np.where(dfd == 1)[0]], axis=0))

compensation_labels = np.asarray(compensation_labels).reshape(-1)
features = np.asarray(features)
print(features.shape)
feature_names = np.asarray(feature_names + feature_names_z + feature_names_acc)

print('the number of features', len(feature_names))
print('comp0', len(np.where(compensation_labels == 0)[0]), 'comp1', len(np.where(compensation_labels == 1)[0]),
      'comp2', len(np.where(compensation_labels == 2)[0]), 'comp3', len(np.where(compensation_labels == 3)[0]),
      'comp4', len(np.where(compensation_labels == 4)[0]), 'comp5', len(np.where(compensation_labels == 5)[0]),
      "total", len(compensation_labels))

# for s in subject_id:
#     all_sub.extend([s] * len(sub_data[s].velfilt_affect_fore_reach))
#     all_sub.extend([s] * len(sub_data[s].velfilt_affect_fore_retract))


all_sub = np.asarray(all_sub)
reach_retract_mask = np.asarray(reach_retract_mask)
print(len(all_sub), len(features))
predicted = np.zeros(compensation_labels.shape)
predicted_testscore = np.zeros(compensation_labels.shape)

r_grid = {'ne': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
          'maxdepth': [None]}
print(r_grid)
num_comb = len(r_grid['ne'])
print('total combination grid search', num_comb)

pca_add_features = np.zeros((len(features), len(feature_names)+2))
for s in subject_id:
    train_feats = features[all_sub != s, :]
    test_feats = features[all_sub == s, :]

    pca = PCA(n_components=2).fit(train_feats)
    pca_feats = pca.transform(test_feats)

    pca_add_features[all_sub == s, :] = np.concatenate((test_feats, pca_feats), axis=1)
feature_names = np.append(feature_names, 'pca0')
feature_names = np.append(feature_names, 'pca1')
print(feature_names.shape, pca_add_features.shape)

for s in subject_id:
    start_time = time.time()

    train_feats = pca_add_features[all_sub != s, :]
    train_labels = compensation_labels[all_sub != s]
    test_labels = compensation_labels[all_sub == s]
    test_feats = pca_add_features[all_sub == s, :]
    train_sub = all_sub[all_sub != s]

    scaler = MinMaxScaler().fit(train_feats)
    train_feats = scaler.transform(train_feats)
    test_feats = scaler.transform(test_feats)

    inner_subject_id = subject_id.copy()
    inner_subject_id = np.delete(inner_subject_id, np.argwhere(inner_subject_id == s))

    # val_scores = np.zeros(num_comb)
    # j = 0
    # for n_e in r_grid['ne']:
    #     predicted_val = np.zeros(len(train_labels))
    #     for i_s in inner_subject_id:
    #         inner_train_feats = features[np.logical_and(all_sub != s, all_sub != i_s), :]
    #         # inner_train_feats = train_feats[train_sub != i_s, :]
    #         inner_train_labels = compensation_labels[np.logical_and(all_sub != s, all_sub != i_s)]
    #         val_feats = features[all_sub == i_s, :]
    #         # val_feats = train_feats[train_sub == i_s, :]
    #         val_labels = compensation_labels[all_sub == i_s]
    #
    #         scaler_inner = RobustScaler().fit(inner_train_feats)
    #         inner_train_feats = scaler_inner.transform(inner_train_feats)
    #         val_feats = scaler_inner.transform(val_feats)
    #
    #         val_model = RandomForestClassifier(n_estimators=n_e, random_state=0)
    #         val_model.fit(inner_train_feats, inner_train_labels)
    #         predicted_val[train_sub == i_s] = val_model.predict(val_feats)
    #     val_scores[j] = accuracy_score(train_labels, predicted_val)
    #     j += 1
    # best_ne = r_grid['ne'][int(np.argmax(val_scores))]
    # print(np.max(val_scores), best_ne)

    # kernel = 1.0 * RBF(1.0) + WhiteKernel()
    # model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, alpha=0)
    model = RandomForestRegressor(n_estimators=400, random_state=0)
    # model = SVR(C= 10, gamma='auto', kernel='rbf')

    model.fit(train_feats, train_labels)
    predicted[all_sub == s] = model.predict(test_feats)
    # predicted_testscore[all_sub == s] = model.predict_proba(test_feats)[:, 1]
    # print(s, 'acc: ', accuracy_score(test_labels, predicted[all_sub == s]))
    print('elapsed time:', time.time() - start_time)



print(compensation_labels)
print(predicted)
fig_reg = plt.figure(figsize=[6, 6])
ax1 = fig_reg.add_subplot(1, 1, 1)
r2, rmse, nrmse = regression_performance(compensation_labels, predicted, ax=ax1)
print('r2:', r2, 'rmse', rmse, 'nrmse', nrmse)
np.save('gp_5class_regress_test', predicted)
plt.show()
