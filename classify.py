import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from scipy.signal import resample
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, chi2, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, classification_report, davies_bouldin_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from ReliefF import ReliefF
# from skfeature.function.statistical_based import CFS
from skfeature.function.information_theoretical_based import FCBF
import seaborn as sns

from extract_features import extract_features
from data_struct import CompData
from utils import ExhaustiveForwardSelect, efs_score, plot_confusion_matrix, magnitude_xy, CFS
import time


num_subjects = 20
exclude_sub_num = 13
direction = ["L", "R"]
n_classes = 2

# list subject ID
subject_id = np.arange(0, num_subjects) + 1
subject_id = np.delete(subject_id, np.argwhere(subject_id == exclude_sub_num))
# with mft
subject_id = np.delete(subject_id, np.argwhere(subject_id == 3))
subject_id = np.delete(subject_id, np.argwhere(subject_id == 15))
subject_id = np.delete(subject_id, np.argwhere(subject_id == 6))
# # exclude 5?
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 5))

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
        macc_xy_reach = magnitude_xy(np.gradient(free_acc_affect_fore_reach[ii], dx, axis=0))
        # residual_reach = abs(mvelnofilt_xy_reach - mvel_xy_reach)
        mvel_xy_reach_norm = resample(mvel_xy_reach / np.mean(mvel_xy_reach), resample_num, np.arange(len(mvel_xy_reach)))[0]

        feat, feature_names = extract_features(mvel_xy_reach, sub_data[sub_id].mft_score, fs=100, prefix='velfilt_')
        feat2, feature_names2 = extract_features(mvelnofilt_xy_reach, None, fs=100, prefix='vel_')
        # feat3, feature_names3 = extract_features(macc_xy_reach, None, fs=100, prefix='acc_')

        # feat3, feature_names3 = extract_features(residual_reach, None, fs=100, prefix='residual_')

        # feat.append(1.0)
        # feature_names.append('reachretract')
        #
        # feat.append(np.max(velfilt_affect_fore_reach[ii][:, 2]))
        # feature_names.append('zheight')
        #
        # feat.append(np.argmax(velfilt_affect_fore_reach[ii][:, 2]))
        # feature_names.append('zmaxdur')

        features.append(feat + feat2)
        all_sub.append(sub_id)
        reach_retract_mask.append(True)

    for ll in range(len(sub_data[sub_id].reach_comp_score)):
        if sub_data[sub_id].reach_comp_score[ll] == 3:
            sub_data[sub_id].reach_comp_score[ll] = 1
        elif sub_data[sub_id].reach_comp_score[ll] != 3: #or sub_data[sub_id].reach_comp_score[ll] == 2:
            sub_data[sub_id].reach_comp_score[ll] = 0

    compensation_labels.extend(sub_data[sub_id].reach_comp_score)

    for ii in range(len(velfilt_affect_fore_retract)):
        mvel_xy_retract = magnitude_xy(velfilt_affect_fore_retract[ii])
        mvelnofilt_xy_retract = magnitude_xy(vel_affect_fore_retract[ii])
        macc_xy_retract = magnitude_xy(np.gradient(free_acc_affect_fore_retract[ii], dx, axis=0))
        # residual_retract = abs(mvelnofilt_xy_retract - mvel_xy_retract)
        mvel_xy_retract_norm = resample(mvel_xy_retract / np.mean(mvel_xy_retract), resample_num, np.arange(len(mvel_xy_retract)))[0]

        feat, feature_names = extract_features(mvel_xy_retract, sub_data[sub_id].mft_score, fs=100, prefix='velfilt_')
        feat2, feature_names2 = extract_features(mvelnofilt_xy_retract, None, fs=100, prefix='vel_')
        # feat3, feature_names3 = extract_features(macc_xy_retract, None, fs=100, prefix='acc_')

        # feat3, feature_names3 = extract_features(residual_retract, None, fs=100, prefix='residual_')

        # feat.append(0.)
        # feature_names.append('reachretract')
        #
        # feat.append(np.max(velfilt_affect_fore_retract[ii][:, 2]))
        # feature_names.append('zheight')
        #
        # feat.append(np.argmax(velfilt_affect_fore_retract[ii][:, 2]))
        # feature_names.append('zmaxdur')

        features.append(feat + feat2)
        all_sub.append(sub_id)
        reach_retract_mask.append(False)

    for ll in range(len(sub_data[sub_id].retract_comp_score)):
        if sub_data[sub_id].retract_comp_score[ll] == 3: #  or sub_data[sub_id].retract_comp_score[ll] == 2:
            sub_data[sub_id].retract_comp_score[ll] = 1
        elif sub_data[sub_id].retract_comp_score[ll] != 3: #or sub_data[sub_id].retract_comp_score[ll] == 2:
            sub_data[sub_id].retract_comp_score[ll] = 0

    compensation_labels.extend(sub_data[sub_id].retract_comp_score)

    # if sub_id == 12:
    #     kkk = np.asarray(features)
    #     dfd = np.asarray(compensation_labels).reshape(-1)
    #     print(np.mean(kkk[np.where(dfd == 0)[0]], axis=0))
    #     print(np.mean(kkk[np.where(dfd == 1)[0]], axis=0))

compensation_labels = np.asarray(compensation_labels).reshape(-1)
features = np.asarray(features)
print(features.shape)
feature_names = np.asarray(feature_names + feature_names2)

print('the number of features', len(feature_names))
print('comp 0', len(np.where(compensation_labels == 0)[0]), 'comp1', len(np.where(compensation_labels == 1)[0]), 'comp2',
      len(np.where(compensation_labels == 2)[0]), 'total', len(compensation_labels))

# for s in subject_id:
#     all_sub.extend([s] * len(sub_data[s].velfilt_affect_fore_reach))
#     all_sub.extend([s] * len(sub_data[s].velfilt_affect_fore_retract))


all_sub = np.asarray(all_sub)
reach_retract_mask = np.asarray(reach_retract_mask)
print(len(all_sub), len(features))
predicted = np.zeros(compensation_labels.shape)
predicted_testscore = np.zeros(compensation_labels.shape)

# p_grid ={'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
p_grid ={'C': [x for x in range(5, 21)], 'gamma': [1]} # better -> take long time
# p_grid ={'C': [5, 10, 15, 20], 'gamma': [0.001]}
r_grid = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
selnum_f_grid = [5, 6, 7] #, 8, 9, 10] #, 15, 20] #[int(x) for x in np.linspace(4, len(feature_names), num=10)]#
print(p_grid, selnum_f_grid)
num_comb = len(p_grid['C']) * len(p_grid['gamma']) * len(selnum_f_grid)
print('total combination grid search', num_comb)
# selnum_f = 17

fpr_set = []
tpr_set = []
for s in subject_id:
    start_time = time.time()

    train_feats = features[all_sub != s, :]
    train_labels = compensation_labels[all_sub != s]
    test_feats = features[all_sub == s, :]
    train_sub = all_sub[all_sub != s]

    scaler = RobustScaler().fit(train_feats)
    train_feats = scaler.transform(train_feats)
    test_feats = scaler.transform(test_feats)

    # fsel = ExhaustiveForwardSelect(score_func=efs_score).fit(train_feats, train_labels)
    # print('Subject {}: '.format(s) + ', '.join(feature_names[fsel.selected_features]))
    # train_feats = fsel.transform(train_feats)
    # test_feats = fsel.transform(test_feats)

    inner_subject_id = subject_id.copy()
    inner_subject_id = np.delete(inner_subject_id, np.argwhere(inner_subject_id == s))

    val_scores = np.zeros(num_comb)
    print('testing!!!!!!', s)

    j = 0
    for n_keep in selnum_f_grid:
        for cval in p_grid['C']:
            for gamval in p_grid['gamma']:
                predicted_val = np.zeros(len(train_labels))
                for i_s in inner_subject_id:
                    inner_train_feats = features[np.logical_and(all_sub != s, all_sub != i_s), :]
                    # inner_train_feats = train_feats[train_sub != i_s, :]
                    inner_train_labels = compensation_labels[np.logical_and(all_sub != s, all_sub != i_s)]
                    val_feats = features[all_sub == i_s, :]
                    # val_feats = train_feats[train_sub == i_s, :]
                    val_labels = compensation_labels[all_sub == i_s]

                    scaler_inner = RobustScaler().fit(inner_train_feats)
                    inner_train_feats = scaler_inner.transform(inner_train_feats)
                    val_feats = scaler_inner.transform(val_feats)

                    fsel = SelectKBest(f_classif, k=n_keep).fit(inner_train_feats, inner_train_labels)
                    # print(i_s, feature_names[fsel.get_support()])
                    inner_train_feats = fsel.transform(inner_train_feats)
                    val_feats = fsel.transform(val_feats)

                    # # # ReliefF
                    # fsel = ReliefF(n_neighbors=10, n_features_to_keep=n_keep)
                    # inner_train_feats = fsel.fit_transform(inner_train_feats, inner_train_labels)
                    # val_feats = fsel.transform(val_feats)

                    # # CFS
                    # fs = CFS().fit(inner_train_feats, inner_train_labels, 'backward')
                    # inner_train_feats = fs.transform(inner_train_feats)
                    # val_feats = fs.transform(val_feats)
                    # print(feature_names[fs.selected_features])
                    # # inner_train_feats = inner_train_feats[:, cfs_idx]
                    # # val_feats = val_feats[:, cfs_idx]

                    # # FCBF
                    # cfs_idx, dfd = FCBF.fcbf(inner_train_feats, inner_train_labels)
                    # print(feature_names[cfs_idx])
                    # print(dfd)
                    # inner_train_feats = inner_train_feats[:, cfs_idx]
                    # val_feats = val_feats[:, cfs_idx]

                    val_model = SVC(C=cval, gamma='auto', kernel='rbf', random_state=0, class_weight='balanced')
                    val_model.fit(inner_train_feats, inner_train_labels)
                    predicted_val[train_sub == i_s] = val_model.predict(val_feats)
                val_scores[j] = accuracy_score(train_labels, predicted_val)
                j += 1

    best_gamma = p_grid['gamma'][int(np.argmax(val_scores) % len(p_grid['gamma']))]
    best_C = p_grid['C'][int(np.argmax(val_scores) // len(p_grid['gamma']) % len(p_grid['C']))]
    best_n_keep = selnum_f_grid[int(np.argmax(val_scores) // (len(p_grid['gamma'])*len(p_grid['C'])))]
    print('r', np.max(val_scores), np.argmax(val_scores), best_C, best_gamma, best_n_keep)

    fsel = SelectKBest(f_classif, k=best_n_keep).fit(train_feats, train_labels)
    print(s, feature_names[fsel.get_support()])
    train_feats = fsel.transform(train_feats)
    test_feats = fsel.transform(test_feats)
    #     fsel = RFE(estimator=RandomForestClassifier(n_estimators=10, random_state=0, class_weight='balanced'),  n_features_to_select=15).fit(train_feats, train_labels)


    # fsel = ReliefF(n_neighbors=10, n_features_to_keep=best_n_keep)
    # train_feats = fsel.fit_transform(train_feats, train_labels)
    # print(s, feature_names[fsel.top_features[:best_n_keep]])
    # test_feats = fsel.transform(test_feats)
    # train_feats = train_feats[:, :best_nf]
    # test_feats = test_feats[:, :best_nf]
    # print(test_feats.shape)

    # fs = CFS().fit(train_feats, train_labels, 'backward')
    # train_feats = fs.transform(train_feats)
    # test_feats = fs.transform(test_feats)
    # print(feature_names[fs.selected_features])

    # cfs_idx, _ = FCBF.fcbf(train_feats, train_labels)
    # train_feats = train_feats[:, cfs_idx]
    # test_feats = test_feats[:, cfs_idx]
    # print(feature_names[cfs_idx])

    # model = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced').fit(train_feats, train_labels)
    model = SVC(kernel='rbf', random_state=0, class_weight='balanced', C=best_C, gamma='auto')

    model.fit(train_feats, train_labels)
    predicted[all_sub == s] = model.predict(test_feats)
    predicted_testscore[all_sub == s] = model.decision_function(test_feats)
    fpr_per_sub, tpr_per_sub, _ = roc_curve(compensation_labels[all_sub == s], predicted_testscore[all_sub == s])
    fpr_set.append(fpr_per_sub)
    tpr_set.append(tpr_per_sub)
    print('elapsed time:', time.time() - start_time)



print(compensation_labels)
print(predicted)
fig_conf = plt.figure(figsize=[6, 6])
ax1 = fig_conf.add_subplot(1, 1, 1)
plot_confusion_matrix(ax1, compensation_labels, predicted, ['Abnormal', 'Normal'], normalize=True)
# ax1.set_title('Total')
# ax2 = fig_conf.add_subplot(1, 3, 2)
# plot_confusion_matrix(ax2, compensation_labels[reach_retract_mask], predicted[reach_retract_mask], ['Abnormal', 'Normal'], normalize=True)
# ax2.set_title('Reaching')
# ax3 = fig_conf.add_subplot(1, 3, 3)
# plot_confusion_matrix(ax3, compensation_labels[~reach_retract_mask], predicted[~reach_retract_mask], ['Abnormal', 'Normal'], normalize=True)
# ax3.set_title('Retracting')
fig_conf.tight_layout()
fig_conf.savefig('confusion_matrix_selectkbest_2class_nofiltadd_test')
np.save('selectkbest_2class_nofiltadd_test', predicted)
np.save('selectkbest_2class_nofiltadd_score_test', predicted_testscore)

print('Accuracy: {:.1f}%'.format(accuracy_score(compensation_labels, predicted) * 100))
print(classification_report(compensation_labels, predicted))
print('weighted per class F1 Score: {:.2f}'.format(f1_score(compensation_labels, predicted, average='weighted')))
print('unweighted per class F1 Score: {:.2f}'.format(f1_score(compensation_labels, predicted, average='macro')))
print('global F1 Score: {:.2f}'.format(f1_score(compensation_labels, predicted, average='micro')))

# print('Reaching Accuracy: {:.1f}%'.format(accuracy_score(compensation_labels[reach_retract_mask], predicted[reach_retract_mask]) * 100))
# print('Retracting Accuracy: {:.1f}%'.format(accuracy_score(compensation_labels[~reach_retract_mask], predicted[~reach_retract_mask]) * 100))

# Compute ROC curve and ROC area for each class

fpr, tpr, _ = roc_curve(compensation_labels, predicted_testscore)
roc_auc = auc(fpr, tpr)
roc_fig = plt.figure()
roc_ax = roc_fig.add_subplot(1, 1, 1)
roc_ax.plot(fpr, tpr, color='red', lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
# for kk in range(len(fpr_set)):
#     roc_ax.plot(fpr_set[kk], tpr_set[kk], label=str(subject_id[kk]))
plt.legend()
roc_fig.savefig('roc_selectkbest_2class_nofiltadd_test')
plt.show()