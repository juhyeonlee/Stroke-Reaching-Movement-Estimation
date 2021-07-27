import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from scipy.signal import resample
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.decomposition import PCA
import pandas as pd
from pingouin import welch_anova, pairwise_gameshowell, homoscedasticity, kruskal, normality
from utils import welch_anova_np, plot_correlation_matrix
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

from ReliefF import ReliefF
import seaborn as sns

sns.set()
sns.set_context('talk')
plt.rcParams["patch.force_edgecolor"] = False  # Turn off histogram borders

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
# with mft scores
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 3))
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 15))
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 6))
# # exclude 5?
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 5))
# subject_id = np.delete(subject_id, np.argwhere(subject_id == 12))


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
    # for reaching movement
    for ii in range(len(velfilt_affect_fore_reach)):
        # compute magnitude of x- and y-axis velocity
        mvel_xy_reach = magnitude_xy(velfilt_affect_fore_reach[ii])
        # compute magnitude of x- and y-axis non-filtered velocity
        mvelnofilt_xy_reach = magnitude_xy(vel_affect_fore_reach[ii])
        # compute magnitude of x- and y-axis acceleration
        macc_xy_reach = magnitude_xy(free_acc_affect_fore_reach[ii])
        # compute magnitude of z-axis velocity
        mvel_z_reach = np.sqrt(np.square(velfilt_affect_fore_reach[ii][:, 2]))
        # residual_reach = abs(mvelnofilt_xy_reach - mvel_xy_reach)
        # normalized magnitude
        mvel_xy_reach_norm = resample(mvel_xy_reach / np.mean(mvel_xy_reach), resample_num, np.arange(len(mvel_xy_reach)))[0]

        # extract features
        feat, feature_names = extract_features(mvel_xy_reach, sub_data[sub_id].mft_score, fs=100, prefix='velfilt_')
        # feat_z, feature_names_z = extract_features(mvel_z_reach, None, z=True, fs=100, prefix='velfilt_z_')
        # feat_nonvel, feature_names_nonvel = extract_features(mvelnofilt_xy_reach, None, fs=100, prefix='vel_')
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

        # use velocify and acceleration features from xy magnitude for BHI21 paper
        features.append(feat + feat_acc)
        all_sub.append(sub_id)
        reach_retract_mask.append(True)

    # list up labels
    for ll in range(len(sub_data[sub_id].reach_fas_score)):
        if sub_data[sub_id].reach_fas_score[ll] == 5:
            sub_data[sub_id].reach_fas_score[ll] = 1
        else:
            sub_data[sub_id].reach_fas_score[ll] = 0

    compensation_labels.extend(sub_data[sub_id].reach_fas_score)

    # for retracting movement
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
    #     # feat_nonvel, feature_names_nonvel = extract_features(mvelnofilt_xy_retract, None, fs=100, prefix='vel_')
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

compensation_labels = np.asarray(compensation_labels).reshape(-1)
features = np.array(features)
print(features.shape)
feature_names = np.asarray(feature_names + feature_names_acc)

print('the number of features', len(feature_names))
print('comp0', len(np.where(compensation_labels == 0)[0]), 'comp1', len(np.where(compensation_labels == 1)[0]),
      'comp2', len(np.where(compensation_labels == 2)[0]), 'comp3', len(np.where(compensation_labels == 3)[0]),
      'comp4', len(np.where(compensation_labels == 4)[0]), 'comp5', len(np.where(compensation_labels == 5)[0]),
      "total", len(compensation_labels))

# investigate correlation between features and labels
n_features = features.shape[1]
corrs = np.zeros((n_features,))
pval = np.zeros((n_features,))
for i in range(n_features):
    corrs[i], pval[i] = pearsonr(features[:, i], compensation_labels)
sorted_corrs = np.flip(np.argsort(np.absolute(corrs)), 0)
for i in sorted_corrs:
    print('{:+.2f} {} {} {:.5f}'.format(corrs[i], i, feature_names[i], pval[i]))
for i in range(n_features):
    print(feature_names[i], pearsonr(features[:, i], features[:, 14]))

inter_corrs = np.zeros((n_features, n_features))
for i in range(n_features):
    for j in range(n_features):
        inter_corrs[i, j] = pearsonr(features[:, i], features[:, j])[0]

# plt.figure()
# plot_correlation_matrix(inter_corrs, feature_names, feature_names)
# plt.show()

# # for paper feature figure
# sel_f=[0, 6, 2]
# # sel_name = ['Log number of peaks \n(velocity)', 'log number of frequency components \n(velocity)',
# #             'Log number of frequency components \n(acceleration)',
# #             'Ratio of the log number of frequency components \nto log duration (velocity)',
# #             'Ratio of the log number of frequency components \nto log duration (acceleration)']
# fig_box = plt.figure(figsize=[25, 5])
# for idx, sel in enumerate(sel_f):
#     stats_dataframe = pd.DataFrame({
#         'Impairment Group': np.hstack([np.array([i for i in compensation_labels])]),
#         'features': np.hstack([np.array([i for i in features[:, sel]])])
#     })
#     ax = fig_box.add_subplot(1, 5, idx+1)
#     data = [features[compensation_labels==0, sel],  features[compensation_labels==1, sel]]
#     print('median', np.median(features[compensation_labels==0, sel]), np.median(features[compensation_labels==1, sel]))
#     ax.boxplot(data)
#     ax.set_xticklabels(['Desirable', 'Undesirable'])
#     # sns.kdeplot(features[compensation_labels==0, sel], legend='Abnormal')
#     # sns.kdeplot(features[compensation_labels==1, sel], legend='Normal')
#     # plt.legend()
#     # plt.title(sel_name[idx])
#     # print('all stat test', welch_anova(stats_dataframe, dv='features', between='Impairment Group'))
#     print('all stat test', welch_anova_np(features[compensation_labels==0, sel],  features[compensation_labels==1, sel]))
#     # print('all stat test - kruskal', kruskal(stats_dataframe, dv='features', between='Impairment Group'))
#     # print('pair', pairwise_gameshowell(stats_dataframe, dv='features', between='Impairment Group'))
#
# for ii in range(n_features):
#     print('all stat test',
#           welch_anova_np(features[compensation_labels == 0, ii], features[compensation_labels == 1, ii]))
# plt.tight_layout()
# plt.savefig('box_plots2.pdf')
# # plt.show()

all_sub = np.asarray(all_sub)
reach_retract_mask = np.asarray(reach_retract_mask)
print(len(all_sub), len(features))
predicted = np.zeros(compensation_labels.shape)
predicted_testscore = np.zeros(compensation_labels.shape)

# extract PCA features with LOSOCV manner
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

# SVM hyperparameter range
C_set = np.logspace(-3, 2, 6)
gamma_set = np.logspace(-3, 2, 6)

# LOSOCV (train and test)
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

    print('testing!!!!!!', s)

    # Gaussian classifier
    kernel = 1.0 * RBF(1.0)
    model = GaussianProcessClassifier(kernel=kernel, random_state=0, n_restarts_optimizer=10, max_iter_predict=100)
    # # Random forest
    # model = RandomForestClassifier(n_estimators=500, random_state=0)
    # # linear SVM
    # clf2 = GridSearchCV(SVC(kernel='linear'), {'C': C_set}, cv=5, n_jobs=-1, refit=False)
    # clf2.fit(train_feats, train_labels)
    # print('C: ', clf2.best_params_['C'])
    # model = SVC(kernel='linear', C=clf2.best_params_['C'], probability=True).fit(train_feats, train_labels)
    # # SVM with rbf kernel
    # clf2 = GridSearchCV(SVC(kernel='rbf'), {'C': C_set, 'gamma': gamma_set}, cv=5, n_jobs=-1, refit=False)
    # clf2.fit(train_feats, train_labels)
    # print('C: ', clf2.best_params_['C'], 'gamma: ', clf2.best_params_['gamma'])
    # model = SVC(kernel='rbf', C=clf2.best_params_['C'], gamma=clf2.best_params_['gamma'], probability=True).fit(train_feats, train_labels)

    model.fit(train_feats, train_labels)
    predicted[all_sub == s] = model.predict(test_feats)
    predicted_testscore[all_sub == s] = model.predict_proba(test_feats)[:, 1]
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

fig_conf = plt.figure(figsize=[6, 6])
ax1 = fig_conf.add_subplot(1, 1, 1)
plot_confusion_matrix(ax1, compensation_labels, predicted, ['Abnormal', 'Normal'], normalize=True)
fig_conf.tight_layout()
fig_conf.savefig('confusion_matrix_gp_before_adjust_thres')
np.save('gp_prediction', predicted)
np.save('gp_prediction_probability', predicted_testscore)

# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(compensation_labels, predicted_testscore)
roc_auc = auc(fpr, tpr)
roc_fig = plt.figure()
roc_ax = roc_fig.add_subplot(1, 1, 1)
roc_ax.plot(fpr, tpr, color='red', lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
roc_ax.plot(fpr, thresholds, markeredgecolor='b',linestyle='dashed', color='b')
plt.legend()
plt.ylim(0, 1)
roc_fig.savefig('roc_gp_before_adjust_thres')
plt.show()