import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from sklearn.decomposition import PCA

from extract_features import extract_features
from data_struct import CompData
from utils import magnitude_xy


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

# subject_id = np.array([1, 5, 8, 9, 10, 11, 12, 17])

save_data_file = open('sub_data_filt_ok.p', 'rb')
sub_data = pickle.load(save_data_file)
# results = np.load('selectkbest_2class_onlyfilt_3to7.npy')
predicted_testscore = np.load('selectkbest_2class_score_reach_1p5hz.npy')
results = (predicted_testscore > 0.10175977690779249).astype(float)

fs = 100
dx = 1 / fs
resample_num = 300

# feature extraction
features = []
compensation_labels = []
all_sub = []
for sub_id in subject_id:
    velfilt_affect_fore_reach = sub_data[sub_id].velfilt_affect_fore_reach
    velfilt_affect_fore_retract = sub_data[sub_id].velfilt_affect_fore_retract
    vel_affect_fore_reach = sub_data[sub_id].vel_affect_fore_reach
    vel_affect_fore_retract = sub_data[sub_id].vel_affect_fore_retract
    free_acc_affect_fore_reach = sub_data[sub_id].free_acc_affect_fore_reach
    free_acc_affect_fore_retract = sub_data[sub_id].free_acc_affect_fore_retract
    target_dist = sub_data[sub_id].target_dist
    for ii in range(len(velfilt_affect_fore_reach)):
        mvel_xy_reach = magnitude_xy(velfilt_affect_fore_reach[ii])
        mvelnofilt_xy_reach = magnitude_xy(vel_affect_fore_reach[ii])
        macc_xy_reach = magnitude_xy(free_acc_affect_fore_reach[ii])
        mvel_xy_reach_norm = resample(mvel_xy_reach / np.mean(mvel_xy_reach), resample_num, np.arange(len(mvel_xy_reach)))[0]
        mvel_z_reach = np.sqrt(np.square(velfilt_affect_fore_reach[ii][:, 2]))

        feat, feature_names = extract_features(mvel_xy_reach, sub_data[sub_id].mft_score, fs=100, prefix='velfilt_')
        feat_z, feature_names_z = extract_features(mvel_z_reach, None, z=True, fs=100, prefix='velfilt_z_')
        # feat2, feature_names2 = extract_features(mvelnofilt_xy_reach, None, fs=100, prefix='vel_')
        feat_acc, feature_names_acc = extract_features(macc_xy_reach, None, z=True, fs=100, prefix='acc_')

        # feat.append(1.0)
        # feature_names.append('reachretract')
        #
        # feat.append(np.max(velfilt_affect_fore_reach[ii][:, 2]))
        # feature_names.append('zheight')
        #
        # feat.append(np.argmax(velfilt_affect_fore_reach[ii][:, 2]))
        # feature_names.append('zmaxdur')

        # features.append(np.hstack((feat, feat2)))
        features.append(feat  + feat_z + feat_acc)
        all_sub.append(sub_id)

    for ll in range(len(sub_data[sub_id].reach_fas_score)):
        if sub_data[sub_id].reach_fas_score[ll] == 5: #or sub_data[sub_id].reach_fas_score[ll] == 4:
            sub_data[sub_id].reach_fas_score[ll] = 1
        else: # elif sub_data[sub_id].reach_fas_score[ll] != 5: #or sub_data[sub_id].reach_comp_score[ll] == 2:
            sub_data[sub_id].reach_fas_score[ll] = 0

    compensation_labels.extend(sub_data[sub_id].reach_fas_score)

    # for ii in range(len(velfilt_affect_fore_retract)):
    #     mvel_xy_retract = magnitude_xy(velfilt_affect_fore_retract[ii])
    #     mvelnofilt_xy_retract = magnitude_xy(vel_affect_fore_retract[ii])
    #     mvel_z_retract = np.sqrt(np.square(velfilt_affect_fore_retract[ii][:, 2]))
    #
    #     mvel_xy_retract_norm = resample(mvel_xy_retract / np.mean(mvel_xy_retract), resample_num, np.arange(len(mvel_xy_retract)))[0]
    #
    #     feat, feature_names = extract_features(mvel_xy_retract, sub_data[sub_id].mft_score, fs=100, prefix='velfilt_')
    #     feat_z, feature_names_z = extract_features(mvel_z_retract, None, z=True, fs=100, prefix='velfilt_zzzzz_')
    #
    #     # feat2, feature_names2 = extract_features(mvelnofilt_xy_retract, None, fs=100, prefix='vel_')
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
    #     # features.append(np.hstack((feat, feat2)))
    #     features.append(feat + feat_z)
    #     all_sub.append(sub_id)
    #
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
all_sub = np.asarray(all_sub)
print(features.shape)
feature_names = np.asarray(feature_names + feature_names_z + feature_names_acc)

print('the number of features', len(feature_names))
print('comp 0', len(np.where(compensation_labels == 0)[0]), 'comp1', len(np.where(compensation_labels == 1)[0]), 'comp2',
      len(np.where(compensation_labels == 2)[0]), 'total', len(compensation_labels))


corr = np.zeros((features.shape[1],))
for i in range(features.shape[1]):
    corr[i], _ = pearsonr(features[np.logical_and(all_sub != 18, all_sub != 20), i], compensation_labels[np.logical_and(all_sub != 18, all_sub != 20)]) #np.isin(all_sub, [5, 14, 16])
for i in np.flip(np.argsort(np.absolute(corr)), 0):
    print('{:+.2f} {}'.format(corr[i], feature_names[i]), i)
print('------------------------------------------')
for i in range(features.shape[1]):
    corr[i], _ = pearsonr(features[:, i], compensation_labels) #np.isin(all_sub, [5, 14, 16])
for i in np.flip(np.argsort(np.absolute(corr)), 0):
    print('{:+.2f} {}'.format(corr[i], feature_names[i]), i)

sns.set()
sns.set_context('talk')
color_palette = sns.color_palette('deep')

pca = PCA(n_components=2).fit(features) #[:, [0, 1, 2, 7, 15, 18, 19, 20, 25, 33]])
all_pca = pca.transform(features) #[:, [0, 1, 2, 7, 15, 18, 19, 20, 25, 33]])
for d in [5, 14, 16]:
    pca_fig = plt.figure()
    pca_ax = pca_fig.add_subplot(1, 1, 1)
    pca_ax.scatter(all_pca[compensation_labels == 0, 0], all_pca[compensation_labels == 0, 1], c='r', s=20, alpha=0.5)
    pca_ax.scatter(all_pca[compensation_labels == 1, 0], all_pca[compensation_labels == 1, 1], c='b', s=20, alpha=0.5)
    # pca_ax2 = pca_fig.add_subplot(1, 2, 2)
    # pca_ax2.scatter(all_pca[results == 0, 0], all_pca[results == 0, 1], c='r', s=20)
    # pca_ax2.scatter(all_pca[results == 1, 0], all_pca[results == 1, 1], c='b', s=20)
    # pca_ax2.scatter(all_pca[compensation_labels != results, 0], all_pca[compensation_labels != results, 1], c='orange', s=20)
    pca_ax.scatter(all_pca[np.logical_and(compensation_labels == 0, all_sub == d), 0], all_pca[np.logical_and(compensation_labels == 0, all_sub == d), 1], c='g', s=20)
    pca_ax.scatter(all_pca[np.logical_and(compensation_labels == 1, all_sub == d), 0], all_pca[np.logical_and(compensation_labels == 1, all_sub == d), 1], c='orange', s=20)
    print(d, np.sum(np.not_equal(compensation_labels[all_sub == d], results[all_sub == d]).astype(int)))

    plt.show()
# pca_ax.scatter(all_pca[np.logical_and(all_sub == 5, compensation_labels == 0), 0], all_pca[np.logical_and(all_sub == 5, compensation_labels == 0), 1], c='m')
# pca_ax.scatter(all_pca[np.logical_and(all_sub == 5, compensation_labels == 1), 0], all_pca[np.logical_and(all_sub == 5, compensation_labels == 1), 1], c='g')

# pca_ax.scatter(all_pca[all_sub == 14, 0], all_pca[all_sub == 14, 1], c='m')
# pca_ax.scatter(all_pca[all_sub == 16, 0], all_pca[all_sub == 16, 1], c='m')
plt.show()
# fig = plt.figure(figsize=(12, 4))
# ax = fig.add_subplot(1, 2, 1)
# fname = 'abs(avg(v))*L'
# features_flat = features[:, feature_names == fname].flatten()
# ax.scatter(features_flat, compensation_labels, s=15)
# fitfunc = np.poly1d(np.polyfit(features_flat, compensation_labels, 1))
# ax.plot(features_flat, fitfunc(features_flat), 'k')
# ax.set_xlabel(fname)
# ax.set_ylabel('D - DI')
# ax.set_title('Pearson {:+.2f}, Spearman {:+.2f}'.format(pearsonr(features_flat, compensation_labels)[0], spearmanr(features_flat, compensation_labels)[0]))
# ax = fig.add_subplot(1, 2, 2)
# ax.hist(features_flat, bins=50)
# ax.set_xlabel(fname)
# ax.set_ylabel('# MEs')
# fig.tight_layout()
# plt.show()

# print(features[np.logical_and(all_sub==5, compensation_labels == results), 2])
# print(features[np.logical_and(all_sub==5, compensation_labels != results), 2])
# print(features[np.logical_and(results==0, compensation_labels != results), 41])
# print(features[np.logical_and(results==1, compensation_labels != results), 41])


savepath = 'features_without_mft'
if not os.path.exists(savepath):
    os.makedirs(savepath)

# for fidx in range(len(feature_names)):
#     for sub in subject_id:
#         if not os.path.exists(os.path.join(savepath, str(sub))):
#             os.makedirs(os.path.join(savepath, str(sub)))
#         feature = features[:, fidx]
#         fig = plt.figure(figsize=(6, 6))
#         ax = fig.add_subplot(1, 1, 1)
#         sns.distplot(feature[np.logical_and(all_sub == sub, compensation_labels == 0)], color='red', label='abnormal', ax=ax)
#         sns.distplot(feature[np.logical_and(all_sub == sub, compensation_labels == 1)], color='green', label='normal', ax=ax)
#         # sns.distplot(feature[compensation_labels == 0], color='red', label='abnormal', ax=ax)
#         # sns.distplot(feature[compensation_labels == 1], color='green', label='normal', ax=ax)
#         # sns.distplot(feature[np.logical_and(all_sub == 5, compensation_labels == 1)], color='red', label='sub5', ax=ax2)
#         plt.title(feature_names[fidx])
#         plt.legend(loc='upper right')
#         # plt.show()
#         fig.savefig(os.path.join(savepath, str(sub), 'reach_' +feature_names[fidx].replace('/', '_')))
#         plt.close()


for fidx in range(len(feature_names)):
    feature = features[:, fidx]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    sns.distplot(feature[compensation_labels == 0], color='red', label='abnormal', ax=ax)
    sns.distplot(feature[compensation_labels == 1], color='green', label='normal', ax=ax)
    # sns.distplot(feature[compensation_labels == 0], color='red', label='abnormal', ax=ax)
    # sns.distplot(feature[compensation_labels == 1], color='green', label='normal', ax=ax)
    # sns.distplot(feature[np.logical_and(all_sub == 5, compensation_labels == 1)], color='red', label='sub5', ax=ax2)
    plt.title(feature_names[fidx])
    plt.legend(loc='upper right')
    # plt.show()
    fig.savefig(os.path.join(savepath, 'reach_' +feature_names[fidx].replace('/', '_')))
    plt.close()

