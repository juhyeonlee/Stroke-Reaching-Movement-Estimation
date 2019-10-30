import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from scipy.stats import entropy
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from scipy.signal import butter, filtfilt, resample, find_peaks, peak_prominences
from scipy.fftpack import fft, fftfreq
from scipy.stats import pearsonr, spearmanr, skew, kurtosis, linregress
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from utils import magnitude, rms, ExhaustiveForwardSelect, efs_score, plot_confusion_matrix, \
    extract_primary_mag, extract_full_mvmt, magnitude_xy
from extract_features import extract_features
from data_struct import CompData


num_subjects = 20
exclude_sub_num = 13
direction = ["L", "R"]

# list subject ID
subject_id = np.arange(0, num_subjects) + 1
subject_id = np.delete(subject_id, np.argwhere(subject_id == exclude_sub_num))
# with mft
subject_id = np.delete(subject_id, np.argwhere(subject_id == 3))
subject_id = np.delete(subject_id, np.argwhere(subject_id == 15))
subject_id = np.delete(subject_id, np.argwhere(subject_id == 6))

# subject_id = np.array([1, 5, 8, 9, 10, 11, 12, 17])

save_data_file = open('sub_data_filt_ok.p', 'rb')
sub_data = pickle.load(save_data_file)

fs = 100
fc_lp = 6
b_lp, a_lp = butter(6, 6 / (fs/2), btype='lowpass')
b_hp, a_hp = butter(2, 0.1 / (fs/2), btype='highpass')
dx = 1 / fs
resample_num = 300

features = []
compensation_labels = []
for sub_id in subject_id:
    vel_affect_fore_reach = sub_data[sub_id].vel_affect_fore_reach
    vel_affect_fore_retract = sub_data[sub_id].vel_affect_fore_retract
    for ii in range(len(vel_affect_fore_reach)):
        feat = []
        mvel_xy_reach = magnitude_xy(vel_affect_fore_reach[ii])
        mvel_xy_reach_norm = resample(mvel_xy_reach / np.mean(mvel_xy_reach), resample_num, np.arange(len(mvel_xy_reach)))[0]
        # peaks_reach, _ = find_peaks(mvel_xy_reach)
        # prominences_reach, *_ = peak_prominences(mvel_xy_reach, peaks_reach)
        #
        # feat.append(len(mvel_xy_reach))
        #
        # feat.append(len(peaks_reach))
        #
        #
        # fft_y = fft(mvel_xy_reach, n=len(mvel_xy_reach))
        # freq = fftfreq(len(mvel_xy_reach), d=1/fs)
        # psd = np.absolute(fft_y / len(mvel_xy_reach))[:int(len(mvel_xy_reach)/2)+1]
        # sortp = np.flip(np.argsort(psd), 0)
        # feat.append(freq[sortp[0]] / freq[sortp[1]]) # ratio between dominant freq and secondary freq
        # feat.append(np.argmax(psd)) # Location of max value (dominant freq)
        # # max_val = np.max(psd)
        # # max_loc = np.argmax(psd)
        # # fft_peaks, _ = find_peaks(psd)
        # # dd = np.argsort(psd[fft_peaks])
        # # fft_peaks_sorted = fft_peaks[dd]
        # # feat.append(fft_peaks[-1])
        # # feat.append(fft_peaks[-2])
        # # feat.append(np.sum(psd) / np.sum(psd[fft_peaks[1:]]))
        #
        #
        # if len(peaks_reach) > 1:
        #     # feat.append(np.diff(peaks_reach).mean())
        #     feat.append(np.diff(peaks_reach).max())
        #     # feat.append(np.diff(peaks_reach).std())
        # else:
        #     # feat.append(0.0)
        #     feat.append(0.0)
        #     # feat.append(0.0)
        feat, feature_names = extract_features(mvel_xy_reach)

        feat.append(sub_data[sub_id].mft_score)

        features.append(feat)

    for ll in range(len(sub_data[sub_id].reach_comp_score)):
        if sub_data[sub_id].reach_comp_score[ll] == 1 or sub_data[sub_id].reach_comp_score[ll] == 2:
            sub_data[sub_id].reach_comp_score[ll] = 1
        elif sub_data[sub_id].reach_comp_score[ll] == 3:
            sub_data[sub_id].reach_comp_score[ll] = 2

    compensation_labels.extend(sub_data[sub_id].reach_comp_score)

    for ii in range(len(vel_affect_fore_retract)):
        feat = []
        mvel_xy_retract = magnitude_xy(vel_affect_fore_retract[ii])
        mvel_xy_retract_norm = resample(mvel_xy_retract / np.mean(mvel_xy_retract), resample_num, np.arange(len(mvel_xy_retract)))[0]
        peaks_retract, _ = find_peaks(mvel_xy_retract)
        prominences_retract, *_ = peak_prominences(mvel_xy_retract, peaks_retract)

        feat.append(len(mvel_xy_retract))

        feat.append(len(peaks_retract))

        fft_y = fft(mvel_xy_retract, n=len(mvel_xy_retract))
        freq = fftfreq(len(mvel_xy_retract), d=1 / fs)
        psd = np.absolute(fft_y / len(mvel_xy_retract))[:int(len(mvel_xy_retract) / 2) + 1]
        sortp = np.flip(np.argsort(psd), 0)
        feat.append(freq[sortp[0]] / freq[sortp[1]])
        feat.append(np.argmax(psd))


        # max_val = np.max(psd)
        # max_loc = np.argmax(psd)
        # fft_peaks, _ = find_peaks(psd)
        # dd = np.argsort(psd[fft_peaks])
        # fft_peaks_sorted = fft_peaks[dd]
        # feat.append(fft_peaks[-1])
        # feat.append(fft_peaks[-2])
        # feat.append(np.sum(psd) / np.sum(psd[fft_peaks[1:]]))

        if len(peaks_retract) > 1:
            # feat.append(np.diff(peaks_retract).mean())
            feat.append(np.diff(peaks_retract).max())
            # feat.append(np.diff(peaks_retract).std())
        else:
            # feat.append(0.0)
            feat.append(0.0)

        feat.append(sub_data[sub_id].mft_score)

        features.append(feat)

    for ll in range(len(sub_data[sub_id].retract_comp_score)):
        if sub_data[sub_id].retract_comp_score[ll] == 1 or sub_data[sub_id].retract_comp_score[ll] == 2:
            sub_data[sub_id].retract_comp_score[ll] = 1
        elif sub_data[sub_id].retract_comp_score[ll] == 3:
            sub_data[sub_id].retract_comp_score[ll] = 2

    compensation_labels.extend(sub_data[sub_id].retract_comp_score)

compensation_labels = np.asarray(compensation_labels).reshape(-1)
features = np.asarray(features)
feature_names = ['dur', 'num_peaks', 'ratiofreq12', 'maxlocpsd', 'maxpeakD']#, 'mft']
feature_names = np.asarray(feature_names)
# assert len(feature_names) == features.shape[1]

# corr = np.zeros((features.shape[1],))
# for i in range(features.shape[1]):
#     corr[i], _ = pearsonr(features[:, i], compensation_labels)
# for i in np.flip(np.argsort(np.absolute(corr)), 0)[:15]:
#     print('{:+.2f} {}'.format(corr[i], feature_names[i]))



# freq_, _ = f_classif(features, compensation_labels.astype(bool))

all_sub = []
for s in subject_id:
    all_sub.extend([s] * len(sub_data[s].vel_affect_fore_reach))
    all_sub.extend([s] * len(sub_data[s].vel_affect_fore_retract))

all_sub = np.asarray(all_sub)
print(len(all_sub), len(features))
predicted = np.zeros(compensation_labels.shape)

p_grid ={'C': np.logspace(-4, 4, 10), 'gamma': [0.01, 0.1, 1.0]}

nested_scores = np.zeros(30)
for tt in range(30):
    train_feats = features #[all_sub != s, :]
    train_labels = compensation_labels #[all_sub != s]
    ss = RobustScaler().fit(train_feats)
    train_feats = ss.transform(train_feats)

    inner_cv = KFold(n_splits=4, shuffle=True, random_state=tt)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=tt)

    model = SVC(kernel='rbf', random_state=0, class_weight='balanced')
    clf = GridSearchCV(estimator=model, param_grid=p_grid, iid=False, verbose=True)
    clf.fit(train_feats, train_labels)
    print(clf.best_estimator_.C, clf.best_estimator_.gamma)

    nested_score = cross_val_score(clf, X=train_feats, y=train_labels, cv=outer_cv)
    nested_scores[tt] = nested_score.mean()
print(nested_scores)

for s in subject_id:
    train_feats = features[all_sub != s, :]
    train_labels = compensation_labels[all_sub != s]
    test_feats = features[all_sub == s, :]

    ss = RobustScaler().fit(train_feats)
    train_feats = ss.transform(train_feats)
    test_feats = ss.transform(test_feats)

    #     fsel = SelectKBest(f_classif, k=20).fit(train_feats, train_labels)
    #     fsel = RFE(estimator=RandomForestClassifier(n_estimators=10, random_state=0, class_weight='balanced'),  n_features_to_select=15).fit(train_feats, train_labels)
    # fsel = ExhaustiveForwardSelect(score_func=efs_score).fit(train_feats, train_labels)
    # print('Subject {}: '.format(s) + ', '.join(feature_names[fsel.selected_features]))
    # train_feats = fsel.transform(train_feats)
    # test_feats = fsel.transform(test_feats)

    # model = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced').fit(train_feats, train_labels)
    model = SVC( kernel='rbf', random_state=0, class_weight='balanced', gamma='scale')
    # clf = GridSearchCV(estimator=model, param_grid=p_grid, iid=False, verbose=True)
    # clf.fit(train_feats, train_labels)
    # # print(clf.best_estimator_.C, clf.best_estimator_.gamma)
    # #
    # nested_score = cross_val_score(clf, X=train_feats, y=train_labels, cv=outer_cv)
    # nested_scores[i] = nested_score.mean()
    model.fit(train_feats, train_labels)
    predicted[all_sub == s] = model.predict(test_feats)


print(compensation_labels)
print(predicted)
fig_conf = plt.figure(figsize=[10, 10])
ax = fig_conf.add_subplot(111)
plot_confusion_matrix(ax, compensation_labels, predicted, ['Comps', 'Poor+Inacc', 'Normal'], normalize=True)
fig_conf.tight_layout()
fig_conf.savefig('confusion_matrix')
# plt.show()

print('Accuracy: {:.1f}%'.format(accuracy_score(compensation_labels, predicted) * 100))
print(classification_report(compensation_labels, predicted))
print('weighted per class F1 Score: {:.2f}'.format(f1_score(compensation_labels, predicted, average='weighted')))
print('unweighted per class F1 Score: {:.2f}'.format(f1_score(compensation_labels, predicted, average='macro')))
print('global F1 Score: {:.2f}'.format(f1_score(compensation_labels, predicted, average='micro')))


# fig_acc = plt.figure()
#
# ax_acc_x = fig_acc.add_subplot(3, 1, 1)
# plt.title('acc: ID {} / pos {} cm / {} deg / affect {} / comp label {}'.format(
#     sub_id, sub_data[sub_id].target_dist[ii], sub_data[sub_id].target_angle[ii],
#     direction[sub_data[sub_id].affect_side], sub_data[sub_id].reach_comp_score[ii]))
# ax_acc_x.plot(np.hstack((hoff_reach, hoff_retract)), label='X')
# ax_acc_x.legend()
# ax_acc_x.set_ylabel('acc')
# ax_acc_x.axhline(0, color='k')
# ax_acc_x.axvline(len(hoff_reach), color='k', linestyle='--')
# plt.show()