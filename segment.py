import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC, SVR
from scipy.signal import butter, filtfilt, resample, find_peaks, peak_prominences
from scipy.stats import pearsonr, spearmanr, skew, kurtosis, linregress
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from utils import magnitude, rms, ExhaustiveForwardSelect, efs_score, plot_confusion_matrix
from data_struct import CompData


num_subjects = 20
exclude_sub_num = 13
direction = ["L", "R"]

# list subject ID
subject_id = np.arange(0, num_subjects) + 1
subject_id = np.delete(subject_id, np.argwhere(subject_id == exclude_sub_num))

save_data_file = open('sub_data_filt.p', 'rb')
sub_data = pickle.load(save_data_file)

fs = 100
fc_lp = 6
b_lp, a_lp = butter(6, 6 / (fs/2), btype='lowpass')
b_hp, a_hp = butter(2, 0.1 / (fs/2), btype='highpass')
dx = 1 / fs

features = []
compensation_labels = []
for sub_id in subject_id:
    vel_affect_fore_reach = sub_data[sub_id].vel_affect_fore_reach
    vel_affect_fore_retract = sub_data[sub_id].vel_affect_fore_retract
    for ii in range(len(vel_affect_fore_reach)):
        feat = []
        mvel_reach = magnitude(vel_affect_fore_reach[ii])

        mvel_norm_reach = mvel_reach / mvel_reach.mean()

        ts = np.linspace(0, 1, mvel_reach.shape[0])
        reach_duration = len(mvel_reach) / 100.
        hoff = 30 * np.power(ts, 4) - 60 * np.power(ts, 3) + 30 * np.power(ts, 2)
        homog_mean = np.pi / 2 * np.sin(np.pi * ts)

        feat.append(1.0)

        # mean/peak
        feat.append(mvel_reach.mean() / mvel_reach.max())

        # log|mean/peak|
        feat.append(np.log10(np.absolute(mvel_reach.mean() / mvel_reach.max())))

        # 1/|peak norm|
        feat.append(1 / np.absolute(mvel_norm_reach).max())

        # |peak norm|
        feat.append(np.absolute(mvel_norm_reach).max())

        # log mean seed
        feat.append(np.log10(np.absolute(mvel_reach.mean())))

        feat.append(rms(hoff - mvel_reach))

        feat.append(rms(homog_mean - mvel_reach))

        peaks_reach, _ = find_peaks(np.absolute(mvel_reach))
        prominences_reach, *_ = peak_prominences(np.absolute(mvel_reach), peaks_reach)

        feat.append(len(peaks_reach))

        feat.append(len(peaks_reach) / reach_duration)

        if len(peaks_reach) > 1:
            feat.append(np.diff(peaks_reach).max())
            feat.append(np.diff(peaks_reach).mean())
            feat.append(np.diff(peaks_reach).std())
            feat.append(np.percentile(np.diff(peaks_reach), 10))
            feat.append(np.percentile(np.diff(peaks_reach), 50))
            feat.append(np.percentile(np.diff(peaks_reach), 90))
            feat.append(prominences_reach.max())
            feat.append(prominences_reach.mean())
            feat.append(prominences_reach.std())
            feat.append(np.percentile(prominences_reach, 10))
            feat.append(np.percentile(prominences_reach, 50))
            feat.append(np.percentile(prominences_reach, 90))
        else:
            feat.extend([0] * 12)

        feat.append(skew(mvel_reach))

        feat.append(kurtosis(mvel_reach))

        feat.append(np.var(mvel_reach))

        features.append(feat)
        print(len(feat))

    compensation_labels.extend(sub_data[sub_id].reach_comp_label)

    for ii in range(len(vel_affect_fore_retract)):
        feat = []
        mvel_retract = magnitude(vel_affect_fore_retract[ii])

        mvel_norm_retract = mvel_retract / mvel_retract.mean()

        ts = np.linspace(0, 1, mvel_retract.shape[0])
        retract_duration = len(mvel_retract) / 100.
        hoff = 30 * np.power(ts, 4) - 60 * np.power(ts, 3) + 30 * np.power(ts, 2)
        homog_mean = np.pi / 2 * np.sin(np.pi * ts)

        feat.append(0.0)
        # mean/peak
        feat.append(mvel_retract.mean() / mvel_retract.max())

        # log|mean/peak|
        feat.append(np.log10(np.absolute(mvel_retract.mean() / mvel_retract.max())))

        # 1/|peak norm|
        feat.append(1 / np.absolute(mvel_norm_retract).max())

        # |peak norm|
        feat.append(np.absolute(mvel_norm_retract).max())

        # log mean seed
        feat.append(np.log10(np.absolute(mvel_retract.mean())))

        feat.append(rms(hoff - mvel_retract))

        feat.append(rms(homog_mean - mvel_retract))


        peaks_retract, _ = find_peaks(np.absolute(mvel_retract))
        prominences_retract, *_ = peak_prominences(np.absolute(mvel_retract), peaks_retract)

        feat.append(len(peaks_retract))

        feat.append(len(peaks_retract) / retract_duration)

        if len(peaks_retract) > 1:
            feat.append(np.diff(peaks_retract).max())
            feat.append(np.diff(peaks_retract).mean())
            feat.append(np.diff(peaks_retract).std())
            feat.append(np.percentile(np.diff(peaks_retract), 10))
            feat.append(np.percentile(np.diff(peaks_retract), 50))
            feat.append(np.percentile(np.diff(peaks_retract), 90))
            feat.append(prominences_retract.max())
            feat.append(prominences_retract.mean())
            feat.append(prominences_retract.std())
            feat.append(np.percentile(prominences_retract, 10))
            feat.append(np.percentile(prominences_retract, 50))
            feat.append(np.percentile(prominences_retract, 90))
        else:
            feat.extend([0] * 12)

        feat.append(skew(mvel_retract))

        feat.append(kurtosis(mvel_retract))

        feat.append(np.var(mvel_retract))

        features.append(feat)
        print(len(feat))
    compensation_labels.extend(sub_data[sub_id].retract_comp_label)

compensation_labels = np.asarray(compensation_labels).reshape(-1)
features = np.asarray(features)
feature_names = ['reachorretract', 'meanpeak', 'logmeanpeak', '1/peaknorm', 'peaknorm', 'logmeanspeed', 'rmshoff', 'rmshomog',
                 'numpeak', 'numpeak/dur', 'maxpeakD', 'meanpeakD', 'stdpeakD', '10peakD', '50peakD', '90peakD',
                 'maxpeakP', 'meanpeakP', 'stdpeakP', '10peakP', '50peakP', '90peakP', 'skew', 'kurtosis', 'var']
feature_names = np.asarray(feature_names)
print(len(feature_names))
F, _ = f_classif(features, compensation_labels.astype(bool))

reach_subj = []
for s in subject_id:
   reach_subj.extend([s] * len(sub_data[s].vel_affect_fore_reach))
   reach_subj.extend([s] * len(sub_data[s].vel_affect_fore_retract))

reach_subj = np.asarray(reach_subj)

predicted = np.zeros(compensation_labels.shape)
for s in subject_id:
    train_feats = features[reach_subj != s, :]
    train_labels = compensation_labels[reach_subj != s]
    test_feats = features[reach_subj == s, :]

    ss = RobustScaler().fit(train_feats)
    train_feats = ss.transform(train_feats)
    test_feats = ss.transform(test_feats)

    #     fsel = SelectKBest(f_classif, k=20).fit(train_feats, train_labels)
    #     fsel = RFE(estimator=RandomForestClassifier(n_estimators=10, random_state=0, class_weight='balanced'),  n_features_to_select=15).fit(train_feats, train_labels)
    fsel = ExhaustiveForwardSelect(score_func=efs_score).fit(train_feats, train_labels)
    print('Subject {}: '.format(s) + ', '.join(feature_names[fsel.selected_features]))
    train_feats = fsel.transform(train_feats)
    test_feats = fsel.transform(test_feats)

    #     model = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced').fit(train_feats, train_labels)
    model = SVC(random_state=0, class_weight='balanced', gamma='scale').fit(train_feats, train_labels)
    predicted[reach_subj == s] = model.predict(test_feats)

print(compensation_labels)
print(predicted)
fig = plt.figure()
ax = fig.add_subplot(111)
plot_confusion_matrix(ax, compensation_labels, predicted, ['Normal', 'Compensation'], normalize=True)

print('Accuracy: {:.1f}%'.format(accuracy_score(compensation_labels, predicted) * 100))
print('F1 Score: {:.2f}'.format(f1_score(compensation_labels, predicted, average='macro')))
fig.savefig('confusion_matrix_svm')

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