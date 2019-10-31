import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from scipy.stats import entropy
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from scipy.signal import butter, filtfilt, resample, find_peaks, peak_prominences
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from utils import magnitude, rms, ExhaustiveForwardSelect, efs_score, plot_confusion_matrix, \
    extract_primary_mag, extract_full_mvmt, magnitude_xy
from extract_features import extract_features
from data_struct import CompData
from ReliefF import ReliefF


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
dx = 1 / fs
resample_num = 300

# feature extraction
features = []
compensation_labels = []
for sub_id in subject_id:
    vel_affect_fore_reach = sub_data[sub_id].vel_affect_fore_reach
    vel_affect_fore_retract = sub_data[sub_id].vel_affect_fore_retract
    for ii in range(len(vel_affect_fore_reach)):
        mvel_xy_reach = magnitude_xy(vel_affect_fore_reach[ii])
        mvel_xy_reach_norm = resample(mvel_xy_reach / np.mean(mvel_xy_reach), resample_num, np.arange(len(mvel_xy_reach)))[0]

        feat, feature_names = extract_features(mvel_xy_reach, sub_data[sub_id].mft_score, fs=100)

        features.append(feat)

    for ll in range(len(sub_data[sub_id].reach_comp_score)):
        if sub_data[sub_id].reach_comp_score[ll] == 1 or sub_data[sub_id].reach_comp_score[ll] == 2:
            sub_data[sub_id].reach_comp_score[ll] = 1
        elif sub_data[sub_id].reach_comp_score[ll] == 3:
            sub_data[sub_id].reach_comp_score[ll] = 2

    compensation_labels.extend(sub_data[sub_id].reach_comp_score)

    for ii in range(len(vel_affect_fore_retract)):
        mvel_xy_retract = magnitude_xy(vel_affect_fore_retract[ii])
        mvel_xy_retract_norm = resample(mvel_xy_retract / np.mean(mvel_xy_retract), resample_num, np.arange(len(mvel_xy_retract)))[0]

        feat, feature_names = extract_features(mvel_xy_retract, sub_data[sub_id].mft_score, fs=100)

        features.append(feat)

    for ll in range(len(sub_data[sub_id].retract_comp_score)):
        if sub_data[sub_id].retract_comp_score[ll] == 1 or sub_data[sub_id].retract_comp_score[ll] == 2:
            sub_data[sub_id].retract_comp_score[ll] = 1
        elif sub_data[sub_id].retract_comp_score[ll] == 3:
            sub_data[sub_id].retract_comp_score[ll] = 2

    compensation_labels.extend(sub_data[sub_id].retract_comp_score)

compensation_labels = np.asarray(compensation_labels).reshape(-1)
features = np.asarray(features)
feature_names = np.asarray(feature_names)

print('the number of features', len(feature_names))
print('comp 0', len(np.where(compensation_labels == 0)[0]), 'comp1', len(np.where(compensation_labels == 1)[0]), 'comp2',
      len(np.where(compensation_labels == 2)[0]), 'total', len(compensation_labels))

all_sub = []
for s in subject_id:
    all_sub.extend([s] * len(sub_data[s].vel_affect_fore_reach))
    all_sub.extend([s] * len(sub_data[s].vel_affect_fore_retract))

all_sub = np.asarray(all_sub)
print(len(all_sub), len(features))
predicted = np.zeros(compensation_labels.shape)

p_grid ={'C': np.logspace(-4, 4, 10), 'gamma': [0.01, 0.1, 1.0]}

for s in subject_id:
    train_feats = features[all_sub != s, :]
    train_labels = compensation_labels[all_sub != s]
    test_feats = features[all_sub == s, :]

    ss = RobustScaler().fit(train_feats)
    train_feats = ss.transform(train_feats)
    test_feats = ss.transform(test_feats)

    # fsel = SelectKBest(f_classif, k=).fit(train_feats, train_labels)
    #     fsel = RFE(estimator=RandomForestClassifier(n_estimators=10, random_state=0, class_weight='balanced'),  n_features_to_select=15).fit(train_feats, train_labels)
    # fsel = ExhaustiveForwardSelect(score_func=efs_score).fit(train_feats, train_labels)
    fsel = ReliefF(n_neighbors=10, n_features_to_keep=20)
    # print('Subject {}: '.format(s) + ', '.join(feature_names[fsel.selected_features]))
    train_feats = fsel.fit_transform(train_feats, train_labels)
    test_feats = fsel.transform(test_feats)
    print(train_feats.shape, test_feats.shape)

    # model = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced').fit(train_feats, train_labels)
    model = SVC(kernel='rbf', random_state=0, class_weight='balanced', gamma='scale')
    # clf = GridSearchCV(estimator=model, param_grid=p_grid, iid=False, verbose=True)
    # clf.fit(train_feats, train_labels)
    # # print(clf.best_estimator_.C, clf.best_estimator_.gamma)
    # #
    # nested_score = cross_val_score(clf, X=train_feats, y=train_labels, cv=outer_cv)
    # nested_scores[i] = nested_score.mean()
    model.fit(train_feats, train_labels)
    # print(feature_names[np.flip(np.argsort(model.feature_importances_))])
    predicted[all_sub == s] = model.predict(test_feats)


print(compensation_labels)
print(predicted)
fig_conf = plt.figure(figsize=[10, 10])
ax = fig_conf.add_subplot(111)
plot_confusion_matrix(ax, compensation_labels, predicted, ['Comps', 'Poor+Inacc', 'Normal'], normalize=True)
fig_conf.tight_layout()
fig_conf.savefig('confusion_matrix')

print('Accuracy: {:.1f}%'.format(accuracy_score(compensation_labels, predicted) * 100))
print(classification_report(compensation_labels, predicted))
print('weighted per class F1 Score: {:.2f}'.format(f1_score(compensation_labels, predicted, average='weighted')))
print('unweighted per class F1 Score: {:.2f}'.format(f1_score(compensation_labels, predicted, average='macro')))
print('global F1 Score: {:.2f}'.format(f1_score(compensation_labels, predicted, average='micro')))
plt.show()
