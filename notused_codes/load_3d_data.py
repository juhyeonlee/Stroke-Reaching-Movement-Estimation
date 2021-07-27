
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.integrate import cumtrapz
import pickle
from utils import magnitude_xy
from extract_features import extract_features
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA



activity_file = '2019ADLs/activity_segments.csv'
activity_subject = np.genfromtxt(activity_file, delimiter=',', skip_header=1, usecols=(0,), dtype=int)
activity_label = np.genfromtxt(activity_file, delimiter=',', skip_header=1, usecols=(1,), dtype=str)
activity_emphasis = np.genfromtxt(activity_file, delimiter=',', skip_header=1, usecols=(2,), dtype=str)
activity_begin = np.genfromtxt(activity_file, delimiter=',', skip_header=1, usecols=(3,), dtype=int)
activity_end = np.genfromtxt(activity_file, delimiter=',', skip_header=1, usecols=(4,), dtype=int)

summary_file = '2019ADLs/Summer2018ScoreTaskSummary.csv'
summary_subject = np.genfromtxt(summary_file, delimiter=',', skip_header=1, usecols=(0,), dtype=int)
summary_affected = np.genfromtxt(summary_file, delimiter=',', skip_header=1, usecols=(2,), dtype=str)
summary_fma = np.genfromtxt(summary_file, delimiter=',', skip_header=1, usecols=(3,), dtype=int)

dir_path = '2019ADLs'
base_csv_name = 'xsens-000_'

sensor_locations = {
    'Right Wrist': '00B4249F',
    'Left Wrist': '00B424B0',
    'Sternum': '00B424B4',
}

fs = 100
dx = 1 / fs
f_lp = 8
# f_bp_l = 0.1
# f_bp_h = 8
f_bp_l = 0.1
f_bp_h = 8
b_lp, a_lp = butter(4, (f_lp / (fs/2)), btype='lowpass')
b_bp, a_bp = butter(2, (f_bp_l /(fs/ 2), f_bp_h / (fs / 2)), btype='bandpass')
#butter(2, (f_bp_l /(fs/ 2), f_bp_h / (fs / 2)), btype='bandpass')

subject_id = np.unique(activity_subject)
print(subject_id)


def correct_packet_counts(p):
    """Packet counts roll over, and are p' = p mod 2^16. Correct this by removing the modulo/overflow."""
    overflow_val = 1 << 16  # 65536
    overflow_count = 0

    p_corrected = np.copy(p)
    for i in range(1, p.shape[0]):
        if p[i] < p[i - 1]:
            overflow_count += 1
        p_corrected[i] = p[i] + (overflow_count * overflow_val)

    return p_corrected


def extract_data_from_csv(file):
    free_acc = np.genfromtxt(file, delimiter=',', skip_header=6, usecols=(13, 14, 15))
    orient = np.genfromtxt(file, delimiter=',', skip_header=6, usecols=(22, 23, 24, 25, 26, 27, 28, 29, 30))
    packet_counter = np.genfromtxt(file, delimiter=',', skip_header=6, usecols=(0,), dtype='int64')
    return free_acc, orient, correct_packet_counts(packet_counter)

sub_list_3d = []
reachvel_list_3d = []
reachacc_list_3d = []
aff_list_3d = []
for sub in [22, 32]:
    if summary_affected[summary_subject == sub] == 'R':
        affected_sensor = 'Right Wrist'
        unaffected_sensor = 'Left Wrist'
    else:
        affected_sensor = 'Left Wrist'
        unaffected_sensor = 'Right Wrist'

    # sternum_filename = os.path.join(dir_path, str(sub), base_csv_name + sensor_locations['Sternum'] + '.csv')
    affected_filename = os.path.join(dir_path, str(sub), base_csv_name + sensor_locations[affected_sensor] + '.csv')
    unaffected_filename = os.path.join(dir_path, str(sub), base_csv_name + sensor_locations[unaffected_sensor] + '.csv')

    # sternum_acc, sternum_orient, sternum_count = extract_data_from_csv(sternum_filename)
    affected_acc, affected_orient, affected_count = extract_data_from_csv(affected_filename)
    unaffected_acc, unaffected_orient, unaffected_count = extract_data_from_csv(unaffected_filename)

    a_begin = activity_begin[activity_subject == sub]
    a_end = activity_end[activity_subject == sub]
    a_label = activity_label[activity_subject == sub]
    a_emphasis = activity_emphasis[activity_subject == sub]

    print(sub)
    print(affected_sensor)
    print(len(a_begin))
    if sub == 22:
        reach_mvmt = [[[40, 103], [43, 134], [809, 902], [930, 1053], [1045, 1138]], None, [[0, 45]], None,
                      [[115, 199], [117, 224], [621, 710], [767, 856], [874, 941]], None, [[0, 81]], None]
        aff_mask = [[False, True, True, True, False], None, [True], None,
                    [True, False, False, False, True, True], None, [True], None]
        unaff_mask = [[True, False, False, False, True], None, [True], None,
                      [False, True, True, False, False], None, [True], None]
    elif sub == 32:
        reach_mvmt = [[[164, 230], [241, 332], [1050, 1176], [1176, 1268], [1409, 1504]], [[355, 419], [168, 237]], None,
                      [[165, 303], [1946, 2088], [2088, 2203], [2916, 3139], [3139, 3328], [4545, 4703]], [[165, 429], [190, 416]], None]
        aff_mask = [[False, True, False, False, False],  [True, False], None,
                    [True, False, False, True, True, True], [True, False], None]
        unaff_mask = [[True, False, True, True, True], [False, True], None,
                      [False, True, True, False, False, False],  [False, True], None]

    for idx in range(len(a_begin)):
        acc_seg_affected = affected_acc[a_begin[idx]-100:a_end[idx]+100]
        accfilt_seg_affected = filtfilt(b_lp, a_lp, acc_seg_affected, axis=0)
        accfilt_seg_affected = accfilt_seg_affected - accfilt_seg_affected.mean()
        vel_seg_affected = cumtrapz(accfilt_seg_affected, dx=dx, axis=0, initial=0)
        velfilt_seg_affected = filtfilt(b_bp, a_bp, vel_seg_affected, axis=0)

        acc_seg_unaffected = unaffected_acc[a_begin[idx]-100:a_end[idx]+100]
        accfilt_seg_unaffected = filtfilt(b_lp, a_lp, acc_seg_unaffected, axis=0)
        accfilt_seg_unaffected = accfilt_seg_unaffected - accfilt_seg_unaffected.mean()
        vel_seg_unaffected = cumtrapz(accfilt_seg_unaffected, dx=dx, axis=0, initial=0)
        velfilt_seg_unaffected = filtfilt(b_bp, a_bp, vel_seg_unaffected, axis=0)

        velfilt_mag_affected = np.sqrt(velfilt_seg_affected[:, 0] ** 2 + velfilt_seg_affected[:, 1] ** 2)
        velfilt_mag_unaffected = np.sqrt(velfilt_seg_unaffected[:, 0] ** 2 + velfilt_seg_unaffected[:, 1] ** 2)

        acc_mag_affected = np.sqrt(acc_seg_affected[:, 0] ** 2 + acc_seg_affected[:, 1] ** 2)
        acc_mag_unaffected = np.sqrt(acc_seg_unaffected[:, 0] ** 2 + acc_seg_unaffected[:, 1] ** 2)

        print(a_label[idx])
        print(a_emphasis[idx])
        print(a_end[idx] - a_begin[idx])
        plt.figure(figsize=(20, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(velfilt_mag_affected, label='x,y velMag')
        # ax1.plot(velfilt_seg_affected[:, 0])
        # ax1.plot(velfilt_seg_affected[:, 1])
        ax1.plot(velfilt_seg_affected[:, 2], label='z vel')
        ax1.set_title('affected')
        ax1.axhline(0, c='k')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(velfilt_mag_unaffected, label='x,y velMag')
        # ax2.plot(velfilt_seg_unaffected[:, 0])
        # ax2.plot(velfilt_seg_unaffected[:, 1])
        ax2.plot(velfilt_seg_unaffected[:, 2], label='z vel')
        ax2.set_title('unaffected')
        ax2.axhline(0, c='k')


        if reach_mvmt[idx] is not None:
            for idx_r in range(len(reach_mvmt[idx])):
                rm_i = reach_mvmt[idx][idx_r][0]
                rm_e = reach_mvmt[idx][idx_r][1] + 1
                print(rm_i, rm_e)
                if aff_mask[idx][idx_r]:
                    # ax1.axvline(rm_i, c='r', linestyle='--')
                    # ax1.axvline(rm_e, c='r', linestyle='--')
                    sub_list_3d.append(sub)
                    reachvel_list_3d.append(velfilt_mag_affected[rm_i:rm_e])
                    reachacc_list_3d.append(acc_mag_affected[rm_i:rm_e])
                    aff_list_3d.append(1)
                if unaff_mask[idx][idx_r]:
                    # ax2.axvline(rm_i, c='r', linestyle='--')
                    # ax2.axvline(rm_e, c='r', linestyle='--')
                    sub_list_3d.append(sub)
                    reachvel_list_3d.append(velfilt_mag_unaffected[rm_i:rm_e])
                    reachacc_list_3d.append(acc_mag_unaffected[rm_i:rm_e])
                    aff_list_3d.append(0)

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
        plt.tight_layout()
        plt.savefig(str(sub) + "_" + a_label[idx] + "_" +  a_emphasis[idx] + ".png")

        # subject 10 : bottle Y 1358 ~ 1426 (bottle up) / 1568 ~ 1640


num_subjects_2d = 20
exclude_sub_num_2d = 13

# list subject ID
subject_id_2d = np.arange(0, num_subjects_2d) + 1
subject_id_2d = np.delete(subject_id_2d, np.argwhere(subject_id_2d == exclude_sub_num_2d))

save_data_file_2d = open('sub_data_filt_ok.p', 'rb')
sub_data_2d = pickle.load(save_data_file_2d)


# feature extraction
features = []
compensation_labels = []
all_sub = []
reach_retract_mask = []
for sub_id in subject_id_2d:
    velfilt_affect_fore_reach = sub_data_2d[sub_id].velfilt_affect_fore_reach
    # velfilt_affect_fore_retract = sub_data_2d[sub_id].velfilt_affect_fore_retract
    free_acc_affect_fore_reach = sub_data_2d[sub_id].free_acc_affect_fore_reach
    # free_acc_affect_fore_retract = sub_data_2d[sub_id].free_acc_affect_fore_retract
    for ii in range(len(velfilt_affect_fore_reach)):
        mvel_xy_reach = magnitude_xy(velfilt_affect_fore_reach[ii])
        macc_xy_reach = magnitude_xy(free_acc_affect_fore_reach[ii])
        feat, feature_names = extract_features(mvel_xy_reach, None, fs=fs, prefix='velfilt_')
        feat_acc, feature_names_acc = extract_features(macc_xy_reach, None, z=True, fs=fs, prefix='acc_')

        features.append(feat + feat_acc)
        all_sub.append(sub_id)

    for ll in range(len(sub_data_2d[sub_id].reach_fas_score)):
        if sub_data_2d[sub_id].reach_fas_score[ll] == 5: #or sub_data[sub_id].reach_fas_score[ll] == 4:
            sub_data_2d[sub_id].reach_fas_score[ll] = 1
        else: # elif sub_data[sub_id].reach_fas_score[ll] != 5: #or sub_data[sub_id].reach_comp_score[ll] == 2:
            sub_data_2d[sub_id].reach_fas_score[ll] = 0

    compensation_labels.extend(sub_data_2d[sub_id].reach_fas_score)

compensation_labels = np.asarray(compensation_labels).reshape(-1)
features = np.asarray(features)
print(features.shape)
feature_names = np.asarray(feature_names + feature_names_acc)
all_sub = np.asarray(all_sub)


pca = PCA(n_components=2).fit(features)
pca_feats = pca.transform(features)
pca_add_features = np.concatenate((features, pca_feats), axis=1)
feature_names = np.append(feature_names, 'pca0')
feature_names = np.append(feature_names, 'pca1')
print(feature_names.shape, pca_add_features.shape)


print('the number of features', len(feature_names))
print('abnormal', len(np.where(compensation_labels == 0)[0]), 'normal', len(np.where(compensation_labels == 1)[0]),
      'total', len(compensation_labels))
# train with 2d
kernel = 1.0 * RBF(1.0)
model = GaussianProcessClassifier(kernel=kernel, random_state=0, n_restarts_optimizer=10, max_iter_predict=100)
model.fit(pca_add_features, compensation_labels)


sub_list_3d = np.asarray(sub_list_3d)
reachvel_list_3d = np.asarray(reachvel_list_3d)
reachacc_list_3d = np.asarray(reachacc_list_3d)
aff_list_3d = np.asarray(aff_list_3d)

predicted = np.zeros(len(reachvel_list_3d))
predicted_testscore = np.zeros(len(reachvel_list_3d))

for tt in range(len(reachvel_list_3d)):
    test_velfeatures, _ = extract_features(reachvel_list_3d[tt], None, fs=fs, prefix='3dvelfilt_')
    test_accfeatures, _ = extract_features(reachacc_list_3d[tt], None, z=True, fs=fs, prefix='3dacc_')

    test_features = test_velfeatures + test_accfeatures
    test_features = np.asarray(test_features)
    test_features = np.reshape(test_features, (1, -1))

    pca_test_feats = pca.transform(test_features)
    pcaadd_test_features = np.concatenate((test_features, pca_test_feats), axis=1)
    predicted[tt] = model.predict(pcaadd_test_features)
    predicted_testscore[tt] = model.predict_proba(pcaadd_test_features)[:, 1]

print(predicted[sub_list_3d == 22], predicted_testscore[sub_list_3d == 22])
print(predicted[np.logical_and(sub_list_3d == 22, aff_list_3d ==1)], predicted_testscore[np.logical_and(sub_list_3d == 22, aff_list_3d ==1)])
print(predicted[sub_list_3d == 32], predicted_testscore[sub_list_3d == 32])
print(predicted[np.logical_and(sub_list_3d == 32, aff_list_3d ==1)], predicted_testscore[np.logical_and(sub_list_3d == 32, aff_list_3d ==1)])
