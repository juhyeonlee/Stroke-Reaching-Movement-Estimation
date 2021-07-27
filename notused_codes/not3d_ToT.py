 # General
import numpy as np
import os
import re
import pandas as pd

#Data Processing
from scipy.integrate import cumtrapz, trapz
from scipy.signal import butter, filtfilt, resample,savgol_filter
from scipy.stats.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def integrate(acc_data):
    fc_lp = 8
    fc_hp = 0.1
    b1, a1 = butter(6, fc_lp/(sample_freq/2), btype='lowpass')
    b2, a2 = butter(2, [fc_hp/(sample_freq/2), fc_lp/(sample_freq/2)], btype='bandpass')
    filt_acc = filtfilt(b1, a1, acc_data, axis=0)
    filt_acc = filt_acc - filt_acc.mean(axis=0)
    vel = cumtrapz(filt_acc, dx=dx, axis=0)
    vel = filtfilt(b2, a2, vel, axis=0)
    vel = vel - vel.mean(axis=0)
    return vel

sample_freq = 100
dx = 1 / sample_freq

num_subjects = 20
exclude_sub_num = [1, 13]

# get file paths
dir_path = './NER17/data'

dir_path2 = './reaching_movement_data'
csv_index_file = 'indices_labels.csv'

# list subject ID
subject_id = np.arange(0, num_subjects) + 1
for e in exclude_sub_num:
    subject_id = np.delete(subject_id, np.argwhere(subject_id == e))
print(len(subject_id))

noise_sub = [5, 6, 11, 15, 16]
noise_idx = [31933, 33149, 32336, 42234, 33899]

for sub in subject_id:
    data_path = os.path.join(dir_path, str(sub)+".csv")
    label_path = os.path.join(dir_path, str(sub)+"_label.csv")
    data = pd.read_csv(data_path, header = 5, usecols = ["PacketCounter","FreeAcc_X","FreeAcc_Y","FreeAcc_Z"])
    label_data = pd.read_csv(label_path, usecols = ["reaching_begin","reaching_end","retracting_begin","retracting_end"])

    indices_data = np.genfromtxt(os.path.join(dir_path2, str(sub), csv_index_file),
                                 delimiter=',', skip_header=1, dtype=int)


    accx = np.array(data["FreeAcc_X"].tolist())
    accy = np.array(data["FreeAcc_Y"].tolist())
    accz = np.array(data["FreeAcc_Z"].tolist())
    # print(accx.size, accy.size, accz.size)

    # interpolate noise value
    if sub in noise_sub:
        idx = np.argwhere(noise_sub == sub)[0][0]
        accx[noise_idx[idx]] = (accx[noise_idx[idx]-1] + accx[noise_idx[idx]+1]) / 2
        accy[noise_idx[idx]] = (accy[noise_idx[idx]-1] + accy[noise_idx[idx]+1]) / 2
        accz[noise_idx[idx]] = (accz[noise_idx[idx]-1] + accz[noise_idx[idx]+1]) / 2

    velx = integrate(tuple(accx))
    vely = integrate(tuple(accy))
    velz = integrate(tuple(accz))

    velx = velx[:]
    vely = vely[:]
    velz = velz[:]
    velmag_xy = (velx ** 2 + vely ** 2 ) ** (0.5)
    # print(label_data)
    rec_beg = label_data["reaching_begin"].tolist()
    rec_end = label_data["reaching_end"].tolist()
    ret_beg = label_data["retracting_begin"].tolist()
    ret_end = label_data["retracting_end"].tolist()

    for i in range(len(rec_beg)):
        if rec_beg[i] != indices_data[i, 0]:
            print(sub, "!!")
        if rec_end[i] != indices_data[i, 1]:
            print(sub, "!!")
        if ret_beg[i] != indices_data[i, 2]:
            print(sub, "!!")
        if ret_end[i] != indices_data[i, 3]:
            print(sub, "!!")

    # plt.figure()
    # # plt.plot(accz)
    # plt.plot(velmag_xy)
    # for i in range(len(rec_beg)):
    #     plt.plot(np.arange(rec_beg[i], rec_end[i]), velmag_xy[rec_beg[i]:rec_end[i]], color='r')
    #     plt.plot(np.arange(ret_beg[i], ret_end[i]), velmag_xy[ret_beg[i]:ret_end[i]], color='m')
    # plt.title(str(sub))
    # plt.show()

