import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import skew, kurtosis, entropy
from scipy.fftpack import fftfreq, fft
from utils import rms


def extract_features(ts, mft_score, fs=100):

    feature_names = []
    feat = []
    # find peaks
    peaks, _ = find_peaks(ts)
    prominences, *_ = peak_prominences(ts, peaks)
    # power spectrum
    ts_m = ts - ts.mean()
    fft_y = fft(ts_m, n=len(ts_m))
    freq = fftfreq(len(ts_m), d=1/fs)
    psd = (np.absolute(fft_y / len(ts_m))[:int(len(ts_m) / 2) + 1]) ** 2
    psd[1:-1] = 2*psd[1:-1]
    sortp = np.flip(np.argsort(psd))
    # hoff, homogeneous mean
    ts_idx = np.linspace(0, 1, len(ts))
    hoff = 30 * np.power(ts_idx, 4) - 60 * np.power(ts_idx, 3) + 30 * np.power(ts_idx, 2)
    homog_mean = np.pi / 2 * np.sin(np.pi * ts_idx)

    feat.append(len(ts))
    feature_names.append('dur')

    feat.append(len(peaks))
    feature_names.append('numpeaks')

    feat.append(len(peaks) / len(ts))
    feature_names.append('numpeaks/dur')

    feat.append(np.mean(ts))
    feature_names.append('mean')

    feat.append(np.mean(ts) / np.max(ts))
    feature_names.append('mean/peak')

    feat.append(np.log10(np.mean(ts) / np.max(ts)))
    feature_names.append('logmean/peak')

    feat.append(np.median(ts))
    feature_names.append('median')

    feat.append(np.max(ts) / len(ts))
    feature_names.append('max/dur')

    feat.append(1 / np.max(ts - ts.mean()))
    feature_names.append('1/peaknorm')

    feat.append(np.max(ts - ts.mean()))
    feature_names.append('peaknorm')

    feat.append(rms(hoff - ts))
    feature_names.append('rmshoff')

    feat.append(rms(homog_mean - ts))
    feature_names.append('rmshomog')

    if len(peaks) > 1:
        feat.append(np.diff(peaks).max())
        feat.append(np.diff(peaks).mean())
        feat.append(np.diff(peaks).std())
        feat.append(np.percentile(np.diff(peaks), 10))
        feat.append(np.percentile(np.diff(peaks), 50))
        feat.append(np.percentile(np.diff(peaks), 90))
        feat.append(prominences.max())
        feat.append(prominences.mean())
        feat.append(prominences.std())
        feat.append(np.percentile(prominences, 10))
        feat.append(np.percentile(prominences, 50))
        feat.append(np.percentile(prominences, 90))
    else:
        feat.extend([0.] * 12)

    feature_names.append('maxpeakdur')
    feature_names.append('meanpeakdur')
    feature_names.append('stdpeakdur')
    feature_names.append('10peakdur')
    feature_names.append('50peakdur')
    feature_names.append('90peaksdur')
    feature_names.append('maxpeakpromi')
    feature_names.append('meanpeakpromi')
    feature_names.append('stdpeakpromi')
    feature_names.append('10peakpromi')
    feature_names.append('50peakpromi')
    feature_names.append('90peakpromi')

    feat.append(skew(ts))
    feature_names.append('skew')

    feat.append(kurtosis(ts))
    feature_names.append('kurtosis')

    feat.append(np.var(ts))
    feature_names.append('var')

    feat.append(entropy(ts))
    feature_names.append('entropy')

    feat.append(sortp[0])
    feature_names.append('loc1stfreq')

    feat.append(sortp[1])
    feature_names.append('loc2stfreq')

    feat.append(freq[sortp[0]])
    feature_names.append('1stfreq')

    feat.append(freq[sortp[1]])
    feature_names.append('2ndfreq')

    feat.append(freq[sortp[0]] / freq[sortp[1]])
    feature_names.append('1stfreq/2ndfreq')

    feat.append(np.max(psd) / np.sum(psd))
    feature_names.append('maxpsd/sumpsd')

    feat.append(np.sum(psd) / np.sum(psd[sortp[1:]]))
    feature_names.append('sumpsd/sumf2ndpsd')

    feat.append(psd[sortp[0]])
    feature_names.append('1stpsd')

    feat.append(psd[sortp[1]])
    feature_names.append('2ndpsd')

    feat.append(len(freq))
    feature_names.append('numfreq')

    feat.append(np.var(freq))
    feature_names.append('varfreq')

    feat.append(mft_score)
    feature_names.append('mftscore')

    assert len(feat) == len(feature_names)

    return feat, feature_names


