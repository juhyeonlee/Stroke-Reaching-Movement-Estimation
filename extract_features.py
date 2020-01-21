import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import skew, kurtosis, entropy, iqr, median_absolute_deviation
from scipy.fftpack import fftfreq, fft
from utils import rms


def extract_features(ts, mft_score, fs=100, prefix='n'):

    feature_names = []
    feat = []
    # find peaks
    peaks, _ = find_peaks(ts)
    mag_peaks_sor = np.flip(np.sort(ts[peaks]))
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

    # feat.append(float(len(ts)))
    # feature_names.append(prefix + 'dur')

    feat.append(float(np.argmax(ts)))
    feature_names.append(prefix + 'peakloc')

    feat.append(float(np.argmax(ts))/len(ts))
    feature_names.append(prefix + 'peakloc/dur')

    feat.append(float(len(peaks)))
    feature_names.append(prefix + 'numpeaks')

    # feat.append(float(len(peaks)) / float(len(ts)))
    # feature_names.append(prefix + 'numpeaks/dur')

    feat.append(np.mean(ts))
    feature_names.append(prefix + 'mean')

    feat.append(np.mean(ts) / np.max(ts))
    feature_names.append(prefix + 'mean/peak')

    feat.append(np.log10(np.mean(ts) / np.max(ts)))
    feature_names.append(prefix + 'logmean/peak')

    feat.append(np.median(ts))
    feature_names.append(prefix + 'median')

    feat.append(np.std(ts) / np.mean(ts))
    feature_names.append(prefix + 'normstd')

    feat.append(np.max(ts) / len(ts))
    feature_names.append(prefix + 'max/dur')

    feat.append(1 / np.max(ts - ts.mean()))
    feature_names.append(prefix + '1/peaknorm')

    feat.append(np.max(ts - ts.mean()))
    feature_names.append(prefix + 'peaknorm')

    feat.append(rms(hoff - ts))
    feature_names.append(prefix + 'rmshoff')

    feat.append(rms(homog_mean - ts))
    feature_names.append(prefix + 'rmshomog')

    feat.append(np.std(np.diff(ts)) / np.mean(np.diff(ts)))
    feature_names.append(prefix + 'smoothness')

    # feat.append(np.std(np.mean(ts_m ** 2)))
    # feature_names.append('rms')

    feat.append(iqr(ts_m))
    feature_names.append(prefix + 'iqr')

    feat.append(median_absolute_deviation(ts_m))
    feature_names.append(prefix + 'mad')

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

    feature_names.append(prefix + 'maxpeakdur')
    feature_names.append(prefix + 'meanpeakdur')
    feature_names.append(prefix + 'stdpeakdur')
    feature_names.append(prefix + '10peakdur')
    feature_names.append(prefix + '50peakdur')
    feature_names.append(prefix + '90peaksdur')
    feature_names.append(prefix + 'maxpeakpromi')
    feature_names.append(prefix + 'meanpeakpromi')
    feature_names.append(prefix + 'stdpeakpromi')
    feature_names.append(prefix + '10peakpromi')
    feature_names.append(prefix + '50peakpromi')
    feature_names.append(prefix + '90peakpromi')

    feat.append(skew(ts))
    feature_names.append(prefix + 'skew')

    feat.append(kurtosis(ts))
    feature_names.append(prefix + 'kurtosis')

    feat.append(np.var(ts))
    feature_names.append(prefix + 'var')

    feat.append(entropy(ts))
    feature_names.append(prefix + 'entropy')

    feat.append(sortp[0])
    feature_names.append(prefix + 'loc1stfreq')

    feat.append(sortp[1])
    feature_names.append(prefix + 'loc2stfreq')

    feat.append(freq[sortp[0]])
    feature_names.append(prefix + '1stfreq')

    feat.append(freq[sortp[1]])
    feature_names.append(prefix + '2ndfreq')

    feat.append(freq[sortp[0]] / freq[sortp[1]])
    feature_names.append(prefix + '1stfreq/2ndfreq')

    # feat.append(np.max(psd) / np.sum(psd))
    # feature_names.append(prefix + 'maxpsd/sumpsd')

    feat.append(np.sum(psd) / np.sum(psd[sortp[1:]]))
    feature_names.append(prefix + 'sumpsd/sumf2ndpsd')

    feat.append(psd[sortp[0]])
    feature_names.append(prefix + '1stpsd')

    feat.append(psd[sortp[1]])
    feature_names.append(prefix + '2ndpsd')
    #
    # feat.append(len(freq))
    # feature_names.append(prefix + 'numfreq')

    feat.append(np.var(freq))
    feature_names.append(prefix + 'varfreq')

    # if len(peaks) >= 2:
    #     feat.append((mag_peaks_sor[0] - mag_peaks_sor[1])/np.max(ts))
    # elif len(peaks)  == 1:
    #     feat.append(mag_peaks_sor[0]/np.max(ts))
    # else:
    #     feat.append(0.0)
    #     print(len(peaks), 'nopeak')
    # feature_names.append(prefix+'peak1peak2magdiff')


    # feat.append(float(len(peaks)) / np.log(len(ts)))
    # feature_names.append(prefix + 'numpeaks/logdur')


    # if mft_score is not None:
    #     feat.append(float(mft_score))
    #     feature_names.append('mftscore')

    assert len(feat) == len(feature_names)

    return feat, feature_names


