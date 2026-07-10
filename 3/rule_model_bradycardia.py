# -*- coding: utf-8 -*-

import numpy as np, os, os.path, sys
from scipy.signal import butter, lfilter
from scipy import stats
from scipy.signal import spectrogram
from scipy.io import loadmat

# bandpass filter
def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

# fing r peak
def findpeaks(data, spacing=1, limit=None):
        len = data.size
        x = np.zeros(len + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len] = data
        peak_candidate = np.zeros(len)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before
            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
        if limit is not None:
            ind = ind[data[ind] > limit]
        return ind

# detect qrs
def detect_qrs(detected_peaks_values, detected_peaks_indices):
    refractory_period = 100  # Change proportionally when adjusting frequency (in samples).
    qrs_peak_filtering_factor = 0.125
    noise_peak_filtering_factor = 0.125
    qrs_noise_diff_weight = 0.25
    qrs_peak_value = 0.0
    noise_peak_value = 0.0
    threshold_value = 0.0


    # Detection results.
    qrs_peaks_indices = np.array([], dtype=int)
    noise_peaks_indices = np.array([], dtype=int)
    
    for detected_peak_index, detected_peaks_value in zip(detected_peaks_indices, detected_peaks_values):

        try:
            last_qrs_index = qrs_peaks_indices[-1]
        except IndexError:
            last_qrs_index = 0

        # After a valid QRS complex detection, there is a 200 ms refractory period before next one can be detected.
        if detected_peak_index - last_qrs_index > refractory_period or not qrs_peaks_indices.size:
            # Peak must be classified either as a noise peak or a QRS peak.
            # To be classified as a QRS peak it must exceed dynamically set threshold value.
            if detected_peaks_value > threshold_value:
                qrs_peaks_indices = np.append(qrs_peaks_indices, detected_peak_index)

                # Adjust QRS peak value used later for setting QRS-noise threshold.
                qrs_peak_value = qrs_peak_filtering_factor * detected_peaks_value + \
                                      (1 - qrs_peak_filtering_factor) * qrs_peak_value
            else:
                noise_peaks_indices = np.append(noise_peaks_indices, detected_peak_index)

                # Adjust noise peak value used later for setting QRS-noise threshold.
                noise_peak_value = noise_peak_filtering_factor * detected_peaks_value + \
                                        (1 - noise_peak_filtering_factor) * noise_peak_value

            # Adjust QRS-noise threshold value based on previously detected QRS or noise peaks value.
            threshold_value = noise_peak_value + \
                                   qrs_noise_diff_weight * (qrs_peak_value - noise_peak_value)

    # Create array containing both input ECG measurements data and QRS detection indication column.
    # We mark QRS detection with '1' flag in 'qrs_detected' log column ('0' otherwise).
    return qrs_peaks_indices

# 检测r peak
def detect_peaks(ecg_measurements,signal_frequency,gain, lead_ = 0, use_other_feas = False):

        filter_lowcut = 0.001
        filter_highcut = 15.0
        filter_order = 1
        integration_window = 30  # Change proportionally when adjusting frequency (in samples).
        findpeaks_limit = 0.35
        findpeaks_spacing = 100  # Change proportionally when adjusting frequency (in samples).

        # Measurements filtering - 0-15 Hz band pass filter.
        filtered_ecg_measurements = bandpass_filter(ecg_measurements, lowcut=filter_lowcut, highcut=filter_highcut, signal_freq=signal_frequency, filter_order=filter_order)

        filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]

        # Derivative - provides QRS slope information.
        differentiated_ecg_measurements = np.ediff1d(filtered_ecg_measurements)

        # Squaring - intensifies values received in derivative.
        squared_ecg_measurements = differentiated_ecg_measurements ** 2

        # Moving-window integration.
        integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window)/integration_window)

        # Fiducial mark - peak detection on integrated measurements.
        cur_findpeaks_limit = max(integrated_ecg_measurements)/10  # replace ori findpeaks_limit
        detected_peaks_indices = findpeaks(data=integrated_ecg_measurements,
                                                     limit=cur_findpeaks_limit,  # 0407 findpeaks_limit
                                                     spacing=findpeaks_spacing)

        detected_peaks_values = integrated_ecg_measurements[detected_peaks_indices]
        
        qrs_peaks_indices = detect_qrs(detected_peaks_values, detected_peaks_indices)
        
        qrs_peaks_values = integrated_ecg_measurements[qrs_peaks_indices]
        
        # return detected_peaks_values, detected_peaks_indices
        return qrs_peaks_values, qrs_peaks_indices
    
def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data
    
def get_RR_interval(data, header_data):
    # data, header_data
    tmp_hea = header_data[0].split(' ')
    tmp_hea1 = header_data[1].split(' ')
    sample_Fs = int(tmp_hea[2])
    gain = int(tmp_hea1[2].split('/')[0])
    
    peaks,idx = detect_peaks(data[0],sample_Fs,gain)
    RR = np.array(np.ediff1d(idx))
    if sample_Fs == 1000:
        RR = np.array([term/2 for term in RR])
    return RR

def judge_bradycardia(data, header_data):
    RR = get_RR_interval(data, header_data)
    bin_RR = ((RR> 500)&(RR< 800))+0
    mean_RR = 0 if len(bin_RR)<6 else bin_RR.mean()

    if mean_RR > 0.5:
        return 1
    elif mean_RR < 0.5:
        return 0
    else:
        return 'error'
    
# test 
'''
input_file =  'D:/pingan/比赛/2020KDDPhysioNet心电/se_resnet/test/Q0060.mat'
predict_bradycardia = judge_bradycardia(input_file) 
'''