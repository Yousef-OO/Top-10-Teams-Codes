""" This file does the preprocessing (e.g. petersburg preprocessing or resampling)
    and stores them in the designated folder as npy files that are used during traing
"""
import os
import wfdb
import tqdm
import shutil
import argparse
import numpy as np

from scipy.io import loadmat
from scipy import signal as sig

import sys
import pathlib as pl
file_path = str(pl.Path(os.path.dirname(
    os.path.realpath(__file__))).joinpath('..').resolve())
sys.path.append(file_path)

from project import incart_path, temp_path, complete_raw_data_path


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data


def get_meta_from_header(header):
    """
    Reads and returns diagnoses, age, sex, sampling freqency and amplitude
    resolution from a given header.

    Arguments
    ---------
    header: list of strings
        The lines from a .hae header file.

    Returns
    -------
    age: int
        Age of the subject.
    gender: int
        The subject's gender, where 0 represents male and 1 represents female.
    freq: int
        The sampling frequency of the record (Hz).
    resolution:
        The amplitude resolution of the record (units/mV).
    classes: list of ints
        List of diagnoses as SNOMED-CT codes.

    """
    classes = []
    age = np.nan
    gender = np.nan
    for line in header:
        if line.startswith('#Dx'):
            tmp = line.split(': ')[1].split(',')
            for c in tmp:
                classes.append(int(c.strip()))
        if line.startswith('#Age'):
            tmp = line.split(': ')[1].strip()
            if tmp != 'NaN':
                age = np.int(tmp)
        if line.startswith('#Sex'):
            tmp = line.split(': ')[1].strip()
            if tmp == 'Male':
                gender = 0
            elif tmp == 'Female':
                gender = 1

    freq = int(header[0].split(' ')[2])
    resolution = int(header[1].split(' ')[2].split('/')[0])

    return age, gender, freq, resolution, classes


def get_all_files(input_path):
    input_files = []
    for entry in input_path.iterdir():
        if entry.name.endswith('mat'):
            input_files.append(entry.name)
    return input_files


def resample(data, source_fs, target_fs):
    """ Resample input ecg with source frequency to target
    frequency using Fourier method along the given axis t.

    Args:
        data (numpy array): 12 lead ecg with dimensions leads X time.
        source_fs: The sampling frequency of the source signal
        target_fs: The sampling frequency of the target signal

    Example:
        resampled_data = resample(np.empty((12, 10000), dtype=np.float64), 250, 500)
        >> resapmled_data.shape != (12, 20000)
    """
    upsampling_factor = np.float(np.float(target_fs) / np.float(source_fs))

    out_length = int(data.shape[1] * upsampling_factor)
    out_data = np.empty((data.shape[0], out_length), dtype=data.dtype)

    for i_lead in range(0, data.shape[0]):
        out_data[i_lead, :] = sig.resample(data[i_lead, :], out_length)

    return out_data


def compute_hr(annotations):
    rr = [annotations[i + 1] - annotations[i] for i in range(0, len(annotations) - 1)]
    hr = np.mean(60 / (np.array(rr) / 500.))
    return hr


def get_diagnose_from_wfdb(name, origin=False, incartdb_path=incart_path):
    if origin:
        head_original = wfdb.rdheader(name, pn_dir='incartdb')
    else:
        head_original = wfdb.rdheader(str(incartdb_path.joinpath(name)))
    # Find diagnoses from free - text annoations
    infos = "___".join(head_original.__dict__['comments']).\
                split("<sex>:")[1][2:].replace("<diagnoses>", "").strip().split("___")

    patient = ""
    diagnose = []
    for info in infos:
        if info[:len("patient")] == "patient":
            patient = info
            continue
        if len(info) == 0:
            continue
        diagnose.append(info)
    free_text = [i.strip() for i in ",".join(diagnose).replace("and", ",").split(",")]
    return free_text, patient


def petersburg_split_and_relabel(name, ecg_500Hz, challenge_labels,
                                 origin=False, incartdb_path=incart_path):
    ecgs, labels = [], []

    # Load physionet labels
    physionet_name = "I" + name.replace('.hea', '')[-2:]
    physionet_labels, _ = get_diagnose_from_wfdb(physionet_name)

    # Load physionet annotations
    if origin:
        annotation = wfdb.rdann(physionet_name, 'atr', pn_dir='incartdb')
    else:
        annotation = wfdb.rdann(str(incartdb_path.joinpath(physionet_name)), 'atr')

    # Resample annotations to 500 Hz
    samples = (annotation.sample / np.float(annotation.fs)) * 500.
    symbols = np.array(annotation.symbol)

    # Split ecg and annotations (each a length of 2 ** 13 + 2 ** 11 = 10240)
    for i in range(0, ecg_500Hz.shape[1], 10240):
        beg = i
        end = i + 10240 if i < ecg_500Hz.shape[1] else ecg_500Hz.shape[1]

        ecg_slice = ecg_500Hz[:, beg:end]
        sl = (samples > float(beg)) & (samples < float(end))
        ann_slice_samples = samples[sl]
        ann_slice_symbols = symbols[sl]

        slow_fast_rhythms = [164889003, 164890007, 426627000, 427084000, 426177001]

        label = []
        # Everything looks normal
        if len(np.unique(ann_slice_symbols)) == 1 and \
                np.unique(ann_slice_symbols)[0] == "N":
            # Add normal class (sinus-rhythm and normal are merged)
            label.append(426783006)
        # Otherwise add the labels from the challenge
        # (except tachycardia/bradycardia/etc..)
        else:
            challenge_labels_snomeds = set(challenge_labels)
            for sfr in slow_fast_rhythms:
                challenge_labels_snomeds.discard(sfr)
            for lbl in challenge_labels_snomeds:
                label.append(lbl)

        # Special Case: Check if tachycardia / bradicardia / flutter / fibrilation
        # Apply a simple rule based mechanism for rule based rhythms
        # tachycardia >= 120 bpm
        # bradycardia <= 50 bpbm

        challenge_labels_snomeds = challenge_labels

        if not len(np.intersect1d(slow_fast_rhythms, challenge_labels_snomeds)):
            # We only add brady / tachy / flutter / fibrilation to the labels
            labels.append(label)
            ecgs.append(ecg_slice)
            continue

        # atrial fibrilations or flutter
        if 164889003 in challenge_labels_snomeds:
            label.append(164889003)
        if 164890007 in challenge_labels_snomeds:
            label.append(164890007)

            # No special rule, all sliced records contain afibs if label is present
            # print("Atrial Fibriations / Flutter",
            #      label,
            #      get_entropy(ann_slice_samples),
            #      get_rmssd(ann_slice_samples),
            #      get_entropy(ann_slice_samples))

        # Bradycardia
        if 426627000 in challenge_labels_snomeds:
            # Brady: hr < 60 bpm
            hr = compute_hr(ann_slice_samples)
            if hr < 60:
                label.append(426627000)

        if 426177001 in challenge_labels_snomeds:
            # Brady: hr < 60 bpm
            hr = compute_hr(ann_slice_samples)
            if hr < 60:
                label.append(426177001)

        # Tachycardia
        if 427084000 in challenge_labels_snomeds:
            hr = compute_hr(ann_slice_samples)
            # Tachy: hr > 100 bpm
            if hr > 100.:
                label.append(427084000)

        labels.append(label)
        ecgs.append(ecg_slice)

    return ecgs, labels


def main(physionet_archive_path=temp_path, processed_data_path=complete_raw_data_path):
    print("Preprocess files from:", str(physionet_archive_path), "to:",
          str(processed_data_path.resolve().absolute()))

    # Get all files in training data
    input_files = get_all_files(physionet_archive_path)
    print("Preprocessing", str(len(input_files)), "records.")

    # If path exists, overwrite them
    shutil.rmtree(processed_data_path, ignore_errors=True)
    processed_data_path.mkdir(parents=True)

    # Iterate over all input files and preprocess data and format labels
    for i, input_file in tqdm.tqdm(enumerate(input_files)):
        data, header = load_challenge_data(
            os.path.join(str(physionet_archive_path), input_file))
        age, gender, freq, resolution, classes = get_meta_from_header(header)
        meta = np.array([age, gender, freq, resolution])
        snomed = np.array(classes)

        # Resample the data to 500 Hz
        target_frequency = np.int(500)
        source_frequency = np.int(meta[2])
        if source_frequency != target_frequency:
            data = resample(data, source_frequency, target_frequency)
            meta[2] = 500

        name = input_file.split('.')[0]

        # Simple way to identify the st petersburg dataset
        if name[0] == 'I':
            ecgs, lbls = petersburg_split_and_relabel(name, data, snomed)
            for j in range(0, len(ecgs)):
                # Attention this is generating multiple records from one
                # record (segmented parts)
                np.save(processed_data_path.joinpath(
                    name + "-" + str(j) + '_data.npy'), ecgs[j])
                np.save(processed_data_path.joinpath(
                    name + "-" + str(j) + '_meta.npy'), meta)
                np.save(processed_data_path.joinpath(
                    name + "-" + str(j) + '_class.npy'), lbls[j])
        # Just store the original data
        else:
            np.save(processed_data_path.joinpath(name + '_data.npy'), data)
            np.save(processed_data_path.joinpath(name + '_meta.npy'), meta)
            np.save(processed_data_path.joinpath(name + '_class.npy'), snomed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Environment definitions for different platforms")

    parser.add_argument('--physionet_archive_path', type=str,
                        default=str(temp_path))
    parser.add_argument('--processed_data_path', type=str,
                        default=str(complete_raw_data_path))

    args = parser.parse_args()
    main(physionet_archive_path=pl.Path(args.physionet_archive_path),
         processed_data_path=pl.Path(args.processed_data_path))
