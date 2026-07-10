import os
import sys
import wfdb
import tqdm
import shutil
import pathlib as pl
import numpy as np

file_path = str(pl.Path(os.path.dirname(
    os.path.realpath(__file__))).joinpath('..').resolve())
sys.path.append(file_path)

from project import mitdb_path, transfer_data_path
from scripts.preprocess_challenge2020 import resample

aux_to_snomed = \
    [['(AB', 'Atrial bigeminy', [251173003]],
     ['(AFIB', 'Atrial fibrillation', [164889003]],
     ['(AFL', 'Atrial flutter', [164890007]],
     ['(B', 'Ventricular bigeminy', [11157007]],
     ['(BII', '2° heart block', [195042002]],
     ['(IVR', 'Idioventricular rhythm', [49260003]],
     ['(N', 'Normal sinus rhythm', [426783006]],
     ['(NOD', 'Nodal (A-V junctional) rhythm', [29320008]],
     ['(P', 'Paced rhythm', [10370003]],
     ['(PREX', 'Pre-excitation (WPW)', [74390002]],
     ['(SBR', 'Sinus bradycardia', [426177001]],
     ['(SVTA', 'Supraventricular tachyarrhythmia', [426761007]],
     ['(T', 'Ventricular trigeminy', [251180001]],
     ['(VFL', 'Ventricular flutter', [111288001]],
     ['(VT', 'Ventricular tachycardia', [164895002]]]

beat_to_snomed = \
    [['N', 'Normal beat', [426783006]],  # Due to miss-leading challenge labels
     ['L', 'Left bundle branch block beat', [164909002]],
     ['R', 'Right bundle branch block beat', [713427006]],
     ['B', 'Bundle branch block beat (unspecified)', [164909002]],  # For simplicity left
     ['A', 'Atrial premature beat', [284470004]],
     ['a', 'Aberrated atrial premature beat', [284470004]],  # Aberrated, we do not care
     ['J', 'Nodal (junctional) premature beat', [29320008]],
     ['S', 'Supraventricular premature or ectopic beat (atrial or nodal)', [63593006]],
     ['V', 'Premature ventricular contraction', [17338001]],
     ['r', 'R-on-T premature ventricular contraction', [17338001]],  # R-on-T
     ['F', 'Fusion of ventricular and normal beat', [13640000]],
     ['e', 'Atrial escape beat', []],
     ['j', 'Nodal (junctional) escape beat', [426995002]],
     ['n', 'Supraventricular escape beat (atrial or nodal)', []],
     ['E', 'Ventricular escape beat', [75532003]],
     ['/', 'Paced beat', [10370003]],  # pacing rhythm
     ['f', 'Fusion of paced and normal beat', [10370003]],  # pacing rhythm
     ['Q', 'Unclassifiable beat', []],  # Ignored
     ['?', 'Beat not classified during learning', []],
     ['x', 'Non-conducted P-wave (blocked APB)', [251170000]],
     ['|', 'Isolated QRS-like artifact', []],  # Ignored
     ['~', 'Change in signal quality', []],  # Ignored
     ['!', 'Ventricular flutter wave', [111288001]],
     ['"', 'Comment annotation', []],
     ['[', 'Start of ventricular flutter/fibrillation', []],  # SPECIAL case (over time)
     [']', 'End of ventricular flutter/fibrillation', []]]  # SPECIAL case (over time)


# modified limb lead is set to Einthoven leads
lead_to_idx = [
    ['MLI', 0], ['MLII', 1], ['MLIII', 2],
    ['avR', 3], ['avL', 4], ['avF', 5],
    ['V1', 6], ['V2', 7], ['V3', 8],
    ['V4', 9], ['V5', 10], ['V6', 11]]

# Special case for "[" and "]" is ventricular flutter/fibrillation
special_snomed = 111288001


def symbol_to_snomed(symbol):
    for beat in beat_to_snomed:
        if beat[0] == symbol:
            return beat[-1]
    print("Attention symbol:", symbol, "can not be translated.")
    return []


def symbols_to_snomeds(symbols):
    snomeds = []
    for symbol in symbols:
        # "+" is only an indicator for aux
        if symbol == '+':
            continue
        snomeds.append(symbol_to_snomed(symbol))
    return np.unique(np.concatenate(snomeds).ravel()).astype(np.int64)


def rhythm_to_snomed(rhythm):
    for aux in aux_to_snomed:
        if aux[0] == rhythm:
            return aux[-1]
    return []


def rhythms_to_snomeds(rhythms):
    snomeds = []
    for rhythm in rhythms:
        snomeds.append(rhythm_to_snomed(rhythm))
    return np.unique(np.concatenate(snomeds).ravel()).astype(np.int64)


def meta_from_header(wfdb_record):
    comments = wfdb_record.comments[0]
    age, sex = comments.split(" ")[:2]
    age = np.int(age) if int(age) > 0 else np.nan
    sex = 0 if sex == 'F' else 1 if sex == 'M' else np.nan
    freq = np.int(wfdb_record.fs)

    return np.array([age, sex, int(freq), 11])


def get_rhythm(idx_samp_beg, idx_samp_end, rhythms):
    window_rhythm = []
    # Find last rhythm from end then break
    for i in range(idx_samp_end, idx_samp_beg, -1):
        if rhythms[i] != '':
            window_rhythm.append(rhythms[i].rstrip('\x00'))

    # Find last rhythm from begin
    for i in range(idx_samp_beg, -1, -1):
        if rhythms[i] != '':
            window_rhythm.append(rhythms[i].rstrip('\x00'))
            break

    # In case we do not find a rhythm add nomal
    if len(window_rhythm) == 0:
        window_rhythm.append('(N')

    return window_rhythm


def special_symbols_to_snomeds(idx_samp_beg, idx_samp_end, symbols):
    for i in range(idx_samp_end, idx_samp_beg, -1):
        if symbols[i] == "]":
            return [special_snomed]
        if symbols[i] == "[":
            return [special_snomed]

    for i in range(idx_samp_beg, -1, -1):
        if symbols[i] == "]":
            return []
        if symbols[i] == "[":
            return [special_snomed]
    return []


def two_lead_to_twelve_monkeys(data, sig_names):
    sig_idx = np.nan

    data_out = np.zeros((12, data.shape[-1]))
    for i, sig_name in enumerate(sig_names):
        for lead in lead_to_idx:
            if lead[0] == sig_name:
                sig_idx = lead[1]
        data_out[sig_idx, :] = data[i]
    return data_out


def main(output_directory=transfer_data_path):
    records = []

    # Create output folder for transfer learning
    shutil.rmtree(output_directory, ignore_errors=True)
    output_directory.mkdir(parents=True)

    for file in mitdb_path.iterdir():
        if file.name.split('.')[-1] == 'hea':
            records.append(file.name.split('.')[0])

    for record in tqdm.tqdm(records):
        # head = wfdb.rdheader(str(mitdb_path.joinpath(record)))
        # data = wfdb.rdsamp(str(mitdb_path.joinpath(record)))

        anno = wfdb.rdann(str(mitdb_path.joinpath(record)), 'atr')
        record = wfdb.rdrecord(str(mitdb_path.joinpath(record)))

        # Get age/sex and format to (nan = unknown, male = 0, female = 1)
        meta = meta_from_header(record)
        leads = record.sig_name

        # Read annotation files and map pathologies to signal
        sample, symbol, rhythm = anno.sample, np.array(anno.symbol), anno.aux_note

        # Get 2 Lead ecg from database
        data = np.swapaxes(record.p_signal, 0, 1)

        # Resample the signal
        target_fs = 500
        source_fs = meta[2]
        if source_fs != target_fs:
            # Resample the ecg
            data = resample(data, source_fs, target_fs)
            # Resample annotations
            sample = np.int64(sample * (int(target_fs) / int(source_fs)))
            # Update meta data
            meta[2] = 500

        # Split the data to a length of 2**12 + 2 ** 10 using no overlap
        sliced_data = []
        sliced_labels = []
        for i in range(0, data.shape[1], 2 ** 12 + 2 ** 10):
            i_begin = i
            i_end = i + 2 ** 12 + 2 ** 10

            idx_samples = np.where((sample >= i_begin) & (sample <= i_end))[0]
            window_symbols = symbol[idx_samples]
            window_rhythms = get_rhythm(idx_samples[0], idx_samples[-1], rhythm)
            window_special = special_symbols_to_snomeds(
                idx_samples[0], idx_samples[-1], symbol)

            snomed_window_symbols = symbols_to_snomeds(window_symbols)
            snomed_window_rhythms = rhythms_to_snomeds(window_rhythms)

            sliced_labels.append(list(np.unique(np.concatenate(
                [window_special, snomed_window_rhythms,
                 snomed_window_symbols]).ravel()).astype(np.int64)))

            sliced_data.append(data[:, i_begin:i_end])

        # Output data as 12 lead ecg, where only 2 leads are non-zero
        for i_slice in range(len(sliced_data)):
            name = "M{}_{:04d}".format(record.record_name, i_slice)
            out_data = two_lead_to_twelve_monkeys(
                sliced_data[i_slice], record.sig_name)
            out_labels = sliced_labels[i_slice]

            np.save(output_directory.joinpath(
                name + '_data.npy'), out_data)
            np.save(output_directory.joinpath(
                name + '_meta.npy'), meta)
            np.save(output_directory.joinpath(
                name + '_class.npy'), out_labels)

        # TODO: Do we want to use the data for something different?
        #  Idea: Train a very simple classifier to relabel ptb-xl?
        #  Idea: Train a complex based classifier instead a sequential?
        #  Idea: Train a model that is used in our ensemble?

    return


if __name__ == '__main__':
    main()
