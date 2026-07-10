"""
CinC/Physionet Challenge 2020, converts it to .npy files and saves a train,
validation and test split of these to a directory data/raw/ as follows:
data/raw/complete:      this directory contains the converted data from the
                        whole data set as .npy files. The original files are
                        converted into
                            - *_class.npy (class as SNOMED-CT code)
                            - *_data.npy (actual 12-lead ecg data)
                            - *_meta.npy (patient's age and gender)
                        files for each record.
data/raw/test/:         contains symlinks to class, data, and meta files
                        in data/raw/complete for each record that is part of
                        test data set
data/raw/train/:        contains symlinks to class, data, and meta files
                        in data/raw/complete for each record that is part of
                        train data set
data/raw/val:           contains symlinks to class, data, and meta files
                        in data/raw/complete for each record that is part of
                        validation data set
data/raw/dx_map.csv:    contains Multi-hot label index for each class,
                        belonging SNOMED-CT code, string representation and
                        their abbreviation

Usage:
$ ls
import_challenge2020.py
$ python import_challenge2020.py

You may need administration rights to execute this script successfully.
"""

import os
import shutil
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

import sys
import pathlib as pl
file_path = str(pl.Path(os.path.dirname(
    os.path.realpath(__file__))).joinpath('..').resolve())
sys.path.append(file_path)

from project import complete_raw_data_path, raw_data_path


def get_all_files(input_path):
    input_files = []
    #for entry in input_path.iterdir():
    for s in os.listdir(str(input_path)):
        entry = pl.Path(s)
        if entry.name.endswith('mat'):
            input_files.append(entry.name)
    return input_files


def make_train_val_test(files, train=0.8, val=0.1, test=0.1, n_releases=6):
    """
    Splits the given files in three subsets of size train, val and test such
    that the different data tranches are proportionally equally represented in
    each set.

    Arguments
    ---------
    files: list of strings
        The file names.
    train: float
        Proportion of the train set
        (default is 0.8)
    val: float
        Proportion of the validation set
        (default is 0.1)
    test: float
        Proportion of the test set
        (default is 0.1)

    Note: train, val and test must sum up to 1

    Returns
    -------
    train_files: list of strings
    val_files: list of strings
    test_files: list of strings
    """

    tranches = []
    for file in files:
        if file[0] not in tranches:
            tranches.append(file[0])
    if n_releases != len(tranches):
        raise Exception('Number of releases does not fit the number of determined '
                        'characters corresponding to releases: (',
                        str(len(tranches)), "/", str(n_releases), ")")
    print(tranches)

    tranch_files = {key: [] for key in tranches}
    for file in files:
        tranch_files[file[0]].append(file)

    train_files, val_files, test_files = [], [], []
    for key in tranches:
        train_f, val_f, test_f = split(tranch_files[key], train=train, val=val, test=test)
        train_files.extend(train_f)
        val_files.extend(val_f)
        test_files.extend(test_f)

    return train_files, val_files, test_files


def split(files, train=0.8, val=0.1, test=0.1):
    """
    Splits the given files in three subsets of size train, val and test.

    Arguments
    ---------
    files: list of strings
        The file names.
    train: float
        Proportion of the train set
        (default is 0.8)
    val: float
        Proportion of the validation set
        (default is 0.1)
    test: float
        Proportion of the test set
        (default is 0.1)

    Note: train, val and test must sum up to 1

    Returns
    -------
    files_train: list of strings
    files_val: list of strings
    files_test: list of strings
    """
    if train + test + val != 1.0:
        assert "Invalid test split"

    test_val_size = np.round(1. - train, 3)
    test_size = test / np.round(test + val, 3)

    files_train, files_test_val = train_test_split(files,
                                                   test_size=test_val_size,
                                                   random_state=42)
    files_test, files_val = train_test_split(files_test_val,
                                             test_size=test_size,
                                             random_state=42)

    return files_train, files_val, files_test


def main(input_directory=complete_raw_data_path, output_directory=raw_data_path):

    # Delete all symlinks and create train/val/test folders
    shutil.rmtree(output_directory.joinpath('train'), ignore_errors=True)
    shutil.rmtree(output_directory.joinpath('val'), ignore_errors=True)
    shutil.rmtree(output_directory.joinpath('test'), ignore_errors=True)

    # Create train/test/val split directories
    output_directory.joinpath('train').mkdir()
    output_directory.joinpath('val').mkdir()
    output_directory.joinpath('test').mkdir()

    records = []
#    for f in input_directory.iterdir():
    for s in os.listdir(str(input_directory)):
        f = pl.Path(os.path.join(str(input_directory), s))
        record, type = f.name.split('.npy')[0].split("_")
        if f.is_file() and type == "data":
            records.append(record)

    print("Total number of imported files:", len(records))

    train_files, val_files, test_files = make_train_val_test(
        records, train=0.9, val=0.09, test=0.01)

    print('Train: ', len(train_files),
          "Test:", len(test_files), "Val:", len(val_files))

    for i, record in enumerate(records):
        if record in train_files:
            dataset = 'train'
        elif record in val_files:
            dataset = 'val'
        elif record in test_files:
            dataset = 'test'
        else:
            raise ValueError('Invalid dataset for record', record)

        for ftype in ("class", "meta", "data"):
            src_path = input_directory.joinpath(record + "_{}.npy".format(ftype))
            dst_path = output_directory.joinpath(dataset).joinpath(
                record + "_{}.npy".format(ftype)
            )
            os.symlink(src_path.resolve().absolute(),
                       dst_path.resolve().absolute())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Environment definitions for different platforms")

    parser.add_argument('--complete_data_path', type=str,
                        default=str(complete_raw_data_path))
    parser.add_argument('--raw_data_path', type=str,
                        default=str(raw_data_path))

    args = parser.parse_args()
    main(input_directory=pl.Path(args.complete_data_path),
         output_directory=pl.Path(args.raw_data_path))
