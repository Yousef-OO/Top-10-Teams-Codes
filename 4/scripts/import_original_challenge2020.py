"""
Get the original .mat and .head for the validation and test set data and
copy them into the val and test folders.
"""

import os
import wfdb
import tqdm
import shutil
import argparse
import numpy as np
import pathlib as pl

from scipy.io import loadmat
from scipy import signal as sig

import sys
import pathlib as pl
file_path = str(pl.Path(os.path.dirname(
    os.path.realpath(__file__))).joinpath('..').resolve())
sys.path.append(file_path)

from project import temp_path, orig_data_path, raw_data_path


def main(input_directory=temp_path,
         raw_directory=raw_data_path,
         output_directory=orig_data_path):
    print("Capturing files from from:", str(input_directory), "to:",
          str(output_directory.resolve().absolute()))

    # Create orig data directory
    shutil.rmtree(output_directory, ignore_errors=True)
    output_directory.mkdir(parents=True)

    # Delete all files and create val/test folders
    for dataset in ['val', 'test', 'train']:
        output_directory.joinpath(dataset).mkdir()

    # Copy the original *.hea and *.mat files
    for dataset in ['val', 'test', 'train']:
        input_files = []
        for entry in raw_directory.joinpath(dataset).iterdir():
            if entry.name.endswith('data.npy'):
                input_files.append(entry.name.split('_')[0].split('-')[0])
        input_files = np.unique(input_files)
        print("Capturing", str(len(input_files)), "records.")
        for entry in input_files:
            shutil.copy(input_directory.joinpath(entry+'.mat'),
                        output_directory.joinpath(dataset).joinpath(entry+'.mat'))
            shutil.copy(input_directory.joinpath(entry+'.hea'),
                        output_directory.joinpath(dataset).joinpath(entry+'.hea'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Environment definitions for different platforms")

    parser.add_argument('--original_data_path', type=str,
                        default=str(temp_path))
    parser.add_argument('--processed_data_path', type=str,
                        default=str(raw_data_path))
    parser.add_argument('--destination_data_path', type=str,
                        default=str(orig_data_path))

    args = parser.parse_args()
    main(input_directory=pl.Path(args.original_data_path),
         raw_directory=pl.Path(args.processed_data_path),
         output_directory=pl.Path(args.destination_data_path))
