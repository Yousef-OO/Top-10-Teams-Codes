""" This file contains some functions for downloading the public
database mitdb, which is not part of the challenge and offers
additional room for validation and verification as well as
the possibility to update the labels contained in the challenge
database

Attention: This should be used for pretraining:
Source: https://physionetchallenges.github.io/2020/faq

Are we allowed to do transfer learning using pre-trained networks?

Yes, most certainly. We encourage you to do this. You do not need
to include your data in the code stack for training the algorithm,
but you do need to include the pre-trained model in the code and
provide code to retrain (continue training) on the training data
we provide. You must also thoroughly document the content of the
database you used to pre-train.

"""

import os
import tqdm
import shutil

from google.cloud import storage

import sys
import pathlib as pl
file_path = str(pl.Path(os.path.dirname(
    os.path.realpath(__file__))).joinpath('..').resolve())
sys.path.append(file_path)

from project import mitdb_path


def download(bucket_name, destination_path):
    valid_endings = ['atr', 'hea', 'dat']

    storage_client = storage.Client.create_anonymous_client()
    # Estimate number of files in google cloud bucket
    files = []
    for file in tqdm.tqdm(storage_client.list_blobs(bucket_name)):
        # Only download valid files (atr, hea and dat)
        if file.name.split('.')[-1] not in valid_endings:
            continue
        # Remove old files (some are outdated e.g. 102-0.atr)
        if file.name.split('-') == 2:
            continue
        files.append(file)

    print("Downloading files:", len(files))
    # Download files in given folder
    for blob in tqdm.tqdm(files):
        destination_file = destination_path.joinpath(blob.name.split('/')[-1])
        blob.download_to_filename(destination_file)


def main():
    # Delete mitdb-folder and create a empty folder
    shutil.rmtree(mitdb_path, ignore_errors=True)
    mitdb_path.mkdir(parents=True)

    # Downloading files from given google cloud bucket
    # https://console.cloud.google.com/storage/browser/mitdb-1.0.0.physionet.org/
    download('mitdb-1.0.0.physionet.org', mitdb_path)


if __name__ == '__main__':
    main()
