""" This file contains some convenience functions for
    downloading the provided training files and extracting
    them accordingly """

import os
import re
import json
import tqdm
import tarfile
import argparse
import numpy as np

from google.cloud import storage
from urllib.request import urlopen

import sys
import pathlib as pl
file_path = str(pl.Path(os.path.dirname(
    os.path.realpath(__file__))).joinpath('..').resolve())
sys.path.append(file_path)

from project import temp_path, scripts_path

# Current data packages released (may be a growing list)
releases = [{'bucket': 'physionet-challenge-2020-12-lead-ecg-public',
             'blob': 'PhysioNetChallenge2020_Training_CPSC.tar.gz',
             'destination': 'training_1.tar.gz'},
            {'bucket': 'physionet-challenge-2020-12-lead-ecg-public',
             'blob': 'PhysioNetChallenge2020_Training_2.tar.gz',
             'destination': 'training_2.tar.gz'},
            {'bucket': 'physionet-challenge-2020-12-lead-ecg-public',
             'blob': 'PhysioNetChallenge2020_Training_StPetersburg.tar.gz',
             'destination': 'training_3.tar.gz'},
            {'bucket': 'physionet-challenge-2020-12-lead-ecg-public',
             'blob': 'PhysioNetChallenge2020_Training_PTB.tar.gz',
             'destination': 'training_4.tar.gz'},
            {'bucket': 'physionet-challenge-2020-12-lead-ecg-public',
             'blob': 'PhysioNetChallenge2020_Training_PTB-XL.tar.gz',
             'destination': 'training_5.tar.gz'},
            {'bucket': 'physionet-challenge-2020-12-lead-ecg-public',
             'blob': 'PhysioNetChallenge2020_Training_E.tar.gz',
             'destination': 'training_6.tar.gz'},
            ]

baseUrl = 'https://browser.ihtsdotools.org/snowstorm/snomed-ct'
edition = 'MAIN'
version = '2019-07-31'


def download(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client.create_anonymous_client()
    print("Downloading Bucket:", bucket_name, "Source:",
          source_blob_name, "Destination", destination_file_name)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def getConceptById(snomed_id):
    url = baseUrl + '/browser/' + edition + '/' + version + '/concepts/' + snomed_id
    response = urlopen(url).read()
    data = json.loads(response.decode('utf-8'))

    return data['fsn']['term']


def getDescriptionById(snomed_id):
    url = baseUrl + '/' + edition + '/' + version + '/descriptions/' + snomed_id
    response = urlopen(url).read()
    data = json.loads(response.decode('utf-8'))

    return data['term']


def formatConcepts(rows, encoder):
    concepts = []
    for row in rows:
        snomed_desc = row[1]
        groups = re.split(r"\((\w+)\)", snomed_desc.replace('observable entity',
                                                            'observable'))[:-1]
        groups[0] = groups[0].replace('Electrocardiographic: ', '')
        groups[0] = groups[0].replace('Electrocardiographic ', '')
        groups[0] = groups[0].replace('Electrocardiogram: ', '')
        groups[0] = groups[0].replace('Electrocardiogram ', '')
        groups[0] = groups[0].lower().strip()
        label = encoder.transform([[row[0]]])[0]
        concepts.append([np.where(label)[0][0], row[0], *groups])

    return concepts


def main(physionet_archive_path=temp_path, compressed_archive_path=scripts_path):
    print("Downlading release blobs to:", compressed_archive_path,
          "Extracting release blobs into:", physionet_archive_path)

    # Ignore if paths already exist otherwise create
    physionet_archive_path.mkdir(parents=True, exist_ok=True)
    compressed_archive_path.mkdir(parents=True, exist_ok=True)

    # Download the data
    for release in releases:
        source_bucket = release['bucket']
        source_blob = release['blob']
        destination_file = compressed_archive_path.joinpath(release['destination'])

        # Download the physionet release
        if destination_file.is_file():
            print("{} already downloaded.".format(destination_file))
        else:
            download(source_bucket, source_blob, destination_file)
            print("{} downloaded to {}.".format(source_blob, destination_file))

        # Extract the files
        archive = tarfile.open(destination_file)
        for compressed in tqdm.tqdm(archive.getmembers()):
            if compressed.isreg():
                compressed.name = os.path.basename(compressed.name)
                if not physionet_archive_path.joinpath(compressed.name).is_file():
                    archive.extract(compressed, physionet_archive_path)
        archive.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Environment definitions for different platforms")

    parser.add_argument('--compressed_archive_path', type=str,
                        default=str(scripts_path))
    parser.add_argument('--physionet_archive_path', type=str,
                        default=str(temp_path))

    args = parser.parse_args()
    main(physionet_archive_path=pl.Path(args.physionet_archive_path),
         compressed_archive_path=pl.Path(args.compressed_archive_path))
