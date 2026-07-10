#!/usr/bin/env python

import json
import torch
import numpy as np
import pathlib as pl
from torchvision import transforms

from scripts.preprocess_challenge2020 import resample
from utils import transforms_from_parameters

from src.networks.base_network import BaseNetwork


def run_12ECG_classifier(data, header, loaded_models):
    # Parameters for testing
    scale_temperature = False
    sigmoid_early = True

    # Load data and header information from header
    age, sex, freq, resolution = read_header_submission(header)

    sex = np.nan_to_num(sex, nan=-1)
    age = (np.nan_to_num(age, nan=-1) - 60) / 10.

    # Resample the data to 500 Hz
    target_frequency = np.int(500)
    source_frequency = np.int(freq)
    if source_frequency != target_frequency:
        print("Resample from:", source_frequency, "to:", target_frequency,
              "Input Shape:", *data.shape)
        data = resample(data, source_frequency, target_frequency)
    print("Data Input Shape:", *data.shape)

    ensemble_out = []
    for model in loaded_models['base_models']:
        network = model['network']
        encoder = model['encoder']
        parameter = model['parameter']
        preprocess = model['preprocess']
        temperature = model['temperature']

        network.eval()

        # Length of the network input
        n_input = parameter['input_length']

        # Function that iterates over multiple sequences in time
        logits = forward_sequential(n_input, network, preprocess, data, age, sex)

        # Model calibration
        if scale_temperature:
            logits = logits / temperature

        if sigmoid_early:
            y_t = torch.sigmoid(torch.from_numpy(logits)).detach().cpu().numpy()
            y_t = y_t.mean(0)
        else:
            # Mean of logits
            y_t = logits.mean(0)

        # List of logits over ensemble
        ensemble_out.append(y_t)

    # Averaging
    y_outs = np.array(ensemble_out).mean(0)

    # Sigmoid
    if not sigmoid_early:
        y_prob = torch.sigmoid(torch.from_numpy(y_outs)).detach().cpu().numpy()
    else:
        y_prob = y_outs

    y_pred = (y_prob > 0.5).astype(np.int64)
    y_snom = [str(s) for s in encoder.values()]

    return y_pred.tolist(), y_prob.tolist(), y_snom


def forward_model(network, preprocess, data, age, sex):
    with torch.no_grad():
        # Check if CUDA available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data = preprocess(data)
        data = torch.tensor(data).float().to(device)
        age = (np.nan_to_num(age, nan=-1)-60)/10.
        age = torch.tensor(age).float().to(device)
        sex = np.nan_to_num(sex, nan=-1)
        sex = torch.tensor(sex).float().to(device)
        network = network.to(device)

        # Reshape input tensors
        data = data.view(-1, *data.shape)
        age = age.view(-1, *age.shape)
        sex = sex.view(-1, *sex.shape)

        # For now we are only using models trained with sigmoid
        logits = network(data, age, sex)

    return logits.detach().cpu().numpy()


def forward_sequential(n_input, network, preprocess, data, age, sex):
    if data.shape[1] <= n_input:
        logits = forward_model(network, preprocess, data, age, sex)
    else:
        logits = []
        # Moving average with window size n_input/2
        stepsize = int(n_input / 2)
        for i in range(0, data.shape[1], stepsize):
            begin = i
            end = i + n_input

            # Corner condition for last split
            if (end - stepsize) > data.shape[1]:
                begin = data.shape[1] - n_input
                end = data.shape[1]

            logits.append(
                forward_model(network,
                        preprocess,
                        data[:, begin:end],
                        age,
                        sex).squeeze())

        logits = np.array(logits)
    return logits


def load_12ECG_model(input_directory):
    # Check environment: CUDA available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0), 'Memory Usage: [Allocated:',
              round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB',
              'Cached:', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB]')

    input_path = pl.Path(input_directory)

    submission_file = input_path.joinpath("submission_08.json")
    with open(submission_file, 'r') as file:
        submissions_models = json.load(file)

    print("Loading models from:", submission_file,
          "supporting:", submissions_models)

    loaded_models = {'base_models': []}
    # Load the models
    for submissions_model in submissions_models['models']:
        # Load model parameter
        parameter_file = input_path.joinpath(
            submissions_model + '.json')
        with open(parameter_file, "r") as file:
            parameter = json.load(file)

        # Load encoder <-> decoder dict for given model
        encoder_file = input_path.joinpath(
            submissions_model + '_encoder.json')
        with open(encoder_file, "r") as file:
            encoder = json.load(file)

        # Load the preprocessing transformations
        _, _, _, preprocess = transforms_from_parameters(parameter)

        # Initialize network from given parameter
        network = BaseNetwork(
            n_sequence=parameter['input_length'],
            n_channels=parameter['input_channels'],
            n_classes=len(encoder.keys()),
            feature_extractor_params=parameter['feature_extractor'],
            classifier_params=parameter['classifier'])

        # Load weights
        network_weights_file = input_path.joinpath(
            submissions_model + '_weights.pth')
        network.load_state_dict(torch.load(network_weights_file))

        # Load temperature (incase we want to apply model scaling)
        try:
            scale_file = input_path.joinpath(
                submissions_model + '_scaled.json')
            with open(scale_file, "r") as file:
                temperature = json.load(file)["temperature"]
        except:
            temperature = 1.

        # Print the network
        print(network)

        loaded_models['base_models'].append({
            'network': network.eval(),
            'encoder': encoder,
            'parameter': parameter,
            'preprocess': transforms.Compose(preprocess),
            'temperature': temperature
        })
    print(loaded_models)

    return loaded_models


def read_header_submission(header):
    age = np.nan
    gender = np.nan
    for line in header:
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

    return age, gender, freq, resolution
