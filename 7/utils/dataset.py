import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms

def get_dataloader(dataset, batch_size, use_sampler=False, shuffle=True, drop_last=False):
  """Return a DataLoader instance that loads ECG data from the given
  dataset.
  """
  extra_params = {'shuffle': shuffle}
  if use_sampler:
    counts = dataset.dataset.groupby(dataset.label).size().values
    weight_per_class = 1. / torch.tensor(counts / max(counts), dtype=torch.float)
    weights = [0] * len(dataset)
    for idx, val in enumerate(dataset.dataset[dataset.label]):
      weights[idx] = weight_per_class[val]
    sampler = WeightedRandomSampler(
      weights=weights,
      num_samples=len(weights),
      replacement=True,
    )
    extra_params = {'sampler': sampler, 'shuffle': False}
  return torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, num_workers=8,
    drop_last=drop_last, **extra_params,
  )
