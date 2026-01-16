import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TestDataset(torch.utils.data.Dataset):
  def __init__(self, test_path):
    super().__init__()

    self.test = torch.load(test_path)
    self.test_keys = list(self.test)

  def __len__(self):
    idx = len(self.test_keys)
    return idx

  def __getitem__(self, idx):
    test_key = self.test_keys[idx]
    test_sample = self.test[test_key]
    test_x = test_sample['x']
    test_adj = test_sample['adj']
    test_y = test_sample['y']
    return test_x, test_adj, test_y