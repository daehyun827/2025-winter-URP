import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 데이터 파일에서 쓸 수 있는 형태로 데이터 꺼내오기
# acc를 key로 갖는 dict 안에 x, adj, parition을 갖는 dict 형태임
class TrainDataset(torch.utils.data.Dataset):
  def __init__(self, train_path):
    super().__init__()

    self.train = torch.load(train_path, map_location="cpu", mmap=True)
    self.train_keys = list(self.train)

  def __len__(self):
    idx = len(self.train_keys)
    return idx

  def __getitem__(self, idx):
    train_key =  self.train_keys[idx]
    train_sample = self.train[train_key]
    x = train_sample['x']
    adj = train_sample['adj']
    y = train_sample['y']
    part = train_sample['Partition']
    return x, adj, y, part