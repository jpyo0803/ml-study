import torch
from config import Config
from torch.utils.data import Dataset

cfg = Config()

class MyDataset(Dataset):
  def __init__(self, src, tgt_input, tgt_output):
    self.src = src
    self.tgt_input = tgt_input
    self.tgt_output = tgt_output

  def __len__(self):
    return self.src.shape[0]
  
  def __getitem__(self, idx):
    return {
      'src': self.src[idx],
      'tgt_input': self.tgt_input[idx],
      'tgt_output': self.tgt_output[idx],
    }

def load_train_dataset():
  data = torch.load(cfg.train_dataset_path)
  dataset = MyDataset(data['src'], data['tgt_input'], data['tgt_output'])
  return dataset

def load_valid_dataset():
  data = torch.load(cfg.valid_dataset_path)
  dataset = MyDataset(data['src'], data['tgt_input'], data['tgt_output'])
  return dataset