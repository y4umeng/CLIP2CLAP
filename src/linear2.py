import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class LinearAligner2(nn.Module):
  def __init__(self, clip_embed_dim, clap_embed_dim):
    super().__init__()
    self.linear = nn.Linear(clip_embed_dim, clap_embed_dim)
    self.linear2 = nn.Linear(clap_embed_dim, clap_embed_dim)

  def forward(self, x):
    x = self.linear(x)
    x = nn.Tanh()(x)
    return torch.clamp(self.linear2(x), min=-0.3, max=0.3)