import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class LinearAligner3(nn.Module):
  def __init__(self, clip_embed_dim, clap_embed_dim):
    super().__init__()
    self.ff1 = nn.Linear(clip_embed_dim, clap_embed_dim)
    self.ff2 = nn.Linear(clap_embed_dim, clap_embed_dim)
    self.ff3 = nn.Linear(clap_embed_dim, clap_embed_dim)
    self.ff4 = nn.Linear(clap_embed_dim, clap_embed_dim)

  def forward(self, x):
    x = self.ff1(x)
    x = nn.Tanh()(x)
    x = self.ff2(x)
    x = nn.Tanh()(x)
    x = self.ff3(x)
    x = nn.Tanh()(x)
    return torch.clamp(self.ff4(x), min=-0.2, max=0.2)