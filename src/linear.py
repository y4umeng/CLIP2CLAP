import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class LinearAligner(nn.Module):
  def __init__(self, clip_embed_dim, clap_embed_dim):
    super().__init__()
    self.linear = nn.Linear(clip_embed_dim, clap_embed_dim)

  def forward(self, x):
    # Pass through linear layer 1
    return self.linear(x)