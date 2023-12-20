import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class LinearAligner4(nn.Module):
  def __init__(self, clip_embed_dim, clap_embed_dim):
    super().__init__()
    self.ff1 = nn.Linear(clip_embed_dim, clap_embed_dim)
    self.ln1 = nn.LayerNorm([512])
    self.ff2 = nn.Linear(clap_embed_dim, clap_embed_dim)
    self.ln2 = nn.LayerNorm([512])
    self.ff3 = nn.Linear(clap_embed_dim, clap_embed_dim)
    self.ln3 = nn.LayerNorm([512])
    self.ff4 = nn.Linear(clap_embed_dim, clap_embed_dim)
    self.ln4 = nn.LayerNorm([512])
    self.t = nn.Parameter(torch.tensor([1]).float())

  def forward(self, x):
    x = self.ff1(x)
    x = nn.Tanh()(x)
    x = self.ln1(x)

    x = self.ff2(x)
    x = nn.Tanh()(x)
    x = self.ln2(x)

    x = self.ff3(x)
    x = nn.Tanh()(x)
    x = self.ln3(x)

    x = self.ff4(x)
    x = nn.Tanh()(x)
    x = self.ln4(x)

    return torch.clamp(x, min=-0.2, max=0.2)