import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class LinearAligner(nn.Module):

  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(512, 1024)
    self.linear2 = nn.Linear(1024, 512)
    self.linear3 = nn.Linear(512, 512)
    self.linear4 = nn.Linear(512, 512)
  def forward(self, x):
    # Pass through linear layer 1
    x = self.linear1(x)

    # Apply tanh
    x = nn.Tanh()(x)

    # Pass through linear layer 2
    x = self.linear2(x)

    # Apply tanh
    x = nn.Tanh()(x)

    # Pass through linear layer 3
    x = self.linear3(x)

    x = self.linear4(x)

    # Return the LogSoftmax of the data
    # This will affect the loss we choose below
    return x
