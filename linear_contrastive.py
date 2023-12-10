import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import torch.nn as nn
import open_clip
from diffusers import AudioLDMPipeline
from data import get_models, get_data, get_embeds, EmbeddingsDataset
from contrastive_train import train_contrastive_model

class LinearAligner(nn.Module):
  def __init__(self, clip_embed_dim, clap_embed_dim):
    super().__init__()
    self.linear = nn.Linear(clip_embed_dim, clap_embed_dim)

  def forward(self, x):
    # Pass through linear layer 1
    return self.linear(x)

def train():
    CLIP, CLAP, TOKENIZER = get_models()
    train_data, test_data = get_data()
    train_dataset = EmbeddingsDataset(train_data)
    test_dataset = EmbeddingsDataset(test_data)
    print(f"Length of test data: {len(test_dataset)}")
    print(f"Length of train data: {len(train_dataset)}")

    torch.manual_seed(420)
 
    # Hyperparameters.
    batch_size = 32
    EPOCHS = 40
    LR = 0.001
    device = 'cuda'
    clip_embed_dim = 768
    clap_embed_dim = 512

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=True)

    model = LinearAligner(clip_embed_dim, clap_embed_dim)
    model.to(device)
    model = nn.DataParallel(model)
    model.train()

    # Print the number of parameters in the model
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(f"Number of parameters: {sum([torch.prod(torch.tensor(p.shape)) for p in model_parameters])}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_contrastive_model(train_dl, test_dl, model, optimizer, None, EPOCHS, "linear_contrastive_loss")

train()