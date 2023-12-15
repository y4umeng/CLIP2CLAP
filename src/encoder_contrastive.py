import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import torch.nn as nn
import open_clip
from diffusers import AudioLDMPipeline
from data import get_models, get_data_gcc, get_embeds, EmbeddingsDataset
from contrastive_train import train_contrastive_model
from encoder4 import AttentionAligner4
from linear3 import LinearAligner3

def train():
    _, _, _ = get_models()
    train_data, test_data = get_data_gcc()
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
    clip_embed_dim = 512
    clap_embed_dim = 512

    print(f"Batch size: {batch_size}")
    print(f"Epochs: {EPOCHS}")
    print(f"LR: {LR}")
    print(f"CLIP embed: {clip_embed_dim}")
    print(f"CLAP embed: {clap_embed_dim}")

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=True)

    model = AttentionAligner4(clip_embed_dim)

    model.to(device)
    model = nn.DataParallel(model)
    # checkpoint = torch.load("../checkpoints/linear_contrastive_loss_epoch_8.pt")
    # model.load_state_dict(checkpoint)
    model.train()

    # Print the number of parameters in the model
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(f"Number of parameters: {sum([torch.prod(torch.tensor(p.shape)) for p in model_parameters])}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_contrastive_model(train_dl, test_dl, model, optimizer, None, EPOCHS, "encoder_contrastive_512", 0)

train()
