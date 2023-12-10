import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import open_clip
from diffusers import AudioLDMPipeline
from data import get_models, get_data, get_embeds, EmbeddingsDataset

CLIP, CLAP, TOKENIZER = get_models()
train_data, test_data = get_data()
train_dataset = EmbeddingsDataset(train_data)
test_dataset = EmbeddingsDataset(test_data)

train_dl = DataLoader(train_dataset, 100, shuffle=True)
test_dl = DataLoader(test_dataset, 100, shuffle=True)

for x_batch, y_batch in train_dl:
    print(x_batch.shape)
    print(y_batch.shape)

