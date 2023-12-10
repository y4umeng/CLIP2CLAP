import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import open_clip
from diffusers import AudioLDMPipeline
from data import get_models, get_data, get_embeds, EmbeddingsDataset
from linear import LinearAligner

def train():
    CLIP, CLAP, TOKENIZER = get_models()
    train_data, test_data = get_data()
    train_dataset = EmbeddingsDataset(train_data)
    test_dataset = EmbeddingsDataset(test_data)
    print(f"Length of test data: {len(test_dataset)}")
    print(f"Length of train data: {len(train_dataset)}")
    
    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 20
    
    train_dl = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

    model = LinearAligner()
    model = nn.DataParallel(model)
    model.to("cuda")
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    SEED = 420
    torch.manual_seed(SEED)
    
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}")
        loss_hist_train = 0
        # Loop through the x and y pairs of data
        for x_batch, y_batch in train_dl:
            # Get the the model predictions
            pred = model(x_batch)

            # Get the loss
            loss = loss_fn(pred, y_batch)

            # Get the gradients
            loss.backward()

            # Add to the loss
            loss_hist_train += loss.item() * len(y_batch)

            # Update the prameters
            optimizer.step()

            # Zero out the gradient
            optimizer.zero_grad()

        loss_hist_train /= len(train_dl.dataset)
        print(f'Train Metrics Epoch {epoch} Loss {loss_hist_train:.4f}')

        torch.save(model.state_dict(), f"./checkpoints/linear_epoch_{epoch}.pt")

        loss_hist_test = 0
        # Get the average value of each metric across the test batches
        with torch.no_grad():
          # Loop through the x and y pairs of data
          for x_batch, y_batch in test_dl:
              # Get he the model predictions
              pred = model(x_batch)

              # Get the loss
              loss = loss_fn(pred, y_batch)

              # Add to the loss
              loss_hist_test += loss.item() * len(y_batch)

          # Normalize the metrics by the right number
          loss_hist_test /= len(test_dl.dataset)
          print(f'Test Metrics Epoch {epoch} Loss {loss_hist_test:.4f}')

train()
