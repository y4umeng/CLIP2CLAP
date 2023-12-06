import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import open_clip
from diffusers import AudioLDMPipeline
from data import get_models, get_data, get_embeds, EmbeddingsDataset
from encoder import AttentionAligner

def train():
    CLIP, CLAP, TOKENIZER = get_models()
    train_data, test_data = get_data()
    train_dataset = EmbeddingsDataset(train_data)
    test_dataset = EmbeddingsDataset(test_data)
    print(f"Length of test data: {len(test_dataset)}")
    print(f"Length of train data: {len(train_dataset)}")

    torch.manual_seed(420)

    # Hyperparameters.
    # I suggest you start with very small values, unless you have a strong PC or are running on the cluster
    batch_size = 64 # How many independent sequences will we process in parallel?
    block_size = 512 # What is the maximum context length for predictions?
    EPOCHS = 20 # Max iterations we run the optimization
    # How often we evaluate across the optimization; every 500 iterations
    eval_interval = 500
    learning_rate = 3e-4

    device = 'cuda'
    # How many batches we use each time we evaluate
    eval_iters = 200
    d_model = 36
    n_head = 6 # This implied that each head has a dimension for the key, query, and values of d_model / 6.
    n_layer = 6 # This implies we have 6 turns to mix the embeddigs; this is "Nx" in the paper
    step_size = 10000
    dropout = 0.2
    # ------------

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=True)

    model = AttentionAligner(step_size, d_model, block_size, n_head, n_layer, dropout)
    model = nn.DataParallel(model)
    model.to("cuda")
    model.train()
    # Print the number of parameters in the model
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(f"Number of parameters: {sum([torch.prod(torch.tensor(p.shape)) for p in model_parameters])}")

    # Create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    for epoch in range(EPOCHS):
        batch = 0
        for xb, yb in train_dl:
            # every once in a while evaluate the loss on test set
            # if iter % eval_interval == 0:
            #     if iter:
            #       scheduler.step()
            #     losses = estimate_loss(model)
            #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Evaluate the loss
            logits, loss = model(xb, yb)
            if batch % 500 == 0:
                print(f"Loss at epoch {epoch}, batch {batch}: {loss}")
            optimizer.zero_grad(set_to_none=True)
            loss.sum().backward()
            optimizer.step()
                
            batch += 1

        torch.save(model.state_dict(), f"./checkpoints/encoder_epoch_{epoch}.pt")

train()
print("TRAINING DONE")
