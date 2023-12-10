import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import torch.nn as nn

def calc_loss(pred, yb, t):
  logits = torch.mm(pred, torch.t(yb)) * torch.exp(t)
  labels = torch.arange(yb.shape[0]).to("cuda")
  loss_1 = nn.CrossEntropyLoss()(logits, labels)
  torch.t(logits)
  loss_2 = nn.CrossEntropyLoss()(logits, labels)
  return (loss_1 + loss_2)/2

def train_contrastive_model(train_dl, test_dl, model, optimizer, scheduler, num_epochs, model_name):
  t = nn.Parameter(torch.tensor([0.0])).to("cuda")
  for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    num_data_points = 0
    num_batch = 0
    for xb, yb in train_dl:
      #get model predictions [B, d_model=512]
      pred = model(xb)

      # contrastive loss
      loss = calc_loss(pred, yb, t)

      optimizer.zero_grad()
      loss.sum().backward()
      optimizer.step()

      train_loss += loss
      num_data_points += xb.shape[0]

      if num_batch % 20 == 0:
        print(f"Epoch: {epoch} Batch: {num_batch} Batch Loss: {loss}")
      num_batch += 1

    print(f"Train average loss after epoch {epoch} is {train_loss/num_data_points}")

    model.eval()
    test_loss = 0.0
    num_data_points = 0
    num_batch = 0
    for xb, yb in test_dl:
      #get model predictions [B, d_model=512]
      pred = model(xb)

      # contrastive loss
      loss = calc_loss(pred, yb, t)

      optimizer.zero_grad()
      loss.sum().backward()
      optimizer.step()

      test_loss += loss
      num_data_points += xb.shape[0]
      num_batch += 1
      if num_batch == 100:
        break

    print(f"Test average loss after epoch {epoch} is {test_loss/num_data_points}")

    #save model
    torch.save(model.state_dict(), f"./checkpoints/{model_name}_epoch_{epoch}.pt")
