import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import torch.nn as nn
from torchmetrics import Accuracy

def calc_loss_cos_similarity(pred, yb, t):
  logits = torch.mm(pred, torch.t(yb)) * torch.exp(t)
  labels = torch.arange(yb.shape[0]).to("cuda")
  loss_1 = nn.CrossEntropyLoss()(logits, labels)
  torch.t(logits)
  loss_2 = nn.CrossEntropyLoss()(logits, labels)
  return (loss_1 + loss_2)/2

def calc_loss_euclid(pred, yb, t, accuracy, margin):
  # calculate distance metric
  logits = torch.cdist(pred, yb, p=2)
  logits = torch.pow(logits, -1) * torch.exp(t)

  print(f"Logit mean: {torch.mean(logits)}")
  print(f"Logit stddev: {torch.dim(logits)}")
  
  # apply margin
  logits = nn.ReLU()(logits - margin) + margin
  
  # do cross entropy along both axes
  labels = torch.arange(yb.shape[0]).to("cuda")
  loss_1 = nn.CrossEntropyLoss()(logits, labels)
  accuracy_1 = accuracy(logits, labels)
  torch.t(logits)
  loss_2 = nn.CrossEntropyLoss()(logits, labels)
  accuracy_2 = accuracy(logits, labels)

  return (loss_1 + loss_2)/2, (accuracy_1 + accuracy_2)/2

def train_contrastive_model(train_dl, test_dl, model, optimizer, scheduler, num_epochs, model_name, start_epoch):
  temperature = nn.Parameter(torch.tensor([0.0])).to("cuda")
  margin = nn.Parameter(torch.tensor([0.0])).to("cuda")
  torch.autograd.set_detect_anomaly(True)
  for epoch in range(start_epoch, start_epoch + num_epochs):
    print(f"EPOCH {epoch} BEGINS")
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    num_data_points = 0
    num_batch = 0
    nans = 0
    for xb, yb in train_dl:
      #get model predictions [B, d_model=512]
      batch_size = xb.shape[0]
      pred = model(xb)

      # contrastive loss
      accuracy = Accuracy(task="multiclass", num_classes=batch_size).to('cuda')
      loss, acc = calc_loss_euclid(pred, yb, temperature, accuracy, margin)

      if torch.isnan(loss):
        nans += 1

      optimizer.zero_grad()
      loss.sum().backward()
      optimizer.step()

      train_loss += loss
      train_acc += acc * batch_size
      num_data_points += batch_size

      if num_batch % 1 == 0:
<<<<<<< HEAD
        print(f"Epoch: {epoch} Batch: {num_batch} Avg Loss: {loss/batch_size} Avg Accuracy: {acc} Nans: {nans} Margin: {margin.item()}")
=======
        print(f"Epoch: {epoch} Batch: {num_batch} Avg Loss: {loss/batch_size} Avg Accuracy: {acc} Nans: {nans} Margin: {margin.item()} Temp: {temperature}")
>>>>>>> 66e3d555342363575f0f3850e34cb3d305f2e7d6

      num_batch += 1

    print(f"\nTrain average loss after epoch {epoch} is {train_loss/num_data_points}")
    print(f"Train average acc after epoch {epoch} is {train_acc/num_data_points}\n")

    #save model
    torch.save(model.state_dict(), f"../checkpoints/{model_name}_epoch_{epoch}.pt")
    print("Checkpoint saved.\n")

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    num_data_points = 0
    for xb, yb in test_dl:
      #get model predictions [B, d_model=512]
      batch_size = xb.shape[0]
      pred = model(xb)

      # contrastive loss
      accuracy = Accuracy(task="multiclass", num_classes=batch_size).to('cuda')
      loss, acc = calc_loss_euclid(pred, yb, temperature, accuracy, margin)

      test_loss += loss
      test_acc += acc * batch_size
      num_data_points += batch_size

    print(f"\nTest average loss after epoch {epoch} is {test_loss/num_data_points}")
    print(f"Test average accuracy after epoch {epoch} is {test_acc/num_data_points}\n")
