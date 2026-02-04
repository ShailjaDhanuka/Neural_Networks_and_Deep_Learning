"""
You will need to validate your NN implementation using PyTorch. You can use any PyTorch functional or modules in this code.

IMPORTANT: DO NOT change any function signatures
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
import torch.nn.functional as F


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SingleLayerMLP(nn.Module):
    """constructing a single layer neural network with Pytorch"""
    def __init__(self, indim, outdim, hidden_layer=100):

        super(SingleLayerMLP, self).__init__()
        
        #making the layers
        self.inputfc = nn.Linear(indim,hidden_layer)
        self.hiddenfc = nn.Linear(hidden_layer,outdim)

    def forward(self, x):
        """
        x shape (batch_size, indim)
        """
        x = self.inputfc(x)
        q = F.relu(x)
        logits = self.hiddenfc(q)
        return logits

class DS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.length = len(X)
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return (x, y)

    def __len__(self):
        return self.length


def validate(loader):
    """takes in a dataloader, then returns the model loss and accuracy on this loader"""
    
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.long().to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


if __name__ == "__main__":
    """The dataset loaders were provided for you.
    You need to implement your own training process.
    You need plot the loss and accuracies during the training process and test process. 
    """


    indim = 60
    outdim = 2
    hidden_dim = 32
    lr = 0.1
    batch_size = 32
    epochs = 500

    #dataset
    Xtrain = pd.read_csv(r"C:\Users\shail\Desktop\Shailja_everything\CMU_courses\NN_and_DL\HW1 (3)\HW1\data\X_train.csv")
    Ytrain = pd.read_csv(r"C:\Users\shail\Desktop\Shailja_everything\CMU_courses\NN_and_DL\HW1 (3)\HW1\data\y_train.csv")
    scaler = MinMaxScaler()
    Xtrain = scaler.fit_transform(Xtrain.to_numpy())
    Ytrain = np.squeeze(Ytrain.to_numpy())
    m1, n1 = Xtrain.shape
    print(m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    Xtest = pd.read_csv(r"C:\Users\shail\Desktop\Shailja_everything\CMU_courses\NN_and_DL\HW1 (3)\HW1\data\X_test.csv")
    Ytest = pd.read_csv(r"C:\Users\shail\Desktop\Shailja_everything\CMU_courses\NN_and_DL\HW1 (3)\HW1\data\y_test.csv")
    Xtest = scaler.transform(Xtest.to_numpy())
    Ytest = np.squeeze(Ytest.to_numpy())
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    #construct the model
    model = SingleLayerMLP(indim, outdim, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    #construct the training process
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.float().to(device)
            y = y.long().to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        test_loss, test_acc = validate(test_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
            )

import os
import pickle

save_dir = r"C:\Users\shail\Desktop\Shailja_everything\CMU_courses\NN_and_DL\HW1 (3)\HW1"
save_path = os.path.join(save_dir, "self_ref_metrics.pkl")

with open(save_path, "wb") as f:
    pickle.dump((train_losses,  train_accs, test_losses, test_accs), f)
print(f"\nMetrics saved to: {save_path}")