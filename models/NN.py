#%%
# DO THIS TO INCLUDE UTILS
import os, sys
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)
# 

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split
from torchmetrics.classification import BinaryFBetaScore

from tqdm      import tqdm
from datetime  import datetime

from utils import process_data, output_csv

"""
To figure out:
-incorporate new feature
-make output of model be predictions
-F2
"""

#%%
# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 2

INPUT_SIZE = 5
OUTPUT_SIZE = 1

TRAIN_BASIC_FEATURES_PATH = '../data/train_basic_features.csv'
TEST_BASIC_FEATURES_PATH = '../data/test_basic_features.csv'

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE_PATH = f'../output/{TIMESTAMP}_output.csv'

X_train, y_train = process_data(TRAIN_BASIC_FEATURES_PATH)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.99)


# print(len(X_train))
# print(len(X_test))


train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

#%%
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True) 

#%%
model = nn.Sequential(
    nn.Flatten(),

    nn.Linear(INPUT_SIZE, 550),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(550, 250),
    nn.ReLU(),

    nn.Linear(250, 100),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(100, 50),
    nn.ReLU(),

    nn.Linear(50, OUTPUT_SIZE),
    nn.Sigmoid()
)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss()
F2_loss = BinaryFBetaScore(beta=2.0) # F2 loss

#%%
# Training loop
model.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Erase accumulated gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate loss
        loss = loss_fn(output, target.unsqueeze(dim=-1))

        # Backward pass
        loss.backward()
        
        # Weight update
        optimizer.step()

    # Track loss each epoch
    print('Train Epoch: %d  Loss: %.4f' % (epoch + 1,  loss.item()))

#%%

model.eval()

test_loss = 0
correct = 0
zeros = 0

# Turning off automatic differentiation
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = torch.round(output)
        test_loss += F2_loss(pred, target.unsqueeze(dim=-1)).item()
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.4f)' %
      (test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))

# %%

output_csv(model, OUTPUT_FILE_PATH, TEST_BASIC_FEATURES_PATH)


# %%
