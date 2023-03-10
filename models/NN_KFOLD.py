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
from torch.utils.data import DataLoader,TensorDataset, SubsetRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torchmetrics.classification import BinaryFBetaScore

from tqdm      import tqdm
from datetime  import datetime

from utils import process_data, output_csv

#%%
# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 10

INPUT_SIZE = 5
OUTPUT_SIZE = 1

TRAIN_BASIC_FEATURES_PATH = '../data/train_basic_features.csv'
TEST_BASIC_FEATURES_PATH = '../data/test_basic_features.csv'

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE_PATH = f'../output/{TIMESTAMP}_output.csv'

X_data, y_data = process_data(TRAIN_BASIC_FEATURES_PATH)

K=10
splits = KFold(n_splits=K, shuffle=True)
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)


total_dataset = TensorDataset(X_data, y_data)

#%%
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True) 

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
for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_dataset)))):
    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(total_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    test_loader = DataLoader(total_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

    # Training loop
    model.train()

    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        train_f2 = 0
        train_correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Erase accumulated gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate loss
            loss = loss_fn(output, target.unsqueeze(dim=-1))

            # Backward pass
            loss.backward()

            # Add loss printing shit
            train_loss += loss.item()
            train_f2 += F2_loss(output, target.unsqueeze(dim=-1)).item()
            pred = torch.round(output)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            # 

            # Weight update
            optimizer.step()

        train_loss /= len(train_loader.sampler)
        train_f2 /= len(train_loader.sampler)

        # Track loss each epoch
        print('Train Epoch: %d  Loss: %.4f' % (epoch + 1,  loss.item()))

        # Track loss each epoch
        print('Average loss: %.4f, F2: %.4f, Accuracy: %d/%d (%.4f)'
            % (train_loss, train_f2, train_correct, len(train_loader.sampler),
           100. * train_correct / len(train_loader.sampler)))

    model.eval()

    test_loss = 0
    correct = 0

    # Turning off automatic differentiation
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F2_loss(output, target.unsqueeze(dim=-1)).item()
            pred = torch.round(output)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.sampler)

    print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.4f)' %
          (test_loss, correct, len(test_loader.sampler),
           100. * correct / len(test_loader.sampler)))

# %%

# output_csv(model, OUTPUT_FILE_PATH, TEST_BASIC_FEATURES_PATH)


# %%
