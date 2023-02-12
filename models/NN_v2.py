#%%
# DO THIS TO INCLUDE UTILS
import os, sys
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)
# 

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from torchmetrics.classification import BinaryFBetaScore
from sklearn.model_selection import train_test_split

import numpy as np

from datetime  import datetime
from tqdm      import tqdm

from utils import process_data, output_csv, plot_tracks
import json

#%%
# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 10

INPUT_SIZE = 7
OUTPUT_SIZE = 1

# /Users/shreshth/Documents/Caltech/cs/cs155/motility-classification/data/train_features_20230211_151647.csv
# /Users/shreshth/Documents/Caltech/cs/cs155/motility-classification/data/test_features_20230211_152003.csv

TRAIN_BASIC_FEATURES_PATH = '../data/train_features_20230211_175449.csv'
TEST_BASIC_FEATURES_PATH = '../data/test_features_20230211_175307.csv'

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE_PATH = f'../output/{TIMESTAMP}_output.csv'

X_train, y_train = process_data(TRAIN_BASIC_FEATURES_PATH)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)


train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# MAKE SURE TO ADD shuffle=True IF YOU'RE
# GOING TO SUBMIT
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

# run from here
#%%
for data, target in test_loader:
    output = model(data)
    pred = torch.round(output)

    print(data[0])
    print(pred[:5])
    print(target[:5])
    break



# %%
MOTILE = 1
NONMOTILE = 0

X_train, y_train, uids = process_data(TRAIN_BASIC_FEATURES_PATH, columns_to_drop=['label'], return_uids=True)
num_examples = 2048
idx = len(X_train) - num_examples
print(f'X_train length: {len(X_train)}')
print(f'num examples examined: {num_examples}')

X_train = X_train[idx:]
y_train = y_train[idx:]
uids = uids[idx:]

output = model(X_train)
pred = torch.round(output)

is_misclassified = pred.squeeze() != y_train
misclassified_uids = uids[is_misclassified]

print(f'Num misclassified: {len(misclassified_uids)}')

misclassified_labels = y_train[is_misclassified]

misclassified_motile_indices = misclassified_labels.int() == MOTILE
misclassified_nonmotile_indices = misclassified_labels.int() != MOTILE

print('of those...')
print(f'{misclassified_motile_indices.sum()} are motile and')
print(f'{misclassified_nonmotile_indices.sum()} are nonmotile')

misclassified_motile_uids = misclassified_uids[misclassified_motile_indices]
misclassified_nonmotile_uids = misclassified_uids[misclassified_nonmotile_indices]

num_lab_misclassified = sum([1 if 'lab' in uid else 0 for uid in misclassified_uids])
num_sim_misclassified = sum([1 if 'sim' in uid else 0 for uid in misclassified_uids])
print(f'of the misclassified uids, {num_lab_misclassified} were lab samples and {num_sim_misclassified} were sim samples')


DATA_LOCATION = '../data/train.json'

# Load the training data
with open(DATA_LOCATION, 'r') as f:
    train_data = json.load(f)

misclassified_motile_tracks = [np.array(train_data[u]['txy']) for u in misclassified_motile_uids]
misclassified_nonmotile_tracks = [np.array(train_data[u]['txy']) for u in misclassified_nonmotile_uids]

plot_tracks(misclassified_motile_tracks, 'Misclassified motile particles')
plot_tracks(misclassified_nonmotile_tracks, 'Misclassified nonmotile particles')

output_csv(model, OUTPUT_FILE_PATH, TEST_BASIC_FEATURES_PATH)

# %%
