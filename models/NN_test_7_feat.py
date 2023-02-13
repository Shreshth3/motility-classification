#%%
# DO THIS TO INCLUDE UTILS
import os, sys
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)
# 

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, WeightedRandomSampler

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
NUM_EPOCHS = 50

INPUT_SIZE = 7
OUTPUT_SIZE = 1

TRAIN_BASIC_FEATURES_PATH = '../data/train_7_feat.csv'
TEST_BASIC_FEATURES_PATH = '../data/test_7_feat.csv'

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE_PATH = f'../output/{TIMESTAMP}_output.csv'

X_train, y_train = process_data(TRAIN_BASIC_FEATURES_PATH)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

ones = torch.sum(y_train).item() / len(y_train)

weights = np.array([ones if t == 0 else 1 - ones for t in y_train])

weights = torch.from_numpy(weights)
sampler = WeightedRandomSampler(weights.type('torch.FloatTensor'), len(weights))

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# MAKE SURE TO ADD shuffle=True IF YOU'RE
# GOING TO SUBMIT
#%%
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True) 

#%%
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(INPUT_SIZE, 550),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(550, 250),
    nn.ReLU(),
    nn.Dropout(0.25),

    nn.Linear(250, 100),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(100, 100),
    nn.ReLU(),

    nn.Linear(100, OUTPUT_SIZE),
    nn.Sigmoid()
)

def torch_f2_loss(output, target):
    tp = torch.sum(output * target).to(torch.float32)
    # tn = torch.sum((1 - output) * (1-target))
    fp = torch.sum((1 - output) * target).to(torch.float32)
    fn = torch.sum(output * (1 - target)).to(torch.float32)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = 5 * p * r / (4 * p + r + 1e-7)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)

    return 1 - torch.mean(f1)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#loss_fn = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0,disable_torch_grad_focal_loss=True)
loss_fn = nn.BCELoss()
#loss_fn = torch_f2_loss
F2_loss = BinaryFBetaScore(beta=2.0) # F2 loss

#%%
# Training loop
model.train()

for epoch in range(NUM_EPOCHS):
    train_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Erase accumulated gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate loss
        loss = loss_fn(output, target.unsqueeze(dim=-1))

        pred = torch.round(output)

        train_loss += F2_loss(pred, target.unsqueeze(dim=-1)).item()

        correct += pred.eq(target.view_as(pred)).sum().item()

        # Backward pass
        loss.backward()
        
        # Weight update
        optimizer.step()
    
    train_loss /= len(train_loader.dataset)

    # Track loss each epoch
    print('Train Epoch: %d  Loss: %.4f Average loss: %.4f, Accuracy: %d/%d (%.4f)' % (epoch + 1, loss.item(), train_loss, correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))

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

print('Test Set: Average loss: %.4f, Accuracy: %d/%d (%.4f)' %
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
