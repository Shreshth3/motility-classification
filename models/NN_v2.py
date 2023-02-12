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

from datetime  import datetime
from tqdm      import tqdm

from utils import process_data, output_csv

#%%
# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 5

INPUT_SIZE = 6
OUTPUT_SIZE = 1


TRAIN_BASIC_FEATURES_PATH = '../data/train_features_20230210_201924.csv'
TEST_BASIC_FEATURES_PATH = '../data/test_features_20230210_202031.csv'

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE_PATH = f'../output/{TIMESTAMP}_output.csv'

X_train, y_train = process_data(TRAIN_BASIC_FEATURES_PATH)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)


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
    #nn.Dropout(0.3),

    nn.Linear(550, 250),
    nn.ReLU(),

    nn.Linear(250, 100),
    nn.ReLU(),
    #nn.Dropout(0.2),

    nn.Linear(100, 50),
    nn.ReLU(),

    nn.Linear(50, OUTPUT_SIZE),
    nn.Sigmoid()
)

def torch_f2_loss(output, target):
    prod = output * target
    #print(prod)
    #print(torch.min(2 * prod + 3 * output - target + 1))
    return torch.sum(1 - 5 * prod / (2 * prod + 3 * output - target + 1 + 1e-10))

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch_f2_loss
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

#%%
for data, target in test_loader:
    output = model(data)
    pred = torch.round(output)

    print(data[0])
    print(pred[:5])
    print(target[:5])
    break



# %%

output_csv(model, OUTPUT_FILE_PATH, TEST_BASIC_FEATURES_PATH)

# %%
