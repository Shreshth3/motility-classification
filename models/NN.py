import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import os
import os.path as op
from pathlib   import Path
from glob      import glob
from tqdm      import tqdm
from datetime  import datetime

import csv
import json





# train_dataset = datasets.MNIST('./data', train=True, download=True,  # Downloads into a directory ../data
#                                transform=transforms.ToTensor())
# test_dataset = datasets.MNIST('./data', train=False, download=False,  # No need to download again
#                               transform=transforms.ToTensor())


# model = nn.Sequential(
#     nn.Flatten(),  
#     nn.Linear(784, 500),
#     nn.ReLU(),
#     nn.Dropout(0.3),
#     nn.Linear(500, 250),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(250, 150),
#     nn.ReLU(),
#     nn.Dropout(0.13),
#     nn.Linear(150,50),
#     nn.Linear(50, 10)
# )
# print(model)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.CrossEntropyLoss()

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True) 

# # Some layers, such as Dropout, behave differently during training
# model.train()

# for epoch in range(20):
#     for batch_idx, (data, target) in enumerate(train_loader):
#         # Erase accumulated gradients
#         optimizer.zero_grad()

#         # Forward pass
#         output = model(data)

#         # Calculate loss
#         loss = loss_fn(output, target)

#         # Backward pass
#         loss.backward()
        
#         # Weight update
#         optimizer.step()

#     # Track loss each epoch
#     print('Train Epoch: %d  Loss: %.4f' % (epoch + 1,  loss.item()))


#     # Putting layers like Dropout into evaluation mode
# model.eval()

# test_loss = 0
# correct = 0

# # Turning off automatic differentiation
# with torch.no_grad():
#     for data, target in test_loader:
#         output = model(data)
#         test_loss += loss_fn(output, target).item()  # Sum up batch loss
#         pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max class score
#         correct += pred.eq(target.view_as(pred)).sum().item()

# test_loss /= len(test_loader.dataset)

# print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.4f)' %
#       (test_loss, correct, len(test_loader.dataset),
#        100. * correct / len(test_loader.dataset)))