import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

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

from sklearn.linear_model import LinearRegression

def df_to_tensor(df):
    return torch.Tensor(df.to_numpy())

def process_data(csv):
    df = pd.read_csv(csv)

    df.dropna(inplace=True)

    Y = df['label'].copy()

    df.drop(columns=['uid', 'label'], inplace=True)


    return df_to_tensor(df), torch.FloatTensor(Y.to_numpy())

def compute_line_of_best_fit(X, y):
  return LinearRegression().fit(X, y)

def compute_residuals(X, y, line_of_best_fit):
  predictions = line_of_best_fit.predict(X)
  residuals = y - predictions

  return residuals


    