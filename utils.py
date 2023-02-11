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


def df_to_tensor(df):
    return torch.Tensor(df.to_numpy())


def process_data(csv):
    df = pd.read_csv(csv)

    df.dropna(inplace=True)
    Y = df['label'].copy()
    df.drop(columns=['uid', 'label'], inplace=True)

    return df_to_tensor(df), torch.FloatTensor(Y.to_numpy())


def process_no_y_data(csv):
    df = pd.read_csv(csv)

    UIDs = df['uid'].copy()
    df.drop(columns=['uid', 'label'], inplace=True)
    df.dropna(inplace=True)

    return df_to_tensor(df), UIDs.to_numpy()


def output_csv(model, filename, test_data):
    real_X_test, Uids_test = process_no_y_data(test_data)
    Uids_test = np.expand_dims(Uids_test, axis=1)

    # get predictions from the model
    predictions_np = torch.round(model(real_X_test)).detach().numpy()

    # combine the two columns. (477, 1) and (477, 1)
    combined = np.concatenate((Uids_test, predictions_np), axis=1)

    # convert to df
    export_df = pd.DataFrame(combined)

    # get rid of the .0
    export_df[1] = pd.to_numeric(export_df[1], downcast='integer')

    export_df.to_csv(filename, header=["UID", "label"], index=False)
