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

def process_data(csv, columns_to_drop=['uid', 'label'], return_uids=False):
    df = pd.read_csv(csv)
    # df = df[df['uid'].str.contains('sim') == False]
    #print(len(df))

    #print(df[len(df)-5:])

    df.dropna(inplace=True)
    Y = df['label'].copy()
    df.drop(columns=columns_to_drop, inplace=True)

    if return_uids:
        uids = df.pop('uid')

        return df_to_tensor(df), torch.FloatTensor(Y.to_numpy()), np.array(uids)

    return df_to_tensor(df), torch.FloatTensor(Y.to_numpy())

def compute_line_of_best_fit(X, y):
  return LinearRegression().fit(X, y)

def compute_residuals(X, y, line_of_best_fit):
  predictions = line_of_best_fit.predict(X)
  residuals = y - predictions

  return residuals


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

def plot_tracks(tracklist, title):
    # plot given tracks
    fig, ax = plt.subplots(figsize=(5,5))
    for t in tracklist:
        ax.plot(t[:,1], t[:,2])
    
    ax.set_xlim([-500,1250])
    ax.set_ylim([1250, -500])
    ax.set_aspect(1.0)
    ax.set_title(title)
    
    fig.show()

def get_misclassified_points(model, test_loader):
    all_misclassified_points = []

    for data, target in test_loader:
        output = model(data)
        pred = torch.round(output)
        pred = pred.squeeze()

        is_misclassified = pred != target

        cur_batch_misclassified_points = data[is_misclassified]
        all_misclassified_points.extend([*cur_batch_misclassified_points])

    return all_misclassified_points

def compute_vector(point_one, point_two):
    x1, y1 = point_one
    x2, y2 = point_two

    return np.array([x2 - x1, y2 - y1])

def compute_angle(v1, v2):
    dot = np.dot(v1, v2)

    v1_magnitude = np.linalg.norm(v1)
    v2_magnitude = np.linalg.norm(v2)

    if v1_magnitude == 0 or v2_magnitude == 0:
        return 0

    cos_angle = dot / (v1_magnitude * v2_magnitude)

    # There are some edge cases where we get
    # -1.0000000000000002 which
    # is not in the domain of arccos
    cos_angle = min(cos_angle, 1.0)
    cos_angle = max(cos_angle, -1.0)

    angle_in_rad = np.arccos(cos_angle)

    angle_in_deg = np.degrees(angle_in_rad)

    return angle_in_deg



