#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import os.path as op
from pathlib   import Path
from glob      import glob
from tqdm      import tqdm
from datetime  import datetime

import csv
import json
import numpy as np
import matplotlib.pyplot as plt

# %%

# Constants
TYPE = "train"
# DATA_LOCATION = '../input/2023-cs155-proj1/train.json'
DATA_LOCATION = f'../data/{TYPE}.json'

# In[2]:

with open(DATA_LOCATION, 'r') as f:
    track_data = json.load(f)

# How many tracks are there?
print(f"n_tracks = {len(track_data.keys())}")

# What do the track Unique IDs (UIDs) look like?
track_uids = list(track_data.keys())
print(f"5 Example Track IDs = {track_uids[:5]}")

# What fields are avaiable for each track?
example_uid = track_uids[0]
print(f"Per-track keys = {track_data[example_uid].keys()}")

# What do the (t, x, y) track coordinates look like?
example_coords = track_data[track_uids[0]]['txy']
example_coords = np.array(example_coords)
np.set_printoptions(threshold=10)
print(f"Coordinate array = \n{example_coords}")

# What does the label look like?
example_label = track_data[track_uids[0]]['label']
print(f"Label = {example_label}")


# ## Implementing Features
#
# Now that we can load the tracks, we need to calculate features. At its core, we need to calculate a single *number* that represents the *entire* track. The features we will implement here are:
#
# * Average per-step speed
# * Standard deviation of the per-step speed
# * Physical length of the track
# * Distance between the start and end of the track
# * Time duration of the track
#
# We will follow the same function pattern for each of these features to simplify the process.

# In[3]:

def template_feature(coords):
    """Name of the Feature

    A short description of the feature goes here. Equations can be useful.

    Parameters
    ----------
    coords: array
        A numpy array containing the (t, x, y) coordinates of the track.

    Returns
    -------
    float
        The feature value for the entire array.

    """

    return 0

def mean_step_speed(coords):
    """Mean step speed of the entire track.

    The average per-step speed. Basically the average of distances between points adjacent in time.

    Returns
    -------
    float
        The average step speed.
    """

    speeds = []

    for i in range(1, coords.shape[0]):
        # Previous coordinate location
        prev = coords[i-1, 1:]
        # Current coordinate location
        curr = coords[i, 1:]

        # Speed in pixels per frame
        curr_speed = np.linalg.norm(curr - prev)

        # Accumulate per-step speeds into a list
        speeds.append(curr_speed)

    # Return the average of the speeds
    return np.mean(speeds)


def stddev_step_speed(coords):
    """Standard deviation of the step speed of the entire track.

    The standard deviation of the per-step speed.

    Returns
    -------
    float
        The stddev of the step speed.
    """

    speeds = []

    for i in range(1, coords.shape[0]):
        # Previous coordinate location
        prev = coords[i-1, 1:]
        # Current coordinate location
        curr = coords[i, 1:]

        # Speed in pixels per frame
        curr_speed = np.linalg.norm(curr - prev)

        # Accumulate per-step speeds into a list
        speeds.append(curr_speed)

    # Return the standard deviation of the speeds
    return np.std(speeds)


def track_length(coords):
    """Standard deviation of the step speed of the entire track.

    The standard deviation of the per-step speed.

    Returns
    -------
    float
        The length of the entire track.
    """

    lengths = []

    for i in range(1, coords.shape[0]):
        # Previous coordinate location
        prev = coords[i-1,1:]
        # Current coordinate location
        curr = coords[i,1:]

        # Speed in pixels per frame
        step_length = np.linalg.norm(curr - prev)

        # Accumulate per-step speeds into a list
        lengths.append(step_length)

    # Return the sum of the lengths
    return np.sum(lengths)


def e2e_distance(coords):
    """End-to-end distance of the track.

    The distance from the start and the end of the given track.

    Returns
    -------
    float
        The end-to-end distance of the entire track.
    """

    # Start and end of the track
    start = coords[0, 1:]
    end = coords[-1, 1:]

    # Return the distance
    return np.linalg.norm(end-start)


def duration(coords):
    """Duration of the track.

    The time duration of the track.

    Returns
    -------
    int
        The end-to-end duration of the entire track.
    """

    # Start and end times of the track
    start_t = coords[0, 0]
    end_t = coords[-1, 0]

    # Return the difference
    return end_t - start_t


######################################
# Implement your own features below! #
######################################


def deviation_from_line(coords):
    """Name of the Feature

    A short description of the feature goes here. Equations can be useful.

    Parameters
    ----------
    coords: array
        A numpy array containing the (t, x, y) coordinates of the track.

    Returns
    -------
    float
        The feature value for the entire array.

    """

    return 0

# ## Implementing Feature cont.
#
# Implementing more features will follow the same pattern as the functions above. You will either:
#
# * Measure per-timestep metrics and calculate some statistic about the distribution
# * Calculate some global feature over the entire track
#
# There are many possible options for additional features.
# Consider visually plotting tracks to identify what geometric properties of a track indicate motility vs. non-motility.

# ## Generating the Feature CSV
#
# **NOTE:** You will also have to do this step when you're predicting the test set for submission to Kaggle.

# In[4]:

FEATURE_LIST = [mean_step_speed, stddev_step_speed, track_length, e2e_distance, duration]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# OUTPUT_FILENAME = f"/kaggle/working/{TYPE}_features_{TIMESTAMP}.csv"
OUTPUT_FILENAME = f"../data/{TYPE}_features_{TIMESTAMP}.csv"


# Generate the feature csv
header = ['uid', 'label']
for featfunc in FEATURE_LIST:
    header.append(featfunc.__name__)

features = []

track_uids = track_data.keys()
for uid in track_uids:
    curr_row = {
        'uid': uid,
        'label': track_data[uid]['label']
    }

    for featfunc in FEATURE_LIST:
        curr_row[featfunc.__name__] = featfunc(np.array(track_data[uid]['txy']))

    features.append(curr_row)

with open(OUTPUT_FILENAME, 'w') as f:
    writer = csv.DictWriter(f, fieldnames = header)
    writer.writeheader()
    for r in features:
        writer.writerow(r)

print("Written to:", OUTPUT_FILENAME)


# %%
