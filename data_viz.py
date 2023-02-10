#!/usr/bin/env python
# coding: utf-8

# In[1]:
get_ipython().run_line_magic('matplotlib', 'inline')
import json
import numpy as np
import matplotlib.pyplot as plt
import random


# In[2]:

# DATA_LOCATION = '../input/2023-cs155-proj1/train.json'
DATA_LOCATION = 'data/train.json'

# Load the training data
with open(DATA_LOCATION, 'r') as f:
    train_data = json.load(f)

# Identify unique IDs (UIDs) that are labeled motile and nonmotile
sim_motile_uids = [x for x in train_data.keys() if train_data[x]['label'] == 1 and 'sim' in x]
sim_nonmotile_uids = [x for x in train_data.keys() if train_data[x]['label'] == 0 and 'sim' in x]
lab_motile_uids = [x for x in train_data.keys() if train_data[x]['label'] == 1 and 'lab' in x]
lab_nonmotile_uids = [x for x in train_data.keys() if train_data[x]['label'] == 0 and 'lab' in x]


# In[3]:


def plot_tracks(tracklist, title):
    # plot given tracks
    fig, ax = plt.subplots(figsize=(5,5))
    for t in tracklist:
        ax.plot(t[:,1], t[:,2])
    
    ax.set_xlim([0,1024])
    ax.set_ylim([1024, 0])
    ax.set_aspect(1.0)
    ax.set_title(title)
    
    fig.show()


# In[4]:


# Plot 10 simulated motile tracks
plot_uids = random.choices(sim_motile_uids, k=5)
to_plot = [np.array(train_data[u]['txy']) for u in plot_uids]

plot_tracks(to_plot, 'Simulated Motile Tracks')

# Plot 10 simulated nonmotile tracks
plot_uids = random.choices(sim_nonmotile_uids, k=5)
to_plot = [np.array(train_data[u]['txy']) for u in plot_uids]

plot_tracks(to_plot, 'Simulated Nonmotile Tracks')


# In[5]:


# Plot 10 lab motile tracks
plot_uids = random.choices(lab_motile_uids, k=5)
to_plot = [np.array(train_data[u]['txy']) for u in plot_uids]

plot_tracks(to_plot, 'Lab Motile Tracks')

# Plot 10 lab nonmotile tracks
plot_uids = random.choices(lab_nonmotile_uids, k=5)
to_plot = [np.array(train_data[u]['txy']) for u in plot_uids]

plot_tracks(to_plot, 'Lab Nonmotile Tracks')


# In[ ]:

