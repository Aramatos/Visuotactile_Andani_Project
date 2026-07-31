# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from pathlib import Path
from utils import *
os.chdir(Path(__file__).parent.parent)
os.getcwd()
# %%
# open an nwb file and grab the units and the stimulus trials

color_map = {
    "F5": "red",
    "F10": "blue",
    "F∞": "green",
    "F20": "orange",
    "vA": "purple",
    "vB": "brown",
    "vC": "pink",
    "vD": "gray",
}

units, trials = load_data("NPBM")

units_good,onsets,offsets,modality,labels,partner = extract_data(units,trials)


bin_edges = (0,0.3)

psth_all_trials = np.zeros( (len(onsets),(len(units_good))), dtype=np.uint8)


for trial_idx, onset in enumerate(onsets):
    for neuron_idx, spikes in enumerate(units_good["spike_times"]):
        spikes_aligned = spikes - onset
        spikes_in_window = spikes_aligned[(spikes_aligned >= bin_edges[0]) & (spikes_aligned <= bin_edges[1])]
        psth_all_trials[trial_idx,neuron_idx] = len(spikes_in_window)


# run LDA to discriminate cross the tactile classes
onsets_tactile = trials[trials["modality"]=="tactile"]["start_time"]
labels_tactile = trials[trials["modality"]=="tactile"]["stimulus"]
psth_tactile_trials = psth_all_trials[trials["modality"]=="tactile"]

# run LDA to discriminate cross the visual classes
onsets_visual = trials[trials["modality"]=="visual"]["start_time"]
labels_visual = trials[trials["modality"]=="visual"]["stimulus"]
psth_visual_trials = psth_all_trials[trials["modality"]=="visual"]

# %%
import elephant
import elephant.spike_train_surrogates as surrogates
from sklearn.decomposition import PCA

























# %%

print(psth_all_trials.shape)
print(psth_tactile_trials.shape)
print(psth_visual_trials.shape)
pca_all = PCA(n_components=3)
pca_all_scores = pca_all.fit_transform(psth_all_trials)

pca_tactile = PCA(n_components=3)
pca_tactile_scores = pca_tactile.fit_transform(psth_tactile_trials)

pca_visual = PCA(n_components=3)
pca_visual_scores = pca_visual.fit_transform(psth_visual_trials)

# %%
fig = go.Figure(data=[go.Scatter3d(
    x=pca_all_scores[:,0],
    y=pca_all_scores[:,1],
    z=pca_all_scores[:,2],
    mode='markers',
    marker=dict(size=2, color=[color_map[label] for label in labels.values]),
    text=labels.values
)])
fig.update_layout(title='PCA All Trials', scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
fig.show()

fig = go.Figure(data=[go.Scatter3d(
    x=pca_tactile_scores[:,0],
    y=pca_tactile_scores[:,1],
    z=pca_tactile_scores[:,2],
    mode='markers',
    marker=dict(size=2 color=[color_map[label] for label in labels_tactile.values]),
    text=labels_tactile.values
)])
fig.update_layout(title='PCA Tactile Trials', scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
fig.show()

fig = go.Figure(data=[go.Scatter3d(
    x=pca_visual_scores[:,0],
    y=pca_visual_scores[:,1],
    z=pca_visual_scores[:,2],
    mode='markers',
    marker=dict(size=2, color=[color_map[label] for label in labels_visual.values]),
    text=labels_visual.values
)])
fig.update_layout(title='PCA Visual Trials', scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
fig.show()

# %%
