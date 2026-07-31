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

# import elephant
import elephant.spike_train_surrogates as surrogates
from sklearn.decomposition import PCA
import neo
import quantities as pq

rng = np.random.default_rng(0)


# build a surrogate version of psth_all_trials: for every neuron, dither its
# real spikes (jitter each spike time within +/- 50 ms, keeping the neuron's
# own rate) to destroy any trial-locked structure, then count the dithered
# spikes in the same [0, 0.3] s post-onset window used for the real PSTH

psth_all_trials_surrogate = np.zeros((len(onsets), len(units_good)), dtype=np.uint8)

for neuron_idx, spike_times in enumerate(units_good["spike_times"]):

    # elephant needs a neo.SpikeTrain, not a raw array
    # trials["stop_time"].max() is the wrong bound here: the recording runs long
    # after the last trial, and this neuron's own spikes go past it, so t_stop
    # has to come from the spike train itself
    t_start = 0.0
    t_stop = spike_times.max()
    spiketrain = neo.SpikeTrain(spike_times * pq.s, t_start=t_start * pq.s, t_stop=t_stop * pq.s)

    # jitter each spike within +/- 50 ms to make one surrogate for this neuron
    dither = 0.05 * pq.s
    surrogate_spiketrain = surrogates.dither_spikes(spiketrain, edges=True, dither=dither, n_surrogates=1)[0]
    surrogate_spike_times = surrogate_spiketrain.magnitude  # back to plain seconds array

    for trial_idx, onset in enumerate(onsets):
        spikes_aligned = surrogate_spike_times - onset
        spikes_in_window = spikes_aligned[(spikes_aligned >= bin_edges[0]) & (spikes_aligned <= bin_edges[1])]
        psth_all_trials_surrogate[trial_idx, neuron_idx] = len(spikes_in_window)

pca_all_surrogate = PCA()
pca_all_surrogate_scores = pca_all_surrogate.fit_transform(psth_all_trials_surrogate)

#%%
pca_all_surrogate.explained_variance_ratio_

# %%
# a second, different null: the cross-neuron shuffle. Instead of dithering each
# neuron's own spikes in time, keep every neuron's column of psth_all_trials
# exactly as it is (same 300 ms post-onset count on every trial) but reorder
# the trials independently for each neuron. So neuron 3's count of 4 spikes,
# wherever it happened, is still a 4 somewhere in neuron 3's column, but it is
# no longer necessarily the same trial that neurons 1 and 2 had their own
# counts on. That destroys whatever made neurons co-vary across trials while
# leaving each neuron's own across-trial distribution of counts untouched.

n_trials_all, n_neurons_all = psth_all_trials.shape
psth_all_trials_neuron_shuffled = np.zeros_like(psth_all_trials)

for neuron_idx in range(n_neurons_all):
    trial_order = rng.permutation(n_trials_all)
    psth_all_trials_neuron_shuffled[:, neuron_idx] = psth_all_trials[trial_order, neuron_idx]

# %%
# check the shuffle did what it claims: same per-neuron counts, weaker
# cross-neuron correlation

totals_before_shuffle = psth_all_trials.sum(axis=0)
totals_after_shuffle = psth_all_trials_neuron_shuffled.sum(axis=0)
totals_identical = np.array_equal(totals_before_shuffle, totals_after_shuffle)

correlation_before_shuffle = np.corrcoef(psth_all_trials, rowvar=False)
correlation_after_shuffle = np.corrcoef(psth_all_trials_neuron_shuffled, rowvar=False)
off_diagonal_all = ~np.eye(n_neurons_all, dtype=bool)

mean_correlation_before_shuffle = np.abs(correlation_before_shuffle[off_diagonal_all]).mean()
mean_correlation_after_shuffle = np.abs(correlation_after_shuffle[off_diagonal_all]).mean()

print(f"every neuron's total spike count unchanged by the shuffle: {totals_identical}")
print(f"mean |correlation| between neuron pairs, real     : {mean_correlation_before_shuffle:.4f}")
print(f"mean |correlation| between neuron pairs, shuffled : {mean_correlation_after_shuffle:.4f}")

# %%
# 3-component PCA on the shuffled matrix, so it can be dropped into the same
# kind of scatter plot as the real and dithered-surrogate scores below

pca_all_neuron_shuffled = PCA(n_components=3)
pca_all_neuron_shuffled_scores = pca_all_neuron_shuffled.fit_transform(psth_all_trials_neuron_shuffled)

# %%
# now the actual comparison: fit PCA with every component kept (not just 3) on
# the real matrix and on the cross-neuron-shuffled matrix, and reduce each
# full spectrum to two numbers, the same two critiques/dimensionality.py uses
# for the same question:
#   PCs for 95% of the variance  -- their own cutoff
#   participation ratio, PR = (sum of eigenvalues)^2 / sum of eigenvalues^2
#                             -- answers to the whole spectrum shape instead
#                                of one cutoff; flat spectrum -> PR near the
#                                neuron count, concentrated spectrum -> PR small

pca_all_full = PCA()
pca_all_full.fit(psth_all_trials)
variance_real = pca_all_full.explained_variance_
cumulative_real = np.cumsum(pca_all_full.explained_variance_ratio_)
pc95_real = int(np.searchsorted(cumulative_real, 0.95) + 1)
pr_real = float(variance_real.sum() ** 2 / np.square(variance_real).sum())

pca_shuffled_full = PCA()
pca_shuffled_full.fit(psth_all_trials_neuron_shuffled)
variance_shuffled = pca_shuffled_full.explained_variance_
cumulative_shuffled = np.cumsum(pca_shuffled_full.explained_variance_ratio_)
pc95_shuffled = int(np.searchsorted(cumulative_shuffled, 0.95) + 1)
pr_shuffled = float(variance_shuffled.sum() ** 2 / np.square(variance_shuffled).sum())

print(f"neurons in the matrix            : {n_neurons_all}")
print(f"PCs for 95% variance, real       : {pc95_real}")
print(f"PCs for 95% variance, shuffled   : {pc95_shuffled}")
print(f"participation ratio, real        : {pr_real:.1f}")
print(f"participation ratio, shuffled    : {pr_shuffled:.1f}")

# something varies from trial to trial that moves many neurons together, rather than each neuron just having its own independent count noise.
# participation ratio shuffled is similar to what is expected for flat.
'''
What "flat" means concretely. PCA’s eigenvalues are just the variances along a set of orthogonal directions in neuron-space, sorted largest to smallest. "Flat" means those variances are all roughly the same size — no direction stands out. Geometrically, the cloud of population activity (one point per trial, in 62-neuron space) looks like a round blob, equally spread in every direction
The fact that the real participation ratio is much lower than the shuffled one means: A concentrated, low-dimensional spectrum is equally consistent with several different stories, and PR can't distinguish them:
- genuine stimulus-driven population coding (different patterns push the population along different directions — encouraging, this is the "real signal" reading)
- slow session drift or state changes
'''


# %%

print(psth_all_trials.shape)
print(psth_tactile_trials.shape)
print(psth_visual_trials.shape)
pca_all = PCA()
pca_all_scores = pca_all.fit_transform(psth_all_trials)



pca_tactile = PCA()
pca_tactile_scores = pca_tactile.fit_transform(psth_tactile_trials)

pca_visual = PCA()
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
    marker=dict(size=2, color=[color_map[label] for label in labels_tactile.values]),
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

fig = go.Figure(data=[go.Scatter3d(
    x=pca_all_surrogate_scores[:,0],
    y=pca_all_surrogate_scores[:,1],
    z=pca_all_surrogate_scores[:,2],
    mode='markers',
    marker=dict(size=2, color=[color_map[label] for label in labels.values]),
    text=labels.values
)])
fig.update_layout(title='PCA All Trials (Surrogate)', scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
fig.show()

fig = go.Figure(data=[go.Scatter3d(
    x=pca_all_neuron_shuffled_scores[:,0],
    y=pca_all_neuron_shuffled_scores[:,1],
    z=pca_all_neuron_shuffled_scores[:,2],
    mode='markers',
    marker=dict(size=2, color=[color_map[label] for label in labels.values]),
    text=labels.values
)])
fig.update_layout(title='PCA All Trials (Cross-Neuron Shuffle)', scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
fig.show()

# %%

plt.plot(np.cumsum(pca_all_full.explained_variance_ratio_), label='Real Data')
plt.plot(np.cumsum(pca_shuffled_full.explained_variance_ratio_), label='Shuffled Data')
plt.plot(np.cumsum(pca_all_surrogate.explained_variance_ratio_), label='Surrogate Data')
plt.plot(np.cumsum(pca_visual.explained_variance_ratio_), label='Visual Data')
plt.plot(np.cumsum(pca_tactile.explained_variance_ratio_), label='Tactile Data')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Component Analysis')
plt.legend()
plt.show()

# %%
np.cumsum(pca_all_surrogate.explained_variance_ratio_)
# %%
