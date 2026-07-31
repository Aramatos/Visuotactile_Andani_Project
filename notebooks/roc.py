# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.linalg import subspace_angles

from pathlib import Path
from utils import *
os.chdir(Path(__file__).parent.parent)
os.getcwd()
# %%
# open an nwb file and grab the units and the stimulus trials


color_map = {
    "F5": "red",
    "F10": "blue",
    "F20": "green",
    "F∞": "orange",
    "vA": "purple",
    "vB": "brown",
    "vC": "pink",
    "vD": "gray",
}


modality_color_map = {
    "tactile": "red",
    "visual": "blue",
}


units, trials = load_data("NPBM")

units_good,onsets,offsets,modality,labels,partner = extract_data(units,trials)


bin_edges = (0,0.050)

psth_all_trials = np.zeros( (len(onsets),(len(units_good))), dtype=np.uint8)


for trial_idx, onset in enumerate(onsets):
    for neuron_idx, spikes in enumerate(units_good["spike_times"]):
        spikes_aligned = spikes - onset
        spikes_in_window = spikes_aligned[(spikes_aligned >= bin_edges[0]) & (spikes_aligned <= bin_edges[1])]
        psth_all_trials[trial_idx,neuron_idx] = len(spikes_in_window)

np.where(spikes_aligned >0)

spikes_aligned[21930:21930+20]

psth_all_trials[1599,60]


# run LDA to discriminate cross the tactile classes
onsets_tactile = trials[trials["modality"]=="tactile"]["start_time"]
labels_tactile = trials[trials["modality"]=="tactile"]["stimulus"]
psth_tactile_trials = psth_all_trials[trials["modality"]=="tactile"]

# run LDA to discriminate cross the visual classes
onsets_visual = trials[trials["modality"]=="visual"]["start_time"]
labels_visual = trials[trials["modality"]=="visual"]["stimulus"]
psth_visual_trials = psth_all_trials[trials["modality"]=="visual"]

print(psth_all_trials)
print(len(psth_all_trials))



# %%

counts_A= psth_all_trials[trials["stimulus"]=="vB"]
counts_B= psth_all_trials[trials["stimulus"]=="F10"]
print(counts_A[:,0])
print(counts_B[:,0])



# %%
# Modality. The spike count is the score, so no classifier and no train/test
# split — roc_auc_score just sweeps a threshold through it.
# True = visual, so AUC > 0.5 means the unit fires more on visual, < 0.5 more on
# tactile. Both are informative, so rank on the distance from 0.5.
is_tactile = modality.values == "tactile"

n_units = psth_all_trials.shape[1]
auc_modality = np.zeros(n_units)



for neuron_idx in range(n_units):
    counts_this_neuron = psth_all_trials[:, neuron_idx]
    auc_modality[neuron_idx] = roc_auc_score(is_tactile, counts_this_neuron)

auc_modality_strength = np.abs(auc_modality - 0.5)

plt.plot(np.sort(auc_modality), marker="o", color="black")

print("modality AUC: min", round(auc_modality.min(), 3),
      " max", round(auc_modality.max(), 3))

# %%

x_1=np.random.normal(5, 5, 800)
x_2=np.random.normal(10, 5, 800)

x=np.concatenate((x_1, x_2))
y=np.concatenate((np.zeros(800), np.ones(800)))

auc_synth = roc_auc_score(y, x)
print(auc_synth)


# %%
# Null: permuting the labels breaks the trial-label link but leaves each unit's
# count distribution intact. Gives the best-of-62 score when there is no signal.
rng = np.random.default_rng(0)
n_shuffles = 200

best_strength_modality_shuffled = np.zeros(n_shuffles)

for shuffle_idx in range(n_shuffles):
    is_visual_shuffled = rng.permutation(is_tactile)

    strengths_this_shuffle = np.zeros(n_units)
    for neuron_idx in range(n_units):
        counts_this_neuron = psth_all_trials[:, neuron_idx]
        auc_shuffled = roc_auc_score(is_visual_shuffled, counts_this_neuron)
        strengths_this_shuffle[neuron_idx] = np.abs(auc_shuffled - 0.5)

    best_strength_modality_shuffled[shuffle_idx] = strengths_this_shuffle.max()

threshold_modality = np.percentile(best_strength_modality_shuffled, 95)
n_above_modality = (auc_modality_strength > threshold_modality).sum()

print("shuffled best |AUC - 0.5|, 95th pct =", round(threshold_modality, 3))
print("real units clearing it:", int(n_above_modality), "of", n_units)

# %%
order_modality = np.argsort(-auc_modality_strength)
sorted_strength_modality = auc_modality_strength[order_modality]

plt.figure(figsize=(12, 4))
plt.bar(range(n_units), sorted_strength_modality, color="black")
plt.axhline(threshold_modality, color="red", linestyle="--",
            label=f"shuffled best, 95th pct = {threshold_modality:.3f}")
plt.xlabel("unit, sorted")
plt.ylabel("|AUC - 0.5|")
plt.title("NPBM — single-unit tactile vs visual")
plt.legend()

# %%
# The curves themselves. Counts run 0-14, so there are few distinct thresholds
# and the ROC is a coarse staircase.
best_unit_modality = order_modality[0]
median_unit_modality = order_modality[n_units // 2]

counts_best_modality = psth_all_trials[:, best_unit_modality]
counts_median_modality = psth_all_trials[:, median_unit_modality]

fpr_best, tpr_best, thresholds_best = roc_curve(is_tactile, counts_best_modality)
fpr_median, tpr_median, thresholds_median = roc_curve(is_tactile, counts_median_modality)

plt.figure(figsize=(5, 5))
plt.plot(fpr_best, tpr_best, marker="o",
         label=f"unit {best_unit_modality}, AUC = {auc_modality_strength[best_unit_modality]:.3f}")
plt.plot(fpr_median, tpr_median, marker="o",
         label=f"unit {median_unit_modality}, AUC = {auc_modality_strength[median_unit_modality]:.3f}")
plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="chance")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("ROC, tactile vs visual")
plt.legend()

# %%
# Tactile patterns. ROC compares two things, so four patterns become six
# two-way questions; a unit counts as informative if it separates any one pair.
pairs_tactile = [
    ("F5", "F10"), ("F5", "F20"), ("F5", "F∞"),
    ("F10", "F20"), ("F10", "F∞"), ("F20", "F∞"),
]
n_pairs = len(pairs_tactile)

auc_tactile = np.zeros((n_units, n_pairs))

for pair_idx, (class_a, class_b) in enumerate(pairs_tactile):
    is_pair = (labels_tactile.values == class_a) | (labels_tactile.values == class_b)
    is_class_b = labels_tactile.values[is_pair] == class_b
    counts_pair = psth_tactile_trials[is_pair]

    for neuron_idx in range(n_units):
        counts_this_neuron = counts_pair[:, neuron_idx]
        auc_tactile[neuron_idx, pair_idx] = roc_auc_score(is_class_b, counts_this_neuron)

auc_tactile_strength = np.abs(auc_tactile - 0.5)
best_pair_tactile = auc_tactile_strength.max(axis=1)

print("tactile best |AUC - 0.5| per unit: max", round(best_pair_tactile.max(), 3),
      " median", round(np.median(best_pair_tactile), 3))

# %%
# Same null, now over 62 units x 6 pairs, so the bar is higher. ~30 s.
best_strength_tactile_shuffled = np.zeros(n_shuffles)

for shuffle_idx in range(n_shuffles):
    labels_tactile_shuffled = rng.permutation(labels_tactile.values)

    strengths_this_shuffle = np.zeros((n_units, n_pairs))
    for pair_idx, (class_a, class_b) in enumerate(pairs_tactile):
        is_pair = (labels_tactile_shuffled == class_a) | (labels_tactile_shuffled == class_b)
        is_class_b = labels_tactile_shuffled[is_pair] == class_b
        counts_pair = psth_tactile_trials[is_pair]

        for neuron_idx in range(n_units):
            counts_this_neuron = counts_pair[:, neuron_idx]
            auc_shuffled = roc_auc_score(is_class_b, counts_this_neuron)
            strengths_this_shuffle[neuron_idx, pair_idx] = np.abs(auc_shuffled - 0.5)

    best_strength_tactile_shuffled[shuffle_idx] = strengths_this_shuffle.max()

threshold_tactile = np.percentile(best_strength_tactile_shuffled, 95)
n_above_tactile = (best_pair_tactile > threshold_tactile).sum()

print("shuffled best |AUC - 0.5|, 95th pct =", round(threshold_tactile, 3))
print("real units clearing it:", int(n_above_tactile), "of", n_units)

# %%
# vmin/vmax set by hand so white lands on 0.5, not on the middle of the data.
max_deviation_tactile = auc_tactile_strength.max()
pair_names_tactile = [f"{class_a} vs {class_b}" for class_a, class_b in pairs_tactile]

plt.figure(figsize=(6, 12))
plt.imshow(auc_tactile, cmap="bwr", aspect="auto",
           vmin=0.5 - max_deviation_tactile, vmax=0.5 + max_deviation_tactile)
plt.colorbar(label="AUC")
plt.xticks(range(n_pairs), pair_names_tactile, rotation=45, ha="right")
plt.ylabel("unit index")
plt.title("NPBM tactile — single-unit pairwise AUC")

# %%
order_tactile = np.argsort(-best_pair_tactile)
sorted_best_tactile = best_pair_tactile[order_tactile]

plt.figure(figsize=(12, 4))
plt.bar(range(n_units), sorted_best_tactile, color="black")
plt.axhline(threshold_tactile, color="red", linestyle="--",
            label=f"shuffled best, 95th pct = {threshold_tactile:.3f}")
plt.xlabel("unit, sorted")
plt.ylabel("best |AUC - 0.5| over 6 pairs")
plt.title("NPBM tactile — any unit separating any pattern pair?")
plt.legend()

# %%
# Visual patterns. Same block, own cell so it keeps its own intermediates.
pairs_visual = [
    ("vA", "vB"), ("vA", "vC"), ("vA", "vD"),
    ("vB", "vC"), ("vB", "vD"), ("vC", "vD"),
]

auc_visual = np.zeros((n_units, n_pairs))

for pair_idx, (class_a, class_b) in enumerate(pairs_visual):
    is_pair = (labels_visual.values == class_a) | (labels_visual.values == class_b)
    is_class_b = labels_visual.values[is_pair] == class_b
    counts_pair = psth_visual_trials[is_pair]

    for neuron_idx in range(n_units):
        counts_this_neuron = counts_pair[:, neuron_idx]
        auc_visual[neuron_idx, pair_idx] = roc_auc_score(is_class_b, counts_this_neuron)

auc_visual_strength = np.abs(auc_visual - 0.5)
best_pair_visual = auc_visual_strength.max(axis=1)

print("visual best |AUC - 0.5| per unit: max", round(best_pair_visual.max(), 3),
      " median", round(np.median(best_pair_visual), 3))

# %%
best_strength_visual_shuffled = np.zeros(n_shuffles)

for shuffle_idx in range(n_shuffles):
    labels_visual_shuffled = rng.permutation(labels_visual.values)

    strengths_this_shuffle = np.zeros((n_units, n_pairs))
    for pair_idx, (class_a, class_b) in enumerate(pairs_visual):
        is_pair = (labels_visual_shuffled == class_a) | (labels_visual_shuffled == class_b)
        is_class_b = labels_visual_shuffled[is_pair] == class_b
        counts_pair = psth_visual_trials[is_pair]

        for neuron_idx in range(n_units):
            counts_this_neuron = counts_pair[:, neuron_idx]
            auc_shuffled = roc_auc_score(is_class_b, counts_this_neuron)
            strengths_this_shuffle[neuron_idx, pair_idx] = np.abs(auc_shuffled - 0.5)

    best_strength_visual_shuffled[shuffle_idx] = strengths_this_shuffle.max()

threshold_visual = np.percentile(best_strength_visual_shuffled, 95)
n_above_visual = (best_pair_visual > threshold_visual).sum()

print("shuffled best |AUC - 0.5|, 95th pct =", round(threshold_visual, 3))
print("real units clearing it:", int(n_above_visual), "of", n_units)

# %%
max_deviation_visual = auc_visual_strength.max()
pair_names_visual = [f"{class_a} vs {class_b}" for class_a, class_b in pairs_visual]

plt.figure(figsize=(6, 12))
plt.imshow(auc_visual, cmap="bwr", aspect="auto",
           vmin=0.5 - max_deviation_visual, vmax=0.5 + max_deviation_visual)
plt.colorbar(label="AUC")
plt.xticks(range(n_pairs), pair_names_visual, rotation=45, ha="right")
plt.ylabel("unit index")
plt.title("NPBM visual — single-unit pairwise AUC")

# %%
order_visual = np.argsort(-best_pair_visual)
sorted_best_visual = best_pair_visual[order_visual]

plt.figure(figsize=(12, 4))
plt.bar(range(n_units), sorted_best_visual, color="black")
plt.axhline(threshold_visual, color="red", linestyle="--",
            label=f"shuffled best, 95th pct = {threshold_visual:.3f}")
plt.xlabel("unit, sorted")
plt.ylabel("best |AUC - 0.5| over 6 pairs")
plt.title("NPBM visual — any unit separating any pattern pair?")
plt.legend()

# %%
