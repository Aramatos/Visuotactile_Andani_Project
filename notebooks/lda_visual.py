# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.linalg import subspace_angles

from pathlib import Path
from utils import *
os.chdir(Path(__file__).parent.parent)
os.getcwd()
# %%
# open an nwb file and grab the units and the stimulus trials

units, trials = load_data("NPBK")

units_good,onsets,offsets,modality,labels,partner = extract_data(units,trials)


bin_edges = (0,0.3)

counts_per_trial_per_neuron = np.zeros( (len(onsets),(len(units_good))), dtype=np.uint8)


for trial_idx, onset in enumerate(onsets):
    for neuron_idx, spikes in enumerate(units_good["spike_times"]):
        spikes_aligned = spikes - onset
        spikes_in_window = spikes_aligned[(spikes_aligned >= bin_edges[0]) & (spikes_aligned <= bin_edges[1])]
        counts_per_trial_per_neuron[trial_idx,neuron_idx] = len(spikes_in_window)


print(counts_per_trial_per_neuron)
print(len(counts_per_trial_per_neuron))

# %%
counts_per_trial_per_neuron.shape


# %%


# run LDA to discriminate cross the visual classes
onsets_visual = trials[trials["modality"]=="visual"]["start_time"]
labels_visual = trials[trials["modality"]=="visual"]["stimulus"]
psth_visual = counts_per_trial_per_neuron[trials["modality"]=="visual"]


lda_3 = LinearDiscriminantAnalysis()
lda_visual=lda_3.fit_transform(psth_visual,labels_visual.values)
lda_visual.shape
print(lda_3.score(psth_visual, labels_visual.values))

plt.figure()
plt.scatter(lda_visual[:,0] ,labels_visual.values)


# %%
color_map = {

    "vA": "purple",
    "vB": "brown",
    "vC": "pink",
    "vD": "gray",
}


modality_color_map = {
    "tactile": "red",
    "visual": "blue",
}

# The row order of each centroid array below is fixed by the order the class
# names are written into it, and nothing in the array itself records that order.
# So the matching list of labels lives right next to it, and every plot indexes
# both with the same row number instead of relying on memory.

classes_visual = ["vA", "vB", "vC", "vD"]

# create centroids for all lda for each class
centroids_lda_3 = np.array([np.mean(lda_visual[labels_visual.values=="vA"], axis=0), np.mean(lda_visual[labels_visual.values=="vB"], axis=0), np.mean(lda_visual[labels_visual.values=="vC"], axis=0), np.mean(lda_visual[labels_visual.values=="vD"], axis=0)])

# %%
# Axis titles. `explained_variance_ratio_` on an LDA is the share of the
# BETWEEN-CLASS separation carried by each discriminant axis — not the share of
# the total variance in the data, which is what the same attribute name means on
# a PCA object. Writing it into the axis title is the only way to tell LD1 from
# LD3 by eye, since all three axes are drawn on the same numeric scale.

percent_visual = 100 * lda_3.explained_variance_ratio_

axis_titles_visual = [
    f"LD1  — {percent_visual[0]:.1f}% ",
    f"LD2  — {percent_visual[1]:.1f}%",
    f"LD3  — {percent_visual[2]:.1f}%",
]

# %%




# %%
# Visual: same construction, separate fit. LD1 here is a different linear
# combination of the same 62 units than LD1 in the tactile figure, and the origin
# is the mean visual trial rather than the mean tactile trial — so a point in one
# figure is not comparable to a point in the other.
traces_visual = []

for row_index, stimulus in enumerate(classes_visual):
    is_this_class = labels_visual.values == stimulus
    scores_this_class = lda_visual[is_this_class]
    centroid_this_class = centroids_lda_3[row_index]

    cloud_trace = go.Scatter3d(
        x=scores_this_class[:, 0],
        y=scores_this_class[:, 1],
        z=scores_this_class[:, 2],
        mode="markers",
        name=f"{stimulus} trials",
        marker=dict(color=color_map[stimulus], size=3, opacity=0.15),
    )
    centroid_trace = go.Scatter3d(
        x=[centroid_this_class[0]],
        y=[centroid_this_class[1]],
        z=[centroid_this_class[2]],
        mode="markers",
        name=f"{stimulus} centroid",
        marker=dict(color=color_map[stimulus], size=12, opacity=1.0,
                    line=dict(color="black", width=3)),
    )

    traces_visual.append(cloud_trace)
    traces_visual.append(centroid_trace)

fig_visual = go.Figure(data=traces_visual)
fig_visual.update_layout(
    title="NPBM visual patterns — LDA discriminant space (in-sample)",
    scene=dict(
        xaxis_title=axis_titles_visual[0],
        yaxis_title=axis_titles_visual[1],
        zaxis_title=axis_titles_visual[2],
        aspectmode="data",
    ),
)
fig_visual.show()


rng = np.random.default_rng(0)


# %%


plt.figure(figsize=(30,10))
plt.imshow(lda_3.scalings_.T,cmap="bwr")
# %%

# obtain the angle in between the first discriminant vectors for all lda
#
# Two things were broken in the one-liner version. `np.dot` on the transposed
# arrays contracted the (1, 62) and (3, 62) shapes against each other, matching 62
# against 3, which raises. And `np.linalg.norm` on a 2D array returns the
# Frobenius norm of the whole matrix, not the length of one axis, so the
# denominator was the wrong quantity even where the shapes happened to line up.
#
# `scalings_` is (n_units, n_axes). A COLUMN is one discriminant axis: a 62-long
# vector of per-unit weights. That column is what an angle is taken between, so
# index the column and leave the array untransposed.

axis_1_visual = lda_3.scalings_[:, 0]


norm_visual = np.linalg.norm(axis_1_visual)

