# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.linalg import subspace_angles
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
from utils import *
os.chdir(Path(__file__).parent.parent)
os.getcwd()
# I think that's to say which neurons are more important and less important but not doesn't say anything else. Strengthen is still nothing else. Oh wait, so a more informative neuron would be closer to one no or closer to zero no no for zero in the top oh in the top yeah yeah yeah closer to five is less informative because there's equal chance of coming from if it's closer to zero then it means that it's on this end so it has a higher false positive so that's not good you want to be in this corner it's closer or is it it is a C score chance number one you see score is yeah or like this plot doesn't show you directly you have to calculate from these curves but yeah zero point seven eight is a Uc score of this line so higher Auc is better yeah but why isn't a lower like I thought Ah no, it's probability that I randomly pick visual from had more spikes. than a randomly pick tactile trial if it's zero percent that means that there's no probability then why does the one with the least probab# %%
# open an nwb file and grab the units and the stimulus trials

units, trials = load_data("NPBM")

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
# run LDA to discriminate across modalities
lda_1 = LinearDiscriminantAnalysis()
x_new=lda_1.fit_transform(counts_per_trial_per_neuron,modality.values)
x_new.shape

plt.figure()
plt.scatter(x_new,modality.values)
print(lda_1.score(counts_per_trial_per_neuron, modality.values))


# run LDA to discriminate cross the tactile classes
onsets_tactile = trials[trials["modality"]=="tactile"]["start_time"]
labels_tactile = trials[trials["modality"]=="tactile"]["stimulus"]
psth_tactile = counts_per_trial_per_neuron[trials["modality"]=="tactile"]

lda_2 = LinearDiscriminantAnalysis()
lda_tactile=lda_2.fit_transform(psth_tactile,labels_tactile.values)
lda_tactile.shape
print(lda_2.score(psth_tactile, labels_tactile.values))

plt.figure()
plt.scatter(lda_tactile[:,0],labels_tactile.values)

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

# The row order of each centroid array below is fixed by the order the class
# names are written into it, and nothing in the array itself records that order.
# So the matching list of labels lives right next to it, and every plot indexes
# both with the same row number instead of relying on memory.
classes_modality = ["tactile", "visual"]
classes_tactile = ["F5", "F10", "F20", "F∞"]
classes_visual = ["vA", "vB", "vC", "vD"]

# create centroids for all lda for each class
centroids_lda_1 = np.array([np.mean(x_new[modality.values=="tactile"], axis=0), np.mean(x_new[modality.values=="visual"], axis=0)])
centroids_lda_2 = np.array([np.mean(lda_tactile[labels_tactile.values=="F5"], axis=0), np.mean(lda_tactile[labels_tactile.values=="F10"], axis=0), np.mean(lda_tactile[labels_tactile.values=="F20"], axis=0), np.mean(lda_tactile[labels_tactile.values=="F∞"], axis=0)])
centroids_lda_3 = np.array([np.mean(lda_visual[labels_visual.values=="vA"], axis=0), np.mean(lda_visual[labels_visual.values=="vB"], axis=0), np.mean(lda_visual[labels_visual.values=="vC"], axis=0), np.mean(lda_visual[labels_visual.values=="vD"], axis=0)])

# %%
# Axis titles. `explained_variance_ratio_` on an LDA is the share of the
# BETWEEN-CLASS separation carried by each discriminant axis — not the share of
# the total variance in the data, which is what the same attribute name means on
# a PCA object. Writing it into the axis title is the only way to tell LD1 from
# LD3 by eye, since all three axes are drawn on the same numeric scale.
percent_tactile = 100 * lda_2.explained_variance_ratio_
percent_visual = 100 * lda_3.explained_variance_ratio_

axis_titles_tactile = [
    f"LD1 (1st discriminant) — {percent_tactile[0]:.1f}% of class separation",
    f"LD2 (2nd discriminant) — {percent_tactile[1]:.1f}%",
    f"LD3 (3rd discriminant) — {percent_tactile[2]:.1f}%",
]
axis_titles_visual = [
    f"LD1 (1st discriminant) — {percent_visual[0]:.1f}% of class separation",
    f"LD2 (2nd discriminant) — {percent_visual[1]:.1f}%",
    f"LD3 (3rd discriminant) — {percent_visual[2]:.1f}%",
]

# %%
# Tactile: the four stimulus patterns in the full 3D discriminant space.
# Two traces per class — the trial cloud at opacity 0.15, the centroid at 1.0.
# The faint cloud is the whole point: at full opacity 200 dots per class occlude
# each other, every class looks compact and cleanly separated, and the figure
# creates exactly the impression the held-out accuracy says is not true.
traces_tactile = []

for row_index, stimulus in enumerate(classes_tactile):
    is_this_class = labels_tactile.values == stimulus
    scores_this_class = lda_tactile[is_this_class]
    centroid_this_class = centroids_lda_2[row_index]

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

    traces_tactile.append(cloud_trace)
    traces_tactile.append(centroid_trace)

fig_tactile = go.Figure(data=traces_tactile)
# aspectmode="data" keeps the three axes on one scale. That matters here: LDA
# normalises every discriminant axis to within-class SD = 1, so on-screen
# distance between two centroids IS a d'. Plotly's default stretches the axes to
# fill a cube, which silently destroys that property.
fig_tactile.update_layout(
    title="NPBM tactile patterns — LDA discriminant space (in-sample)",
    scene=dict(
        xaxis_title=axis_titles_tactile[0],
        yaxis_title=axis_titles_tactile[1],
        zaxis_title=axis_titles_tactile[2],
        aspectmode="data",
    ),
)
fig_tactile.show()

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

# %%
# Modality: 2 classes, so LDA returns exactly ONE axis (n_classes - 1 = 1) and
# there is no second axis to plot against. Each class gets its own horizontal
# band, and the vertical spread within a band is random jitter added here purely
# so that 800 dots do not stack into one opaque line. Only the x position is data;
# the y position carries nothing.
rng = np.random.default_rng(0)

traces_modality = []

for row_index, this_modality in enumerate(classes_modality):
    is_this_class = modality.values == this_modality
    scores_this_class = x_new[is_this_class, 0]
    centroid_this_class = centroids_lda_1[row_index, 0]

    n_trials_this_class = len(scores_this_class)
    jitter = rng.uniform(-0.15, 0.15, size=n_trials_this_class)
    y_this_class = row_index + jitter

    cloud_trace = go.Scatter(
        x=scores_this_class,
        y=y_this_class,
        mode="markers",
        name=f"{this_modality} trials",
        marker=dict(color=modality_color_map[this_modality], size=4, opacity=0.15),
    )
    centroid_trace = go.Scatter(
        x=[centroid_this_class],
        y=[row_index],
        mode="markers",
        name=f"{this_modality} centroid",
        marker=dict(color=modality_color_map[this_modality], size=18, opacity=1.0,
                    line=dict(color="black", width=3)),
    )

    traces_modality.append(cloud_trace)
    traces_modality.append(centroid_trace)

fig_modality = go.Figure(data=traces_modality)
# The "100%" in the x title is not a result — with one axis it has nowhere else
# to go. Read the separation off the gap between the two solid centroids, in
# units of within-class SD.
fig_modality.update_layout(
    title="NPBM tactile vs visual — the single LDA axis (in-sample)",
    xaxis_title="LD1 (the only discriminant axis) — 100% by construction",
    yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=classes_modality,
               title="vertical jitter only — carries no information"),
)
fig_modality.show()

# %%

plt.figure(figsize=(30,10))
plt.imshow(lda_1.scalings_.T,cmap="bwr")

plt.figure(figsize=(30,10))
plt.imshow(lda_2.scalings_.T,cmap="bwr")

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
axis_1_modality = lda_1.scalings_[:, 0]
axis_1_tactile = lda_2.scalings_[:, 0]
axis_1_visual = lda_3.scalings_[:, 0]

norm_modality = np.linalg.norm(axis_1_modality)
norm_tactile = np.linalg.norm(axis_1_tactile)
norm_visual = np.linalg.norm(axis_1_visual)

dot_modality_tactile = np.dot(axis_1_modality, axis_1_tactile)
dot_modality_visual = np.dot(axis_1_modality, axis_1_visual)
dot_tactile_visual = np.dot(axis_1_tactile, axis_1_visual)

cos_modality_tactile = dot_modality_tactile / (norm_modality * norm_tactile)
cos_modality_visual = dot_modality_visual / (norm_modality * norm_visual)
cos_tactile_visual = dot_tactile_visual / (norm_tactile * norm_visual)

# The sign of a discriminant axis is arbitrary — refit on other trials and a whole
# axis can flip end for end without any geometry changing. So a cosine of +0.30 and
# one of -0.30 describe the same relationship, and only the ACUTE angle carries
# meaning. Taking the absolute value collapses that ambiguity and puts every
# result in [0, 90] degrees.
abs_cos_modality_tactile = np.abs(cos_modality_tactile)
abs_cos_modality_visual = np.abs(cos_modality_visual)
abs_cos_tactile_visual = np.abs(cos_tactile_visual)

# clip only guards against a cosine arriving as 1.0000000002 from rounding, which
# would make arccos return nan
clipped_modality_tactile = np.clip(abs_cos_modality_tactile, -1.0, 1.0)
clipped_modality_visual = np.clip(abs_cos_modality_visual, -1.0, 1.0)
clipped_tactile_visual = np.clip(abs_cos_tactile_visual, -1.0, 1.0)

radians_modality_tactile = np.arccos(clipped_modality_tactile)
radians_modality_visual = np.arccos(clipped_modality_visual)
radians_tactile_visual = np.arccos(clipped_tactile_visual)

angle_modality_tactile = np.degrees(radians_modality_tactile)
angle_modality_visual = np.degrees(radians_modality_visual)
angle_tactile_visual = np.degrees(radians_tactile_visual)

print("acute angle between LD1 weight vectors, in degrees")
print("  modality vs tactile:", angle_modality_tactile)
print("  modality vs visual :", angle_modality_visual)
print("  tactile  vs visual :", angle_tactile_visual)

# %%
# The angles above mean nothing without a chance level, and the chance level is
# not 90 degrees with any tolerance you would guess. Two RANDOM directions in
# 62-dimensional space are already very nearly orthogonal: their cosine has mean 0
# and standard deviation 1/sqrt(62) = 0.127. After the absolute value, a cosine
# only clears the 95th percentile of chance above 1.96 * 0.127 = 0.249 — an angle
# of about 76 degrees. So an angle of 80 degrees is NOT "these two readouts are
# nearly independent"; it is indistinguishable from two vectors drawn at random.
n_units = lda_1.scalings_.shape[0]
cos_sd_by_chance = 1.0 / np.sqrt(n_units)
cos_threshold_95 = 1.96 * cos_sd_by_chance
radians_threshold_95 = np.arccos(cos_threshold_95)
angle_threshold_95 = np.degrees(radians_threshold_95)

print("chance level: any angle above", round(angle_threshold_95, 1),
      "degrees is what two unrelated random directions look like")

# %%
# LD1-vs-LD1 is the wrong comparison for the two four-class fits in any case.
# LD1, LD2 and LD3 are ordered only by eigenvalue, and lda_3's eigenvalues are
# 0.39 / 0.35 / 0.26 — close enough that which direction gets called "LD1" is
# essentially arbitrary, so an angle measured to it is not a stable quantity.
#
# The stable question ignores the choice of basis inside each fit and asks how the
# two discriminant SUBSPACES sit relative to each other. `subspace_angles` returns
# min(k1, k2) angles in descending order: 90 degrees means the subspaces share
# nothing along that direction, 0 degrees means they coincide. Unlike the cosine
# above, this is invariant to both sign flips and rotations within each fit.
principal_radians_modality_tactile = subspace_angles(lda_1.scalings_, lda_2.scalings_)
principal_radians_modality_visual = subspace_angles(lda_1.scalings_, lda_3.scalings_)
principal_radians_tactile_visual = subspace_angles(lda_2.scalings_, lda_3.scalings_)

principal_angles_modality_tactile = np.degrees(principal_radians_modality_tactile)
principal_angles_modality_visual = np.degrees(principal_radians_modality_visual)
principal_angles_tactile_visual = np.degrees(principal_radians_tactile_visual)

print("principal angles between discriminant subspaces, in degrees (largest first)")
print("  modality vs tactile:", principal_angles_modality_tactile)
print("  modality vs visual :", principal_angles_modality_visual)
print("  tactile  vs visual :", principal_angles_tactile_visual)

# %%
# `argsort` was being fed to cosine_similarity instead of the weight vectors
# themselves — that measured similarity between rank-order permutations, not
# between the discriminant axes. Pass the raw scalings_ columns directly.

cosine_similarity(lda_1.scalings_[:, 0].reshape(1,-1), lda_2.scalings_[:, 0].reshape(1,-1))

# %%
cosine_similarity(lda_2.scalings_[:, 1].reshape(1,-1), lda_3.scalings_[:, 1].reshape(1,-1))

# %%


cos_sim_axes = dict()
for i_comp in range(lda_2.scalings_.shape[-1]):
    cos_sim_axes[f'lda1_to_lda2comp{i_comp}'] = cosine_similarity(np.abs(lda_1.scalings_[:, 0].reshape(1,-1)), np.abs(lda_2.scalings_[:, i_comp].reshape(1,-1)))

    cos_sim_axes[f'lda1_to_lda3comp{i_comp}'] = cosine_similarity(np.abs(lda_1.scalings_[:, 0].reshape(1,-1)), np.abs(lda_3.scalings_[:, i_comp].reshape(1,-1)))

    cos_sim_axes[f'lda2_to_lda3comp{i_comp}'] = cosine_similarity(np.abs(lda_2.scalings_[:, 0].reshape(1,-1)), np.abs(lda_3.scalings_[:, i_comp].reshape(1,-1)))

# %%
cos_sim_axes

# %%
