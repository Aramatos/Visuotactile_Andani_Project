# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, KFold, permutation_test_score, cross_val_score
from scipy.linalg import subspace_angles, orth

from pathlib import Path
from utils import *
os.chdir(Path(__file__).parent.parent)
os.getcwd()


# %% spike count per trial per neuron, 0-300 ms after onset

units, trials = load_data("NPBM")
units_good, onsets, offsets, modality, labels, partner = extract_data(units, trials)

bin_edges = (0, 0.3)
counts_per_trial_per_neuron = np.zeros((len(onsets), len(units_good)), dtype=np.uint8)

for trial_idx, onset in enumerate(onsets):
    for neuron_idx, spikes in enumerate(units_good["spike_times"]):
        spikes_aligned = spikes - onset
        spikes_in_window = spikes_aligned[(spikes_aligned >= bin_edges[0])
                                          & (spikes_aligned <= bin_edges[1])]
        counts_per_trial_per_neuron[trial_idx, neuron_idx] = len(spikes_in_window)

print(counts_per_trial_per_neuron.shape, "(trial, neuron)")


# %% colours and class names

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

classes_modality = ["tactile", "visual"]
classes_tactile = ["F5", "F10", "F20", "F∞"]
classes_visual = ["vA", "vB", "vC", "vD"]

rng = np.random.default_rng(0)

OUT_DIR = "figures/lda"
os.makedirs(OUT_DIR, exist_ok=True)


# %% three fits. 2 classes give 1 discriminant axis, 4 classes give 3.

lda_1 = LinearDiscriminantAnalysis()
x_new = lda_1.fit_transform(counts_per_trial_per_neuron, modality.values)
print("modality", x_new.shape, lda_1.score(counts_per_trial_per_neuron, modality.values))

onsets_tactile = trials[trials["modality"] == "tactile"]["start_time"]
labels_tactile = trials[trials["modality"] == "tactile"]["stimulus"]
psth_tactile = counts_per_trial_per_neuron[trials["modality"] == "tactile"]

lda_2 = LinearDiscriminantAnalysis()
lda_tactile = lda_2.fit_transform(psth_tactile, labels_tactile.values)
print("tactile ", lda_tactile.shape, lda_2.score(psth_tactile, labels_tactile.values))

onsets_visual = trials[trials["modality"] == "visual"]["start_time"]
labels_visual = trials[trials["modality"] == "visual"]["stimulus"]
psth_visual = counts_per_trial_per_neuron[trials["modality"] == "visual"]

lda_3 = LinearDiscriminantAnalysis()
lda_visual = lda_3.fit_transform(psth_visual, labels_visual.values)
print("visual  ", lda_visual.shape, lda_3.score(psth_visual, labels_visual.values))


# %% does each fit actually decode? the scores above are IN SAMPLE, and 62
# features on 800 trials overfits, so nothing downstream means anything until
# these three come out above chance.

shuffled_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
blocked_folds = KFold(n_splits=5, shuffle=False)

decoding_problems = [
    ("modality", counts_per_trial_per_neuron, modality.values, 0.50),
    ("tactile ", psth_tactile, labels_tactile.values, 0.25),
    ("visual  ", psth_visual, labels_visual.values, 0.25),
]

# fraction of trials classified correctly. held-out is the real one; shuffled is
# what chance actually is; blocked folds keep drift out; shrunk would rescue it
# if overfitting were the limit. p is the share of 200 label shuffles that
# matched it, floored at 1/201.
print(f"{'':10s}{'in-sample':>10s}{'held-out':>10s}{'blocked':>9s}"
      f"{'shrunk':>9s}{'shuffled':>10s}{'p':>8s}{'chance':>8s}")
for name, features, target, nominal_chance in decoding_problems:
    plain = LinearDiscriminantAnalysis()
    shrunk = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    in_sample = plain.fit(features, target).score(features, target)
    held_out = cross_val_score(plain, features, target, cv=shuffled_folds).mean()
    blocked = cross_val_score(plain, features, target, cv=blocked_folds).mean()
    regularised = cross_val_score(shrunk, features, target, cv=shuffled_folds).mean()
    observed, permuted, p_value = permutation_test_score(
        plain, features, target, cv=shuffled_folds, n_permutations=200,
        random_state=0, n_jobs=-1)

    print(f"{name:10s}{in_sample:10.3f}{held_out:10.3f}{blocked:9.3f}"
          f"{regularised:9.3f}{permuted.mean():10.3f}{p_value:8.3f}"
          f"{nominal_chance:8.2f}")


# %%
# on an LDA, explained_variance_ratio_ is the share of the BETWEEN-class
# separation per axis, not the share of total variance as on a PCA
percent_tactile = 100 * lda_2.explained_variance_ratio_
percent_visual = 100 * lda_3.explained_variance_ratio_

axis_titles_tactile = [
    f"LD1  — {percent_tactile[0]:.1f}% ",
    f"LD2  — {percent_tactile[1]:.1f}%",
    f"LD3  — {percent_tactile[2]:.1f}%",
]
axis_titles_visual = [
    f"LD1  — {percent_visual[0]:.1f}% ",
    f"LD2  — {percent_visual[1]:.1f}%",
    f"LD3  — {percent_visual[2]:.1f}%",
]


# %% LD1 vs LD2. equal aspect because LDA sets within-class SD to 1, so a
# distance on the plot is a d-prime; clouds at alpha 0.15 or they occlude.

# modality has only one axis, so y here is jitter
fig_lda_modality, ax_modality = plt.subplots(figsize=(7, 3.5))

for row_index, this_modality in enumerate(classes_modality):
    is_this_class = modality.values == this_modality
    scores_this_class = x_new[is_this_class, 0]
    jitter = rng.uniform(-0.15, 0.15, size=len(scores_this_class))
    y_this_class = row_index + jitter
    ax_modality.scatter(scores_this_class, y_this_class, s=8, alpha=0.15,
                        color=modality_color_map[this_modality], label=this_modality)

ax_modality.set_yticks([0, 1])
ax_modality.set_yticklabels(classes_modality)
ax_modality.set_xlabel("LD1 — 100% (2 classes give only one axis)")
ax_modality.set_ylabel("jitter\n(no information)")
ax_modality.set_title(f"NPBM tactile vs visual — in-sample "
                      f"{lda_1.score(counts_per_trial_per_neuron, modality.values):.2f}")
ax_modality.legend(markerscale=2, framealpha=1)
fig_lda_modality.tight_layout()
fig_lda_modality.savefig(f"{OUT_DIR}/lda_modality_LD1.png", dpi=150, facecolor="white")
plt.show()

fig_lda_tactile, ax_tactile = plt.subplots(figsize=(6, 6))

for stimulus in classes_tactile:
    is_this_class = labels_tactile.values == stimulus
    scores_this_class = lda_tactile[is_this_class]
    ax_tactile.scatter(scores_this_class[:, 0], scores_this_class[:, 1], s=8,
                       alpha=0.15, color=color_map[stimulus])

for stimulus in classes_tactile:
    is_this_class = labels_tactile.values == stimulus
    centroid_this_class = lda_tactile[is_this_class].mean(axis=0)
    ax_tactile.scatter(centroid_this_class[0], centroid_this_class[1], s=200,
                       color=color_map[stimulus], edgecolor="black", linewidth=2,
                       label=stimulus, zorder=3)

ax_tactile.set_xlabel(axis_titles_tactile[0])
ax_tactile.set_ylabel(axis_titles_tactile[1])
ax_tactile.set_aspect("equal")
ax_tactile.set_title(f"NPBM tactile patterns — LD1 vs LD2, in-sample "
                     f"{lda_2.score(psth_tactile, labels_tactile.values):.2f}")
ax_tactile.legend(title="centroids", framealpha=1)
fig_lda_tactile.tight_layout()
fig_lda_tactile.savefig(f"{OUT_DIR}/lda_tactile_LD1_LD2.png", dpi=150, facecolor="white")
plt.show()

fig_lda_visual, ax_visual = plt.subplots(figsize=(6, 6))

for stimulus in classes_visual:
    is_this_class = labels_visual.values == stimulus
    scores_this_class = lda_visual[is_this_class]
    ax_visual.scatter(scores_this_class[:, 0], scores_this_class[:, 1], s=8,
                      alpha=0.15, color=color_map[stimulus])

for stimulus in classes_visual:
    is_this_class = labels_visual.values == stimulus
    centroid_this_class = lda_visual[is_this_class].mean(axis=0)
    ax_visual.scatter(centroid_this_class[0], centroid_this_class[1], s=200,
                      color=color_map[stimulus], edgecolor="black", linewidth=2,
                      label=stimulus, zorder=3)

ax_visual.set_xlabel(axis_titles_visual[0])
ax_visual.set_ylabel(axis_titles_visual[1])
ax_visual.set_aspect("equal")
ax_visual.set_title(f"NPBM visual patterns — LD1 vs LD2, in-sample "
                    f"{lda_3.score(psth_visual, labels_visual.values):.2f}")
ax_visual.legend(title="centroids", framealpha=1)
fig_lda_visual.tight_layout()
fig_lda_visual.savefig(f"{OUT_DIR}/lda_visual_LD1_LD2.png", dpi=150, facecolor="white")
plt.show()


# %% the same three fits in plotly, full 3D discriminant space

centroids_lda_1 = np.array([np.mean(x_new[modality.values == "tactile"], axis=0),
                            np.mean(x_new[modality.values == "visual"], axis=0)])
centroids_lda_2 = np.array([np.mean(lda_tactile[labels_tactile.values == "F5"], axis=0),
                            np.mean(lda_tactile[labels_tactile.values == "F10"], axis=0),
                            np.mean(lda_tactile[labels_tactile.values == "F20"], axis=0),
                            np.mean(lda_tactile[labels_tactile.values == "F∞"], axis=0)])
centroids_lda_3 = np.array([np.mean(lda_visual[labels_visual.values == "vA"], axis=0),
                            np.mean(lda_visual[labels_visual.values == "vB"], axis=0),
                            np.mean(lda_visual[labels_visual.values == "vC"], axis=0),
                            np.mean(lda_visual[labels_visual.values == "vD"], axis=0)])

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
fig_modality.update_layout(
    title="NPBM tactile vs visual — the single LDA axis (in-sample)",
    xaxis_title="LD1 (the only discriminant axis) — 100% by construction",
    yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=classes_modality,
               title="vertical jitter only — carries no information"),
)
fig_modality.show()

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
        marker=dict(color=color_map[stimulus], size=3, opacity=0.25),
    )
    centroid_trace = go.Scatter3d(
        x=[centroid_this_class[0]],
        y=[centroid_this_class[1]],
        z=[centroid_this_class[2]],
        mode="markers",
        name=f"{stimulus} centroid",
        marker=dict(color=color_map[stimulus], size=5, opacity=1.0,
                    line=dict(color="black", width=3)),
    )

    traces_tactile.append(cloud_trace)
    traces_tactile.append(centroid_trace)

fig_tactile = go.Figure(data=traces_tactile)
# aspectmode="data" keeps the axes on one scale, so a centroid distance is a d'
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
        marker=dict(color=color_map[stimulus], size=3, opacity=0.20),
    )
    centroid_trace = go.Scatter3d(
        x=[centroid_this_class[0]],
        y=[centroid_this_class[1]],
        z=[centroid_this_class[2]],
        mode="markers",
        name=f"{stimulus} centroid",
        marker=dict(color=color_map[stimulus], size=5, opacity=1.0,
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


# %% per-unit weights of each fit

plt.figure(figsize=(30, 10))
plt.imshow(lda_1.scalings_.T, cmap="bwr")

plt.figure(figsize=(30, 10))
plt.imshow(lda_2.scalings_.T, cmap="bwr")

plt.figure(figsize=(30, 10))
plt.imshow(lda_3.scalings_.T, cmap="bwr")


# %% ===================================================================
#    HOW DIFFERENT ARE THE THREE SPACES?
#    ===================================================================
#
# Two candidate vectors per contrast, both 62 long:
#   readout weight   w = Sigma^-1 (mu_a - mu_b)   what LDA fits, whitened by the
#                                                 within-class noise covariance
#   coding direction d = mu_a - mu_b              where the population actually moves
#
# Here they sit 57.8 degrees apart, so which one is measured decides the answer.
# The question is about where activity goes, so it is d.

counts_float = counts_per_trial_per_neuron.astype(float)
stimulus_names = trials["stimulus"].to_numpy()
in_tactile = modality.values == "tactile"
in_visual = modality.values == "visual"
all_trials = np.ones(len(counts_float), dtype=bool)

mean_tactile_trial = counts_float[in_tactile].mean(axis=0)
mean_visual_trial = counts_float[in_visual].mean(axis=0)
delta_mu = mean_tactile_trial - mean_visual_trial

# pooled within-class covariance: subtract each trial's own class mean, so only
# trial-to-trial noise is left
residuals = counts_float.copy()
residuals[in_tactile] = residuals[in_tactile] - mean_tactile_trial
residuals[in_visual] = residuals[in_visual] - mean_visual_trial
within_class_covariance = (residuals.T @ residuals) / (len(counts_float) - 2)

weight_from_sklearn = lda_1.scalings_[:, 0]
weight_from_formula = np.linalg.solve(within_class_covariance, delta_mu)

cosine = np.dot(weight_from_sklearn, weight_from_formula)
cosine = cosine / (np.linalg.norm(weight_from_sklearn) * np.linalg.norm(weight_from_formula))
angle_formula_vs_sklearn = np.degrees(np.arccos(np.clip(abs(cosine), -1.0, 1.0)))

cosine = np.dot(weight_from_sklearn, delta_mu)
cosine = cosine / (np.linalg.norm(weight_from_sklearn) * np.linalg.norm(delta_mu))
angle_weight_vs_coding = np.degrees(np.arccos(np.clip(abs(cosine), -1.0, 1.0)))

print(f"scalings_ is Sigma^-1 delta_mu   : {angle_formula_vs_sklearn:.3f} deg apart")
print(f"readout weight vs coding direction: {angle_weight_vs_coding:.1f} deg apart")


# %% the two coding subspaces


def modality_coding_direction(counts, is_tactile, is_visual, trial_mask):
    """mean(tactile) - mean(visual), as an (n_units, 1) column."""
    mean_tactile = counts[trial_mask & is_tactile].mean(axis=0)
    mean_visual = counts[trial_mask & is_visual].mean(axis=0)
    difference = mean_tactile - mean_visual
    return difference.reshape(-1, 1)


def pattern_coding_subspace(counts, stimulus_names, in_modality, trial_mask):
    """The 4 class means within one modality, centred so only how the patterns
    differ from EACH OTHER survives. 4 centred points span 3 dims."""
    selected = trial_mask & in_modality
    names_present = sorted(set(stimulus_names[selected]))
    class_means = np.zeros((len(names_present), counts.shape[1]))
    for row_index, name in enumerate(names_present):
        this_class = selected & (stimulus_names == name)
        class_means[row_index] = counts[this_class].mean(axis=0)
    centered_means = class_means - class_means.mean(axis=0)
    return orth(centered_means.T)


coding_modality = modality_coding_direction(counts_float, in_tactile, in_visual, all_trials)
coding_tactile = pattern_coding_subspace(counts_float, stimulus_names, in_tactile, all_trials)
coding_visual = pattern_coding_subspace(counts_float, stimulus_names, in_visual, all_trials)

print("shapes:", coding_modality.shape, coding_tactile.shape, coding_visual.shape)


# %% principal angles.
#
# Two subspaces of dim k1, k2 meet at min(k1, k2) angles: the first is the closest
# any direction in A gets to any direction in B, the next is the same after
# deleting that pair, and so on. Smallest = what they share, largest = what is
# private. With a 1D side, cos^2 of the one angle is the share of that vector
# sitting inside the other subspace.

coding_radians_modality_tactile = subspace_angles(coding_modality, coding_tactile)
coding_radians_modality_visual = subspace_angles(coding_modality, coding_visual)
coding_radians_tactile_visual = subspace_angles(coding_tactile, coding_visual)

print("coding directions, degrees")
print("  modality vs tactile:", np.degrees(coding_radians_modality_tactile))
print("  modality vs visual :", np.degrees(coding_radians_modality_visual))
print("  tactile  vs visual :", np.degrees(coding_radians_tactile_visual))

share_inside_tactile = np.cos(coding_radians_modality_tactile[0]) ** 2
share_inside_visual = np.cos(coding_radians_modality_visual[0]) ** 2
print(f"  modality direction is {100 * share_inside_tactile:.0f}% inside tactile, "
      f"{100 * share_inside_visual:.0f}% inside visual")

# same measure on the LDA weights, for contrast
print("readout weights, degrees")
print("  modality vs tactile:", np.degrees(subspace_angles(lda_1.scalings_, lda_2.scalings_)))
print("  modality vs visual :", np.degrees(subspace_angles(lda_1.scalings_, lda_3.scalings_)))
print("  tactile  vs visual :", np.degrees(subspace_angles(lda_2.scalings_, lda_3.scalings_)))


# %% FLOOR — random subspaces of matched dimension. In 62 dims unrelated is not
# 90 degrees, and a 1D vector gets three chances against a 3D subspace.

rng = np.random.default_rng(0)
n_units = counts_float.shape[1]
n_draws = 2000

floor_1_vs_3 = np.zeros(n_draws)
for draw in range(n_draws):
    random_a = rng.standard_normal((n_units, 1))
    random_b = rng.standard_normal((n_units, 3))
    floor_1_vs_3[draw] = np.degrees(subspace_angles(random_a, random_b)).min()

floor_3_vs_3 = np.zeros(n_draws)
for draw in range(n_draws):
    random_a = rng.standard_normal((n_units, 3))
    random_b = rng.standard_normal((n_units, 3))
    floor_3_vs_3[draw] = np.degrees(subspace_angles(random_a, random_b)).min()

print(f"floor 1D vs 3D: {np.median(floor_1_vs_3):.1f}  (5th pct "
      f"{np.percentile(floor_1_vs_3, 5):.1f})")
print(f"floor 3D vs 3D: {np.median(floor_3_vs_3):.1f}  (5th pct "
      f"{np.percentile(floor_3_vs_3, 5):.1f})")


# %% the isotropic floor above is too easy to beat. Four noisy class means are
# not four random directions: they inherit the population's own covariance and
# land in its high-variance directions, which BOTH modalities share. So the
# honest floor keeps the data and destroys only the labels — shuffle which trial
# had which stimulus, rebuild the same subspaces, remeasure.

n_shuffles = 200
shuffled_modality_tactile = np.zeros(n_shuffles)
shuffled_modality_visual = np.zeros(n_shuffles)
shuffled_tactile_visual = np.zeros(n_shuffles)

for shuffle_index in range(n_shuffles):
    fake_stimulus = stimulus_names.copy()
    fake_stimulus[in_tactile] = rng.permutation(stimulus_names[in_tactile])
    fake_stimulus[in_visual] = rng.permutation(stimulus_names[in_visual])
    fake_modality_order = rng.permutation(len(counts_float))
    fake_tactile = in_tactile[fake_modality_order]
    fake_visual = ~fake_tactile

    null_modality = modality_coding_direction(counts_float, fake_tactile, fake_visual, all_trials)
    null_tactile = pattern_coding_subspace(counts_float, fake_stimulus, in_tactile, all_trials)
    null_visual = pattern_coding_subspace(counts_float, fake_stimulus, in_visual, all_trials)

    shuffled_modality_tactile[shuffle_index] = np.degrees(subspace_angles(null_modality, null_tactile)).min()
    shuffled_modality_visual[shuffle_index] = np.degrees(subspace_angles(null_modality, null_visual)).min()
    shuffled_tactile_visual[shuffle_index] = np.degrees(subspace_angles(null_tactile, null_visual)).min()

print(f"label-shuffled floor, modality vs tactile: {np.median(shuffled_modality_tactile):.1f}")
print(f"label-shuffled floor, modality vs visual : {np.median(shuffled_modality_visual):.1f}")
print(f"label-shuffled floor, tactile  vs visual : {np.median(shuffled_tactile_visual):.1f}")


# %% CEILING — the same quantity from two disjoint halves. An angle cannot beat
# its own reproducibility. The split also stops the modality direction and the
# tactile subspace from sharing trials, which would inflate their alignment.
# Halves are stratified by stimulus so both contain all eight classes.

n_splits = 200

cross_modality_tactile = np.zeros(n_splits)
cross_modality_visual = np.zeros(n_splits)
cross_tactile_visual = np.zeros(n_splits)
ceiling_modality = np.zeros(n_splits)
ceiling_tactile = np.zeros(n_splits)
ceiling_visual = np.zeros(n_splits)

for split in range(n_splits):
    half_a = np.zeros(len(counts_float), dtype=bool)
    for name in sorted(set(stimulus_names)):
        rows_this_class = np.flatnonzero(stimulus_names == name)
        shuffled = rng.permutation(rows_this_class)
        half_a[shuffled[: len(shuffled) // 2]] = True
    half_b = ~half_a

    modality_a = modality_coding_direction(counts_float, in_tactile, in_visual, half_a)
    modality_b = modality_coding_direction(counts_float, in_tactile, in_visual, half_b)
    tactile_a = pattern_coding_subspace(counts_float, stimulus_names, in_tactile, half_a)
    tactile_b = pattern_coding_subspace(counts_float, stimulus_names, in_tactile, half_b)
    visual_a = pattern_coding_subspace(counts_float, stimulus_names, in_visual, half_a)
    visual_b = pattern_coding_subspace(counts_float, stimulus_names, in_visual, half_b)

    cross_modality_tactile[split] = np.degrees(subspace_angles(modality_a, tactile_b)).min()
    cross_modality_visual[split] = np.degrees(subspace_angles(modality_a, visual_b)).min()
    cross_tactile_visual[split] = np.degrees(subspace_angles(tactile_a, visual_b)).min()

    ceiling_modality[split] = np.degrees(subspace_angles(modality_a, modality_b)).min()
    ceiling_tactile[split] = np.degrees(subspace_angles(tactile_a, tactile_b)).min()
    ceiling_visual[split] = np.degrees(subspace_angles(visual_a, visual_b)).min()


# %% read the observed angle against its ceiling and its floor

print(f"{'':22s}{'observed':>10s}{'ceiling':>9s}{'shuf':>8s}{'random':>8s}")
print(f"{'modality vs tactile':22s}{np.median(cross_modality_tactile):10.1f}"
      f"{max(np.median(ceiling_modality), np.median(ceiling_tactile)):9.1f}"
      f"{np.median(shuffled_modality_tactile):8.1f}{np.median(floor_1_vs_3):8.1f}")
print(f"{'modality vs visual':22s}{np.median(cross_modality_visual):10.1f}"
      f"{max(np.median(ceiling_modality), np.median(ceiling_visual)):9.1f}"
      f"{np.median(shuffled_modality_visual):8.1f}{np.median(floor_1_vs_3):8.1f}")
print(f"{'tactile vs visual':22s}{np.median(cross_tactile_visual):10.1f}"
      f"{max(np.median(ceiling_tactile), np.median(ceiling_visual)):9.1f}"
      f"{np.median(shuffled_tactile_visual):8.1f}{np.median(floor_3_vs_3):8.1f}")
print(f"\nself-consistency: modality {np.median(ceiling_modality):.1f}, "
      f"tactile {np.median(ceiling_tactile):.1f}, visual {np.median(ceiling_visual):.1f}")


# %% the same thing as two distributions. left panel 1D vs 3D, right 3D vs 3D,
# so every curve in a panel is a comparison of the same shape. Grey is unrelated
# (random subspaces), dashed is a subspace against ITSELF from the other half of
# the trials, filled colour is the real cross-comparison.

bins = np.arange(0, 91, 1.5)
GREY = "#999999"
BLUE = "#0072B2"       # anything tactile
ORANGE = "#D55E00"     # anything visual
GREEN = "#009E73"      # tactile against visual

fig_angles, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

shuffled_1_vs_3 = np.concatenate([shuffled_modality_tactile, shuffled_modality_visual])

ax_left.hist(shuffled_1_vs_3, bins=bins, density=True, color=GREY, alpha=0.55,
             label="labels shuffled (no information)")
ax_left.hist(cross_modality_tactile, bins=bins, density=True, color=BLUE, alpha=0.65,
             label="modality vs tactile")
ax_left.hist(cross_modality_visual, bins=bins, density=True, color=ORANGE, alpha=0.65,
             label="modality vs visual")
ax_left.hist(ceiling_modality, bins=bins, density=True, histtype="step", lw=1.8,
             linestyle="--", color="black",
             label="modality vs itself (the limit)")
ax_left.set_title("one direction vs a 3D subspace")
ax_left.set_xlabel("smallest principal angle (degrees)")
ax_left.set_ylabel("density")
ax_left.legend(fontsize=8.5, framealpha=1)

ax_right.hist(shuffled_tactile_visual, bins=bins, density=True, color=GREY, alpha=0.55,
              label="labels shuffled (no information)")
ax_right.hist(cross_tactile_visual, bins=bins, density=True, color=GREEN, alpha=0.65,
              label="tactile vs visual")
ax_right.hist(ceiling_tactile, bins=bins, density=True, histtype="step", lw=1.8,
              linestyle="--", color=BLUE, label="tactile vs itself (the limit)")
ax_right.hist(ceiling_visual, bins=bins, density=True, histtype="step", lw=1.8,
              linestyle="--", color=ORANGE, label="visual vs itself (the limit)")
ax_right.set_title("3D subspace vs 3D subspace")
ax_right.set_xlabel("smallest principal angle (degrees)")
ax_right.legend(fontsize=8.5, framealpha=1)

for a in (ax_left, ax_right):
    a.set_xlim(0, 90)
    a.set_xticks(np.arange(0, 91, 15))
    a.grid(True, axis="x", color="0.88", lw=0.6)
    a.set_axisbelow(True)

fig_angles.suptitle("NPBM — where the observed subspace angles sit between "
                    "label-shuffled and self-identical", x=0.01, ha="left", fontsize=11)
fig_angles.tight_layout()
fig_angles.savefig(f"{OUT_DIR}/subspace_angle_distributions.png", dpi=150,
                   facecolor="white")
plt.show()
