# %%
"""PSTH for ONE session, with the simplest plotting that does the job.

psth_all_sessions.py loops over all five and places its panels by hand so the
2x2 / 2x4 grids come out right. This file drops all of that: pick a session at the
top, run the cells in order, get one figure with a column per stimulus class.

Set EXP and go. NPBI is a poor choice here -- its visuo-tactile onsets are about
243 ms late, so its panels are aligned to the wrong time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from pathlib import Path
from utils import *
import os
os.chdir(Path(__file__).parent.parent)
os.getcwd()

EXP = "NPBM"


# alphabetical would put F10 before F5, so name the order explicitly
STIM_ORDER = ["F5", "F10", "F20", "F∞", "vA", "vB", "vC", "vD"]



# %% load the session

with NWBHDF5IO(f"nwb/{EXP}.nwb", "r") as io:
    nwb = io.read()
    units = nwb.units.to_dataframe()
    trials = nwb.trials.to_dataframe()

units_good = units[units["quality"] == "good"]
onsets = trials["start_time"].to_numpy()

print(f"{EXP}: {len(units_good)} good units, {len(onsets)} trials")


# %% name each trial by what was presented on it

modality = trials["modality"].astype(str)
stimulus = trials["stimulus"].astype(str)
partner = trials["tactile_with"].fillna("").astype(str)

labels = []
for i in range(len(trials)):
    if partner.iloc[i] == "":
        labels.append(f"{modality.iloc[i]}/{stimulus.iloc[i]}")
    else:
        labels.append(f"{modality.iloc[i]}/{stimulus.iloc[i]}+{partner.iloc[i]}")
labels = np.array(labels)

# put the classes in plotting order: modality first, then STIM_ORDER
found = sorted(set(labels))
classes = []
for modality_name in sorted({c.split("/")[0] for c in found}):
    for pattern in STIM_ORDER:
        for cls in found:
            if cls.split("/")[0] == modality_name and cls.split("/")[1].startswith(pattern):
                classes.append(cls)

print(f"{len(classes)} classes: {classes}")


# %% bin every unit around every onset
BIN_S = 0.010              # paper: 2 ms bins
PRE_S = 0.300               # paper: 300 ms prestimulus baseline
POST_S = 0.600              # patterns run 200-340 ms, so 300 would cut the offset


bin_edges = np.arange(-PRE_S, POST_S + BIN_S, BIN_S)
times = bin_edges[:-1] + BIN_S / 2

binned_per_neuron = np.zeros((len(units_good), len(onsets), len(times)), dtype=np.uint8)
for row, spikes in enumerate(units_good["spike_times"]):
    for i, onset in enumerate(onsets):
        spikes_aligned = spikes - onset
        in_window = (spikes_aligned >= -PRE_S) & (spikes_aligned <= POST_S)
        binned_per_neuron[row, i, :] = np.histogram(spikes_aligned[in_window],
                                                    bins=bin_edges)[0]

print(f"binned: {binned_per_neuron.shape} (unit, trial, bin)")


# %% average per class

heatmaps = []
means = []
n_trials = []
offsets = []
for cls in classes:
    in_class = labels == cls
    per_neuron = binned_per_neuron[:, in_class, :].mean(axis=1)
    heatmaps.append(per_neuron)
    means.append(per_neuron.mean(axis=0) / BIN_S)              # in Hz
    n_trials.append(int(in_class.sum()))
    offsets.append(float(trials.loc[in_class, "duration_s"].median()))

vmax = np.percentile(heatmaps, 99.5)        # one colour scale for the figure
ymax = 1.15 * np.max(means)                 # one y scale for the mean traces


# %% plot: one column per class, heatmap over the population mean

fig, axes = plt.subplots(2, 4, figsize=(3.6 * len(classes), 6),
                         height_ratios=[3, 1], sharex=True, squeeze=False)

for i, cls in enumerate(classes):
    k = i // 4
    j = i % 4
    top = axes[0, j]
    bottom = axes[1, j]

    image = top.imshow(heatmaps[i], aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                       extent=[times[0], times[-1], len(units_good), 0])
    top.set_title(f"{cls}   n={n_trials[i]}", fontsize=10)

    bottom.plot(times, means[i], color="black", lw=1)
    bottom.set_ylim(0, ymax)
    bottom.set_xlabel("time from onset (s)")

    top.axvline(0, color="green", lw=1.5)
    top.axvline(offsets[i], color="red", lw=1.5)
    bottom.axvline(0, color="green", lw=1.5)
    bottom.axvline(offsets[i], color="red", lw=1.5)

axes[0, 0].set_ylabel("neurons")
axes[1, 0].set_ylabel("mean rate (Hz)")
fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.01, fraction=0.02,
             label="mean spikes / bin")
fig.suptitle(f"{EXP}   ({len(units_good)} good units, {BIN_S * 1000:.0f} ms bins, "
             f"green = onset, red = offset)", x=0.02, ha="left")

fig.savefig(f"figures/psth_{EXP}_single.png", dpi=150, facecolor="white",
            bbox_inches="tight")
plt.show()

# %%
