# %%

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from pathlib import Path
from utils import *

os.chdir(Path(__file__).parent.parent)
os.getcwd()
# %%

EXPERIMENTS = [ "NPBK", "NPBM", "NPBN", "NPBO"]

# Short modality description per experiment (from the dataset description).
MODALITY = {
    "NPBI": "visual patterns + visuo-tactile",
    "NPBK": "visual patterns",
    "NPBM": "tactile patterns and visual patterns",
    "NPBN": "tactile patterns",
    "NPBO": "tactile patterns",
}


# alphabetical would put F10 before F5, so order the patterns explicitly
STIM_ORDER = ["F5", "F10", "F20", "F∞", "vA", "vB", "vC", "vD"]

OUT_DIR = "figures/psth_sessions"
DATA_DIR = "processed/psth"
RECOMPUTE = True                   # True to re-read the nwb files from scratch

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)



# %% processing -- every experiment in one loop, everything kept in `data`




data = {}
for exp in EXPERIMENTS:
    cache = f"{DATA_DIR}/{exp}.npz"

            # NPBM is 17 GB and NPBK 6 GB, so the binning is done once and cached.
    if os.path.exists(cache) and not RECOMPUTE:
        # allow_pickle because the first caches were written with an
        # object-dtype label array; these are our own files
        stored = np.load(cache, allow_pickle=True)
        data[exp] = {k: stored[k] for k in stored.files}
        print(f"{exp}: loaded {cache}")
        continue
    else:
        units, trials = load_data(exp)


    units_good,onsets,offsets,modality,stimulus,partner=extract_data(units,trials)
    
    labels = []
    for i in range(len(trials)):
        if partner.iloc[i] == "":
            labels.append(f"{modality.iloc[i]}/{stimulus.iloc[i]}")
        else:
            labels.append(f"{modality.iloc[i]}/{stimulus.iloc[i]}+{partner.iloc[i]}")
    labels = np.array(labels)

    # Put the classes in the order we want to plot them: modality first, then
    # the patterns in the order STIM_ORDER lists them. Sorting the names
    # alphabetically instead would put F10 before F5.
    found = sorted(set(labels))
    classes = []
    for modality_name in sorted({c.split("/")[0] for c in found}):
        for pattern in STIM_ORDER:
            for cls in found:
                if (cls.split("/")[0] == modality_name
                        and cls.split("/")[1].startswith(pattern)):
                    classes.append(cls)

    # a pattern missing from STIM_ORDER would be dropped here without a word
    if len(classes) != len(found):
        raise ValueError(f"{exp}: {set(found) - set(classes)} not in STIM_ORDER")

    print(f"{exp} ({MODALITY[exp]}): {len(units_good)} good units, "
          f"{len(onsets)} trials, {len(classes)} classes")

    # binned spike train for every unit, aligned to every stimulus onset.
    # uint8 because a 2 ms bin never holds more than a couple of spikes, and the
    # int64 version of this array is 400 MB on the bigger sessions

    

    pre_s=.3
    post_s=.6
    bin_s=0.002





    bin_edges= np.arange(pre_s,post_s+bin_s,bin_s)
    n_bins= len(bin_edges)-1
    binned_per_neuron = get_psth(units_good,onsets,bin_edges,pre_s,post_s)
    

    # per class: the unit x time heatmap (mean over that class's trials), and the
    # global mean over units in Hz
    heatmaps = np.zeros((len(classes), len(units_good), n_bins))
    means = np.zeros((len(classes), n_bins))
    n_trials = np.zeros(len(classes), dtype=int)
    durations = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        class_mask = labels == cls
        heatmaps[i] = binned_per_neuron[:, class_mask, :].mean(axis=1)
        means[i] = heatmaps[i].mean(axis=0) / bin_s
        n_trials[i] = class_mask.sum()
        durations[i] = trials.loc[class_mask, "duration_s"].median()

    # kept per trial as well, so the trial-by-trial checks in npbi_check.py do
    # not have to re-read the nwb file
    pop_per_trial = binned_per_neuron.mean(axis=0) / bin_s

    data[exp] = {"classes": np.array(classes), "heatmaps": heatmaps,
                 "means": means, "n_trials": n_trials, "durations": durations,
                  "onsets": onsets, "labels": labels,
                 "pop_per_trial": pop_per_trial,
                 "cluster_ids": units_good["cluster_id"].to_numpy()}
    np.savez_compressed(cache, **data[exp])
    print(f"   wrote {cache}")


# %% plotting -- takes the processed data, opens no nwb file

def plot_session(exp, d):
    """One figure per experiment: the classes on a 2-row grid, each one a
    unit x time heatmap with the global mean underneath it.

    4 classes -> 2x2, 8 classes -> 2x4. Classes are sorted by modality first,
    so the top row is the unpaired conditions and the bottom row the paired
    ones, column by column on the same pattern.
    """
    classes, times = d["classes"], d["times"]
    n_rows = 2
    n_cols = int(np.ceil(len(classes) / n_rows))

    fig = plt.figure(figsize=(3.4 * n_cols + 0.9, 3.6 * n_rows))

    # one colour and y scale for the whole figure, otherwise the classes cannot
    # be compared by eye
    vmax = np.percentile(d["heatmaps"], 99.5)
    ymax = 1.15 * d["means"].max()

    # Placed by hand. Both layout engines leave a band of dead space between the
    # two class-rows here, because the colourbar spans axes from two different
    # gridspecs; tight_layout also cannot place a colourbar at all.
    block = 0.78 / n_rows                       # figure fraction per class-row
    for row in range(n_rows):
        top = 0.90 - row * (block + 0.06)
        # each class-row is a tall heatmap over a short mean trace
        inner = fig.add_gridspec(2, n_cols, height_ratios=[3, 1],
                                 top=top, bottom=top - block,
                                 left=0.06, right=0.90, hspace=0.06, wspace=0.12)
        for col in range(n_cols):
            i = row * n_cols + col
            if i >= len(classes):
                break
            ax = fig.add_subplot(inner[0, col])
            axm = fig.add_subplot(inner[1, col], sharex=ax)

            im = ax.imshow(d["heatmaps"][i], aspect="auto", cmap="magma",
                           vmin=0, vmax=vmax,
                           extent=[times[0], times[-1], len(d["heatmaps"][i]), 0])
            ax.set_title(f"{classes[i]}   n={d['n_trials'][i]}", fontsize=9.5)
            ax.tick_params(labelbottom=False)

            axm.plot(times, d["means"][i], color="black", lw=1)
            axm.set_ylim(0, ymax)

            for a in (ax, axm):
                a.axvline(0, color="green", lw=1.5)
                a.axvline(d["durations"][i], color="red", lw=1.5)

            if row == n_rows - 1:
                axm.set_xlabel("time from onset (s)")
            if col == 0:
                ax.set_ylabel("neurons")
                axm.set_ylabel("mean (Hz)")
            else:
                ax.set_yticks([])
                axm.tick_params(labelleft=False)

    fig.colorbar(im, cax=fig.add_axes([0.915, 0.90 - block, 0.012, block]),
                 label="mean spikes / bin")
    fig.suptitle(f"{exp} — {MODALITY[exp]}    "
                 f"({len(d['heatmaps'][0])} good units, {BIN_S * 1000:.0f} ms bins, "
                 f"green = onset, red = offset)", x=0.06, y=0.96, ha="left",
                 fontsize=11)

    path = f"{OUT_DIR}/psth_{exp}.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.show()
    return path


for exp in EXPERIMENTS:
    print(f"wrote {plot_session(exp, data[exp])}")

# %%
