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

# The PSTH window and bin width. Everything downstream is expressed in Hz, which
# is just the spike count in a bin divided by BIN_S -- so BIN_S is the only place
# the bin width appears and the rates stay comparable if it is changed.
BIN_S = 0.010               # 10 ms bins
PRE_S = 0.200               # 200 ms of prestimulus baseline (the paper uses 300)
POST_S = 0.600              # patterns run 200-340 ms, so 300 would cut the offset
TICK_S = 0.100              # x ticks and gridlines every 100 ms

OUT_DIR = "figures/psth_sessions"
DATA_DIR = "processed/psth"
RECOMPUTE = False                  # True to re-read the nwb files from scratch

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)



# %% processing -- every experiment in one loop, everything kept in `data`




data = {}
for exp in EXPERIMENTS:
    # the whole window is part of the cache name, so changing PRE_S, POST_S or
    # BIN_S can never read back an array binned to the old one
    cache = (f"{DATA_DIR}/{exp}_pre{PRE_S * 1000:.0f}"
             f"_post{POST_S * 1000:.0f}_bin{BIN_S * 1000:.0f}ms.npz")

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
    # uint8 because a 10 ms bin never holds more than a handful of spikes, and the
    # int64 version of this array is 400 MB on the bigger sessions

    # The window runs from -PRE_S to +POST_S, so the first edge is NEGATIVE.
    # A previous version started the edges at +PRE_S, which silently binned only
    # the 300-600 ms tail while the plots still labelled the axis -0.3 to 0.6.
    bin_edges = np.arange(-PRE_S, POST_S + BIN_S, BIN_S)
    times = bin_edges[:-1] + BIN_S / 2
    n_bins = len(bin_edges) - 1
    binned_per_neuron = get_psth(units_good, onsets, bin_edges, PRE_S, POST_S)

    # One neuron order for the whole session, taken from how fast each unit fires
    # in the PRE-stimulus bins pooled over every trial. Pre-stimulus only, so the
    # ordering is not itself a response and the same rows mean the same neurons in
    # every panel; without it the heatmaps are salt-and-pepper and unreadable.
    pre_bins = times < 0
    baseline_hz = binned_per_neuron[:, :, pre_bins].mean(axis=(1, 2)) / BIN_S
    order = np.argsort(baseline_hz)[::-1]

    # per class: the unit x time heatmap in Hz (mean over that class's trials,
    # divided by the bin width), and the mean over units, also in Hz
    heatmaps = np.zeros((len(classes), len(units_good), n_bins))
    means = np.zeros((len(classes), n_bins))
    n_trials = np.zeros(len(classes), dtype=int)
    durations = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        class_mask = labels == cls
        counts_per_unit = binned_per_neuron[:, class_mask, :].mean(axis=1)
        rates_per_unit = counts_per_unit / BIN_S
        heatmaps[i] = rates_per_unit[order]
        means[i] = heatmaps[i].mean(axis=0)
        n_trials[i] = class_mask.sum()
        durations[i] = trials.loc[class_mask, "duration_s"].median()

    # kept per trial as well, so the trial-by-trial checks in npbi_check.py do
    # not have to re-read the nwb file
    pop_per_trial = binned_per_neuron.mean(axis=0) / BIN_S

    data[exp] = {"classes": np.array(classes), "heatmaps": heatmaps,
                 "means": means, "n_trials": n_trials, "durations": durations,
                 "times": times, "bin_edges": bin_edges,
                 "onsets": onsets, "labels": labels,
                 "pop_per_trial": pop_per_trial,
                 "baseline_hz": baseline_hz[order],
                 "cluster_ids": units_good["cluster_id"].to_numpy()[order]}
    np.savez_compressed(cache, **data[exp])
    print(f"   wrote {cache}")


# %% plotting -- takes the processed data, opens no nwb file

def plot_session(exp, d):
    """One figure per experiment: the classes on a 2-row grid, each one a
    unit x time heatmap in Hz with the population mean rate underneath it.

    4 classes -> 2x2, 8 classes -> 2x4. Classes are sorted by modality first,
    so the top row is the unpaired conditions and the bottom row the paired
    ones, column by column on the same pattern.

    Everything it draws is already in `d`; it computes no rates of its own.
    """
    classes = d["classes"]
    times = d["times"]
    bin_edges = d["bin_edges"]
    n_rows = 2
    n_cols = int(np.ceil(len(classes) / n_rows))

    # The axes run edge-to-edge of the binned window, not centre-to-centre of the
    # first and last bin, so the tick at 0.0 sits on the true onset and the tick
    # at -0.2 / 0.6 sits on the true window edge instead of half a bin inside it.
    x_lo = bin_edges[0]
    x_hi = bin_edges[-1]
    # arange on floats lands on things like 0.30000000000000004, which matplotlib
    # would print in full, so the tick positions are rounded to the millisecond
    xticks = np.arange(x_lo, x_hi + TICK_S / 2, TICK_S)
    xticks = np.round(xticks, 3)

    fig_w = 3.3 * n_cols + 1.9
    fig_h = 3.7 * n_rows + 1.1
    fig = plt.figure(figsize=(fig_w, fig_h))

    # one colour and y scale for the whole figure, otherwise the classes cannot
    # be compared by eye. Both are in Hz.
    vmax = np.percentile(d["heatmaps"], 99.5)
    ymax = 1.15 * d["means"].max()

    # Placed by hand. Both layout engines leave a band of dead space between the
    # two class-rows here, because the colourbar spans axes from two different
    # gridspecs; tight_layout also cannot place a colourbar at all.
    #
    # The margins are decided in INCHES and only then turned into the figure
    # fractions matplotlib wants. A fixed fraction would be a different physical
    # width on a 4-class figure (2 columns, ~8 in wide) than on an 8-class one
    # (4 columns, ~15 in wide), and the "pop. rate (Hz)" label ran off the left
    # edge of the narrow ones.
    left = 1.05 / fig_w                          # room for the y tick labels
    right = 1 - 1.35 / fig_w                     # room for the colourbar
    top_edge = 1 - 0.90 / fig_h                  # room for suptitle + panel titles
    bottom_edge = 0.60 / fig_h                   # room for the x label
    gap = 0.55 / fig_h                           # dead space between class-rows
    span = top_edge - bottom_edge
    block = (span - gap * (n_rows - 1)) / n_rows  # figure fraction per class-row

    for row in range(n_rows):
        top = top_edge - row * (block + gap)
        # each class-row is a tall heatmap over a short mean-rate trace
        # wspace has to clear the tick LABELS, not just the axes: at one label
        # every 100 ms the "0.6" of one column and the "-0.2" of the next sit on
        # top of each other at the 0.10 the panels alone would need
        inner = fig.add_gridspec(2, n_cols, height_ratios=[3, 1],
                                 top=top, bottom=top - block,
                                 left=left, right=right, hspace=0.07, wspace=0.17)
        for col in range(n_cols):
            i = row * n_cols + col
            if i >= len(classes):
                break
            ax = fig.add_subplot(inner[0, col])
            axm = fig.add_subplot(inner[1, col], sharex=ax)

            n_units = len(d["heatmaps"][i])
            im = ax.imshow(d["heatmaps"][i], aspect="auto", cmap="magma",
                           vmin=0, vmax=vmax,
                           extent=[x_lo, x_hi, n_units, 0],
                           interpolation="nearest")
            ax.set_title(f"{classes[i]}   n={d['n_trials'][i]}", fontsize=10)
            ax.tick_params(labelbottom=False)

            axm.plot(times, d["means"][i], color="black", lw=1.2)
            axm.set_ylim(0, ymax)

            # the pre-stimulus level of THIS class, so the evoked change can be
            # read off the trace in Hz instead of guessed
            pre_bins = times < 0
            baseline = d["means"][i][pre_bins].mean()
            axm.axhline(baseline, color="0.55", lw=0.8, ls="--")

            for a in (ax, axm):
                a.axvline(0, color="#00c000", lw=1.5, zorder=3)
                a.axvline(d["durations"][i], color="#e03030", lw=1.5, zorder=3)
                a.set_xlim(x_lo, x_hi)
                a.set_xticks(xticks)
                a.tick_params(labelsize=8)

            # The same 100 ms ruling on the heatmap and on the trace under it, so
            # a feature can be carried down from one to the other by eye. On the
            # heatmap it has to be drawn ON TOP of the image, which is what
            # set_axisbelow(False) does; matplotlib otherwise hides it underneath.
            ax.set_axisbelow(False)
            ax.grid(True, axis="x", color="white", alpha=0.30, lw=0.6)
            axm.grid(True, axis="x", color="0.80", lw=0.6)

            if row == n_rows - 1:
                axm.set_xlabel("time from onset (s)")
            if col == 0:
                ax.set_ylabel("neuron (sorted by baseline Hz)", fontsize=9)
                axm.set_ylabel("pop. rate\n(Hz)", fontsize=9)
            else:
                ax.tick_params(labelleft=False)
                axm.tick_params(labelleft=False)

    cax = fig.add_axes([right + 0.25 / fig_w, bottom_edge, 0.16 / fig_w, span])
    fig.colorbar(im, cax=cax, label="firing rate (Hz)")

    fig.suptitle(f"{exp} — {MODALITY[exp]}    "
                 f"({len(d['heatmaps'][0])} good units, {BIN_S * 1000:.0f} ms bins, "
                 f"rates in Hz, green = onset, red = offset)",
                 x=left, y=1 - 0.28 / fig_h, ha="left", fontsize=11)

    path = f"{OUT_DIR}/psth_{exp}.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.show()
    return path


for exp in EXPERIMENTS:
    print(f"wrote {plot_session(exp, data[exp])}")

# %%
