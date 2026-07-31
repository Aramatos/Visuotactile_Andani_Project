# %%
"""PSTH for ONE session, laid out as a 2 x N grid of stimulus classes.
psth_all_sessions.py loops over all five sessions and hides the plotting inside a
function. This file does one session with everything written out straight, cell by
cell, so every intermediate can be looked at.

Every rate on the figure is in Hz: the spike count in a BIN_S-wide bin divided by
BIN_S. The heatmap is per-neuron Hz, the trace underneath it is the mean over
neurons, also Hz.

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
BIN_S = 0.010               # 10 ms bins
PRE_S = 0.200               # 200 ms of prestimulus baseline (the paper uses 300)
POST_S = 0.600              # patterns run 200-340 ms, so 300 would cut the offset
TICK_S = 0.100              # x ticks and gridlines every 100 ms


bin_edges = np.arange(-PRE_S, POST_S + BIN_S, BIN_S)
times = bin_edges[:-1] + BIN_S / 2

binned_per_neuron = np.zeros((len(units_good), len(onsets), len(times)), dtype=np.uint8)
for row, spikes in enumerate(units_good["spike_times"]):
    for i, onset in enumerate(onsets):
        spikes_aligned = spikes - onset
        in_window = (spikes_aligned >= -PRE_S) & (spikes_aligned <= POST_S)
        binned_per_neuron[row, i, :] = np.histogram(spikes_aligned[in_window],
                                                    bins=bin_edges)[0]

print(f"binned: {binned_per_neuron.shape} (unit, trial, bin), "
      f"{len(times)} bins of {BIN_S * 1000:.0f} ms")


# %% one neuron order for the whole figure
#
# Each unit gets its mean firing rate over the PRE-stimulus bins, pooled across
# every trial of every class, and the heatmap rows are then sorted fastest-first.
# Pre-stimulus only, so the sort is a property of the neuron and not of any
# response, and row 7 is the same neuron in all eight panels. Left unsorted the
# heatmap is salt-and-pepper and nothing can be read off it.

pre_bins = times < 0
baseline_hz = binned_per_neuron[:, :, pre_bins].mean(axis=(1, 2)) / BIN_S
order = np.argsort(baseline_hz)[::-1]

print(f"baseline rates: {baseline_hz.min():.2f} to {baseline_hz.max():.2f} Hz, "
      f"median {np.median(baseline_hz):.2f} Hz")


# %% average per class, in Hz

heatmaps = []
means = []
n_trials = []
offsets = []
baselines = []
for cls in classes:
    in_class = labels == cls
    counts_per_neuron = binned_per_neuron[:, in_class, :].mean(axis=1)
    rates_per_neuron = counts_per_neuron / BIN_S            # spikes/bin -> Hz
    population_rate = rates_per_neuron.mean(axis=0)         # mean over neurons, Hz
    heatmaps.append(rates_per_neuron[order])
    means.append(population_rate)
    n_trials.append(int(in_class.sum()))
    offsets.append(float(trials.loc[in_class, "duration_s"].median()))
    baselines.append(float(population_rate[pre_bins].mean()))

vmax = np.percentile(heatmaps, 99.5)        # one colour scale for the figure, Hz
ymax = 1.15 * np.max(means)                 # one y scale for the mean traces, Hz

print(f"colour scale 0 - {vmax:.1f} Hz, trace scale 0 - {ymax:.1f} Hz")


# %% plot: a 2 x N grid of classes, each one a heatmap over its population rate
#
# The grid is built as one gridspec per CLASS-ROW, and inside each of those a
# 2 x n_cols block: the block's top row holds the heatmaps and its bottom row the
# mean traces. So one cell of the visible 2 x 4 grid is really two stacked axes.
# A flat subplots(2, 4) cannot express that -- which is why the previous version
# drew classes 4-7 straight on top of classes 0-3.

n_rows = 2
n_cols = int(np.ceil(len(classes) / n_rows))

# The axes run edge-to-edge of the binned window, not centre-to-centre of the
# first and last bin, so the tick at 0.0 sits on the true onset and the tick at
# -0.2 / 0.6 sits on the true window edge instead of half a bin inside it.
x_lo = bin_edges[0]
x_hi = bin_edges[-1]
# arange on floats lands on things like 0.30000000000000004, which matplotlib
# would print in full, so the tick positions are rounded to the millisecond
xticks = np.arange(x_lo, x_hi + TICK_S / 2, TICK_S)
xticks = np.round(xticks, 3)

fig_w = 3.3 * n_cols + 1.9
fig_h = 3.7 * n_rows + 1.1
fig = plt.figure(figsize=(fig_w, fig_h))

# The margins are decided in INCHES and only then turned into the figure fractions
# matplotlib wants. A fixed fraction would be a different physical width on a
# 4-class session (2 columns, ~8 in wide) than on an 8-class one (4 columns, ~15 in
# wide), and the "pop. rate (Hz)" label ran off the left edge of the narrow ones.
left = 1.05 / fig_w                           # room for the y tick labels
right = 1 - 1.35 / fig_w                      # room for the colourbar
top_edge = 1 - 0.90 / fig_h                   # room for suptitle + panel titles
bottom_edge = 0.60 / fig_h                    # room for the x label
gap = 0.55 / fig_h                            # dead space between class-rows
span = top_edge - bottom_edge
block = (span - gap * (n_rows - 1)) / n_rows  # figure fraction per class-row

for row in range(n_rows):
    top = top_edge - row * (block + gap)
    # wspace has to clear the tick LABELS, not just the axes: at one label every
    # 100 ms the "0.6" of one column and the "-0.2" of the next sit on top of
    # each other at the 0.10 the panels alone would need
    inner = fig.add_gridspec(2, n_cols, height_ratios=[3, 1],
                             top=top, bottom=top - block,
                             left=left, right=right, hspace=0.07, wspace=0.17)
    for col in range(n_cols):
        i = row * n_cols + col
        if i >= len(classes):
            break
        ax = fig.add_subplot(inner[0, col])
        axm = fig.add_subplot(inner[1, col], sharex=ax)

        image = ax.imshow(heatmaps[i], aspect="auto", cmap="magma",
                          vmin=0, vmax=vmax,
                          extent=[x_lo, x_hi, len(units_good), 0],
                          interpolation="nearest")
        ax.set_title(f"{classes[i]}   n={n_trials[i]}", fontsize=10)
        ax.tick_params(labelbottom=False)

        axm.plot(times, means[i], color="black", lw=1.2)
        axm.set_ylim(0, ymax)
        # this class's own pre-stimulus level, so the evoked change can be read
        # off the trace in Hz instead of guessed
        axm.axhline(baselines[i], color="0.55", lw=0.8, ls="--")

        ax.axvline(0, color="#00c000", lw=1.5, zorder=3)
        ax.axvline(offsets[i], color="#e03030", lw=1.5, zorder=3)
        axm.axvline(0, color="#00c000", lw=1.5, zorder=3)
        axm.axvline(offsets[i], color="#e03030", lw=1.5, zorder=3)
        ax.set_xlim(x_lo, x_hi)
        axm.set_xlim(x_lo, x_hi)
        ax.set_xticks(xticks)
        axm.set_xticks(xticks)
        ax.tick_params(labelsize=8)
        axm.tick_params(labelsize=8)

        # The same 100 ms ruling on the heatmap and on the trace under it, so a
        # feature can be carried down from one to the other by eye. On the heatmap
        # it has to be drawn ON TOP of the image, which is what set_axisbelow(False)
        # does; matplotlib otherwise hides it underneath.
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
fig.colorbar(image, cax=cax, label="firing rate (Hz)")

fig.suptitle(f"{EXP}   ({len(units_good)} good units, {BIN_S * 1000:.0f} ms bins, "
             f"rates in Hz, green = onset, red = offset)",
             x=left, y=1 - 0.28 / fig_h, ha="left", fontsize=11)

fig.savefig(f"figures/psth_{EXP}_single.png", dpi=150, facecolor="white")
plt.show()

# %%
