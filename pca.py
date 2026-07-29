# %%


import os
import json
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # hundreds of figures get written, none shown
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pynwb import NWBHDF5IO

EXPERIMENTS = ["NPBI", "NPBK", "NPBM", "NPBN", "NPBO"]


CHOSEN_EXPERIMENTS = ["NPBN"]  # for development, to avoid writing hundreds of figures
# Short modality description per experiment (from the dataset description).
MODALITY = {
    "NPBI": "visual patterns + visuo-tactile",
    "NPBK": "visual patterns",
    "NPBM": "tactile patterns and visual patterns",
    "NPBN": "tactile patterns",
    "NPBO": "tactile patterns",
}

BIN_S = 0.002                       # paper: 2 ms bins
PRE_S = 0.300                       # paper: 300 ms prestimulus baseline
DURAION = 0.300                       # paper: 300 ms poststimulus test window
POST_S = 0.300                      # paper: 300 ms poststimulus test window
N_SD = 3.0                          # paper used 2; stricter here
MIN_BINS = 10                       # flag when MORE than this many bins exceed


OUT_DIR = "figures/pca"

STIM_ORDER = ["F5", "F10", "F20", "F∞"]


# open an nwb file and grab the units and the stimulus trials
with NWBHDF5IO("nwb/NPBN.nwb", "r") as io:
    nwb = io.read()
    print(nwb.units)
    units = nwb.units.to_dataframe()
    trials = nwb.trials.to_dataframe()


# take the very first stimulus and find when it starts and ends
first = trials.iloc[0]
onset = first["start_time"]
offset = onset + first["duration_s"]

pre = 0.3
post = 0.3

# %%
# get the good units
units_good = units[units["quality"] == "good"]

# raster: one row of ticks per unit, lined up to the stimulus onset
fig, ax = plt.subplots(figsize=(10, 7))

for row, spikes in enumerate(units_good["spike_times"]):
    spikes = spikes - onset
    spikes = spikes[(spikes >= -pre) & (spikes <= post)]
    ax.plot(spikes, np.full_like(spikes, row), "|", color="black", markersize=3)

ax.axvline(0, color="green", lw=2, label="onset")
ax.axvline(offset - onset, color="red", lw=2, label="offset")

ax.set_xlim(-pre, post)
ax.set_xlabel("time from onset (s)")
ax.set_ylabel("unit")
ax.set_title(f"first stimulus  ({first['modality']} {first['stimulus']})")
ax.legend()

plt.show()

# %%
