# %%
import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA



# open an nwb file and grab the units and the stimulus trials
with NWBHDF5IO("nwb/NPBK.nwb", "r") as io:
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

# get the good units
units = units[units["quality"] == "good"]

# raster: one row of ticks per unit, lined up to the stimulus onset
fig, ax = plt.subplots(figsize=(10, 7))

for row, spikes in enumerate(units["spike_times"]):
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
