# %%
import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA


# Short modality description per experiment (from the dataset description).
MODALITY = {
    "NPBI": "visual patterns + visuo-tactile",
    "NPBK": "visual patterns",
    "NPBM": "tactile patterns and visual patterns",
    "NPBN": "tactile patterns",
    "NPBO": "tactile patterns",
}

#EXPERIMENTS = ["NPBI"]
EXPERIMENTS = ["NPBI", "NPBK", "NPBM", "NPBN", "NPBO"]
trial_arrays = []
for exp in EXPERIMENTS:
    print(f"Experiment {exp}: {MODALITY[exp]}")
    with NWBHDF5IO(f"nwb/{exp}.nwb", "r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        trials = nwb.trials.to_dataframe()
        trial_arrays.append(trials)


# %%
# print head of all 5 expeirment types
for exp, trials in zip(EXPERIMENTS, trial_arrays):
    print(f"Experiment {exp}: {MODALITY[exp]}")
    print(trials.head())

# %%
# Load the LFP. NPBK, NPBM, NPBN and NPBO have it; NPBI does not.
with NWBHDF5IO("nwb/NPBO.nwb", "r") as io:
    nwb = io.read()
    lfp = nwb.processing["ecephys"]["LFP"]["lfp_imec0"]

    print(lfp.data.shape, "= (samples, channels)")
    print(lfp.rate, "Hz, starts at", lfp.starting_time, "s")

    # data is int16 -> multiply by conversion for volts (here 1e6 for microvolts)
    seconds, channel = 2, 300
    n = int(seconds * lfp.rate)
    trace = lfp.data[:n, channel] * lfp.conversion * 1e6
    time = np.arange(n) / lfp.rate

plt.figure(figsize=(9, 3))
plt.plot(time, trace, lw=0.8)
plt.xlabel("time (s)")
plt.ylabel("µV")
plt.title(f"NPBO — LFP, channel {channel}")
plt.tight_layout()
plt.show()

# %%
# Average the LFP around the tactile stimuli (the evoked response).
with NWBHDF5IO("nwb/NPBO.nwb", "r") as io:
    nwb = io.read()
    lfp = nwb.processing["ecephys"]["LFP"]["lfp_imec0"]
    onsets = nwb.trials.to_dataframe()["start_time"].to_numpy()[:100]

    pre, post = int(0.1 * lfp.rate), int(0.4 * lfp.rate)
    chunks = []
    for t in onsets:
        i = int(round(t * lfp.rate))
        seg = lfp.data[i - pre : i + post, 240:] * lfp.conversion * 1e6
        chunks.append(seg - seg[:pre].mean(0))     # subtract the pre-stimulus level
    evoked = np.mean(chunks, axis=0).mean(axis=1)  # average trials, then channels

plt.figure(figsize=(6, 3))
plt.plot((np.arange(-pre, post)) / lfp.rate * 1000, evoked)
plt.axvline(0, color="0.6", lw=0.8)
plt.xlabel("time from stimulus (ms)")
plt.ylabel("µV")
plt.title("NPBO — tactile evoked LFP (100 trials, upper channels)")
plt.tight_layout()
plt.show()

# %%
