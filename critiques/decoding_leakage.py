# %% [ THE ARGUMENT ]
"""Critique B: the decoding accuracy is inflated by cross-validation leakage.

  THE CLAIM      Evoked activity is separable from spontaneous activity, F1 = 0.93
                 for tactile and 0.81 for visual, and F1 keeps climbing out to
                 PC #50 with no saturation.

  THE WEAKNESS   Their data points are individual 1 ms samples, and step 3 of
                 their pipeline convolves everything with a 10 ms Gaussian. That
                 makes neighbouring samples near-copies of each other. The
                 train/test split is then random ACROSS SAMPLES, so t = 147 ms can
                 land in training while t = 148 ms of the SAME trial lands in
                 test. kNN does not have to learn anything about spontaneous
                 versus evoked: it can find the test point's own smoothed
                 neighbour, carrying the same label, and copy it.

  THE FIX        Split by TRIAL. Every sample from a trial goes entirely to train
                 or entirely to test, so no test point has a near-copy of itself
                 on the other side of the split.

  THE TEST       Run the identical classifier both ways and see how far F1 falls.
                 A real effect survives trial-wise splitting; leakage does not.
                 There IS a real tactile response in this data, so the expectation
                 is a drop, not a collapse to chance. The size of the drop is the
                 measurement.

  WHY NPBI IS    Its onsets are about 243 ms late, so its "spontaneous" and
  NOT HERE       "evoked" windows are both mislabelled in time.

Steps 1-7 walk through NPBN. Then NPBK, NPBM and NPBO each get their own cell.
"""

# %% [ STEP 0 ] settings, and the pieces we reuse

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from pynwb import NWBHDF5IO

WINDOW_S = 0.300            # 300 ms before onset = spontaneous, after = evoked
BIN_S = 0.001               # paper: 1 ms bins for the decoding
SMOOTH_MS = 10              # paper: 10 ms Gaussian kernel
GROUP = 50                  # paper: average 50 sequential trials before decoding
K = 5                       # paper: k = 5 nearest neighbours
VARIANCE = 0.95             # paper: PCs explaining 95% of spontaneous variance
N_REPEATS = 5               # paper used 50; 5 is enough to separate the two schemes

# Subsampled on purpose: kNN is O(train x test), and all 800 trials x 300 ms x 2
# classes is 480k points, which does not finish. 40 trials per class keeps the 1 ms
# resolution -- and 1 ms resolution is the whole point, because the leakage lives
# between neighbouring milliseconds. Stated here rather than hidden.
TRIALS_PER_CLASS = 40

rng = np.random.default_rng(0)


def load_windows(exp):
    """Spike counts either side of each onset, smoothed as the paper smooths them.

    Returns (trial, neuron, ms) for spontaneous and for evoked, same trial order.
    """
    with NWBHDF5IO(f"nwb/{exp}.nwb", "r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        trials = nwb.trials.to_dataframe()

    good = units[units["quality"] == "good"]
    onsets = trials["start_time"].to_numpy()
    n_ms = int(WINDOW_S / BIN_S)

    spontaneous = np.zeros((len(onsets), len(good), n_ms), dtype=np.float32)
    evoked = np.zeros((len(onsets), len(good), n_ms), dtype=np.float32)
    pre_edges = np.arange(n_ms + 1) * BIN_S - WINDOW_S
    post_edges = np.arange(n_ms + 1) * BIN_S

    for neuron, spikes in enumerate(good["spike_times"]):
        spikes = np.sort(np.asarray(spikes, dtype=np.float64))
        for trial, onset in enumerate(onsets):
            lo, hi = np.searchsorted(spikes, [onset - WINDOW_S, onset + WINDOW_S])
            relative = spikes[lo:hi] - onset
            spontaneous[trial, neuron] = np.histogram(relative, bins=pre_edges)[0]
            evoked[trial, neuron] = np.histogram(relative, bins=post_edges)[0]

    # the paper's step 3, applied along time within each trial
    sigma = (SMOOTH_MS / (BIN_S * 1000)) / 2.355
    return (gaussian_filter1d(spontaneous, sigma, axis=2),
            gaussian_filter1d(evoked, sigma, axis=2))


def make_points(spontaneous, evoked, trials_per_class, average_group=None):
    """Turn the windows into (sample, neuron) points, with labels and group ids.

    label 0 = spontaneous, 1 = evoked. The group id is what a trial-wise split
    holds out: a single trial normally, or one averaged trace when average_group
    is set.

    average_group reproduces the paper's step 2 -- average that many SEQUENTIAL
    trials together. Sequential matters: it welds each averaged trace to one chunk
    of session time, which is why critique D can reuse this.
    """
    chosen = rng.choice(spontaneous.shape[0], trials_per_class, replace=False)
    chosen = np.sort(chosen)               # keep session order for the averaging

    all_x = []
    all_y = []
    all_groups = []
    for label, source in ((0, spontaneous), (1, evoked)):
        block = source[chosen]                              # (trial, neuron, ms)

        if average_group:
            n_groups = block.shape[0] // average_group
            keep = block[:n_groups * average_group]
            grouped = keep.reshape(n_groups, average_group, keep.shape[1],
                                   keep.shape[2])
            block = grouped.mean(axis=1)
            group_ids = np.arange(n_groups)
        else:
            group_ids = chosen

        n_items, n_neurons, n_ms = block.shape
        time_last = block.transpose(0, 2, 1)

        all_x.append(time_last.reshape(-1, n_neurons))
        all_y.append(np.full(n_items * n_ms, label))
        all_groups.append(np.repeat(group_ids, n_ms))

    return np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_groups)


def decode(x, y, trial_id, split_by_trial, shuffle_labels=False):
    """kNN with a 50/50 split, either over samples or over whole trials."""
    scores = []
    for repeat in range(N_REPEATS):
        labels = rng.permutation(y) if shuffle_labels else y
        if split_by_trial:
            trials = np.unique(trial_id)
            train_trials = rng.choice(trials, len(trials) // 2, replace=False)
            is_train = np.isin(trial_id, train_trials)
        else:
            is_train = rng.random(len(y)) < 0.5

        # PCA on the spontaneous training points only, as the paper fits it on
        # spontaneous data -- and fit on TRAIN only, or the split leaks again
        pca = PCA(n_components=VARIANCE).fit(x[is_train & (labels == 0)])
        model = KNeighborsClassifier(n_neighbors=K).fit(
            pca.transform(x[is_train]), labels[is_train])
        predicted = model.predict(pca.transform(x[~is_train]))
        scores.append(f1_score(labels[~is_train], predicted))
    return float(np.mean(scores)), float(np.std(scores)), pca.n_components_


# %% [ STEP 1 ] NPBN: load the two windows

npbn_spont, npbn_evoked = load_windows("NPBN")
print(f"NPBN: {npbn_spont.shape[1]} neurons, {npbn_spont.shape[0]} trials")
print(f"spontaneous window mean rate {npbn_spont.mean() / BIN_S:.2f} Hz")
print(f"evoked window mean rate      {npbn_evoked.mean() / BIN_S:.2f} Hz")
print("\nThe evoked window really is different -- there is a genuine response to")
print("decode. The question is only whether F1 = 0.93 measures it honestly.")


# %% [ STEP 2 ] NPBN: see the leakage, before doing any decoding
#
# If neighbouring 1 ms samples are near-copies, then a random split over samples
# hands the classifier the answer. Measure how near-copy they are.

x, y, trial_id = make_points(npbn_spont, npbn_evoked, TRIALS_PER_CLASS)
print(f"{len(x)} data points, {x.shape[1]} neurons, "
      f"{TRIALS_PER_CLASS} trials per class")

one_trial = npbn_evoked[0].T                       # (ms, neuron)
step1 = np.corrcoef(one_trial[:-1].ravel(), one_trial[1:].ravel())[0, 1]
step10 = np.corrcoef(one_trial[:-10].ravel(), one_trial[10:].ravel())[0, 1]
step50 = np.corrcoef(one_trial[:-50].ravel(), one_trial[50:].ravel())[0, 1]
print(f"\ncorrelation between a sample and the sample 1 ms later : {step1:.4f}")
print(f"...10 ms later                                        : {step10:.4f}")
print(f"...50 ms later                                        : {step50:.4f}")
print("\nAt 1 ms apart the two vectors are essentially the same point. A random")
print("split over samples therefore puts near-duplicates on both sides.")


# %% [ STEP 3 ] NPBN: their split -- random over samples

sample_f1, sample_sd, n_pcs = decode(x, y, trial_id, split_by_trial=False)
print(f"sample-wise split : F1 = {sample_f1:.3f} (SD {sample_sd:.3f}), "
      f"{n_pcs} PCs kept")
print(f"paper reported F1 = 0.93 for tactile with this scheme")


# %% [ STEP 4 ] NPBN: the honest split -- whole trials held out

trial_f1, trial_sd, n_pcs = decode(x, y, trial_id, split_by_trial=True)
print(f"trial-wise split  : F1 = {trial_f1:.3f} (SD {trial_sd:.3f}), "
      f"{n_pcs} PCs kept")
print(f"\ndrop from closing the leak: {sample_f1 - trial_f1:+.3f}")
print("Whatever survives here is the real, decodable difference between")
print("spontaneous and evoked population activity.")


# %% [ STEP 4b ] NPBN: their ACTUAL pipeline -- 50 sequential trials averaged first
#
# Steps 3 and 4 used single trials, and landed near 0.6 rather than their 0.93.
# The missing step is their trial averaging: 50 sequential trials collapsed into
# one trace, which raises the signal-to-noise enormously. Run it their way, or the
# comparison is against a pipeline they never used.
#
# Every trial is used here, not the subsample, because averaging 50 at a time
# leaves only 800/50 = 16 traces per class and kNN on that is instant.

x_averaged, y_averaged, group_averaged = make_points(
    npbn_spont, npbn_evoked, npbn_spont.shape[0], average_group=GROUP)

n_traces = len(np.unique(group_averaged))
print(f"{len(x_averaged)} data points from {n_traces} averaged traces per class, "
      f"{GROUP} sequential trials in each")

averaged_sample_f1, averaged_sample_sd, n_pcs = decode(
    x_averaged, y_averaged, group_averaged, split_by_trial=False)
print(f"averaged, sample-wise split : F1 = {averaged_sample_f1:.3f} "
      f"(SD {averaged_sample_sd:.3f}), {n_pcs} PCs kept")
print("paper reported F1 = 0.93 for tactile")


# %% [ STEP 4c ] NPBN: the same averaged data, holding out whole traces
#
# With only 16 traces per class this split is coarse, but it is the honest one:
# no test sample has a near-copy of itself in the training set.

averaged_trial_f1, averaged_trial_sd, n_pcs = decode(
    x_averaged, y_averaged, group_averaged, split_by_trial=True)
print(f"averaged, trace-wise split  : F1 = {averaged_trial_f1:.3f} "
      f"(SD {averaged_trial_sd:.3f}), {n_pcs} PCs kept")
print(f"\ndrop from closing the leak, on their own pipeline: "
      f"{averaged_sample_f1 - averaged_trial_f1:+.3f}")


# %% [ STEP 5 ] NPBN: both schemes against shuffled labels
#
# The control tells us where chance actually sits. If the sample-wise scheme beats
# chance even on SHUFFLED labels, the leak is proven outright, because shuffled
# labels contain no real information at all.

shuffled_sample_f1, shuffled_sample_sd, n_pcs = decode(
    x, y, trial_id, split_by_trial=False, shuffle_labels=True)
shuffled_trial_f1, shuffled_trial_sd, n_pcs = decode(
    x, y, trial_id, split_by_trial=True, shuffle_labels=True)
print(f"sample-wise, labels shuffled : F1 = {shuffled_sample_f1:.3f}")
print(f"trial-wise,  labels shuffled : F1 = {shuffled_trial_f1:.3f}")
print("\nBoth should sit near chance. If the sample-wise one does not, the split")
print("itself is leaking and no result computed that way can be trusted.")


# %% [ STEP 6 ] NPBN: does F1 really keep climbing with more PCs?
#
# The paper reads "F1 still rising at PC #50" as evidence of high-dimensional
# information. But if the gain comes from leakage, more PCs simply sharpen the
# near-duplicate matching. Sweep the PC count under both splits.

pc_counts = [1, 3, 5, 10, 20, 40]
npbn_pc_sweep = []
for n_pc in pc_counts:
    for by_trial in (False, True):
        pca = PCA(n_components=n_pc).fit(x[y == 0])
        projected = pca.transform(x)
        if by_trial:
            trials = np.unique(trial_id)
            train_trials = rng.choice(trials, len(trials) // 2, replace=False)
            is_train = np.isin(trial_id, train_trials)
        else:
            is_train = rng.random(len(y)) < 0.5
        model = KNeighborsClassifier(n_neighbors=K).fit(projected[is_train],
                                                        y[is_train])
        npbn_pc_sweep.append({
            "n_pcs": n_pc,
            "split": "trial-wise" if by_trial else "sample-wise",
            "f1": f1_score(y[~is_train], model.predict(projected[~is_train]))})
npbn_pc_sweep = pd.DataFrame(npbn_pc_sweep)
print(npbn_pc_sweep.pivot(index="n_pcs", columns="split", values="f1").round(3).to_string())


# %% [ STEP 7 ] NPBN: the figure


def plot_leakage(pc_sweep, sample, trial, shuffled, name, path):
    """Left: F1 under the two splits. Right: F1 against the number of PCs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    bars = ax.bar(["their split\n(over samples)", "honest split\n(over trials)"],
                  [sample, trial], color=["#c0392b", "#2c3e50"], width=0.55)
    ax.axhline(shuffled, color="#7b8794", lw=1, ls="--")
    ax.text(1.45, shuffled, " shuffled labels", va="center", fontsize=8,
            color="#7b8794")
    for bar, value in zip(bars, [sample, trial]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1, spontaneous vs evoked")
    ax.set_title(f"{name}: what the leak was worth", loc="left", fontsize=10)

    ax = axes[1]
    for split, colour in (("sample-wise", "#c0392b"), ("trial-wise", "#2c3e50")):
        sel = pc_sweep[pc_sweep.split == split]
        ax.plot(sel.n_pcs, sel.f1, "o-", color=colour, lw=1.5, ms=4, label=split)
    ax.set_xlabel("number of PCs")
    ax.set_ylabel("F1")
    ax.set_title("does F1 keep climbing with more PCs?", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(f"Critique B on {name} — random splits over 1 ms samples leak "
                 f"across a 10 ms smoothing kernel", x=0.02, ha="left",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150, facecolor="white")
    return path


os.makedirs("figures/critique", exist_ok=True)
figure_path = plot_leakage(npbn_pc_sweep, sample_f1, trial_f1,
                           shuffled_trial_f1, "NPBN",
                           "figures/critique/decoding_leakage_NPBN.png")
print(f"wrote {figure_path}")


# %% [ THE SAME TEST, ONE CELL PER SESSION ]


def one_session_story(exp):
    """Steps 3, 4 and 5 for one session, as a one-row summary."""
    spont, evoked = load_windows(exp)
    x, y, trial_id = make_points(spont, evoked, TRIALS_PER_CLASS)
    by_sample = decode(x, y, trial_id, False)
    by_trial = decode(x, y, trial_id, True)
    shuffled_sample = decode(x, y, trial_id, False, shuffle_labels=True)
    return pd.Series({
        "session": exp,
        "neurons": spont.shape[1],
        "f1_sample_split": round(by_sample[0], 3),
        "f1_trial_split": round(by_trial[0], 3),
        "drop": round(by_sample[0] - by_trial[0], 3),
        "f1_sample_split_shuffled": round(shuffled_sample[0], 3),
        "pcs_kept": by_trial[2],
    })


# %% NPBN's row (walked through above)

story_npbn = one_session_story("NPBN")
print(story_npbn.to_string())

# %% NPBK -- visual patterns only

story_npbk = one_session_story("NPBK")
print(story_npbk.to_string())

# %% NPBM -- tactile and visual patterns

story_npbm = one_session_story("NPBM")
print(story_npbm.to_string())

# %% NPBO -- tactile patterns

story_npbo = one_session_story("NPBO")
print(story_npbo.to_string())

# %% all four side by side

everyone = pd.DataFrame([story_npbn, story_npbk, story_npbm, story_npbo])
print(everyone.to_string(index=False))
print("\nThe 'drop' column is the size of the leak. What is left in")
print("f1_trial_split is the result the paper should have reported.")

# %%
