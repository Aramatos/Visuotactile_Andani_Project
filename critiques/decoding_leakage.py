# %% [ THE ARGUMENT ]
"""Critique B: the decoding accuracy is inflated by cross-validation leakage.

  THE CLAIM      Evoked activity is separable from spontaneous activity, F1 = 0.93
                 for tactile and 0.81 for visual, and F1 keeps climbing out to
                 PC #50 with no saturation, which they read as evidence that the
                 information is high-dimensional.

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

  READ THIS      STEPS 3-4 use single trials and land near 0.6, not their 0.93.
  ORDER          The missing ingredient is their 50-sequential-trial averaging,
                 which raises signal-to-noise enormously. STEPS 5-6 run their
                 ACTUAL pipeline, and that pair is the one to quote -- comparing
                 against a pipeline they never used would be unfair.

  WHY NPBI IS    Its onsets are about 243 ms late, so its "spontaneous" and
  NOT HERE       "evoked" windows are both mislabelled in time. See CLAUDE.md.

Steps 1-9 walk through NPBN. Then NPBK, NPBM and NPBO each get their own cell.
"""

# %% [ STEP 0 ] settings

import os
from pathlib import Path

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
N_REPEATS = 5               # paper used 50; 5 separates the two schemes cleanly

# Subsampled on purpose: kNN is O(train x test), and all 800 trials x 300 ms x 2
# classes is 480k points, which does not finish. 40 trials per class keeps the
# 1 ms resolution -- and 1 ms resolution is the whole point, because the leakage
# lives between neighbouring milliseconds. Stated here rather than hidden.
TRIALS_PER_CLASS = 40

# Every path below hangs off the repo root, so this file runs whether the working
# directory is the repo, the critiques folder, or wherever the VS Code
# interactive window happens to start. That window has no __file__, hence the
# two cases.
if "__file__" in globals():
    ROOT = Path(__file__).resolve().parent.parent
else:
    ROOT = Path.cwd()
if ROOT.name == "critiques":
    ROOT = ROOT.parent

OUT_DIR = ROOT / "figures/critique"
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(0)


# %% [ STEP 0b ] the three operations
#
# Each does one thing and calls none of the others.


def load_windows(exp):
    """Spike counts either side of each onset, smoothed as the paper smooths them.

    Returns (spontaneous, evoked), each (trial, neuron, ms), same trial order.
    """
    with NWBHDF5IO(ROOT / "nwb" / f"{exp}.nwb", "r") as io:
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
    smoothed_spontaneous = gaussian_filter1d(spontaneous, sigma, axis=2)
    smoothed_evoked = gaussian_filter1d(evoked, sigma, axis=2)
    return smoothed_spontaneous, smoothed_evoked


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
    """kNN with a 50/50 split, either over samples or over whole trials.

    Returns (mean F1, SD F1, PCs kept). With shuffle_labels the labels are
    permuted before splitting, which is the chance control for THAT split scheme
    -- the two schemes do not share a chance level, so each is measured with its
    own control.
    """
    scores = []
    for repeat in range(N_REPEATS):
        labels = rng.permutation(y) if shuffle_labels else y
        if split_by_trial:
            trials = np.unique(trial_id)
            train_trials = rng.choice(trials, len(trials) // 2, replace=False)
            is_train = np.isin(trial_id, train_trials)
        else:
            is_train = rng.random(len(y)) < 0.5

        # PCA on the spontaneous TRAINING points only. The paper fits it on
        # spontaneous data; fitting on everything would leak the split again.
        pca = PCA(n_components=VARIANCE).fit(x[is_train & (labels == 0)])
        model = KNeighborsClassifier(n_neighbors=K).fit(
            pca.transform(x[is_train]), labels[is_train])
        predicted = model.predict(pca.transform(x[~is_train]))
        scores.append(f1_score(labels[~is_train], predicted))
    return float(np.mean(scores)), float(np.std(scores)), pca.n_components_


# %% [ STEP 1 ] NPBN: load the two windows

npbn_spont, npbn_evoked = load_windows("NPBN")

spont_rate = npbn_spont.mean() / BIN_S
evoked_rate = npbn_evoked.mean() / BIN_S

print(f"NPBN: {npbn_spont.shape[1]} neurons, {npbn_spont.shape[0]} trials")
print(f"spontaneous window mean rate {spont_rate:.2f} Hz")
print(f"evoked window mean rate      {evoked_rate:.2f} Hz")
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


# %% [ STEP 3 ] NPBN: single trials, their split -- random over samples

single_sample_f1, single_sample_sd, single_sample_pcs = decode(
    x, y, trial_id, split_by_trial=False)
print(f"single trials, sample-wise split : F1 = {single_sample_f1:.3f} "
      f"(SD {single_sample_sd:.3f}), {single_sample_pcs} PCs kept")


# %% [ STEP 4 ] NPBN: single trials, the honest split -- whole trials held out

single_trial_f1, single_trial_sd, single_trial_pcs = decode(
    x, y, trial_id, split_by_trial=True)
print(f"single trials, trial-wise split  : F1 = {single_trial_f1:.3f} "
      f"(SD {single_trial_sd:.3f}), {single_trial_pcs} PCs kept")
print(f"\ndrop from closing the leak: {single_sample_f1 - single_trial_f1:+.3f}")


# %% [ STEP 5 ] NPBN: their ACTUAL pipeline -- 50 sequential trials averaged first
#
# This is the comparison that matters. Their 0.93 was measured after averaging 50
# sequential trials into one trace, which raises signal-to-noise enormously.
# Every trial is used here rather than the subsample, because averaging 50 at a
# time leaves only 800/50 = 16 traces per class and kNN on that is instant.

x_averaged, y_averaged, group_averaged = make_points(
    npbn_spont, npbn_evoked, npbn_spont.shape[0], average_group=GROUP)

n_traces = len(np.unique(group_averaged))
print(f"{len(x_averaged)} data points from {n_traces} averaged traces per class, "
      f"{GROUP} sequential trials in each")

averaged_sample_f1, averaged_sample_sd, averaged_sample_pcs = decode(
    x_averaged, y_averaged, group_averaged, split_by_trial=False)
print(f"averaged, sample-wise split : F1 = {averaged_sample_f1:.3f} "
      f"(SD {averaged_sample_sd:.3f}), {averaged_sample_pcs} PCs kept")
print("paper reported F1 = 0.93 for tactile")


# %% [ STEP 6 ] NPBN: the same averaged data, holding out whole traces
#
# With only 16 traces per class this split is coarse, but it is the honest one:
# no test sample has a near-copy of itself in the training set.

averaged_trial_f1, averaged_trial_sd, averaged_trial_pcs = decode(
    x_averaged, y_averaged, group_averaged, split_by_trial=True)
print(f"averaged, trace-wise split  : F1 = {averaged_trial_f1:.3f} "
      f"(SD {averaged_trial_sd:.3f}), {averaged_trial_pcs} PCs kept")
print(f"\ndrop from closing the leak, on their own pipeline: "
      f"{averaged_sample_f1 - averaged_trial_f1:+.3f}")


# %% [ STEP 7 ] NPBN: a shuffled-label control for EVERY scheme
#
# Chance is not 0.5 here. F1 depends on the split scheme, so each scheme gets its
# own control rather than being compared to one shared line. If the sample-wise
# scheme beats its own shuffled control by more than the trial-wise scheme beats
# its own, the extra is leakage, because shuffled labels contain no information.

shuffled_single_sample_f1, _, _ = decode(x, y, trial_id, split_by_trial=False,
                                         shuffle_labels=True)
shuffled_single_trial_f1, _, _ = decode(x, y, trial_id, split_by_trial=True,
                                        shuffle_labels=True)
shuffled_averaged_sample_f1, _, _ = decode(x_averaged, y_averaged, group_averaged,
                                           split_by_trial=False, shuffle_labels=True)
shuffled_averaged_trial_f1, _, _ = decode(x_averaged, y_averaged, group_averaged,
                                          split_by_trial=True, shuffle_labels=True)

control_rows = [
    {"data": "single trials", "split": "sample-wise",
     "f1": single_sample_f1, "f1_shuffled": shuffled_single_sample_f1},
    {"data": "single trials", "split": "trial-wise",
     "f1": single_trial_f1, "f1_shuffled": shuffled_single_trial_f1},
    {"data": "50-trial averages", "split": "sample-wise",
     "f1": averaged_sample_f1, "f1_shuffled": shuffled_averaged_sample_f1},
    {"data": "50-trial averages", "split": "trace-wise",
     "f1": averaged_trial_f1, "f1_shuffled": shuffled_averaged_trial_f1},
]
npbn_controls = pd.DataFrame(control_rows)
npbn_controls["above_own_control"] = (npbn_controls.f1
                                      - npbn_controls.f1_shuffled)
print(npbn_controls.round(3).to_string(index=False))
print("\nRead the last column only. It is the only quantity comparable across")
print("rows, because each row is measured against its own chance level.")


# %% [ STEP 8 ] NPBN: does F1 really keep climbing with more PCs?
#
# The paper reads "F1 still rising at PC #50" as evidence of high-dimensional
# information. But if the gain comes from leakage, more PCs simply sharpen the
# near-duplicate matching. Sweep the PC count under both splits.

pc_counts = [1, 3, 5, 10, 20, 40]
sweep_rows = []
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
        predicted = model.predict(projected[~is_train])
        score = f1_score(y[~is_train], predicted)
        sweep_rows.append({"n_pcs": n_pc,
                           "split": "trial-wise" if by_trial else "sample-wise",
                           "f1": score})

npbn_pc_sweep = pd.DataFrame(sweep_rows)
print(npbn_pc_sweep.pivot(index="n_pcs", columns="split", values="f1")
      .round(3).to_string())
print("\nRising under one split and flat or falling under the other is the")
print("signature of leakage, not of high-dimensional information.")


# %% [ STEP 9 ] NPBN: the figure


def plot_leakage(controls, pc_sweep, name, path):
    """Three panels, drawn from finished tables. No computation happens here."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    single = controls[controls.data == "single trials"]
    averaged = controls[controls.data == "50-trial averages"]

    for ax, table, title in ((axes[0], single, "single trials"),
                             (axes[1], averaged, "their pipeline: 50-trial averages")):
        positions = np.arange(len(table))
        ax.bar(positions, table.f1, color=["#c0392b", "#2c3e50"], width=0.55)
        ax.plot(positions, table.f1_shuffled, "_", color="#7b8794", ms=28, mew=2)
        for position, value in enumerate(table.f1):
            ax.text(position, value + 0.015, f"{value:.3f}", ha="center", fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels(["their split\n(over samples)", "honest split\n(held-out)"],
                           fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1, spontaneous vs evoked")
        ax.set_title(f"{name}: {title}", loc="left", fontsize=10)
    axes[1].text(1.35, averaged.f1_shuffled.iloc[-1], " grey dash =\n shuffled labels",
                 fontsize=8, color="#7b8794", va="center")

    ax = axes[2]
    for split, colour in (("sample-wise", "#c0392b"), ("trial-wise", "#2c3e50")):
        selected = pc_sweep[pc_sweep.split == split]
        ax.plot(selected.n_pcs, selected.f1, "o-", color=colour, lw=1.5, ms=4,
                label=split)
    ax.set_xlabel("number of PCs")
    ax.set_ylabel("F1")
    ax.set_title("does F1 keep climbing with more PCs?", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(f"Critique B on {name} — random splits over 1 ms samples leak "
                 f"across a 10 ms smoothing kernel", x=0.02, ha="left", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150, facecolor="white")
    return path


figure_path = plot_leakage(npbn_controls, npbn_pc_sweep, "NPBN",
                           f"{OUT_DIR}/decoding_leakage_NPBN.png")
print(f"wrote {figure_path}")


# %% [ THE SAME TEST, ONE CELL PER SESSION ]
#
# Each cell is STEPS 5 to 7 on their own pipeline -- the averaged data under both
# splits, each against its own shuffled control. Written out per session rather
# than looped.


# %% NPBN's row (walked through above)

story_npbn = pd.Series({
    "session": "NPBN",
    "neurons": npbn_spont.shape[1],
    "f1_sample_split": round(averaged_sample_f1, 3),
    "f1_sample_shuffled": round(shuffled_averaged_sample_f1, 3),
    "f1_trial_split": round(averaged_trial_f1, 3),
    "f1_trial_shuffled": round(shuffled_averaged_trial_f1, 3),
    "leak": round((averaged_sample_f1 - shuffled_averaged_sample_f1)
                  - (averaged_trial_f1 - shuffled_averaged_trial_f1), 3),
})
print(story_npbn.to_string())


# %% NPBK -- visual patterns only

npbk_spont, npbk_evoked = load_windows("NPBK")
npbk_x, npbk_y, npbk_groups = make_points(npbk_spont, npbk_evoked,
                                          npbk_spont.shape[0], average_group=GROUP)

npbk_sample_f1, npbk_sample_sd, npbk_sample_pcs = decode(
    npbk_x, npbk_y, npbk_groups, split_by_trial=False)
npbk_trial_f1, npbk_trial_sd, npbk_trial_pcs = decode(
    npbk_x, npbk_y, npbk_groups, split_by_trial=True)
npbk_sample_shuffled, _, _ = decode(npbk_x, npbk_y, npbk_groups,
                                    split_by_trial=False, shuffle_labels=True)
npbk_trial_shuffled, _, _ = decode(npbk_x, npbk_y, npbk_groups,
                                   split_by_trial=True, shuffle_labels=True)

story_npbk = pd.Series({
    "session": "NPBK",
    "neurons": npbk_spont.shape[1],
    "f1_sample_split": round(npbk_sample_f1, 3),
    "f1_sample_shuffled": round(npbk_sample_shuffled, 3),
    "f1_trial_split": round(npbk_trial_f1, 3),
    "f1_trial_shuffled": round(npbk_trial_shuffled, 3),
    "leak": round((npbk_sample_f1 - npbk_sample_shuffled)
                  - (npbk_trial_f1 - npbk_trial_shuffled), 3),
})
print(story_npbk.to_string())


# %% NPBM -- tactile and visual patterns, 1600 trials

npbm_spont, npbm_evoked = load_windows("NPBM")
npbm_x, npbm_y, npbm_groups = make_points(npbm_spont, npbm_evoked,
                                          npbm_spont.shape[0], average_group=GROUP)

npbm_sample_f1, npbm_sample_sd, npbm_sample_pcs = decode(
    npbm_x, npbm_y, npbm_groups, split_by_trial=False)
npbm_trial_f1, npbm_trial_sd, npbm_trial_pcs = decode(
    npbm_x, npbm_y, npbm_groups, split_by_trial=True)
npbm_sample_shuffled, _, _ = decode(npbm_x, npbm_y, npbm_groups,
                                    split_by_trial=False, shuffle_labels=True)
npbm_trial_shuffled, _, _ = decode(npbm_x, npbm_y, npbm_groups,
                                   split_by_trial=True, shuffle_labels=True)

story_npbm = pd.Series({
    "session": "NPBM",
    "neurons": npbm_spont.shape[1],
    "f1_sample_split": round(npbm_sample_f1, 3),
    "f1_sample_shuffled": round(npbm_sample_shuffled, 3),
    "f1_trial_split": round(npbm_trial_f1, 3),
    "f1_trial_shuffled": round(npbm_trial_shuffled, 3),
    "leak": round((npbm_sample_f1 - npbm_sample_shuffled)
                  - (npbm_trial_f1 - npbm_trial_shuffled), 3),
})
print(story_npbm.to_string())


# %% NPBO -- tactile patterns

npbo_spont, npbo_evoked = load_windows("NPBO")
npbo_x, npbo_y, npbo_groups = make_points(npbo_spont, npbo_evoked,
                                          npbo_spont.shape[0], average_group=GROUP)

npbo_sample_f1, npbo_sample_sd, npbo_sample_pcs = decode(
    npbo_x, npbo_y, npbo_groups, split_by_trial=False)
npbo_trial_f1, npbo_trial_sd, npbo_trial_pcs = decode(
    npbo_x, npbo_y, npbo_groups, split_by_trial=True)
npbo_sample_shuffled, _, _ = decode(npbo_x, npbo_y, npbo_groups,
                                    split_by_trial=False, shuffle_labels=True)
npbo_trial_shuffled, _, _ = decode(npbo_x, npbo_y, npbo_groups,
                                   split_by_trial=True, shuffle_labels=True)

story_npbo = pd.Series({
    "session": "NPBO",
    "neurons": npbo_spont.shape[1],
    "f1_sample_split": round(npbo_sample_f1, 3),
    "f1_sample_shuffled": round(npbo_sample_shuffled, 3),
    "f1_trial_split": round(npbo_trial_f1, 3),
    "f1_trial_shuffled": round(npbo_trial_shuffled, 3),
    "leak": round((npbo_sample_f1 - npbo_sample_shuffled)
                  - (npbo_trial_f1 - npbo_trial_shuffled), 3),
})
print(story_npbo.to_string())


# %% all four side by side

everyone = pd.DataFrame([story_npbn, story_npbk, story_npbm, story_npbo])
print(everyone.to_string(index=False))
print("\n'leak' is how much more the sample-wise scheme beats its own chance")
print("level than the honest scheme beats its own. That is the part of their")
print("0.93 that the split scheme manufactured.")

everyone.to_csv(f"{OUT_DIR}/decoding_leakage_summary.csv", index=False)
print(f"\nwrote {OUT_DIR}/decoding_leakage_summary.csv")

# %%
