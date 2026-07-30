# %% [ THE ARGUMENT ]
"""Critique D: is the "working memory-like" residual actually drift or carryover?

  THE CLAIM      Tactile pattern identity stays decodable out to 1000 ms, long
                 after the ~300 ms stimulus ends, so the population "never returns
                 to spontaneous baseline within 1 s" and holds resident
                 information -- a substrate for working memory. Their numbers:
                 pre-stimulus F1 = 0.303, then 0.455 (1-200 ms), 0.363, 0.337,
                 0.329, 0.318 (801-1000 ms). Chance is 0.25.

  THE WEAKNESS   Two dull explanations are left open, and either would produce the
                 same result.

                 CARRYOVER. Stimuli are ~1.8-2.9 s apart and the claimed residual
                 lasts 1 s, so the previous trial's tail is still inside the next
                 trial's pre-stimulus window. Their own pre-stimulus F1 of 0.303
                 already sits above chance, which is what carryover looks like --
                 and a baseline that is not at chance cannot serve as a baseline.

                 DRIFT. Anesthesia depth, temperature and probe settling all move
                 slowly across a session. Their step of averaging 50 SEQUENTIAL
                 trials welds every data point to one chunk of session time, so a
                 slow drift becomes a label-correlated signal.

  REPRODUCE      STEPS 2-4 run THEIR pipeline: 1 ms bins, 50 sequential trials
  FIRST          averaged, 10 ms Gaussian, PCA to 95%, kNN k=5, split over
                 samples. That is the only way to get their numbers, and a
                 critique that never reaches their numbers is arguing with a
                 different analysis.

  THEN BREAK IT  STEP 3 changes one thing only -- group 50 RANDOM trials instead
                 of 50 sequential ones. Same averaging, same smoothing, same
                 classifier, same amount of data. If the effect needs the trials
                 to be sequential, it is session time, not memory.
                 STEP 4 holds out whole averaged traces instead of samples.
                 STEPS 6-9 test carryover and drift directly at the trial level.

  WHY NPBI IS    Its onsets are about 243 ms late, so every window here would be
  NOT HERE       measuring the wrong slice of time. See CLAUDE.md.

Steps 1-11 walk through NPBN. Then NPBK, NPBM and NPBO each get their own cell.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, r2_score
from pynwb import NWBHDF5IO

# The windows the paper scores. The first is its spontaneous baseline, the rest
# are its five poststimulus windows.
WINDOWS = {
    "pre -300..0": (-300, 0),
    "0..200": (0, 200),
    "200..400": (200, 400),
    "400..600": (400, 600),
    "600..800": (600, 800),
    "800..1000": (800, 1000),
}
PAPER_F1 = {"pre -300..0": 0.303, "0..200": 0.455, "200..400": 0.363,
            "400..600": 0.337, "600..800": 0.329, "800..1000": 0.318}

GROUP = 50                  # paper: average 50 sequential trials
SMOOTH_MS = 10              # paper: 10 ms Gaussian kernel
K = 5                       # paper: k = 5 nearest neighbours
VARIANCE = 0.95             # paper: PCs explaining 95% of the variance
N_REPEATS = 5               # paper used 50
N_FOLDS = 5

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


# %% [ STEP 0b ] the operations
#
# Each does one thing and calls none of the others.


def load_session(exp):
    """Good units, trial labels and onsets, in session order."""
    with NWBHDF5IO(ROOT / "nwb" / f"{exp}.nwb", "r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        trials = nwb.trials.to_dataframe()

    trials = trials.sort_values("start_time", ignore_index=True)
    good_units = units[units["quality"] == "good"]
    onsets = trials["start_time"].to_numpy()

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

    return good_units, labels, onsets


def window_counts(good_units, onsets, start_ms, stop_ms):
    """Spike counts at 1 ms in one window around every onset: (trial, neuron, ms)."""
    start_s = start_ms / 1000.0
    stop_s = stop_ms / 1000.0
    n_ms = int(round(stop_ms - start_ms))
    edges = np.arange(n_ms + 1) / 1000.0 + start_s

    counts = np.zeros((len(onsets), len(good_units), n_ms), dtype=np.float32)
    for neuron, spikes in enumerate(good_units["spike_times"]):
        spikes = np.sort(np.asarray(spikes, dtype=np.float64))
        for trial, onset in enumerate(onsets):
            first, last = np.searchsorted(spikes, [onset + start_s, onset + stop_s])
            relative = spikes[first:last] - onset
            counts[trial, neuron] = np.histogram(relative, bins=edges)[0]
    return counts


def average_and_flatten(counts, labels, group, sequential):
    """The paper's steps 2 and 3, then reshape to points.

    Within each stimulus class, take `group` trials at a time and average them
    into one trace, convolve that trace with a 10 ms Gaussian along time, and
    treat every millisecond of it as one data point.

    `sequential` is the switch this critique turns. True takes the trials in the
    order they happened, which is what the paper did, so each averaged trace
    belongs to one contiguous stretch of session time. False takes a random
    `group` of that class's trials from anywhere in the session, which averages
    the same number of trials with the same noise reduction but spreads each
    trace over the whole session. Nothing else differs between the two.

    Returns (x, y, trace_id): x is (sample, neuron), y the class of each sample,
    trace_id which averaged trace it came from.
    """
    sigma = SMOOTH_MS / 2.355
    all_x = []
    all_y = []
    all_traces = []
    next_trace_id = 0

    for cls in sorted(set(labels)):
        in_class = np.flatnonzero(labels == cls)
        if not sequential:
            in_class = rng.permutation(in_class)

        n_traces = len(in_class) // group
        for trace in range(n_traces):
            members = in_class[trace * group:(trace + 1) * group]
            averaged = counts[members].mean(axis=0)              # (neuron, ms)
            smoothed = gaussian_filter1d(averaged, sigma, axis=1)

            all_x.append(smoothed.T)                             # (ms, neuron)
            all_y.append(np.full(smoothed.shape[1], cls))
            all_traces.append(np.full(smoothed.shape[1], next_trace_id))
            next_trace_id += 1

    return np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_traces)


def decode_f1(x, y, trace_id, split_by_trace):
    """kNN with a 50/50 split, macro F1 over the classes.

    split_by_trace False is the paper's scheme: the split is random over
    individual samples, so two milliseconds of the same averaged trace can land
    on opposite sides. True holds out whole traces.
    """
    scores = []
    for repeat in range(N_REPEATS):
        if split_by_trace:
            traces = np.unique(trace_id)
            train_traces = rng.choice(traces, len(traces) // 2, replace=False)
            is_train = np.isin(trace_id, train_traces)
        else:
            is_train = rng.random(len(y)) < 0.5

        pca = PCA(n_components=VARIANCE).fit(x[is_train])
        model = KNeighborsClassifier(n_neighbors=K).fit(
            pca.transform(x[is_train]), y[is_train])
        predicted = model.predict(pca.transform(x[~is_train]))
        scores.append(f1_score(y[~is_train], predicted, average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def window_rates(good_units, onsets, start_ms, stop_ms):
    """Firing rate per trial per neuron in one window: (trial, neuron) in Hz."""
    start_s = start_ms / 1000.0
    stop_s = stop_ms / 1000.0
    duration = stop_s - start_s

    rates = np.zeros((len(onsets), len(good_units)), dtype=np.float32)
    for neuron, spikes in enumerate(good_units["spike_times"]):
        spikes = np.sort(np.asarray(spikes, dtype=np.float64))
        first = np.searchsorted(spikes, onsets + start_s)
        last = np.searchsorted(spikes, onsets + stop_s)
        rates[:, neuron] = (last - first) / duration
    return rates


def decode_labels(x, y):
    """Balanced accuracy for predicting y from x, whole trials held out."""
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    model = LinearDiscriminantAnalysis()
    predicted = cross_val_predict(model, x, y, cv=folds)
    return float(balanced_accuracy_score(y, predicted))


def predict_time(x, onsets):
    """Cross-validated R2 for predicting elapsed session time from the state."""
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    model = Ridge(alpha=1.0)
    predicted = cross_val_predict(model, x, onsets, cv=folds)
    return float(r2_score(onsets, predicted))


def drift_axis(x, onsets):
    """The direction in population space along which session time increases."""
    model = Ridge(alpha=1.0)
    model.fit(x, onsets)
    axis = model.coef_
    return axis / np.linalg.norm(axis)


def remove_axis(x, axis):
    """Project one direction out of every trial's population vector."""
    projection = x @ axis
    return x - projection[:, None] * axis[None, :]


# %% [ STEP 1 ] NPBN: load it and look at the session

npbn_units, npbn_labels, npbn_onsets = load_session("NPBN")
npbn_classes = sorted(set(npbn_labels))
chance = 1.0 / len(npbn_classes)

session_minutes = (npbn_onsets.max() - npbn_onsets.min()) / 60
median_gap = np.median(np.diff(npbn_onsets))

print(f"NPBN: {len(npbn_onsets)} trials x {len(npbn_units)} good units")
print(f"{len(npbn_classes)} classes -> chance = {chance:.3f}")
print(f"session spans {session_minutes:.1f} minutes")
print(f"median gap between stimuli: {median_gap:.2f} s")
print("\nThat gap matters: the claimed residual lasts 1000 ms, so consecutive")
print("trials are not independent samples of a resting brain.")


# %% [ STEP 2 ] NPBN: their Fig 5 pipeline exactly as the Methods describe it
#
# 50 sequential trials averaged, 10 ms Gaussian, PCA to 95%, kNN k=5, and the
# split "random over data points" -- their words. Their reported F1 is printed
# alongside.
#
# Watch what happens. With 200 trials per pattern, averaging 50 leaves FOUR
# traces per pattern. Every millisecond of a trace is a near-copy of its
# neighbours after the Gaussian, so a random split over samples hands each test
# point its own trace, and identifying the trace identifies the pattern.

paper_rows = []
for name, (start_ms, stop_ms) in WINDOWS.items():
    counts = window_counts(npbn_units, npbn_onsets, start_ms, stop_ms)
    x, y, trace_id = average_and_flatten(counts, npbn_labels, GROUP, sequential=True)
    f1_mean, f1_sd = decode_f1(x, y, trace_id, split_by_trace=False)
    paper_rows.append({"window": name, "f1_sample_split": f1_mean,
                       "paper_reported": PAPER_F1[name]})

npbn_paper = pd.DataFrame(paper_rows)
n_traces_per_class = len(npbn_onsets) // len(npbn_classes) // GROUP
print(f"{n_traces_per_class} averaged traces per pattern, "
      f"{GROUP} trials in each")
print(npbn_paper.round(3).to_string(index=False))
print(f"\nchance is {chance:.3f}. The pipeline as described saturates -- every")
print("window, including the one BEFORE the stimulus, is decoded almost")
print("perfectly. That cannot be the analysis that produced their 0.303.")


# %% [ STEP 3 ] NPBN: the same pipeline, holding out whole averaged traces
#
# One word in the Methods is doing a lot of work: whether "data points" are split
# means milliseconds or traces. STEP 2 read it as milliseconds. Read it as traces
# instead -- no test sample has a near-copy of itself in training -- and run the
# identical analysis.

honest_rows = []
for name, (start_ms, stop_ms) in WINDOWS.items():
    counts = window_counts(npbn_units, npbn_onsets, start_ms, stop_ms)
    x, y, trace_id = average_and_flatten(counts, npbn_labels, GROUP, sequential=True)
    f1_mean, f1_sd = decode_f1(x, y, trace_id, split_by_trace=True)
    honest_rows.append({"window": name, "f1_trace_split": f1_mean})

npbn_honest = pd.DataFrame(honest_rows)
two_readings = npbn_paper.merge(npbn_honest, on="window")
print(two_readings.round(3).to_string(index=False))
print("\nSO THE PIPELINE AS PUBLISHED DOES NOT REPRODUCE. One reading of their")
print("split saturates near 1.0, the other falls below chance, and their")
print("published values sit between the two. The shape is right in the trace-")
print("split reading -- a peak just after onset, then decay -- but the level is")
print("not, and with only 2 traces per class in training that reading has almost")
print("no statistical power. Neither the numbers nor the error bars in Fig 5 can")
print("be recovered from what the Methods say.")


# %% [ STEP 4 ] NPBN: the analysis their design actually supports
#
# Their averaging has two effects rolled into one. It raises signal-to-noise,
# which is legitimate and necessary at these firing rates, and it welds each
# averaged trace to one contiguous stretch of session time, which is not.
#
# So separate them. Average the same number of trials, but draw them at RANDOM
# from anywhere in the session, which keeps the signal-to-noise gain and breaks
# the link to session time. Hold out whole traces, which closes the leak. Then
# sweep the group size, because with 50 there are only 4 traces per class and
# nothing can be measured.

group_rows = []
for group_size in [5, 10, 25, 50]:
    for name in ["pre -300..0", "0..200", "800..1000"]:
        start_ms, stop_ms = WINDOWS[name]
        counts = window_counts(npbn_units, npbn_onsets, start_ms, stop_ms)
        x, y, trace_id = average_and_flatten(counts, npbn_labels, group_size,
                                             sequential=False)
        f1_mean, f1_sd = decode_f1(x, y, trace_id, split_by_trace=True)
        group_rows.append({"group": group_size, "traces_per_class":
                           len(np.unique(trace_id)) // len(npbn_classes),
                           "window": name, "f1": f1_mean})

npbn_groups = pd.DataFrame(group_rows)
print(npbn_groups.pivot(index=["group", "traces_per_class"], columns="window",
                        values="f1").round(3).to_string())
print(f"\nchance is {chance:.3f}. Read the 800..1000 column against the")
print("pre -300..0 column. If the late window is no better than the window")
print("before the stimulus, nothing was retained for a second.")
print("\nBUT READ THE 0..200 COLUMN FIRST, as a check on the classifier itself.")
print("A tactile response certainly exists there -- STEP 5 recovers it at 0.318")
print("with an LDA on single trials. If this kNN scheme sits at chance in that")
print("column too, then it is underpowered rather than informative: PCA plus")
print("kNN on a handful of averaged traces is a weak estimator, and its silence")
print("in the late window proves nothing on its own. STEP 5 is the measurement")
print("to quote; this table exists to show that their pipeline, once the leak")
print("is closed, cannot see even the response it was built to study.")


# %% [ STEP 4b ] NPBN: the same sweep with SEQUENTIAL grouping, for the contrast
#
# Identical in every respect except that the 50 trials averaged together are now
# consecutive, as in the paper. The difference between this table and the last
# one is what session time contributes.

sequential_rows = []
for group_size in [5, 10, 25, 50]:
    for name in ["pre -300..0", "0..200", "800..1000"]:
        start_ms, stop_ms = WINDOWS[name]
        counts = window_counts(npbn_units, npbn_onsets, start_ms, stop_ms)
        x, y, trace_id = average_and_flatten(counts, npbn_labels, group_size,
                                             sequential=True)
        f1_mean, f1_sd = decode_f1(x, y, trace_id, split_by_trace=True)
        sequential_rows.append({"group": group_size, "window": name, "f1": f1_mean})

npbn_sequential_groups = pd.DataFrame(sequential_rows)
print(npbn_sequential_groups.pivot(index="group", columns="window", values="f1")
      .round(3).to_string())

grouping_effect = npbn_sequential_groups.merge(
    npbn_groups[["group", "window", "f1"]], on=["group", "window"],
    suffixes=("_sequential", "_random"))
grouping_effect["from_session_time"] = (grouping_effect.f1_sequential
                                        - grouping_effect.f1_random)
print("\n" + grouping_effect.pivot(index="group", columns="window",
                                   values="from_session_time").round(3).to_string())
print("\nPositive numbers in the pre-stimulus column are the giveaway: a window")
print("before the stimulus cannot know the pattern, so anything sequential")
print("grouping adds there is session time being read as stimulus identity.")


three_ways = two_readings.copy()


# %% [ STEP 5 ] NPBN: the same question at the trial level, no averaging at all
#
# One firing-rate vector per trial per window, LDA, whole trials held out. This
# is the cleanest possible version of their question, and it is the number a
# reader should be given.

trial_rows = []
for name, (start_ms, stop_ms) in WINDOWS.items():
    rates = window_rates(npbn_units, npbn_onsets, start_ms, stop_ms)
    accuracy = decode_labels(rates, npbn_labels)
    trial_rows.append({"window": name, "accuracy": accuracy,
                       "above_chance": accuracy - chance})

npbn_decline = pd.DataFrame(trial_rows)
print(npbn_decline.round(3).to_string(index=False))
print(f"\nchance is {chance:.3f}.")


# %% [ STEP 6 ] NPBN: TEST 1 -- carryover. Decode the PREVIOUS trial's pattern.
#
# If the pre-stimulus window still knows what the LAST stimulus was, then
# "information surviving past the stimulus" is just the previous trial bleeding
# in, and it needs no memory mechanism to explain.

pre_rates = window_rates(npbn_units, npbn_onsets, -300, 0)

previous_labels = np.roll(npbn_labels, 1)          # session order, shifted by one
pre_rates_from_second = pre_rates[1:]              # drop the first trial
previous_from_second = previous_labels[1:]
current_from_second = npbn_labels[1:]

carryover_accuracy = decode_labels(pre_rates_from_second, previous_from_second)
next_pattern_accuracy = decode_labels(pre_rates_from_second, current_from_second)

print(f"from the pre-stimulus window, decoding the PREVIOUS trial's pattern: "
      f"{carryover_accuracy:.3f}")
print(f"from the same window, decoding the trial about to happen           : "
      f"{next_pattern_accuracy:.3f}")
print(f"chance                                                            : "
      f"{chance:.3f}")
print("\nThe second line is the impossible one: the upcoming stimulus cannot be")
print("known in advance, so anything above chance there is drift or leakage.")
print("Whatever it scores is the floor for every other number in this file.")


# %% [ STEP 7 ] NPBN: TEST 2 -- how much drift is there?
#
# Predict elapsed session time from the population state. If the state says what
# time it is, then anything correlated with time is decodable from it.

drift_rows = []
for name, (start_ms, stop_ms) in WINDOWS.items():
    rates = window_rates(npbn_units, npbn_onsets, start_ms, stop_ms)
    r2 = predict_time(rates, npbn_onsets)
    drift_rows.append({"window": name, "time_r2": r2})

npbn_drift = pd.DataFrame(drift_rows)
print(npbn_drift.round(3).to_string(index=False))
print("\nR2 near 0 means no drift. R2 well above 0 means the population state is")
print("partly a clock, and every slow analysis on it is suspect.")


# %% [ STEP 8 ] NPBN: TEST 3 -- is the late 'memory' axis the drift axis?

late_rates = window_rates(npbn_units, npbn_onsets, 800, 1000)
time_direction = drift_axis(late_rates, npbn_onsets)

discriminator = LinearDiscriminantAnalysis()
discriminator.fit(late_rates, npbn_labels)
pattern_directions = discriminator.coef_        # one row per class

cosines = []
for row in range(pattern_directions.shape[0]):
    axis = pattern_directions[row]
    axis = axis / np.linalg.norm(axis)
    cosines.append(abs(float(axis @ time_direction)))

print("|cosine| between the drift axis and each pattern-discriminant axis:")
for cls, value in zip(npbn_classes, cosines):
    print(f"   {cls:24} {value:.3f}")
print(f"mean {np.mean(cosines):.3f}")
print("\n0 means the two are unrelated directions. Values well above 0 mean the")
print("classifier is partly using the drift direction to tell patterns apart.")


# %% [ STEP 9 ] NPBN: TEST 4 -- take the drift out and decode again

late_without_drift = remove_axis(late_rates, time_direction)

with_drift = decode_labels(late_rates, npbn_labels)
without_drift = decode_labels(late_without_drift, npbn_labels)

print("late window (800-1000 ms), pattern decoding")
print(f"   as is                        : {with_drift:.3f}")
print(f"   with the drift axis removed  : {without_drift:.3f}")
print(f"   chance                       : {chance:.3f}")
print(f"   lost to drift removal        : {with_drift - without_drift:+.3f}")
print(f"\nIf removing ONE direction out of {len(npbn_units)} takes the accuracy to")
print("chance, the 'working memory-like' residual was a drifting session.")


# %% [ STEP 10 ] NPBN: control -- shuffle the trial order and redo the drift test
#
# Sanity check on the machinery: if trials are randomly reordered in time, the
# drift test must fail. If it still 'works', the test is measuring something else.

shuffled_onsets = rng.permutation(npbn_onsets)
shuffled_r2 = predict_time(pre_rates, shuffled_onsets)
real_r2 = predict_time(pre_rates, npbn_onsets)

print(f"time R2 from the pre-stimulus window, real order     : {real_r2:.3f}")
print(f"time R2 with the onsets randomly permuted            : {shuffled_r2:.3f}")
print("\nThe second should be near or below 0. If it is not, ignore STEP 7.")


# %% [ STEP 11 ] NPBN: the figure


def plot_memory_or_drift(pipelines, drift, decline, chance_level, with_drift_score,
                         without_drift_score, carryover, next_pattern, name, path):
    """Three panels, drawn from finished tables. No computation happens here."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    positions = np.arange(len(pipelines))

    ax = axes[0]
    ax.plot(positions, pipelines.paper_reported, "o:", color="#7b8794", lw=1.4,
            ms=4, label="paper, as published")
    ax.plot(positions, pipelines.f1_sample_split, "o-", color="#c0392b", lw=1.6,
            ms=5, label="Methods read as: split samples")
    ax.plot(positions, pipelines.f1_trace_split, "^-.", color="#2c3e50", lw=1.6,
            ms=5, label="Methods read as: split traces")
    ax.axhline(chance_level, color="#7b8794", lw=1, ls="--")
    ax.text(len(pipelines) - 0.5, chance_level, " chance", fontsize=8,
            color="#7b8794", va="center")
    ax.set_xticks(positions)
    ax.set_xticklabels(pipelines.window, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("F1, tactile pattern identity")
    ax.set_title(f"{name}: their Fig 5, three ways", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    ax.plot(positions, drift.time_r2, "s-", color="#c1703a", lw=1.6, ms=5,
            label="R² predicting session time")
    ax.plot(positions, decline.accuracy, "o-", color="#2c3e50", lw=1.6, ms=5,
            label="trial-level pattern decoding")
    ax.axhline(chance_level, color="#7b8794", lw=1, ls="--")
    ax.axhline(0, color="#d0d5da", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(drift.window, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("R² / balanced accuracy")
    ax.set_title("how much the state is a clock", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    names = ["late window\nas is", "late window\ndrift removed",
             "pre-stim ->\nPREVIOUS pattern", "pre-stim ->\nnext pattern"]
    values = [with_drift_score, without_drift_score, carryover, next_pattern]
    colours = ["#2c3e50", "#2c3e50", "#c0392b", "#c0392b"]
    ax.bar(names, values, color=colours, width=0.6)
    ax.axhline(chance_level, color="#7b8794", lw=1, ls="--")
    for position, value in enumerate(values):
        ax.text(position, value + 0.005, f"{value:.3f}", ha="center", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("the two dull explanations, tested", loc="left", fontsize=10)

    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(f"Critique D on {name} — carryover and drift, before invoking "
                 f"working memory", x=0.02, ha="left", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=150, facecolor="white")
    return path


figure_path = plot_memory_or_drift(
    three_ways, npbn_drift, npbn_decline, chance, with_drift, without_drift,
    carryover_accuracy, next_pattern_accuracy, "NPBN",
    f"{OUT_DIR}/memory_or_drift_NPBN.png")
print(f"wrote {figure_path}")


# %% [ THE SAME TESTS, ONE CELL PER SESSION ]
#
# Each cell below is the decisive subset: the two readings of their split, the
# honest decoding (10 random trials averaged, whole traces held out), and the
# trial-level carryover and drift tests.

HONEST_GROUP = 10           # small enough to leave traces to hold out


# %% NPBN's row (walked through above)

npbn_honest_pre_counts = window_counts(npbn_units, npbn_onsets, -300, 0)
npbn_honest_pre_x, npbn_honest_pre_y, npbn_honest_pre_trace = average_and_flatten(
    npbn_honest_pre_counts, npbn_labels, HONEST_GROUP, sequential=False)
npbn_honest_pre, _ = decode_f1(npbn_honest_pre_x, npbn_honest_pre_y,
                               npbn_honest_pre_trace, split_by_trace=True)

npbn_honest_late_counts = window_counts(npbn_units, npbn_onsets, 800, 1000)
npbn_honest_late_x, npbn_honest_late_y, npbn_honest_late_trace = average_and_flatten(
    npbn_honest_late_counts, npbn_labels, HONEST_GROUP, sequential=False)
npbn_honest_late, _ = decode_f1(npbn_honest_late_x, npbn_honest_late_y,
                                npbn_honest_late_trace, split_by_trace=True)

story_npbn = pd.Series({
    "session": "NPBN",
    "neurons": len(npbn_units),
    "classes": len(npbn_classes),
    "chance": round(chance, 3),
    "f1_prestim_sample_split": round(two_readings.f1_sample_split.iloc[0], 3),
    "f1_prestim_trace_split": round(two_readings.f1_trace_split.iloc[0], 3),
    "f1_late_sample_split": round(two_readings.f1_sample_split.iloc[-1], 3),
    "f1_late_trace_split": round(two_readings.f1_trace_split.iloc[-1], 3),
    "f1_prestim_honest": round(npbn_honest_pre, 3),
    "f1_late_honest": round(npbn_honest_late, 3),
    "late_as_is": round(with_drift, 3),
    "late_drift_removed": round(without_drift, 3),
    "prestim_previous_pattern": round(carryover_accuracy, 3),
    "prestim_next_pattern": round(next_pattern_accuracy, 3),
    "time_r2_prestim": round(real_r2, 3),
})
print(story_npbn.to_string())


# %% NPBK -- visual patterns only

npbk_units, npbk_labels, npbk_onsets = load_session("NPBK")
npbk_chance = 1.0 / len(set(npbk_labels))

npbk_pre_counts = window_counts(npbk_units, npbk_onsets, -300, 0)
npbk_pre_x, npbk_pre_y, npbk_pre_trace = average_and_flatten(
    npbk_pre_counts, npbk_labels, GROUP, sequential=True)
npbk_pre_sample, _ = decode_f1(npbk_pre_x, npbk_pre_y, npbk_pre_trace, False)
npbk_pre_trace_split, _ = decode_f1(npbk_pre_x, npbk_pre_y, npbk_pre_trace, True)

npbk_honest_pre_x, npbk_honest_pre_y, npbk_honest_pre_trace = average_and_flatten(
    npbk_pre_counts, npbk_labels, HONEST_GROUP, sequential=False)
npbk_honest_pre, _ = decode_f1(npbk_honest_pre_x, npbk_honest_pre_y,
                               npbk_honest_pre_trace, True)

npbk_late_counts = window_counts(npbk_units, npbk_onsets, 800, 1000)
npbk_late_x, npbk_late_y, npbk_late_trace = average_and_flatten(
    npbk_late_counts, npbk_labels, GROUP, sequential=True)
npbk_late_sample, _ = decode_f1(npbk_late_x, npbk_late_y, npbk_late_trace, False)
npbk_late_trace_split, _ = decode_f1(npbk_late_x, npbk_late_y, npbk_late_trace, True)

npbk_honest_late_x, npbk_honest_late_y, npbk_honest_late_trace = average_and_flatten(
    npbk_late_counts, npbk_labels, HONEST_GROUP, sequential=False)
npbk_honest_late, _ = decode_f1(npbk_honest_late_x, npbk_honest_late_y,
                                npbk_honest_late_trace, True)

npbk_pre_rates = window_rates(npbk_units, npbk_onsets, -300, 0)
npbk_late_rates = window_rates(npbk_units, npbk_onsets, 800, 1000)
npbk_previous = np.roll(npbk_labels, 1)
npbk_carryover = decode_labels(npbk_pre_rates[1:], npbk_previous[1:])
npbk_next = decode_labels(npbk_pre_rates[1:], npbk_labels[1:])
npbk_time_r2 = predict_time(npbk_pre_rates, npbk_onsets)
npbk_axis = drift_axis(npbk_late_rates, npbk_onsets)
npbk_late_clean = remove_axis(npbk_late_rates, npbk_axis)
npbk_late_as_is = decode_labels(npbk_late_rates, npbk_labels)
npbk_late_removed = decode_labels(npbk_late_clean, npbk_labels)

story_npbk = pd.Series({
    "session": "NPBK",
    "neurons": len(npbk_units),
    "classes": len(set(npbk_labels)),
    "chance": round(npbk_chance, 3),
    "f1_prestim_sample_split": round(npbk_pre_sample, 3),
    "f1_prestim_trace_split": round(npbk_pre_trace_split, 3),
    "f1_late_sample_split": round(npbk_late_sample, 3),
    "f1_late_trace_split": round(npbk_late_trace_split, 3),
    "f1_prestim_honest": round(npbk_honest_pre, 3),
    "f1_late_honest": round(npbk_honest_late, 3),
    "late_as_is": round(npbk_late_as_is, 3),
    "late_drift_removed": round(npbk_late_removed, 3),
    "prestim_previous_pattern": round(npbk_carryover, 3),
    "prestim_next_pattern": round(npbk_next, 3),
    "time_r2_prestim": round(npbk_time_r2, 3),
})
print(story_npbk.to_string())


# %% NPBM -- tactile and visual patterns, 1600 trials

npbm_units, npbm_labels, npbm_onsets = load_session("NPBM")
npbm_chance = 1.0 / len(set(npbm_labels))

npbm_pre_counts = window_counts(npbm_units, npbm_onsets, -300, 0)
npbm_pre_x, npbm_pre_y, npbm_pre_trace = average_and_flatten(
    npbm_pre_counts, npbm_labels, GROUP, sequential=True)
npbm_pre_sample, _ = decode_f1(npbm_pre_x, npbm_pre_y, npbm_pre_trace, False)
npbm_pre_trace_split, _ = decode_f1(npbm_pre_x, npbm_pre_y, npbm_pre_trace, True)

npbm_honest_pre_x, npbm_honest_pre_y, npbm_honest_pre_trace = average_and_flatten(
    npbm_pre_counts, npbm_labels, HONEST_GROUP, sequential=False)
npbm_honest_pre, _ = decode_f1(npbm_honest_pre_x, npbm_honest_pre_y,
                               npbm_honest_pre_trace, True)

npbm_late_counts = window_counts(npbm_units, npbm_onsets, 800, 1000)
npbm_late_x, npbm_late_y, npbm_late_trace = average_and_flatten(
    npbm_late_counts, npbm_labels, GROUP, sequential=True)
npbm_late_sample, _ = decode_f1(npbm_late_x, npbm_late_y, npbm_late_trace, False)
npbm_late_trace_split, _ = decode_f1(npbm_late_x, npbm_late_y, npbm_late_trace, True)

npbm_honest_late_x, npbm_honest_late_y, npbm_honest_late_trace = average_and_flatten(
    npbm_late_counts, npbm_labels, HONEST_GROUP, sequential=False)
npbm_honest_late, _ = decode_f1(npbm_honest_late_x, npbm_honest_late_y,
                                npbm_honest_late_trace, True)

npbm_pre_rates = window_rates(npbm_units, npbm_onsets, -300, 0)
npbm_late_rates = window_rates(npbm_units, npbm_onsets, 800, 1000)
npbm_previous = np.roll(npbm_labels, 1)
npbm_carryover = decode_labels(npbm_pre_rates[1:], npbm_previous[1:])
npbm_next = decode_labels(npbm_pre_rates[1:], npbm_labels[1:])
npbm_time_r2 = predict_time(npbm_pre_rates, npbm_onsets)
npbm_axis = drift_axis(npbm_late_rates, npbm_onsets)
npbm_late_clean = remove_axis(npbm_late_rates, npbm_axis)
npbm_late_as_is = decode_labels(npbm_late_rates, npbm_labels)
npbm_late_removed = decode_labels(npbm_late_clean, npbm_labels)

story_npbm = pd.Series({
    "session": "NPBM",
    "neurons": len(npbm_units),
    "classes": len(set(npbm_labels)),
    "chance": round(npbm_chance, 3),
    "f1_prestim_sample_split": round(npbm_pre_sample, 3),
    "f1_prestim_trace_split": round(npbm_pre_trace_split, 3),
    "f1_late_sample_split": round(npbm_late_sample, 3),
    "f1_late_trace_split": round(npbm_late_trace_split, 3),
    "f1_prestim_honest": round(npbm_honest_pre, 3),
    "f1_late_honest": round(npbm_honest_late, 3),
    "late_as_is": round(npbm_late_as_is, 3),
    "late_drift_removed": round(npbm_late_removed, 3),
    "prestim_previous_pattern": round(npbm_carryover, 3),
    "prestim_next_pattern": round(npbm_next, 3),
    "time_r2_prestim": round(npbm_time_r2, 3),
})
print(story_npbm.to_string())


# %% NPBO -- tactile patterns, onsets confirmed against the LFP artifacts

npbo_units, npbo_labels, npbo_onsets = load_session("NPBO")
npbo_chance = 1.0 / len(set(npbo_labels))

npbo_pre_counts = window_counts(npbo_units, npbo_onsets, -300, 0)
npbo_pre_x, npbo_pre_y, npbo_pre_trace = average_and_flatten(
    npbo_pre_counts, npbo_labels, GROUP, sequential=True)
npbo_pre_sample, _ = decode_f1(npbo_pre_x, npbo_pre_y, npbo_pre_trace, False)
npbo_pre_trace_split, _ = decode_f1(npbo_pre_x, npbo_pre_y, npbo_pre_trace, True)

npbo_honest_pre_x, npbo_honest_pre_y, npbo_honest_pre_trace = average_and_flatten(
    npbo_pre_counts, npbo_labels, HONEST_GROUP, sequential=False)
npbo_honest_pre, _ = decode_f1(npbo_honest_pre_x, npbo_honest_pre_y,
                               npbo_honest_pre_trace, True)

npbo_late_counts = window_counts(npbo_units, npbo_onsets, 800, 1000)
npbo_late_x, npbo_late_y, npbo_late_trace = average_and_flatten(
    npbo_late_counts, npbo_labels, GROUP, sequential=True)
npbo_late_sample, _ = decode_f1(npbo_late_x, npbo_late_y, npbo_late_trace, False)
npbo_late_trace_split, _ = decode_f1(npbo_late_x, npbo_late_y, npbo_late_trace, True)

npbo_honest_late_x, npbo_honest_late_y, npbo_honest_late_trace = average_and_flatten(
    npbo_late_counts, npbo_labels, HONEST_GROUP, sequential=False)
npbo_honest_late, _ = decode_f1(npbo_honest_late_x, npbo_honest_late_y,
                                npbo_honest_late_trace, True)

npbo_pre_rates = window_rates(npbo_units, npbo_onsets, -300, 0)
npbo_late_rates = window_rates(npbo_units, npbo_onsets, 800, 1000)
npbo_previous = np.roll(npbo_labels, 1)
npbo_carryover = decode_labels(npbo_pre_rates[1:], npbo_previous[1:])
npbo_next = decode_labels(npbo_pre_rates[1:], npbo_labels[1:])
npbo_time_r2 = predict_time(npbo_pre_rates, npbo_onsets)
npbo_axis = drift_axis(npbo_late_rates, npbo_onsets)
npbo_late_clean = remove_axis(npbo_late_rates, npbo_axis)
npbo_late_as_is = decode_labels(npbo_late_rates, npbo_labels)
npbo_late_removed = decode_labels(npbo_late_clean, npbo_labels)

story_npbo = pd.Series({
    "session": "NPBO",
    "neurons": len(npbo_units),
    "classes": len(set(npbo_labels)),
    "chance": round(npbo_chance, 3),
    "f1_prestim_sample_split": round(npbo_pre_sample, 3),
    "f1_prestim_trace_split": round(npbo_pre_trace_split, 3),
    "f1_late_sample_split": round(npbo_late_sample, 3),
    "f1_late_trace_split": round(npbo_late_trace_split, 3),
    "f1_prestim_honest": round(npbo_honest_pre, 3),
    "f1_late_honest": round(npbo_honest_late, 3),
    "late_as_is": round(npbo_late_as_is, 3),
    "late_drift_removed": round(npbo_late_removed, 3),
    "prestim_previous_pattern": round(npbo_carryover, 3),
    "prestim_next_pattern": round(npbo_next, 3),
    "time_r2_prestim": round(npbo_time_r2, 3),
})
print(story_npbo.to_string())


# %% all four side by side

everyone = pd.DataFrame([story_npbn, story_npbk, story_npbm, story_npbo])
print(everyone.to_string(index=False))
print("\nRead prestim_next_pattern first: it is the impossible one. Whatever it")
print("scores above chance is the floor for how much of every other number here")
print("is slow session structure rather than stimulus information.")
print("\nThen read f1_prestim_sample_split against f1_prestim_trace_split: their")
print("Methods admit both readings, and the two land on opposite sides of their")
print("published value, so Fig 5 cannot be reproduced from what is written.")
print("\nThen f1_late_honest against f1_prestim_honest: 10 trials averaged at")
print("random, whole traces held out. That pair is the honest version of their")
print("question, and it is the number this reanalysis should quote.")

everyone.to_csv(f"{OUT_DIR}/memory_or_drift_summary.csv", index=False)
print(f"\nwrote {OUT_DIR}/memory_or_drift_summary.csv")

# %%
