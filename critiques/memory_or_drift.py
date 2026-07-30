# %% [ THE ARGUMENT ]
"""Critique D: is the "working memory-like" residual actually drift or carryover?

  THE CLAIM      Tactile pattern identity stays decodable out to 1000 ms, long
                 after the ~300 ms stimulus ends, so the population "never returns
                 to spontaneous baseline within 1 s" and holds resident
                 information -- a substrate for working memory. Their numbers:
                 pre-stimulus F1 = 0.303, then 0.455 (1-200 ms), 0.363, 0.337,
                 0.329, 0.318 (801-1000 ms).

  THE WEAKNESS   Two dull explanations are left open, and either would produce the
                 same result.

                 CARRYOVER. Stimuli are ~1.8 s apart and the claimed residual
                 lasts 1 s, so the previous trial's tail is still inside the next
                 trial's pre-stimulus window. Note their own pre-stimulus F1 of
                 0.303 already sits above the 0.25 chance level for four classes,
                 which is what carryover looks like.

                 DRIFT. Anesthesia depth, temperature and probe settling all move
                 slowly across a session. Their step of averaging 50 SEQUENTIAL
                 trials welds every data point to one chunk of session time, so a
                 slow drift becomes a label-correlated signal.

  THE TESTS      1. Decode the PREVIOUS trial's pattern from the current
                    pre-stimulus window. Above chance = carryover exists.
                 2. Predict elapsed session time from the population state. High
                    R2 = strong drift.
                 3. Ask whether the late "pattern identity" axis IS the drift
                    axis, by comparing their directions.
                 4. Project the drift axis out of the population state and re-run
                    the late-window decoding. If it collapses, the residual was
                    drift.

  WHY NPBI IS    Its onsets are about 243 ms late, so every window here would be
  NOT HERE       measuring the wrong slice of time.

Steps 1-8 walk through NPBN. Then NPBK, NPBM and NPBO each get their own cell.
"""

# %% [ STEP 0 ] settings

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, r2_score
from pynwb import NWBHDF5IO

# The windows we score, in milliseconds relative to onset. The first is the
# paper's spontaneous baseline; the rest match their late-window analysis.
WINDOWS = {
    "pre -300..0": (-300, 0),
    "0..200": (0, 200),
    "200..400": (200, 400),
    "400..600": (400, 600),
    "600..800": (600, 800),
    "800..1000": (800, 1000),
}

N_FOLDS = 5
OUT_DIR = "figures/critique"
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(0)


# %% [ STEP 0b ] the operations


def load_window_rates(exp):
    """Firing rate per neuron per trial, in each window of WINDOWS.

    Returns (rates, labels, onsets) where rates is a dict of window name ->
    (trial, neuron) array in Hz. Trials come back in session order, which the
    carryover and drift tests both need.
    """
    with NWBHDF5IO(f"nwb/{exp}.nwb", "r") as io:
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

    rates = {}
    for name, (start_ms, stop_ms) in WINDOWS.items():
        start_s = start_ms / 1000.0
        stop_s = stop_ms / 1000.0
        duration = stop_s - start_s

        counts = np.zeros((len(onsets), len(good_units)), dtype=np.float32)
        for neuron, spikes in enumerate(good_units["spike_times"]):
            spikes = np.sort(np.asarray(spikes, dtype=np.float64))
            first = np.searchsorted(spikes, onsets + start_s)
            last = np.searchsorted(spikes, onsets + stop_s)
            counts[:, neuron] = (last - first) / duration
        rates[name] = counts

    return rates, labels, onsets


def decode_labels(x, y):
    """Balanced accuracy for predicting y from x, held out by trial."""
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


# %% [ STEP 1 ] NPBN: load it

npbn_rates, npbn_labels, npbn_onsets = load_window_rates("NPBN")
npbn_classes = sorted(set(npbn_labels))
chance = 1.0 / len(npbn_classes)

n_trials, n_neurons = npbn_rates["pre -300..0"].shape
session_minutes = (npbn_onsets.max() - npbn_onsets.min()) / 60

print(f"NPBN: {n_trials} trials x {n_neurons} neurons")
print(f"{len(npbn_classes)} classes -> chance = {chance:.3f} balanced accuracy")
print(f"session spans {session_minutes:.1f} minutes")
print(f"median gap between stimuli: {np.median(np.diff(npbn_onsets)):.2f} s")
print("\nThat gap matters: the claimed residual lasts 1000 ms, so consecutive")
print("trials are not independent samples of a resting brain.")


# %% [ STEP 2 ] NPBN: reproduce the decline they read as memory

decline_rows = []
for name in WINDOWS:
    x = npbn_rates[name]
    accuracy = decode_labels(x, npbn_labels)
    decline_rows.append({"window": name, "accuracy": accuracy,
                         "above_chance": accuracy - chance})

npbn_decline = pd.DataFrame(decline_rows)
print(npbn_decline.round(3).to_string(index=False))
print(f"\nchance is {chance:.3f}. The paper's shape is a jump just after onset")
print("then a slow decay that never reaches the pre-stimulus level.")


# %% [ STEP 3 ] NPBN: TEST 1 -- carryover. Decode the PREVIOUS trial's pattern.
#
# If the pre-stimulus window still knows what the LAST stimulus was, then
# "information surviving past the stimulus" is just the previous trial bleeding
# in, and it needs no memory mechanism to explain.

previous_labels = np.roll(npbn_labels, 1)          # session order, shifted by one
x_pre = npbn_rates["pre -300..0"][1:]              # drop the first trial
y_previous = previous_labels[1:]
y_current = npbn_labels[1:]

carryover_accuracy = decode_labels(x_pre, y_previous)
current_from_pre = decode_labels(x_pre, y_current)

print(f"from the pre-stimulus window, decoding the PREVIOUS trial's pattern: "
      f"{carryover_accuracy:.3f}")
print(f"from the same window, decoding the trial about to happen           : "
      f"{current_from_pre:.3f}")
print(f"chance                                                            : "
      f"{chance:.3f}")
print("\nThe second line is the sanity check: the upcoming stimulus cannot be")
print("known in advance, so anything above chance there is drift or leakage, not")
print("prediction. If both lines are above chance by a similar amount, the")
print("pre-stimulus window is carrying slow session structure, not memory.")


# %% [ STEP 4 ] NPBN: TEST 2 -- how much drift is there?
#
# Predict elapsed session time from the population state. If the state says what
# time it is, then anything correlated with time is decodable from it -- including
# stimulus labels, if the design presented them in a time-correlated way.

drift_rows = []
for name in WINDOWS:
    x = npbn_rates[name]
    r2 = predict_time(x, npbn_onsets)
    drift_rows.append({"window": name, "time_r2": r2})

npbn_drift = pd.DataFrame(drift_rows)
print(npbn_drift.round(3).to_string(index=False))
print("\nR2 near 0 means no drift. R2 near 1 means the population state is")
print("essentially a clock, and every slow analysis on it is suspect.")


# %% [ STEP 5 ] NPBN: TEST 3 -- is the late 'memory' axis the drift axis?
#
# Take the direction along which session time increases, and the directions the
# classifier uses to tell patterns apart in the last window. If they are the same
# direction, the late decoding is reading the clock.

late = npbn_rates["800..1000"]
time_direction = drift_axis(late, npbn_onsets)

discriminator = LinearDiscriminantAnalysis()
discriminator.fit(late, npbn_labels)
pattern_directions = discriminator.coef_        # one row per class

cosines = []
for row in range(pattern_directions.shape[0]):
    axis = pattern_directions[row]
    axis = axis / np.linalg.norm(axis)
    cosines.append(abs(float(axis @ time_direction)))

print(f"|cosine| between the drift axis and each pattern-discriminant axis:")
for cls, value in zip(npbn_classes, cosines):
    print(f"   {cls:24} {value:.3f}")
print(f"mean {np.mean(cosines):.3f}")
print("\n0 means the two are unrelated directions. Values well above 0 mean the")
print("classifier is partly using the drift direction to tell patterns apart.")


# %% [ STEP 6 ] NPBN: TEST 4 -- take the drift out and decode again
#
# The decisive one. Remove the drift direction from every trial's population
# vector, then repeat the late-window decoding. What remains is pattern
# information that is not explainable as slow session change.

late_without_drift = remove_axis(late, time_direction)

with_drift = decode_labels(late, npbn_labels)
without_drift = decode_labels(late_without_drift, npbn_labels)

print(f"late window (800-1000 ms), pattern decoding")
print(f"   as is                        : {with_drift:.3f}")
print(f"   with the drift axis removed  : {without_drift:.3f}")
print(f"   chance                       : {chance:.3f}")
print(f"   lost to drift removal        : {with_drift - without_drift:+.3f}")
print("\nIf removing ONE direction out of {0} takes the accuracy to chance, the".format(n_neurons))
print("'working memory-like' residual was a drifting session, not held information.")


# %% [ STEP 7 ] NPBN: control -- shuffle the trial order and redo the drift test
#
# Sanity check on the machinery: if trials are randomly reordered in time, the
# drift test must fail. If it still 'works', the test is measuring something else.

shuffled_onsets = rng.permutation(npbn_onsets)
shuffled_r2 = predict_time(npbn_rates["pre -300..0"], shuffled_onsets)
real_r2 = predict_time(npbn_rates["pre -300..0"], npbn_onsets)

print(f"time R2 from the pre-stimulus window, real order     : {real_r2:.3f}")
print(f"time R2 with the onsets randomly permuted            : {shuffled_r2:.3f}")
print("\nThe second should be near or below 0. If it is not, ignore step 4.")


# %% [ STEP 8 ] NPBN: the figure


def plot_memory_or_drift(decline, drift, chance_level, with_drift, without_drift,
                         carryover, current_from_pre, name, path):
    """Left: the decline they call memory. Middle: drift. Right: the two tests."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    ax = axes[0]
    positions = np.arange(len(decline))
    ax.plot(positions, decline.accuracy, "o-", color="#2c3e50", lw=1.6, ms=5)
    ax.axhline(chance_level, color="#7b8794", lw=1, ls="--")
    ax.text(len(decline) - 0.5, chance_level, " chance", fontsize=8,
            color="#7b8794", va="center")
    ax.set_xticks(positions)
    ax.set_xticklabels(decline.window, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("pattern decoding (balanced accuracy)")
    ax.set_title(f"{name}: the decline they read as memory", loc="left", fontsize=10)

    ax = axes[1]
    ax.plot(positions, drift.time_r2, "s-", color="#c1703a", lw=1.6, ms=5)
    ax.axhline(0, color="#7b8794", lw=1, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(drift.window, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("R² predicting session time")
    ax.set_title("how much the state is a clock", loc="left", fontsize=10)

    ax = axes[2]
    names = ["late window\nas is", "late window\ndrift removed",
             "pre-stim ->\nPREVIOUS pattern", "pre-stim ->\nnext pattern"]
    values = [with_drift, without_drift, carryover, current_from_pre]
    colours = ["#2c3e50", "#2c3e50", "#c0392b", "#c0392b"]
    ax.bar(names, values, color=colours, width=0.6)
    ax.axhline(chance_level, color="#7b8794", lw=1, ls="--")
    for position, value in enumerate(values):
        ax.text(position, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
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
    npbn_decline, npbn_drift, chance, with_drift, without_drift,
    carryover_accuracy, current_from_pre, "NPBN",
    f"{OUT_DIR}/memory_or_drift_NPBN.png")
print(f"wrote {figure_path}")


# %% [ THE SAME TESTS, ONE CELL PER SESSION ]


def one_session_story(exp):
    """Tests 1 to 4 for one session, as a one-row summary."""
    rates, labels, onsets = load_window_rates(exp)
    classes = sorted(set(labels))
    chance_level = 1.0 / len(classes)

    pre = rates["pre -300..0"]
    late_window = rates["800..1000"]

    previous = np.roll(labels, 1)
    carryover = decode_labels(pre[1:], previous[1:])
    next_pattern = decode_labels(pre[1:], labels[1:])

    time_r2 = predict_time(pre, onsets)
    time_direction_here = drift_axis(late_window, onsets)
    cleaned = remove_axis(late_window, time_direction_here)

    return pd.Series({
        "session": exp,
        "neurons": pre.shape[1],
        "classes": len(classes),
        "chance": round(chance_level, 3),
        "late_as_is": round(decode_labels(late_window, labels), 3),
        "late_drift_removed": round(decode_labels(cleaned, labels), 3),
        "prestim_previous_pattern": round(carryover, 3),
        "prestim_next_pattern": round(next_pattern, 3),
        "time_r2_prestim": round(time_r2, 3),
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
print("\nRead prestim_next_pattern first: it is the impossible one. Whatever it")
print("scores above chance is the floor for how much of every other number here")
print("is slow session structure rather than stimulus information.")

# %%
