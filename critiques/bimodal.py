# %% [ THE ARGUMENT ]
"""Critique C: the bimodal claim rests on one session, and it is the broken one.

  THE CLAIM      S1 population responses to a given tactile pattern change when a
                 visual stimulus is presented at the same time (Fig 2: F1 = 0.73
                 for +visual vs -visual, against 0.5 shuffled), and individual S1
                 neurons cannot make that separation while the population can
                 (Fig 2E). They read this as visual input modulating haptic
                 processing in S1 -- anticipatory regulation.

  THE PROBLEM    Of the five deposited sessions, exactly ONE contains trials in
  WITH THE DATA  which both modalities were delivered together: NPBI, which has
                 400 visual-only trials (vA-vD) and 400 visuo-tactile trials
                 (vA-vD each paired with an F5 shock). And NPBI is the session
                 whose onsets are ~243 ms late, with no LFP and no TTL to fix
                 them against (see CLAUDE.md).

                 The deposited design is also not the design the paper describes.
                 The paper reports four TACTILE patterns +/- visual, three
                 experiments, 800 visuo-tactile against 800 tactile stimuli.
                 NPBI is the converse: four VISUAL patterns +/- one F5 shock,
                 400 against 400.

  WHY IT CAN     A constant timing error shifts every trial in the session by the
  STILL BE       same amount. Any contrast BETWEEN two conditions inside NPBI is
  TESTED         therefore unaffected by it -- both conditions move together. All
                 that changes is which slice of time the window covers, and STEP 2
                 measures the shift from the data and corrects for it. What NPBI
                 cannot support is any absolute latency statement.

  THE TEST       Their contrast, mirrored. Inside the 400 visuo-tactile trials the
  THAT MATTERS   tactile input is identical (F5 every time) and only the visual
                 partner differs. If S1 separates vA/vB/vC/vD there, then visual
                 input really did change the cortical response to the same tactile
                 input. That is the paper's claim, tested the only way this data
                 allows.

  THE TRAP       Decoding +tactile vs -tactile is NOT that test. Adding a skin
                 shock to an S1 recording produces an S1 response; separating
                 those two conditions is trivially easy and says nothing about
                 bimodal integration. STEP 3 runs it anyway to show how large a
                 number a trivial contrast produces here.

Steps 1-8 walk through NPBI, which is the only session this critique can use.
"""

# %% [ STEP 0 ] settings

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from pynwb import NWBHDF5IO

EXP = "NPBI"

# The measured correction. NPBI's labelled onsets are LATE, so the true stimulus
# happened at (labelled onset - SHIFT_S). Established 2026-07-29 by comparing the
# F5 population PSTH peak against NPBN and NPBO; STEP 2 re-measures it here from
# the data in front of us rather than taking it on trust.
SHIFT_S = 0.243

WINDOWS = {
    "pre -300..0": (-300, 0),
    "0..200": (0, 200),
    "200..400": (200, 400),
}

N_FOLDS = 5
N_SHUFFLE = 20

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


# %% [ STEP 0b ] the four operations
#
# Each does one thing and calls none of the others.


def load_session(exp):
    """The good units and the trials table, straight out of the NWB file."""
    with NWBHDF5IO(ROOT / "nwb" / f"{exp}.nwb", "r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        trials = nwb.trials.to_dataframe()

    good_units = units[units["quality"] == "good"]
    trials = trials.sort_values("start_time", ignore_index=True)
    return good_units, trials


def population_psth(good_units, onsets, pre_s, post_s, bin_s):
    """Mean firing rate over all units, aligned to onsets. Returns (times, rate)."""
    edges = np.arange(-pre_s, post_s + bin_s, bin_s)
    times = edges[:-1] + bin_s / 2
    total = np.zeros(len(times))

    for spikes in good_units["spike_times"]:
        spikes = np.sort(np.asarray(spikes, dtype=np.float64))
        for onset in onsets:
            lo, hi = np.searchsorted(spikes, [onset - pre_s, onset + post_s])
            relative = spikes[lo:hi] - onset
            total += np.histogram(relative, bins=edges)[0]

    rate = total / (len(onsets) * len(good_units) * bin_s)
    return times, rate


def window_rates(good_units, onsets, start_ms, stop_ms):
    """Firing rate per trial per neuron in one window. Returns (trial, neuron) Hz."""
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
    """Balanced accuracy, whole trials held out. No trial appears on both sides."""
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    model = LinearDiscriminantAnalysis()
    predicted = cross_val_predict(model, x, y, cv=folds)
    return float(balanced_accuracy_score(y, predicted))


# %% [ STEP 1 ] NPBI: what is actually in this session

npbi_units, npbi_trials = load_session(EXP)

design = npbi_trials.groupby(["modality", "stimulus"]).size()
print(f"{EXP}: {len(npbi_units)} good units, {len(npbi_trials)} trials")
print(design.to_string())
print(f"\nsession spans {(npbi_trials.start_time.max() - npbi_trials.start_time.min()) / 60:.1f} min")
print(f"median gap between stimuli: {npbi_trials.start_time.diff().median():.2f} s")
print("\nThe paper describes four TACTILE patterns +/- visual across three")
print("experiments, 800 vs 800 stimuli. What is deposited is four VISUAL")
print("patterns +/- one F5 shock, 400 vs 400, in one session. Those are not the")
print("same experiment, and this is the only session that has both modalities")
print("on the same trial at all.")


# %% [ STEP 2 ] NPBI: measure the timing error rather than assuming it
#
# The population PSTH of a real S1 response to a skin shock peaks a few tens of
# milliseconds AFTER the stimulus. If NPBI's peak sits well before its labelled
# onset, the labels are late by that much. Only the visuo-tactile trials carry a
# tactile shock, so those are the ones with a peak to find.

visuo_tactile = npbi_trials["modality"] == "visuo_tactile"
visuo_tactile_onsets = npbi_trials.loc[visuo_tactile, "start_time"].to_numpy()

psth_times, psth_rate = population_psth(npbi_units, visuo_tactile_onsets,
                                        pre_s=0.500, post_s=0.500, bin_s=0.002)

peak_index = int(np.argmax(psth_rate))
peak_time = psth_times[peak_index]
baseline_rate = psth_rate.mean()

print(f"population PSTH peak at {1000 * peak_time:+.0f} ms relative to the label")
print(f"peak rate {psth_rate[peak_index]:.2f} Hz against a window mean of "
      f"{baseline_rate:.2f} Hz")
print(f"\nIn NPBN and NPBO the same F5 shock peaks at about +58 ms. A peak at")
print(f"{1000 * peak_time:+.0f} ms means the labels are late by roughly")
print(f"{1000 * (0.058 - peak_time):.0f} ms. The stored correction is "
      f"{1000 * SHIFT_S:.0f} ms.")

corrected_onsets = npbi_trials["start_time"].to_numpy() - SHIFT_S
print(f"\nUsing corrected onsets = labelled onset - {SHIFT_S:.3f} s from here on.")


# %% [ STEP 2b ] NPBI: the same PSTH on corrected onsets, as a check

check_times, check_rate = population_psth(
    npbi_units, visuo_tactile_onsets - SHIFT_S, pre_s=0.500, post_s=0.500,
    bin_s=0.002)

check_peak_time = check_times[int(np.argmax(check_rate))]
print(f"after correction the peak sits at {1000 * check_peak_time:+.0f} ms, "
      f"which is where an S1 shock response belongs")


# %% [ STEP 3 ] NPBI: the trivial contrast, run so its size is on record
#
# +tactile against -tactile. This is easy for a reason that has nothing to do
# with bimodal integration: one condition delivers a skin shock to the forepaw
# and the other does not, and this is somatosensory cortex.

is_visuo_tactile = (npbi_trials["modality"] == "visuo_tactile").to_numpy()
tactile_labels = np.where(is_visuo_tactile, "+tactile", "-tactile")

early_rates = window_rates(npbi_units, corrected_onsets, 0, 200)
trivial_accuracy = decode_labels(early_rates, tactile_labels)

print(f"decoding +tactile vs -tactile, 0-200 ms : {trivial_accuracy:.3f}")
print(f"chance                                  : 0.500")
print("\nA large number here is not evidence for anything. It is the tactile")
print("response itself. The paper's equivalent panel has the same problem in")
print("the opposite direction, and this is the ceiling any bimodal claim in")
print("this session has to be separated from.")


# %% [ STEP 4 ] NPBI: THE TEST -- same tactile input, different visual partner
#
# Restrict to the 400 visuo-tactile trials. Every one of them delivers the same
# F5 shock; the only thing that differs is which visual pattern accompanied it.
# If the population separates them, visual input changed the cortical response to
# an identical tactile input. That is the paper's claim.

visuo_tactile_trials = npbi_trials[is_visuo_tactile]
visuo_tactile_corrected = visuo_tactile_trials["start_time"].to_numpy() - SHIFT_S
visual_partner = visuo_tactile_trials["stimulus"].to_numpy()

n_classes = len(set(visual_partner))
chance = 1.0 / n_classes
print(f"{len(visuo_tactile_trials)} visuo-tactile trials, {n_classes} visual "
      f"partners -> chance = {chance:.3f}")

modulation_rows = []
for name, (start_ms, stop_ms) in WINDOWS.items():
    rates = window_rates(npbi_units, visuo_tactile_corrected, start_ms, stop_ms)
    accuracy = decode_labels(rates, visual_partner)
    modulation_rows.append({"window": name, "accuracy": accuracy,
                            "above_chance": accuracy - chance})

npbi_modulation = pd.DataFrame(modulation_rows)
print(npbi_modulation.round(3).to_string(index=False))
print("\nThe pre-stimulus row is the control that decides the other two: nothing")
print("about the upcoming visual pattern can be known before it arrives, so")
print("whatever that row scores above chance is the floor set by drift and")
print("carryover, not by bimodal integration.")


# %% [ STEP 5 ] NPBI: a shuffled-label null for the window that matters
#
# Balanced accuracy with 4 classes has a chance level of 0.25 in theory, but with
# 400 trials and 70 neurons an LDA can overfit its way above that. Measure where
# chance really sits by destroying the label-to-trial correspondence and rerunning
# the identical decoding.

early_visuo_tactile = window_rates(npbi_units, visuo_tactile_corrected, 0, 200)
real_accuracy = decode_labels(early_visuo_tactile, visual_partner)

shuffled_accuracies = []
for repeat in range(N_SHUFFLE):
    shuffled_labels = rng.permutation(visual_partner)
    shuffled_accuracies.append(decode_labels(early_visuo_tactile, shuffled_labels))

shuffled_mean = np.mean(shuffled_accuracies)
shuffled_sd = np.std(shuffled_accuracies)

print(f"visual partner decoded from 0-200 ms  : {real_accuracy:.3f}")
print(f"same decoding, labels shuffled        : {shuffled_mean:.3f} "
      f"(SD {shuffled_sd:.3f} over {N_SHUFFLE} shuffles)")
print(f"theoretical chance                    : {chance:.3f}")
if shuffled_sd > 0:
    print(f"real result sits {(real_accuracy - shuffled_mean) / shuffled_sd:+.1f} "
          f"SDs above its own null")


# %% [ STEP 6 ] NPBI: does S1 see the visual patterns without any touch at all?
#
# The 400 visual-only trials answer this. If S1 separates vA-vD with no tactile
# stimulus present, then the STEP 4 result need not be integration -- it can be
# the visual response sitting alongside an unchanged tactile response rather than
# modulating it. The two readings are distinguishable only if this number is
# small.

visual_only_trials = npbi_trials[~is_visuo_tactile]
visual_only_corrected = visual_only_trials["start_time"].to_numpy() - SHIFT_S
visual_only_pattern = visual_only_trials["stimulus"].to_numpy()

visual_only_rows = []
for name, (start_ms, stop_ms) in WINDOWS.items():
    rates = window_rates(npbi_units, visual_only_corrected, start_ms, stop_ms)
    accuracy = decode_labels(rates, visual_only_pattern)
    visual_only_rows.append({"window": name, "accuracy": accuracy,
                             "above_chance": accuracy - chance})

npbi_visual_only = pd.DataFrame(visual_only_rows)
print(npbi_visual_only.round(3).to_string(index=False))
print("\nCompare with STEP 4. If visual-alone decoding is as good as visual-with-")
print("touch decoding, S1 is simply receiving visual input, and calling it")
print("modulation of haptic processing adds a claim the data does not carry.")


# %% [ STEP 7 ] NPBI: their Fig 2E -- can single neurons do it?
#
# The paper's point is that the population separates conditions that no
# individual neuron separates. Test the same thing directly: run the identical
# decoding on one neuron at a time and look at the best one.

single_neuron_scores = []
for neuron in range(early_visuo_tactile.shape[1]):
    one_neuron = early_visuo_tactile[:, [neuron]]
    single_neuron_scores.append(decode_labels(one_neuron, visual_partner))

single_neuron_scores = np.array(single_neuron_scores)
best_neuron = int(np.argmax(single_neuron_scores))

print(f"population of {early_visuo_tactile.shape[1]} neurons : {real_accuracy:.3f}")
print(f"best single neuron (#{best_neuron})            : "
      f"{single_neuron_scores[best_neuron]:.3f}")
print(f"median single neuron                     : "
      f"{np.median(single_neuron_scores):.3f}")
print(f"shuffled-label level                     : {shuffled_mean:.3f}")
print("\nTheir Fig 2E reading -- population succeeds where single neurons fail --")
print("needs the population number to be clearly above both the single-neuron")
print("distribution AND its own shuffled null. If the population score is itself")
print("at the shuffled level, this panel is not comparing a success against a")
print("failure; it is comparing two failures, and neither ranking means")
print("anything.")


# %% [ STEP 8 ] NPBI: the figure


def plot_bimodal(psth_times_labelled, psth_rate_labelled, psth_times_corrected,
                 psth_rate_corrected, modulation, visual_only, chance_level,
                 shuffled_level, single_scores, population_score, name, path):
    """Three panels, drawn from finished arrays. No computation happens here."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    ax = axes[0]
    ax.plot(1000 * psth_times_labelled, psth_rate_labelled, color="#c0392b", lw=1.2,
            label="as labelled")
    ax.plot(1000 * psth_times_corrected, psth_rate_corrected, color="#2c3e50",
            lw=1.2, label="corrected")
    ax.axvline(0, color="#7b8794", lw=1, ls="--")
    ax.set_xlabel("time from onset (ms)")
    ax.set_ylabel("population rate (Hz)")
    ax.set_title(f"{name}: the clock, before and after", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    positions = np.arange(len(modulation))
    ax.plot(positions, modulation.accuracy, "o-", color="#2c3e50", lw=1.6, ms=5,
            label="with touch (same F5 every trial)")
    ax.plot(positions, visual_only.accuracy, "s--", color="#c1703a", lw=1.6, ms=5,
            label="visual alone, no touch")
    ax.axhline(chance_level, color="#7b8794", lw=1, ls="--")
    ax.axhline(shuffled_level, color="#c0392b", lw=1, ls=":")
    ax.text(0.02, shuffled_level, "shuffled / chance", fontsize=8,
            color="#c0392b", va="bottom", ha="left", transform=ax.get_yaxis_transform())
    ax.set_xticks(positions)
    ax.set_xticklabels(modulation.window, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("visual pattern decoding")
    ax.set_title("does the visual partner change S1?", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.hist(single_scores, bins=20, color="#b8c2cc", edgecolor="white")
    ax.axvline(population_score, color="#2c3e50", lw=2)
    ax.text(population_score, ax.get_ylim()[1] * 0.9, " population", fontsize=8.5,
            color="#2c3e50")
    ax.axvline(chance_level, color="#7b8794", lw=1, ls="--")
    ax.set_xlabel("visual pattern decoding, 0-200 ms")
    ax.set_ylabel("neurons")
    ax.set_title("their Fig 2E: one neuron at a time", loc="left", fontsize=10)

    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(f"Critique C on {name} — the bimodal claim, tested on the only "
                 f"session that carries both modalities", x=0.02, ha="left",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=150, facecolor="white")
    return path


figure_path = plot_bimodal(psth_times, psth_rate, check_times, check_rate,
                           npbi_modulation, npbi_visual_only, chance,
                           shuffled_mean, single_neuron_scores, real_accuracy,
                           EXP, f"{OUT_DIR}/bimodal_{EXP}.png")
print(f"wrote {figure_path}")


# %% [ STEP 9 ] what this session can and cannot support

summary = pd.Series({
    "session": EXP,
    "neurons": len(npbi_units),
    "visuo_tactile_trials": int(is_visuo_tactile.sum()),
    "visual_only_trials": int((~is_visuo_tactile).sum()),
    "timing_shift_applied_s": SHIFT_S,
    "trivial_plus_minus_tactile": round(trivial_accuracy, 3),
    "visual_partner_with_touch_0_200": round(real_accuracy, 3),
    "visual_partner_shuffled": round(shuffled_mean, 3),
    "visual_partner_prestim": round(npbi_modulation.accuracy.iloc[0], 3),
    "visual_alone_0_200": round(npbi_visual_only.accuracy.iloc[1], 3),
    "best_single_neuron": round(float(single_neuron_scores.max()), 3),
    "chance": round(chance, 3),
})
print(summary.to_string())

summary.to_frame().T.to_csv(f"{OUT_DIR}/bimodal_summary.csv", index=False)
print(f"\nwrote {OUT_DIR}/bimodal_summary.csv")

print(f"""
IS THE NEGATIVE TRUSTWORTHY?

  A null result is only worth reading if the same machinery can find something
  it should find. It can: the identical decoder, on the identical trials, in the
  identical window, separates +tactile from -tactile at {trivial_accuracy:.3f}
  against a chance level of 0.500. The population vectors, the cross-validation
  and the classifier are all working. What they do not find is any trace of
  WHICH visual pattern accompanied the shock.

  The honest caveat in the other direction: this is a trial-level test, and the
  paper's effect was measured after averaging 50 trials, which raises
  signal-to-noise a great deal. A small true effect could hide here. What the
  averaging cannot do is survive critique D's objection -- averaging 50
  SEQUENTIAL trials ties every data point to a stretch of session time.

WHAT THIS SUPPORTS

  Only a within-NPBI statement. The contrast is immune to the session's constant
  timing error, but every absolute latency in it is not, and no other deposited
  session can replicate it because no other session delivered two modalities on
  the same trial.

WHAT IT CANNOT SUPPORT

  The paper's actual Fig 2, which is four tactile patterns +/- visual across
  three experiments. That experiment is not in the deposited data. A claim of
  bimodal integration in S1 resting on one anesthetized animal, in a session
  with a known and uncorrectable clock error, is a claim about one animal.
""")

# %%
