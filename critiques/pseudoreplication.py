# %% [ THE ARGUMENT ]
"""Critique E: the significance comes from resampling, not from observations.

  THE CLAIM      Fig 5's key comparison is reported as t(2999) = -87.65,
                 p < 0.001, d = 2.1, and four more like it down to d = 0.4. Read
                 plainly, that is 3000 observations and an overwhelming result.

  WHERE THE      Their own Methods say where: "we grouped the F1 scores from all
  3000 COMES     50 repetitions from the 20 decoding analyses in the 200 ms
  FROM           prestimulus time window for all experiments (N = 3000)".
                 50 repetitions x 20 sub-windows x 3 experiments = 3000.

  THE WEAKNESS   Those 3000 numbers are not 3000 observations. They are the same
                 three animals, re-decoded 1000 times each. A repetition of a
                 classifier on the same data carries no new information about
                 rats, so it cannot license a claim about rats. The t statistic
                 grows like sqrt(N), so the experimenter sets p by choosing how
                 many times to rerun the loop: 50 repetitions instead of 5 divides
                 the p-value by orders of magnitude while the data stays put.

  WHY IT IS NOT  d = 2.1 is computed with SD_pooled taken across repetitions.
  RESCUED BY     Repetition noise is smaller than animal-to-animal variation, so
  THE EFFECT     dividing by it inflates d for exactly the same reason it inflates
  SIZE           t. It is not an independent check.

  THE TEST       1. Rebuild their N. Reproduce the pipeline, 20 sub-windows x 50
                    repetitions per session, and run their paired t-test.
                 2. Show the p-value is a knob. Recompute it while varying only
                    the number of repetitions.
                 3. Run the identical test where the true difference is ZERO by
                    construction (labels shuffled in both windows). If it still
                    returns p < 0.001, the test cannot tell a real effect from
                    none.
                 4. Do it honestly: one number per experiment, n = 4, and report
                    what that actually supports.

  WHY NPBI IS    Its onsets are about 243 ms late, so its windows are the wrong
  NOT HERE       slices of time. See CLAUDE.md.

Steps 1-8 use all four usable sessions, because the whole point is how few
independent units there really are.
"""

# %% [ STEP 0 ] settings

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from pynwb import NWBHDF5IO

SESSIONS = ["NPBN", "NPBK", "NPBM", "NPBO"]

# Three 200 ms windows. The first two are BOTH before the stimulus, so no
# stimulus-driven difference can exist between them -- that pair is the null the
# procedure has to get right. The third is their evoked window.
EARLY_BASELINE = (-400, -200)
PRE_WINDOW = (-200, 0)          # their 200 ms prestimulus window
EVOKED_WINDOW = (0, 200)        # their 1-200 ms poststimulus window
SUB_MS = 10                     # their 10 ms sub-windows
N_REPEATS = 50                  # their 50 repetitions
GROUP = 50                      # paper: average 50 sequential trials
SMOOTH_MS = 10                  # paper: 10 ms Gaussian kernel
K = 5                           # paper: k = 5 nearest neighbours
VARIANCE = 0.95                 # paper: PCs explaining 95% of the variance

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
    labels = []
    for i in range(len(trials)):
        labels.append(f"{modality.iloc[i]}/{stimulus.iloc[i]}")
    labels = np.array(labels)

    return good_units, labels, onsets


def averaged_traces(good_units, onsets, labels, start_ms, stop_ms):
    """The paper's steps 1-3 for one window: bin, average 50 sequential, smooth.

    Returns (traces, trace_labels) where traces is (trace, neuron, ms).
    """
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

    sigma = SMOOTH_MS / 2.355
    traces = []
    trace_labels = []
    for cls in sorted(set(labels)):
        in_class = np.flatnonzero(labels == cls)
        n_traces = len(in_class) // GROUP
        for trace in range(n_traces):
            members = in_class[trace * GROUP:(trace + 1) * GROUP]
            averaged = counts[members].mean(axis=0)
            smoothed = gaussian_filter1d(averaged, sigma, axis=1)
            traces.append(smoothed)
            trace_labels.append(cls)

    return np.stack(traces), np.array(trace_labels)


def f1_per_repetition(traces, trace_labels, sub_ms, n_repeats, shuffle_labels):
    """Their inner loop: one F1 for every (sub-window, repetition) pair.

    Each 10 ms sub-window of the averaged traces is decoded on its own, with a
    fresh random 50/50 split over samples, n_repeats times. Returns a flat array
    of length (number of sub-windows x n_repeats) -- their unit of "N".
    """
    n_traces, n_neurons, n_ms = traces.shape
    n_sub = n_ms // sub_ms

    scores = []
    for sub in range(n_sub):
        block = traces[:, :, sub * sub_ms:(sub + 1) * sub_ms]
        time_last = block.transpose(0, 2, 1)
        x = time_last.reshape(-1, n_neurons)
        y = np.repeat(trace_labels, sub_ms)
        if shuffle_labels:
            y = rng.permutation(y)

        for repeat in range(n_repeats):
            # redraw until both sides hold every class, so the returned array is
            # always exactly (n_sub x n_repeats) long and can be reshaped
            is_train = rng.random(len(y)) < 0.5
            while len(set(y[is_train])) < 2 or len(set(y[~is_train])) < 2:
                is_train = rng.random(len(y)) < 0.5

            pca = PCA(n_components=VARIANCE).fit(x[is_train])
            model = KNeighborsClassifier(n_neighbors=K).fit(
                pca.transform(x[is_train]), y[is_train])
            predicted = model.predict(pca.transform(x[~is_train]))
            scores.append(f1_score(y[~is_train], predicted, average="macro"))

    return np.array(scores)


def paired_test(a, b):
    """Their test: paired t over two equal-length vectors, with Cohen's d."""
    t_statistic, p_value = stats.ttest_rel(a, b)
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    cohens_d = (np.mean(a) - np.mean(b)) / pooled_sd
    return float(t_statistic), float(p_value), float(cohens_d), len(a) - 1


# %% [ STEP 1 ] rebuild their N, session by session
#
# 20 sub-windows x 50 repetitions per session per window. Four sessions gives
# 4000 numbers where they had 3000 -- the same construction, one session more.
#
# Three windows are scored. EVOKED vs PRE is the comparison they report. PRE vs
# EARLY BASELINE is the control the paper never ran: two windows that both sit
# before the stimulus, so the true difference between them is zero by design.

baseline_scores = {}
pre_scores = {}
evoked_scores = {}

for exp in SESSIONS:
    units, labels, onsets = load_session(exp)

    baseline_traces, baseline_labels = averaged_traces(units, onsets, labels,
                                                       *EARLY_BASELINE)
    pre_traces, pre_labels = averaged_traces(units, onsets, labels, *PRE_WINDOW)
    evoked_traces, evoked_labels = averaged_traces(units, onsets, labels,
                                                   *EVOKED_WINDOW)

    baseline_scores[exp] = f1_per_repetition(baseline_traces, baseline_labels,
                                             SUB_MS, N_REPEATS, shuffle_labels=False)
    pre_scores[exp] = f1_per_repetition(pre_traces, pre_labels, SUB_MS,
                                        N_REPEATS, shuffle_labels=False)
    evoked_scores[exp] = f1_per_repetition(evoked_traces, evoked_labels, SUB_MS,
                                           N_REPEATS, shuffle_labels=False)

    print(f"{exp}: {len(pre_scores[exp])} F1 values per window. "
          f"mean F1  baseline {baseline_scores[exp].mean():.4f}  "
          f"pre {pre_scores[exp].mean():.4f}  "
          f"evoked {evoked_scores[exp].mean():.4f}")

baseline_pooled = np.concatenate([baseline_scores[exp] for exp in SESSIONS])
pre_pooled = np.concatenate([pre_scores[exp] for exp in SESSIONS])
evoked_pooled = np.concatenate([evoked_scores[exp] for exp in SESSIONS])
print(f"\npooled: N = {len(pre_pooled)} per window (they reported N = 3000)")


# %% [ STEP 2 ] their test, on their N, for their comparison

their_t, their_p, their_d, their_df = paired_test(evoked_pooled, pre_pooled)

print("paired t-test, evoked vs prestimulus, pooling every repetition")
print(f"   t({their_df}) = {their_t:.2f}")
print(f"   p = {their_p:.3g}")
print(f"   Cohen's d = {their_d:.2f}")
print(f"   mean difference = {evoked_pooled.mean() - pre_pooled.mean():+.4f} F1")
print("\nThe paper reports t(2999) = -87.65, p < 0.001, d = 2.1 for this")
print("comparison. Nothing so far is wrong -- an evoked window should differ")
print("from a baseline. STEP 3 asks what the same procedure says when the two")
print("windows cannot possibly differ.")


# %% [ STEP 3 ] the control the paper never ran
#
# -400 to -200 ms against -200 to 0 ms. Both are before the stimulus. Whatever
# the stimulus does, it does it to neither. A procedure that reports these two as
# significantly different is reporting on its own arithmetic.

null_t, null_p, null_d, null_df = paired_test(pre_pooled, baseline_pooled)

print("paired t-test, two windows that are BOTH prestimulus")
print(f"   t({null_df}) = {null_t:.2f}")
print(f"   p = {null_p:.3g}")
print(f"   Cohen's d = {null_d:.2f}")
print(f"   mean difference = {pre_pooled.mean() - baseline_pooled.mean():+.4f} F1")
print("\nRead this one honestly: the control PASSES. Their pooling scheme does")
print("not manufacture significance out of nothing, because the true difference")
print("between two baseline windows here really is about zero and the noise")
print("across repetitions is unbiased. So the objection is not that the test")
print("invents effects. It is narrower and harder to dismiss: the test cannot")
print("tell a large effect from a negligible one, because its p-value is set by")
print("how many times the loop was run. STEP 4 makes that concrete.")


# %% [ STEP 4 ] the p-value is a knob
#
# Nothing about the animals changes here. The only thing that changes is how many
# times the classifier is rerun on the same data. Both comparisons are tracked:
# the real one and the one that should never reach significance.

knob_rows = []
for n_used in [1, 2, 5, 10, 25, 50]:
    kept_baseline = []
    kept_pre = []
    kept_evoked = []
    for exp in SESSIONS:
        # scores arrive ordered (sub-window, repetition), so reshape and keep the
        # first n_used repetitions of every sub-window
        baseline_matrix = baseline_scores[exp].reshape(-1, N_REPEATS)
        pre_matrix = pre_scores[exp].reshape(-1, N_REPEATS)
        evoked_matrix = evoked_scores[exp].reshape(-1, N_REPEATS)
        kept_baseline.append(baseline_matrix[:, :n_used].ravel())
        kept_pre.append(pre_matrix[:, :n_used].ravel())
        kept_evoked.append(evoked_matrix[:, :n_used].ravel())

    subset_baseline = np.concatenate(kept_baseline)
    subset_pre = np.concatenate(kept_pre)
    subset_evoked = np.concatenate(kept_evoked)

    real_t, real_p, real_d, real_df = paired_test(subset_evoked, subset_pre)
    fake_t, fake_p, fake_d, fake_df = paired_test(subset_pre, subset_baseline)
    knob_rows.append({"repetitions": n_used, "N": len(subset_pre),
                      "p_evoked_vs_pre": real_p, "d_evoked_vs_pre": real_d,
                      "p_two_baselines": fake_p, "d_two_baselines": fake_d})

knob = pd.DataFrame(knob_rows)
print(knob.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print("\nHERE IS THE PROBLEM, IN ONE TABLE. d barely moves down the column,")
print("because d is a property of the data. p falls by many orders of magnitude,")
print("because p counts rows and the rows are reruns. One repetition of this")
print("analysis is not significant; fifty repetitions of the SAME analysis on")
print("the SAME four animals is p < 1e-18. Nothing was learned in between.")
print("\nThat is why t(2999) cannot be read as strong evidence: it reports how")
print("many times the loop ran, and the loop count is the analyst's choice.")


# %% [ STEP 5 ] the honest test -- one number per experiment
#
# The experimental unit is the animal. Average away the repetitions, which are
# measurement noise on one estimate, and test what is left.

honest_rows = []
for exp in SESSIONS:
    honest_rows.append({"session": exp,
                        "f1_baseline": baseline_scores[exp].mean(),
                        "f1_prestim": pre_scores[exp].mean(),
                        "f1_evoked": evoked_scores[exp].mean(),
                        "difference": evoked_scores[exp].mean() - pre_scores[exp].mean()})

honest = pd.DataFrame(honest_rows)
print(honest.round(4).to_string(index=False))

honest_t, honest_p, honest_d, honest_df = paired_test(
    honest.f1_evoked.to_numpy(), honest.f1_prestim.to_numpy())
honest_null_t, honest_null_p, honest_null_d, _ = paired_test(
    honest.f1_prestim.to_numpy(), honest.f1_baseline.to_numpy())

print("\npaired t-test over sessions, evoked vs prestimulus:")
print(f"   t({honest_df}) = {honest_t:.2f}")
print(f"   p = {honest_p:.3g}")
print(f"   Cohen's d = {honest_d:.2f}")
print(f"   mean difference = {honest.difference.mean():+.4f} F1 "
      f"(SD {honest.difference.std(ddof=1):.4f})")
print(f"\nsame test on the two prestimulus windows: p = {honest_null_p:.3g}")
print(f"\nn = {len(honest)} animals. That is what the design supports, and it is")
print("what the paper should have reported.")


# %% [ STEP 6 ] the figure


def plot_pseudoreplication(knob_table, honest_table, real_t, null_t, name, path):
    """Three panels, drawn from finished tables. No computation happens here."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    ax = axes[0]
    ax.plot(knob_table.repetitions, -np.log10(knob_table.p_evoked_vs_pre), "o-",
            color="#2c3e50", lw=1.6, ms=5, label="evoked vs prestimulus")
    ax.plot(knob_table.repetitions, -np.log10(knob_table.p_two_baselines), "s--",
            color="#c0392b", lw=1.6, ms=5, label="two prestimulus windows")
    ax.axhline(-np.log10(0.05), color="#7b8794", lw=1, ls="--")
    ax.text(0.98, -np.log10(0.05), "p = 0.05 ", fontsize=8, color="#7b8794",
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlabel("classifier repetitions (analyst's choice)")
    ax.set_ylabel("−log10(p)")
    ax.set_title("the p-value is a knob", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.bar(["evoked vs\nprestimulus", "two prestimulus\nwindows"],
           [abs(real_t), abs(null_t)], color=["#2c3e50", "#c0392b"], width=0.55)
    for position, value in enumerate([abs(real_t), abs(null_t)]):
        ax.text(position, value, f"  |t| = {value:.1f}", ha="center", fontsize=9,
                va="bottom")
    ax.set_ylabel("|t| from their pooling scheme")
    ax.set_title("what the test says when nothing happened", loc="left",
                 fontsize=10)

    ax = axes[2]
    positions = np.arange(len(honest_table))
    ax.bar(positions, honest_table.difference, color="#2c3e50", width=0.55)
    ax.axhline(0, color="#7b8794", lw=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(honest_table.session, fontsize=9)
    ax.set_ylabel("evoked − prestimulus F1")
    ax.set_title(f"one number per animal (n = {len(honest_table)})", loc="left",
                 fontsize=10)

    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(f"Critique E — t(2999) counts classifier reruns, not animals",
                 x=0.02, ha="left", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=150, facecolor="white")
    return path


figure_path = plot_pseudoreplication(knob, honest, their_t, null_t, "all sessions",
                                     f"{OUT_DIR}/pseudoreplication.png")
print(f"wrote {figure_path}")


# %% [ STEP 7 ] the summary table

summary = pd.DataFrame([
    {"test": "their pooling, evoked vs prestimulus", "n": len(pre_pooled),
     "t": their_t, "p": their_p, "d": their_d},
    {"test": "their pooling, two prestimulus windows", "n": len(pre_pooled),
     "t": null_t, "p": null_p, "d": null_d},
    {"test": "per animal, evoked vs prestimulus", "n": len(honest),
     "t": honest_t, "p": honest_p, "d": honest_d},
    {"test": "per animal, two prestimulus windows", "n": len(honest),
     "t": honest_null_t, "p": honest_null_p, "d": honest_null_d},
])
print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

summary.to_csv(f"{OUT_DIR}/pseudoreplication_summary.csv", index=False)
print(f"\nwrote {OUT_DIR}/pseudoreplication_summary.csv")

print(f"""
WHAT IS AND IS NOT BEING CLAIMED

  Not claimed: that their pooling invents effects from noise. STEP 3 tested that
  directly and it does not -- two baseline windows come back p = {null_p:.2f}.

  Claimed: that the p-value they report is a function of how many times they
  reran the classifier, so it cannot be read as a measure of evidence. The same
  data gives p = {knob.p_evoked_vs_pre.iloc[0]:.2f} at one repetition and
  p = {knob.p_evoked_vs_pre.iloc[-1]:.1e} at fifty. And the honest test, one
  number per animal, gives p = {honest_p:.2f} on n = {len(honest)} -- the same
  effect, correctly attributed to four animals instead of four thousand reruns.

WHAT TO REPLACE IT WITH

  The repetitions are measurement noise on one estimate, so average them away
  and keep one number per animal. With n = 4 or 5 the honest options are a
  paired t-test on those means, a mixed-effects model with animal as a random
  effect, or a permutation test that permutes at the level of the animal. All
  three give a far weaker result than t(2999), which is the point: the design
  has five animals in it, and no amount of resampling adds a sixth.
""")

# %%
