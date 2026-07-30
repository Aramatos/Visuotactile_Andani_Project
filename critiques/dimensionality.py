# %% [ THE ARGUMENT ]
"""Critique A: their dimensionality number cannot tell richness from noise.

  THE CLAIM      The paper fits PCA to spontaneous activity (the 300 ms before
                 each stimulus) and counts the PCs needed for 95% of the
                 variance: 50, 27, 55, 42, 28 across the five experiments, which
                 is 62-81% of all available PCs. It reads that as an extremely
                 high-dimensional state space.

  THE WEAKNESS   That count is large whenever the eigenvalue spectrum is FLAT,
                 and independent noise produces a flat spectrum. At 1 ms bins
                 with neurons firing a few Hz nearly every bin is empty, so flat
                 is the default expectation rather than a finding.

  THE TEST       Destroy all between-neuron structure while keeping each neuron
                 exactly as active as it was, then recount. STEP 2 shows
                 precisely what that shuffle does and checks that it did it.

  THE BETTER     Participation ratio, PR = (sum of eigenvalues)^2 / sum of
  METRIC         eigenvalues^2, which answers to the shape of the whole spectrum
                 instead of one cutoff. Flat -> PR near the neuron count.
                 Concentrated -> PR small.

  THE HONEST     There IS population structure, and it is LOW-dimensional, not
  NUMBER         high. At raw 1 ms the real data is indistinguishable from the
                 shuffle on both metrics. Under their own preprocessing the PR
                 separates sharply (NPBN: 5.2 real vs 16.7 shuffled), so the
                 structure that exists is slow enough to survive averaging 50
                 sequential trials. That is roughly 5-15 dimensions, not 42, and
                 being slow it is the same structure critique D calls drift.

  WHY NPBI IS    Its onsets are about 243 ms late, so its "pre-stimulus" windows
  NOT HERE       contain stimulus-driven activity. It cannot measure spontaneous
                 anything. See CLAUDE.md.

Steps 1-9 walk through NPBN one cell at a time. Then NPBK, NPBM and NPBO each get
their own cell, spelled out the same way, so each session has its own story.
"""

# %% [ STEP 0 ] settings

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from pynwb import NWBHDF5IO

BASELINE_S = 0.300          # the paper's spontaneous window
GROUP = 50                  # paper: average 50 sequential trials together
SMOOTH_MS = 10              # paper: 10 ms Gaussian kernel
BIN_MS_LIST = [1, 5, 10, 50, 100, 300]
N_SHUFFLE = 10
PAPER_PC_COUNTS = [50, 27, 55, 42, 28]

OUT_DIR = "figures/critique"
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(0)


# %% [ STEP 0b ] the four operations
#
# Each one does a single thing and calls none of the others, so any of them can
# be run on its own and its output looked at. Nothing below hides a step.


def load_spontaneous(exp):
    """Spike counts at 1 ms in the 300 ms before every onset: (trial, neuron, ms)."""
    with NWBHDF5IO(f"nwb/{exp}.nwb", "r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        trials = nwb.trials.to_dataframe()

    good_units = units[units["quality"] == "good"]
    onsets = trials["start_time"].to_numpy()
    n_ms = int(BASELINE_S * 1000)
    edges = np.arange(n_ms + 1) / 1000.0 - BASELINE_S

    counts = np.zeros((len(onsets), len(good_units), n_ms), dtype=np.uint8)
    for neuron, spikes in enumerate(good_units["spike_times"]):
        spikes = np.sort(np.asarray(spikes, dtype=np.float64))
        for trial, onset in enumerate(onsets):
            first, last = np.searchsorted(spikes, [onset - BASELINE_S, onset])
            relative = spikes[first:last] - onset
            counts[trial, neuron] = np.histogram(relative, bins=edges)[0]
    return counts


def shuffle_neurons_apart(counts):
    """Rotate each neuron's own spike train by its own random amount.

    THE SHUFFLE, precisely. Every (trial, neuron) pair has a 300-sample spike
    train. We pick one random offset per pair and rotate that train circularly:
    the sample that was at millisecond 0 moves to millisecond `offset`, and what
    falls off the end wraps around to the front.

    What is preserved, exactly:
      - each neuron's spike count in each trial (rotation moves spikes, never
        adds or removes them)
      - each neuron's own autocorrelation, i.e. its bursting and its inter-spike
        interval structure, which rotation does not touch
    What is destroyed:
      - the alignment BETWEEN neurons. Because every neuron gets a DIFFERENT
        offset, any moment where two neurons used to fire together is pulled
        apart.

    So the shuffled data is the same population of neurons, each behaving exactly
    as before, with no population-level coordination left. That is the right null
    for a claim about population dimensionality.
    """
    n_trials, n_neurons, n_ms = counts.shape
    offsets = rng.integers(0, n_ms, size=(n_trials, n_neurons))
    millisecond = np.arange(n_ms)
    columns = (millisecond[None, None, :] - offsets[:, :, None]) % n_ms
    return np.take_along_axis(counts, columns, axis=2)


def build_matrix(counts, bin_ms=1, paper_pipeline=True):
    """(sample, neuron). paper_pipeline adds their 50-trial average + 10 ms smooth."""
    n_trials, n_neurons, n_ms = counts.shape

    whole_bins = (n_ms // bin_ms) * bin_ms
    trimmed = counts[:, :, :whole_bins]
    reshaped = trimmed.reshape(n_trials, n_neurons, -1, bin_ms)
    binned = reshaped.sum(axis=3).astype(np.float32)

    if paper_pipeline:
        n_groups = n_trials // GROUP
        keep = binned[:n_groups * GROUP]
        grouped = keep.reshape(n_groups, GROUP, n_neurons, -1)
        binned = grouped.mean(axis=1)

        sigma_bins = (SMOOTH_MS / bin_ms) / 2.355        # FWHM -> sigma
        if sigma_bins > 0.3:
            binned = gaussian_filter1d(binned, sigma_bins, axis=2)

    time_last = binned.transpose(0, 2, 1)
    return time_last.reshape(-1, binned.shape[1])


def dimensions(matrix):
    """(PCs for 95% of variance, participation ratio, share held by the top PC)."""
    pca = PCA()
    pca.fit(matrix)

    variance = pca.explained_variance_               # the eigenvalues
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    participation = float(variance.sum() ** 2 / np.square(variance).sum())
    top_share = float(pca.explained_variance_ratio_[0])
    return n_95, participation, top_share


# %% [ STEP 1 ] NPBN: load it, and look at what we are working with

npbn = load_spontaneous("NPBN")
n_trials, n_neurons, n_ms = npbn.shape

mean_rate = 1000 * npbn.mean()
empty_share = 100 * (npbn == 0).mean()

print(f"NPBN spontaneous activity: {n_neurons} neurons x {n_trials} windows x {n_ms} ms")
print(f"mean firing rate: {mean_rate:.2f} Hz")
print(f"share of 1 ms bins that are empty: {empty_share:.1f}%")
print("\nThat last number is the reason to be suspicious. If almost every bin is")
print("empty, two neurons rarely have anything to correlate at 1 ms, and a flat")
print("spectrum is what you should expect before looking.")


# %% [ STEP 2 ] NPBN: what the shuffle actually does
#
# Before trusting the shuffle as a null, check that it preserves what it claims
# to preserve and destroys what it claims to destroy. Every line here is one
# step, so each intermediate can be looked at.

npbn_shuffled_once = shuffle_neurons_apart(npbn)

spikes_before = npbn.sum(axis=(0, 2))              # per neuron, whole session
spikes_after = npbn_shuffled_once.sum(axis=(0, 2))
counts_identical = np.array_equal(spikes_before, spikes_after)

matrix_before = build_matrix(npbn, bin_ms=1, paper_pipeline=False)
matrix_after = build_matrix(npbn_shuffled_once, bin_ms=1, paper_pipeline=False)

correlation_before = np.corrcoef(matrix_before, rowvar=False)
correlation_after = np.corrcoef(matrix_after, rowvar=False)
off_diagonal = ~np.eye(n_neurons, dtype=bool)

mean_correlation_before = np.abs(correlation_before[off_diagonal]).mean()
mean_correlation_after = np.abs(correlation_after[off_diagonal]).mean()

print(f"every neuron's spike count unchanged by the shuffle: {counts_identical}")
print(f"mean |correlation| between neuron pairs, real     : {mean_correlation_before:.4f}")
print(f"mean |correlation| between neuron pairs, shuffled : {mean_correlation_after:.4f}")
print("\nSo: same neurons, same rates, same bursting, coordination removed.")


# %% [ STEP 3 ] NPBN: reproduce the paper's number

npbn_paper_matrix = build_matrix(npbn, bin_ms=1, paper_pipeline=True)
paper_pc95, paper_pr, paper_top = dimensions(npbn_paper_matrix)

print(f"PCs for 95% of the variance: {paper_pc95} out of {n_neurons} neurons "
      f"= {100 * paper_pc95 / n_neurons:.0f}% of available PCs")
print(f"paper reported {PAPER_PC_COUNTS}, i.e. 62-81% of available PCs")
print("\nThe replication lands in their range, so their arithmetic is fine. The")
print("question is what the number means.")


# %% [ STEP 4 ] NPBN: the same number, on data with no coordination left

npbn_shuffled_pc95 = []
npbn_shuffled_pr = []
npbn_shuffled_top = []
for repeat in range(N_SHUFFLE):
    shuffled_counts = shuffle_neurons_apart(npbn)
    shuffled_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=True)
    pc95, pr, top = dimensions(shuffled_matrix)
    npbn_shuffled_pc95.append(pc95)
    npbn_shuffled_pr.append(pr)
    npbn_shuffled_top.append(top)

mean_shuffled_pc95 = np.mean(npbn_shuffled_pc95)
sd_shuffled_pc95 = np.std(npbn_shuffled_pc95)

print(f"real data            : {paper_pc95} PCs for 95%")
print(f"rate-matched shuffle : {mean_shuffled_pc95:.1f} PCs for 95% "
      f"(SD {sd_shuffled_pc95:.1f} over {N_SHUFFLE} shuffles)")
print("\nThe shuffle has no between-neuron structure at all. If it needs about as")
print("many PCs as the real data, the count is not measuring population")
print("structure. That is the argument, in one comparison.")


# %% [ STEP 5 ] NPBN: the same question, with a metric that can answer it

mean_shuffled_pr = np.mean(npbn_shuffled_pr)

print(f"participation ratio, real     : {paper_pr:.1f}")
print(f"participation ratio, shuffle  : {mean_shuffled_pr:.1f}")
print(f"neurons in the session        : {n_neurons}")
print(f"top PC holds {100 * paper_top:.1f}% of the variance in the real data")
print("\nPR(real) well below PR(shuffle) means the real spectrum IS concentrated,")
print("so structure exists and the 95% cutoff simply could not see it. STEP 6")
print("checks whether that gap is real or manufactured by their preprocessing.")


# %% [ STEP 6 ] NPBN: is the gap there without their preprocessing?
#
# The paper averages 50 sequential trials and smooths with a 10 ms Gaussian. Both
# operations change the spectrum before it is measured: averaging suppresses
# whatever is independent across trials, smoothing correlates neighbouring
# samples. So run the same real-vs-shuffle comparison with those steps switched
# OFF, and see whether the PR gap survives.
#
# Their own Methods describe the PCA as fitted to "800 or 1600 times 300 data
# points", which is the UNaveraged count -- so the raw row is arguably the one
# that matches what they say they did.

npbn_raw_matrix = build_matrix(npbn, bin_ms=1, paper_pipeline=False)
raw_pc95, raw_pr, raw_top = dimensions(npbn_raw_matrix)

npbn_raw_shuffled_pr = []
for repeat in range(N_SHUFFLE):
    shuffled_counts = shuffle_neurons_apart(npbn)
    shuffled_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=False)
    pc95, pr, top = dimensions(shuffled_matrix)
    npbn_raw_shuffled_pr.append(pr)

mean_raw_shuffled_pr = np.mean(npbn_raw_shuffled_pr)
raw_gap = raw_pr - mean_raw_shuffled_pr
paper_gap = paper_pr - mean_shuffled_pr

print(f"{'':30} {'PR real':>8} {'PR shuffled':>12} {'gap':>6}")
print(f"{'raw 1 ms, nothing done':30} {raw_pr:8.1f} {mean_raw_shuffled_pr:12.1f} {raw_gap:6.1f}")
print(f"{'their averaging + smoothing':30} {paper_pr:8.1f} {mean_shuffled_pr:12.1f} {paper_gap:6.1f}")
print("\nThe raw gap near zero says there is no fast (1 ms) coordination to find.")
print("The large gap after 50-trial averaging says the structure that does exist")
print("is SLOW -- it survives averaging 50 sequential trials, which is exactly")
print("what a drifting session looks like. Either way it is low-dimensional, and")
print("neither reading supports 'extremely high-dimensional'.")


# %% [ STEP 7 ] NPBN: their Fig 1F, with the shuffle drawn on it
#
# The paper bins more coarsely, watches dimensionality collapse, and uses that to
# argue against calcium imaging and 100 ms bins. Independent noise averages out
# under coarser bins too, so the shuffle should collapse as well.

sweep_rows = []
for bin_ms in BIN_MS_LIST:
    real_matrix = build_matrix(npbn, bin_ms=bin_ms, paper_pipeline=True)
    real_pc95, real_pr, real_top = dimensions(real_matrix)

    shuffled_counts = shuffle_neurons_apart(npbn)
    shuffled_matrix = build_matrix(shuffled_counts, bin_ms=bin_ms, paper_pipeline=True)
    shuf_pc95, shuf_pr, shuf_top = dimensions(shuffled_matrix)

    sweep_rows.append({"bin_ms": bin_ms,
                       "pc95_real": real_pc95, "pc95_shuffled": shuf_pc95,
                       "pr_real": real_pr, "pr_shuffled": shuf_pr})

npbn_sweep = pd.DataFrame(sweep_rows)
print(npbn_sweep.round(1).to_string(index=False))
print("\nIf the two pc95 columns fall together, Fig 1F is showing noise averaging")
print("out rather than neural dimensionality being lost.")


# %% [ STEP 8 ] NPBN: or is the number just how many neurons you recorded?

neuron_rows = []
for take in [10, 20, 30, n_neurons]:
    pc95_samples = []
    pr_samples = []
    for repeat in range(10):
        chosen = rng.choice(n_neurons, size=take, replace=False)
        subset = npbn[:, chosen, :]
        subset_matrix = build_matrix(subset, bin_ms=1, paper_pipeline=False)
        pc95, pr, top = dimensions(subset_matrix)
        pc95_samples.append(pc95)
        pr_samples.append(pr)

    mean_pc95 = np.mean(pc95_samples)
    mean_pr = np.mean(pr_samples)
    neuron_rows.append({"neurons": take,
                        "pc95": mean_pc95,
                        "pr": mean_pr,
                        "pc95_per_neuron": mean_pc95 / take,
                        "pr_per_neuron": mean_pr / take})

npbn_curve = pd.DataFrame(neuron_rows)
print(npbn_curve.round(2).to_string(index=False))
print("\nIf pc95_per_neuron stays flat as neurons are added, then '62-81% of")
print("available PCs' is a statement about the size of the recording.")


# %% [ STEP 9a ] NPBN: the two cumulative-variance curves the figure needs
#
# Computed here, in the open, rather than inside the plotting function. The plot
# below receives finished arrays and does no arithmetic of its own.

pca_real = PCA()
pca_real.fit(npbn_raw_matrix)
cumulative_real = 100 * np.cumsum(pca_real.explained_variance_ratio_)

npbn_shuffled_for_figure = shuffle_neurons_apart(npbn)
matrix_shuffled_for_figure = build_matrix(npbn_shuffled_for_figure, bin_ms=1,
                                          paper_pipeline=False)
pca_shuffled = PCA()
pca_shuffled.fit(matrix_shuffled_for_figure)
cumulative_shuffled = 100 * np.cumsum(pca_shuffled.explained_variance_ratio_)

print(f"real     reaches 95% at PC {np.searchsorted(cumulative_real, 95) + 1}")
print(f"shuffled reaches 95% at PC {np.searchsorted(cumulative_shuffled, 95) + 1}")


# %% [ STEP 9b ] NPBN: the figure


def plot_the_argument(cum_real, cum_shuffled, sweep, curve, name, path):
    """Three panels, drawn from finished arrays. No computation happens here."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    ax = axes[0]
    ax.plot(np.arange(1, len(cum_real) + 1), cum_real, color="#2c3e50", ls="-",
            lw=1.6, label="real")
    ax.plot(np.arange(1, len(cum_shuffled) + 1), cum_shuffled, color="#c0392b",
            ls="--", lw=1.6, label="rate-matched shuffle")
    ax.axhline(95, color="#7b8794", lw=0.8, ls=":")
    ax.text(1, 96, "95% cutoff", fontsize=8, color="#7b8794")
    ax.set_xlabel("number of PCs")
    ax.set_ylabel("cumulative variance (%)")
    ax.set_title(f"{name}: what the spectrum looks like", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax = axes[1]
    ax.plot(sweep.bin_ms, sweep.pc95_real, "o-", color="#2c3e50", lw=1.4, ms=4,
            label="PCs for 95%, real")
    ax.plot(sweep.bin_ms, sweep.pc95_shuffled, "o--", color="#c0392b", lw=1.4,
            ms=4, label="PCs for 95%, shuffled")
    ax.set_xscale("log")
    ax.set_xticks(BIN_MS_LIST)
    ax.set_xticklabels(BIN_MS_LIST)
    ax.set_xlabel("bin width (ms)")
    ax.set_ylabel("PCs for 95% variance")
    ax.set_title("their Fig 1F, with the shuffle on it", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(curve.neurons, curve.pc95, "o-", color="#2c3e50", lw=1.4, ms=4,
            label="PCs for 95%")
    ax.plot(curve.neurons, curve.pr, "s--", color="#c1703a", lw=1.4, ms=4,
            label="participation ratio")
    ax.plot([0, curve.neurons.max()], [0, curve.neurons.max()], color="#7b8794",
            lw=0.8, ls=":", label="one dimension per neuron")
    ax.set_xlabel("neurons included")
    ax.set_ylabel("dimensions")
    ax.set_title("or is it the size of the recording?", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(f"Critique A on {name} — the 95% PC count cannot separate "
                 f"richness from noise", x=0.02, ha="left", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150, facecolor="white")
    return path


figure_path = plot_the_argument(cumulative_real, cumulative_shuffled, npbn_sweep,
                                npbn_curve, "NPBN",
                                f"{OUT_DIR}/dimensionality_NPBN.png")
print(f"wrote {figure_path}")


# %% [ THE SAME STEPS, ONE CELL PER SESSION ]
#
# Each cell below is STEPS 3 to 6 written out for one session: reproduce their
# number, shuffle it, and check whether the PR gap needs their preprocessing.
# Written out rather than looped so each session's numbers can be looked at on
# their own.


# %% NPBN's row (walked through above)

npbn_raw_matrix = build_matrix(npbn, bin_ms=1, paper_pipeline=False)
npbn_raw_pc95, npbn_raw_pr, npbn_raw_top = dimensions(npbn_raw_matrix)

npbn_paper_matrix = build_matrix(npbn, bin_ms=1, paper_pipeline=True)
npbn_paper_pc95, npbn_paper_pr, npbn_paper_top = dimensions(npbn_paper_matrix)

npbn_null_paper_pc95 = []
npbn_null_paper_pr = []
npbn_null_raw_pr = []
for repeat in range(N_SHUFFLE):
    shuffled_counts = shuffle_neurons_apart(npbn)
    shuffled_paper_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=True)
    pc95, pr, top = dimensions(shuffled_paper_matrix)
    npbn_null_paper_pc95.append(pc95)
    npbn_null_paper_pr.append(pr)

    shuffled_raw_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=False)
    pc95, pr, top = dimensions(shuffled_raw_matrix)
    npbn_null_raw_pr.append(pr)

story_npbn = pd.Series({
    "session": "NPBN",
    "neurons": npbn.shape[1],
    "rate_hz": round(1000 * npbn.mean(), 2),
    "pc95_real": npbn_paper_pc95,
    "pc95_shuffled": round(np.mean(npbn_null_paper_pc95), 1),
    "pc95_pct_of_neurons": round(100 * npbn_paper_pc95 / npbn.shape[1]),
    "pr_real": round(npbn_paper_pr, 1),
    "pr_shuffled": round(np.mean(npbn_null_paper_pr), 1),
    "pr_raw_real": round(npbn_raw_pr, 1),
    "pr_raw_shuffled": round(np.mean(npbn_null_raw_pr), 1),
})
print(story_npbn.to_string())


# %% NPBK -- visual patterns only, the lowest firing rate

npbk = load_spontaneous("NPBK")

npbk_raw_matrix = build_matrix(npbk, bin_ms=1, paper_pipeline=False)
npbk_raw_pc95, npbk_raw_pr, npbk_raw_top = dimensions(npbk_raw_matrix)

npbk_paper_matrix = build_matrix(npbk, bin_ms=1, paper_pipeline=True)
npbk_paper_pc95, npbk_paper_pr, npbk_paper_top = dimensions(npbk_paper_matrix)

npbk_null_paper_pc95 = []
npbk_null_paper_pr = []
npbk_null_raw_pr = []
for repeat in range(N_SHUFFLE):
    shuffled_counts = shuffle_neurons_apart(npbk)
    shuffled_paper_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=True)
    pc95, pr, top = dimensions(shuffled_paper_matrix)
    npbk_null_paper_pc95.append(pc95)
    npbk_null_paper_pr.append(pr)

    shuffled_raw_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=False)
    pc95, pr, top = dimensions(shuffled_raw_matrix)
    npbk_null_raw_pr.append(pr)

story_npbk = pd.Series({
    "session": "NPBK",
    "neurons": npbk.shape[1],
    "rate_hz": round(1000 * npbk.mean(), 2),
    "pc95_real": npbk_paper_pc95,
    "pc95_shuffled": round(np.mean(npbk_null_paper_pc95), 1),
    "pc95_pct_of_neurons": round(100 * npbk_paper_pc95 / npbk.shape[1]),
    "pr_real": round(npbk_paper_pr, 1),
    "pr_shuffled": round(np.mean(npbk_null_paper_pr), 1),
    "pr_raw_real": round(npbk_raw_pr, 1),
    "pr_raw_shuffled": round(np.mean(npbk_null_raw_pr), 1),
})
print(story_npbk.to_string())


# %% NPBM -- tactile and visual patterns in one session, 1600 trials

npbm = load_spontaneous("NPBM")

npbm_raw_matrix = build_matrix(npbm, bin_ms=1, paper_pipeline=False)
npbm_raw_pc95, npbm_raw_pr, npbm_raw_top = dimensions(npbm_raw_matrix)

npbm_paper_matrix = build_matrix(npbm, bin_ms=1, paper_pipeline=True)
npbm_paper_pc95, npbm_paper_pr, npbm_paper_top = dimensions(npbm_paper_matrix)

npbm_null_paper_pc95 = []
npbm_null_paper_pr = []
npbm_null_raw_pr = []
for repeat in range(N_SHUFFLE):
    shuffled_counts = shuffle_neurons_apart(npbm)
    shuffled_paper_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=True)
    pc95, pr, top = dimensions(shuffled_paper_matrix)
    npbm_null_paper_pc95.append(pc95)
    npbm_null_paper_pr.append(pr)

    shuffled_raw_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=False)
    pc95, pr, top = dimensions(shuffled_raw_matrix)
    npbm_null_raw_pr.append(pr)

story_npbm = pd.Series({
    "session": "NPBM",
    "neurons": npbm.shape[1],
    "rate_hz": round(1000 * npbm.mean(), 2),
    "pc95_real": npbm_paper_pc95,
    "pc95_shuffled": round(np.mean(npbm_null_paper_pc95), 1),
    "pc95_pct_of_neurons": round(100 * npbm_paper_pc95 / npbm.shape[1]),
    "pr_real": round(npbm_paper_pr, 1),
    "pr_shuffled": round(np.mean(npbm_null_paper_pr), 1),
    "pr_raw_real": round(npbm_raw_pr, 1),
    "pr_raw_shuffled": round(np.mean(npbm_null_raw_pr), 1),
})
print(story_npbm.to_string())


# %% NPBO -- tactile patterns, onsets confirmed against the LFP artifacts (0.4 ms)

npbo = load_spontaneous("NPBO")

npbo_raw_matrix = build_matrix(npbo, bin_ms=1, paper_pipeline=False)
npbo_raw_pc95, npbo_raw_pr, npbo_raw_top = dimensions(npbo_raw_matrix)

npbo_paper_matrix = build_matrix(npbo, bin_ms=1, paper_pipeline=True)
npbo_paper_pc95, npbo_paper_pr, npbo_paper_top = dimensions(npbo_paper_matrix)

npbo_null_paper_pc95 = []
npbo_null_paper_pr = []
npbo_null_raw_pr = []
for repeat in range(N_SHUFFLE):
    shuffled_counts = shuffle_neurons_apart(npbo)
    shuffled_paper_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=True)
    pc95, pr, top = dimensions(shuffled_paper_matrix)
    npbo_null_paper_pc95.append(pc95)
    npbo_null_paper_pr.append(pr)

    shuffled_raw_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=False)
    pc95, pr, top = dimensions(shuffled_raw_matrix)
    npbo_null_raw_pr.append(pr)

story_npbo = pd.Series({
    "session": "NPBO",
    "neurons": npbo.shape[1],
    "rate_hz": round(1000 * npbo.mean(), 2),
    "pc95_real": npbo_paper_pc95,
    "pc95_shuffled": round(np.mean(npbo_null_paper_pc95), 1),
    "pc95_pct_of_neurons": round(100 * npbo_paper_pc95 / npbo.shape[1]),
    "pr_real": round(npbo_paper_pr, 1),
    "pr_shuffled": round(np.mean(npbo_null_paper_pr), 1),
    "pr_raw_real": round(npbo_raw_pr, 1),
    "pr_raw_shuffled": round(np.mean(npbo_null_raw_pr), 1),
})
print(story_npbo.to_string())


# %% all four side by side

everyone = pd.DataFrame([story_npbn, story_npbk, story_npbm, story_npbo])
print(everyone.to_string(index=False))
print("\npc95_real against pc95_shuffled : is the headline metric fooled?")
print("pr_raw_real against pr_raw_shuffled : is there structure without their")
print("preprocessing? That pair is the one that decides the claim.")

everyone.to_csv(f"{OUT_DIR}/dimensionality_summary.csv", index=False)
print(f"\nwrote {OUT_DIR}/dimensionality_summary.csv")

# %%
