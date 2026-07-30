# %% [ THE ARGUMENT ]
"""Critique A: their dimensionality number cannot tell richness from noise.

  THE CLAIM      The paper fits PCA to spontaneous activity (the 300 ms before
                 each stimulus) and counts the PCs needed for 95% of the
                 variance: 50, 27, 55, 42, 28 across the five experiments, which
                 is 62-81% of all available PCs. It reads that as an extremely
                 high-dimensional state space.

  THE WEAKNESS   That count is large whenever the eigenvalue spectrum is FLAT, and
                 independent noise produces a flat spectrum. At 1 ms bins with
                 neurons firing a few Hz, nearly every bin is empty, so flat is
                 the default expectation rather than a finding.

  THE TEST       Destroy all between-neuron structure while keeping each neuron
                 exactly as active as it was, then recount. Step 2 below shows
                 precisely what that shuffle does.

  THE BETTER     Participation ratio, PR = (sum of eigenvalues)^2 / sum of
  METRIC         eigenvalues^2, which answers to the shape of the whole spectrum
                 instead of one cutoff. Flat -> PR near the neuron count.
                 Concentrated -> PR small.

  WHY NPBI IS    Its visuo-tactile onsets are about 243 ms late, so its
  NOT HERE       "pre-stimulus" windows contain stimulus-driven activity. It
                 cannot measure spontaneous anything.

Steps 1-9 walk through NPBN, one cell at a time. Then NPBK, NPBM and NPBO each get
their own cell so the four can be compared.
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


# %% [ STEP 0b ] the four operations, each one thing


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
# Before trusting the shuffle as a null, check that it preserves what it claims to
# preserve and destroys what it claims to destroy.

npbn_shuffled_once = shuffle_neurons_apart(npbn)

spikes_before = npbn.sum(axis=(0, 2))              # per neuron, whole session
spikes_after = npbn_shuffled_once.sum(axis=(0, 2))
counts_identical = np.array_equal(spikes_before, spikes_after)

# pairwise correlation between neurons, at 1 ms, before and after
matrix_before = build_matrix(npbn, bin_ms=1, paper_pipeline=False)
matrix_after = build_matrix(npbn_shuffled_once, bin_ms=1, paper_pipeline=False)

correlation_before = np.corrcoef(matrix_before, rowvar=False)
correlation_after = np.corrcoef(matrix_after, rowvar=False)
off_diagonal = ~np.eye(n_neurons, dtype=bool)

print(f"every neuron's spike count unchanged by the shuffle: {counts_identical}")
print(f"mean |correlation| between neuron pairs, real     : "
      f"{np.abs(correlation_before[off_diagonal]).mean():.4f}")
print(f"mean |correlation| between neuron pairs, shuffled : "
      f"{np.abs(correlation_after[off_diagonal]).mean():.4f}")
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

shuffled_pc95 = []
shuffled_pr = []
shuffled_top = []
for repeat in range(N_SHUFFLE):
    shuffled_counts = shuffle_neurons_apart(npbn)
    shuffled_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=True)
    pc95, pr, top = dimensions(shuffled_matrix)
    shuffled_pc95.append(pc95)
    shuffled_pr.append(pr)
    shuffled_top.append(top)

print(f"real data            : {paper_pc95} PCs for 95%")
print(f"rate-matched shuffle : {np.mean(shuffled_pc95):.1f} PCs for 95% "
      f"(SD {np.std(shuffled_pc95):.1f} over {N_SHUFFLE} shuffles)")
print("\nThe shuffle has no between-neuron structure at all. If it needs about as")
print("many PCs as the real data, the count is not measuring population")
print("structure. That is the argument, in one comparison.")


# %% [ STEP 5 ] NPBN: the same question, with a metric that can answer it

print(f"participation ratio, real     : {paper_pr:.1f}")
print(f"participation ratio, shuffle  : {np.mean(shuffled_pr):.1f}")
print(f"neurons in the session        : {n_neurons}")
print(f"top PC holds {100 * paper_top:.1f}% of the variance in the real data")
print("\nPR(real) well below PR(shuffle) would mean the real spectrum IS")
print("concentrated, so structure exists and the 95% cutoff simply could not see")
print("it. Step 6 checks whether that gap is real or manufactured.")


# %% [ STEP 6 ] NPBN: is the gap there without their preprocessing?
#
# The paper averages 50 sequential trials and smooths with a 10 ms Gaussian. Both
# operations change the spectrum before it is measured: averaging suppresses
# whatever is independent across trials, smoothing correlates neighbouring
# samples. So run the same real-vs-shuffle comparison with those steps switched
# OFF, and see whether the PR gap survives.

npbn_raw_matrix = build_matrix(npbn, bin_ms=1, paper_pipeline=False)
raw_pc95, raw_pr, raw_top = dimensions(npbn_raw_matrix)

raw_shuffled_pr = []
for repeat in range(N_SHUFFLE):
    shuffled_counts = shuffle_neurons_apart(npbn)
    shuffled_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=False)
    pc95, pr, top = dimensions(shuffled_matrix)
    raw_shuffled_pr.append(pr)

print(f"{'':30} {'PR real':>8} {'PR shuffled':>12} {'gap':>6}")
print(f"{'raw 1 ms, nothing done':30} {raw_pr:8.1f} "
      f"{np.mean(raw_shuffled_pr):12.1f} {raw_pr - np.mean(raw_shuffled_pr):6.1f}")
print(f"{'their averaging + smoothing':30} {paper_pr:8.1f} "
      f"{np.mean(shuffled_pr):12.1f} {paper_pr - np.mean(shuffled_pr):6.1f}")
print("\nIf the gap is near zero raw and only opens after their preprocessing,")
print("then the 'structure' is something the pipeline created, and the honest")
print("statement is the raw one.")


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
    shuffled_matrix = build_matrix(shuffled_counts, bin_ms=bin_ms,
                                   paper_pipeline=True)
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

    neuron_rows.append({"neurons": take,
                        "pc95": np.mean(pc95_samples),
                        "pr": np.mean(pr_samples),
                        "pc95_per_neuron": np.mean(pc95_samples) / take,
                        "pr_per_neuron": np.mean(pr_samples) / take})

npbn_curve = pd.DataFrame(neuron_rows)
print(npbn_curve.round(2).to_string(index=False))
print("\nIf pc95_per_neuron stays flat as neurons are added, then '62-81% of")
print("available PCs' is a statement about the size of the recording.")


# %% [ STEP 9 ] NPBN: the figure


def plot_the_argument(real_counts, name, sweep, curve, path):
    """Three panels: the spectrum, the bin-width collapse, the neuron-count curve."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    ax = axes[0]
    real_matrix = build_matrix(real_counts, bin_ms=1, paper_pipeline=False)
    shuffled_counts = shuffle_neurons_apart(real_counts)
    shuffled_matrix = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=False)

    for matrix, colour, style, tag in ((real_matrix, "#2c3e50", "-", "real"),
                                       (shuffled_matrix, "#c0392b", "--",
                                        "rate-matched shuffle")):
        pca = PCA()
        pca.fit(matrix)
        cumulative = 100 * np.cumsum(pca.explained_variance_ratio_)
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative, color=colour,
                ls=style, lw=1.6, label=tag)

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


figure_path = plot_the_argument(npbn, "NPBN", npbn_sweep, npbn_curve,
                               f"{OUT_DIR}/dimensionality_NPBN.png")
print(f"wrote {figure_path}")


# %% [ THE SAME STEPS, ONE CELL PER SESSION ]


def one_session_story(counts, name):
    """Steps 3 to 6 for one session, as a one-row summary."""
    paper_matrix = build_matrix(counts, bin_ms=1, paper_pipeline=True)
    paper_pc95, paper_pr, paper_top = dimensions(paper_matrix)

    raw_matrix = build_matrix(counts, bin_ms=1, paper_pipeline=False)
    raw_pc95, raw_pr, raw_top = dimensions(raw_matrix)

    paper_shuffled_pc95 = []
    paper_shuffled_pr = []
    raw_shuffled_pr = []
    for repeat in range(N_SHUFFLE):
        shuffled_counts = shuffle_neurons_apart(counts)

        shuffled_paper = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=True)
        pc95, pr, top = dimensions(shuffled_paper)
        paper_shuffled_pc95.append(pc95)
        paper_shuffled_pr.append(pr)

        shuffled_raw = build_matrix(shuffled_counts, bin_ms=1, paper_pipeline=False)
        pc95, pr, top = dimensions(shuffled_raw)
        raw_shuffled_pr.append(pr)

    return pd.Series({
        "session": name,
        "neurons": counts.shape[1],
        "rate_hz": round(1000 * counts.mean(), 2),
        "pc95_real": paper_pc95,
        "pc95_shuffled": round(np.mean(paper_shuffled_pc95), 1),
        "pc95_pct_of_neurons": round(100 * paper_pc95 / counts.shape[1]),
        "pr_real": round(paper_pr, 1),
        "pr_shuffled": round(np.mean(paper_shuffled_pr), 1),
        "pr_raw_real": round(raw_pr, 1),
        "pr_raw_shuffled": round(np.mean(raw_shuffled_pr), 1),
    })


# %% NPBN's row (walked through above)

story_npbn = one_session_story(npbn, "NPBN")
print(story_npbn.to_string())

# %% NPBK -- visual patterns only, the lowest firing rate

npbk = load_spontaneous("NPBK")
story_npbk = one_session_story(npbk, "NPBK")
print(story_npbk.to_string())

# %% NPBM -- tactile and visual patterns in one session

npbm = load_spontaneous("NPBM")
story_npbm = one_session_story(npbm, "NPBM")
print(story_npbm.to_string())

# %% NPBO -- tactile patterns, onsets confirmed against the LFP artifacts

npbo = load_spontaneous("NPBO")
story_npbo = one_session_story(npbo, "NPBO")
print(story_npbo.to_string())

# %% all four side by side

everyone = pd.DataFrame([story_npbn, story_npbk, story_npbm, story_npbo])
print(everyone.to_string(index=False))
print("\npc95_real against pc95_shuffled : is the headline metric fooled?")
print("pr_raw_real against pr_raw_shuffled : is there structure without their")
print("preprocessing? That pair is the one that decides the claim.")

# %%
