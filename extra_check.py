# %%
"""Per-neuron PSTH responsiveness check, every unit against every stimulus class.

The paper decides which neurons "count" with a PSTH test (Methods): bin spikes at
2 ms, take the mean firing rate in the 300 ms prestimulus window as baseline, and
call a neuron responsive if the rate exceeds baseline + 2 SD for two consecutive
bins in the 300 ms poststimulus window. That test is run once, on tactile trials.

Here we run a stricter version of it -- baseline + 3 SD, and more than 10 bins
above threshold -- and we run it for EVERY unit against EVERY stimulus class
separately, rather than once per neuron. Separating the classes is the point: a
neuron driven by tactile F20 but blind to visual vA is invisible to a single
pooled test, and so is a neuron that follows the pulse structure of one pattern
but not another.

The window is always -300 ms to +300 ms around onset, regardless of how long the
pattern itself lasts (tactile patterns run 183-321 ms, visual 215 ms). The true
offset is drawn on every panel so the fixed window can be read against it.

Bins above threshold are counted anywhere in the poststimulus window, not
required to be consecutive. A neuron entrained to a pulse train fires in short
bursts that track the pattern, which scatters supra-threshold bins rather than
producing one long run; the longest consecutive run is recorded alongside so both
readings are available.

Caveat worth keeping in view: at 2 ms bins with neurons firing a few Hz, most
baseline bins hold zero spikes, so the baseline SD is dominated by counting noise
rather than by real rate variability. The 3 SD threshold is closer to "a bin with
a couple of spikes in it" than to a calibrated statistical test for low-rate
units. This is inherited from the paper's method; 3 SD / 10 bins is considerably
more conservative than their 2 SD / 2 bins. Units silent across the whole
baseline window have SD = 0, which would make the threshold 0 and flag a single
spike -- those are marked untestable instead of flagged.

Run:  conda run -n neural_analysis python psth_check.py
      or cell-by-cell in the VS Code interactive window, which additionally
      draws every session summary plus the first SHOW_DETAIL neurons inline.
Out:  figures/psth/psth_flags.csv          every unit x class, nothing dropped
      figures/psth/summary_<EXP>.png       flag matrix + per-class rates
      figures/psth/<EXP>_<quality>_unit<id>.png    one flagged neuron, all classes
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib

# Run as a plain script this writes hundreds of figures, so it forces a
# file-only backend. Under the VS Code interactive window or Jupyter, leave the
# inline backend alone -- otherwise nothing can ever be drawn to the window.
INTERACTIVE = "ipykernel" in sys.modules
if not INTERACTIVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pynwb import NWBHDF5IO

EXPERIMENTS = ["NPBI", "NPBK", "NPBM", "NPBN", "NPBO"]

# Short modality description per experiment (from the dataset description).
MODALITY = {
    "NPBI": "visual patterns + visuo-tactile",
    "NPBK": "visual patterns",
    "NPBM": "tactile patterns and visual patterns",
    "NPBN": "tactile patterns",
    "NPBO": "tactile patterns",
}

BIN_S = 0.002                       # paper: 2 ms bins
PRE_S = 0.300                       # paper: 300 ms prestimulus baseline
POST_S = 0.300                      # paper: 300 ms poststimulus test window
N_SD = 3.0                          # paper used 2; stricter here
MIN_BINS = 10                       # flag when MORE than this many bins exceed

QUALITY = ("good", "mua")           # which spike-sorting labels to analyse
MAX_DETAIL_MUA = 20                 # per session, ranked by supra-threshold bins
SHOW_DETAIL = 3                     # detail figures drawn inline per session;
                                    # every one is still written to disk

OUT_DIR = "figures/psth"

# Semantic roles, not a series palette. Each also carries a label or fixed
# position, so nothing is identified by colour alone.
INK = "#2c3e50"                     # the neuron's own PSTH
ALERT = "#c0392b"                   # bins over threshold, and the threshold line
ONSET = "#2e7d5b"                   # stimulus onset
OFFSET = "#c1703a"                  # true stimulus offset, and the mua series
MUTED = "#7b8794"

N_PRE = int(round(PRE_S / BIN_S))
N_POST = int(round(POST_S / BIN_S))
EDGES = np.linspace(-PRE_S, POST_S, N_PRE + N_POST + 1)
CENTERS_MS = 1000.0 * (EDGES[:-1] + EDGES[1:]) / 2.0
POST_MS = CENTERS_MS[N_PRE:]

# F5/F10/F20/F-inf are tactile patterns, vA-vD visual. Alphabetical order would
# put F10 before F5, so order them explicitly.
STIM_ORDER = ["F5", "F10", "F20", "F∞", "vA", "vB", "vC", "vD"]

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                     "axes.edgecolor": MUTED, "text.color": "#23303a"})


# %% The measurement

def class_of(trials):
    """Label each trial by what was actually presented.

    modality/stimulus, plus the concurrent tactile pattern where there is one
    (NPBI pairs every visuo-tactile visual pattern with F5).
    """
    label = trials["modality"].astype(str) + "/" + trials["stimulus"].astype(str)
    with_ = trials["tactile_with"].fillna("").astype(str)
    return np.where(with_.str.len() > 0, label + "+" + with_, label)


def class_sort_key(cls):
    modality, _, stimulus = cls.partition("/")
    base = stimulus.split("+")[0]
    return (modality, STIM_ORDER.index(base) if base in STIM_ORDER else 99, cls)


def psth(spike_times, onsets):
    """Trial-averaged firing rate in Hz, one value per BIN_S bin of the window."""
    lo = np.searchsorted(spike_times, onsets - PRE_S)
    hi = np.searchsorted(spike_times, onsets + POST_S)
    relative = np.concatenate(
        [spike_times[a:b] - t for a, b, t in zip(lo, hi, onsets)]
    )
    counts, _ = np.histogram(relative, bins=EDGES)
    return counts / (len(onsets) * BIN_S)


def longest_run(mask):
    """Length of the longest stretch of consecutive True values."""
    if not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return int((edges[1::2] - edges[::2]).max())


def score(rate):
    """Apply the threshold test to one unit's PSTH for one class."""
    baseline = rate[:N_PRE]
    mean, sd = float(baseline.mean()), float(baseline.std())
    post = rate[N_PRE:]

    # A unit that never fires during baseline has SD 0, which would put the
    # threshold at 0 and flag on a single spike. Not a response; not testable.
    untestable = sd == 0.0
    threshold = mean + N_SD * sd
    above = post > threshold
    peak = int(np.argmax(post))

    return {
        "baseline_hz": mean,
        "baseline_sd": sd,
        "threshold_hz": threshold,
        "n_supra": int(above.sum()),
        "run_max": longest_run(above),
        "peak_hz": float(post[peak]),
        "peak_lat_ms": float(POST_MS[peak]),
        "untestable": untestable,
        "flagged": bool(not untestable and above.sum() > MIN_BINS),
    }


def analyse(exp):
    """Every unit x every class for one session. Returns (rows, rates, meta)."""
    with NWBHDF5IO(f"nwb/{exp}.nwb", "r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        trials = nwb.trials.to_dataframe()

    trials = trials.assign(cls=class_of(trials))
    classes = sorted(trials["cls"].unique(), key=class_sort_key)
    onsets = {c: trials.loc[trials["cls"] == c, "start_time"].to_numpy()
              for c in classes}
    # Duration is a property of the pattern, so it is constant within a class.
    duration_ms = {c: 1000.0 * float(trials.loc[trials["cls"] == c, "duration_s"].median())
                   for c in classes}

    units = units[units["quality"].isin(QUALITY)]
    print(f"{exp} ({MODALITY[exp]}): {len(units)} units x {len(classes)} classes "
          f"= {len(units) * len(classes)} PSTHs")

    rows, rates = [], {}
    for cluster_id, quality, spikes in zip(units["cluster_id"], units["quality"],
                                           units["spike_times"]):
        spikes = np.sort(np.asarray(spikes, dtype=np.float64))
        for cls in classes:
            rate = psth(spikes, onsets[cls])
            rates[(cluster_id, cls)] = rate.astype(np.float32)
            rows.append({"exp": exp, "cluster_id": int(cluster_id),
                         "quality": quality, "cls": cls,
                         "n_trials": len(onsets[cls]),
                         "duration_ms": duration_ms[cls], **score(rate)})

    meta = {"classes": classes, "duration_ms": duration_ms}
    return pd.DataFrame(rows), rates, meta


# %% Figures

def finish(fig, path, show=False):
    """Write the figure to disk, and draw it inline when running interactively."""
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    if show and INTERACTIVE:
        plt.show()
    plt.close(fig)
    return path


def draw_panel(ax, rate, row, duration_ms):
    """One neuron, one class: the PSTH with the test drawn on top of it."""
    above = rate[N_PRE:] > row["threshold_hz"]

    ax.axvspan(0, duration_ms, color=OFFSET, alpha=0.07, lw=0)
    ax.bar(CENTERS_MS, rate, width=BIN_S * 1000, color=INK, lw=0)
    if above.any():
        ax.bar(POST_MS[above], rate[N_PRE:][above], width=BIN_S * 1000,
               color=ALERT, lw=0)

    ax.axhline(row["baseline_hz"], color=MUTED, lw=0.8, ls="--")
    ax.axhline(row["threshold_hz"], color=ALERT, lw=0.9, ls=":")
    ax.axvline(0, color=ONSET, lw=1.2)
    ax.axvline(duration_ms, color=OFFSET, lw=1.2)
    ax.axvline(POST_S * 1000, color=MUTED, lw=0.8, ls="--")

    verdict = ("UNTESTABLE" if row["untestable"]
               else "FLAGGED" if row["flagged"] else "not flagged")
    ax.set_title(f"{row['cls']}   {verdict}", loc="left", fontsize=9,
                 color=ALERT if row["flagged"] else "#23303a")
    ax.text(0.98, 0.94,
            f"{row['n_supra']} bins > {N_SD:.0f} SD  (longest run {row['run_max']})",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.set_xlim(-PRE_S * 1000, POST_S * 1000)
    # Headroom so the bin-count annotation never sits on top of the data.
    ax.set_ylim(0, 1.3 * max(float(rate.max()), float(row["threshold_hz"]), 1e-9))
    ax.spines[["right", "top"]].set_visible(False)


def detail_figure(exp, cluster_id, quality, unit_rows, rates, meta, show=False):
    """All of one neuron's PSTHs, one panel per class."""
    classes = meta["classes"]
    ncols = min(4, len(classes))
    nrows = int(np.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.5 * nrows),
                             dpi=150, squeeze=False)

    # drop=False: draw_panel reads row["cls"] for its title, so the column has
    # to survive becoming the index.
    indexed = unit_rows.set_index("cls", drop=False)
    for ax, cls in zip(axes.flat, classes):
        draw_panel(ax, rates[(cluster_id, cls)], indexed.loc[cls],
                   meta["duration_ms"][cls])
    for ax in axes.flat[len(classes):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time from onset (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("rate (Hz)")

    n_flagged = int(indexed["flagged"].sum())
    fig.suptitle(
        f"{exp}  unit {cluster_id} ({quality})  —  flagged in {n_flagged} of "
        f"{len(classes)} classes    "
        f"[green = onset, orange = true offset, dashed = +300 ms window edge]",
        x=0.012, ha="left", fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = f"{OUT_DIR}/{exp}_{quality}_unit{cluster_id}.png"
    return finish(fig, path, show)


def summary_figure(exp, df, rates, meta):
    """Session overview: who flagged, on what, and what the population does."""
    classes = meta["classes"]
    # Constrained layout rather than tight_layout: the third panel carries a
    # colorbar, which tight_layout cannot place.
    fig = plt.figure(figsize=(13.5, 4.6), dpi=200, layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.85, 1.5])

    # -- flag matrix: one row per unit, one column per class ------------------
    # 0 tested and quiet, 1 flagged, 2 untestable. Units are ordered good-first
    # and then by how many classes they flagged in, so structure is visible.
    wide = df.pivot(index="cluster_id", columns="cls", values="flagged")[classes]
    bad = df.pivot(index="cluster_id", columns="cls", values="untestable")[classes]
    quality = df.groupby("cluster_id")["quality"].first()

    matrix = wide.astype(int).where(~bad, 2)
    order = pd.DataFrame({"q": (quality != "good").astype(int),
                          "n": wide.sum(axis=1)}).sort_values(
        ["q", "n"], ascending=[True, False]).index
    matrix = matrix.loc[order]
    n_good = int((quality.loc[order] == "good").sum())

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest",
              cmap=ListedColormap(["#e8edf1", ALERT, "#c3ccd4"]), vmin=0, vmax=2)
    ax.axhline(n_good - 0.5, color=INK, lw=1.2)
    ax.text(len(classes) - 0.4, n_good / 2, f"good\nn={n_good}", fontsize=7.5,
            va="center", color=MUTED)
    ax.text(len(classes) - 0.4, (n_good + len(matrix)) / 2,
            f"mua\nn={len(matrix) - n_good}", fontsize=7.5, va="center", color=MUTED)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.split("/")[-1] for c in classes], rotation=45, ha="right",
                       fontsize=8)
    ax.set_ylabel("unit")
    ax.set_yticks([])
    ax.set_title("Flagged (red) per unit and class", loc="left", fontsize=10.5)

    # -- how many units flagged, per class -----------------------------------
    ax = fig.add_subplot(gs[0, 1])
    frac = (df[df["quality"] == "good"].groupby("cls")["flagged"].mean() * 100).reindex(classes)
    frac_mua = (df[df["quality"] == "mua"].groupby("cls")["flagged"].mean() * 100).reindex(classes)
    x = np.arange(len(classes))
    ax.bar(x - 0.2, frac, width=0.36, color=INK, label="good")
    ax.bar(x + 0.2, frac_mua, width=0.36, color=OFFSET, label="mua")
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("/")[-1] for c in classes], rotation=45, ha="right",
                       fontsize=8)
    ax.set_ylabel("units flagged (%)")
    ax.legend(frameon=False, fontsize=8.5, handlelength=1.2)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_title("Share of units driven, by class", loc="left", fontsize=10.5)

    # -- population response, baseline-normalised ----------------------------
    # Diverging and symmetric around zero so suppression reads as clearly as
    # drive. Pulse-locked entrainment shows up here as vertical stripes.
    good_ids = quality[quality == "good"].index
    stack = []
    for cls in classes:
        z = []
        for cluster_id in good_ids:
            rate = rates[(cluster_id, cls)].astype(np.float64)
            base = rate[:N_PRE]
            if base.std() > 0:
                z.append((rate - base.mean()) / base.std())
        stack.append(np.mean(z, axis=0) if z else np.zeros(len(CENTERS_MS)))
    stack = np.asarray(stack)

    ax = fig.add_subplot(gs[0, 2])
    limit = float(np.abs(stack).max()) or 1.0
    im = ax.imshow(stack, aspect="auto", interpolation="nearest", cmap="RdBu_r",
                   vmin=-limit, vmax=limit,
                   extent=[-PRE_S * 1000, POST_S * 1000, len(classes) - 0.5, -0.5])
    ax.axvline(0, color=ONSET, lw=1.2)
    ax.axvline(POST_S * 1000, color=MUTED, lw=0.8, ls="--")
    for i, cls in enumerate(classes):
        ax.plot([meta["duration_ms"][cls]] * 2, [i - 0.5, i + 0.5], color=OFFSET, lw=1.4)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("time from onset (ms)")
    ax.set_title("Population mean, good units (SD from baseline)", loc="left",
                 fontsize=10.5)
    fig.colorbar(im, ax=ax, pad=0.015, fraction=0.035).outline.set_visible(False)

    fig.suptitle(f"{exp} — {MODALITY[exp]}   "
                 f"(2 ms bins, baseline + {N_SD:.0f} SD, flag at > {MIN_BINS} bins "
                 f"in the 300 ms after onset)",
                 x=0.012, ha="left", fontsize=11.5)
    path = f"{OUT_DIR}/summary_{exp}.png"
    return finish(fig, path, show=True)


# %% Run every session

os.makedirs(OUT_DIR, exist_ok=True)
all_rows, overview = [], {}

for exp in EXPERIMENTS:
    df, rates, meta = analyse(exp)
    summary_figure(exp, df, rates, meta)

    # Detail figures for every flagged good unit, plus the strongest MUA units.
    # The cap is on plots only -- the CSV keeps every unit x class row.
    flagged = df[df["flagged"]]
    strength = flagged.groupby(["quality", "cluster_id"])["n_supra"].max()
    good_ids = list(strength.get("good", pd.Series(dtype=float)).index)
    mua_ranked = list(strength.get("mua", pd.Series(dtype=float))
                      .sort_values(ascending=False).index)
    mua_ids, mua_skipped = mua_ranked[:MAX_DETAIL_MUA], len(mua_ranked) - MAX_DETAIL_MUA

    drawn = {}
    for quality, ids in (("good", good_ids), ("mua", mua_ids)):
        for cluster_id in ids:
            unit_rows = df[(df["cluster_id"] == cluster_id) & (df["quality"] == quality)]
            drawn[(quality, cluster_id)] = detail_figure(
                exp, cluster_id, quality, unit_rows, rates, meta,
                show=len(drawn) < SHOW_DETAIL)

    df["detail_figure"] = [drawn.get((q, c), "") for q, c
                           in zip(df["quality"], df["cluster_id"])]
    all_rows.append(df)

    n_units = df["cluster_id"].nunique()
    n_flagged_units = flagged["cluster_id"].nunique()
    overview[exp] = {
        "modality": MODALITY[exp],
        "n_units": int(n_units),
        "n_classes": len(meta["classes"]),
        "n_pairs": int(len(df)),
        "n_pairs_flagged": int(df["flagged"].sum()),
        "n_pairs_untestable": int(df["untestable"].sum()),
        "n_units_flagged": int(n_flagged_units),
        "flagged_pct_good": round(float(
            df[df["quality"] == "good"]["flagged"].mean() * 100), 1),
        "flagged_pct_mua": round(float(
            df[df["quality"] == "mua"]["flagged"].mean() * 100), 1),
        "detail_figures": len(drawn),
        "mua_detail_skipped": int(max(mua_skipped, 0)),
    }
    o = overview[exp]
    print(f"   {o['n_pairs_flagged']}/{o['n_pairs']} unit-class pairs flagged "
          f"(good {o['flagged_pct_good']}%, mua {o['flagged_pct_mua']}%); "
          f"{o['n_units_flagged']}/{n_units} units flagged in >=1 class; "
          f"{o['n_pairs_untestable']} untestable")
    print(f"   wrote summary_{exp}.png + {len(drawn)} detail figures"
          + (f"  ({max(mua_skipped, 0)} flagged mua units not plotted, "
             f"cap MAX_DETAIL_MUA={MAX_DETAIL_MUA})" if mua_skipped > 0 else ""))


# %% Everything, in one table

results = pd.concat(all_rows, ignore_index=True)
results.to_csv(f"{OUT_DIR}/psth_flags.csv", index=False)
with open(f"{OUT_DIR}/psth_summary.json", "w") as fh:
    json.dump({"bin_ms": BIN_S * 1000, "pre_ms": PRE_S * 1000, "post_ms": POST_S * 1000,
               "n_sd": N_SD, "min_bins": MIN_BINS, "quality": list(QUALITY),
               "sessions": overview}, fh, indent=2, ensure_ascii=False)

print(f"\nwrote {OUT_DIR}/psth_flags.csv  ({len(results)} unit-class rows)")
print(f"wrote {OUT_DIR}/psth_summary.json")
