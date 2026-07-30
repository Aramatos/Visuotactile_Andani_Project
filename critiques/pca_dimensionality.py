# %%
"""Recreate the paper's PCA dimensionality result -- and check the ruler it used.

The paper counts how many principal components are needed to reach 95% of the
variance, gets a big number (50 of 61 neurons, 27 of ~40, ...), and concludes the
cortical state is "extremely high-dimensional".

The catch: a FLAT eigenvalue spectrum also needs almost every component to reach
95%. And a flat spectrum is what independent noise looks like -- at 1 ms bins
with neurons firing a few Hz, most bins hold 0 or 1 spike. So "needed 50 of 61"
is equally consistent with rich structure and with mush. The metric cannot tell
them apart.

Participation ratio can:

    PR = (sum of eigenvalues)^2 / (sum of squared eigenvalues)

Flat spectrum -> PR near n_neurons. Variance concentrated in a few axes -> PR
small. We compute both numbers on the real data and on a null that keeps each
neuron's own firing statistics but destroys the correlations BETWEEN neurons
(independent circular shifts of each neuron's rate). If the real PR sits well
below the null PR, the population is far more structured than independent
neurons would be, and the paper's "high-dimensional" reading is an artifact of
the ruler rather than a property of cortex.

Run:  conda run -n neural_analysis python pca_dimensionality.py
Out:  figures/pca_dimensionality.png
"""

import os
import json

import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from neo.core import SpikeTrain
from elephant.statistics import instantaneous_rate
from elephant.kernels import GaussianKernel
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA

EXP = "NPBN"
T_START, T_STOP = 0.0, 300.0            # seconds of recording analysed
BIN = 1 * pq.ms                         # paper: 1 ms bins
KERNEL = GaussianKernel(10 * pq.ms)     # paper: 10 ms Gaussian smoothing
N_NULL = 10                             # shuffle repeats

REAL, NULL = "#2c3e50", "#c1703a"
OUT = "figures/pca_dimensionality.png"


# %% Spike trains of the curated "good" units -> instantaneous rate

with NWBHDF5IO(f"nwb/{EXP}.nwb", "r") as io:
    units = io.read().units.to_dataframe()

good = units[units["quality"] == "good"]
trains = [
    SpikeTrain(
        np.asarray(st)[(np.asarray(st) >= T_START) & (np.asarray(st) < T_STOP)] * pq.s,
        t_start=T_START * pq.s, t_stop=T_STOP * pq.s,
    )
    for st in good["spike_times"]
]
print(f"{EXP}: {len(trains)} good units, {T_STOP - T_START:.0f} s")

rate = instantaneous_rate(trains, sampling_period=BIN, kernel=KERNEL)
X = np.asarray(rate, dtype=np.float32)          # (n_time, n_neurons)
print("rate matrix:", X.shape)


# %% The two rulers, on the real data and on an independent-neuron null

def spectrum(mat):
    return PCA().fit(mat).explained_variance_


def n_pcs_95(lam):
    """The paper's metric: how many components to reach 95% of the variance."""
    return int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1)


def participation_ratio(lam):
    """One number for how concentrated the spectrum is."""
    return lam.sum() ** 2 / (lam ** 2).sum()


def shuffle_independent(mat, rng):
    """Keep each neuron's own time course; destroy correlations between neurons."""
    return np.stack(
        [np.roll(mat[:, i], rng.integers(mat.shape[0])) for i in range(mat.shape[1])],
        axis=1,
    )


rng = np.random.default_rng(0)
lam_real = spectrum(X)
null_spectra = [spectrum(shuffle_independent(X, rng)) for _ in range(N_NULL)]
lam_null = np.mean(null_spectra, axis=0)

n_units = X.shape[1]
pr_real, pr_null = participation_ratio(lam_real), participation_ratio(lam_null)
n95_real, n95_null = n_pcs_95(lam_real), n_pcs_95(lam_null)

print(f"\n  paper's metric : {n95_real}/{n_units} PCs for 95% variance "
      f"(shuffled null: {n95_null}/{n_units})")
print(f"  participation  : PR = {pr_real:.1f}   (shuffled null: {pr_null:.1f})")
print(f"  -> real data is {pr_null / pr_real:.1f}x more concentrated than "
      f"independent neurons")


# %% Figure

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                     "axes.edgecolor": "#7b8794", "text.color": "#23303a"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7), dpi=220)

k = np.arange(1, n_units + 1)
ax1.semilogy(k, lam_real / lam_real.sum(), color=REAL, lw=2, label="real data")
ax1.semilogy(k, lam_null / lam_null.sum(), color=NULL, lw=2,
             label="shuffled (independent neurons)")
ax1.legend(loc="lower left", frameon=False, fontsize=9.5, handlelength=1.4)
ax1.set_xlabel("principal component")
ax1.set_ylabel("share of variance")
ax1.set_title("The spectrum", loc="left", fontsize=11)
ax1.spines[["right", "top"]].set_visible(False)

cum_real = np.cumsum(lam_real) / lam_real.sum()
cum_null = np.cumsum(lam_null) / lam_null.sum()
ax2.plot(k, cum_real, color=REAL, lw=2)
ax2.plot(k, cum_null, color=NULL, lw=2)
ax2.axhline(0.95, color="0.65", lw=0.9, ls="--")
ax2.text(1, 0.965, "95%", color="0.45", fontsize=8.5)
for n95, col in [(n95_real, REAL), (n95_null, NULL)]:
    ax2.plot([n95, n95], [0, 0.95], color=col, lw=0.9, ls=":")
    ax2.plot(n95, 0.95, "o", color=col, ms=5)
ax2.set_xlabel("principal component")
ax2.set_ylabel("cumulative variance")
ax2.set_ylim(0, 1.04)
ax2.set_title("The paper's ruler — and a better one", loc="left", fontsize=11)
ax2.spines[["right", "top"]].set_visible(False)
ax2.text(
    0.97, 0.06,
    f"95% needs {n95_real} of {n_units} PCs   (shuffled: {n95_null})\n"
    f"participation ratio {pr_real:.1f}   (shuffled: {pr_null:.1f})",
    transform=ax2.transAxes, ha="right", va="bottom", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.45", fc="#dceaf6", ec="none"),
)

fig.suptitle(
    f"{EXP}: both rulers on the same data — the count says "
    f"\"high-dimensional\", the spread says otherwise",
    x=0.012, ha="left", fontsize=11.5,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
os.makedirs("figures", exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"\nwrote {OUT}")

# Numbers for the presentation slide, so the text can never drift from the figure.
with open("figures/pca_summary.json", "w") as fh:
    json.dump({"exp": EXP, "n_units": n_units, "seconds": T_STOP - T_START,
               "n95_real": n95_real, "n95_null": n95_null,
               "pr_real": round(float(pr_real), 1),
               "pr_null": round(float(pr_null), 1)}, fh, indent=2)
print("wrote figures/pca_summary.json")
