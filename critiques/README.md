# Critiques — claim, test, current number

Each file here is one argument against one claim in Kristensen, Kesgin & Jörntell
(2024), *Comm Biol* 7:1043. Every file follows the same shape: reproduce their
number with their pipeline, change exactly one thing, report what happens, and
state the honest number that should replace theirs.

Run any of them cell by cell, or whole:

```bash
MPLBACKEND=Agg /home/hector/miniconda3/envs/neural_analysis/bin/python critiques/dimensionality.py
```

NPBI is excluded from A, B, D and E — its onsets are ~243 ms late and
unverifiable. It appears in C only, because C is a within-session contrast where
a constant offset cancels. See `../CLAUDE.md`.

Figures land in `figures/critique/`, summary tables next to them as CSV. Neither
is in git (`.gitignore` excludes `*.png`), so the numbers below are the record.

---

## Summary

| | claim | verdict | the number that matters |
|---|---|---|---|
| **A** `dimensionality.py` | spontaneous activity is extremely high-dimensional (50/27/55/42/28 PCs = 62–81% of available) | **not supported as stated** — the true dimensionality is low | PCs for 95%: real 38 vs rate-matched shuffle 40.2 (NPBN). Participation ratio at raw 1 ms: real 24.5 vs shuffle 27.5 |
| **B** `decoding_leakage.py` | spontaneous vs evoked F1 = 0.93, still climbing at PC #50 | **strongly inflated by CV leakage** | their pipeline, split over samples: **0.981**. Same data, whole traces held out: **0.634**. Leak worth 0.36–0.45 F1 across four sessions |
| **C** `bimodal.py` | visual input modulates the S1 response to tactile input (F1 = 0.73) | **not testable on the deposited data; no effect where it is** | in the only bimodal session, visual-partner decoding = **0.217** against a shuffled null of 0.250 |
| **D** `memory_or_drift.py` | pattern identity persists to 1000 ms — working-memory-like | **not reproducible from the Methods; no residual past ~400 ms** | their described pipeline gives 0.96–0.98 in *every* window incl. pre-stimulus; published values are 0.30–0.46. Trial-level decoding returns to chance by 400 ms |
| **E** `pseudoreplication.py` | t(2999) = −87.65, p < 0.001, d = 2.1 | **p reports the rerun count, not the evidence** | same data: p = 0.21 at 1 repetition, p = 1.6e-18 at 50. Per animal (n = 4): p = 0.139 |

---

## A — dimensionality

**Test.** Rotate each neuron's spike train by its own random offset. Rates,
bursting and ISI structure are preserved exactly; between-neuron coordination is
destroyed. Then recount PCs and compute the participation ratio.

| session | neurons | PCs for 95% real | shuffled | PR raw real | PR raw shuffled | PR paper-pipeline real | shuffled |
|---|---|---|---|---|---|---|---|
| NPBN | 61 | 38 | 40.2 | 24.5 | 27.5 | 5.2 | 17.3 |
| NPBK | 40 | 24 | 24.5 | 23.9 | 24.0 | 6.4 | 6.2 |
| NPBM | 62 | 48 | 50.0 | 38.8 | 39.8 | 15.1 | 22.0 |
| NPBO | 36 | 25 | 26.5 | 21.5 | 21.8 | 8.4 | 13.1 |

**Findings.**
- The headline count is reproduced (38/61 = 62%, in their 62–81% range) and is
  indistinguishable from a shuffle with no population structure at all.
- 99.5% of 1 ms bins are empty at 5.25 Hz. A flat spectrum is the prior
  expectation, not a finding.
- Fig 1F's collapse with coarser bins happens to the shuffle too.
- PCs-for-95% per neuron stays at 0.75–0.83 as neurons are added, so the metric
  tracks recording size.
- **The positive result:** under their own preprocessing the participation ratio
  does separate (NPBN 5.2 real vs 17.3 shuffled). Structure exists — roughly
  5–15 dimensions, not 42 — and it is slow enough to survive averaging 50
  sequential trials, which is the same structure critique D calls drift. NPBK
  shows no gap at all.

---

## B — decoding leakage

**Test.** Their split is random over 1 ms samples, after a 10 ms Gaussian.
Neighbouring samples correlate at r = 0.990. Split by trial instead.

| session | their split | own shuffled control | traces held out | own shuffled control | leak |
|---|---|---|---|---|---|
| NPBN | 0.981 | 0.496 | 0.634 | 0.508 | 0.359 |
| NPBK | 0.984 | 0.508 | 0.536 | 0.505 | 0.446 |
| NPBM | 0.994 | 0.503 | 0.610 | 0.508 | 0.389 |
| NPBO | 0.986 | 0.500 | 0.603 | 0.498 | 0.381 |

"Leak" = how much more the sample-wise scheme beats its own chance level than
the honest scheme beats its own. Each scheme gets its own shuffled control
because F1 chance is not 0.5 under both.

**Findings.**
- Running their actual pipeline (50 sequential trials averaged) gives **0.981**,
  above their published 0.93 — so the claim reproduces and then some.
- Closing the leak costs 0.35 F1. What survives, ~0.6, is the real
  spontaneous-vs-evoked difference and is the number to report.
- The PC sweep is the diagnostic panel: sample-wise F1 climbs monotonically to
  PC #40 (0.402 → 0.602), trial-wise peaks at 3 PCs (0.448) and then *declines*
  to 0.414. Their "F1 still rising at PC #50 means high-dimensional information"
  is what near-duplicate matching looks like, not what information looks like.

---

## C — the bimodal claim

**The data problem.** Of five deposited sessions, only NPBI has both modalities
on the same trial: 400 visual-only (vA–vD) and 400 visuo-tactile (vA–vD each
with an F5 shock). The paper describes four *tactile* patterns ± visual across
three experiments, 800 vs 800. That experiment is not in the deposited data.

**The timing.** NPBI's clock error is re-derived here from the data rather than
assumed: the population PSTH peaks at −185 ms against the labelled onset; after
the −243 ms correction it peaks at **+59 ms**, where an S1 shock response
belongs. The ± contrast is within-session, so a constant offset cancels.

**Findings** (70 units, chance 0.250, whole trials held out):

| test | result |
|---|---|
| +tactile vs −tactile, 0–200 ms *(positive control)* | **0.835** (chance 0.500) |
| visual partner, same F5 every trial, 0–200 ms | **0.217** |
| same, labels shuffled | 0.250 (SD 0.028) |
| same, pre-stimulus window | 0.280 |
| visual pattern with no touch at all, 0–200 ms | 0.248 |
| best single neuron | 0.290 (median 0.247) |

- The decoder works — it separates ±tactile at 0.835 on the same trials in the
  same window. It finds no trace of *which* visual pattern accompanied the shock.
- S1 does not separate the visual patterns even when presented alone (0.248).
- Their Fig 2E logic (population succeeds where single neurons fail) is not
  assessable here: the population score is itself at its shuffled null, so the
  comparison is between two failures.
- **Caveat kept in the file:** this is a trial-level test; their effect was
  measured after averaging 50 trials, so a small true effect could hide. What
  averaging cannot escape is critique D's objection to *sequential* averaging.

---

## D — memory or drift

**The reproduction failure.** Their Methods say data points were "split into
training and test sets (50/50)" after averaging 50 sequential trials. With 200
trials per pattern that leaves four traces per pattern, and "data points" can
mean milliseconds or traces. Both readings were run:

| window | Methods read as: split samples | published | read as: split traces |
|---|---|---|---|
| pre −300..0 | 0.962 | 0.303 | 0.132 |
| 0..200 | 0.799 | 0.455 | 0.188 |
| 200..400 | 0.969 | 0.363 | 0.227 |
| 400..600 | 0.976 | 0.337 | 0.177 |
| 600..800 | 0.973 | 0.329 | 0.170 |
| 800..1000 | 0.971 | 0.318 | 0.203 |

One reading saturates near 1.0 *in every window including the one before the
stimulus*; the other falls below chance. The published values sit between them.
**Fig 5 cannot be recovered from what the Methods say.** The trace-split reading
does reproduce the *shape* — peak just after onset, then decay — but not the
level, and with 2 traces per class in training it has almost no power.

**The honest measurement** (per-trial firing rates, LDA, whole trials held out,
NPBN, chance 0.250):

| window | accuracy |
|---|---|
| pre −300..0 | 0.238 |
| 0..200 | **0.318** |
| 200..400 | 0.292 |
| 400..600 | 0.245 |
| 600..800 | 0.249 |
| 800..1000 | 0.251 |

Pattern information is gone by 400 ms. There is no residual to explain at
1000 ms — in any of the four sessions (late-window decoding: 0.251, 0.251, 0.108
vs chance 0.125, 0.264).

**The two dull explanations, measured anyway:**
- **Drift is real and large.** R² predicting session time from the pre-stimulus
  population state: NPBN 0.377, NPBK 0.554, NPBM 0.523, NPBO 0.518. With onsets
  permuted it is −0.139, so the test is behaving.
- **Carryover is weak and inconsistent.** Decoding the *previous* trial's pattern
  from the pre-stimulus window: NPBN 0.282 (vs 0.250 for the impossible
  "next-pattern" control), but NPBK 0.242 vs 0.275 and NPBO 0.259 vs 0.257 —
  i.e. the impossible control scores as high as the real one. Read as a floor on
  the other numbers, not as a demonstration of carryover.
- Removing the drift axis from the late window changes nothing (0.251 → 0.256)
  because there was nothing there to remove.

**Known weakness of this file:** the leak-free version of *their* classifier
(random 10-trial groups, traces held out) sits at chance even in the 0–200 ms
window where an LDA finds 0.318. So that table shows their pipeline is
underpowered once the leak is closed — it does not independently establish
absence. The trial-level LDA is the measurement to quote.

---

## E — pseudoreplication

**Where their N comes from.** Their own Methods: 50 repetitions × 20 sub-windows
× 3 experiments = 3000. Rebuilt here with four sessions = 4000.

| test | n | t | p | d |
|---|---|---|---|---|
| their pooling, evoked vs prestimulus | 4000 | −8.83 | 1.6e-18 | −0.20 |
| their pooling, two prestimulus windows *(control)* | 4000 | 1.02 | 0.31 | 0.02 |
| per animal, evoked vs prestimulus | 4 | −2.00 | 0.139 | −1.00 |
| per animal, two prestimulus windows | 4 | 0.80 | 0.48 | 0.19 |

**The knob** — same data, same effect size, only the rerun count changes:

| repetitions | N | p (evoked vs pre) | d |
|---|---|---|---|
| 1 | 80 | 0.21 | −0.19 |
| 5 | 400 | 0.013 | −0.18 |
| 10 | 800 | 2.8e-06 | −0.24 |
| 50 | 4000 | 1.6e-18 | −0.20 |

**What is and is not claimed.** Their scheme does *not* manufacture effects from
nothing — the two-baseline control comes back p = 0.31, correctly. The narrower
and harder objection is that the p-value is set by how many times the loop ran.
One repetition is not significant; fifty repetitions of the same analysis on the
same four animals is p < 1e-18, and nothing was learned in between. Their
reported difference is worth 0.014 F1.

**Replacement:** average the repetitions away, keep one number per animal, then
a paired t-test, a mixed-effects model with animal as a random effect, or a
permutation test at the animal level.

---

## Still open

- **Confirm the NPBI design mismatch against the Figshare source.** The modality
  labels come from our own conversion (`convert_to_nwb.py` in the other repo,
  which reads `stim_type != 'visual'` out of the deposited stimulus CSVs). The
  claim "the deposited design is not the design the paper describes" should be
  checked against the raw files before it goes in writing.
- **Ask the lab for NPBI's missing run**, which is the only thing that would make
  that session usable for anything onset-locked.
- **Neuropixels 1.0 vs 2.0** — the raw metas say 1.0, the paper says 2.0. Worth
  a line in any write-up; not yet a critique file.
- **Unit inclusion.** `../extra_check.py` runs a stricter version of their
  responsiveness criterion (3 SD / 10 bins vs their 2 SD / 2 bins). At 2 ms bins
  and a few Hz, their threshold is closer to "a bin with two spikes in it" than
  to a test. Not yet written up as a critique.
- **Anesthesia.** Ketamine/xylazine UP/DOWN states are the obvious mechanism
  behind the drift in D and the slow structure in A. Currently an argument in
  prose only, with no measurement attached.
