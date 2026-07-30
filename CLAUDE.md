# Visuotactile Andani Project — context

Reanalysis and critique of:

> Kristensen, Kesgin & Jörntell (2024). *High-dimensional cortical signals reveal
> rich bimodal and working memory-like representations among S1 neuron
> populations.* Communications Biology 7:1043.
> Full PDF in the repo root; plain text in the conversion repo as `article.txt`.

**Goal:** Hector's own reanalysis paper. The critiques in `critiques/` are the
motivation section; the deliverable is a better analysis of the same data. So
every critique must (1) reproduce the paper's own number with the paper's own
pipeline, then (2) show what changes when the flaw is fixed, then (3) state the
honest number that should replace it.

## The two repos

| repo | what lives there |
|---|---|
| `~/Projects/Visuotactile_Andani_Project` (here) | analysis, PSTHs, `critiques/` |
| `~/Projects/Tutorials/ANDANI/Wokrshop_Dataset` | NWB conversion, raw-data and dataset-integrity checks, `HANDOFF_LFP.md`, the slides |

Raw LFP is on the T9 drive: `/run/media/hector/T9/Henrik_Datasets/Sofie/<EXP>`.
`HANDOFF_LFP.md` in the conversion repo is the reference for anything about raw
files, runs, probes or LFP — read it before touching raw data.

## The paper's three claims

1. **Spontaneous activity is extremely high-dimensional.** 50/27/55/42/28 PCs for
   95% of the variance = 62–81% of all available PCs. Fig 1F: dimensionality
   collapses with coarser bins, used to argue against calcium imaging.
2. **Rich bimodal representation.** Spontaneous vs evoked F1 = 0.93 tactile /
   0.81 visual, still climbing at PC #50; visual input modulates tactile
   responses (F1 = 0.73 vs 0.5 shuffled, Fig 2); fine visual pattern nuances
   separable at population but not single-neuron level (Fig 3).
3. **Working memory-like residual.** Tactile pattern identity decodable to
   1000 ms post-stimulus: pre-stim F1 = 0.303 → 0.455 (1–200 ms) → 0.318
   (801–1000 ms). "Never returns to the spontaneous baseline" (Fig 5).

Their pipeline: 1 ms bins → average **50 sequential trials** → 10 ms Gaussian →
PCA on the 300 ms pre-stimulus window (95% variance) → kNN k=5, 50/50 split,
50 repeats → F1.

## The data

Five sessions in `nwb/`, converted by `convert_to_nwb.py` in the other repo.
Units table holds every cluster with `quality` = `good` (curated) or `mua`.

| exp | trials | good units | conditions | span |
|---|---|---|---|---|
| NPBI | 800 | 70 | visual vA–vD (100 each) **+ visuo-tactile vA–vD+F5 (100 each)** | 1316 s |
| NPBK | 800 | 40 | visual vA–vD (200 each) | 2714 s |
| NPBM | 1600 | 62 | tactile F5/F10/F20/F∞ **and** visual vA–vD (~200 each) | 6712 s |
| NPBN | 800 | 61 | tactile F5/F10/F20/F∞ (200 each) | 2500 s |
| NPBO | 800 | 36 | tactile F5/F10/F20/F∞ (200 each) | 2508 s |

Trials columns: `start_time, stop_time, modality, stimulus, tactile_with,
duration_s, onset_index`. Median inter-stimulus interval 1.6–2.9 s.

### Established data facts (do not re-derive)

- **NPBI's onsets are ~243 ms late and cannot be fixed.** Measured 2026-07-29:
  PSTH peak −185 ms vs +58 ms for the same F5 pattern in NPBN/NPBO;
  cross-correlation −244/−242 ms. All four NPBI LF files have zero stimulation
  artifact at the labelled onsets at any lag, and channel 384 (SY sync) is
  constant 0, so there is no TTL to check against. NPBI's spikes span 2245 s,
  matching neither run on the drive (5284 s, 4460 s) — the sorted run is a third
  recording nobody has. **Exclude NPBI from anything onset-locked, and from
  spontaneous-activity analyses too** (its "pre-stimulus" windows contain
  stimulus-driven activity). The one exception is a *within-session ±condition
  contrast*, where a constant offset applies to both conditions and cancels —
  see `critiques/bimodal.py`.
- **NPBI is the only session with combined bimodal trials.** Claim #2 therefore
  rests entirely on the session with the broken clock. The deposited design also
  does not match the paper's text: the paper describes four *tactile* patterns
  ± visual across three experiments (800+800 stimuli); NPBI is the converse —
  four *visual* patterns ± one F5 shock, 400+400.
- **The probes are Neuropixels 1.0, not 2.0.** Raw metas say `imDatPrb_type=0`,
  `PRB_1_4_0480_1_C`. The paper's Methods say 2.0. Trust the files.
- **Measured stimulus durations** (recovered from stimulation artifacts, not the
  paper): F5 0.283 s, F10 0.290 s, F20 0.321 s, F∞ **0.183 s**, vA–vD 0.215 s,
  visuo-tactile 0.283 s. The paper only says "200 to 340 ms"; a flat 340 ms
  overstates F∞ by 157 ms.
- NPBK and NPBM recorded **two probes**; which is S1 is unsettled, so both are
  stored (`lfp_imec0`, `lfp_imec1`). Always key off `imDatPrb_sn` — the
  imec0/imec1 assignment swaps between experiments.
- **NPBI and NPBO both have a run named `secondPub_g0`.** File tags alone are
  ambiguous. Always put the experiment name in figure titles and filenames.

## The critiques

Each file in `critiques/` is one argument, self-contained, run cell by cell.

| file | claim attacked | one-line argument |
|---|---|---|
| `dimensionality.py` | #1 | the 95% PC count is large whenever the spectrum is flat, and independent noise is flat; a rate-matched shuffle scores the same |
| `decoding_leakage.py` | #2 | random splits over 1 ms samples leak across their own 10 ms smoothing kernel; split by trial instead |
| `bimodal.py` | #2 | the ±visual contrast tested on the only session that has it, shift-corrected, plus what the deposited data cannot support |
| `memory_or_drift.py` | #3 | carryover from the previous trial and slow session drift both produce their result without any memory |
| `pseudoreplication.py` | all | their `t(2999)` comes from 50 CV repetitions of the same 5 animals, not 3000 independent observations |

`critiques/README.md` is the index: claim → test → current number.

## Code style (important)

Hector runs these files cell by cell in the VS Code interactive window and needs
to stop mid-way and inspect every intermediate. Rejected outright:

- a function call inside another function call — `dimensions(build_matrix(x))`
  must be two lines with a named intermediate
- one project function calling another project function; ops are flat and
  independent, and plotting functions receive **only precomputed arrays**
- list comprehensions that assemble a DataFrame, `**dict(zip(...))`, a helper
  defined inside another function
- a `for` loop over the five sessions — each session gets its own cell so it has
  its own story
- hand-rolled numerics where a library exists: sklearn `PCA`, not `np.linalg`

Also: explain what a transformation *does* in words next to it (e.g. what
exactly a "shuffle" permutes), not just its name. Long and obvious beats short
and clever.

## Running things

```bash
/home/hector/miniconda3/envs/neural_analysis/bin/python critiques/dimensionality.py
```

`conda run -n neural_analysis python` swallows stdout here — use the interpreter
path directly. Add `MPLBACKEND=Agg` when running headless.

Every critique file resolves its paths from a `ROOT` constant computed at the top
of the file, so it runs from the repo root, from `critiques/`, or from the VS Code
interactive window (which has no `__file__`). **Don't add `os.chdir` to fix a path
problem** — it moves the process into `critiques/` and breaks `ROOT / "nwb"` for
everything downstream. Use `ROOT / ...` instead.

Gotchas:
- `figures/critique/dimensionality.csv` has a `kind` column whose values are
  `real` and `shuffled`. An older version wrote `null`, which pandas silently
  reads back as NaN — pass `keep_default_na=False` if you meet an old file.
- `nwb/NPBM.nwb` is 17 GB and `NPBK.nwb` 5.9 GB, but that is LFP; loading only
  `units`/`trials` is fast (~0.1 s), so no caching is needed for spike analyses.
- `.gitignore` excludes `*.nwb`, `*.png`, `*.npz`, `*.npy` — figures are not in
  git, so numbers quoted in prose must also live in `critiques/README.md`.
