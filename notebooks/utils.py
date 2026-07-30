"""
Shared loading or binning steps used to preform analysis on the dataset 

Used by:
  - `psth_all_sessions.py` — one PSTH figure per session (NPBK, NPBM, NPBN, NPBO)
  - `lda.py`              — LDA on per-trial spike counts for one session (NPBM)


"""
import pandas as pd
from pynwb import NWBHDF5IO
import numpy as np
from sklearn.decomposition import PCA

def load_data(exp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open `nwb/{exp}.nwb` and return its units and trials tables as DataFrames.

    `exp` is the experiment name, e.g. "NPBM". Both tables are converted to
    pandas *inside* the `with` block, so the numbers are copied into memory
    before the HDF5 file closes and the caller gets plain DataFrames it can keep
    using after this returns.
    """

    with NWBHDF5IO(f"nwb/{exp}.nwb", "r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        trials = nwb.trials.to_dataframe()
        return units, trials

def extract_data(units: pd.DataFrame, trials: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pulls the spike trains, onsets times, offset times,
    modality, stimulus labels, and tactile partner from the units and trials
    tables for ONE experiment. Returns them in a tuple.

    Returns, in order:
      units_good — the units table with the `mua` clusters dropped, i.e. only
                   the manually curated `good` clusters survive. Every later
                   count is per *good* unit, so this row count is what the
                   notebooks print as "N good units".
      onsets     — trial start times in seconds on the session clock, one per
                   trial. This is the alignment point for everything downstream.
      offsets    — trial stop times, same clock. Neither notebook uses these
                   right now (`psth_all_sessions.py` draws its red offset line
                   from the trials table's `duration_s` column instead), but
                   they are returned so the caller can check onset/offset
                   spacing without re-reading the file.
      modality   — "tactile" / "visual" per trial, as str.
      labels     — the stimulus pattern per trial ("F5", "vA", ...), as str.
      partner    — for a paired visuo-tactile trial, the pattern delivered in
                   the *other* modality; empty string when the trial was
                   unpaired. NaN is filled with "" first so the caller can test
                   `partner == ""` rather than having to handle NaN.
    """

    units_good = units[units["quality"] == "good"]
    onsets = trials["start_time"].to_numpy()
    offsets = trials["stop_time"].to_numpy()
    modality = trials["modality"].astype(str)
    labels = trials["stimulus"].astype(str)
    partner = trials["tactile_with"].fillna("").astype(str)

    return units_good,onsets,offsets,modality,labels,partner



def get_psth(units_good: pd.DataFrame, onsets: np.ndarray, bin_edges: np.ndarray, pre_s: float, post_s: float) -> np.ndarray:
    """Bin every good unit's spike train around every stimulus onset.

    Returns one array of shape (n_units, n_trials, n_bins) holding raw spike
    counts per bin — not rates, not means, nothing averaged yet. The caller does
    all the averaging, so this array is the one object every later number in
    `psth_all_sessions.py` is derived from:
      heatmaps      = mean over the trials of one class      -> (n_classes, n_units, n_bins)
      means         = that heatmap averaged over units, / bin_s -> Hz
      pop_per_trial = mean over units, kept per trial, / bin_s

    Arguments:
      units_good — the curated units table from `extract_data`; the loop reads
                   its `spike_times` column, one ragged array of spike times per
                   unit, in seconds on the session clock.
      onsets     — the alignment points, also from `extract_data`.
      bin_edges  — the bin edges *relative to onset*, in seconds, passed
                   straight to `np.histogram`. n_bins is one less than the
                   number of edges.
      pre_s/post_s — the window half-widths in seconds. These only pre-filter
                   the spikes to the window before histogramming; they do not
                   define the bins. If they disagree with `bin_edges` the bins
                   outside [-pre_s, post_s] simply come out empty rather than
                   raising, so the two must be kept consistent by the caller.

    `spikes - onset` re-expresses each spike as a time *relative to that trial's
    stimulus onset*, so the same bin index means the same latency on every
    trial; that realignment is the whole point of the function.


    """
    n_bins=len(bin_edges)-1
    binned_per_neuron = np.zeros((len(units_good), len(onsets), n_bins), dtype=np.uint8)
    # outer loop over units, inner over trials, so the array is filled in the
    # (unit, trial, bin) order it is declared in
    for row, spikes in enumerate(units_good["spike_times"]):
        for i, onset in enumerate(onsets):
            spikes_aligned = spikes - onset
            spikes_in_window = spikes_aligned[(spikes_aligned >= -pre_s)
                                              & (spikes_aligned <= post_s)]
            binned_per_neuron[row, i, :] = np.histogram(spikes_in_window,
                                                        bins=bin_edges)[0]

    return binned_per_neuron
