"""Question 11: which brain area contains most information about variable A?

Final variable: per mouse and area, the mean over units of each unit's
*excess* Skaggs information about A (bits/spike): the unit's information on
the in-trial clock minus the mean of its 1000 circular-shift null scores.
Inter-trial intervals are excluded; the whole trial is the analysis window,
because the relevant segment for A is not known a priori.

Run: ``python -m questions.q11_information --animal 1`` then ``--aggregate``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from con2phys import cli, io, plotting, stats
from con2phys.spikes import draw_circular_shifts, valid_segments
from questions.q11_figures import plot_animal, plot_distributions

KEY = "q11"
QUESTION = "Q11. Which brain area contains most information about variable A?"

# ---------------------------------------------------------------------------
# Copied unchanged from the proof_of_concept repository:
# skaggs_information_batch  <- src/statistics/information_scores.py
# empirical_p_values  <- src/utils/general_utils.py
# _bin_indicator_spectra, shuffled_binned_sums  <- src/utils/cell_detection_utils.py
# ---------------------------------------------------------------------------


def skaggs_information_batch(rates: np.ndarray, occupancy: np.ndarray,
                             valid_bins: np.ndarray) -> np.ndarray:
    """Vectorised Skaggs information (bits per unit activity) over many tuning curves.

    Args:
        rates: Mean activity per bin, shape ``(..., n_bins)``; negative values are
            clipped to zero.
        occupancy: Occupancy per bin, shape ``(n_bins,)``, shared by every curve
            (the behaviour is never shuffled).
        valid_bins: Boolean mask ``(n_bins,)`` of bins that enter the sum.

    Returns:
        Information per curve, shape ``rates.shape[:-1]``; NaN when no bin is
        valid, 0 when the occupancy-weighted mean activity is zero.

    Scientific rationale:
        I = Σ p_i (λ_i/λ̄) log₂(λ_i/λ̄): how much knowing the bin (here the
        value of A) tells about the next spike. Evaluated for the real curve
        and all shuffled curves at once, so the null is computed exactly like
        the observed statistic.
    """
    rates = np.maximum(np.asarray(rates, dtype=np.float64), 0.0)
    occupancy = np.asarray(occupancy, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_bins, dtype=bool).reshape(-1) & np.isfinite(occupancy) & (occupancy > 0)
    if not np.any(valid):
        return np.full(rates.shape[:-1], np.nan)
    p_i = occupancy[valid] / occupancy[valid].sum()
    lam = rates[..., valid]
    lam = np.where(np.isfinite(lam), lam, 0.0)
    lam_mean = np.sum(lam * p_i, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lam_mean > 0, lam / lam_mean, 0.0)
        terms = np.where(ratio > 0, p_i * ratio * np.log2(ratio), 0.0)
    info = terms.sum(axis=-1)
    info[lam_mean[..., 0] <= 0] = 0.0
    return info


def empirical_p_values(real_scores: np.ndarray, shuffle_scores: np.ndarray) -> np.ndarray:
    """Upper-tail permutation p-values ``(k + 1) / (n + 1)`` against each row's own null.

    Scientific rationale:
        Counting the observed value as one member of the null (Phipson &
        Smyth 2010) keeps the p-value valid and never zero.
    """
    real = np.asarray(real_scores, dtype=np.float64).reshape(-1)
    null = np.asarray(shuffle_scores, dtype=np.float64)
    finite = np.isfinite(null)
    n_valid = finite.sum(axis=1)
    p = ((finite & (null >= real[:, None])).sum(axis=1) + 1.0) / (n_valid + 1.0)
    p[~np.isfinite(real) | (n_valid == 0)] = np.nan
    return p


def _bin_indicator_spectra(bin_index: np.ndarray, n_bins: int) -> np.ndarray:
    """FFT of the 0/1 indicator of every bin along the frame axis."""
    indicators = np.zeros((int(n_bins), bin_index.size), dtype=np.float64)
    keep = bin_index >= 0
    indicators[bin_index[keep], np.flatnonzero(keep)] = 1.0
    return np.fft.rfft(indicators, axis=1)


def shuffled_binned_sums(traces: np.ndarray, bin_index: np.ndarray, n_bins: int,
                         shifts: np.ndarray,
                         indicator_spectra: np.ndarray | None = None) -> np.ndarray:
    """Bin sums of circularly shifted traces, ``(n_units, n_shifts, n_bins)``.

    Scientific rationale:
        ``Σ_t roll(s, k)[t] 1[b_t = j]`` is the circular cross-correlation of
        the trace with bin j's indicator, so all shifts come from one FFT per
        unit and bin; equal to binning ``np.roll(trace, k)``.
    """
    traces = np.asarray(traces, dtype=np.float64)
    n_frames = traces.shape[1]
    spectra = (_bin_indicator_spectra(np.asarray(bin_index), n_bins)
               if indicator_spectra is None else indicator_spectra)
    signal_spectra = np.conj(np.fft.rfft(traces, axis=1))
    out = np.empty((traces.shape[0], shifts.shape[1], int(n_bins)), dtype=np.float64)
    for j in range(int(n_bins)):
        correlation = np.fft.irfft(signal_spectra * spectra[j], n=n_frames, axis=1)
        out[:, :, j] = np.take_along_axis(correlation, shifts, axis=1)
    return out


# ---------------------------------------------------------------------------
# Question 11
# ---------------------------------------------------------------------------


def valid_trial_mask(trials: io.TrialAnnotations) -> np.ndarray:
    """Trials with a finite start, a later finite end and a label for A.

    Scientific rationale:
        The analysis window is the whole trial, so missing stimulus or
        outcome times do not matter.
    """
    labels = np.asarray(trials.variable_a, dtype=float)
    return valid_segments(trials.event("trial_start"), trials.event("trial_end")) \
        & np.isfinite(labels)


def in_trial_bins(starts: np.ndarray, ends: np.ndarray, labels: np.ndarray,
                  bin_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Cut every trial into equal bins and join them into an in-trial clock.

    Returns:
        Start time of every bin and the A label of its trial, in trial order.
        The incomplete last bin of each trial is dropped.

    Scientific rationale:
        Joining only in-trial bins removes the inter-trial intervals from the
        rate map and from the circular-shift null; no bin straddles two trials.
    """
    per_trial = np.floor((ends - starts) / bin_s + 1e-9).astype(int)  # float-rounding tolerance
    trial_of_bin = np.repeat(np.arange(len(starts)), per_trial)
    first_bin = np.repeat(np.cumsum(per_trial) - per_trial, per_trial)
    offsets = np.arange(per_trial.sum()) - first_bin
    return starts[trial_of_bin] + offsets * bin_s, labels[trial_of_bin]


def binned_spike_counts(trains: list[np.ndarray], bin_starts: np.ndarray,
                        bin_s: float) -> np.ndarray:
    """Spike count of every unit in every half-open bin ``[start, start + bin_s)``."""
    counts = np.empty((len(trains), len(bin_starts)))
    for unit, train in enumerate(trains):
        times = np.sort(train)
        counts[unit] = (np.searchsorted(times, bin_starts + bin_s, side="left")
                        - np.searchsorted(times, bin_starts, side="left"))
    return counts


def label_information(counts: np.ndarray, labels: np.ndarray, bin_s: float, n_shuffles: int,
                      min_shift_bins: int, rng: np.random.Generator,
                      chunk_units: int) -> dict[str, np.ndarray]:
    """Skaggs information about a binary label and its circular-shift null.

    Args:
        counts: ``(n_units, n_bins)`` spike counts on the in-trial clock.
        labels: Label (0 or 1) of every bin.
        bin_s: Bin width (s), to express rates in spikes/s.
        n_shuffles: Circular shifts.
        min_shift_bins: Minimum shift, in bins.
        rng: Seeded generator; one set of offsets is shared by all units.
        chunk_units: Units per FFT batch (bounds memory).

    Returns:
        ``information`` and ``p_value`` ``(n_units,)``, ``null``
        ``(n_units, n_shuffles)``, ``rate_a0`` / ``rate_a1`` (spikes/s),
        ``n_spikes`` and ``occupancy_s`` ``(2,)``. Silent units get NaN.

    Scientific rationale:
        Shifting the spike train while keeping labels and occupancy fixed keeps
        the unit's rate and temporal structure but breaks its alignment with
        A, so the null mean is the unit's own small-sample bias.
    """
    labels = np.asarray(labels).astype(int)
    if not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("labels must be 0 or 1")
    occupancy = np.bincount(labels, minlength=2).astype(float)
    valid = occupancy > 0
    seconds = occupancy * bin_s
    sums = np.stack([counts[:, labels == a].sum(axis=1) for a in (0, 1)], axis=1)
    rates = sums / seconds
    information = skaggs_information_batch(rates, occupancy, valid)
    shifts = np.repeat(draw_circular_shifts(1, n_shuffles, labels.size, min_shift_bins, rng),
                       len(counts), axis=0)
    spectra = _bin_indicator_spectra(labels, 2)
    null = np.empty((len(counts), n_shuffles))
    for start in range(0, len(counts), chunk_units):
        stop = start + chunk_units
        shuffled = shuffled_binned_sums(counts[start:stop], labels, 2, shifts[start:stop], spectra)
        null[start:stop] = skaggs_information_batch(shuffled / seconds, occupancy, valid)
    silent = sums.sum(axis=1) == 0
    information[silent] = np.nan
    null[silent] = np.nan
    return {"information": information, "null": null,
            "p_value": empirical_p_values(information, null),
            "rate_a0": rates[:, 0], "rate_a1": rates[:, 1],
            "n_spikes": sums.sum(axis=1), "occupancy_s": seconds}


def mean_excess(real: np.ndarray, null: np.ndarray) -> float:
    """Mean over (non-silent) units of real information minus the unit's null mean.

    Scientific rationale:
        Each unit's excess has expected value 0 without information, so the
        mean over units is unbiased (the median is not: the null is skewed).
    """
    finite = np.isfinite(real)
    if not finite.any():
        return float("nan")
    return float((real[finite] - null[finite].mean(axis=1)).mean())


def analyze_mouse(trains: list[np.ndarray], unit_ids: np.ndarray, unit_areas: np.ndarray,
                  trials: io.TrialAnnotations, areas: list[int], settings: dict[str, Any],
                  rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Information about A and its null for every unit of one animal.

    Returns:
        Flat arrays keyed ``"{area}__{name}"`` plus animal-level trial counts,
        ready for ``np.savez``. ``null_one`` holds one random null score per
        unit, for display.

    Scientific rationale:
        All units share the same in-trial clock and labels, so areas are
        compared on exactly the same trials.
    """
    keep = valid_trial_mask(trials)
    starts = trials.event("trial_start")[keep]
    ends = trials.event("trial_end")[keep]
    labels = np.asarray(trials.variable_a, dtype=float)[keep]
    bin_s = float(settings["bin_s"])
    bin_starts, bin_labels = in_trial_bins(starts, ends, labels, bin_s)
    counts = binned_spike_counts(trains, bin_starts, bin_s)
    n_shuffles = int(settings["n_shuffles"])
    result = label_information(counts, bin_labels, bin_s, n_shuffles,
                               int(round(float(settings["min_shift_s"]) / bin_s)), rng,
                               int(settings["chunk_units"]))
    result["null_one"] = result["null"][np.arange(len(counts)),
                                        rng.integers(n_shuffles, size=len(counts))]
    result["unit_ids"] = np.asarray(unit_ids)
    occupancy = result.pop("occupancy_s")
    record = {"n_trials_total": len(keep), "n_trials_used": int(keep.sum()),
              "n_trials_a1": int(labels.sum()), "occupancy_s": occupancy,
              "n_overlapping_trials": int(np.sum(starts[1:] < ends[:-1]))}
    for area in areas:
        in_area = np.asarray(unit_areas) == area
        record |= {f"{area}__{name}": values[in_area] for name, values in result.items()}
    return record


def by_area(saved: Any, areas: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """Split ``{area}__{name}`` arrays into one dict per area."""
    return {area: {key.split("__", 1)[1]: np.asarray(saved[key]) for key in saved
                   if key.startswith(f"{area}__")} for area in areas}


def run_animal(config: dict[str, Any], mouse: int) -> dict[str, Any]:
    """Per-unit information and null of one animal (arrays saved as ``.npz``)."""
    settings = config[KEY]
    areas = [int(a) for a in config["areas"]]
    folder = io.mouse_dir(config, mouse)
    animal = io.load_spikes(folder)
    rng = np.random.default_rng([int(config["random_seed"]), mouse])
    arrays = analyze_mouse(io.unit_spike_trains(animal), animal.unit_ids, animal.unit_areas,
                           io.load_trials(folder / "trial_data.xlsx"), areas, settings, rng)
    npz = cli.animal_path(config, KEY, mouse, "npz")
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    split = by_area(arrays, areas)
    plot_animal(mouse, split, cli.output_dir(config, KEY) / "figures" / f"mouse_{mouse:02d}.png",
                int(config["dpi"]))
    alpha = float(config["significance_level"])
    record: dict[str, Any] = {k: arrays[k] for k in ("n_trials_total", "n_trials_used",
                                                     "n_trials_a1", "n_overlapping_trials")}
    record["occupancy_s"] = arrays["occupancy_s"].tolist()
    record["areas"] = {}
    for area, data in split.items():
        scored = np.isfinite(data["p_value"])
        record["areas"][str(area)] = {
            "n_units": int(len(data["unit_ids"])), "n_silent": int((~scored).sum()),
            "mean_excess": mean_excess(data["information"], data["null"]),
            "fraction_p_below_alpha": (float(np.mean(data["p_value"][scored] < alpha))
                                       if scored.any() else float("nan"))}
    return record


def aggregate(config: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Across-mouse summary, decision and figures."""
    labels = [str(a) for a in config["areas"]]
    values = {m: [r["areas"][a]["mean_excess"] for a in labels] for m, r in records.items()}
    used = {m: v for m, v in values.items() if np.all(np.isfinite(v))}
    matrix = np.array(list(used.values()))
    summary = stats.summarize_conditions(matrix, labels, config, ranking="highest",
                                         reference=0.0)
    folder = cli.output_dir(config, KEY) / "figures"
    plotting.plot_summary(matrix, labels, summary,
                          "Excess information about A (bits/spike)", "Q11",
                          folder / "summary.png", reference=0.0)
    plot_distributions(_unit_table(config, list(used), [int(a) for a in labels]),
                       [int(a) for a in labels], folder / "unit_distributions.png",
                       int(config["dpi"]))
    return cli.final_report(
        QUESTION, "mean over units of (Skaggs information about A on the in-trial clock − "
        "mean of 1000 circular-shift null scores)", "bits/spike", used, labels, summary,
        dropped_mice=sorted(set(values) - set(used)),
        n_trials_used={str(m): r["n_trials_used"] for m, r in records.items()},
        n_silent_units={a: int(sum(r["areas"][a]["n_silent"] for r in records.values()))
                        for a in labels})


def _unit_table(config: dict[str, Any], mice: list[int], areas: list[int]
                ) -> list[dict[str, Any]]:
    """One row per unit (area, information, null mean, one null draw, spikes) for figures."""
    rows = []
    for mouse in mice:
        with np.load(cli.animal_path(config, KEY, mouse, "npz")) as saved:
            for area, data in by_area(saved, areas).items():
                null_mean = np.full(len(data["unit_ids"]), np.nan)  # silent units stay NaN
                finite = np.isfinite(data["null"]).any(axis=1)
                null_mean[finite] = data["null"][finite].mean(axis=1)
                rows += [{"area": area, "information": float(data["information"][i]),
                          "null_one": float(data["null_one"][i]),
                          "null_mean": float(null_mean[i]),
                          "n_spikes": float(data["n_spikes"][i])}
                         for i in range(len(data["unit_ids"]))]
    return rows


if __name__ == "__main__":
    cli.main(KEY, run_animal, aggregate, __doc__)
