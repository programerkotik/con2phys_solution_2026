"""Question 15: in which brain area is the neural signal most complex?

Final variable: per mouse and area, the plug-in Shannon entropy (bits per
10 ms bin) of each unit's spike counts within one stimulus → trial-end
segment, averaged over the area's units and then over trials.

Run: ``python -m questions.q15_entropy --animal 1`` then ``--aggregate``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

from con2phys import cli, io, plotting, stats
from con2phys.spikes import valid_segments
from questions.q15_figures import plot_animal, plot_rate_dependence

KEY = "q15"
QUESTION = "Q15. In which brain area is the neural signal most complex?"


def trial_bin_counts(sorted_train: np.ndarray, start: float, n_bins: int,
                     bin_s: float) -> np.ndarray:
    """Spike counts of one unit in consecutive half-open bins from a segment start.

    Scientific rationale:
        Half-open bins count every spike exactly once; spikes after the last
        complete bin (the incomplete remainder) are ignored.
    """
    edges = start + bin_s * np.arange(n_bins + 1)
    return np.diff(np.searchsorted(sorted_train, edges, side="left"))


def count_entropy_bits(counts: np.ndarray) -> float:
    """Plug-in Shannon entropy H = −Σ p_k log₂ p_k of a spike-count distribution, in bits.

    Scientific rationale:
        H is the average uncertainty about how many spikes fall in a bin: 0 if
        every bin holds the same count (e.g. a silent unit), larger when the
        counts are more variable.
    """
    p = np.bincount(counts) / len(counts)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() + 0.0)  # + 0.0 turns -0.0 into 0.0


def trial_entropies(trains: list[np.ndarray], starts: np.ndarray, ends: np.ndarray,
                    bin_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Entropy (bits per bin) and rate (spikes/s) of every unit in every trial, ``[n_trials × n_units]``.

    Scientific rationale:
        Entropy is computed within each trial; the rate over the same bins
        shows how strongly the entropy follows the firing rate.
    """
    n_bins = np.floor((ends - starts) / bin_s + 1e-9).astype(int)  # tolerance for float rounding
    entropy = np.empty((len(starts), len(trains)))
    rate = np.empty_like(entropy)
    for unit, train in enumerate(trains):
        train = np.sort(train)
        for trial, (start, n) in enumerate(zip(starts, n_bins)):
            counts = trial_bin_counts(train, start, n, bin_s)
            entropy[trial, unit] = count_entropy_bits(counts)
            rate[trial, unit] = counts.sum() / (n * bin_s)
    return entropy, rate


def analyze_mouse(trains: list[np.ndarray], unit_areas: np.ndarray, trials: io.TrialAnnotations,
                  areas: list[int], settings: dict[str, Any]) -> dict[str, Any]:
    """Trial-mean entropy of every area for one animal.

    Returns:
        Per area the mouse-level entropy and rate (units then trials averaged),
        the per-trial entropy, and per-unit trial-mean entropy and rate.

    Scientific rationale:
        Per trial, the mean over an area's units does not grow with the number
        of units; the mean over trials gives every trial the same weight,
        whatever its duration.
    """
    bin_s = float(settings["bin_s"])
    start_event, end_event = settings["segment"]
    starts, ends = trials.event(start_event), trials.event(end_event)
    keep = valid_segments(starts, ends) & (ends - starts >= bin_s)
    entropy, rate = trial_entropies(trains, starts[keep], ends[keep], bin_s)
    record: dict[str, Any] = {"n_trials_total": int(len(starts)), "n_trials_used": int(keep.sum()),
                              "starts_s": starts[keep].tolist(),
                              "durations_s": (ends - starts)[keep].tolist(), "areas": {}}
    for area in areas:
        units = unit_areas == area
        per_trial = entropy[:, units].mean(axis=1)
        record["areas"][str(area)] = {
            "n_units": int(units.sum()), "entropy_bits": float(per_trial.mean()),
            "rate_hz": float(rate[:, units].mean(axis=1).mean()),
            "trial_entropy_bits": per_trial.tolist(),
            "unit_entropy_bits": entropy[:, units].mean(axis=0).tolist(),
            "unit_rate_hz": rate[:, units].mean(axis=0).tolist(),
        }
    return record


def entropy_rate_correlation(unit_entropy: np.ndarray, unit_rate: np.ndarray) -> float:
    """Spearman ρ between units' trial-mean entropy and rate.

    Scientific rationale:
        At 10 ms bins the count entropy is close to the binary entropy of
        rate × bin, so ρ near 1 means the entropy ranking mostly re-measures
        the firing-rate ranking.
    """
    return float(spearmanr(unit_entropy, unit_rate).statistic)


def run_animal(config: dict[str, Any], mouse: int) -> dict[str, Any]:
    """Entropy of every area of one animal, with its step figure."""
    settings = config[KEY]
    folder = io.mouse_dir(config, mouse)
    animal = io.load_spikes(folder)
    trains = io.unit_spike_trains(animal)
    areas = [int(a) for a in config["areas"]]
    unit_areas = np.asarray(animal.unit_areas)
    record = analyze_mouse(trains, unit_areas, io.load_trials(folder / "trial_data.xlsx"),
                           areas, settings)
    plot_animal(mouse, trains, unit_areas, record, areas, float(settings["bin_s"]),
                cli.output_dir(config, KEY) / "figures" / f"mouse_{mouse:02d}.png",
                int(config["dpi"]))
    return record


def aggregate(config: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Across-mouse summary, decision and figures."""
    labels = [str(a) for a in config["areas"]]
    bin_s = float(config[KEY]["bin_s"])
    values = {m: [r["areas"][a]["entropy_bits"] for a in labels] for m, r in records.items()}
    matrix = np.array(list(values.values()))
    summary = stats.summarize_conditions(matrix, labels, config, ranking="highest")
    folder = cli.output_dir(config, KEY) / "figures"
    plotting.plot_summary(matrix, labels, summary, f"Entropy (bits per {bin_s * 1000:.0f} ms bin)",
                          "Q15", folder / "summary.png")
    plot_rate_dependence(records, labels, bin_s, folder / "entropy_vs_rate.png",
                         int(config["dpi"]))
    unit = {a: {k: np.concatenate([r["areas"][a][k] for r in records.values()])
                for k in ("unit_entropy_bits", "unit_rate_hz")} for a in labels}
    return cli.final_report(
        QUESTION, "plug-in Shannon entropy of each unit's 10 ms spike counts per stimulus→"
        "trial-end segment, averaged over units, then trials", "bits per 10 ms bin",
        values, labels, summary,
        unit_entropy_rate_spearman={a: entropy_rate_correlation(u["unit_entropy_bits"],
                                                                u["unit_rate_hz"])
                                    for a, u in unit.items()},
        mean_rate_hz_mouse_weighted={a: float(np.mean([r["areas"][a]["rate_hz"]
                                                       for r in records.values()]))
                                     for a in labels},
        n_trials_used={str(m): r["n_trials_used"] for m, r in records.items()})


if __name__ == "__main__":
    cli.main(KEY, run_animal, aggregate, __doc__)
