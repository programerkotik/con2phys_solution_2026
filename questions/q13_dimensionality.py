"""Question 13: in which brain area is the dimensionality of neural activity highest?

Final variable: per mouse and area, k/n averaged over 500 random subsets of n
units (n = the smallest area of the mouse, after dropping flat units), where k
is the number of principal components needed to explain 80% of the variance
of the soft-normalised, trial-averaged, time-normalised stimulus → outcome
trajectory.

Run: ``python -m questions.q13_dimensionality --animal 1`` then ``--aggregate``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from con2phys import cli, io, plotting, stats
from con2phys.spikes import valid_segments
from questions.q13_figures import plot_animal, plot_size_curves

KEY = "q13"
QUESTION = "Q13. In which brain area is the dimensionality of neural activity highest?"


def trial_averaged_trajectory(trains: list[np.ndarray], starts: np.ndarray,
                              ends: np.ndarray, n_bins: int) -> np.ndarray:
    """Time-normalised firing rates averaged over trials, ``[n_bins × n_units]`` (spikes/s).

    Scientific rationale:
        Each segment is cut into the same number of half-open bins, so bin i
        is the same fraction of the stimulus-to-outcome interval in every
        trial. Averaging over trials keeps the activity locked to that interval
        and removes most private, trial-to-trial spiking noise.
    """
    edges = starts[:, None] + (ends - starts)[:, None] * np.linspace(0.0, 1.0, n_bins + 1)
    widths = (ends - starts)[:, None] / n_bins
    trajectory = np.empty((n_bins, len(trains)))
    for unit, train in enumerate(trains):
        cumulative = np.searchsorted(np.sort(train), edges, side="left")
        trajectory[:, unit] = (np.diff(cumulative, axis=1) / widths).mean(axis=0)
    return trajectory


def soft_normalize(trajectory: np.ndarray, soft_norm_hz: float) -> np.ndarray:
    """Divide each unit's trajectory by its range plus a constant (spikes/s).

    Scientific rationale:
        Soft normalisation (Churchland et al. 2012) stops high-rate units from
        dominating the variance, while weakly modulated units keep a small
        weight instead of being scaled up to unit variance as with z-scoring.
    """
    span = trajectory.max(axis=0) - trajectory.min(axis=0)
    return trajectory / (span + soft_norm_hz)


def explained_variance_curve(trajectory: np.ndarray) -> np.ndarray:
    """Cumulative fraction of variance explained by successive principal components.

    Scientific rationale:
        PCA of the per-unit-centred trajectory (rows = time bins, columns =
        units) finds the orthogonal population patterns of its time course;
        squared singular values are the variances along them.
    """
    centred = trajectory - trajectory.mean(axis=0)
    variances = np.linalg.svd(centred, compute_uv=False) ** 2
    return np.cumsum(variances) / variances.sum()


def pcs_for_variance(trajectory: np.ndarray, threshold: float) -> int:
    """Smallest number of principal components explaining ``threshold`` of the variance."""
    curve = explained_variance_curve(trajectory)
    return int(np.searchsorted(curve, threshold - 1e-9) + 1)  # tolerance for float rounding


def subsampled_dimensionality(trajectory: np.ndarray, n_units: int, n_subsets: int,
                              threshold: float, rng: np.random.Generator) -> np.ndarray:
    """k for random subsets of ``n_units`` units drawn without replacement.

    Scientific rationale:
        k grows with the number of units, so areas are compared at the same n.
        Drawing without replacement avoids duplicated (perfectly correlated)
        units, which would artificially lower k.
    """
    total = trajectory.shape[1]
    return np.array([pcs_for_variance(trajectory[:, rng.choice(total, n_units, replace=False)],
                                      threshold) for _ in range(n_subsets)])


def area_dimensionality(trajectory: np.ndarray, matched_n: int, settings: dict[str, Any],
                        rng: np.random.Generator) -> dict[str, Any]:
    """Matched-n dimensionality of one area and its subset-size curve.

    Scientific rationale:
        k/n is the fraction of PCs per unit needed for the variance threshold,
        which puts mice with different matched n on one scale. The subset-size
        curve shows whether the area ranking depends on the chosen n.
    """
    threshold = float(settings["variance_threshold"])
    n_subsets = int(settings["n_subsets"])
    k = subsampled_dimensionality(trajectory, matched_n, n_subsets, threshold, rng)
    curve = {str(n): float(np.mean(subsampled_dimensionality(
                 trajectory, n, n_subsets, threshold, rng)) / n)
             for n in settings["curve_subset_sizes"] if n <= trajectory.shape[1]}
    return {"n_units_used": int(trajectory.shape[1]), "mean_k": float(k.mean()),
            "dimensionality": float(k.mean() / matched_n),
            "k_counts": np.bincount(k, minlength=matched_n + 1)[1:].tolist(),
            "size_curve": curve}


def analyze_mouse(trains: list[np.ndarray], unit_areas: np.ndarray,
                  trials: io.TrialAnnotations, areas: list[int], settings: dict[str, Any],
                  rng: np.random.Generator) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    """Unit-matched trajectory dimensionality of every area for one animal.

    Returns:
        A JSON-serialisable record and the normalised trajectory of each area
        (non-flat units only), for plotting.

    Scientific rationale:
        All areas share the same trials, time bins and matched n, so within a
        mouse they differ only in their units' activity.
    """
    start_event, end_event = settings["segment"]
    starts, ends = trials.event(start_event), trials.event(end_event)
    keep = valid_segments(starts, ends)
    trajectory = soft_normalize(
        trial_averaged_trajectory(trains, starts[keep], ends[keep], int(settings["n_time_bins"])),
        float(settings["soft_norm_hz"]))
    non_flat = np.ptp(trajectory, axis=0) > 0
    by_area = {a: trajectory[:, (unit_areas == a) & non_flat] for a in areas}
    matched_n = min(t.shape[1] for t in by_area.values())
    record: dict[str, Any] = {
        "n_trials_total": int(len(starts)), "n_trials_used": int(keep.sum()),
        "durations_s": (ends - starts)[keep].tolist(), "matched_n": int(matched_n), "areas": {},
    }
    for area, area_trajectory in by_area.items():
        record["areas"][str(area)] = {"n_units": int((unit_areas == area).sum()),
                                      **area_dimensionality(area_trajectory, matched_n,
                                                            settings, rng)}
    return record, by_area


def run_animal(config: dict[str, Any], mouse: int) -> dict[str, Any]:
    """Dimensionality of every area of one animal, with its step figure."""
    settings = config[KEY]
    folder = io.mouse_dir(config, mouse)
    animal = io.load_spikes(folder)
    rng = np.random.default_rng([int(config["random_seed"]), mouse])
    record, trajectories = analyze_mouse(io.unit_spike_trains(animal),
                                         np.asarray(animal.unit_areas),
                                         io.load_trials(folder / "trial_data.xlsx"),
                                         [int(a) for a in config["areas"]], settings, rng)
    plot_animal(mouse, trajectories, record, settings, rng,
                cli.output_dir(config, KEY) / "figures" / f"mouse_{mouse:02d}.png",
                int(config["dpi"]))
    return record


def aggregate(config: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Across-mouse summary, decision and figures."""
    labels = [str(a) for a in config["areas"]]
    values = {m: [r["areas"][a]["dimensionality"] for a in labels] for m, r in records.items()}
    matrix = np.array(list(values.values()))
    summary = stats.summarize_conditions(matrix, labels, config, ranking="highest")
    folder = cli.output_dir(config, KEY) / "figures"
    plotting.plot_summary(matrix, labels, summary, "Dimensionality k/n (PCs per unit for 80%)",
                          "Q13", folder / "summary.png")
    plot_size_curves(records, [int(a) for a in labels], folder / "subset_size.png",
                     int(config["dpi"]))
    per_mouse = {str(m): {"matched_n": r["matched_n"], "n_trials_used": r["n_trials_used"],
                          "mean_k": {a: r["areas"][a]["mean_k"] for a in labels},
                          "n_units_used": {a: r["areas"][a]["n_units_used"] for a in labels}}
                 for m, r in records.items()}
    return cli.final_report(QUESTION, "mean over 500 matched-n unit subsets of k/n, k = PCs "
                            "for 80% variance of the trial-averaged stimulus→outcome trajectory",
                            "fraction of PCs per unit (dimensionless)", values, labels, summary,
                            per_mouse=per_mouse)


if __name__ == "__main__":
    cli.main(KEY, run_animal, aggregate, __doc__)
