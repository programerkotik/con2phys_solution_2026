"""Question 4: in which brain area are pairwise spike-train interactions strongest at 100 ms?

Final variable: per mouse and area, the mean over all unordered unit pairs of
the zero-lag Pearson correlation between the units' spike counts in 100 ms
bins over the whole recording (the "spike-count correlation", r, unitless).

Run: ``python -m questions.q04_spike_correlation --animal 1`` then ``--aggregate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from con2phys import cli, io, plotting, stats

KEY = "q04"
QUESTION = ("Q4. In which brain area are pairwise spike train interactions strongest "
            "at the 100 ms timescale?")


def bin_area(trains: list[np.ndarray], duration_s: float, bin_s: float) -> Any:
    """Bin the spike trains of one area over the whole recording (Elephant).

    Args:
        trains: Spike times (s) of each unit.
        duration_s: Recording duration (s).
        bin_s: Bin width (s).

    Returns:
        ``elephant.conversion.BinnedSpikeTrain`` over the complete bins from
        0 s; spikes outside them are ignored.

    Scientific rationale:
        A 100 ms bin is the timescale the question names: two units interact
        at this scale if their spike counts in the same 100 ms windows
        co-vary.
    """
    import neo
    import quantities as pq
    from elephant.conversion import BinnedSpikeTrain

    t_stop = int(np.floor(duration_s / bin_s + 1e-9)) * bin_s
    spike_trains = []
    for train in trains:
        times = np.sort(np.asarray(train, dtype=float))
        times = times[(times >= 0) & (times < t_stop)]
        spike_trains.append(neo.SpikeTrain(times * pq.s, t_start=0 * pq.s, t_stop=t_stop * pq.s))
    return BinnedSpikeTrain(spike_trains, bin_size=bin_s * pq.s, t_start=0 * pq.s,
                            t_stop=t_stop * pq.s)


def pairwise_correlations(binned: Any) -> tuple[np.ndarray, np.ndarray]:
    """Zero-lag Pearson r of every unordered unit pair (Elephant).

    Args:
        binned: ``BinnedSpikeTrain`` of one area (output of ``bin_area``).

    Returns:
        The full ``[n_units × n_units]`` correlation matrix and the
        upper-triangle pair values (NaN for pairs involving a silent unit).

    Scientific rationale:
        Pearson r of binned counts is the standard spike-count correlation:
        independent of each unit's rate scale and signed, so co-activation
        (r > 0) and opposition (r < 0) are both kept.
    """
    from elephant.spike_train_correlation import correlation_coefficient

    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.atleast_2d(correlation_coefficient(binned))
    return matrix, matrix[np.triu_indices(len(matrix), k=1)]


def run_animal(config: dict[str, Any], mouse: int) -> dict[str, Any]:
    """Pairwise 100 ms spike-count correlations of every area of one animal."""
    folder = io.mouse_dir(config, mouse)
    bin_s = float(config[KEY]["bin_s"])
    animal = io.load_spikes(folder)
    duration_s = io.recording_duration_s(folder, float(config["lfp_sampling_rate_hz"]))
    io.report_out_of_range_spikes(mouse, animal.spikes_s, duration_s)
    trains = io.unit_spike_trains(animal)
    record: dict[str, Any] = {"duration_s": duration_s, "areas": {}}
    example: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for area in config["areas"]:
        units = np.flatnonzero(animal.unit_areas == area)
        binned = bin_area([trains[i] for i in units], duration_s, bin_s)
        matrix, pairs = pairwise_correlations(binned)
        finite = pairs[np.isfinite(pairs)]
        record["areas"][str(area)] = {
            "mean_r": float(finite.mean()), "n_units": int(len(units)),
            "n_pairs": int(pairs.size), "n_nan_pairs": int(pairs.size - finite.size),
            "pair_r": finite.tolist()}
        if mouse == int(config["example_mouse"]):
            example[area] = (binned.to_array(), matrix)
    if example:
        plot_example(config, mouse, example, record, bin_s)
    return record


def aggregate(config: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Across-mouse summary, decision and figures."""
    labels = [str(a) for a in config["areas"]]
    values = {m: [r["areas"][a]["mean_r"] for a in labels] for m, r in records.items()}
    matrix = np.array(list(values.values()))
    summary = stats.summarize_conditions(matrix, labels, config, ranking="highest")
    folder = cli.output_dir(config, KEY) / "figures"
    plotting.plot_summary(matrix, labels, summary, "Mean pairwise spike-count r (100 ms)",
                          "Q4", folder / "summary.png", reference=0.0)
    plot_pair_distributions(records, labels, folder / "pair_r_all_mice.png")
    totals = {a: {k: int(sum(r["areas"][a][k] for r in records.values()))
                  for k in ("n_units", "n_pairs", "n_nan_pairs")} for a in labels}
    return cli.final_report(QUESTION, "mean over unit pairs of the zero-lag Pearson r of "
                            "spike counts in 100 ms bins (whole recording)", "r (unitless)",
                            values, labels, summary, counts=totals)


def plot_example(config: dict[str, Any], mouse: int,
                 example: dict[int, tuple[np.ndarray, np.ndarray]], record: dict[str, Any],
                 bin_s: float) -> None:
    """Step figures for one mouse: count traces, correlation matrices, pair-r histograms."""
    folder = cli.output_dir(config, KEY) / "figures"
    n_show = int(round(60 / bin_s))
    figure, axes = plt.subplots(len(example), 1, figsize=(13, 2.6 * len(example)), sharex=True)
    for axis, (area, (counts, _)) in zip(axes, example.items()):
        start = counts.shape[1] // 2
        busiest = np.argsort(counts.sum(axis=1))[::-1][:5]
        for k, unit in enumerate(busiest):
            axis.step(np.arange(n_show) * bin_s, counts[unit, start:start + n_show] + 8 * k,
                      where="post", color=plotting.area_color(area), linewidth=0.6)
        axis.set(ylabel=f"Area {area}\ncounts (offset)", yticks=[])
    axes[-1].set_xlabel("Time (s, 60 s from the middle of the recording; 100 ms bins)")
    figure.suptitle(f"Mouse {mouse}: spike counts of the 5 most active units per area")
    plotting.save(figure, folder / "example_counts.png")

    figure, axes = plt.subplots(1, len(example) + 1, figsize=(5 * (len(example) + 1), 4.3))
    for axis, (area, (_, matrix)) in zip(axes, example.items()):
        shown = np.where(np.eye(len(matrix), dtype=bool), np.nan, matrix)
        limit = np.nanmax(np.abs(shown))
        image = axis.imshow(shown, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set(title=f"Area {area}: unit × unit r", xlabel="Unit", ylabel="Unit")
        figure.colorbar(image, ax=axis, shrink=0.8)
        pairs = np.asarray(record["areas"][str(area)]["pair_r"])
        axes[-1].hist(pairs, bins=np.linspace(-0.2, 0.6, 61), histtype="step", linewidth=1.5,
                      color=plotting.area_color(area),
                      label=f"Area {area}: mean {pairs.mean():.3f}")
    axes[-1].axvline(0, color="black", linewidth=0.8)
    axes[-1].set(xlabel="Pair r (100 ms counts)", ylabel="Pairs", title="Pair distributions")
    axes[-1].legend(fontsize=8)
    figure.suptitle(f"Mouse {mouse}")
    plotting.save(figure, folder / "example_matrices.png")


def plot_pair_distributions(records: dict[int, dict[str, Any]], labels: list[str],
                            path: Path) -> None:
    """Pair-r distributions per area, pooled over mice."""
    figure, axis = plt.subplots(figsize=(6, 4))
    for area in labels:
        pairs = np.concatenate([r["areas"][area]["pair_r"] for r in records.values()])
        axis.hist(pairs, bins=np.linspace(-0.2, 0.6, 81), histtype="step", linewidth=1.5,
                  density=True, color=plotting.area_color(area),
                  label=f"Area {area} ({pairs.size} pairs)")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Pair r (100 ms counts)", ylabel="Density", title="All pairs, all mice")
    axis.legend(fontsize=8)
    plotting.save(figure, path)


if __name__ == "__main__":
    cli.main(KEY, run_animal, aggregate, __doc__)
