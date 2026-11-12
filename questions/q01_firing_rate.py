"""Question 1: which brain area has the lowest firing rate over the entire recording?

Final variable: per mouse and area, the mean over units of each unit's firing
rate (spikes / recording duration, Hz). Lowest mean across mice wins if the
shared decision rule supports it (``con2phys.stats.summarize_conditions``).

Run: ``python -m questions.q01_firing_rate --animal 1`` then ``--aggregate``.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from con2phys import cli, io, plotting, stats

KEY = "q01"
QUESTION = "Q1. Which brain area has the lowest firing rate over the entire recording?"


def unit_firing_rate(spike_times_s: np.ndarray, duration_s: float) -> float:
    """Mean firing rate of one unit over the whole recording, with Elephant.

    Args:
        spike_times_s: Spike times of the unit (s).
        duration_s: Recording duration (s), from the LFP length.

    Returns:
        Rate in Hz; spikes outside ``[0, duration_s]`` are ignored and a unit
        without spikes has 0 Hz.

    Scientific rationale:
        "Over the entire recording" means every unit's spike count is divided
        by the same exposure, the full recording length.
    """
    import elephant.statistics
    import neo
    import quantities as pq

    times = np.asarray(spike_times_s, dtype=float)
    times = times[(times >= 0) & (times <= duration_s)]
    if times.size == 0:
        return 0.0
    train = neo.SpikeTrain(times * pq.s, t_start=0 * pq.s, t_stop=duration_s * pq.s)
    return float(elephant.statistics.mean_firing_rate(train).rescale(pq.Hz).magnitude)


def run_animal(config: dict[str, Any], mouse: int) -> dict[str, Any]:
    """Unit rates of one animal and their mean per area.

    Scientific rationale:
        Averaging units within each mouse makes the mouse the independent
        observation, so animals with many sorted units do not dominate.
    """
    folder = io.mouse_dir(config, mouse)
    animal = io.load_spikes(folder)
    duration_s = io.recording_duration_s(folder, float(config["lfp_sampling_rate_hz"]))
    io.report_out_of_range_spikes(mouse, animal.spikes_s, duration_s)
    rates = np.array([unit_firing_rate(t, duration_s) for t in io.unit_spike_trains(animal)])
    record: dict[str, Any] = {"duration_s": duration_s, "areas": {}}
    for area in config["areas"]:
        area_rates = rates[animal.unit_areas == area]
        if area_rates.size == 0:
            raise ValueError(f"Mouse {mouse} has no units in area {area}")
        record["areas"][str(area)] = {"mean_rate_hz": float(area_rates.mean()),
                                      "unit_rates_hz": area_rates.tolist()}
    if mouse == int(config["example_mouse"]):
        plot_example(config, mouse, animal, duration_s, record)
    return record


def aggregate(config: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Across-mouse summary, decision and figures."""
    labels = [str(a) for a in config["areas"]]
    values = {m: [r["areas"][a]["mean_rate_hz"] for a in labels] for m, r in records.items()}
    matrix = np.array(list(values.values()))
    summary = stats.summarize_conditions(matrix, labels, config, ranking="lowest")
    folder = cli.output_dir(config, KEY) / "figures"
    plotting.plot_summary(matrix, labels, summary, "Mean unit firing rate (Hz)", "Q1",
                          folder / "summary.png")
    plot_unit_distributions(records, labels, folder / "unit_rates_all_mice.png")
    n_units = {a: int(sum(len(r["areas"][a]["unit_rates_hz"]) for r in records.values()))
               for a in labels}
    return cli.final_report(QUESTION, "mean over units of each unit's firing rate "
                            "(spike count / recording duration)", "Hz", values, labels, summary,
                            n_units=n_units)


def plot_example(config: dict[str, Any], mouse: int, animal: io.AnimalSpikes,
                 duration_s: float, record: dict[str, Any]) -> None:
    """Step figure: a 60 s raster per area and the unit-rate distribution of one mouse."""
    trains = io.unit_spike_trains(animal)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]})
    start = duration_s / 2
    row = 0
    for area in config["areas"]:
        for index in np.flatnonzero(animal.unit_areas == area):
            t = trains[index]
            t = t[(t >= start) & (t < start + 60)] - start
            axes[0].vlines(t, row, row + 0.8, color=plotting.area_color(area), linewidth=0.5)
            row += 1
    axes[0].set(xlabel="Time (s, 60 s from the middle of the recording)", ylabel="Unit",
                title=f"Mouse {mouse}: spike raster, coloured by area")
    for i, area in enumerate(config["areas"]):
        rates = np.array(record["areas"][str(area)]["unit_rates_hz"])
        axes[1].scatter(np.full(rates.size, i) + np.random.default_rng(i).uniform(-0.15, 0.15,
                                                                                    rates.size),
                        rates, s=10, color=plotting.area_color(area), alpha=0.7)
        axes[1].hlines(rates.mean(), i - 0.3, i + 0.3, color="black")
    axes[1].set_yscale("symlog", linthresh=0.1)
    axes[1].set_xticks(range(len(config["areas"])), [f"Area {a}" for a in config["areas"]])
    axes[1].set(ylabel="Unit firing rate (Hz)", title="Units (bar = mouse value)")
    plotting.save(figure, cli.output_dir(config, KEY) / "figures" / "example_mouse.png")


def plot_unit_distributions(records: dict[int, dict[str, Any]], labels: list[str], path) -> None:
    """Distribution of all units' rates per area, pooled over mice (log-spaced bins)."""
    figure, axis = plt.subplots(figsize=(6, 4))
    bins = np.logspace(-3, 2.5, 50)
    for area in labels:
        rates = np.concatenate([r["areas"][area]["unit_rates_hz"] for r in records.values()])
        axis.hist(np.clip(rates, 1e-3, None), bins=bins, histtype="step", linewidth=1.5,
                  color=plotting.area_color(area), label=f"Area {area} ({rates.size} units)")
    axis.set_xscale("log")
    axis.set(xlabel="Unit firing rate (Hz; silent units at 1e-3)", ylabel="Units",
             title="All units, all mice")
    axis.legend(fontsize=8)
    plotting.save(figure, path)


if __name__ == "__main__":
    cli.main(KEY, run_animal, aggregate, __doc__)
