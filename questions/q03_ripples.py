"""Question 3: which brain area has the highest density of (hippocampal-type) ripples?

Final variable: per mouse and area, the number of consensus ripple events per
minute of recording. An event is a consensus event when the Kay, Karlsson and
Roumis detectors (``ripple_detection``), each run on the area's multichannel
150–240 Hz signal, all report an event centre within 20 ms. Events that
coincide (within 20 ms) in all three areas are removed as non-local.

Run: ``python -m questions.q03_ripples --animal 1`` then ``--aggregate``.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from con2phys import cli, io, plotting, signal, stats

KEY = "q03"
QUESTION = "Q3. Which brain area (if any) has the highest density of ripples?"
DETECTOR_NAMES = ("Kay", "Karlsson", "Roumis")
FLAG_REMOVED_FRACTION = 0.10  # exclusion removing < 10% of events flags a mouse


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detector_event_centers(filtered_lfp: np.ndarray, sampling_rate_hz: float
                           ) -> dict[str, np.ndarray]:
    """Run the Kay, Karlsson and Roumis detectors on one area.

    Args:
        filtered_lfp: Ripple-band ``[channels × samples]`` LFP.
        sampling_rate_hz: Sampling frequency.

    Returns:
        Detector name → event centres (s), the midpoint of each event.

    Scientific rationale:
        Three independent ripple definitions on the same multichannel signal
        make consensus an explicit agreement criterion instead of one
        detector's threshold. Speed is set to 0: the data contain no running
        speed, so no immobility gating is possible.
    """
    if filtered_lfp.ndim != 2 or not np.isfinite(filtered_lfp).all():
        raise ValueError("filtered_lfp must be a finite channels-by-time array")
    import ripple_detection

    detectors = {"Kay": ripple_detection.Kay_ripple_detector,
                 "Karlsson": ripple_detection.Karlsson_ripple_detector,
                 "Roumis": ripple_detection.Roumis_ripple_detector}
    time_s = np.arange(filtered_lfp.shape[1], dtype=float) / sampling_rate_hz
    speed = np.zeros(time_s.size, dtype=float)
    centers = {}
    for name in DETECTOR_NAMES:
        table = detectors[name](time_s, filtered_lfp.T, speed, sampling_rate_hz)
        centers[name] = np.asarray((table["start_time"].to_numpy()
                                    + table["end_time"].to_numpy()) / 2, dtype=float)
    return centers


def merge_close_events(events_s: np.ndarray, tolerance_s: float) -> np.ndarray:
    """Merge sorted event centres closer than ``tolerance_s`` into their mean.

    Scientific rationale:
        One ripple matched by two candidate events of the same detector would
        otherwise produce two consensus events; merging guarantees at most one
        event per ripple.
    """
    events = np.sort(np.asarray(events_s, dtype=float))
    if events.size == 0:
        return events
    groups = np.split(events, np.flatnonzero(np.diff(events) > tolerance_s + 1e-12) + 1)
    return np.array([group.mean() if group.size > 1 else group[0] for group in groups])


def detector_consensus(detector_centers: dict[str, np.ndarray], tolerance_s: float
                       ) -> np.ndarray:
    """Events found by all three detectors within ``tolerance_s`` of each other.

    Args:
        detector_centers: Event centres keyed by detector name.
        tolerance_s: Maximum range of the three matched centres.

    Returns:
        Sorted consensus event centres (s): the mean of each matched triple,
        with triples closer than the tolerance merged into one event.
    """
    if tolerance_s <= 0 or any(name not in detector_centers for name in DETECTOR_NAMES):
        raise ValueError("all detector centers and a positive tolerance are required")
    centers = [np.asarray(detector_centers[name], dtype=float) for name in DETECTOR_NAMES]
    tolerance = tolerance_s + 1e-12
    matches = []
    for first in centers[0]:
        near = [np.flatnonzero(np.abs(values - first) <= tolerance) for values in centers[1:]]
        for second, third in product(*near):
            triple = [first, centers[1][second], centers[2][third]]
            if max(triple) - min(triple) <= tolerance:
                matches.append(float(np.mean(triple)))
    return merge_close_events(np.unique(np.round(matches, decimals=6)), tolerance_s)


def remove_cross_area_events(events_by_area: list[np.ndarray], tolerance_s: float
                             ) -> list[np.ndarray]:
    """Remove events that coincide (range ≤ tolerance) in all three areas.

    Scientific rationale:
        A hippocampal ripple is local. An event detected simultaneously in
        three non-adjacent areas is a shared (reference or artefact) signal
        and is not counted as three regional ripples.
    """
    if len(events_by_area) != 3 or tolerance_s <= 0:
        raise ValueError("cross-area exclusion requires three areas and a positive tolerance")
    tolerance = tolerance_s + 1e-12
    marked: list[set[int]] = [set(), set(), set()]
    for first_index, first in enumerate(events_by_area[0]):
        second = np.flatnonzero(np.abs(events_by_area[1] - first) <= tolerance)
        third = np.flatnonzero(np.abs(events_by_area[2] - first) <= tolerance)
        for j, k in product(second, third):
            triple = [first, events_by_area[1][j], events_by_area[2][k]]
            if max(triple) - min(triple) <= tolerance:
                marked[0].add(first_index)
                marked[1].add(int(j))
                marked[2].add(int(k))
    return [np.delete(np.asarray(events, dtype=float), sorted(indices))
            for events, indices in zip(events_by_area, marked)]


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------


def trial_intervals(trial_start_s: np.ndarray, trial_end_s: np.ndarray) -> np.ndarray:
    """Union of valid ``[trial_start, trial_end]`` intervals, ``(n, 2)``.

    Scientific rationale:
        Merging overlapping trials avoids double counting trial time.
    """
    starts = np.asarray(trial_start_s, dtype=float).reshape(-1)
    ends = np.asarray(trial_end_s, dtype=float).reshape(-1)
    valid = np.isfinite(starts) & np.isfinite(ends) & (ends >= starts)
    merged: list[list[float]] = []
    for start, end in sorted(zip(starts[valid], ends[valid])):
        if not merged or start > merged[-1][1]:
            merged.append([float(start), float(end)])
        else:
            merged[-1][1] = max(merged[-1][1], float(end))
    return np.asarray(merged, dtype=float).reshape(-1, 2)


def events_inside_trials(events_s: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Whether each event lies inside any (inclusive) trial interval."""
    events = np.asarray(events_s, dtype=float).reshape(-1)
    if intervals.size == 0:
        return np.zeros(events.size, dtype=bool)
    return np.any((events[:, None] >= intervals[None, :, 0])
                  & (events[:, None] <= intervals[None, :, 1]), axis=1)


def trial_time_fraction(intervals: np.ndarray, duration_s: float) -> float:
    """Fraction of the recording covered by trials (intervals clipped to the recording).

    Scientific rationale:
        The fraction of events inside trials is only interpretable against
        the fraction of time inside trials: their ratio is the enrichment
        (> 1 means ripples are over-represented during trials).
    """
    if intervals.size == 0:
        return 0.0
    clipped = np.clip(intervals, 0.0, duration_s)
    return float(np.sum(clipped[:, 1] - clipped[:, 0]) / duration_s)


# ---------------------------------------------------------------------------
# Per animal
# ---------------------------------------------------------------------------


def run_animal(config: dict[str, Any], mouse: int) -> dict[str, Any]:
    """Detect consensus ripples in every area of one animal and count them."""
    settings = config[KEY]
    fs = float(config["lfp_sampling_rate_hz"])
    tolerance = float(settings["detector_match_tolerance_s"])
    folder = io.mouse_dir(config, mouse)
    detections, consensus, shapes = [], [], []
    for area in config["areas"]:
        raw = io.load_lfp(io.lfp_path(folder, area), fs, dtype=np.float32)
        filtered = signal.bandpass(raw, fs, tuple(settings["detector_band_hz"]),
                                   int(settings["filter_order"]))
        detections.append(detector_event_centers(filtered, fs))
        consensus.append(detector_consensus(detections[-1], tolerance))
        shapes.append(raw.shape)
        del raw, filtered
        cli.log(f"q03 mouse {mouse} area {area}: {len(consensus[-1])} consensus events")
    final = remove_cross_area_events(consensus, tolerance)
    duration_s = shapes[0][1] / fs
    trials = io.load_trials(folder / "trial_data.xlsx")
    intervals = trial_intervals(trials.trial_start_s, trials.trial_end_s)
    record: dict[str, Any] = {"duration_s": duration_s,
                              "trial_time_fraction": trial_time_fraction(intervals, duration_s),
                              "areas": {}}
    for area, detected, before, events, shape in zip(config["areas"], detections, consensus,
                                                     final, shapes):
        record["areas"][str(area)] = {
            "n_channels": int(shape[0]),
            "detector_counts": {name: int(len(detected[name])) for name in DETECTOR_NAMES},
            "consensus_before_exclusion": int(len(before)),
            "event_count": int(len(events)),
            "rate_per_minute": float(len(events) / duration_s * 60.0),
            "inside_trial_count": int(events_inside_trials(events, intervals).sum()),
            "events_s": events.tolist()}
    if mouse == int(config["example_mouse"]):
        plot_example(config, mouse, record)
    return record


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def removed_fraction(record: dict[str, Any]) -> float:
    """Fraction of consensus events (all areas) removed by the cross-area exclusion."""
    before = sum(a["consensus_before_exclusion"] for a in record["areas"].values())
    after = sum(a["event_count"] for a in record["areas"].values())
    return 1.0 - after / before if before else 0.0


def trial_enrichment(record: dict[str, Any]) -> dict[str, float]:
    """Fraction of final events inside trials, trial time fraction, and their ratio."""
    events = sum(a["event_count"] for a in record["areas"].values())
    inside = sum(a["inside_trial_count"] for a in record["areas"].values())
    event_fraction = inside / events if events else float("nan")
    time_fraction = record["trial_time_fraction"]
    return {"events_inside_fraction": event_fraction, "trial_time_fraction": time_fraction,
            "enrichment": event_fraction / time_fraction if time_fraction else float("nan")}


def aggregate(config: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Across-mouse summary, decision, caveat diagnostics and figures."""
    labels = [str(a) for a in config["areas"]]
    values = {m: [r["areas"][a]["rate_per_minute"] for a in labels] for m, r in records.items()}
    matrix = np.array(list(values.values()))
    summary = stats.summarize_conditions(matrix, labels, config, ranking="highest")
    removed = {m: removed_fraction(r) for m, r in records.items()}
    flagged = [m for m, f in removed.items() if f < FLAG_REMOVED_FRACTION]
    folder = cli.output_dir(config, KEY) / "figures"
    plotting.plot_summary(matrix, labels, summary, "Consensus ripples per minute", "Q3",
                          folder / "summary.png")
    plot_counts(records, labels, flagged, folder / "detector_counts.png")
    plot_rates_per_mouse(matrix, list(values), labels, flagged, folder / "rates_per_mouse.png")
    enrichment = {m: trial_enrichment(r) for m, r in records.items()}
    inside = sum(a["inside_trial_count"] for r in records.values() for a in r["areas"].values())
    total = sum(a["event_count"] for r in records.values() for a in r["areas"].values())
    pooled_time = (sum(r["trial_time_fraction"] * r["duration_s"] for r in records.values())
                   / sum(r["duration_s"] for r in records.values()))
    return cli.final_report(
        QUESTION, "consensus ripple events (3 detectors agree within 20 ms, minus events "
        "shared by all 3 areas) per minute of recording", "events/min", values, labels, summary,
        detector_counts={m: {a: r["areas"][a]["detector_counts"] for a in labels}
                         for m, r in records.items()},
        consensus_before_exclusion={m: {a: r["areas"][a]["consensus_before_exclusion"]
                                        for a in labels} for m, r in records.items()},
        event_counts={m: {a: r["areas"][a]["event_count"] for a in labels}
                      for m, r in records.items()},
        removed_by_exclusion_fraction=removed, flagged_mice=flagged,
        flag_rule=f"cross-area exclusion removed < {FLAG_REMOVED_FRACTION:.0%} of events",
        median_rate_per_minute={a: float(np.median(matrix[:, i])) for i, a in enumerate(labels)},
        trial_enrichment_by_mouse=enrichment,
        trial_enrichment_pooled={"events_inside_fraction": inside / total,
                                 "trial_time_fraction": pooled_time,
                                 "enrichment": inside / total / pooled_time})


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _event_average(lfp: np.ndarray, events_s: np.ndarray, fs: float, half_s: float
                   ) -> np.ndarray:
    """Mean ``[channels × time]`` window around events with a complete window."""
    half = int(round(half_s * fs))
    centers = np.round(np.asarray(events_s) * fs).astype(int)
    centers = centers[(centers - half >= 0) & (centers + half + 1 <= lfp.shape[1])]
    if centers.size == 0:
        return np.full((lfp.shape[0], 2 * half + 1), np.nan)
    return np.mean([lfp[:, c - half:c + half + 1] for c in centers], axis=0)


def plot_example(config: dict[str, Any], mouse: int, record: dict[str, Any]) -> None:
    """Step figures for one mouse: event traces, event-triggered averages, 50–250 Hz PSD."""
    settings = config[KEY]
    fs = float(config["lfp_sampling_rate_hz"])
    folder = io.mouse_dir(config, mouse)
    out = cli.output_dir(config, KEY) / "figures"
    areas = config["areas"]
    traces, averages, spectra = plt.subplots(2, len(areas), figsize=(15, 7)), \
        plt.subplots(2, len(areas), figsize=(15, 7)), plt.subplots(figsize=(6, 4.5))
    for i, area in enumerate(areas):
        raw = io.load_lfp(io.lfp_path(folder, area), fs, dtype=np.float32)
        events = np.asarray(record["areas"][str(area)]["events_s"])
        detector = signal.bandpass(raw, fs, tuple(settings["detector_band_hz"]),
                                   int(settings["filter_order"]))
        _plot_event_traces(traces[1][:, i], raw, detector, events, fs, area)
        waveform = signal.bandpass(raw, fs, tuple(settings["waveform_band_hz"]),
                                   int(settings["filter_order"]))
        _plot_event_average(averages[1][:, i], raw, waveform, events, fs,
                            float(settings["waveform_half_window_s"]), area)
        frequencies, psd = signal.welch_psd(raw, fs, float(config["welch"]["window_s"]),
                                            float(config["welch"]["overlap_fraction"]))
        keep = frequencies >= 50
        spectra[1].semilogy(frequencies[keep], psd[:, keep].mean(axis=0),
                            color=plotting.area_color(area), label=f"Area {area}")
        del raw, detector, waveform
    spectra[1].axvspan(*settings["detector_band_hz"], color="0.85", label="detector band")
    spectra[1].set(xlabel="Frequency (Hz)", ylabel="Mean channel PSD (V²/Hz)",
                   title=f"Mouse {mouse}: high-frequency PSD (Nyquist 250 Hz)")
    spectra[1].legend(fontsize=8)
    traces[0].suptitle(f"Mouse {mouse}: raw LFP (top) and 150–240 Hz signal (bottom) around "
                       "one final consensus event of each area (red lines = final events)")
    averages[0].suptitle(f"Mouse {mouse}: event-triggered averages over all final events")
    plotting.save(traces[0], out / "example_event_traces.png")
    plotting.save(averages[0], out / "example_event_average.png")
    plotting.save(spectra[0], out / "example_high_frequency_psd.png")


def _plot_event_traces(axes: Any, raw: np.ndarray, filtered: np.ndarray, events: np.ndarray,
                       fs: float, area: int) -> None:
    """Raw and detector-band traces (6 channels) in a 2 s window around one event."""
    axes[0].set_title(f"Area {area}")
    if events.size == 0:
        return
    center = int(round(events[len(events) // 2] * fs))
    window = slice(max(center - int(fs), 0), min(center + int(fs), raw.shape[1]))
    time = np.arange(window.start, window.stop) / fs - events[len(events) // 2]
    channels = np.linspace(0, raw.shape[0] - 1, 6).astype(int)
    for axis, data in zip(axes, (raw, filtered)):
        spacing = 4 * np.median(np.std(data[channels, window], axis=1)) + 1e-12
        for k, channel in enumerate(channels):
            axis.plot(time, data[channel, window] + k * spacing, linewidth=0.5,
                      color=plotting.area_color(area))
        nearby = events[np.abs(events - events[len(events) // 2]) < 1.0] - events[len(events) // 2]
        for event in nearby:
            axis.axvline(event, color="red", linewidth=0.8, alpha=0.6)
        axis.set_yticks([])
        axis.set_xlabel("Time from event (s)")


def _plot_event_average(axes: Any, raw: np.ndarray, filtered: np.ndarray, events: np.ndarray,
                        fs: float, half_s: float, area: int) -> None:
    """Channel × time images of the raw and 50–240 Hz event-triggered averages."""
    for axis, data, name in zip(axes, (raw, filtered), ("raw", "50–240 Hz")):
        average = _event_average(data, events, fs, half_s)
        limit = np.nanpercentile(np.abs(average), 99) if np.isfinite(average).any() else 1.0
        axis.imshow(average, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-limit,
                    vmax=limit, extent=(-half_s, half_s, -0.5, average.shape[0] - 0.5))
        axis.set(xlabel="Time from event (s)", ylabel="Channel (0 = deepest)",
                 title=f"Area {area}, {name} (n = {len(events)} events)")


def plot_counts(records: dict[int, dict[str, Any]], labels: list[str], flagged: list[int],
                path: Path) -> None:
    """Per mouse and area: detector counts, consensus and final counts (log scale)."""
    mice = list(records)
    figure, axes = plt.subplots(1, len(labels), figsize=(16, 4.5), sharey=True)
    stages = [*DETECTOR_NAMES, "consensus", "final"]
    for axis, area in zip(axes, labels):
        for s, stage in enumerate(stages):
            counts = [records[m]["areas"][area]["detector_counts"][stage] if stage in
                      DETECTOR_NAMES else records[m]["areas"][area][
                          "consensus_before_exclusion" if stage == "consensus" else "event_count"]
                      for m in mice]
            axis.plot(range(len(mice)), np.maximum(counts, 0.8), "o--" if stage == "Kay" else "o-",
                      markersize=5 if stage == "Kay" else 3,  # Kay ≈ Roumis would hide it
                      linewidth=0.8, label=stage, color=f"C{s + 3}" if s < 3 else
                      ("black" if stage == "consensus" else plotting.area_color(area)))
        for i, m in enumerate(mice):
            if m in flagged:
                axis.axvspan(i - 0.4, i + 0.4, color="red", alpha=0.12)
        axis.set_xticks(range(len(mice)), mice, fontsize=7)
        axis.set(yscale="log", xlabel="Mouse (red = exclusion removed < 10%)",
                 title=f"Area {area}")
    axes[0].set_ylabel("Events in the recording")
    axes[0].legend(fontsize=7)
    figure.suptitle("Detector counts → three-detector consensus → after cross-area exclusion")
    plotting.save(figure, path)


def plot_rates_per_mouse(matrix: np.ndarray, mice: list[int], labels: list[str],
                         flagged: list[int], path: Path) -> None:
    """Heatmap of the final rate for every mouse and area (flagged mice marked)."""
    figure, axis = plt.subplots(figsize=(5, 7))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white")
    axis.set_xticks(range(len(labels)), [f"Area {a}" for a in labels])
    axis.set_yticks(range(len(mice)), [f"{m}{' *' if m in flagged else ''}" for m in mice])
    axis.set(ylabel="Mouse (* flagged)", title="Final ripples per minute")
    figure.colorbar(image, ax=axis, label="events/min")
    plotting.save(figure, path)


if __name__ == "__main__":
    cli.main(KEY, run_animal, aggregate, __doc__)
