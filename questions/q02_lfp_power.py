"""Question 2: which brain area has the highest broadband LFP power (1–100 Hz)?

Final variable: per mouse and area, the mean over channels of each channel's
Welch PSD integrated from 1 to 100 Hz (linear units, signal units², likely V²).

Run: ``python -m questions.q02_lfp_power --animal 1`` then ``--aggregate``.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from con2phys import cli, io, plotting, signal, stats

KEY = "q02"
QUESTION = "Q2. Which brain area has the highest broadband LFP power (1–100 Hz)?"
BANDS_HZ = {"delta 1–4": (1.0, 4.0), "theta 4–10": (4.0, 10.0), "beta 13–30": (13.0, 30.0),
            "gamma 30–100": (30.0, 100.0)}  # for the descriptive band breakdown only


def area_power(lfp: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray]:
    """Welch PSD and 1–100 Hz power of every channel of one area.

    Returns:
        Frequencies, PSD ``[channels × frequencies]`` and power per channel.

    Scientific rationale:
        Broadband power is the area under the linear PSD in the band, i.e.
        the variance of the 1–100 Hz signal; each channel is a recording site
        and gets equal weight in the area mean.
    """
    fs = float(config["lfp_sampling_rate_hz"])
    frequencies, psd = signal.welch_psd(lfp, fs, float(config["welch"]["window_s"]),
                                        float(config["welch"]["overlap_fraction"]))
    power = signal.bandpower(frequencies, psd, tuple(config[KEY]["frequency_band_hz"]))
    return frequencies, psd, power


def run_animal(config: dict[str, Any], mouse: int) -> dict[str, Any]:
    """Channel powers and mean PSD of every area of one animal."""
    folder = io.mouse_dir(config, mouse)
    fs = float(config["lfp_sampling_rate_hz"])
    record: dict[str, Any] = {"areas": {}}
    lfps = {}
    for area in config["areas"]:
        lfp = io.load_lfp(io.lfp_path(folder, area), fs)
        frequencies, psd, power = area_power(lfp, config)
        record["frequencies_hz"] = frequencies.tolist()
        record["areas"][str(area)] = {
            "mean_power": float(power.mean()), "n_channels": int(lfp.shape[0]),
            "channel_power": power.tolist(), "mean_psd": psd.mean(axis=0).tolist(),
            "band_power": {name: float(signal.bandpower(frequencies, psd, band).mean())
                           for name, band in BANDS_HZ.items()}}
        if mouse == int(config["example_mouse"]):
            lfps[area] = (lfp[:, : int(10 * fs)], psd)
    if lfps:
        plot_example(config, mouse, np.asarray(record["frequencies_hz"]), lfps, record)
    return record


def aggregate(config: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Across-mouse summary, decision and figures."""
    labels = [str(a) for a in config["areas"]]
    values = {m: [r["areas"][a]["mean_power"] for a in labels] for m, r in records.items()}
    matrix = np.array(list(values.values()))
    summary = stats.summarize_conditions(matrix, labels, config, ranking="highest")
    folder = cli.output_dir(config, KEY) / "figures"
    plotting.plot_summary(matrix, labels, summary, "Mean channel power 1–100 Hz (V²)", "Q2",
                          folder / "summary.png", log_scale=True)
    plot_group_spectra(records, labels, folder / "group_psd_and_bands.png")
    return cli.final_report(QUESTION, "mean over channels of the Welch PSD integrated over "
                            "1–100 Hz", "V² (dataset units squared)", values, labels, summary,
                            n_channels={a: int(sum(r["areas"][a]["n_channels"]
                                                   for r in records.values())) for a in labels})


def plot_example(config: dict[str, Any], mouse: int, frequencies: np.ndarray,
                 lfps: dict[int, tuple[np.ndarray, np.ndarray]], record: dict[str, Any]) -> None:
    """Step figures for one mouse: raw traces, channel PSDs, power across depth."""
    fs = float(config["lfp_sampling_rate_hz"])
    folder = cli.output_dir(config, KEY) / "figures"
    figure, axes = plt.subplots(1, len(lfps), figsize=(15, 6), sharey=False)
    for axis, (area, (snippet, _)) in zip(axes, lfps.items()):
        channels = np.linspace(0, snippet.shape[0] - 1, 8).astype(int)
        spacing = 4 * np.median(np.std(snippet, axis=1))
        time = np.arange(snippet.shape[1]) / fs
        for k, channel in enumerate(channels):
            axis.plot(time, snippet[channel] + k * spacing, color=plotting.area_color(area),
                      linewidth=0.5)
        axis.set_yticks([k * spacing for k in range(len(channels))],
                        [f"ch {c}" for c in channels])
        axis.set(xlabel="Time (s)", title=f"Area {area}: raw LFP, 8 of {snippet.shape[0]} "
                 "channels (deep → superficial)")
    plotting.save(figure, folder / "example_traces.png")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    keep = (frequencies >= 0.5) & (frequencies <= 200)
    for area, (_, psd) in lfps.items():
        color = plotting.area_color(area)
        axes[0].loglog(frequencies[keep], psd[:, keep].T, color=color, alpha=0.15, linewidth=0.5)
        axes[0].loglog(frequencies[keep], psd[:, keep].mean(axis=0), color=color, linewidth=2,
                       label=f"Area {area}")
        power = record["areas"][str(area)]["channel_power"]
        axes[1].plot(power, np.arange(len(power)), "-o", markersize=3, color=color,
                     label=f"Area {area}")
        axes[2].bar(np.arange(len(BANDS_HZ)) + 0.25 * (int(area) - 2),
                    list(record["areas"][str(area)]["band_power"].values()), width=0.25,
                    color=color, label=f"Area {area}")
    for f in config[KEY]["frequency_band_hz"]:
        axes[0].axvline(f, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)",
                title="Channel PSDs (thin) and area mean (thick)")
    axes[1].set(xscale="log", xlabel="1–100 Hz power (V²)",
                ylabel="Channel index (0 = deepest)", title="Power across depth")
    axes[2].set_xticks(range(len(BANDS_HZ)), list(BANDS_HZ), fontsize=8)
    axes[2].set(yscale="log", ylabel="Mean channel power (V²)", title="Power per band")
    for axis in axes:
        axis.legend(fontsize=8)
    figure.suptitle(f"Mouse {mouse}")
    plotting.save(figure, folder / "example_spectra.png")


def plot_group_spectra(records: dict[int, dict[str, Any]], labels: list[str], path) -> None:
    """Mouse-averaged PSD per area (with each mouse thin) and band power per area."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    frequencies = np.asarray(next(iter(records.values()))["frequencies_hz"])
    keep = (frequencies >= 0.5) & (frequencies <= 200)
    for area in labels:
        psds = np.array([r["areas"][area]["mean_psd"] for r in records.values()])[:, keep]
        color = plotting.area_color(area)
        axes[0].loglog(frequencies[keep], psds.T, color=color, alpha=0.15, linewidth=0.5)
        axes[0].loglog(frequencies[keep], psds.mean(axis=0), color=color, linewidth=2,
                       label=f"Area {area}")
        bands = np.array([list(r["areas"][area]["band_power"].values())
                          for r in records.values()])
        axes[1].errorbar(np.arange(len(BANDS_HZ)) + 0.1 * (int(area) - 2), bands.mean(axis=0),
                         yerr=bands.std(axis=0), fmt="o", color=color, capsize=3,
                         label=f"Area {area}")
    axes[0].set(xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)",
                title="Area-mean PSD per mouse (thin) and across mice (thick)")
    axes[1].set_xticks(range(len(BANDS_HZ)), list(BANDS_HZ), fontsize=8)
    axes[1].set(yscale="log", ylabel="Power (V²), mean ± SD over mice", title="Power per band")
    for axis in axes:
        axis.legend(fontsize=8)
    plotting.save(figure, path)


if __name__ == "__main__":
    cli.main(KEY, run_animal, aggregate, __doc__)
