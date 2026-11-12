"""Question 12: during which trial segment is variable C best decoded?

Final variable: per mouse and segment, the balanced accuracy of a linear SVM
decoding C (C = 0 vs C = 2; the near-absent C = 1 is dropped) from the firing
rates of all units of all areas in that segment, averaged over 5 unshuffled
stratified folds. Chance = 0.5.

Run: ``python -m questions.q12_decoding --animal 1`` then ``--aggregate``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from con2phys import cli, io, plotting, stats
from questions.q12_figures import SEGMENT_LABELS, plot_durations, plot_per_mouse

KEY = "q12"
QUESTION = "Q12. During which trial segment is variable C best decoded?"


def valid_trial_mask(trials: io.TrialAnnotations, segments: list[tuple[str, str]],
                     excluded_labels: list[int]) -> np.ndarray:
    """Trials usable for decoding C in every segment.

    Args:
        trials: Trial annotations of one animal.
        segments: ``(start_event, end_event)`` pairs defining the segments.
        excluded_labels: Values of C to drop (here the near-absent C = 1).

    Returns:
        Boolean mask over trials.

    Scientific rationale:
        A trial is kept only if C is known and not excluded and all segments
        have finite, strictly positive durations. All segments are then
        decoded from the same trials and folds, so the comparison is paired.
    """
    labels = np.asarray(trials.variable_c, dtype=float)
    keep = np.isfinite(labels) & ~np.isin(labels, excluded_labels)
    for start_event, end_event in segments:
        duration = trials.event(end_event) - trials.event(start_event)
        keep &= np.isfinite(duration) & (duration > 0)
    return keep


def segment_rates(trains: list[np.ndarray], starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Firing rate (spikes/s) of every unit in every half-open segment, ``[n_trials × n_units]``.

    Scientific rationale:
        Dividing by the segment's own duration removes trial-to-trial
        differences in segment length, which would otherwise let the decoder
        read C from timing instead of from neural activity.
    """
    counts = np.empty((len(starts), len(trains)))
    for unit, train in enumerate(trains):
        times = np.sort(train)
        counts[:, unit] = (np.searchsorted(times, ends, side="left")
                           - np.searchsorted(times, starts, side="left"))
    return counts / (ends - starts)[:, None]


def decode_segment(features: np.ndarray, labels: np.ndarray, n_folds: int,
                   svm_c: float) -> np.ndarray:
    """Cross-validated balanced accuracy of a linear SVM, one value per test fold.

    Args:
        features: ``[n_trials × n_units]`` rates, trials in chronological order.
        labels: C per trial (two classes).
        n_folds: Number of stratified folds.
        svm_c: SVM regularisation strength.

    Scientific rationale:
        Unshuffled stratified folds split each class chronologically, so test
        trials come from limited stretches of the session (only partly
        contiguous blocks), which limits leakage from slow drift. Balanced
        accuracy has chance 0.5 whatever the class proportions. The scaler is
        fitted on training trials only.
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have one row per trial")
    scores = []
    for train, test in StratifiedKFold(n_splits=n_folds, shuffle=False).split(features, labels):
        model = make_pipeline(StandardScaler(),
                              SVC(kernel="linear", C=svm_c, class_weight="balanced"))
        model.fit(features[train], labels[train])
        scores.append(balanced_accuracy_score(labels[test], model.predict(features[test])))
    return np.asarray(scores)


def analyze_mouse(trains: list[np.ndarray], trials: io.TrialAnnotations,
                  settings: dict[str, Any]) -> dict[str, Any]:
    """Decode C in every segment for one animal (all units of all areas pooled).

    Scientific rationale:
        One decoder per animal and segment treats the animal as the
        independent unit and compares segments on the same trials and units.
    """
    segments = {name: tuple(bounds) for name, bounds in settings["segments"].items()}
    labels = np.asarray(trials.variable_c, dtype=float)
    excluded = np.isin(labels, settings["excluded_c_values"])
    keep = valid_trial_mask(trials, list(segments.values()), settings["excluded_c_values"])
    y = labels[keep].astype(int)
    record: dict[str, Any] = {
        "n_trials_total": int(len(labels)), "n_units": len(trains),
        "n_dropped_excluded_c": int(excluded.sum()),
        "n_dropped_invalid": int((~keep & ~excluded).sum()),
        "class_counts": {str(c): int((y == c).sum()) for c in np.unique(y)},
        "segments": {},
    }
    for name, (start_event, end_event) in segments.items():
        starts = trials.event(start_event)[keep]
        ends = trials.event(end_event)[keep]
        folds = decode_segment(segment_rates(trains, starts, ends), y,
                               int(settings["n_folds"]), float(settings["svm_c"]))
        record["segments"][name] = {"balanced_accuracy": float(folds.mean()),
                                    "fold_scores": folds.tolist(),
                                    "durations_s": (ends - starts).tolist()}
    return record


def run_animal(config: dict[str, Any], mouse: int) -> dict[str, Any]:
    """Balanced accuracy of every segment for one animal."""
    folder = io.mouse_dir(config, mouse)
    animal = io.load_spikes(folder)
    return analyze_mouse(io.unit_spike_trains(animal), io.load_trials(folder / "trial_data.xlsx"),
                         config[KEY])


def aggregate(config: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Across-mouse summary, decision and figures."""
    settings = config[KEY]
    labels = list(settings["segments"])
    chance = float(settings["chance_level"])
    values = {m: [r["segments"][n]["balanced_accuracy"] for n in labels]
              for m, r in records.items()}
    matrix = np.array(list(values.values()))
    names = {n: SEGMENT_LABELS[n].replace("\n", " ") for n in labels}
    summary = stats.summarize_conditions(matrix, labels, config, ranking="highest",
                                         answer_labels=names, reference=chance)
    folder = cli.output_dir(config, KEY) / "figures"
    dpi = int(config["dpi"])
    plotting.plot_summary(matrix, labels, summary, "Balanced accuracy (C = 0 vs C = 2)", "Q12",
                          folder / "summary.png", tick_labels=[SEGMENT_LABELS[n] for n in labels],
                          reference=chance)
    plot_per_mouse(records, labels, chance, folder / "per_mouse_folds.png", dpi)
    plot_durations(records, labels, folder / "segment_durations.png", dpi)
    per_mouse = {str(m): {k: v for k, v in r.items() if k not in ("segments", "mouse")}
                 for m, r in records.items()}
    return cli.final_report(QUESTION, "balanced accuracy of a linear SVM decoding C (0 vs 2) "
                            "from all units' segment firing rates, mean of 5 folds",
                            "balanced accuracy (chance = 0.5)", values, labels, summary,
                            trials_per_mouse=per_mouse)


if __name__ == "__main__":
    cli.main(KEY, run_animal, aggregate, __doc__)
