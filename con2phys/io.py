"""Loading the CON²PHYS dataset: configuration, animals, spikes, LFP and trials.

Every question reads its data through this module, so all analyses see the
same unit-to-area mapping, the same recording duration and the same trial
table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, whosmat


# ---------------------------------------------------------------------------
# Configuration and animals
# ---------------------------------------------------------------------------


def read_config(path: Path) -> dict[str, Any]:
    """Read the YAML configuration, the single source of all parameters.

    Args:
        path: Path to ``config.yaml``.

    Returns:
        The complete configuration mapping.
    """
    import yaml

    return yaml.safe_load(Path(path).read_text())


def discover_mice(data_root: Path, excluded_mice: set[int]) -> tuple[int, ...]:
    """Discover numeric animal folders while applying the project exclusion.

    Args:
        data_root: Directory containing one numeric folder per animal.
        excluded_mice: Animal identifiers excluded by the dataset protocol.

    Returns:
        Sorted included animal identifiers.

    Scientific rationale:
        The animal is the independent unit for regional inference, so every
        question must use the same explicit inclusion rule.
    """
    mice = {int(path.name) for path in Path(data_root).iterdir()
            if path.is_dir() and path.name.isdigit()}
    return tuple(sorted(mice - {int(mouse) for mouse in excluded_mice}))


def included_mice(config: dict[str, Any]) -> tuple[int, ...]:
    """Animals included by ``config.yaml`` (all data folders minus ``excluded_mice``)."""
    return discover_mice(Path(config["data_root"]), {int(m) for m in config["excluded_mice"]})


def mouse_dir(config: dict[str, Any], mouse: int) -> Path:
    """Data folder of one animal."""
    return Path(config["data_root"]) / str(mouse)


# ---------------------------------------------------------------------------
# Spikes and units
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnimalSpikes:
    """Spike data and the unit metadata for one animal.

    ``spikes_s`` and ``clusters`` describe individual spikes; ``unit_ids`` and
    ``unit_areas`` are aligned per unit and assign each spike train to an
    anonymised brain area.
    """

    spikes_s: np.ndarray
    clusters: np.ndarray
    unit_ids: np.ndarray
    unit_areas: np.ndarray


def _load_vector(path: Path, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    """Load and flatten a named MATLAB vector, rejecting non-finite floats."""
    values = np.asarray(loadmat(path, variable_names=[name])[name], dtype=dtype).reshape(-1)
    if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
        raise ValueError(f"{path}:{name} contains non-finite values")
    return values


def _load_unit_metadata(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load aligned ``(unit_ids, unit_areas)`` from the ``brain_areas`` struct.

    Args:
        path: ``brain_area.mat`` containing the scalar ``brain_areas`` struct.

    Returns:
        Integer arrays of unit cluster IDs and their area labels (1–3).

    Raises:
        ValueError: If the struct, its fields, or their alignment are invalid.

    Scientific rationale:
        The unit-to-area mapping is the authoritative way to assign each
        cluster's spike train to a recording region.
    """
    data = loadmat(path, variable_names=["brain_areas"])
    if "brain_areas" not in data:
        raise ValueError(f"Missing brain_areas in {path}")
    struct_array = np.asarray(data["brain_areas"])
    if struct_array.size != 1 or struct_array.dtype.names is None:
        raise ValueError(f"Expected one brain_areas struct in {path}")
    record = struct_array.reshape(-1)[0]
    if not {"cluster_id", "brain_area"}.issubset(record.dtype.names or ()):
        raise ValueError(f"brain_areas in {path} lacks cluster_id/brain_area")
    unit_ids = np.asarray(record["cluster_id"], dtype=np.int64).reshape(-1)
    unit_areas = np.asarray(record["brain_area"], dtype=np.int64).reshape(-1)
    if len(unit_ids) != len(unit_areas):
        raise ValueError(f"Unit IDs and area labels have different lengths in {path}")
    if len(np.unique(unit_ids)) != len(unit_ids):
        raise ValueError(f"Duplicate unit IDs in {path}")
    return unit_ids, unit_areas


def load_spikes(animal_dir: Path) -> AnimalSpikes:
    """Load spike events and the unit-to-area mapping of one animal.

    Args:
        animal_dir: Directory with ``spikes.mat``, ``clusters.mat`` and
            ``brain_area.mat``.

    Returns:
        Validated spike events and unit metadata.

    Raises:
        ValueError: If spike and cluster arrays are not aligned.

    Scientific rationale:
        Every analysis must compare spikes belonging to known units; an
        alignment failure would silently assign activity to the wrong area.
    """
    spikes_s = _load_vector(animal_dir / "spikes.mat", "spikes", np.dtype(float))
    clusters = _load_vector(animal_dir / "clusters.mat", "clusters", np.dtype(np.int64))
    unit_ids, unit_areas = _load_unit_metadata(animal_dir / "brain_area.mat")
    if len(spikes_s) != len(clusters):
        raise ValueError(f"Spikes and clusters have different lengths in {animal_dir}")
    return AnimalSpikes(spikes_s, clusters, unit_ids, unit_areas)


def spikes_by_cluster(spikes_s: np.ndarray, clusters: np.ndarray) -> dict[int, np.ndarray]:
    """Group spike timestamps by cluster ID in one sorted pass.

    Args:
        spikes_s: Spike timestamps in seconds.
        clusters: Cluster ID for each spike.

    Returns:
        Mapping from cluster ID to its spike timestamps (in recorded order).
    """
    order = np.argsort(clusters, kind="stable")
    sorted_clusters = clusters[order]
    sorted_spikes = spikes_s[order]
    groups: dict[int, np.ndarray] = {}
    starts = np.flatnonzero(np.r_[True, sorted_clusters[1:] != sorted_clusters[:-1]])
    stops = np.r_[starts[1:], len(sorted_clusters)]
    for start, stop in zip(starts, stops):
        groups[int(sorted_clusters[start])] = sorted_spikes[start:stop]
    return groups


def unit_spike_trains(animal: AnimalSpikes) -> list[np.ndarray]:
    """Spike times of every unit, in the order of ``animal.unit_ids``.

    Units without spikes get an empty array, so the list stays aligned with
    ``animal.unit_areas``.
    """
    groups = spikes_by_cluster(animal.spikes_s, animal.clusters)
    return [groups.get(int(unit), np.empty(0)) for unit in animal.unit_ids]


def report_out_of_range_spikes(mouse: int, spikes_s: np.ndarray, duration_s: float) -> int:
    """Print and return how many spikes fall outside ``[0, duration_s]``.

    Scientific rationale:
        The data are used as provided (spikes, LFP and trials share one clock).
        Mouse 2 has spikes before 0 s; analyses that bin or clip to the
        recording drop them. Logging the count keeps that visible.
    """
    n_outside = int(np.sum((spikes_s < 0) | (spikes_s > duration_s)))
    if n_outside:
        print(f"mouse {mouse}: {n_outside} spikes outside [0, {duration_s:.1f}] s "
              "are ignored by binned or clipped analyses", flush=True)
    return n_outside


# ---------------------------------------------------------------------------
# LFP
# ---------------------------------------------------------------------------


def _sole_variable(path: Path) -> tuple[str, tuple[int, ...]]:
    """Name and shape of the only data variable in a MATLAB file (header only)."""
    variables = [(name, shape) for name, shape, _ in whosmat(path) if not name.startswith("__")]
    if len(variables) != 1:
        raise ValueError(f"Expected one data variable in {path}, found {variables}")
    return variables[0]


def lfp_path(animal_dir: Path, area: int) -> Path:
    """LFP file of one area (``lfp_<area>.mat``)."""
    return animal_dir / f"lfp_{area}.mat"


def recording_sample_count(animal_dir: Path) -> int:
    """Number of LFP samples, read from the area-1 file header without loading it.

    Scientific rationale:
        The LFP defines the recording exposure more reliably than the last
        spike. All three areas share the same length in every animal.
    """
    name, shape = _sole_variable(lfp_path(animal_dir, 1))
    if len(shape) != 2:
        raise ValueError(f"Expected a 2D LFP array in {animal_dir}, found {name}{shape}")
    return int(shape[1])


def recording_duration_s(animal_dir: Path, sampling_rate_hz: float) -> float:
    """Recording duration in seconds (LFP sample count divided by sampling rate)."""
    return recording_sample_count(animal_dir) / float(sampling_rate_hz)


def load_lfp(path: Path, sampling_rate_hz: float, duration_s: float | None = None,
             dtype: type = np.float64) -> np.ndarray:
    """Load one area's LFP as a finite ``[channels × samples]`` array.

    Args:
        path: MATLAB file containing one LFP variable.
        sampling_rate_hz: LFP sampling frequency.
        duration_s: Load only the first ``duration_s`` seconds; ``None`` loads
            the complete recording.
        dtype: Floating dtype of the returned array (float32 halves memory
            for the long ripple and theta analyses).

    Returns:
        Channels-by-time LFP, channels ordered deep to superficial.

    Raises:
        ValueError: If the file does not hold one finite 2-D array.
    """
    name, _ = _sole_variable(path)
    lfp = np.asarray(loadmat(path, variable_names=[name])[name], dtype=dtype)
    if lfp.ndim != 2 or lfp.shape[1] < 2:
        raise ValueError(f"Expected a 2-D channels-by-time LFP in {path}")
    if duration_s is not None:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive or None")
        lfp = lfp[:, : min(lfp.shape[1], int(round(duration_s * sampling_rate_hz)))]
    if not np.isfinite(lfp).all():
        raise ValueError(f"LFP contains non-finite values: {path}")
    return lfp


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------


EVENTS = ("trial_start", "stim_start", "outcome", "trial_end")


@dataclass(frozen=True)
class TrialAnnotations:
    """Trial event times (s; missing = NaN) and behavioural labels of one animal."""

    trial_start_s: np.ndarray
    stim_start_s: np.ndarray
    outcome_s: np.ndarray
    trial_end_s: np.ndarray
    variable_a: np.ndarray
    variable_b: np.ndarray
    variable_c: np.ndarray

    def event(self, name: str) -> np.ndarray:
        """Times of one event (``trial_start``, ``stim_start``, ``outcome``, ``trial_end``)."""
        if name not in EVENTS:
            raise ValueError(f"Unknown trial event {name!r}")
        return getattr(self, f"{name}_s")


def load_trials(path: Path) -> TrialAnnotations:
    """Load trial event times and behavioural labels from ``trial_data.xlsx``.

    Args:
        path: Animal-specific trial spreadsheet.

    Returns:
        Trial annotations aligned to the electrophysiology clock. Missing
        cells become NaN, so every consumer gets float arrays.
    """
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    rows = workbook.active.iter_rows(values_only=True)
    headers = [str(value).strip().lower() for value in next(rows)]
    columns: dict[str, list[Any]] = {header: [] for header in headers}
    for row in rows:
        for header, value in zip(headers, row):
            columns[header].append(value)
    workbook.close()
    names = [*EVENTS, "variable_a", "variable_b", "variable_c"]
    missing = set(names) - columns.keys()
    if missing:
        raise ValueError(f"Missing trial columns in {path}: {sorted(missing)}")
    as_float = [np.array([np.nan if v is None else v for v in columns[n]], dtype=float)
                for n in names]
    return TrialAnnotations(*as_float)
