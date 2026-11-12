"""Trial-segment selection and circular-shift nulls shared by the spike questions."""

from __future__ import annotations

import numpy as np


def valid_segments(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Trials whose segment has finite bounds and a strictly positive length.

    Scientific rationale:
        A segment is only defined when both events were recorded and occur in
        the expected order; a few trials have a missing event or an outcome
        after the trial end.
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    return np.isfinite(starts) & np.isfinite(ends) & (ends > starts)


def draw_circular_shifts(n_rows: int, n_shifts: int, n_frames: int, min_shift: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Draw circular-shift offsets in ``[min_shift, n_frames - min_shift)``.

    Args:
        n_rows: Number of independent rows (e.g. units), each with its own offsets.
        n_shifts: Number of shuffles per row.
        n_frames: Length of the series being shifted.
        min_shift: Minimum offset in frames.
        rng: Seeded generator.

    Returns:
        Integer array ``(n_rows, n_shifts)``.

    Raises:
        ValueError: If the series is too short to leave any admissible offset.

    Scientific rationale:
        A circular shift keeps each trace's autocorrelation and event
        statistics but breaks its alignment to behaviour (or to other units).
    """
    n_frames, min_shift = int(n_frames), int(min_shift)
    if n_frames <= 2 * min_shift:
        raise ValueError(f"Series of {n_frames} frames is too short for circular shifts "
                         f"of at least {min_shift} frames.")
    return rng.integers(min_shift, n_frames - min_shift, size=(int(n_rows), int(n_shifts)))
