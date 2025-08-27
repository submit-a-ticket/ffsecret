from __future__ import annotations

from typing import Tuple

import numpy as np

from .pilot import generate_pilot, PILOT_BITS, PILOT_SIZE
from .core import _mid_band_coords  # reuse private helper (still in same package)

__all__ = ["locate_pilot"]


def _pilot_coords(tile_size: int, key: int):
    """Return the list of conjugate-pair coordinates reserved for the pilot."""
    coords = _mid_band_coords(tile_size, tile_size)
    rng = np.random.default_rng(key)
    rng.shuffle(coords)
    needed = PILOT_BITS
    if needed > len(coords):
        raise ValueError("Tile size too small for pilot pattern")
    return coords[:needed]


def locate_pilot(F: np.ndarray, key: int, tile_size: int) -> Tuple[int, int, float]:
    """Vectorised pilot-location using *one* tile correlation.

    Rather than iterating over every tile for every possible phase, we
    correlate the pilot against a single tile-sized window sliding over the
    range ``0..tile_size-1`` in x / y.  Because every tile uses the same pilot,
    the offset that maximises correlation inside *one* tile is identical to the
    global offset.
    """

    H, W = F.shape
    if H < tile_size or W < tile_size:
        raise ValueError("Image smaller than tile size")

    mag = np.abs(F)

    pilot_bits = generate_pilot(key).flatten()
    pilot_coords = _pilot_coords(tile_size, key)
    signs = np.where(pilot_bits == 1, 1.0, -1.0)

    # Precompute arrays of coordinate offsets
    py = np.array([c[0] for c in pilot_coords])
    px = np.array([c[1] for c in pilot_coords])
    py_sym = tile_size - 1 - py
    px_sym = tile_size - 1 - px

    scores = np.full((tile_size, tile_size), -np.inf, dtype=np.float32)

    for dy in range(tile_size):
        if dy > H - tile_size:
            break
        y_indices = dy + py
        y_indices_sym = dy + py_sym
        for dx in range(tile_size):
            if dx > W - tile_size:
                break
            x_indices = dx + px
            x_indices_sym = dx + px_sym

            c1 = mag[y_indices, x_indices]
            c2 = mag[y_indices_sym, x_indices_sym]
            diff = (c1 - c2) * signs
            scores[dy, dx] = float(diff.mean())

    best_flat = int(np.argmax(scores))
    best_dy, best_dx = divmod(best_flat, tile_size)
    confidence = scores[best_dy, best_dx] - np.nanmean(scores)
    return best_dy, best_dx, float(confidence) 