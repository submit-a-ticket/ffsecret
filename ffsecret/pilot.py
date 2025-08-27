from __future__ import annotations

import numpy as np

__all__ = ["generate_pilot", "ALPHA_PILOT", "PILOT_SIZE", "PILOT_BITS"]

# Fixed parameters for the pilot pattern. Keeping them here makes them easily
# discoverable and modifiable without touching the embedding/extraction logic.
PILOT_SIZE: int = 32  # side length of the square pilot matrix
ALPHA_PILOT: float = 0.6  # additional strength factor exclusively for pilot marks

# The number of pilot bits that will be embedded in each tile (flattened row-major)
PILOT_BITS: int = PILOT_SIZE * PILOT_SIZE  # 1024 bits by default


def generate_pilot(key: int, size: int = PILOT_SIZE) -> np.ndarray:
    """Return a deterministic binary (0/1) PN matrix of shape *(size, size)*.

    The sequence is generated using NumPy's PCG64 RNG seeded with *key* to make
    it secret-key dependent yet fully deterministic.  The caller is expected to
    treat the resulting matrix as read-only and embed the flattened bitstream
    into the frequency domain of each tile before the actual payload data.
    """
    rng = np.random.default_rng(key)
    # Use uint8 to keep memory low; convert to Python int when iterating.
    return rng.integers(0, 2, size=(size, size), dtype=np.uint8) 