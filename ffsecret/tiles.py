"""Tile utilities for crop-tolerant watermarking.

choose_tile_size(width, height) -> int
iter_tiles(y_array, tile_size) -> iterator of (ix, iy, y_subarray, (y0,y1,x0,x1))
"""
from typing import Iterator, Tuple
import numpy as np

__all__ = ["choose_tile_size", "iter_tiles"]

def _round_pow2(n: int) -> int:
    """Round *n* to the nearest power-of-two (128,256,512)."""
    powers = [128, 256, 512]
    return min(powers, key=lambda p: abs(p - n))

def choose_tile_size(w: int, h: int) -> int:
    shortest = min(w, h)
    approx = int((10_000) ** 0.5)  # ≈100 px
    ts = _round_pow2(approx)
    ts = max(128, ts)
    ts = min(512, ts)
    # ensure at least 2 tiles along shortest side
    while shortest // ts < 2 and ts > 128:
        ts //= 2
    return ts

def iter_tiles(y: np.ndarray, tile_size: int) -> Iterator[Tuple[int,int,np.ndarray,Tuple[int,int,int,int]]]:
    h, w = y.shape
    nx = (w + tile_size - 1) // tile_size
    ny = (h + tile_size - 1) // tile_size
    for iy in range(ny):
        for ix in range(nx):
            y0 = iy * tile_size
            x0 = ix * tile_size
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            yield ix, iy, y[y0:y1, x0:x1], (y0, y1, x0, x1) 