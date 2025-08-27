import numpy as np
from PIL import Image
from reedsolo import RSCodec
from typing import Union, List, Tuple

# Pilot-related imports
from .pilot import generate_pilot, ALPHA_PILOT, PILOT_BITS

from .tiles import choose_tile_size, iter_tiles

__all__ = ["embed", "extract"]

# -----------------------------------------------------------------------------
# Tile helpers (reuse existing FFT logic)
# -----------------------------------------------------------------------------


def embed_tile(
    tile_arr: np.ndarray,
    payload_bytes: bytes,
    key: int,
    alpha: float,
    spread: int,
):
    """Embed pilot pattern **and** payload into a tile (in-place).

    Embedding order:
    1. Pilot pattern (single coefficient pair per bit, high *ALPHA_PILOT*).
    2. Data bits (replicated *spread* times, normal *alpha*).
    """

    # ------------- Prepare data bitstream -------------
    length_prefix = len(payload_bytes).to_bytes(2, "big")
    data_bits = _bytes_to_bits(length_prefix + payload_bytes)

    h, w = tile_arr.shape
    F = np.fft.fftshift(np.fft.fft2(tile_arr))

    # All candidate mid-band coordinates (one per conjugate pair)
    coords = _mid_band_coords(h, w)
    rng = np.random.default_rng(key)
    rng.shuffle(coords)

    # Reserve the first chunk for the pilot
    pilot_bits = generate_pilot(key).flatten().tolist()
    n_pilot = len(pilot_bits)  # == PILOT_BITS

    if n_pilot + len(data_bits) * spread > len(coords):
        raise ValueError("Tile payload too large (pilot + data)")

    pair_idx = 0

    # ------------- Embed pilot (no magnitude skipping) -------------
    for bit in pilot_bits:
        if pair_idx >= len(coords):
            raise RuntimeError("Ran out of coefficient pairs while embedding pilot bits")

        y_idx, x_idx = coords[pair_idx]
        pair_idx += 1

        y_sym = h - 1 - y_idx
        x_sym = w - 1 - x_idx

        # Ensure correct ordering
        mag1 = np.abs(F[y_idx, x_idx])
        mag2 = np.abs(F[y_sym, x_sym])
        if (bit == 1 and mag1 < mag2) or (bit == 0 and mag1 > mag2):
            F[y_idx, x_idx], F[y_sym, x_sym] = F[y_sym, x_sym], F[y_idx, x_idx]

        strength = ALPHA_PILOT
        if bit == 1:
            F[y_idx, x_idx] *= (1 + strength)
            F[y_sym, x_sym] *= (1 - strength)
        else:
            F[y_idx, x_idx] *= (1 - strength)
            F[y_sym, x_sym] *= (1 + strength)

    # Remaining coordinates for data start at pair_idx
    data_coords = coords[pair_idx:]

    # ------------- Embed data -------------
    pair_idx = 0
    skip_threshold = 5e-4  # must match extract_tile
    for bit in data_bits:
        used = 0
        while used < spread:
            if pair_idx >= len(data_coords):
                raise RuntimeError("Ran out of coefficient pairs while embedding data bits")

            y_idx, x_idx = data_coords[pair_idx]
            pair_idx += 1

            y_sym = h - 1 - y_idx
            x_sym = w - 1 - x_idx

            mag1 = np.abs(F[y_idx, x_idx])
            mag2 = np.abs(F[y_sym, x_sym])

            if max(mag1, mag2) < skip_threshold:
                continue  # decoder will skip this too; keep alignment

            # Ensure ordering matches bit
            if (bit == 1 and mag1 < mag2) or (bit == 0 and mag1 > mag2):
                F[y_idx, x_idx], F[y_sym, x_sym] = F[y_sym, x_sym], F[y_idx, x_idx]

            # Reinforce ordering
            if bit == 1:
                F[y_idx, x_idx] *= (1 + alpha)
                F[y_sym, x_sym] *= (1 - alpha)
            else:
                F[y_idx, x_idx] *= (1 - alpha)
                F[y_sym, x_sym] *= (1 + alpha)

            used += 1

    # ------------- Back to spatial domain -------------
    tile_arr[:, :] = np.real(np.fft.ifft2(np.fft.ifftshift(F)))


# Skip-type hints unchanged
def extract_tile(tile_arr: np.ndarray, key: int, alpha: float, spread: int):
    """Attempt to extract payload bytes from tile, returns bytes or None."""
    h, w = tile_arr.shape
    F = np.fft.fftshift(np.fft.fft2(tile_arr))
    coords_full = _mid_band_coords(h, w)
    rng = np.random.default_rng(key)
    rng.shuffle(coords_full)

    # Skip pilot coordinates – they occupy the first PILOT_BITS entries.
    from .pilot import PILOT_BITS  # local import to avoid cycle at top
    coords = coords_full[PILOT_BITS:]
    skip_threshold = 5e-4

    def read_bits(n_bits: int, start_idx: int = 0):
        bits = []
        idx = start_idx
        for _ in range(n_bits):
            votes = []
            while len(votes) < spread and idx < len(coords):
                y_idx, x_idx = coords[idx]
                idx += 1
                y_sym = h - 1 - y_idx
                x_sym = w - 1 - x_idx
                c1 = np.abs(F[y_idx, x_idx])
                c2 = np.abs(F[y_sym, x_sym])
                if max(c1, c2) < skip_threshold:
                    continue
                votes.append(1 if c1 > c2 else 0)
            if len(votes) < spread:
                raise ValueError("Not enough votes")
            bits.append(1 if sum(votes) > (spread // 2) else 0)
        return bits, idx

    try:
        length_bits, cursor = read_bits(16)
        length = int("".join(str(b) for b in length_bits), 2)
        if length == 0 or length > 4096:
            # Likely a false-positive tile; signal caller to skip
            return None
        total_bits = (length) * 8
        payload_bits, _ = read_bits(total_bits, cursor)
        payload = _bits_to_bytes(payload_bits)
        return payload
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _to_y_channel(img: Image.Image):
    """Return Y channel as float32 array and the untouched Cb/Cr images."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    return np.asarray(y, dtype=np.float32), cb, cr


def _merge_y_channel(y_channel: np.ndarray, cb: Image.Image, cr: Image.Image) -> Image.Image:
    """Recombine modified Y channel with original chroma and return RGB image."""
    y_img = Image.fromarray(np.uint8(np.clip(y_channel, 0, 255)), mode="L")
    img_ycbcr = Image.merge("YCbCr", (y_img, cb, cr))
    return img_ycbcr.convert("RGB")


def _bytes_to_bits(data: bytes) -> List[int]:
    """Convert bytes to list of bits (big-endian)."""
    bits: List[int] = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_bytes(bits: List[int]) -> bytes:
    """Convert bit list back to bytes."""
    if len(bits) % 8:
        raise ValueError("Bit list length must be multiple of 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        out.append(byte)
    return bytes(out)


# -----------------------------------------------------------------------------
# Frequency domain coordinate helper
# -----------------------------------------------------------------------------


def _mid_band_coords(h: int, w: int) -> np.ndarray:
    """Return coordinates in mid-frequency band, one per conjugate pair.

    The returned list only contains coordinates in the upper-left half-plane
    (excluding the exact centre line) so that each selected \(y, x\) has a
    unique conjugate partner \(h-1-y, w-1-x\). This avoids double-modifying
    coefficients which would otherwise corrupt the watermark.
    """
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    low = 0.15 * min(h, w) / 2.0
    high = 0.45 * min(h, w) / 2.0
    mask = (dist >= low) & (dist <= high)

    # Keep only one half of the symmetric spectrum (y < cy OR (y == cy AND x < cx))
    half_plane = (Y < cy) | ((Y == cy) & (X < cx))
    mask &= half_plane

    coords = np.column_stack(np.nonzero(mask))
    return coords


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def embed(
    input_path: str,
    output_path: str,
    data: Union[str, bytes],
    key: int,
    *,
    alpha: float = 0.1,
    spread: int = 5,
    crop_safe: bool = False,
) -> None:
    """Embed *data* into *input_path* and write result to *output_path*.

    Args:
        input_path: Path to the cover image (read).
        output_path: Where to save the stego-image.
        data: Bytes or UTF-8 string payload to hide.
        key: Integer embedding key (seed for PRNG).
        alpha: Strength factor (0 < alpha < 1).
        spread: Number of coefficient pairs used per payload bit.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    if crop_safe:
        # ------------------------------------------------------------------
        # Crop-robust tiled embedding (replicate full payload per tile)
        # ------------------------------------------------------------------
        img = Image.open(input_path)
        y_orig, cb, cr = _to_y_channel(img)

        # RS-encode once
        rsc = RSCodec(32)
        encoded = rsc.encode(data)

        for tile_size in (128, 256, 512):
            y = y_orig.copy()
            success_tiles = 0
            for _, _, tile_arr, _ in iter_tiles(y, tile_size):
                try:
                    embed_tile(tile_arr, encoded, key, alpha, spread)
                    success_tiles += 1
                except ValueError:
                    # Tile too smooth / small, skip it
                    continue
            if success_tiles > 0:
                out_img = _merge_y_channel(y, cb, cr)
                out_img.save(output_path)
                return

        raise ValueError("Payload too large even for 512px tiles; reduce data size or spread.")

    # Forward-error-correction (Reed-Solomon)
    rsc = RSCodec(32)  # 32 RS parity bytes -> can correct up to 16 bytes
    encoded = rsc.encode(data)

    # Prepend payload length (16-bit big-endian)
    length_prefix = len(data).to_bytes(2, "big")
    bitstream: List[int] = _bytes_to_bits(length_prefix + encoded)

    # ------------------------------------------------------------------
    # Image → frequency domain
    # ------------------------------------------------------------------
    try:
        img = Image.open(input_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cover image not found: {input_path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to open image '{input_path}': {e}") from e
    y, cb, cr = _to_y_channel(img)
    h, w = y.shape

    F = np.fft.fftshift(np.fft.fft2(y))

    # Candidate coordinates (one per conjugate pair)
    coords = _mid_band_coords(h, w)

    needed_pairs = len(bitstream) * spread

    # Shuffle deterministically with secret key
    rng = np.random.default_rng(key)
    rng.shuffle(coords)

    # Threshold for negligible coefficients (skip during embedding)
    skip_threshold = 1e-7  # constant threshold

    # Final capacity check (optimistic – skipping may advance pair_idx further)
    if needed_pairs > len(coords):
        raise ValueError("Payload too large for this image or parameters. Try smaller data or lower spread.")

    # ------------------------------------------------------------------
    # Embed bits
    # ------------------------------------------------------------------
    pair_idx = 0
    for bit in bitstream:
        used = 0
        while used < spread:
            if pair_idx >= len(coords):
                raise RuntimeError("Ran out of coefficient pairs while embedding; try lower payload or spread.")

            y_idx, x_idx = coords[pair_idx]
            pair_idx += 1

            y_sym = h - 1 - y_idx
            x_sym = w - 1 - x_idx

            mag1 = np.abs(F[y_idx, x_idx])
            mag2 = np.abs(F[y_sym, x_sym])

            if max(mag1, mag2) < skip_threshold:
                continue  # too weak, skip this pair

            # Ensure correct ordering (bit 1 => mag1 > mag2)
            if (bit == 1 and mag1 < mag2) or (bit == 0 and mag1 > mag2):
                F[y_idx, x_idx], F[y_sym, x_sym] = F[y_sym, x_sym], F[y_idx, x_idx]
                mag1, mag2 = mag2, mag1

            # Reinforce ordering
            if bit == 1:
                F[y_idx, x_idx] *= (1.0 + alpha)
                F[y_sym, x_sym] *= (1.0 - alpha)
            else:
                F[y_idx, x_idx] *= (1.0 - alpha)
                F[y_sym, x_sym] *= (1.0 + alpha)

            used += 1

    # ------------------------------------------------------------------
    # Back to spatial domain and save
    # ------------------------------------------------------------------
    y_stego = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    y_stego = np.clip(y_stego, 0, 255)

    out_img = _merge_y_channel(y_stego, cb, cr)
    out_img.save(output_path)


def extract(
    input_path: str,
    key: int,
    *,
    alpha: float = 0.1,
    spread: int = 5,
    **kwargs,
) -> bytes:
    """Extract hidden payload. Tries multiple orientations (flip / rotate)."""
    crop_safe = kwargs.pop("crop_safe", False) if isinstance(locals().get('kwargs', {}), dict) else False

    def _attempt_full(im: Image.Image):
        y, _, _ = _to_y_channel(im)
        h, w = y.shape

        F = np.fft.fftshift(np.fft.fft2(y))

        coords = _mid_band_coords(h, w)

        rng = np.random.default_rng(key)
        rng.shuffle(coords)

        skip_threshold = 5e-4

        def _read_bits(n_bits: int, start_idx: int = 0):
            bits: List[int] = []
            idx = start_idx
            for _ in range(n_bits):
                votes = []
                while len(votes) < spread:
                    if idx >= len(coords):
                        raise ValueError("Insufficient coefficient pairs while extracting – likely wrong key or damaged image.")

                    y_idx, x_idx = coords[idx]
                    idx += 1
                    y_sym = h - 1 - y_idx
                    x_sym = w - 1 - x_idx
                    c1 = np.abs(F[y_idx, x_idx])
                    c2 = np.abs(F[y_sym, x_sym])
                    if max(c1, c2) < skip_threshold:
                        continue
                    votes.append(1 if c1 > c2 else 0)
                bits.append(1 if sum(votes) > (spread // 2) else 0)
            return bits, idx

        length_bits, cursor = _read_bits(16, 0)
        length = int("".join(str(b) for b in length_bits), 2)
        if length == 0 or length > 4096:
            raise ValueError(
                "Watermark length header invalid – image may be corrupted or key/preset mismatch.")
        required_pairs = (16 + (length + 32) * 8) * spread
        if required_pairs > len(coords):
            raise ValueError("implausible length")

        total_bits = (length + 32) * 8
        payload_bits, _ = _read_bits(total_bits, cursor)
        coded_bytes = _bits_to_bytes(payload_bits)

        rsc = RSCodec(32)
        decoded_tuple = rsc.decode(coded_bytes)
        decoded_msg = decoded_tuple[0] if isinstance(decoded_tuple, tuple) else decoded_tuple
        if len(decoded_msg) == 0:
            return None  # false positive
        return bytes(decoded_msg)

    if not crop_safe:
        def _attempt(im: Image.Image):
            return _attempt_full(im)

        # First attempt original orientation
        try:
            with Image.open(input_path) as base_img:
                return _attempt(base_img)
        except Exception:
            pass

        # Try fallback orientations as before
        with Image.open(input_path) as base_img:
            orientations = [
                base_img.transpose(Image.FLIP_LEFT_RIGHT),
                base_img.transpose(Image.FLIP_TOP_BOTTOM),
                base_img.rotate(90, expand=True),
                base_img.rotate(180, expand=True),
                base_img.rotate(270, expand=True),
            ]
            for im in orientations:
                try:
                    return _attempt(im)
                except Exception:
                    continue
        raise ValueError("Failed to extract payload – image may have been heavily modified or wrong key.")
    else:
        # Crop-safe path with pilot-based alignment search
        from .align import locate_pilot  # local import to avoid circular dependency

        with Image.open(input_path) as img:
            y, _, _ = _to_y_channel(img)
            F_full = np.fft.fftshift(np.fft.fft2(y))  # compute once

            rsc = RSCodec(32)

            for tile_size in (512, 256, 128):
                if tile_size > min(y.shape):
                    continue

                dy, dx, conf = locate_pilot(F_full, key, tile_size)

                # Empirical threshold – may be tuned; require positive margin
                if conf < 0.02:  # fallback to brute search if low confidence
                    align_offsets = None
                else:
                    align_offsets = (dy, dx)

                searched_any = False

                if align_offsets is not None:
                    dy0, dx0 = align_offsets
                    for y0 in range(dy0, y.shape[0] - tile_size + 1, tile_size):
                        for x0 in range(dx0, y.shape[1] - tile_size + 1, tile_size):
                            tile_arr = y[y0 : y0 + tile_size, x0 : x0 + tile_size]
                            payload = extract_tile(tile_arr, key, alpha, spread)
                            if not payload:
                                continue
                            try:
                                decoded_tuple = rsc.decode(payload)
                                decoded_msg = (
                                    decoded_tuple[0]
                                    if isinstance(decoded_tuple, tuple)
                                    else decoded_tuple
                                )
                                if len(decoded_msg):
                                    return bytes(decoded_msg)
                            except Exception:
                                continue
                            searched_any = True

                # Fallback brute-force window search (as previous version) if nothing found
                phase_steps = [0, tile_size // 4, tile_size // 2, (3 * tile_size) // 4]
                step = tile_size // 2
                for dy_f in phase_steps:
                    for dx_f in phase_steps:
                        for y0 in range(dy_f, y.shape[0] - tile_size + 1, step):
                            for x0 in range(dx_f, y.shape[1] - tile_size + 1, step):
                                tile_arr = y[y0 : y0 + tile_size, x0 : x0 + tile_size]
                                payload = extract_tile(tile_arr, key, alpha, spread)
                                if not payload:
                                    continue
                                try:
                                    decoded_tuple = rsc.decode(payload)
                                    decoded_msg = (
                                        decoded_tuple[0]
                                        if isinstance(decoded_tuple, tuple)
                                        else decoded_tuple
                                    )
                                    if len(decoded_msg):
                                        return bytes(decoded_msg)
                                except Exception:
                                    continue

        raise ValueError("Crop-safe extraction failed – no valid tile found.") 