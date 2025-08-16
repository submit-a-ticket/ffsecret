import numpy as np
from PIL import Image
from reedsolo import RSCodec
from typing import Union, List, Tuple

__all__ = ["embed", "extract"]


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
    skip_threshold = 5e-4  # constant threshold

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
) -> bytes:
    """Extract hidden payload from *input_path* using *key*.

    Returns the raw payload bytes (without length header or FEC).
    """
    try:
        img = Image.open(input_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Stego image not found: {input_path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to open image '{input_path}': {e}") from e
    y, _, _ = _to_y_channel(img)
    h, w = y.shape

    F = np.fft.fftshift(np.fft.fft2(y))

    # Recreate same coordinate list
    coords = _mid_band_coords(h, w)

    # No magnitude sub-selection to ensure coordinate sets match embed

    rng = np.random.default_rng(key)
    rng.shuffle(coords)

    skip_threshold = 5e-4

    # Internal helper to read *n* bits from spectrum
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
                    continue  # skip weak pair
                votes.append(1 if c1 > c2 else 0)

            bits.append(1 if sum(votes) > (spread // 2) else 0)

        return bits, idx

    # Read 16-bit length prefix first
    length_bits, cursor = _read_bits(16, 0)
    length = int("".join(str(b) for b in length_bits), 2)

    # Sanity-check length against capacity to guard against corrupted length bits
    max_pairs = len(coords)
    required_pairs = (16 + (length + 32) * 8) * spread
    if required_pairs > max_pairs:
        raise ValueError(
            "Extracted length header is implausible – likely wrong key or heavy image damage."
        )

    # Now read RS-encoded payload of (length + 32) bytes
    total_bits = (length + 32) * 8
    payload_bits, _ = _read_bits(total_bits, cursor)

    coded_bytes = _bits_to_bytes(payload_bits)

    # RS decode
    rsc = RSCodec(32)
    try:
        decoded_tuple = rsc.decode(coded_bytes)
        decoded_msg = decoded_tuple[0] if isinstance(decoded_tuple, tuple) else decoded_tuple
        decoded = bytes(decoded_msg)
    except Exception as exc:
        raise ValueError("Reed-Solomon decoding failed – image may be corrupted or wrong key.") from exc

    # Strip any padding and return exactly *length* bytes
    return decoded[:length] 