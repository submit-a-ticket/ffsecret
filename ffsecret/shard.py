"""Simple shard / parity helper for tile payloads."""
from typing import List, Tuple
import zlib
import math

HEADER_SIZE = 8  # bytes

def _crc16(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFF


def split_into_shards(payload: bytes, max_shard_payload: int) -> List[bytes]:
    shards = [payload[i:i + max_shard_payload] for i in range(0, len(payload), max_shard_payload)]
    return shards


def add_parity(shards: List[bytes], parity_count: int) -> List[bytes]:
    if parity_count == 0:
        return shards
    max_len = max(len(s) for s in shards)
    # pad shards equal
    shards_eq = [s.ljust(max_len, b"\0") for s in shards]
    for p in range(parity_count):
        parity = bytearray(max_len)
        for s in shards_eq:
            for i, b in enumerate(s):
                parity[i] ^= b
        shards.append(bytes(parity))
    return shards


def build_headers(shards: List[bytes], k_data: int) -> List[bytes]:
    out = []
    total_k = k_data
    for shard_id, shard_payload in enumerate(shards):
        is_parity = 1 if shard_id >= k_data else 0
        crc = _crc16(shard_payload)
        header = shard_id.to_bytes(2, "big") + total_k.to_bytes(2, "big") + bytes([is_parity, 0]) + crc.to_bytes(2, "big")
        out.append(header + shard_payload)
    return out


def shard_payload(payload: bytes, num_tiles: int, parity_ratio: float = 0.3, max_per_tile: int = 128) -> List[bytes]:
    data_shards = split_into_shards(payload, max_per_tile)
    k = len(data_shards)
    parity_count = max(1, int(math.ceil(k * parity_ratio)))
    shards = data_shards.copy()
    shards = add_parity(shards, parity_count)
    return build_headers(shards, k)


def parse_header(tile_bytes: bytes) -> Tuple[int, int, int, bytes]:
    header = tile_bytes[:HEADER_SIZE]
    shard_id = int.from_bytes(header[0:2], "big")
    total_k = int.from_bytes(header[2:4], "big")
    is_parity = header[4]
    crc = int.from_bytes(header[6:8], "big")
    payload = tile_bytes[HEADER_SIZE:]
    if _crc16(payload) != crc:
        raise ValueError("CRC mismatch")
    return shard_id, total_k, is_parity, payload 