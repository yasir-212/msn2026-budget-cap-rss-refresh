"""Information-theoretic Wegman--Carter authentication for BOLT-RSS v3.

The universal family is polynomial evaluation over GF(2^128), represented
with the irreducible polynomial x^128+x^7+x^2+x+1.  A session consumes a
fresh 128-bit field key and a fresh 128-bit one-time pad.  Every 256-bit
string is valid authentication material, so 256 extracted uniform bits can
replenish the next session without rejection sampling or a computational KDF.

For a message encoded as B field coefficients, a nonzero difference
polynomial has at most B roots, hence the substitution-forgery probability
is at most B/2^128.
"""
from __future__ import annotations

import os
import numpy as np

_MASK128 = (1 << 128) - 1
# Lower terms of x^128+x^7+x^2+x+1.
_REDUCTION = (1 << 7) | (1 << 2) | (1 << 1) | 1
AUTH_KEY_BITS = 256
AUTH_TAG_BITS = 128
_BLOCK_BYTES = 16


def _gf_mul(a: int, b: int) -> int:
    a &= _MASK128
    b &= _MASK128
    out = 0
    while b:
        if b & 1:
            out ^= a
        b >>= 1
        carry = a >> 127
        a = (a << 1) & _MASK128
        if carry:
            a ^= _REDUCTION
    return out & _MASK128


def _coefficients(message: bytes) -> list[int]:
    # Full 16-byte chunks are field elements.  A final length element makes
    # the encoding prefix-free, including messages ending in zero bytes.
    coeffs = [int.from_bytes(message[i:i + _BLOCK_BYTES].ljust(_BLOCK_BYTES, b"\0"), "big")
              for i in range(0, len(message), _BLOCK_BYTES)]
    coeffs.append(len(message) & _MASK128)
    return coeffs


def poly_hash_gf128(key: int, message: bytes) -> int:
    acc = 0
    r = int(key) & _MASK128
    for coeff in _coefficients(message):
        acc = _gf_mul(acc ^ int(coeff), r)
    return acc


def wc_tag(hash_key: int, pad: int, message: bytes) -> int:
    return (poly_hash_gf128(hash_key, message) ^ (int(pad) & _MASK128)) & _MASK128


def wc_verify(hash_key: int, pad: int, message: bytes, tag: int) -> bool:
    return wc_tag(hash_key, pad, message) == (int(tag) & _MASK128)


def tag_to_bytes(tag: int) -> bytes:
    return int(tag).to_bytes(16, "big")


def parse_auth_material(material: bytes) -> tuple[int, int]:
    if len(material) != 32:
        raise ValueError("authentication material must be exactly 32 bytes")
    return int.from_bytes(material[:16], "big"), int.from_bytes(material[16:], "big")


def fresh_auth_material(rng: np.random.Generator | None = None) -> bytes:
    if rng is None:
        return os.urandom(32)
    return rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()


def forgery_bound(message_len_bytes: int) -> float:
    blocks = max(1, (int(message_len_bytes) + _BLOCK_BYTES - 1) // _BLOCK_BYTES) + 1
    return float(blocks / (1 << 128))
