#!/usr/bin/env python3
"""Dequantization fixtures for halo::quant against llama.cpp's gguf-py (numpy).

    LLAMA_CPP=<llama.cpp checkout> python3 tools/gen_quant_fixtures.py > tests/core/data/quant.tsv

quant.tsv: type  block_hex  floats_hex
  One block (or 32 halves for F16/BF16) per line; floats_hex is the
  little-endian float32 bits of every dequantized element, as gguf-py computes
  them. The C++ dequantizers share helpers with the fused dot products, so the
  property that the two agree cannot catch a slip in a shared helper; this
  independent reference can.
"""
import os
import random
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.environ['LLAMA_CPP'], 'gguf-py'))
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as T  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

SEED = 20261004
BLOCKS = 12

# Byte offsets of each block's fp16 scale fields (ggml-common.h layouts).
SCALES = {
    T.Q8_0: [0], T.Q5_1: [0, 2], T.Q4_K: [0, 2], T.Q5_K: [0, 2],
    T.Q6_K: [208], T.Q3_K: [108], T.IQ4_NL: [0], T.IQ4_XS: [0], T.IQ3_S: [0],
}


def scale(rng, i):
    """A finite fp16 scale; the first blocks pin the sign and subnormal cases."""
    sign = (i & 1) if i < 4 else rng.getrandbits(1)
    if i in (2, 3):
        return (sign << 15) | rng.randrange(1, 1024)  # subnormal
    return (sign << 15) | (rng.randrange(5, 19) << 10) | rng.randrange(1024)


def main():
    rng = random.Random(SEED)
    out = sys.stdout
    for t, offs in SCALES.items():
        _, size = GGML_QUANT_SIZES[t]
        for i in range(BLOCKS):
            b = bytearray(rng.getrandbits(8) for _ in range(size))
            for o in offs:
                b[o:o + 2] = struct.pack('<H', scale(rng, i + offs.index(o)))
            vals = dequantize(np.frombuffer(bytes(b), dtype=np.uint8), t).astype('<f4')
            out.write(f'{t.name}\t{b.hex()}\t{vals.tobytes().hex()}\n')
    for t in (T.F16, T.BF16):
        for i in range(BLOCKS):
            halves = []
            while len(halves) < 32:
                h = rng.getrandbits(16)
                if (h >> 10) & 31 == 31 and t == T.F16:
                    continue  # no inf/NaN: a NaN payload is not a value to compare
                if (h >> 7) & 255 == 255 and t == T.BF16:
                    continue
                halves.append(h)
            b = struct.pack('<32H', *halves)
            if t == T.F16:
                vals = np.frombuffer(b, dtype='<f2').astype('<f4')
            else:  # gguf-py's BF16 path widens the bits exactly like this
                vals = (np.frombuffer(b, dtype='<u2').astype('<u4') << 16).view('<f4')
            out.write(f'{t.name}\t{b.hex()}\t{vals.tobytes().hex()}\n')


if __name__ == '__main__':
    main()
