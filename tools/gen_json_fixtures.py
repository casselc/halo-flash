#!/usr/bin/env python3
"""Fixtures for halo::json against CPython's json module.

json.tsv: input_hex  status  dumps_indent2_hex  dumps_compact_hex
  status is "ok" when json.loads accepts the input (and the input contains no
  lone surrogates, which halo rejects by design), else "err".
float.tsv: double_bits_hex  repr  (Python repr of the float as json emits it)
"""
import json
import random
import struct
import sys

SEED = 20261003


def has_lone_surrogate(v):
    if isinstance(v, str):
        return any(0xD800 <= ord(c) <= 0xDFFF for c in v)
    if isinstance(v, list):
        return any(has_lone_surrogate(x) for x in v)
    if isinstance(v, dict):
        return any(has_lone_surrogate(k) or has_lone_surrogate(x) for k, x in v.items())
    return False


def rand_value(rng, depth=0):
    r = rng.random()
    if depth > 3 or r < 0.35:
        k = rng.randrange(8)
        if k == 0: return None
        if k == 1: return rng.random() < 0.5
        if k == 2: return rng.randrange(-10**20, 10**20)
        if k == 3: return rng.choice([0.0, -0.0, 1.0, 0.1, 1e-5, 1e16, 1e15, 123456789.125,
                                      2.5e-308, 1.7976931348623157e308, 5e-324, -3.25, 1/3])
        if k == 4: return struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if k == 5: return rng.uniform(-1e6, 1e6)
        return "".join(rng.choice(["a", "é", "\n", "\t", '"', "\\", "\x01", "中", "🦙", "/", " ", " "])
                       for _ in range(rng.randrange(0, 8)))
    if r < 0.65:
        return [rand_value(rng, depth + 1) for _ in range(rng.randrange(0, 4))]
    return {rng.choice(["a", "b", "key", "é", "x y", ""]): rand_value(rng, depth + 1)
            for _ in range(rng.randrange(0, 4))}


def main(out):
    rng = random.Random(SEED)
    cases = [
        '{}', '[]', '1', '-0', '0.0', '-0.0', '1e2', '1E-7', '1.5e300', '1e400', '-1e400',
        '123456789012345678901234567890', '3.141592653589793', 'NaN', 'Infinity', '-Infinity',
        '"\\u00e9\\ud83e\\udd99\\u2028"', '"\\ud800"', '"\\udc00x"', '{"a":1,"a":2,"b":3}',
        '[1,2,]', '{"a":1,}', '01', '1.', '.5', '+1', '"\x01"', '"\\x"', '[', '', ' 1 ',
        '{"nested":{"deep":[1,[2,[3,{"x":null}]]]}}', '"\xff"', '"\xc3\xa9"', 'true false',
        '{"a" 1}', '[1 2]', '"\\/"', '1e', '1e+', '-', '--1', '"\t"',
        # Escapes, \u case and the UTF-8 length boundaries (added for Mull survivors).
        '"\\b\\f\\n\\r\\t\\"\\\\\\/"', '"\\u00E9\\u00c9\\uABCD\\uFFFF\\uD83E\\uDD99"',
        '"\\u0041\\u007f\\u0080\\u07ff\\u0800\\uffff"', '"\\ud800\\udc00"', '"\\udbff\\udfff"',
        '"\\ud83e\\udbff"', '"\\ud83e\\ue000"', '"\\udfff"', '"\\udc00"', '"\\u12"', '"\\u12G4"',
        '"\\ud83e"', '"\\ud83e\\u0041"', '"\\', '"abc', '[1,{"a":[]},"x",{}]', '{"":[{}]}',
        '[ 1 , 2 ]', '{ "a" : 1 , "b" : [ ] }', ' { } ', '[\n]', '[1,\n2]\r\n', '{"a":1 , }',
        '1E+2', '-1.5E-3', '1e-7', '1.5e+300', '-0e0', '[-]', '[1e]', 'tru', 'nul', 'nulL',
        '[true,false,null]', '"\\u00aa\\u0a0A\\uFEFF\\uabcd"', '"\\u00a', '"\\u00af"',
        b'"\x7f"', b'"\xc2\x80"', b'"\xdf\xbf"', b'"\xc1\xbf"', b'"\xc0\x80"', b'"\xe0\x80\x80"',
        b'"\xe0\xa0\x80"', b'"\xef\xbf\xbf"', b'"\xed\x9f\xbf"', b'"\xed\xa0\x80"', b'"\xed\xbf\xbf"',
        b'"\xee\x80\x80"', b'"\xf0\x90\x80\x80"', b'"\xf0\x8f\xbf\xbf"', b'"\xf4\x8f\xbf\xbf"',
        b'"\xf4\x90\x80\x80"', b'"\xf5\x80\x80\x80"', b'"\xe2\x82"', b'"\xc3"', b'"\xc3\x28"',
        b'"\xe2\x28\xa1"', b'"\xf0\x90\x28\xbc"', b'"\x80"', b'"\xf8\x88\x80\x80\x80"',
    ]
    for _ in range(1500):
        v = rand_value(rng)
        cases.append(json.dumps(v, ensure_ascii=rng.random() < 0.3,
                                indent=rng.choice([None, 1, 2]), allow_nan=True))
    with open(f"{out}/json.tsv", "w") as f:
        f.write("# CPython json\n")
        for c in cases:
            raw = c.encode("utf-8", "surrogatepass") if isinstance(c, str) else c
            try:
                v = json.loads(raw.decode("utf-8"))
                ok = not has_lone_surrogate(v)
            except Exception:
                ok = False
            if ok:
                a = json.dumps(v, indent=2, ensure_ascii=False).encode()
                b = json.dumps(v, ensure_ascii=False).encode()
                f.write(f"{raw.hex()}\tok\t{a.hex()}\t{b.hex()}\n")
            else:
                f.write(f"{raw.hex()}\terr\t\t\n")
    with open(f"{out}/float.tsv", "w") as f:
        f.write("# CPython float repr via json.dumps\n")
        for _ in range(20000):
            bits = rng.getrandbits(64)
            d = struct.unpack("<d", struct.pack("<Q", bits))[0]
            f.write(f"{bits:016x}\t{json.dumps(d)}\n")
        for d in [1e16, 1e15, 9999999999999998.0, 1e-4, 1e-5, 0.0001, 0.00012, 123.0, 1e22, 1.5, 5e-324]:
            bits = struct.unpack("<Q", struct.pack("<d", d))[0]
            f.write(f"{bits:016x}\t{json.dumps(d)}\n")


if __name__ == "__main__":
    main(sys.argv[1])
