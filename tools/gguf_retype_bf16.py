#!/usr/bin/env python3
"""Copies a GGUF, converting the F32 tensors whose names match a regex to
BF16 (round to nearest even). llama-quantize refuses to touch QSA indexer
projections, which UD-Q4_K_XL stores in BF16; tools/make_small_quant.sh uses
this to match that.

  LLAMA_CPP=<llama.cpp> gguf_retype_bf16.py <in.gguf> <out.gguf> <name-regex>
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.environ["LLAMA_CPP"], "gguf-py"))
import gguf  # noqa: E402


def main():
    src, dst, pattern = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
    reader = gguf.GGUFReader(src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    data = {}
    converted = 0
    for t in reader.tensors:
        d, ty = t.data, t.tensor_type
        if ty == gguf.GGMLQuantizationType.F32 and pattern.search(t.name):
            d, ty = gguf.quants.quantize(d, gguf.GGMLQuantizationType.BF16), gguf.GGMLQuantizationType.BF16
            converted += 1
        data[t.name] = d
        writer.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, ty)
    if converted == 0:
        sys.exit(f"no F32 tensor matches {pattern.pattern}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(data[t.name], tensor_endianess=reader.endianess)
    writer.close()
    print(f"{converted} tensors converted to BF16")


if __name__ == "__main__":
    main()
