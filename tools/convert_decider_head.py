#!/usr/bin/env python3
"""Packs a strands-decider checkpoint's readout head (head.safetensors) and
its inference settings (hobson_config.json / strands_decider_config.json)
into a small GGUF, so halo-flash loads everything through one format.

Tensors (F32): head.norm.{weight,bias}, head.q.{weight,bias}, head.k.{weight,bias}
Metadata: decider.head_type, decider.pointer_dim, decider.temperature,
decider.temperature.{noul,choice,score}, decider.ordinal_smoothing,
decider.max_length, decider.base_model.

Usage: LLAMA_CPP=<llama.cpp> convert_decider_head.py <checkpoint-dir> <out.gguf>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.environ["LLAMA_CPP"], "gguf-py"))
import gguf  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    cfg_path = next(p for p in (os.path.join(ckpt, "strands_decider_config.json"),
                                os.path.join(ckpt, "hobson_config.json")) if os.path.exists(p))
    cfg = json.load(open(cfg_path))
    if cfg.get("head_type") != "pointer":
        raise SystemExit("only pointer heads are supported")
    head = load_file(os.path.join(ckpt, "head.safetensors"))
    w = gguf.GGUFWriter(out, "decider-head")
    w.add_type("decider-head")
    w.add_string("decider.head_type", "pointer")
    w.add_uint32("decider.pointer_dim", int(cfg["pointer_dim"]))
    w.add_float32("decider.temperature", float(cfg.get("temperature", 1.0)))
    for kind, t in (cfg.get("temperature_by_kind") or {}).items():
        w.add_float32(f"decider.temperature.{kind}", float(t))
    w.add_float32("decider.ordinal_smoothing", float(cfg.get("ordinal_smoothing", 0.0)))
    w.add_uint32("decider.max_length", int(cfg.get("max_length", 4096)))
    w.add_string("decider.base_model", cfg["base_model"])
    for name in ("norm.weight", "norm.bias", "q.weight", "q.bias", "k.weight", "k.bias"):
        w.add_tensor("head." + name, head[name].astype("float32"))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
