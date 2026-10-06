#!/usr/bin/env python3
"""Converts a PEFT LoRA adapter for Qwen4Exp (Qwen3.8-Flash-Next) or Qwen3.5
(e.g. strands-decider's torso adapter) to a GGUF LoRA adapter in llama.cpp's
layout (`<tensor>.lora_a` [in, r] and
`<tensor>.lora_b` [r, out] in GGUF dimension order, adapter.lora.alpha).

llama.cpp's convert_lora_to_gguf.py cannot do this: the base converter
reorders Gated DeltaNet value heads along the input dimension of out_proj,
which its LoRA tensor wrapper cannot reshape. This tool reuses the base
converter's own transforms exactly, without re-implementing them. It pushes
an index-valued tensor with the base weight's shape through
`modify_tensors`, reads back which source row and column each output element
came from, and applies that row map to B and column map to A
(W = s * B @ A, so row transforms act on B and column transforms on A).
Non-separable transforms are rejected.

Usage: LLAMA_CPP=<llama.cpp> convert_lora_gguf.py <base-hf-dir> <peft-dir> <out.gguf>
The base model's architecture (config.json) picks the converter class.
"""
import json
import os
import re
import sys
from pathlib import Path

llama = os.environ["LLAMA_CPP"]
sys.path[:0] = [llama, os.path.join(llama, "gguf-py")]

import gguf  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from conversion import qwen, qwen4exp  # noqa: E402

CLASSES = {
    "Qwen4ExpForConditionalGeneration": (qwen4exp.Qwen4ExpTextModel, "qwen4exp"),
    "Qwen4ExpForCausalLM": (qwen4exp.Qwen4ExpTextModel, "qwen4exp"),
    "Qwen3_5ForConditionalGeneration": (qwen.Qwen3_5TextModel, "qwen35"),
    "Qwen3_5ForCausalLM": (qwen.Qwen3_5TextModel, "qwen35"),
}
from safetensors.torch import load_file  # noqa: E402


def main():
    base_dir, peft_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    cfg = json.load(open(peft_dir / "adapter_config.json"))
    if cfg.get("use_rslora") or cfg.get("use_dora"):
        raise SystemExit("rsLoRA/DoRA adapters are not supported")
    rank, alpha = int(cfg["r"]), float(cfg["lora_alpha"])

    hf_config = json.load(open(base_dir / "config.json"))
    arch = hf_config["architectures"][0]
    cls, gguf_arch = CLASSES[arch]
    # A base without MTP layers (e.g. the tiny test models) converts like
    # convert_hf_to_gguf.py --no-mtp; adapters never touch MTP tensors.
    if not hf_config.get("text_config", hf_config).get("mtp_num_hidden_layers") and getattr(cls, "supports_mtp_export", False):
        cls.no_mtp = True
    model = cls(base_dir, gguf.LlamaFileType.ALL_F32, out, eager=True, dry_run=True)
    types = model.hparams.get("layer_types") or []
    model.hparams["layer_types"] = ["full_attention" if t == "indexed_attention" else t for t in types]

    state = load_file(peft_dir / "adapter_model.safetensors")
    modules = sorted({k.split(".lora_A.")[0] for k in state if ".lora_A." in k})
    writer = gguf.GGUFWriter(out, gguf_arch)
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", alpha)

    count = 0
    for module in modules:
        A = state[module + ".lora_A.weight"].double()   # [r, in]
        B = state[module + ".lora_B.weight"].double()   # [out, r]
        if A.shape[0] != rank or B.shape[1] != rank:
            raise SystemExit(f"{module}: rank mismatch")
        # PEFT prefixes: base_model.model.model.layers... when attached to a
        # *ForCausalLM, base_model.model.layers... when attached to the bare
        # text model (strands-decider's torso).
        hf_name = re.sub(r"^base_model\.model\.(model\.)?", "model.", module) + ".weight"
        m = re.search(r"\.layers\.(\d+)\.", hf_name)
        bid = int(m.group(1)) if m else None
        rows_n, cols_n = B.shape[0], A.shape[1]
        probe = torch.arange(rows_n * cols_n, dtype=torch.float64).reshape(rows_n, cols_n)
        for gguf_name, t in model.modify_tensors(probe, hf_name, bid):
            t = torch.as_tensor(t).to(torch.float64)
            if t.ndim != 2:
                raise SystemExit(f"{hf_name} -> {gguf_name}: non-2D transform")
            src_rows = (t[:, 0] // cols_n).long()
            src_cols = (t[0, :] % cols_n).long()
            if not torch.equal(t, (src_rows[:, None] * cols_n + src_cols[None, :]).double()):
                raise SystemExit(f"{hf_name} -> {gguf_name}: transform is not a row/column map")
            a = A[:, src_cols].to(torch.float32).contiguous().numpy()   # [r, in']
            b = B[src_rows, :].to(torch.float32).contiguous().numpy()   # [out', r]
            name = gguf_name[:-len(".weight")] if gguf_name.endswith(".weight") else gguf_name
            writer.add_tensor(name + ".weight.lora_a", a)
            writer.add_tensor(name + ".weight.lora_b", b)
            count += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {count} adapted tensors (rank {rank}, alpha {alpha}) to {out}")


if __name__ == "__main__":
    main()
