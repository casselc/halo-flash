#!/usr/bin/env python3
"""Builds a tiny random Qwen3.5 dense text model with HF transformers
(modeling_qwen3_5.py) and records its logits, as the reference for
halo::model::Qwen35Reference.

  make_tiny_qwen35_reference.py <qwen3.5 tokenizer dir> <out-dir>
  python <llama.cpp>/convert_hf_to_gguf.py <out-dir>/hf --outtype f32 --outfile <out-dir>/tiny-qwen35-f32.gguf
"""
import json
import math
import os
import shutil
import sys

import torch
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

SEED = 20261006
T = 32


def main():
    tok_dir, out = sys.argv[1], sys.argv[2]
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    cfg = Qwen3_5TextConfig(
        vocab_size=248320, hidden_size=64, intermediate_size=96, num_hidden_layers=8,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, rms_norm_eps=1e-6, tie_word_embeddings=True,
        max_position_embeddings=4096, eos_token_id=248044, pad_token_id=None,
    )
    model = mod.Qwen3_5ForCausalLM(cfg).eval().to(torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.copy_(torch.empty_like(p).uniform_(-0.3, 0.3, generator=gen)
                        + (1.0 if name.endswith("linear_attn.norm.weight") else 0.0))
            elif name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(1.0, 16.0, generator=gen)))
            elif name.endswith("dt_bias") or "conv1d" in name:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif "embed_tokens" in name:
                p.copy_(torch.randn(p.shape, generator=gen))
            elif p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=gen) / math.sqrt(p.shape[-1]))
    hf = os.path.join(out, "hf")
    model.save_pretrained(hf, safe_serialization=True)
    c = json.load(open(os.path.join(hf, "config.json")))
    c["architectures"] = ["Qwen3_5ForCausalLM"]
    json.dump(c, open(os.path.join(hf, "config.json"), "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(tok_dir, f)):
            shutil.copy(os.path.join(tok_dir, f), hf)
    g = torch.Generator().manual_seed(SEED + 1)
    ids = torch.randint(0, 248000, (1, T), generator=g)
    with torch.no_grad():
        o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    ids[0].to(torch.int32).numpy().tofile(os.path.join(out, "tokens.i32"))
    o.logits[0].float().numpy().tofile(os.path.join(out, "logits.f32"))
    o.hidden_states[-1][0].float().numpy().tofile(os.path.join(out, "hidden.f32"))
    json.dump({"seed": SEED, "T": T, "transformers": transformers.__version__}, open(os.path.join(out, "meta.json"), "w"))
    print("ok")


if __name__ == "__main__":
    main()
