#!/usr/bin/env python3
"""Attaches a random PEFT LoRA to the tiny Flash-Next made by
make_tiny_hf_reference.py and records HF + PEFT logits as the reference for
halo-flash's LoRA path.

  make_tiny_lora_reference.py <tiny-dir>
  LLAMA_CPP=... tools/convert_qwen4exp_lora.py --base <tiny-dir>/hf \
      <tiny-dir>/lora --outtype f32 --outfile <tiny-dir>/lora-f32.gguf

Writes <tiny-dir>/lora/ (adapter_config.json, adapter_model.safetensors) and
<tiny-dir>/logits_lora.f32 ([T, vocab], same tokens as tokens.i32).
PEFT initializes B to zero, which would make the adapter a no-op; both A and B
are randomized so every adapted projection changes the output.
"""
import json
import os
import sys

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers.models.qwen4_exp import configuration_qwen4_exp as cfgmod
from transformers.models.qwen4_exp import modeling_qwen4_exp as mod

SEED = 20261004
TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",                      # gated attention
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",  # Gated DeltaNet
    "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj",
]


def load_tiny(tiny):
    cfg = json.load(open(os.path.join(tiny, "hf", "config.json")))
    cfg.pop("architectures", None)
    cfg.pop("torch_dtype", None)
    config = cfgmod.Qwen4ExpTextConfig(**{k: v for k, v in cfg.items() if not k.startswith("_")})
    model = mod.Qwen4ExpForCausalLM(config).eval().to(torch.float32)
    state = load_file(os.path.join(tiny, "hf", "model.safetensors"))
    merged = {}
    shards = {}
    for k, v in state.items():
        k2 = k.replace("model.language_model.", "model.", 1)
        if ".ngram_embedding.shard_" in k2:
            base, idx = k2.split(".shard_")
            shards.setdefault(base, {})[int(idx.split(".")[0])] = v
        else:
            merged[k2] = v
    for base, parts in shards.items():
        merged[base + ".weight"] = torch.cat([parts[i] for i in sorted(parts)])
    missing, unexpected = model.load_state_dict(merged, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    return model


def main():
    tiny = sys.argv[1]
    model = load_tiny(tiny)
    ids = torch.from_numpy(np.fromfile(os.path.join(tiny, "tokens.i32"), dtype=np.int32)).long().view(1, -1)
    with torch.no_grad():
        base_logits = model(input_ids=ids, use_cache=False).logits[0].float()
    ref = np.fromfile(os.path.join(tiny, "logits_prefill.f32"), dtype=np.float32).reshape(base_logits.shape)
    rel = float(np.linalg.norm(base_logits.numpy() - ref) / np.linalg.norm(ref))
    assert rel < 1e-5, f"reloaded tiny model drifted from the recorded reference: {rel}"

    lcfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS, bias="none")
    model = get_peft_model(model, lcfg)
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator=gen) * (0.5 / p.shape[-1] ** 0.5))
        logits = model(input_ids=ids, use_cache=False).logits[0].float()
    model.save_pretrained(os.path.join(tiny, "lora"))
    logits.numpy().tofile(os.path.join(tiny, "logits_lora.f32"))
    delta = float((logits - base_logits).norm() / base_logits.norm())
    adapted = sorted({n.split(".lora_")[0] for n, _ in model.named_parameters() if "lora_A" in n})
    json.dump({"seed": SEED, "rank": 4, "alpha": 8, "targets": TARGETS,
               "adapted_modules": len(adapted), "lora_vs_base_rel_l2": delta},
              open(os.path.join(tiny, "lora_meta.json"), "w"), indent=2)
    print(f"adapted {len(adapted)} modules; LoRA changes logits by rel L2 {delta:.3f}")


if __name__ == "__main__":
    main()
