#!/usr/bin/env python3
"""PyTorch autograd reference for the stage-1 trainer's gradients
(src/train/last_k.cpp), on the tiny Qwen3.5 model of
make_tiny_qwen35_reference.py, in float64.

The residual stream entering layer L-k is replaced (forward pre-hook) by a
leaf tensor, so autograd covers exactly what the trainer covers: layers
L-k .. L-1, the final norm and a head. Two cases:

  lm   LoRA (PEFT, rank 4, alpha 8, random A and B) on every projection of
       the last k layers; mean cross-entropy of the frozen tied LM head at
       a few positions.
  cls  the same LoRA plus a classifier head (LayerNorm + Linear, random
       init) on the last position's final hidden state; cross-entropy of
       one label.

Recorded (float32 files): the boundary input, d(loss)/d(residual entering
each of the last k layers and the final norm), the loss, and every LoRA and
head gradient. LoRA tensors (initial values and gradients) are written in
GGUF layout through tools/convert_lora_gguf.py, which applies the base
converter's own row/column maps (value-head reordering); a gradient maps
like its parameter because the maps are permutations.

Usage: LLAMA_CPP=<llama.cpp> qwen35_grad_reference.py <tiny-dir> <out-dir> [k]
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

SEED = 20261006
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
           "out_proj", "gate_proj", "up_proj", "down_proj"]
RANK, ALPHA, CLASSES = 4, 8, 3


def load_tiny(tiny):
    cfg = json.load(open(os.path.join(tiny, "hf", "config.json")))
    for k in ("architectures", "torch_dtype", "dtype"):
        cfg.pop(k, None)
    config = Qwen3_5TextConfig(**{k: v for k, v in cfg.items() if not k.startswith("_")})
    model = mod.Qwen3_5ForCausalLM(config).eval()
    state = load_file(os.path.join(tiny, "hf", "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if "inv_freq" not in m and not (m == "lm_head.weight" and config.tie_word_embeddings)]
    model.tie_weights()
    assert not missing and not unexpected, (missing, unexpected)
    return model.to(torch.float64)


def run_case(model, ids, boundary, k, case, gen, out, tiny):
    L = model.config.num_hidden_layers
    first = L - k
    layers = [f"model.layers.{i}." for i in range(first, L)]
    targets = [n for n, m in model.named_modules()
               if any(n.startswith(p) for p in layers) and n.split(".")[-1] in TARGETS and isinstance(m, torch.nn.Linear)]
    peft = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, target_modules=targets,
                                            bias="none"))
    with torch.no_grad():
        for n, p in peft.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float64) * (0.5 / p.shape[-1] ** 0.5))
    lora_params = {n: p for n, p in peft.named_parameters() if "lora_" in n}
    for p in lora_params.values():
        p.requires_grad_(True)
    base = peft.base_model.model  # Qwen3_5ForCausalLM

    x0 = boundary.clone().requires_grad_(True)
    captured = {}

    def replace(module, args, kwargs):
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = x0
            return args, kwargs
        return (x0,) + tuple(args[1:]), kwargs

    def keep(name):
        def hook(module, args, kwargs):
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            h.retain_grad()
            captured[name] = h
        return hook

    hooks = [base.model.layers[first].register_forward_pre_hook(replace, with_kwargs=True)]
    for i in range(first, L):
        hooks.append(base.model.layers[i].register_forward_pre_hook(keep(f"in{i - first}"), with_kwargs=True))
    hooks.append(base.model.norm.register_forward_pre_hook(keep(f"in{k}"), with_kwargs=True))
    final = {}
    hooks.append(base.model.norm.register_forward_hook(lambda m, a, o: final.setdefault("h", o)))

    T = ids.shape[1]
    meta = {"case": case, "k": k, "T": T, "rank": RANK, "alpha": ALPHA}
    head = None
    if case == "lm":
        logits = base(input_ids=ids, use_cache=False).logits[0]
        positions = list(range(T - 8, T - 1))
        tgt = ids[0, [p + 1 for p in positions]]
        loss = torch.nn.functional.cross_entropy(logits[positions], tgt)
        meta["positions"] = positions
        meta["targets"] = tgt.tolist()
    else:
        base.model(input_ids=ids, use_cache=False)
        H = model.config.hidden_size
        head = {
            "norm.weight": (1 + 0.3 * torch.randn(H, generator=gen, dtype=torch.float64)).requires_grad_(True),
            "norm.bias": (0.1 * torch.randn(H, generator=gen, dtype=torch.float64)).requires_grad_(True),
            "out.weight": (torch.randn(CLASSES, H, generator=gen, dtype=torch.float64) / H ** 0.5).requires_grad_(True),
            "out.bias": (0.1 * torch.randn(CLASSES, generator=gen, dtype=torch.float64)).requires_grad_(True),
        }
        hfin = final["h"][0, T - 1]
        ln = torch.nn.functional.layer_norm(hfin, (H,), head["norm.weight"], head["norm.bias"], eps=1e-5)
        logits = head["out.weight"] @ ln + head["out.bias"]
        label = 1
        loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([label]))
        meta["label"] = label
        meta["label_position"] = T - 1
        meta["classes"] = CLASSES
    loss.backward()
    for h in hooks:
        h.remove()
    meta["loss"] = float(loss)

    prefix = os.path.join(out, case)
    d_res = torch.stack([captured[f"in{i}"].grad[0] for i in range(k + 1)])
    d_res.float().numpy().tofile(prefix + "_d_residual.f32")
    if head is not None:
        for name, p in head.items():
            p.detach().float().numpy().tofile(f"{prefix}_head_{name}.f32")
            p.grad.float().numpy().tofile(f"{prefix}_head_{name}.grad.f32")

    # LoRA values and gradients through the GGUF converter (PEFT layout -> GGUF layout).
    for what in ("init", "grad"):
        d = os.path.join(out, f"{case}_lora_{what}")
        os.makedirs(d, exist_ok=True)
        tensors = {}
        for n, p in lora_params.items():
            key = n.replace(".default", "")
            tensors[key] = (p.detach() if what == "init" else p.grad).float().contiguous()
        save_file(tensors, os.path.join(d, "adapter_model.safetensors"))
        json.dump({"r": RANK, "lora_alpha": ALPHA, "peft_type": "LORA", "target_modules": TARGETS},
                  open(os.path.join(d, "adapter_config.json"), "w"))
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "convert_lora_gguf.py"),
                        os.path.join(tiny, "hf"), d, f"{prefix}_lora_{what}.gguf"], check=True)
    json.dump(meta, open(prefix + ".json", "w"), indent=2)
    print(f"{case}: loss {meta['loss']:.6f}, {len(lora_params)} LoRA tensors")
    # Leave the base model unwrapped for the next case.
    return peft.unload()


def main():
    tiny, out = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(SEED)
    model = load_tiny(tiny)
    ids = torch.from_numpy(np.fromfile(os.path.join(tiny, "tokens.i32"), dtype=np.int32)).long().view(1, -1)
    L = model.config.num_hidden_layers
    with torch.no_grad():
        o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    # hidden_states[i] is the input of layer i (i < L).
    boundary = o.hidden_states[L - k].float().double()
    boundary[0].float().numpy().tofile(os.path.join(out, "boundary.f32"))
    gen = torch.Generator().manual_seed(SEED)
    for case in ("lm", "cls"):
        model = run_case(model, ids, boundary, k, case, gen, out, tiny)
    import transformers
    json.dump({"seed": SEED, "k": k, "torch": torch.__version__, "transformers": transformers.__version__,
               "dtype": "float64"}, open(os.path.join(out, "meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
