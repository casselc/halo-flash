#!/usr/bin/env python3
"""Builds a tiny random Qwen3.8-Flash-Next with HF transformers' own modeling
code (modeling_qwen4_exp.py) and records its outputs as the reference for
halo-flash's CPU oracle.

The checkpoint mimics the real one's tensor names and layout
(model.language_model.*, stacked experts, the PLE table split into
`ngram_embedding.shard_N` pieces), so llama.cpp's converter runs the same code
path it runs on the real model:

  make_tiny_hf_reference.py <tokenizer-dir> <out-dir> [--small-quant]
  python <llama.cpp>/convert_hf_to_gguf.py <out-dir>/hf --outtype f32 \
      --outfile <out-dir>/tiny-f32.gguf

--small-quant builds a larger "small" model whose widths the real quant mix
accepts (hidden 256, DeltaNet and indexer head dim 128, attention head dim
256, mixer rank 32), so the GPU's quantized fast paths run end to end;
tools/make_small_quant.sh then quantizes it like UD-Q4_K_XL.

<tokenizer-dir> holds Flash-Next's tokenizer.json and tokenizer_config.json.
Outputs in <out-dir>:
  hf/                   config.json, model.safetensors, tokenizer files
  tokens.i32            the reference sequence (int32)
  logits_prefill.f32    [T, vocab] logits from one full forward
  logits_decode.f32     [T, vocab] logits from token-by-token cached decode
  hidden.f32            [n, T, hc_dim] residual streams recorded by HF
  head_input.f32        [T, hidden] head-mixer output (LM head input)
  meta.json             shapes, versions, and the prefill/decode agreement
"""
import json
import math
import os
import shutil
import sys

import torch
import transformers
from safetensors.torch import save_file
from transformers.models.qwen4_exp import configuration_qwen4_exp as cfgmod
from transformers.models.qwen4_exp import modeling_qwen4_exp as mod

SEED = 20261003
T = 48           # sequence length: 12 complete QSA blocks against a budget of 2
EOS = 248044     # <|endoftext|>, the PLE n-gram reset token
SPLIT = 4        # PLE table shards


def tiny_config(**overrides):
    layers = 8
    kw = dict(
        vocab_size=248320, hidden_size=64, num_hidden_layers=layers,
        layer_types=["linear_attention"] * 3 + ["full_attention"]
                    + ["linear_attention"] * 3 + ["full_attention"],
        full_attention_interval=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
        indexer_budget=8, indexer_compress_ratio=4,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, norm_topk_prob=True,
        hc_count=4, hc_lowrank=16,
        ple_layer_ids=[2], ple_embed_dim=128, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=101,
        make_ngram_vocab_size_divisible_by=128, split_ngram_parts=SPLIT,
        output_gate_type="sigmoid", hidden_act="silu",
        rms_norm_eps=1e-6, max_position_embeddings=4096,
        eos_token_id=EOS, bos_token_id=EOS, pad_token_id=None,
        tie_word_embeddings=False, attention_bias=False,
    )
    kw.update(overrides)
    return cfgmod.Qwen4ExpTextConfig(**kw)


def small_quant_config():
    """Same layer pattern; every width a multiple of the quant block it meets."""
    return tiny_config(
        hidden_size=256, head_dim=256,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25,
                         "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        indexer_head_dim=128, linear_key_head_dim=128, linear_value_head_dim=128,
        num_experts=16, num_experts_per_tok=4, moe_intermediate_size=128,
        shared_expert_intermediate_size=128, hc_lowrank=32, ple_embed_dim=512)


def randomize(model, gen):
    """Weights scaled to keep activations O(1) in tiny dimensions, and every
    norm gamma moved off zero so a missing or doubled (1 + w) is visible."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.copy_(torch.empty_like(p).uniform_(-0.3, 0.3, generator=gen))
            elif name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(1.0, 16.0, generator=gen)))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif "conv1d" in name:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif "ngram_embedding" in name or "embed_tokens" in name:
                p.copy_(torch.randn(p.shape, generator=gen))
            elif p.ndim >= 2:
                fan_in = p.shape[-1]
                p.copy_(torch.randn(p.shape, generator=gen) / math.sqrt(fan_in))
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.1)


def checkpoint_state(model, config):
    """HF state dict renamed to the real checkpoint's layout."""
    out = {}
    for k, v in model.state_dict().items():
        if k.endswith("rotary_emb.inv_freq") or k.endswith("original_inv_freq"):
            continue
        name = k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k
        if name.endswith("ngram_embedding.weight"):
            rows = v.shape[0]
            per = math.ceil(rows / config.split_ngram_parts)
            for i in range(config.split_ngram_parts):
                out[name.replace("ngram_embedding.weight", f"ngram_embedding.shard_{i}.weight")] = \
                    v[i * per:(i + 1) * per].contiguous()
            continue
        out[name] = v.contiguous()
    return out


def main():
    tok_dir, out_dir = sys.argv[1], sys.argv[2]
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    config = small_quant_config() if "--small-quant" in sys.argv[3:] else tiny_config()
    model = mod.Qwen4ExpForCausalLM(config).eval().to(torch.float32)
    randomize(model, gen)

    hf_dir = os.path.join(out_dir, "hf")
    os.makedirs(hf_dir, exist_ok=True)
    cfg = config.to_dict()
    cfg["architectures"] = ["Qwen4ExpForCausalLM"]
    cfg["torch_dtype"] = "float32"
    # transformers main serializes attention layers as "indexed_attention";
    # the real Flash-Next config.json says "full_attention", which is the only
    # value llama.cpp's converter recognizes (anything else silently writes
    # zero QSA compression ratios). Match the real checkpoint.
    # (transformers 5.16 writes "qwen_sparse_attention": every non-linear type maps.)
    cfg["layer_types"] = ["full_attention" if t != "linear_attention" else t
                          for t in cfg["layer_types"]]
    json.dump(cfg, open(os.path.join(hf_dir, "config.json"), "w"), indent=2)
    save_file(checkpoint_state(model, config), os.path.join(hf_dir, "model.safetensors"),
              metadata={"format": "pt"})
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(tok_dir, f), hf_dir)

    # Reference sequence: random ordinary tokens with an EOS in the middle
    # (resets the PLE n-gram window) and a repeated bigram (exercises the
    # same rows twice).
    g = torch.Generator().manual_seed(SEED + 1)
    tokens = torch.randint(0, 248000, (T,), generator=g)
    tokens[20] = EOS
    tokens[30], tokens[31] = tokens[5], tokens[6]
    ids = tokens.view(1, T)

    with torch.no_grad():
        full = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        prefill = full.logits[0].float()
        hc_dim = config.hc_count * config.hidden_size
        hidden = torch.stack([h[0] for h in full.hidden_states if h.shape[-1] == hc_dim]).float()
        head_in = [h[0] for h in full.hidden_states if h.shape[-1] != hc_dim][-1].float()
        cache = None
        decode = []
        for t in range(T):
            o = model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            decode.append(o.logits[0, -1].float())
        decode = torch.stack(decode)

    tokens.to(torch.int32).numpy().tofile(os.path.join(out_dir, "tokens.i32"))
    prefill.numpy().tofile(os.path.join(out_dir, "logits_prefill.f32"))
    decode.numpy().tofile(os.path.join(out_dir, "logits_decode.f32"))
    hidden.numpy().tofile(os.path.join(out_dir, "hidden.f32"))
    head_in.numpy().tofile(os.path.join(out_dir, "head_input.f32"))
    rel = ((prefill - decode).norm() / prefill.norm()).item()
    meta = {
        "seed": SEED, "T": T, "vocab": config.vocab_size,
        "hidden_shape": list(hidden.shape),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "prefill_vs_decode_rel_l2": rel,
        "prefill_vs_decode_argmax_agree": int((prefill.argmax(-1) == decode.argmax(-1)).sum()),
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
