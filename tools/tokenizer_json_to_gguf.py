#!/usr/bin/env python3
"""Writes a tokenizer-only GGUF from an HF byte-level BPE tokenizer.json, in
llama.cpp's tokenizer.ggml.* layout, so halo-flash loads it like any model's
tokenizer. The split regex picks tokenizer.ggml.pre: "qwen35" when letter runs
include \\p{M}, "qwen2" for the older pattern (strands-decider ships that one).

Usage: LLAMA_CPP=<llama.cpp> tokenizer_json_to_gguf.py <tokenizer.json> <out.gguf>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.environ["LLAMA_CPP"], "gguf-py"))
import gguf  # noqa: E402

QWEN35 = r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
QWEN2 = r"[^\r\n\p{L}\p{N}]?\p{L}+"


def main():
    src, out = sys.argv[1], sys.argv[2]
    spec = json.load(open(src, encoding="utf-8"))
    model = spec["model"]
    assert model["type"] == "BPE" and spec["normalizer"]["type"] == "NFC"
    pattern = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    pre = "qwen35" if QWEN35 in pattern else "qwen2" if QWEN2 in pattern else None
    if pre is None:
        raise SystemExit(f"unrecognized split regex: {pattern}")
    vocab = dict(model["vocab"])
    added = {t["id"]: t for t in spec["added_tokens"]}
    n = max(max(vocab.values()), max(added) if added else 0) + 1
    tokens = [f"[PAD{i}]" for i in range(n)]
    types = [gguf.TokenType.UNUSED] * n
    for text, i in vocab.items():
        tokens[i], types[i] = text, gguf.TokenType.NORMAL
    for i, t in added.items():
        tokens[i] = t["content"]
        types[i] = gguf.TokenType.CONTROL if t["special"] else gguf.TokenType.USER_DEFINED
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    w = gguf.GGUFWriter(out, "qwen35")
    w.add_type("tokenizer")
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre(pre)
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    eos = next((i for i, t in added.items() if t["content"] == "<|endoftext|>"), None)
    if eos is not None:
        w.add_eos_token_id(eos)
        w.add_pad_token_id(eos)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{out}: {n} tokens, {len(merges)} merges, pre={pre}")


if __name__ == "__main__":
    main()
