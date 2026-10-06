#!/usr/bin/env python3
"""Runs llama.cpp's convert_hf_to_gguf.py on a Qwen4Exp checkpoint with one
fix: transformers >= 5.19 normalizes attention layer types to
"indexed_attention", while the converter only recognizes "full_attention" and
otherwise writes all-zero QSA compression ratios (silently disabling sparse
attention). This maps the names back before metadata is written.

Usage: LLAMA_CPP=<llama.cpp checkout> convert_qwen4exp.py <convert args...>
       LLAMA_CONVERT=convert_lora_to_gguf.py ... for adapters (same fix).
"""
import os
import sys

llama = os.environ["LLAMA_CPP"]
sys.path[:0] = [llama, os.path.join(llama, "gguf-py")]
sys.argv[0] = os.path.join(llama, os.environ.get("LLAMA_CONVERT", "convert_hf_to_gguf.py"))

from conversion import qwen4exp  # noqa: E402

_original = qwen4exp.Qwen4ExpTextModel.set_gguf_parameters


def _set_gguf_parameters(self):
    types = self.hparams.get("layer_types")
    if types:
        # transformers 5.16 says "qwen_sparse_attention": every non-linear type is the sparse attention layer.
        self.hparams["layer_types"] = ["full_attention" if t != "linear_attention" else t for t in types]
    return _original(self)


qwen4exp.Qwen4ExpTextModel.set_gguf_parameters = _set_gguf_parameters

import runpy  # noqa: E402

runpy.run_path(sys.argv[0], run_name="__main__")
