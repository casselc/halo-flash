#!/usr/bin/env bash
# Quantizes the --small-quant model (tools/make_tiny_hf_reference.py) with
# the per-tensor types of unsloth's UD-Q4_K_XL Flash-Next, so the GPU's
# quantized kernels run end to end against the CPU reference:
#   Q8_0   dense projections, LM head, embeddings, shared expert, mixers
#   Q4_K   routed gate/up (Q5_K in layer 5, as in one real layer)
#   Q5_1   routed down (Q8_0 in layer 0, as in five real layers)
#   F32    router, shared-expert gate, DeltaNet alpha/beta, mixer inject, convs
#   BF16   QSA indexer projections
#   IQ4_NL PLE n-gram table
#
#   LLAMA_CPP=<llama.cpp checkout with qwen4exp> make_small_quant.sh <dir>
# Reads <dir>/tiny-f32.gguf, writes <dir>/small-q.gguf.
set -euo pipefail
dir=${1:?usage: make_small_quant.sh <dir>}
quantize=$(ls "${LLAMA_CPP:?set LLAMA_CPP}"/build*/bin/llama-quantize | head -1)
args=()
for t in 'ffn_gate_inp=f32' 'ffn_gate_inp_shexp=f32' 'ssm_alpha=f32' 'ssm_beta=f32' \
         'hc_attn_inject=f32' 'hc_ffn_inject=f32' 'ssm_conv1d=f32' 'ple_conv1d=f32' \
         'per_layer_token_embd=iq4_nl' \
         'blk\.5\.ffn_gate_exps=q5_k' 'blk\.5\.ffn_up_exps=q5_k' \
         'ffn_gate_exps=q4_k' 'ffn_up_exps=q4_k' \
         'blk\.0\.ffn_down_exps=q8_0' 'ffn_down_exps=q5_1'; do
  args+=(--tensor-type "$t")
done
"$quantize" --pure "${args[@]}" "$dir/tiny-f32.gguf" "$dir/small-q.tmp.gguf" Q8_0
# llama-quantize never quantizes indexer projections; convert them here.
python3 "$(dirname "$0")/gguf_retype_bf16.py" "$dir/small-q.tmp.gguf" "$dir/small-q.gguf" 'indexer\.[qk]_proj\.weight$'
rm "$dir/small-q.tmp.gguf"
