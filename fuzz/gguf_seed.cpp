// Writes a small seed corpus of valid GGUF images for gguf_fuzz.
#include <cstdio>
#include <string>
#include <vector>

#include "core/gguf_writer.hpp"

using halo::core::GgmlType;
using halo::core::GgufWriter;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: gguf_seed <out-dir>\n");
    return 2;
  }
  const std::string dir = argv[1];
  for (int variant = 0; variant < 6; ++variant) {
    GgufWriter w(variant % 2 ? 64 : 32);
    w.Set("general.architecture", std::string("qwen4exp"));
    w.Set("qwen4exp.block_count", std::uint32_t(variant + 1));
    w.Set("qwen4exp.rope.freq_base", 1.0e7F);
    w.Set("tokenizer.ggml.tokens", std::vector<std::string>{"a", "b", "<|im_end|>"});
    w.Set("qwen4exp.attention.compress_ratios", std::vector<std::int32_t>{0, 0, 0, 4});
    w.Set("split.count", std::uint32_t(1));
    w.Set("flag", variant % 3 == 0);
    std::vector<std::uint8_t> f32(4 * 8 * (variant + 1), 0x3f);
    w.AddTensor("blk.0.attn_norm.weight", {8, std::uint64_t(variant + 1)}, GgmlType::kF32, f32);
    std::vector<std::uint8_t> q8(34 * 2, 0x11);
    w.AddTensor("blk.0.attn_q.weight", {32, 2}, GgmlType::kQ8_0, q8);
    std::vector<std::uint8_t> q4k(144, 0x5a);
    w.AddTensor("blk.0.ffn_up_exps.weight", {256, 1, 1}, GgmlType::kQ4_K, q4k);
    std::string error;
    if (!w.WriteFile(dir + "/seed" + std::to_string(variant) + ".gguf", &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
  }
  return 0;
}
