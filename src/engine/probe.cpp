// PROBE ONLY: times batch-shaped WMMA kernels (src/kernels/batch_decode.hip, "PROBE ONLY") inside the engine, on the
// real weights, so they run at the clocks and cache state of a decode. Nothing here feeds a decode result.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "engine/flash_next_gpu.hpp"

namespace halo::engine {

namespace {
constexpr std::uint32_t kTab = 1 + 80 + 80 + 640;  // ucount, uexp[80], unum[80], upair[80 * 8]
}

bool FlashNextGpu::ProbeSetRouting(std::uint32_t B, const std::vector<std::vector<std::uint32_t>>& ids,
                                   std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t k = c.num_experts_used, L = c.num_layers;
  if (!probe_x_) {
    const std::size_t xs = 8ull * 12288;
    probe_x_ = Scratch(xs, error);
    probe_part_ = Scratch(std::max<std::size_t>(16ull * 8 * 12288, 8ull * 3 * c.vocab_size), error);
    probe_y_ = Scratch(8ull * c.vocab_size, error);
    probe_h_ = Scratch(80ull * c.expert_ff, error);
    probe_ep_ = Scratch(80ull * c.hidden_size, error);
    probe_tab_stride_ = kTab;
    probe_tab_ = Scratch(std::size_t{L} * kTab, error);
    if (!probe_x_ || !probe_part_ || !probe_y_ || !probe_h_ || !probe_ep_ || !probe_tab_) return false;
    std::vector<float> x(xs);
    for (std::size_t i = 0; i < xs; ++i) x[i] = 0.05f * std::sin(0.37f * static_cast<float>(i));
    if (!dev_.Upload(probe_x_, 0, x.data(), xs * 4, error) || !dev_.Fill(probe_h_, 0, error)) return false;
  }
  std::vector<std::uint32_t> tab(std::size_t{L} * kTab, 0);
  for (std::uint32_t l = 0; l < L && l < ids.size(); ++l) {
    std::uint32_t* t = tab.data() + std::size_t{l} * kTab;
    const auto& v = ids[l];
    std::uint32_t U = 0;
    for (std::uint32_t p = 0; p < B * k && p < v.size(); ++p) {
      std::uint32_t u = 0;
      while (u < U && t[1 + u] != v[p]) ++u;
      if (u == U) t[1 + U++] = v[p];
      t[1 + 80 + 80 + u * 8 + t[1 + 80 + u]++] = p;
    }
    t[0] = U;
  }
  probe_B_ = B;
  return dev_.Upload(probe_tab_, 0, tab.data(), tab.size() * 4, error);
}

bool FlashNextGpu::ProbeRun(std::uint32_t B, bool check, std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t H = c.hidden_size, k = c.num_experts_used;
  const auto gemv = [&](const model::TensorRef& t, const char* tag) -> bool {
    const auto& d = tensors_.at(t.data);
    if (!d.qs) return true;  // not a repacked Q8 matrix: not probed
    const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
    // Split K so that the launch has >= ~240 blocks (and every split fits 1024 columns of LDS).
    const std::uint32_t row_blocks = (rows + 127) / 128;
    std::uint32_t S = std::max<std::uint32_t>((cols + 1023) / 1024, std::min<std::uint32_t>(16, (240 + row_blocks - 1) / row_blocks));
    std::uint32_t kspan = ((cols + S - 1) / S + 31) / 32 * 32;
    S = (cols + kspan - 1) / kspan;
    (void)tag;
    static const char* kname = [] {
      const char* v = std::getenv("HALO_PROBE_V");
      return v && v[0] == '1'   ? "halo_probe_q8r_wmma"
             : v && v[0] == '2' ? "halo_probe_q8r_wmma2"
             : v && v[0] == '4' ? "halo_probe_q8r_wmma4"
                                : "halo_probe_q8r_wmma3";
    }();
    if (!Launch(kname, {row_blocks, S}, {256},
                {rt::Arg::Buf(d.qs), rt::Arg::Buf(d.d), rt::Arg::Buf(probe_x_), rt::Arg::Buf(probe_part_),
                 rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val(d.qstride), rt::Arg::Val(B), rt::Arg::Val(kspan),
                 rt::Arg::Val<std::uint32_t>(12288)},
                error))
      return false;
    return Launch("halo_probe_reduce", {(rows * B + 255) / 256}, {256},
                  {rt::Arg::Buf(probe_part_), rt::Arg::Buf(probe_y_), rt::Arg::Val(rows), rt::Arg::Val(B), rt::Arg::Val(S)},
                  error);
  };
  if (check) {
    const auto& l = w_.layers[0];
    const auto& t = l.linear ? l.ssm_qkv : l.attn_q;
    const auto rows = static_cast<std::uint32_t>(t.rows);
    std::vector<float> a(rows), b(rows);
    if (!gemv(t, "check") || !dev_.Download(probe_y_, 0, a.data(), rows * 4, error) ||
        !MatVec(t, 0, probe_x_, lora_out_, error) || !dev_.Download(lora_out_, 0, b.data(), rows * 4, error))
      return false;
    double dd = 0, mm = 0;
    for (std::uint32_t i = 0; i < rows; ++i) dd += double(a[i] - b[i]) * (a[i] - b[i]), mm += double(b[i]) * b[i];
    std::printf("probe check: WMMA fp16 GEMV vs decode GEMV, %s column 0: relative L2 %.3g\n", t.name.data(),
                std::sqrt(dd / mm));
  }
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const auto& l = w_.layers[il];
    if (!gemv(l.hc_attn.down, "mix down") || !gemv(l.hc_attn.up, "mix up")) return false;
    if (l.linear) {
      if (!gemv(l.ssm_qkv, "qkv") || !gemv(l.ssm_gate, "z") || !gemv(l.ssm_out, "ssm out")) return false;
    } else if (!gemv(l.attn_q, "q") || !gemv(l.attn_k, "k") || !gemv(l.attn_v, "v") || !gemv(l.attn_out, "attn out")) {
      return false;
    }
    if (!gemv(l.hc_ffn.down, "mix down") || !gemv(l.hc_ffn.up, "mix up")) return false;
    const auto& g = l.ffn_gate_exps;
    const auto& dn = l.ffn_down_exps;
    if (g.type == core::GgmlType::kQ4_K && dn.type == core::GgmlType::kQ5_1) {
      const auto ff = static_cast<std::uint32_t>(g.rows);
      const rt::Buffer tab{static_cast<char*>(probe_tab_.device) + std::size_t{il} * kTab * 4, kTab * 4, nullptr};
      const auto at = [&](std::size_t off) { return rt::Arg::Buf(tab, off * 4); };
      if (!Launch("halo_probe_q4k_gu_wmma", {(ff + 127) / 128, B * k}, {256},
                  {rt::Arg::Buf(tensors_.at(g.data).buf), rt::Arg::Buf(tensors_.at(l.ffn_up_exps.data).buf), at(0), at(1),
                   at(81), at(161), rt::Arg::Buf(probe_x_), rt::Arg::Buf(probe_h_),
                   rt::Arg::Val<std::uint64_t>(g.RowBytes() * g.rows), rt::Arg::Val(ff), rt::Arg::Val(H),
                   rt::Arg::Val<std::uint64_t>(g.RowBytes()), rt::Arg::Val(k)},
                  error) ||
          !Launch("halo_probe_q51_down_wmma", {(H + 127) / 128, B * k}, {256},
                  {rt::Arg::Buf(tensors_.at(dn.data).buf), at(0), at(1), at(81), at(161), rt::Arg::Buf(probe_h_),
                   rt::Arg::Buf(probe_ep_), rt::Arg::Val<std::uint64_t>(dn.RowBytes() * dn.rows), rt::Arg::Val(H),
                   rt::Arg::Val(ff), rt::Arg::Val<std::uint64_t>(dn.RowBytes())},
                  error))
        return false;
    }
    if (!gemv(l.shexp_gate, "shexp gate") || !gemv(l.shexp_up, "shexp up") || !gemv(l.shexp_down, "shexp down"))
      return false;
  }
  return gemv(w_.hc_head.down, "head mix down") && gemv(w_.hc_head.up, "head mix up") && gemv(w_.output, "lm head");
}

}  // namespace halo::engine
