// halo_run: greedy decoding on the real model, on the CPU reference, the
// GPU engine, or both in lockstep (reporting their logit difference per
// step). Used for the P1/P2 gates against gufo's greedy output.
//
//   halo_run --model <shard-00001.gguf> --prompt "text" [--n 16]
//            [--device cpu|hip|hrx|both-hip|both-hrx] [--max-context 512]
//   halo_run --model <shard-00001.gguf> --prefill-bench 2048 --max-context 2048
//            [--batched-prefill 0|1] [--prefill-int8 0|1] [--reps 3]
//            [--prefill-file text] [--prefill-warm text]
//
// --prefill-bench N times Prefill over N pseudo-random tokens (fixed seed,
// as llama-bench's pp test uses random tokens) from a reset engine, --reps
// times, and reports the median. --prefill-file takes the tokens from a
// text file instead (the first N of its raw encoding), and --prefill-warm
// prefills another file once, untimed, first: gufo's model_bench protocol
// (synthetic prose over a fixed vocabulary, a warm-up request first).
//
// The prompt is encoded raw (no chat template), as `gufo prompt --raw`.
// Output: one line per generated token (id, top-2 logit margin, piece),
// then the decoded text and timing.

#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include <iterator>
#include <map>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "halo_toolchain.h"
#include "core/gguf.hpp"
#include "core/sampling.hpp"
#include "engine/flash_next_gpu.hpp"
#include "model/ngram.hpp"
#include "model/reference.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"
#include "text/tokenizer.hpp"

using namespace halo;

namespace {

struct Top2 {
  std::uint32_t best;
  float margin;
};

Top2 Argmax(const std::vector<float>& l) {
  std::uint32_t b = 0, s = 1;
  if (l[s] > l[b]) std::swap(b, s);
  for (std::uint32_t i = 2; i < l.size(); ++i) {
    if (l[i] > l[b]) {
      s = b;
      b = i;
    } else if (l[i] > l[s]) {
      s = i;
    }
  }
  return {b, l[b] - l[s]};
}

double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0, m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    d += double(a[i] - b[i]) * (a[i] - b[i]);
    m += double(b[i]) * b[i];
  }
  return std::sqrt(d / m);
}

// --profile: every launch is followed by a synchronize and timed, then
// accumulated per kernel name (eager mode). The per-launch floor (launch +
// sync of an empty grid) is reported so it can be subtracted.
struct Profiling final : rt::Device {
  std::unique_ptr<rt::Device> d;
  std::map<rt::KernelId, std::string> names;
  std::map<std::string, std::pair<double, long>> time;  // seconds, launches
  bool on = false;
  rt::Backend backend() const noexcept override { return d->backend(); }
  rt::Buffer Alloc(std::size_t b, std::string* e) override { return d->Alloc(b, e); }
  void Free(rt::Buffer& b) noexcept override { d->Free(b); }
  bool Upload(const rt::Buffer& a, std::size_t o, const void* s, std::size_t b, std::string* e) override {
    return d->Upload(a, o, s, b, e);
  }
  bool Download(const rt::Buffer& a, std::size_t o, void* s, std::size_t b, std::string* e) override {
    return d->Download(a, o, s, b, e);
  }
  bool Fill(const rt::Buffer& a, std::uint8_t v, std::string* e) override { return Timed("(fill)", [&] { return d->Fill(a, v, e); }); }
  void* AllocHost(std::size_t b, std::string* e) override { return d->AllocHost(b, e); }
  void FreeHost(void* p) noexcept override { d->FreeHost(p); }
  bool LoadModule(std::span<const std::uint8_t> h, std::string* e) override { return d->LoadModule(h, e); }
  rt::KernelId Kernel(std::string_view n, std::string* e) override {
    const auto id = d->Kernel(n, e);
    names[id] = std::string(n);
    return id;
  }
  bool Launch(rt::KernelId k, rt::Dim3 g, rt::Dim3 b, std::span<const rt::Arg> a, std::string* e) override {
    return Timed(names[k], [&] { return d->Launch(k, g, b, a, e); });
  }
  bool Synchronize(std::string* e) override { return d->Synchronize(e); }
  bool BeginCapture(std::string* e) override { return d->BeginCapture(e); }
  bool EndCapture(rt::Graph* g, std::string* e) override { return d->EndCapture(g, e); }
  bool LaunchGraph(const rt::Graph& g, std::string* e) override { return d->LaunchGraph(g, e); }
  void FreeGraph(rt::Graph& g) noexcept override { d->FreeGraph(g); }
  template <typename F>
  bool Timed(const std::string& name, F&& f) {
    if (!on) return f();
    std::string e;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = f() && d->Synchronize(&e);
    auto& [secs, n] = time[name];
    secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++n;
    return ok;
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string model, prompt = "The capital of France is", device = "hip";
  int n = 16, prefill_bench = 0, reps = 3;
  std::string prefill_file, prefill_warm;
  bool kernel_timing = false;  // --kernel-timing 1: sync-free per-kernel times of the last prefill rep
  // Decode measurement: --prompt-file F (raw text prompt), --prefill-prompt 1 (all but the prompt's last token
  // through Prefill, then stepping), --decode-logits-out F (every decode step's logits, fp32, appended),
  // --decode-timing 1 (sync-free per-kernel times of the steady decode steps; eager launches).
  std::string prompt_file, decode_logits_out;
  bool prefill_prompt = false, decode_timing = false;
  // --span-hsaco FILE --span-kernel NAME --span-occurrence N --span-out FILE [--span-sample N]: load the traced
  // code object and record the N-th launch (from 0) of kernel NAME in the last prefill rep (docs/TRACING.md).
  std::string span_hsaco, span_out;
  std::string logits_out;  // --logits-out FILE: the last prefill's logits (fp32), for bitwise comparison of builds
  std::string route_trace;
  // --batch B: P6 batched decode of B sessions (prompts from --batch-prompts, comma-separated files, cycled with
  // different lengths), --n steps; --device-sample 1 times greedy steps, else --batch-check 1 compares every
  // entry's logits with the same session replayed alone.
  std::uint32_t batch = 0;
  std::string batch_prompts;
  bool batch_check = false;
  // --session-bench N: prefill N tokens (--prompt-file repeated), then time and size fork / bind / snapshot /
  // restore / drop of that session (P6 design numbers).
  std::uint32_t session_bench = 0;
  // --mtp-model FILE --mtp-prompts a,b,c: P5 acceptance measurement. Per prompt: pass 1 decodes --n tokens AR
  // (greedy, or sampled at --mtp-temp T) and keeps every target distribution; pass 2 replays the sequence
  // token by token and at every position drafts --mtp-depth tokens with the MTP block (greedy: the block's own
  // argmax chain, exact acceptance; sampled: the chain teacher-forced on the target path, acceptance
  // sum_x min(p, q) per depth). --mtp-check N compares the first N draft logits with the CPU reference.
  std::string mtp_model, mtp_prompts, mtp_out, mtp_vocab;
  // --mtp-dump PREFIX: pass 2 writes PREFIX.<prompt>.{tok,hid,dft}: the token sequence (int32), the prompt length
  // and every position's trunk hidden (hc_dim floats), and the GPU's greedy drafts per draft position (K int32).
  // --mtp-cpu-replay PREFIX.<prompt> --mtp-cpu-n N (with --device cpu): the CPU reference draft block (fp32
  // arithmetic on the dequantized sidecar weights) on those hiddens, greedy chains at the first N draft positions,
  // acceptance against the same tokens, and agreement with the GPU's drafts.
  std::string mtp_dump, mtp_cpu_replay;
  std::uint32_t mtp_cpu_n = 40;  // --mtp-vocab FILE: int32 ids of the trimmed draft head
  // --wmma-probe a.bin,b.bin (route traces): PROBE ONLY. Per B = 1..8, six decode steps (eager, kernel timing on)
  // each followed by the batch-shaped WMMA probe kernels for B columns; routing of the B columns drawn from the
  // traces (different traces / positions, or --wmma-probe-consec 1: B consecutive steps of one trace).
  std::string wmma_probe;
  // --count-files LIST --count-top N --mtp-vocab OUT: the N most frequent tokens over the files named in LIST
  // (one path a line), filled up by lowest id (BPE merge order), plus the chat-control tokens; int32 ids to OUT.
  std::string count_files;
  std::uint32_t count_top = 32768;
  bool wmma_consec = false;
  std::uint32_t mtp_depth = 4, mtp_check = 0;
  double mtp_temp = 0;  // --route-trace FILE: routed expert ids of every step ([layers][k] uint32)
  std::uint32_t span_sample = 1;
  bool eager = false, profile = false, device_sample = false;
  bool ngram_resident = false;  // --ngram-resident 1: the n-gram table in RAM (~29 GB)
  std::uint32_t moe_topk = 0;  // --moe-topk N: experts per token (0: the file's count; CODESIGN arm B)
  double ngram_cache_gb = 4.0;  // --ngram-cache-gb N: cache of n-gram rows read from disk (0: off)
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::uint32_t max_context = 512;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--model") model = argv[i + 1];
    else if (a == "--prompt") prompt = argv[i + 1];
    else if (a == "--device") device = argv[i + 1];
    else if (a == "--n") n = std::atoi(argv[i + 1]);
    else if (a == "--max-context") max_context = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--eager") eager = std::string(argv[i + 1]) == "1";
    else if (a == "--profile") profile = std::string(argv[i + 1]) == "1";
    else if (a == "--device-sample") device_sample = std::string(argv[i + 1]) == "1";
    else if (a == "--int8-moe") engine::FlashNextGpu::int8_act_moe_ = std::string(argv[i + 1]) == "1";
    else if (a == "--prefill-bench") prefill_bench = std::atoi(argv[i + 1]);
    else if (a == "--span-hsaco") span_hsaco = argv[i + 1];
    else if (a == "--logits-out") logits_out = argv[i + 1];
    else if (a == "--moe-blocked") engine::FlashNextGpu::prefill_moe_blocked_ = std::string(argv[i + 1]) == "1";
    else if (a == "--span-kernel") engine::FlashNextGpu::span_kernel_ = argv[i + 1];
    else if (a == "--span-occurrence") engine::FlashNextGpu::span_occurrence_ = std::atoll(argv[i + 1]);
    else if (a == "--span-out") span_out = argv[i + 1];
    else if (a == "--span-sample") span_sample = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--prefill-file") prefill_file = argv[i + 1];
    else if (a == "--split-gdn") engine::FlashNextGpu::prefill_split_gdn_ = std::string(argv[i + 1]) == "1";
    else if (a == "--mq-attention") engine::FlashNextGpu::prefill_mq_attention_ = std::string(argv[i + 1]) == "1";
    else if (a == "--parallel-bucket") engine::FlashNextGpu::prefill_parallel_bucket_ = std::string(argv[i + 1]) == "1";
    else if (a == "--skinny-f32") engine::FlashNextGpu::prefill_skinny_f32_ = std::string(argv[i + 1]) == "1";
    else if (a == "--fused-combine") engine::FlashNextGpu::prefill_fused_combine_ = std::string(argv[i + 1]) == "1";
    else if (a == "--fused-inject") engine::FlashNextGpu::prefill_fused_inject_ = std::string(argv[i + 1]) == "1";
    else if (a == "--ple-overlap") engine::FlashNextGpu::prefill_ple_overlap_ = std::string(argv[i + 1]) == "1";
    else if (a == "--fa-attention") engine::FlashNextGpu::prefill_fa_attention_ = std::string(argv[i + 1]) == "1";
    else if (a == "--fused-moe-combine") engine::FlashNextGpu::prefill_fused_moe_combine_ = std::string(argv[i + 1]) == "1";
    else if (a == "--tok-combine") engine::FlashNextGpu::prefill_tok_combine_ = std::string(argv[i + 1]) == "1";
    else if (a == "--moe-in-combine") engine::FlashNextGpu::prefill_moe_in_combine_ = std::string(argv[i + 1]) == "1";
    else if (a == "--q-fused") engine::FlashNextGpu::prefill_q_fused_ = std::string(argv[i + 1]) == "1";
    else if (a == "--s640") engine::FlashNextGpu::prefill_s640_ = std::string(argv[i + 1]) == "1";
    else if (a == "--f16-out") engine::FlashNextGpu::prefill_f16_out_ = std::string(argv[i + 1]) == "1";
    else if (a == "--moe-vec") engine::FlashNextGpu::prefill_moe_vec_ = std::string(argv[i + 1]) == "1";
    else if (a == "--kq-fused") engine::FlashNextGpu::prefill_kq_fused_ = std::string(argv[i + 1]) == "1";
    else if (a == "--gated-wave") engine::FlashNextGpu::prefill_gated_wave_ = std::string(argv[i + 1]) == "1";
    else if (a == "--gate-in-place") engine::FlashNextGpu::prefill_gate_in_place_ = std::string(argv[i + 1]) == "1";
    else if (a == "--banded-gemm") engine::FlashNextGpu::prefill_banded_gemm_ = std::string(argv[i + 1]) == "1";
    else if (a == "--reg-gemm") engine::FlashNextGpu::prefill_reg_gemm_ = std::string(argv[i + 1]) == "1";
    else if (a == "--blocked-xh") engine::FlashNextGpu::prefill_blocked_xh_ = std::string(argv[i + 1]) == "1";
    else if (a == "--gdn-chunked") {
      engine::FlashNextGpu::prefill_gdn_chunked_ = std::string(argv[i + 1]) != "0";
      engine::FlashNextGpu::prefill_gdn_c16_u3_ = std::string(argv[i + 1]) != "u2";
    }
    else if (a == "--ngram-prefetch") engine::FlashNextGpu::prefill_ngram_prefetch_ = std::string(argv[i + 1]) == "1";
    else if (a == "--kv16") engine::FlashNextGpu::prefill_kv16_ = std::string(argv[i + 1]) == "1";
    else if (a == "--scores-wmma") engine::FlashNextGpu::prefill_scores_wmma_ = std::string(argv[i + 1]) == "1";
    else if (a == "--qsel") engine::FlashNextGpu::prefill_qsel_attention_ = std::string(argv[i + 1]) == "1";
    else if (a == "--qsa-select256") engine::FlashNextGpu::prefill_select256_ = std::string(argv[i + 1]) != "0";
    else if (a == "--qsel-h2") engine::FlashNextGpu::prefill_qsel_h2_ = std::string(argv[i + 1]) != "0";
    else if (a == "--gemm-slice") engine::FlashNextGpu::prefill_gemm_slice_ = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--dense-prefix") engine::FlashNextGpu::prefill_dense_prefix_ = std::string(argv[i + 1]) != "0";
    else if (a == "--prefill-chunk") engine::FlashNextGpu::prefill_chunk_ = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--prefill-long-chunk") engine::FlashNextGpu::prefill_long_chunk_ = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--router-bf16x3") engine::FlashNextGpu::prefill_router_bf16x3_ = std::string(argv[i + 1]) == "1";
    else if (a == "--prompt-file") prompt_file = argv[i + 1];
    else if (a == "--prefill-prompt") prefill_prompt = std::string(argv[i + 1]) == "1";
    else if (a == "--decode-logits-out") decode_logits_out = argv[i + 1];
    else if (a == "--decode-attn-b") engine::FlashNextGpu::decode_attn_batched_ = std::string(argv[i + 1]) == "1";
    else if (a == "--decode-opt") engine::FlashNextGpu::decode_opt_ = static_cast<unsigned>(std::strtoul(argv[i + 1], nullptr, 0));
    else if (a == "--route-trace") route_trace = argv[i + 1], engine::FlashNextGpu::route_trace_ = true;
    else if (a == "--batch") batch = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--batch-prompts") batch_prompts = argv[i + 1];
    else if (a == "--session-bench") session_bench = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--mtp-model") mtp_model = argv[i + 1];
    else if (a == "--mtp-prompts") mtp_prompts = argv[i + 1];
    else if (a == "--mtp-depth") mtp_depth = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--mtp-temp") mtp_temp = std::atof(argv[i + 1]);
    else if (a == "--mtp-check") mtp_check = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--mtp-out") mtp_out = argv[i + 1];
    else if (a == "--mtp-vocab") mtp_vocab = argv[i + 1];
    else if (a == "--mtp-dump") mtp_dump = argv[i + 1];
    else if (a == "--mtp-cpu-replay") mtp_cpu_replay = argv[i + 1];
    else if (a == "--mtp-cpu-n") mtp_cpu_n = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--count-files") count_files = argv[i + 1];
    else if (a == "--count-top") count_top = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--wmma-probe") wmma_probe = argv[i + 1];
    else if (a == "--wmma-probe-consec") wmma_consec = std::string(argv[i + 1]) == "1";
    else if (a == "--batch-check") batch_check = std::string(argv[i + 1]) == "1";
    else if (a == "--batch-decode") engine::FlashNextGpu::batch_decode_ = std::string(argv[i + 1]) == "1";
    else if (a == "--decode-timing") decode_timing = std::string(argv[i + 1]) == "1";
    else if (a == "--kernel-timing") kernel_timing = std::string(argv[i + 1]) == "1";
    else if (a == "--ngram-resident") ngram_resident = std::string(argv[i + 1]) == "1";
    else if (a == "--ngram-cache-gb") ngram_cache_gb = std::atof(argv[i + 1]);
    else if (a == "--moe-topk") moe_topk = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--prefill-warm") prefill_warm = argv[i + 1];
    else if (a == "--reps") reps = std::max(1, std::atoi(argv[i + 1]));
    else if (a == "--batched-prefill") engine::FlashNextGpu::batched_prefill_ = std::string(argv[i + 1]) == "1";
    else if (a == "--small-f16") engine::FlashNextGpu::prefill_small_f16_ = std::string(argv[i + 1]) == "1";
    else if (a == "--mix-h16") engine::FlashNextGpu::prefill_mix_h16_ = std::string(argv[i + 1]) == "1";
    else if (a == "--shexp-w8a8") engine::FlashNextGpu::prefill_shexp_w8a8_ = std::string(argv[i + 1]) == "1";
    else if (a == "--down-w8a8") engine::FlashNextGpu::prefill_down_w8a8_ = std::string(argv[i + 1]) == "1";
    else if (a == "--gdn-io16") engine::FlashNextGpu::prefill_gdn_io16_ = std::string(argv[i + 1]) == "1";
    else if (a == "--gdn-f16") engine::FlashNextGpu::prefill_gdn_f16_ = std::atoi(argv[i + 1]);
    else if (a == "--mixed32-skip") engine::FlashNextGpu::prefill_mixed32_skip_ = std::string(argv[i + 1]) == "1";
    else if (a == "--router-tile") engine::FlashNextGpu::prefill_router_tile_ = std::atoi(argv[i + 1]);
    else if (a == "--res16") engine::FlashNextGpu::prefill_res16_ = std::atoi(argv[i + 1]);
    else if (a == "--gdn-scan-h") engine::FlashNextGpu::prefill_gdn_scan_h_ = std::string(argv[i + 1]) == "1";
    else if (a == "--router-f16") engine::FlashNextGpu::prefill_router_f16_ = std::atoi(argv[i + 1]);
    else if (a == "--prefill-int8") engine::FlashNextGpu::prefill_int8_gemm_ = std::string(argv[i + 1]) == "1";
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (model.empty()) {
    std::fprintf(stderr, "usage: halo_run --model <shard-00001.gguf> [--prompt text] [--n N] [--device ...]\n");
    return 2;
  }
  std::string error;
  const auto t0 = std::chrono::steady_clock::now();
  auto reader = core::GgufReader::OpenFile(model, &error);
  if (!reader) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  auto weights = model::ModelWeights::Bind(*reader, &error);
  if (!weights) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  if (moe_topk && !weights->SetExpertsUsed(moe_topk, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 2;
  auto tok = text::Tokenizer::FromGguf(*reader, &error);
  if (!tok) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  const auto& c = weights->config;
  const auto& t = weights->ple_table;
  auto ngram = model::NgramTable::Open(reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset, t.rows,
                                       c.ple_head_dim, t.type, &error,
                                       static_cast<std::size_t>(ngram_cache_gb * (1ull << 30)));
  if (!ngram) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  std::printf("toolchain: %s\n", HALO_TOOLCHAIN);
  std::printf("model: %u layers, hidden %u, vocab %u, experts %u/%u, flash-next geometry %s\n", c.num_layers,
              c.hidden_size, c.vocab_size, c.num_experts_used, c.num_experts, c.IsFlashNextGeometry() ? "yes" : "no");

  const bool use_cpu = device == "cpu" || device.starts_with("both");
  const bool use_gpu = device != "cpu";
  const rt::Backend backend = device.ends_with("hrx") ? rt::Backend::kHrx : rt::Backend::kHip;
  std::unique_ptr<rt::Device> dev;
  std::unique_ptr<engine::FlashNextGpu> gpu;
  if (!span_hsaco.empty()) {
    std::ifstream f(span_hsaco, std::ios::binary);
    engine::FlashNextGpu::span_hsaco_.assign(std::istreambuf_iterator<char>(f), {});
    if (engine::FlashNextGpu::span_hsaco_.empty()) return std::fprintf(stderr, "cannot read %s\n", span_hsaco.c_str()), 1;
  }
  if (use_gpu) {
    dev = rt::Device::Create(backend, &error);
    if (!dev) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    if (profile) {
      auto p = std::make_unique<Profiling>();
      p->d = std::move(dev);
      dev = std::move(p);
      eager = true;
    }
    const auto tu = std::chrono::steady_clock::now();
    std::vector<rt::FileSource> sources;
    for (const auto& r : reader->GetMappedRegions()) sources.push_back({r.data, r.size, r.file_descriptor});
    gpu = engine::FlashNextGpu::Create(*dev, *weights, ngram.get(), nullptr, max_context, &error, sources);
    if (!gpu) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    std::printf("gpu (%s) ready in %.1f s\n", rt::BackendName(backend),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tu).count());
    gpu->SetUseGraphs(!eager);
    if (!span_hsaco.empty()) {  // recording off until the traced launch; buffer cleared
      const std::uint32_t off = 0, zero_offset = 0;
      std::vector<std::uint8_t> zeros(std::size_t{1} << 26);
      if (!dev->WriteGlobal("halo_span_enable", &off, 4, &error) ||
          !dev->WriteGlobal("halo_span_sample", &span_sample, 4, &error) ||
          !dev->WriteGlobal("halo_span_offset", &zero_offset, 4, &error) ||
          !dev->WriteGlobal("halo_spans", zeros.data(), zeros.size(), &error) ||
          !dev->WriteGlobal("halo_span_dropped", &off, 4, &error))
        return std::fprintf(stderr, "span trace: %s\n", error.c_str()), 1;
    }
  }
  // After the engine: its allocations come first (see BENCHMARKS.md).
  if (ngram_resident) {
    const auto tr = std::chrono::steady_clock::now();
    if (!ngram->LoadResident(&error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    std::fprintf(stderr, "n-gram table resident (%.1f GB) in %.1f s\n", ngram->Rows() * ngram->RowBytes() / 1e9,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - tr).count());
  }
  if (prefill_bench > 0) {
    if (!gpu) return std::fprintf(stderr, "--prefill-bench needs a GPU device\n"), 2;
    const auto from_file = [&](const std::string& path) {
      std::ifstream f(path);
      const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      const auto ids = tok->Encode(text);
      std::vector<std::int32_t> out(ids.begin(), ids.begin() + std::min<std::size_t>(ids.size(), prefill_bench));
      if (out.size() < static_cast<std::size_t>(prefill_bench))
        std::fprintf(stderr, "%s: only %zu tokens\n", path.c_str(), out.size());
      return out;
    };
    std::vector<std::int32_t> toks(static_cast<std::size_t>(prefill_bench));
    std::uint32_t seed = 1234;
    for (auto& x : toks) {
      seed = seed * 1664525u + 1013904223u;
      x = static_cast<std::int32_t>((seed >> 4) % c.vocab_size);
    }
    if (!prefill_file.empty()) toks = from_file(prefill_file);
    std::vector<float> logits(c.vocab_size);
    if (!prefill_warm.empty()) {
      const auto warm = from_file(prefill_warm);
      if (!gpu->Reset(&error) || !gpu->Prefill(warm, logits, &error))
        return std::fprintf(stderr, "warm-up: %s\n", error.c_str()), 1;
    }
    std::vector<double> secs;
    if (profile) static_cast<Profiling*>(dev.get())->on = true;
    for (int r = 0; r < reps; ++r) {
      if (!gpu->Reset(&error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
      const bool timed = kernel_timing && r + 1 == reps;
      if (!span_hsaco.empty() && r + 1 == reps) gpu->SpanArm();
      if (timed && !gpu->StartKernelTiming(1 << 20, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
      const auto st = std::chrono::steady_clock::now();
      if (!gpu->Prefill(toks, logits, &error)) return std::fprintf(stderr, "prefill: %s\n", error.c_str()), 1;
      if (timed) {
        double span = 0, rate = 0;
        const auto times = gpu->StopKernelTiming(&span, &rate, &error);
        double total = 0;
        for (const auto& k : times) total += k.ms;
        std::printf("kernel timing (last rep, sync-free): span %.1f ms, counter %.0f ticks/ms\n", span, rate);
        for (const auto& k : times)
          std::printf("  %-58s %9.2f ms %5.1f%% %7llu calls %9.1f us/call\n", k.key.c_str(), k.ms, 100 * k.ms / total,
                      static_cast<unsigned long long>(k.calls), 1e3 * k.ms / k.calls);
      }
      secs.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - st).count());
      std::printf("prefill %zu tokens: %.3f s (%.1f tok/s), argmax %u\n", toks.size(), secs.back(),
                  toks.size() / secs.back(), Argmax(logits).best);
    }
    if (!logits_out.empty()) {
      std::ofstream f(logits_out, std::ios::binary);
      f.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(logits[0])));
    }
    if (!span_hsaco.empty()) {
      std::vector<std::uint8_t> recs(std::size_t{1} << 26);
      std::uint32_t dropped = 0;
      if (!dev->ReadGlobal("halo_spans", recs.data(), recs.size(), &error) ||
          !dev->ReadGlobal("halo_span_dropped", &dropped, 4, &error))
        return std::fprintf(stderr, "span trace: %s\n", error.c_str()), 1;
      if (dropped)
        std::printf("span trace: WARNING %u traced waves past the buffer recorded nothing (raise --span-sample)\n",
                    dropped);
      std::vector<std::uint8_t> kept;
      for (std::size_t i = 0; i < recs.size() / 16; ++i)
        if (recs[i * 16 + 15]) kept.insert(kept.end(), recs.begin() + i * 16, recs.begin() + i * 16 + 16);
      const auto n = static_cast<std::uint32_t>(kept.size() / 16);
      std::ofstream f(span_out, std::ios::binary);
      f.write(reinterpret_cast<const char*>(&n), 4);
      f.write(reinterpret_cast<const char*>(kept.data()), static_cast<std::streamsize>(kept.size()));
      std::printf("span trace: %u records of %s launch %lld (of %lld in the last rep) -> %s\n", n,
                  engine::FlashNextGpu::span_kernel_.c_str(), static_cast<long long>(engine::FlashNextGpu::span_occurrence_),
                  static_cast<long long>(gpu->SpanSeen()), span_out.c_str());
    }
    if (ngram) {
      const auto st = ngram->Stats();
      std::printf("n-gram cache: %.1f GiB, %llu hits, %llu misses (%.1f%% hits)\n", st.bytes / double(1ull << 30),
                  static_cast<unsigned long long>(st.hits), static_cast<unsigned long long>(st.misses),
                  st.hits + st.misses ? 100.0 * st.hits / (st.hits + st.misses) : 0.0);
    }
    std::sort(secs.begin(), secs.end());
    std::printf("prefill median: %.1f tok/s over %d reps of %zu tokens (batched %d, int8 GEMMs %d)\n",
                toks.size() / secs[secs.size() / 2], reps, toks.size(), engine::FlashNextGpu::batched_prefill_ ? 1 : 0,
                engine::FlashNextGpu::prefill_int8_gemm_ ? 1 : 0);
    if (profile) {
      auto* p = static_cast<Profiling*>(dev.get());
      std::vector<std::pair<std::string, std::pair<double, long>>> rows(p->time.begin(), p->time.end());
      std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second.first > b.second.first; });
      double total = 0;
      for (auto& r : rows) total += r.second.first;
      std::printf("profile over %d prefills: %.3f s each (each launch timed with a synchronize)\n", reps, total / reps);
      for (auto& [name, v] : rows)
        std::printf("  %-28s %8.1f ms/prefill %6.1f%%  %8ld launches/prefill\n", name.c_str(), 1e3 * v.first / reps,
                    100 * v.first / total, v.second / reps);
    }
    return 0;
  }
  if (session_bench > 0) {
    if (!gpu) return std::fprintf(stderr, "--session-bench needs a GPU device\n"), 2;
    std::string text = prompt;
    if (!prompt_file.empty()) {
      std::ifstream f(prompt_file);
      text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    const auto enc = tok->Encode(text);
    if (enc.empty()) return std::fprintf(stderr, "empty prompt\n"), 2;
    std::vector<std::int32_t> toks;
    while (toks.size() < session_bench) toks.insert(toks.end(), enc.begin(), enc.end());
    toks.resize(session_bench);
    using clk = std::chrono::steady_clock;
    const auto ms = [](clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); };
    auto t = clk::now();
    if (!gpu->Prefill(toks, {}, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    if (!dev->Synchronize(&error)) return 1;
    std::printf("session bench: %u tokens prefilled in %.0f ms (page %zu B)\n", session_bench, ms(t), dev->PageBytes());
    const auto show = [&](const char* what, engine::FlashNextGpu::SessionId id) {
      const auto m = gpu->SessionMemory(id);
      std::printf("  %-26s recurrent %8.1f MB, KV mapped %8.1f MB, KV exclusive %8.1f MB\n", what, m.recurrent / 1e6,
                  m.kv_mapped / 1e6, m.kv_exclusive / 1e6);
    };
    show("session 0", 0);
    engine::FlashNextGpu::SessionId f1, f2;
    t = clk::now();
    if (!gpu->ForkSession(0, &f1, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    std::printf("  fork: %.2f ms\n", ms(t));
    t = clk::now();
    if (!gpu->ForkSession(0, &f2, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    std::printf("  fork again: %.2f ms\n", ms(t));
    show("session 0 after 2 forks", 0);
    show("fork", f1);
    t = clk::now();
    if (!gpu->BindSession(f1, &error)) return 1;
    std::printf("  bind: %.3f ms\n", ms(t));
    std::vector<float> lg(c.vocab_size);
    t = clk::now();
    if (!gpu->Step(toks[0], lg, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    std::printf("  first step of the fork (graph capture): %.1f ms\n", ms(t));
    for (int i = 0; i < 3; ++i) {
      t = clk::now();
      if (!gpu->Step(toks[1 + i], lg, &error)) return 1;
      std::printf("  step of the fork: %.1f ms\n", ms(t));
    }
    show("fork after 4 steps", f1);
    auto snap = gpu->NewSnapshot(&error);
    if (!snap) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    t = clk::now();
    if (!gpu->Save(*snap, &error)) return 1;
    std::printf("  Save (whole-state snapshot copy): %.2f ms\n", ms(t));
    t = clk::now();
    if (!gpu->Restore(*snap, &error)) return 1;
    std::printf("  Restore: %.2f ms\n", ms(t));
    if (!gpu->BindSession(0, &error)) return 1;
    t = clk::now();
    if (!gpu->DropSession(f1, &error) || !gpu->DropSession(f2, &error)) return 1;
    std::printf("  drop 2 forks: %.2f ms\n", ms(t));
    std::size_t fr = 0;
    (void)fr;
    return 0;
  }
  if (!count_files.empty()) {
    std::ifstream list(count_files);
    std::vector<std::uint64_t> cnt(c.vocab_size, 0);
    std::string path;
    std::size_t files = 0, total = 0;
    while (std::getline(list, path)) {
      std::ifstream f(path);
      const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      for (auto t : tok->Encode(text, true))
        if (static_cast<std::uint32_t>(t) < c.vocab_size) ++cnt[t], ++total;
      ++files;
    }
    std::vector<std::int32_t> order(c.vocab_size);
    for (std::uint32_t i = 0; i < c.vocab_size; ++i) order[i] = static_cast<std::int32_t>(i);
    std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) { return cnt[a] > cnt[b]; });
    std::vector<std::int32_t> keep;
    std::vector<char> in(c.vocab_size, 0);
    for (const char* s : {"<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>"}) {
      const auto ids = tok->Encode(s, true);
      if (ids.size() == 1 && !in[ids[0]]) in[ids[0]] = 1, keep.push_back(ids[0]);
    }
    std::size_t seen = 0;
    for (auto t : order) {
      if (keep.size() >= count_top || cnt[t] == 0) break;
      if (!in[t]) in[t] = 1, keep.push_back(t), seen += cnt[t];
    }
    for (std::uint32_t t = 0; t < c.vocab_size && keep.size() < count_top; ++t)
      if (!in[t]) in[t] = 1, keep.push_back(static_cast<std::int32_t>(t));
    std::sort(keep.begin(), keep.end());
    std::ofstream o(mtp_vocab, std::ios::binary);
    o.write(reinterpret_cast<const char*>(keep.data()), static_cast<std::streamsize>(keep.size() * 4));
    std::printf("counted %zu tokens in %zu files; kept %zu ids covering %.4f of the corpus tokens\n", total, files,
                keep.size(), double(seen) / double(std::max<std::size_t>(total, 1)));
    return 0;
  }
  if (!mtp_cpu_replay.empty()) {
    if (mtp_model.empty()) return std::fprintf(stderr, "--mtp-cpu-replay needs --mtp-model\n"), 2;
    auto mreader = core::GgufReader::OpenFile(mtp_model, &error);
    if (!mreader) return 1;
    auto mtp = model::MtpWeights::Bind(*mreader, c, &error);
    if (!mtp) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    const auto read_all = [](const std::string& f, auto& v) {
      std::ifstream in(f, std::ios::binary | std::ios::ate);
      v.resize(static_cast<std::size_t>(in.tellg()) / sizeof(v[0]));
      in.seekg(0);
      in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(v[0])));
    };
    std::vector<std::int32_t> tk, gd;
    std::vector<float> hid;
    read_all(mtp_cpu_replay + ".tok", tk);
    read_all(mtp_cpu_replay + ".hid", hid);
    read_all(mtp_cpu_replay + ".dft", gd);
    const std::size_t L = static_cast<std::size_t>(tk[0]), D = c.HcDim(), V = c.vocab_size, K = std::max(1u, mtp_depth);
    std::vector<std::int32_t> toks(tk.begin() + 1, tk.end());
    const std::size_t npos = hid.size() / D;
    auto ref = std::make_unique<model::ReferenceModel>(*weights, ngram.get(), static_cast<std::uint32_t>(npos + K + 2));
    std::vector<float> l(V);
    const auto argmax_v = [&](const std::vector<float>& v) { return static_cast<std::int32_t>(std::max_element(v.begin(), v.end()) - v.begin()); };
    std::vector<double> cond_num(K, 0), cond_den(K, 0), chain(K, 0);
    std::size_t positions = 0, agree = 0, compared = 0;
    for (std::size_t t = 0; t + 1 < toks.size() && t < npos; ++t) {
      const std::span<const float> h(hid.data() + t * D, D);
      const bool draft = t + 1 >= L && positions < mtp_cpu_n && t + 1 + K < toks.size();
      if (!draft) {
        if (!ref->MtpStep(*mtp, toks[t + 1], h, {}, nullptr, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
        if (t + 1 >= L && positions >= mtp_cpu_n) break;
        continue;
      }
      std::vector<std::int32_t> d(K);
      if (!ref->MtpStep(*mtp, toks[t + 1], h, l, nullptr, &error)) return 1;
      auto chain_ref = std::make_unique<model::ReferenceModel>(*ref);  // the chain runs on a copy
      d[0] = argmax_v(l);
      for (std::size_t j = 1; j < K; ++j) {
        if (!chain_ref->MtpStep(*mtp, d[j - 1], {}, l, nullptr, &error)) return 1;
        d[j] = argmax_v(l);
      }
      double run = 1;
      for (std::size_t j = 0; j < K; ++j) {
        const double a = d[j] == toks[t + 2 + j] ? 1.0 : 0.0;
        if (run > 0) cond_num[j] += a, cond_den[j] += 1;
        run *= a;
        chain[j] += run;
        const std::size_t gi = positions * K + j;
        if (gi < gd.size()) compared++, agree += gd[gi] == d[j];
      }
      ++positions;
      std::fprintf(stderr, "cpu position %zu done\n", positions);
    }
    std::printf("cpu reference drafts (%s): %zu positions; GPU drafts agree %zu of %zu\n", mtp_cpu_replay.c_str(), positions,
                agree, compared);
    double e = 1;
    for (std::size_t j = 0; j < K; ++j) {
      e += chain[j] / std::max<std::size_t>(positions, 1);
      std::printf("  depth %zu: conditional acceptance %.3f (%.0f of %.0f), E[tokens/cycle] %.3f\n", j + 1,
                  cond_num[j] / std::max(1.0, cond_den[j]), cond_num[j], cond_den[j], e);
    }
    return 0;
  }
  if (!wmma_probe.empty()) {
    if (!gpu) return 2;
    std::vector<std::vector<std::uint32_t>> traces;  // [file][steps * L * k]
    for (std::size_t p = 0; p < wmma_probe.size();) {
      const auto q = wmma_probe.find(',', p);
      std::ifstream f(wmma_probe.substr(p, q == std::string::npos ? std::string::npos : q - p), std::ios::binary | std::ios::ate);
      std::vector<std::uint32_t> v(static_cast<std::size_t>(f.tellg()) / 4);
      f.seekg(0);
      f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
      traces.push_back(std::move(v));
      p = q == std::string::npos ? wmma_probe.size() : q + 1;
    }
    const std::uint32_t L = c.num_layers, k = c.num_experts_used;
    const std::size_t per = std::size_t{L} * k;
    std::mt19937 rng(7);
    const auto ids_tok = tok->Encode("The quick brown fox jumps over the lazy dog. In the beginning");
    std::vector<std::int32_t> warm(ids_tok.begin(), ids_tok.end());
    if (!gpu->Prefill(warm, {}, &error)) return 1;
    std::int32_t token = warm.back();
    for (std::uint32_t B = 1; B <= 8; ++B) {
      std::vector<std::vector<std::uint32_t>> ids(L);
      const auto& t0 = traces[rng() % traces.size()];
      const std::size_t base = rng() % (t0.size() / per - 8);
      for (std::uint32_t b = 0; b < B; ++b) {
        const auto& tr = wmma_consec ? t0 : traces[rng() % traces.size()];
        const std::size_t step = wmma_consec ? base + b : rng() % (tr.size() / per);
        for (std::uint32_t l = 0; l < L; ++l)
          ids[l].insert(ids[l].end(), tr.begin() + static_cast<std::ptrdiff_t>(step * per + l * k),
                        tr.begin() + static_cast<std::ptrdiff_t>(step * per + (l + 1) * k));
      }
      double U = 0;
      for (const auto& v : ids) {
        std::vector<std::uint32_t> s(v);
        std::sort(s.begin(), s.end());
        U += static_cast<double>(std::unique(s.begin(), s.end()) - s.begin());
      }
      if (!gpu->ProbeSetRouting(B, ids, &error) || (B == 1 && !gpu->ProbeRun(1, true, &error)) ||
          !gpu->StartKernelTiming(1 << 20, &error))
        return std::fprintf(stderr, "%s\n", error.c_str()), 1;
      constexpr int kSteps = 6;
      for (int s = 0; s < kSteps; ++s)
        if (!gpu->Step(token, {}, &error) || !gpu->ProbeRun(B, false, &error))
          return std::fprintf(stderr, "%s\n", error.c_str()), 1;
      double span = 0, rate = 0;
      const auto times = gpu->StopKernelTiming(&span, &rate, &error);
      std::printf("probe B %u (expert union %.1f a layer)\n", B, U / L);
      for (const auto& kt : times)
        std::printf("  B%u %-58s %8.3f ms/step %6.1f calls/step %8.1f us/call\n", B, kt.key.c_str(), kt.ms / kSteps,
                    kt.calls / double(kSteps), 1e3 * kt.ms / kt.calls);
    }
    return 0;
  }
  if (!mtp_model.empty()) {
    if (!gpu) return std::fprintf(stderr, "--mtp-model needs a GPU device\n"), 2;
    auto mreader = core::GgufReader::OpenFile(mtp_model, &error);
    if (!mreader) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    auto mtp = model::MtpWeights::Bind(*mreader, c, &error);
    if (!mtp) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    if (!gpu->LoadMtp(*mtp, &error)) return std::fprintf(stderr, "LoadMtp: %s\n", error.c_str()), 1;
    const std::uint32_t K = std::max(1u, mtp_depth);
    const std::size_t V = c.vocab_size;
    std::vector<std::int32_t> sub;  // trimmed draft vocabulary (empty: full)
    if (!mtp_vocab.empty()) {
      std::ifstream f(mtp_vocab, std::ios::binary | std::ios::ate);
      sub.resize(static_cast<std::size_t>(f.tellg()) / 4);
      f.seekg(0);
      f.read(reinterpret_cast<char*>(sub.data()), static_cast<std::streamsize>(sub.size() * 4));
      if (!gpu->SetDraftVocab(sub, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
      std::printf("trimmed draft head: %zu tokens\n", sub.size());
    }
    const std::size_t QV = sub.empty() ? V : sub.size();
    const auto qid = [&](std::int32_t i) { return sub.empty() ? i : sub[static_cast<std::size_t>(i)]; };
    std::vector<std::string> files;
    for (std::size_t p = 0; p < mtp_prompts.size();) {
      const auto q = mtp_prompts.find(',', p);
      files.push_back(mtp_prompts.substr(p, q == std::string::npos ? std::string::npos : q - p));
      p = q == std::string::npos ? mtp_prompts.size() : q + 1;
    }
    std::mt19937_64 rng(12345);
    const auto probs = [&](const float* l, std::vector<double>& p) {  // softmax at mtp_temp (> 0)
      p.resize(V);
      double mx = -1e300, sum = 0;
      for (std::size_t i = 0; i < V; ++i) mx = std::max(mx, static_cast<double>(l[i]));
      for (std::size_t i = 0; i < V; ++i) sum += p[i] = std::exp((l[i] - mx) / mtp_temp);
      for (auto& x : p) x /= sum;
    };
    const auto argmax = [&](const float* l) { return static_cast<std::int32_t>(std::max_element(l, l + V) - l); };
    std::ofstream out;
    if (!mtp_out.empty()) out.open(mtp_out, std::ios::app);
    double t_head_sum = 0, t_base_sum = 0;
    std::size_t t_head_n = 0, t_base_n = 0, t_chain_n = 0;
    double t_chain_sum = 0;
    for (const auto& file : files) {
      std::ifstream f(file);
      const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      const auto enc = tok->Encode(text, true);
      std::vector<std::int32_t> toks(enc.begin(), enc.end());
      const std::size_t L = toks.size();
      // Pass 1: AR decode, every generated position's target logits kept.
      if (!gpu->Reset(&error)) return 1;
      if (L > 1 && !gpu->Prefill(std::span<const std::int32_t>(toks).first(L - 1), {}, &error))
        return std::fprintf(stderr, "%s\n", error.c_str()), 1;
      std::vector<float> tl(static_cast<std::size_t>(n) * V);  // tl[g]: logits after the step at position L-1+g
      std::vector<double> p;
      for (int g = 0; g < n; ++g) {
        float* lg = tl.data() + static_cast<std::size_t>(g) * V;
        if (!gpu->Step(toks.back(), std::span<float>(lg, V), &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
        std::int32_t next = argmax(lg);
        if (mtp_temp > 0) {
          probs(lg, p);
          double u = std::uniform_real_distribution<double>(0, 1)(rng), cum = 0;
          for (std::size_t i = 0; i < V; ++i)
            if ((cum += p[i]) > u) {
              next = static_cast<std::int32_t>(i);
              break;
            }
        }
        toks.push_back(next);
      }
      // Pass 2: replay token by token; MTP base step at every position, drafts at generated positions.
      if (!gpu->Reset(&error)) return 1;
      gpu->MtpRewind(0);
      std::unique_ptr<model::ReferenceModel> ref;
      if (mtp_check > 0) ref = std::make_unique<model::ReferenceModel>(*weights, ngram.get(), mtp_check + 1);
      std::vector<std::vector<float>> q(K, std::vector<float>(QV));
      std::vector<float> hidden, cpu_l(V);
      // greedy: hits[j] = positions whose drafts 1..j+1 all matched; sampled: sum over positions of prod alpha
      std::vector<double> chain(K, 0.0), cond_num(K, 0.0), cond_den(K, 0.0);
      std::size_t positions = 0;
      using clk = std::chrono::steady_clock;
      std::ofstream dhid, ddft;
      const std::string pname = file.substr(file.find_last_of('/') + 1);
      if (!mtp_dump.empty()) {
        std::ofstream dt(mtp_dump + "." + pname + ".tok", std::ios::binary);
        const std::int32_t lp = static_cast<std::int32_t>(L);
        dt.write(reinterpret_cast<const char*>(&lp), 4);
        dt.write(reinterpret_cast<const char*>(toks.data()), static_cast<std::streamsize>(toks.size() * 4));
        dhid.open(mtp_dump + "." + pname + ".hid", std::ios::binary);
        ddft.open(mtp_dump + "." + pname + ".dft", std::ios::binary);
      }
      for (std::size_t t = 0; t + 1 < toks.size(); ++t) {
        if (!gpu->Step(toks[t], {}, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
        if (dhid.is_open()) {
          if (!gpu->TrunkHidden(&hidden, &error)) return 1;
          dhid.write(reinterpret_cast<const char*>(hidden.data()), static_cast<std::streamsize>(hidden.size() * 4));
        }
        const std::size_t g0 = t + 1 >= L ? t + 1 - L : SIZE_MAX;  // tl index of the step at position t
        // Drafts only where the targets of all K depths are known: q_j vs tl[g0 + j].
        const bool draft = t + 1 >= L && g0 + K < static_cast<std::size_t>(n);
        const bool check = t < mtp_check;
        static const bool mtp_timing = std::getenv("HALO_MTP_TIMING") != nullptr;
        if (mtp_timing && t == 20 && !gpu->StartKernelTiming(1 << 20, &error)) return 1;
        if (!dev->Synchronize(&error)) return 1;  // the trunk step's time is not the draft block's
        auto st = clk::now();
        if (mtp_timing && t == 30) {
          double span = 0, rate = 0;
          const auto times = gpu->StopKernelTiming(&span, &rate, &error);
          for (const auto& k : times)
            std::printf("  mtp-timing %-58s %8.3f ms/10 steps %6.1f calls\n", k.key.c_str(), k.ms, double(k.calls));
        }
        if (!gpu->MtpStep(toks[t + 1], true, draft || check ? std::span<float>(q[0]) : std::span<float>{}, &error))
          return std::fprintf(stderr, "mtp: %s\n", error.c_str()), 1;
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - st).count();
        if (draft || check) t_head_sum += ms, ++t_head_n;
        else t_base_sum += ms, ++t_base_n;
        if (check) {
          if (!gpu->TrunkHidden(&hidden, &error) ||
              !ref->MtpStep(*mtp, toks[t + 1], hidden, cpu_l, nullptr, &error))
            return std::fprintf(stderr, "check: %s\n", error.c_str()), 1;
          double d = 0, m = 0;
          for (std::size_t i = 0; i < V && QV == V; ++i) d += (q[0][i] - cpu_l[i]) * double(q[0][i] - cpu_l[i]), m += cpu_l[i] * double(cpu_l[i]);
          std::printf("mtp check position %zu: relL2 %.3g, argmax gpu %d cpu %d\n", t, std::sqrt(d / m), argmax(q[0].data()),
                      argmax(cpu_l.data()));
        }
        if (!draft) continue;
        const std::uint32_t base = gpu->MtpPosition();
        std::vector<std::int32_t> d(K);
        const auto qarg = [&](const std::vector<float>& l) {
          return qid(static_cast<std::int32_t>(std::max_element(l.begin(), l.end()) - l.begin()));
        };
        d[0] = qarg(q[0]);
        for (std::uint32_t j = 1; j < K; ++j) {
          const std::int32_t in = mtp_temp > 0 ? toks[t + 1 + j] : d[j - 1];
          const auto sc = clk::now();
          if (!gpu->MtpStep(in, false, q[j], &error)) return std::fprintf(stderr, "mtp: %s\n", error.c_str()), 1;
          t_chain_sum += std::chrono::duration<double, std::milli>(clk::now() - sc).count(), ++t_chain_n;
          d[j] = qarg(q[j]);
        }
        gpu->MtpRewind(base);
        if (ddft.is_open()) ddft.write(reinterpret_cast<const char*>(d.data()), static_cast<std::streamsize>(d.size() * 4));
        ++positions;
        double run = 1;
        for (std::uint32_t j = 0; j < K; ++j) {
          double a;
          if (mtp_temp > 0) {
            std::vector<double> pq;
            probs(tl.data() + (g0 + 1 + j) * V, p);
            a = 0;
            if (sub.empty()) {
              probs(q[j].data(), pq);
              for (std::size_t i = 0; i < V; ++i) a += std::min(p[i], pq[i]);
            } else {  // q over the subset only
              double mx = -1e300, sum = 0;
              for (float v : q[j]) mx = std::max(mx, double(v));
              pq.resize(QV);
              for (std::size_t i = 0; i < QV; ++i) sum += pq[i] = std::exp((q[j][i] - mx) / mtp_temp);
              for (std::size_t i = 0; i < QV; ++i) a += std::min(p[static_cast<std::size_t>(sub[i])], pq[i] / sum);
            }
          } else {
            a = d[j] == toks[t + 2 + j] ? 1.0 : 0.0;
          }
          if (mtp_temp > 0 || run > 0) cond_num[j] += a, cond_den[j] += 1;  // greedy: given drafts 1..j matched
          run *= a;
          chain[j] += run;
        }
      }
      std::string name = file.substr(file.find_last_of('/') + 1);
      std::printf("mtp %s %s: %zu prompt tokens, %d generated, %zu draft positions\n", name.c_str(),
                  mtp_temp > 0 ? ("T=" + std::to_string(mtp_temp)).c_str() : "greedy", L, n, positions);
      double e = 1;
      for (std::uint32_t j = 0; j < K; ++j) {
        e += chain[j] / static_cast<double>(positions);
        std::printf("  depth %u: conditional acceptance %.3f (%.0f of %.0f), P(drafts 1..%u accepted) %.3f, E[tokens/cycle | depth %u] %.3f\n",
                    j + 1, cond_num[j] / std::max(1.0, cond_den[j]), cond_num[j], cond_den[j], j + 1,
                    chain[j] / static_cast<double>(positions), j + 1, e);
        if (out.is_open())
          out << name << ' ' << (mtp_temp > 0 ? mtp_temp : 0.0) << ' ' << j + 1 << ' ' << cond_num[j] << ' ' << cond_den[j]
              << ' ' << chain[j] << ' ' << positions << '\n';
      }
      std::printf("  text: %s\n", tok->Decode(std::vector<text::TokenId>(toks.begin() + static_cast<std::ptrdiff_t>(L), toks.end())).c_str());
    }
    std::printf("mtp chained draft step (eager, with head, download): %.2f ms (%zu)\n", t_chain_sum / std::max<std::size_t>(1, t_chain_n), t_chain_n);
    std::printf("mtp step (eager): with head %.2f ms (%zu), headless %.2f ms (%zu)\n", t_head_sum / std::max<std::size_t>(1, t_head_n),
                t_head_n, t_base_sum / std::max<std::size_t>(1, t_base_n), t_base_n);
    return 0;
  }
  if (batch > 0) {
    if (!gpu) return std::fprintf(stderr, "--batch needs a GPU device\n"), 2;
    std::vector<std::string> files;
    for (std::size_t p = 0; p < batch_prompts.size();) {
      const auto q = batch_prompts.find(',', p);
      files.push_back(batch_prompts.substr(p, q == std::string::npos ? std::string::npos : q - p));
      p = q == std::string::npos ? batch_prompts.size() : q + 1;
    }
    if (files.empty()) return std::fprintf(stderr, "--batch needs --batch-prompts\n"), 2;
    using Sid = engine::FlashNextGpu::SessionId;
    std::vector<Sid> sids(batch);
    std::vector<std::vector<std::int32_t>> prompts(batch);
    for (std::uint32_t b = 0; b < batch; ++b) {
      std::ifstream f(files[b % files.size()]);
      const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      const auto enc = tok->Encode(text);
      // Session b: its file's tokens, shortened by 7 b tokens so equal files still differ.
      const std::size_t drop = std::min<std::size_t>(enc.size() - 1, 7 * (b / files.size()));
      prompts[b].assign(enc.begin(), enc.end() - static_cast<std::ptrdiff_t>(drop));
      if (b == 0) sids[b] = 0;
      else if (!gpu->NewSession(&sids[b], &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
      if (!gpu->BindSession(sids[b], &error) ||
          !gpu->Prefill(std::span<const std::int32_t>(prompts[b]).first(prompts[b].size() - 1), {}, &error))
        return std::fprintf(stderr, "prefill: %s\n", error.c_str()), 1;
    }
    if (!gpu->BindSession(0, &error)) return 1;
    std::printf("batch %u: batched kernels %s\n", batch, gpu->BatchSupported() ? "yes" : gpu->BatchUnsupported().c_str());
    static sampling::SamplingConfig greedy;
    std::vector<std::int32_t> next(batch);
    for (std::uint32_t b = 0; b < batch; ++b) next[b] = prompts[b].back();
    std::vector<std::vector<std::int32_t>> fed(batch);
    std::vector<std::vector<float>> logits(batch, std::vector<float>(device_sample ? 0 : c.vocab_size));
    std::vector<std::vector<float>> all(batch);  // every step's logits per entry (check mode)
    std::vector<engine::FlashNextGpu::SampleRequest> reqs(batch);
    std::vector<engine::FlashNextGpu::SampleResult> res(batch);
    std::vector<double> ms;
    for (int step = 0; step < n; ++step) {
      std::vector<engine::FlashNextGpu::BatchEntry> entries(batch);
      for (std::uint32_t b = 0; b < batch; ++b) {
        entries[b].session = sids[b];
        entries[b].token = next[b];
        if (device_sample) {
          reqs[b] = {&greedy, {}, 0};
          entries[b].sample = &reqs[b];
          entries[b].result = &res[b];
        } else {
          entries[b].logits = logits[b];
        }
      }
      if (decode_timing && step == 4 && !gpu->StartKernelTiming(1 << 22, &error))
        return std::fprintf(stderr, "%s\n", error.c_str()), 1;
      const auto st = std::chrono::steady_clock::now();
      if (!gpu->StepBatch(entries, &error)) return std::fprintf(stderr, "batch step: %s\n", error.c_str()), 1;
      if (step >= 4) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - st).count());
      for (std::uint32_t b = 0; b < batch; ++b) {
        fed[b].push_back(next[b]);
        if (device_sample) {
          next[b] = res[b].token;
        } else {
          next[b] = static_cast<std::int32_t>(Argmax(logits[b]).best);
          if (batch_check) all[b].insert(all[b].end(), logits[b].begin(), logits[b].end());
        }
      }
    }
    if (decode_timing) {
      double span = 0, rate = 0, total = 0;
      const auto times = gpu->StopKernelTiming(&span, &rate, &error);
      const double steps = static_cast<double>(ms.size());
      for (const auto& k : times) total += k.ms;
      std::printf("batch kernel timing (eager, %.0f steps): kernels %.2f ms/step\n", steps, total / steps);
      for (const auto& k : times)
        std::printf("  %-58s %8.3f ms/step %5.1f%% %6.1f calls/step %8.1f us/call\n", k.key.c_str(), k.ms / steps,
                    100 * k.ms / total, k.calls / steps, 1e3 * k.ms / k.calls);
    }
    if (!ms.empty()) {
      std::sort(ms.begin(), ms.end());
      const double med = ms[ms.size() / 2];
      std::printf("batch %u steady: median %.2f ms/step, %.2f tok/s aggregate, %.2f tok/s per session, min %.2f, over %zu steps\n",
                  batch, med, 1000.0 * batch / med, 1000.0 / med, ms.front(), ms.size());
    }
    for (std::uint32_t b = 0; b < batch && b < 4; ++b)
      std::printf("session %u: %s\n", b, tok->Decode(std::vector<text::TokenId>(fed[b].begin() + 1, fed[b].end())).c_str());
    if (batch_check && !device_sample) {
      // Each session again, alone, from the same prompt: Prefill + Step with the same tokens.
      std::size_t same = 0, total = 0;
      for (std::uint32_t b = 0; b < batch; ++b) {
        Sid solo;
        if (!gpu->NewSession(&solo, &error) || !gpu->BindSession(solo, &error) ||
            !gpu->Prefill(std::span<const std::int32_t>(prompts[b]).first(prompts[b].size() - 1), {}, &error))
          return std::fprintf(stderr, "%s\n", error.c_str()), 1;
        std::vector<float> l(c.vocab_size);
        for (std::size_t t = 0; t < fed[b].size(); ++t) {
          if (!gpu->Step(fed[b][t], l, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
          ++total;
          same += std::memcmp(l.data(), all[b].data() + t * c.vocab_size, c.vocab_size * 4) == 0;
        }
        if (!gpu->BindSession(0, &error) || !gpu->DropSession(solo, &error)) return 1;
      }
      std::printf("batch check: %zu of %zu entry-steps byte-identical to the session stepped alone%s\n", same, total,
                  same == total ? " (EXACT)" : " (DIFFERS)");
    }
    return 0;
  }
  std::unique_ptr<model::ReferenceModel> cpu;
  if (use_cpu) cpu = std::make_unique<model::ReferenceModel>(*weights, ngram.get(), max_context);

  if (!prompt_file.empty()) {
    std::ifstream f(prompt_file);
    prompt.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  }
  std::vector<text::TokenId> ids = tok->Encode(prompt);
  std::printf("prompt: %zu tokens\n", ids.size());
  std::vector<float> lc(c.vocab_size), lg(c.vocab_size);
  std::vector<text::TokenId> generated;
  std::vector<double> step_ms;  // decode steps after warm-up (graph capture etc.)
  std::ofstream decode_logits, route_out;
  if (!route_trace.empty()) route_out.open(route_trace, std::ios::binary);
  std::vector<std::uint32_t> route_ids;
  if (!decode_logits_out.empty()) decode_logits.open(decode_logits_out, std::ios::binary);
  std::size_t first_step = 0;
  if (prefill_prompt && ids.size() > 1) {
    if (!gpu || cpu) return std::fprintf(stderr, "--prefill-prompt needs a GPU-only run\n"), 2;
    const std::vector<std::int32_t> head(ids.begin(), ids.end() - 1);
    const auto tp = std::chrono::steady_clock::now();
    if (!gpu->Prefill(head, {}, &error)) return std::fprintf(stderr, "prefill: %s\n", error.c_str()), 1;
    std::printf("prefill %zu prompt tokens: %.3f s\n", head.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
    first_step = ids.size() - 1;
  }
  bool timing_on = false;
  const auto ts = std::chrono::steady_clock::now();
  double worst = 0;
  for (std::size_t step = first_step; step < ids.size() + static_cast<std::size_t>(n) - 1; ++step) {
    if (decode_timing && !timing_on && generated.size() == 4) {  // after warm-up, as step_ms
      if (!gpu->StartKernelTiming(1 << 20, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
      timing_on = true;
    }
    const auto token = static_cast<std::int32_t>(step < ids.size() ? ids[step] : generated.back());
    const bool want = step + 1 >= ids.size();
    if (cpu && !cpu->Step(token, want ? std::span<float>(lc) : std::span<float>{}, {}, &error))
      return std::fprintf(stderr, "cpu: %s\n", error.c_str()), 1;
    const auto st = std::chrono::steady_clock::now();
    if (gpu && want && device_sample && !cpu) {  // greedy on the device: only the token comes back
      static sampling::SamplingConfig greedy;
      engine::FlashNextGpu::SampleResult r;
      if (!gpu->StepSample(token, {&greedy, {}, 0}, &r, &error)) return std::fprintf(stderr, "gpu: %s\n", error.c_str()), 1;
      std::fill(lg.begin(), lg.end(), 0.0f);
      lg[static_cast<std::size_t>(r.token)] = 1.0f;
    } else if (gpu && !gpu->Step(token, want ? std::span<float>(lg) : std::span<float>{}, &error)) {
      return std::fprintf(stderr, "gpu: %s\n", error.c_str()), 1;
    }
    if (route_out.is_open() && gpu) {  // [layers][k] uint32 per step, prompt steps included
      if (!gpu->RouteTrace(&route_ids, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
      route_out.write(reinterpret_cast<const char*>(route_ids.data()), static_cast<std::streamsize>(route_ids.size() * 4));
    }
    if (want && generated.size() >= 4) step_ms.push_back(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - st).count());
    if (step < 2 || step + 1 == ids.size())
      std::printf("step %zu: %.3f s\n", step, std::chrono::duration<double>(std::chrono::steady_clock::now() - st).count());
    if (profile && step == ids.size()) static_cast<Profiling*>(dev.get())->on = true;  // after warm-up
    if (!want) continue;
    const Top2 top = Argmax(gpu ? lg : lc);
    std::string line;
    if (cpu && gpu) {
      const Top2 ct = Argmax(lc);
      const double rel = RelL2(lg, lc);
      worst = std::max(worst, rel);
      line = " cpu " + std::to_string(ct.best) + (ct.best == top.best ? "" : " DIFFERS") + " relL2 " +
             std::to_string(rel);
    }
    if (decode_logits.is_open())
      decode_logits.write(reinterpret_cast<const char*>(lg.data()), static_cast<std::streamsize>(lg.size() * sizeof(float)));
    generated.push_back(top.best);
    const auto piece = tok->Decode(std::span<const text::TokenId>(&generated.back(), 1));
    std::printf("%3zu  %6u  margin %7.3f%s  %s\n", generated.size(), top.best, top.margin, line.c_str(),
                piece.c_str());
    std::fflush(stdout);
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
  if (timing_on) {
    double span = 0, rate = 0, total = 0;
    const auto times = gpu->StopKernelTiming(&span, &rate, &error);
    const double steps = static_cast<double>(step_ms.size());
    for (const auto& k : times) total += k.ms;
    std::printf("decode kernel timing (sync-free, eager, %.0f steps): span %.1f ms, kernels %.2f ms/step\n", steps,
                span, total / steps);
    for (const auto& k : times)
      std::printf("  %-58s %8.3f ms/step %5.1f%% %6.1f calls/step %8.1f us/call\n", k.key.c_str(), k.ms / steps,
                  100 * k.ms / total, k.calls / steps, 1e3 * k.ms / k.calls);
  }
  if (!step_ms.empty()) {
    std::sort(step_ms.begin(), step_ms.end());
    std::printf("steady decode: median %.2f ms/step (%.2f tok/s), min %.2f ms, over %zu steps\n",
                step_ms[step_ms.size() / 2], 1000.0 / step_ms[step_ms.size() / 2], step_ms.front(), step_ms.size());
  }
  std::printf("text: %s%s\n", prompt.c_str(), tok->Decode(generated).c_str());
  std::printf("%zu steps in %.2f s (%.2f steps/s)%s\n", ids.size() + generated.size() - 1, secs,
              (ids.size() + generated.size() - 1) / secs,
              cpu && gpu ? (" worst relL2 " + std::to_string(worst)).c_str() : "");
  if (profile) {
    auto* p = static_cast<Profiling*>(dev.get());
    std::vector<std::pair<std::string, std::pair<double, long>>> rows(p->time.begin(), p->time.end());
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second.first > b.second.first; });
    double total = 0;
    long launches = 0;
    for (auto& r : rows) total += r.second.first, launches += r.second.second;
    const auto steps = static_cast<double>(generated.size() - 1);
    std::printf("profile over %.0f steps: %.1f ms/step, %.0f launches/step (each timed with a synchronize)\n", steps,
                1e3 * total / steps, launches / steps);
    for (auto& [name, v] : rows)
      std::printf("  %-28s %8.2f ms/step %6.1f%%  %7.0f launches/step  %7.1f us/launch\n", name.c_str(),
                  1e3 * v.first / steps, 100 * v.first / total, v.second / steps, 1e6 * v.first / v.second);
  }
  std::printf("total %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}
