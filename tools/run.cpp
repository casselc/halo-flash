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
  // --span-hsaco FILE --span-kernel NAME --span-occurrence N --span-out FILE [--span-sample N]: load the traced
  // code object and record the N-th launch (from 0) of kernel NAME in the last prefill rep (docs/TRACING.md).
  std::string span_hsaco, span_out;
  std::string logits_out;  // --logits-out FILE: the last prefill's logits (fp32), for bitwise comparison of builds
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
    else if (a == "--prefill-chunk") engine::FlashNextGpu::prefill_chunk_ = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (a == "--router-bf16x3") engine::FlashNextGpu::prefill_router_bf16x3_ = std::string(argv[i + 1]) == "1";
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
  std::unique_ptr<model::ReferenceModel> cpu;
  if (use_cpu) cpu = std::make_unique<model::ReferenceModel>(*weights, ngram.get(), max_context);

  std::vector<text::TokenId> ids = tok->Encode(prompt);
  std::printf("prompt: %zu tokens\n", ids.size());
  std::vector<float> lc(c.vocab_size), lg(c.vocab_size);
  std::vector<text::TokenId> generated;
  std::vector<double> step_ms;  // decode steps after warm-up (graph capture etc.)
  const auto ts = std::chrono::steady_clock::now();
  double worst = 0;
  for (std::size_t step = 0; step < ids.size() + static_cast<std::size_t>(n) - 1; ++step) {
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
    generated.push_back(top.best);
    const auto piece = tok->Decode(std::span<const text::TokenId>(&generated.back(), 1));
    std::printf("%3zu  %6u  margin %7.3f%s  %s\n", generated.size(), top.best, top.margin, line.c_str(),
                piece.c_str());
    std::fflush(stdout);
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
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
