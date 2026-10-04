// halo_run: greedy decoding on the real model, on the CPU reference, the
// GPU engine, or both in lockstep (reporting their logit difference per
// step). Used for the P1/P2 gates against gufo's greedy output.
//
//   halo_run --model <shard-00001.gguf> --prompt "text" [--n 16]
//            [--device cpu|hip|hrx|both-hip|both-hrx] [--max-context 512]
//
// The prompt is encoded raw (no chat template), as `gufo prompt --raw`.
// Output: one line per generated token (id, top-2 logit margin, piece),
// then the decoded text and timing.

#include <algorithm>
#include <chrono>
#include <map>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
  int n = 16;
  bool eager = false, profile = false, device_sample = false;
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
  auto tok = text::Tokenizer::FromGguf(*reader, &error);
  if (!tok) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  const auto& c = weights->config;
  const auto& t = weights->ple_table;
  auto ngram = model::NgramTable::Open(reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset, t.rows,
                                       c.ple_head_dim, t.type, &error);
  if (!ngram) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  std::printf("model: %u layers, hidden %u, vocab %u, experts %u/%u, flash-next geometry %s\n", c.num_layers,
              c.hidden_size, c.vocab_size, c.num_experts_used, c.num_experts, c.IsFlashNextGeometry() ? "yes" : "no");

  const bool use_cpu = device == "cpu" || device.starts_with("both");
  const bool use_gpu = device != "cpu";
  const rt::Backend backend = device.ends_with("hrx") ? rt::Backend::kHrx : rt::Backend::kHip;
  std::unique_ptr<rt::Device> dev;
  std::unique_ptr<engine::FlashNextGpu> gpu;
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
