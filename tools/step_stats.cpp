// Launches, host syncs and transfers per Flash-Next decode step, and wall time
// per step and per launch, on every built backend. P3 uses it to track
// dispatch overhead. Usage: HALO_TINY_HF_DIR=<dir> halo_step_stats [gguf]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "core/gguf.hpp"
#include "engine/flash_next_gpu.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
using namespace halo;
struct Counting : rt::Device {
  std::unique_ptr<rt::Device> d; long launches = 0, syncs = 0, xfers = 0, graphs = 0;
  rt::Backend backend() const noexcept override { return d->backend(); }
  rt::Buffer Alloc(std::size_t b, std::string* e) override { return d->Alloc(b, e); }
  void Free(rt::Buffer& b) noexcept override { d->Free(b); }
  bool Upload(const rt::Buffer& a, std::size_t o, const void* s, std::size_t b, std::string* e) override { ++xfers; return d->Upload(a, o, s, b, e); }
  bool Download(const rt::Buffer& a, std::size_t o, void* s, std::size_t b, std::string* e) override { ++xfers; return d->Download(a, o, s, b, e); }
  bool Fill(const rt::Buffer& a, std::uint8_t v, std::string* e) override { ++launches; return d->Fill(a, v, e); }
  bool LoadModule(std::span<const std::uint8_t> h, std::string* e) override { return d->LoadModule(h, e); }
  rt::KernelId Kernel(std::string_view n, std::string* e) override { return d->Kernel(n, e); }
  bool Launch(rt::KernelId k, rt::Dim3 g, rt::Dim3 b, std::span<const rt::Arg> a, std::string* e) override { ++launches; return d->Launch(k, g, b, a, e); }
  bool Synchronize(std::string* e) override { ++syncs; return d->Synchronize(e); }
  bool BeginCapture(std::string* e) override { return d->BeginCapture(e); }
  bool EndCapture(rt::Graph* g, std::string* e) override { return d->EndCapture(g, e); }
  bool LaunchGraph(const rt::Graph& g, std::string* e) override { ++graphs; return d->LaunchGraph(g, e); }
  void FreeGraph(rt::Graph& g) noexcept override { d->FreeGraph(g); }
};
int main(int argc, char** argv) {
  std::string err;
  const char* dir = std::getenv("HALO_TINY_HF_DIR");
  const std::string path = argc > 1 ? argv[1] : dir ? std::string(dir) + "/tiny-f32.gguf" : "";
  if (path.empty()) {
    std::fprintf(stderr, "usage: halo_step_stats <model.gguf> (or set HALO_TINY_HF_DIR)\n");
    return 2;
  }
  auto r = core::GgufReader::OpenFile(path, &err);
  if (!r) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  auto w = model::ModelWeights::Bind(*r, &err);
  const auto& t = w->ple_table;
  auto ng = model::NgramTable::Open(r->GetMappedRegions()[t.shard].file_descriptor, t.file_offset, t.rows, w->config.ple_head_dim, t.type, &err);
  for (auto be : {rt::Backend::kHip, rt::Backend::kHrx}) {
    for (bool graphs : {false, true}) {
      Counting c;
      c.d = rt::Device::Create(be, &err);
      if (!c.d) continue;
      auto m = engine::FlashNextGpu::Create(c, *w, ng.get(), nullptr, 64, &err);
      if (!m) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
      }
      m->SetUseGraphs(graphs);
      std::vector<float> logits(w->config.vocab_size);
      for (int i = 0; i < 4; ++i)  // warm up, including both graph variants
        if (!m->Step(i + 100, i % 2 ? std::span<float>(logits) : std::span<float>{}, &err)) {
          std::fprintf(stderr, "%s\n", err.c_str());
          return 1;
        }
      c.launches = c.syncs = c.xfers = c.graphs = 0;
      const int N = 40;
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < N; ++i) (void)m->Step(i + 200, i + 1 == N ? std::span<float>(logits) : std::span<float>{}, &err);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("%s %-6s: %d layers, %6.1f launches + %.1f graphs/step, %.1f syncs, %.1f transfers, %.3f ms/step\n",
                  rt::BackendName(be), graphs ? "graph" : "eager", w->config.num_layers, double(c.launches) / N,
                  double(c.graphs) / N, double(c.syncs) / N, double(c.xfers) / N, ms / N);
    }
  }
}
