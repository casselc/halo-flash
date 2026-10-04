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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/gguf.hpp"
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

}  // namespace

int main(int argc, char** argv) {
  std::string model, prompt = "The capital of France is", device = "hip";
  int n = 16;
  bool eager = false;
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
  const auto ts = std::chrono::steady_clock::now();
  double worst = 0;
  for (std::size_t step = 0; step < ids.size() + static_cast<std::size_t>(n) - 1; ++step) {
    const auto token = static_cast<std::int32_t>(step < ids.size() ? ids[step] : generated.back());
    const bool want = step + 1 >= ids.size();
    if (cpu && !cpu->Step(token, want ? std::span<float>(lc) : std::span<float>{}, {}, &error))
      return std::fprintf(stderr, "cpu: %s\n", error.c_str()), 1;
    const auto st = std::chrono::steady_clock::now();
    if (gpu && !gpu->Step(token, want ? std::span<float>(lg) : std::span<float>{}, &error))
      return std::fprintf(stderr, "gpu: %s\n", error.c_str()), 1;
    if (step < 2 || step + 1 == ids.size())
      std::printf("step %zu: %.3f s\n", step, std::chrono::duration<double>(std::chrono::steady_clock::now() - st).count());
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
  std::printf("text: %s%s\n", prompt.c_str(), tok->Decode(generated).c_str());
  std::printf("%zu steps in %.2f s (%.2f steps/s)%s\n", ids.size() + generated.size() - 1, secs,
              (ids.size() + generated.size() - 1) / secs,
              cpu && gpu ? (" worst relL2 " + std::to_string(worst)).c_str() : "");
  std::printf("total %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}
