// Prefix reuse across retained sessions (server/session_cache): which prefill splits leave the engine's state
// byte-identical to one cold prefill, and that the cache's reuse is exact.
//   HALO_SMALL_QUANT_DIR    holds small-q.gguf (the small quantized model: every fast kernel runs)
//   HALO_FLASH_NEXT_GGUF    shard 00001 of the real model, plus HALO_PREFIX_PROBE_TEXT (a long text file): the
//                           real-model probe (minutes; run alone under the GPU lock)

#include <hegel/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "engine/flash_next_gpu.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"
#include "server/flash_next_generator.hpp"
#include "text/tokenizer.hpp"

namespace gs = hegel::generators;
using namespace halo;

namespace {

struct Model {
  std::unique_ptr<core::GgufReader> reader;
  std::optional<model::ModelWeights> weights;
  std::unique_ptr<model::NgramTable> ngram;
  std::unique_ptr<text::Tokenizer> tok;
  std::unique_ptr<rt::Device> dev;
};

std::unique_ptr<Model> Open(const std::string& path, bool tokenizer) {
  auto m = std::make_unique<Model>();
  std::string e;
  m->reader = core::GgufReader::OpenFile(path, &e);
  if (!m->reader) return nullptr;
  m->weights = model::ModelWeights::Bind(*m->reader, &e);
  if (!m->weights) return nullptr;
  if (tokenizer && !(m->tok = text::Tokenizer::FromGguf(*m->reader, &e))) return nullptr;
  const auto& p = m->weights->ple_table;
  m->ngram = model::NgramTable::Open(m->reader->GetMappedRegions()[p.shard].file_descriptor, p.file_offset, p.rows,
                                     m->weights->config.ple_head_dim, p.type, &e);
  if (!m->ngram) return nullptr;
  m->dev = rt::Device::Create(rt::Backend::kHip, &e);
  return m->dev ? std::move(m) : nullptr;
}

bool SameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Logits after prefilling tokens[0, n) (split at `cuts`, each later piece on a fork of the session the earlier
// pieces ran on, as the cache does) and then `steps` greedy-fed steps of the given tokens.
std::vector<std::vector<float>> RunSplit(engine::FlashNextGpu& e, const std::vector<std::int32_t>& tokens, std::size_t n,
                                         const std::vector<std::size_t>& cuts, std::size_t steps, std::size_t V) {
  std::string error;
  std::vector<std::vector<float>> out;
  EXPECT_TRUE(e.BindSession(0, &error)) << error;
  EXPECT_TRUE(e.Reset(&error)) << error;
  std::vector<engine::FlashNextGpu::SessionId> made;
  std::size_t at = 0;
  std::vector<std::size_t> ends = cuts;
  ends.push_back(n);
  for (std::size_t i = 0; i < ends.size(); ++i) {
    if (i > 0) {
      engine::FlashNextGpu::SessionId f = 0;
      EXPECT_TRUE(e.ForkSession(e.BoundSession(), &f, &error)) << error;
      EXPECT_TRUE(e.BindSession(f, &error)) << error;
      made.push_back(f);
    }
    std::vector<float> l(V);
    const bool last = i + 1 == ends.size();
    EXPECT_TRUE(e.Prefill(std::span<const std::int32_t>(tokens.data() + at, ends[i] - at),
                          last ? std::span<float>(l) : std::span<float>{}, &error)) << error;
    if (last) out.push_back(std::move(l));
    at = ends[i];
  }
  for (std::size_t s = 0; s < steps; ++s) {
    std::vector<float> l(V);
    EXPECT_TRUE(e.Step(tokens[n + s], l, &error)) << error;
    out.push_back(std::move(l));
  }
  EXPECT_TRUE(e.BindSession(0, &error)) << error;
  for (auto f : made) EXPECT_TRUE(e.DropSession(f, &error)) << error;
  return out;
}

}  // namespace

// Probe (not a gate): on the real model, which split points give a state byte-identical to one cold prefill?
// Prints IDENTICAL / DIFFERS per split; asserts only that the unsplit rerun is identical (determinism).
TEST(SessionCache, RealModelSplitProbe) {
  const char* path = std::getenv("HALO_FLASH_NEXT_GGUF");
  const char* text_path = std::getenv("HALO_PREFIX_PROBE_TEXT");
  if (!path || !text_path) GTEST_SKIP() << "set HALO_FLASH_NEXT_GGUF and HALO_PREFIX_PROBE_TEXT";
  auto m = Open(path, true);
  ASSERT_TRUE(m);
  std::ifstream in(text_path);
  std::stringstream ss;
  ss << in.rdbuf();
  const auto ids = m->tok->Encode(ss.str(), false);
  const std::size_t n = std::min<std::size_t>(ids.size() - 8, std::getenv("HALO_PREFIX_PROBE_N") ? std::atoi(std::getenv("HALO_PREFIX_PROBE_N")) : 5000);
  std::vector<std::int32_t> tokens(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(n + 8));
  std::string e;
  std::vector<rt::FileSource> sources;
  for (const auto& r : m->reader->GetMappedRegions()) sources.push_back({r.data, r.size, r.file_descriptor});
  auto engine = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, 16384, &e, sources);
  ASSERT_TRUE(engine) << e;
  const std::size_t V = m->weights->config.vocab_size;
  const auto cold = RunSplit(*engine, tokens, n, {}, 4, V);
  ASSERT_TRUE(SameBits(cold[0], RunSplit(*engine, tokens, n, {}, 4, V)[0])) << "cold prefill is not deterministic";
  const char* env_cuts = std::getenv("HALO_PREFIX_PROBE_CUTS");
  std::vector<std::vector<std::size_t>> cases;
  if (env_cuts) {
    std::stringstream cs(env_cuts);
    std::string one;
    while (std::getline(cs, one, ';')) {
      std::vector<std::size_t> cuts;
      std::stringstream ps(one);
      std::string c;
      while (std::getline(ps, c, ',')) cuts.push_back(std::stoul(c));
      cases.push_back(cuts);
    }
  } else {
    cases = {{2048}, {4096}, {2048, 4096}, {1024}, {512}, {256}, {128}, {64}, {2048 + 128}, {2048 + 512}, {3072},
             {1037}, {2048 + 37}, {4000}};
  }
  for (const auto& cuts : cases) {
    const auto got = RunSplit(*engine, tokens, n, cuts, 4, V);
    bool prefill_same = SameBits(got[0], cold[0]), steps_same = true;
    double worst = 0;
    for (std::size_t s = 0; s < got.size(); ++s) {
      if (s && !SameBits(got[s], cold[s])) steps_same = false;
      double d = 0, ref = 0;
      for (std::size_t i = 0; i < V; ++i) {
        d += (got[s][i] - cold[s][i]) * (got[s][i] - cold[s][i]);
        ref += cold[s][i] * cold[s][i];
      }
      worst = std::max(worst, std::sqrt(d / ref));
    }
    std::string label;
    for (auto c : cuts) label += std::to_string(c) + " ";
    std::printf("split at %-14s prefill %s, 4 steps %s (worst relative L2 %.3g)\n", label.c_str(),
                prefill_same ? "IDENTICAL" : "DIFFERS", steps_same ? "IDENTICAL" : "DIFFERS", worst);
  }
}

namespace {

// The alignment the server's exact mode uses on the small model (SessionCacheOptions::align).
constexpr std::uint32_t kAlign = 64;

std::unique_ptr<Model> OpenSmall() {
  const char* dir = std::getenv("HALO_SMALL_QUANT_DIR");
  return dir ? Open(std::string(dir) + "/small-q.gguf", false) : nullptr;
}

}  // namespace

// Splits at multiples of kAlign leave the state byte-identical to one cold prefill (prefill logits and the steps
// after it); positive control: unaligned splits differ (the comparison can see a split).
TEST(SessionCache, SmallModelAlignedSplitsAreByteIdentical) {
  auto m = OpenSmall();
  if (!m) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  std::string e;
  auto engine = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, 1024, &e);
  ASSERT_TRUE(engine) << e;
  const std::size_t V = m->weights->config.vocab_size;
  std::uint32_t seed = 12345;
  std::vector<std::int32_t> tokens(900);
  for (auto& t : tokens) {
    seed = seed * 1664525u + 1013904223u;
    t = static_cast<std::int32_t>((seed >> 8) % V);
  }
  std::size_t unaligned_differ = 0;
  for (const std::size_t n : {300u, 517u, 880u}) {
    const auto cold = RunSplit(*engine, tokens, n, {}, 4, V);
    for (const auto& cuts : std::vector<std::vector<std::size_t>>{
             {kAlign}, {2 * kAlign}, {kAlign, 3 * kAlign}, {(n - 1) / kAlign * kAlign}, {4 * kAlign}}) {
      if (cuts.back() >= n) continue;
      const auto got = RunSplit(*engine, tokens, n, cuts, 4, V);
      for (std::size_t s = 0; s < got.size(); ++s)
        EXPECT_TRUE(SameBits(got[s], cold[s])) << n << " tokens, cut " << cuts[0] << ", output " << s;
    }
    for (const std::size_t cut : {37u, kAlign + 5, kAlign / 2 + kAlign}) {
      const auto got = RunSplit(*engine, tokens, n, {cut}, 4, V);
      bool same = true;
      for (std::size_t s = 0; s < got.size(); ++s) same = same && SameBits(got[s], cold[s]);
      unaligned_differ += !same;
    }
  }
  std::printf("unaligned splits that differ: %zu of 9\n", unaligned_differ);
  EXPECT_GT(unaligned_differ, 0u) << "positive control: no unaligned split differed, so the check cannot see a split";
}

// The cache's exactness contract, through the server's generator: random agent conversations whose next prompt
// keeps the previous prompt up to its generation prompt and then either echoes what was generated (opencode,
// pi) or renders that turn differently (Hermes drops reasoning), with unrelated requests interleaved
// (opencode's title request). Every request's tokens (greedy, or sampled with a seed: a sampled token moves with
// any change in the logits) from the exact-mode cache equal a cold run's (a second engine, cache off, every
// state from one prefill); and the cache did reuse (else the comparison proves nothing). Fast mode runs the
// same conversations on a third engine and reports how often it differs from cold (the documented tradeoff).
TEST(SessionCache, ReuseIsByteIdenticalToColdPrefill) {
  auto m = OpenSmall();
  if (!m) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  constexpr std::uint32_t kCtx = 2048;
  std::string e;
  auto ea = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  auto eb = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  auto ec = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  ASSERT_TRUE(ea && eb && ec) << e;
  const auto V = static_cast<std::int32_t>(m->weights->config.vocab_size);
  std::size_t requests = 0, reused = 0, prompt_tokens = 0, fast_differs = 0, fast_reused = 0;
  hegel::test([&](hegel::TestCase& tc) {
    // Positive control (manual): HALO_TEST_CACHE_ALIGN=1 parks checkpoints anywhere, which must fail here.
    const char* align_env = std::getenv("HALO_TEST_CACHE_ALIGN");
    const std::uint32_t align = align_env ? static_cast<std::uint32_t>(std::atoi(align_env)) : kAlign;
    server::SessionCacheOptions exact{.mode = server::SessionCacheOptions::Mode::kExact, .align = align,
                                      .max_retained = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 4}))};
    server::FlashNextGenerator a(*ea, std::vector<text::TokenId>{}, kCtx, exact);
    server::FlashNextGenerator b(*eb, std::vector<text::TokenId>{}, kCtx, {.mode = server::SessionCacheOptions::Mode::kOff});
    server::FlashNextGenerator c(*ec, std::vector<text::TokenId>{}, kCtx, {.mode = server::SessionCacheOptions::Mode::kFast});
    const auto tokens = [&](std::uint32_t lo, std::uint32_t hi) {
      std::vector<std::int32_t> v(tc.draw(gs::integers<std::uint32_t>({.min_value = lo, .max_value = hi})));
      for (auto& t : v) t = tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = V - 1}));
      return v;
    };
    const std::vector<std::int32_t> gen_prompt = {7, 11, 13};  // the "<|im_start|>assistant\n<think>\n" stand-in
    std::vector<std::int32_t> prompt = tokens(20, 300);
    std::size_t stable = prompt.size();
    prompt.insert(prompt.end(), gen_prompt.begin(), gen_prompt.end());
    const int turns = tc.draw(gs::integers<int>({.min_value = 2, .max_value = 5}));
    for (int turn = 0; turn < turns && prompt.size() + 40 < kCtx; ++turn) {
      server::TextGenerator::Params p;
      p.max_tokens = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 12}));
      if (tc.draw(gs::booleans())) {
        p.sampling.temperature = 1.0F;
        p.sampling.top_k = 0;
        p.sampling.top_p = 1.0F;
        p.sampling.seed = tc.draw(gs::integers<std::int64_t>({.min_value = 0, .max_value = 1 << 20}));
      }
      std::vector<std::int32_t> got, want, fast;
      const auto collect = [](std::vector<std::int32_t>* out) {
        return [out](text::TokenId t) { return out->push_back(static_cast<std::int32_t>(t)), true; };
      };
      const auto ra = a.GenerateIds(prompt, stable, p, collect(&got));
      const auto rb = b.GenerateIds(prompt, stable, p, collect(&want));
      const auto rc = c.GenerateIds(prompt, stable, p, collect(&fast));
      ASSERT_TRUE(ra.error.empty() && rb.error.empty() && rc.error.empty()) << ra.error << rb.error << rc.error;
      ASSERT_EQ(got, want) << "turn " << turn << ", prompt " << prompt.size() << ", reused " << ra.cached_tokens;
      EXPECT_EQ(rb.cached_tokens, 0u);
      ++requests;
      reused += ra.cached_tokens;
      prompt_tokens += prompt.size();
      fast_differs += fast != want;
      fast_reused += rc.cached_tokens;
      // An unrelated request in between (title request): must not cost the conversation its checkpoints.
      if (tc.draw(gs::booleans())) {
        auto other = tokens(10, 120);
        std::vector<std::int32_t> oa, ob, oc;
        server::TextGenerator::Params q;
        q.max_tokens = 4;
        ASSERT_TRUE(a.GenerateIds(other, other.size() - 1, q, collect(&oa)).error.empty());
        ASSERT_TRUE(b.GenerateIds(other, other.size() - 1, q, collect(&ob)).error.empty());
        ASSERT_TRUE(c.GenerateIds(other, other.size() - 1, q, collect(&oc)).error.empty());
        ASSERT_EQ(oa, ob);
      }
      // The next prompt: this one up to its generation prompt, then this turn as the client renders it.
      std::vector<std::int32_t> next(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(stable));
      next.insert(next.end(), gen_prompt.begin(), gen_prompt.end());
      if (tc.draw(gs::booleans())) {
        next.insert(next.end(), got.begin(), got.end());  // echoed as generated
      } else {
        const auto other = tokens(1, 20);  // rendered differently (reasoning dropped)
        next.insert(next.end(), other.begin(), other.end());
      }
      const auto tool = tokens(1, 150);  // the tool result
      next.insert(next.end(), tool.begin(), tool.end());
      stable = next.size();
      next.insert(next.end(), gen_prompt.begin(), gen_prompt.end());
      prompt = std::move(next);
    }
  }, {.test_cases = 40});
  std::printf("exact: %zu requests, %zu of %zu prompt tokens reused; fast: %zu reused, %zu requests differ from cold\n",
              requests, reused, prompt_tokens, fast_reused, fast_differs);
  EXPECT_GT(reused, prompt_tokens / 10) << "the cache hardly reused anything: the comparison is close to vacuous";
}
