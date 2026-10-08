// Prefix reuse across retained sessions (server/session_cache): which prefill splits leave the engine's state
// byte-identical to one cold prefill, and that the cache's reuse is exact.
//   HALO_SMALL_QUANT_DIR    holds small-q.gguf (the small quantized model: every fast kernel runs)
//   HALO_FLASH_NEXT_GGUF    shard 00001 of the real model, plus HALO_PREFIX_PROBE_TEXT (a long text file): the
//                           real-model probe (minutes; run alone under the GPU lock)

#include <hegel/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/gguf.hpp"
#include "engine/flash_next_gpu.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"
#include "server/flash_next_generator.hpp"
#include "server/spill_keys.hpp"
#include "server/yard_keys.hpp"
#include "../server/yard_test_util.hpp"
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

// SR4: the exact cache with a disk tier. Eviction spills (max_retained 1-2), and between turns an admin spills
// every resident entry, restores some, or flips a bit in a spilled file. Every request's tokens equal a cache-off
// engine's (exact mode's contract survives spill and restore), a damaged file only costs a re-prefill, and spilled
// states are actually restored and reused (non-vacuous).
TEST(SessionCache, SpilledStatesServeByteIdenticalToColdPrefill) {
  auto m = OpenSmall();
  if (!m) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  const char* d = std::getenv("HALO_SPILL_TEST_DIR");
  const std::string dir = (d && *d ? std::filesystem::path(d) : std::filesystem::temp_directory_path() / "halo-spill-test") /
                          ("cache-" + std::to_string(::getpid()));
  constexpr std::uint32_t kCtx = 2048;
  std::string e;
  auto ea = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  auto eb = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  ASSERT_TRUE(ea && eb) << e;
  const auto V = static_cast<std::int32_t>(m->weights->config.vocab_size);
  std::size_t restored = 0, restore_failed = 0, spilled = 0, reused = 0, corrupted = 0, rotations = 0;
  hegel::test([&](hegel::TestCase& tc) {
    server::SessionCacheOptions exact{.mode = server::SessionCacheOptions::Mode::kExact, .align = kAlign,
                                      .max_retained = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 2}))};
    exact.spill_dir = dir;
    exact.spill_on_evict = true;
    // Encrypted spills (v2) under a master provider; admin 4 rotates its epoch (crypto-shredding).
    core::crypt::SecretBytes master(32);
    ASSERT_TRUE(core::crypt::RandomBytes({master.data(), 32}));
    std::string ke;
    auto keys = server::MasterSpillKeys::Create("test", std::move(master), "", &ke);
    ASSERT_TRUE(keys) << ke;
    exact.spill_keys = keys.get();
    exact.max_spilled = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 6}));
    server::FlashNextGenerator a(*ea, std::vector<text::TokenId>{}, kCtx, exact);
    server::FlashNextGenerator b(*eb, std::vector<text::TokenId>{}, kCtx, {.mode = server::SessionCacheOptions::Mode::kOff});
    const auto tokens = [&](std::uint32_t lo, std::uint32_t hi) {
      std::vector<std::int32_t> v(tc.draw(gs::integers<std::uint32_t>({.min_value = lo, .max_value = hi})));
      for (auto& t : v) t = tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = V - 1}));
      return v;
    };
    const auto collect = [](std::vector<std::int32_t>* out) {
      return [out](text::TokenId t) { return out->push_back(static_cast<std::int32_t>(t)), true; };
    };
    const std::vector<std::int32_t> gen_prompt = {7, 11, 13};
    std::set<std::uint64_t> damaged;  // entries whose file had a bit flipped
    std::vector<std::int32_t> prompt = tokens(20, 300);
    std::size_t stable = prompt.size();
    prompt.insert(prompt.end(), gen_prompt.begin(), gen_prompt.end());
    const int turns = tc.draw(gs::integers<int>({.min_value = 2, .max_value = 5}));
    for (int turn = 0; turn < turns && prompt.size() + 40 < kCtx; ++turn) {
      server::TextGenerator::Params p;
      p.max_tokens = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 8}));
      std::vector<std::int32_t> got, want;
      const auto ra = a.GenerateIds(prompt, stable, p, collect(&got));
      const auto rb = b.GenerateIds(prompt, stable, p, collect(&want));
      ASSERT_TRUE(ra.error.empty() && rb.error.empty()) << ra.error << rb.error;
      ASSERT_EQ(got, want) << "turn " << turn << ", prompt " << prompt.size() << ", reused " << ra.cached_tokens;
      reused += ra.cached_tokens;
      // Unrelated requests push the conversation's checkpoint out (spilled at max_retained 1-2).
      for (int k = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 2})); k > 0; --k) {
        auto other = tokens(10, 120);
        std::vector<std::int32_t> oa, ob;
        server::TextGenerator::Params q;
        q.max_tokens = 3;
        ASSERT_TRUE(a.GenerateIds(other, other.size() - 1, q, collect(&oa)).error.empty());
        ASSERT_TRUE(b.GenerateIds(other, other.size() - 1, q, collect(&ob)).error.empty());
        ASSERT_EQ(oa, ob);
      }
      // Admin between turns.
      const int admin = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4}));
      a.WithCache([&](server::SessionCache& c) {
        std::string why;
        // Every file in the spill directory is encrypted (v2) (checked before this turn's bit flips; a file
        // flipped earlier has been restored-and-refused or still holds a flip outside the magic's byte).
        for (const auto& f : std::filesystem::directory_iterator(dir)) {
          const std::string name = f.path().filename().string();
          if (damaged.contains(std::strtoull(name.c_str() + name.find('-') + 1, nullptr, 10))) continue;
          char magic[8] = {};
          std::ifstream(f.path(), std::ios::binary).read(magic, 8);
          EXPECT_EQ(std::string(magic, 8), "HALOSPL2") << f.path();
        }
        for (const auto& en : c.List()) {
          if (admin == 1 && !en.spilled) EXPECT_TRUE(c.Spill(en.key, &why)) << why;
          if (admin == 2 && en.spilled) {
            const bool ok = c.Restore(en.key, &why);
            EXPECT_TRUE(ok || damaged.contains(en.key)) << "an intact spilled entry did not restore: " << why;
          }
        }
        if (admin == 3)
          for (const auto& f : std::filesystem::directory_iterator(dir)) {
            std::fstream io(f.path(), std::ios::in | std::ios::out | std::ios::binary);
            const auto size = std::filesystem::file_size(f.path());
            const auto at = static_cast<std::streamoff>(tc.draw(gs::integers<std::uint64_t>({.min_value = 0, .max_value = size - 1})));
            char ch;
            io.seekg(at);
            io.get(ch);
            io.seekp(at);
            io.put(static_cast<char>(ch ^ 0x20));
            ++corrupted;
            const std::string name = f.path().filename().string();  // s<pid>-<key>.spill
            damaged.insert(std::strtoull(name.c_str() + name.find('-') + 1, nullptr, 10));
          }
        if (admin == 4) {  // a new epoch: every spilled entry is unreadable, erased, its file deleted
          std::size_t had = 0;
          for (const auto& en : c.List()) had += en.spilled;
          EXPECT_TRUE(keys->FenceEpoch(&why));
          EXPECT_EQ(c.DropUnreadable(), had);
          for (const auto& en : c.List()) EXPECT_FALSE(en.spilled);
          ++rotations;
        }
        return true;
      });
      std::vector<std::int32_t> next(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(stable));
      next.insert(next.end(), gen_prompt.begin(), gen_prompt.end());
      const auto other = tokens(1, 20);
      next.insert(next.end(), other.begin(), other.end());
      const auto tool = tokens(1, 150);
      next.insert(next.end(), tool.begin(), tool.end());
      stable = next.size();
      next.insert(next.end(), gen_prompt.begin(), gen_prompt.end());
      prompt = std::move(next);
    }
    const auto& k = a.cache().spill_counters();
    restored += k.restored;
    restore_failed += k.restore_failed;
    spilled += k.spilled;
  }, {.test_cases = 40, .phases = {hegel::Phase::Explicit, hegel::Phase::Reuse, hegel::Phase::Generate}});
  std::filesystem::remove_all(dir);
  std::printf("spill cache: %zu spilled, %zu restored, %zu restores failed (%zu bit flips, %zu rotations), %zu prompt "
              "tokens reused\n", spilled, restored, restore_failed, corrupted, rotations, reused);
  EXPECT_GT(restored, 5u) << "spilled states were hardly ever restored: the comparison is close to vacuous";
}

namespace {
// A provider whose seal and open keys can be withheld (a yard session without a registered key, a failing TPM).
struct GateKeys final : server::SpillKeys {
  explicit GateKeys(server::MasterSpillKeys& k) : inner(k) {}
  server::MasterSpillKeys& inner;
  bool seal = true, open = true;
  std::string Describe() const override { return "gate:" + inner.Describe(); }
  bool SealLease(std::uint64_t entry, std::string_view label, bool bound, server::SpillLease* out,
                 std::string* why) override {
    if (!seal) return *why = "key withheld", false;
    return inner.SealLease(entry, label, bound, out, why);
  }
  bool OpenLease(std::string_view label, std::string_view id, server::SpillLease* out, std::string* why) override {
    if (!open) return *why = "key withheld", false;
    return inner.OpenLease(label, id, out, why);
  }
  bool MayOpen(std::string_view label, std::string_view id) override { return inner.MayOpen(label, id); }
};
std::size_t FilesIn(const std::string& dir) {
  std::size_t n = 0;
  std::error_code ec;
  for (const auto& f : std::filesystem::directory_iterator(dir, ec)) n += f.is_regular_file();
  return n;
}
}  // namespace

// Spill keys in the cache (docs/SESSIONS.md section 10): a provider without a key refuses the spill and nothing is
// written (never a plaintext fallback; eviction then drops); no provider refuses unless plaintext was chosen; a
// plaintext v1 file restores only with spill_accept_v1; an entry whose open key is withheld stays spilled and
// untouched until the key is there; a rotated epoch makes spilled files unreadable (DropUnreadable erases them and
// the next request re-prefills, still equal to a cold engine's tokens).
TEST(SessionCache, SpillKeysRefuseKeepAndShred) {
  auto m = OpenSmall();
  if (!m) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  const char* d = std::getenv("HALO_SPILL_TEST_DIR");
  const std::string dir = (d && *d ? std::filesystem::path(d) : std::filesystem::temp_directory_path() / "halo-spill-test") /
                          ("keys-" + std::to_string(::getpid()));
  constexpr std::uint32_t kCtx = 1024;
  std::string e;
  auto ea = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  auto eb = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  ASSERT_TRUE(ea && eb) << e;
  core::crypt::SecretBytes master(32);
  ASSERT_TRUE(core::crypt::RandomBytes({master.data(), 32}));
  auto master_keys = server::MasterSpillKeys::Create("test", std::move(master), "", &e);
  ASSERT_TRUE(master_keys) << e;
  GateKeys gate(*master_keys);
  std::vector<std::int32_t> prompt(200);
  for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = static_cast<std::int32_t>((i * 7919 + 3) % 1000);
  const auto collect = [](std::vector<std::int32_t>* out) {
    return [out](text::TokenId t) { return out->push_back(static_cast<std::int32_t>(t)), true; };
  };
  server::FlashNextGenerator cold(*eb, std::vector<text::TokenId>{}, kCtx, {.mode = server::SessionCacheOptions::Mode::kOff});
  server::TextGenerator::Params p;
  p.max_tokens = 4;
  std::vector<std::int32_t> want;
  ASSERT_TRUE(cold.GenerateIds(prompt, 190, p, collect(&want)).error.empty());
  // One request parks a checkpoint (at 128: align 64); returns its key.
  const auto run = [&](server::FlashNextGenerator& g, std::size_t* reused = nullptr) {
    std::vector<std::int32_t> got;
    const auto r = g.GenerateIds(prompt, 190, p, collect(&got));
    EXPECT_TRUE(r.error.empty()) << r.error;
    EXPECT_EQ(got, want) << "tokens differ from a cold engine's";
    if (reused) *reused = r.cached_tokens;
  };
  const auto first_key = [](server::SessionCache& c) {
    const auto l = c.List();
    return l.empty() ? 0 : l.front().key;
  };
  for (int mode = 0; mode < 5; ++mode) {
    std::filesystem::remove_all(dir);
    server::SessionCacheOptions o{.mode = server::SessionCacheOptions::Mode::kExact, .align = 64, .max_retained = 4};
    o.spill_dir = dir;
    gate.seal = gate.open = true;
    if (mode == 0) o.spill_keys = &gate;            // provider withholds the seal key
    if (mode == 1) o.spill_keys = nullptr;          // no provider, plaintext not allowed
    if (mode == 2 || mode == 3) o.spill_plaintext_ok = true, o.spill_accept_v1 = mode == 3;  // plaintext v1
    if (mode == 4) o.spill_keys = &gate;            // encrypted; open key withheld, then granted, then rotated
    server::FlashNextGenerator g(*ea, std::vector<text::TokenId>{}, kCtx, o);
    run(g);
    g.WithCache([&](server::SessionCache& c) {
      const auto key = first_key(c);
      EXPECT_NE(key, 0u);
      std::string why;
      bool kept = false;
      if (mode == 0) {
        gate.seal = false;
        EXPECT_FALSE(c.Spill(key, &why)) << "spilled without a key";
        EXPECT_EQ(FilesIn(dir), 0u) << "a file was written without a key (plaintext fallback?)";
        EXPECT_EQ(c.spill_counters().no_key, 1u);
        EXPECT_FALSE(c.List().front().spilled);
      } else if (mode == 1) {
        EXPECT_FALSE(c.Spill(key, &why));
        EXPECT_EQ(FilesIn(dir), 0u);
      } else if (mode == 2 || mode == 3) {
        EXPECT_TRUE(c.Spill(key, &why)) << why;
        EXPECT_EQ(c.List().front().key_id, "");
        EXPECT_EQ(c.Restore(key, &why), mode == 3) << why;  // v1 only with accept_v1; refused: entry erased
        EXPECT_EQ(c.List().size(), mode == 3 ? 1u : 0u);
      } else {
        EXPECT_TRUE(c.Spill(key, &why)) << why;
        EXPECT_NE(c.List().front().key_id, "");
        gate.open = false;
        EXPECT_FALSE(c.Restore(key, &why, &kept));
        EXPECT_TRUE(kept) << "an entry waiting for its key was dropped";
        EXPECT_TRUE(c.List().front().spilled);
        EXPECT_EQ(FilesIn(dir), 1u);
        EXPECT_EQ(c.DropUnreadable(), 0u) << "a withheld key is not a rotated one";
        gate.open = true;
        EXPECT_TRUE(c.Restore(key, &why, &kept)) << why;
        EXPECT_TRUE(c.Spill(key, &why)) << why;
        EXPECT_TRUE(master_keys->FenceEpoch(&why));
        EXPECT_FALSE(c.Restore(key, &why, &kept)) << "a file of the previous epoch restored";
        EXPECT_FALSE(kept);
        EXPECT_EQ(c.List().size(), 0u);
        EXPECT_EQ(FilesIn(dir), 0u);
      }
      return true;
    });
    // Whatever happened, the next request's tokens equal a cold engine's.
    std::size_t reused = 0;
    run(g, &reused);
    if (mode == 3) EXPECT_GT(reused, 0u) << "the restored v1 entry was not reused";
  }
  // Eviction with a provider that has no key: the entry is dropped, nothing is written.
  {
    std::filesystem::remove_all(dir);
    server::SessionCacheOptions o{.mode = server::SessionCacheOptions::Mode::kExact, .align = 64, .max_retained = 1};
    o.spill_dir = dir;
    o.spill_on_evict = true;
    o.spill_keys = &gate;
    gate.seal = false;
    server::FlashNextGenerator g(*ea, std::vector<text::TokenId>{}, kCtx, o);
    run(g);
    std::vector<std::int32_t> other(100, 5), got;
    server::TextGenerator::Params q;
    q.max_tokens = 2;
    ASSERT_TRUE(g.GenerateIds(other, 99, q, collect(&got)).error.empty());
    EXPECT_EQ(FilesIn(dir), 0u) << "eviction wrote a file without a key";
    g.WithCache([&](server::SessionCache& c) {
      EXPECT_GE(c.spill_counters().no_key, 1u);
      return true;
    });
  }
  std::filesystem::remove_all(dir);
}

namespace {

// GTT used (amdgpu sysfs) and the host's MemAvailable, in bytes (0 when unreadable).
std::pair<std::size_t, std::size_t> GttUsedAndMemAvailable() {
  std::size_t used = 0, avail = 0;
  for (int card = 0; card < 8 && !used; ++card) {
    std::ifstream u("/sys/class/drm/card" + std::to_string(card) + "/device/mem_info_gtt_used");
    if (!(u >> used)) used = 0;
  }
  std::ifstream m("/proc/meminfo");
  std::string key, unit;
  std::size_t kb = 0;
  while (m >> key >> kb >> unit)
    if (key == "MemAvailable:") {
      avail = kb * 1024;
      break;
    }
  return {used, avail};
}

}  // namespace

// Probe (not a gate; HALO_PAGE_PROBE=1): does a released VMM page stay charged to GTT (docs/SESSIONS.md "Released
// pages stay charged"), so that rt::Device::AvailableBytes cannot see what an eviction frees? Prints the readings
// after creating and mapping pages, after unmapping and releasing them, and after creating them again; then the
// same through the engine (a session prefilled, dropped, and a new one prefilled).
TEST(SessionCache, ReleasedPageChargeProbe) {
  if (!std::getenv("HALO_PAGE_PROBE")) GTEST_SKIP() << "set HALO_PAGE_PROBE=1 (and HALO_SMALL_QUANT_DIR)";
  std::string e;
  auto dev = rt::Device::Create(rt::Backend::kHip, &e);
  ASSERT_TRUE(dev) << e;
  const std::size_t pb = dev->PageBytes();
  ASSERT_GT(pb, 0u);
  const std::size_t n = (std::size_t{4} << 30) / pb;
  const auto show = [&](const char* what) {
    const auto [used, mem] = GttUsedAndMemAvailable();
    std::printf("%-34s AvailableBytes %7.3f GB  GTT used %7.3f GB  MemAvailable %7.3f GB\n", what,
                dev->AvailableBytes() / 1e9, used / 1e9, mem / 1e9);
  };
  rt::Buffer range = dev->ReserveRange(n * pb, &e);
  ASSERT_TRUE(range.device) << e;
  std::vector<void*> pages(n);
  const auto create = [&] {
    for (std::size_t i = 0; i < n; ++i) {
      pages[i] = dev->CreatePage(&e);
      ASSERT_TRUE(pages[i]) << e;
      ASSERT_TRUE(dev->MapPage(range, i * pb, pages[i], &e)) << e;
      ASSERT_TRUE(dev->Fill(rt::Buffer{static_cast<char*>(range.device) + i * pb, pb, nullptr}, 1, &e)) << e;
    }
    ASSERT_TRUE(dev->Synchronize(&e)) << e;
  };
  const auto release = [&] {
    for (std::size_t i = 0; i < n; ++i) {
      ASSERT_TRUE(dev->UnmapPage(range, i * pb, &e)) << e;
      dev->ReleasePage(pages[i]);
    }
    ASSERT_TRUE(dev->Synchronize(&e)) << e;
  };
  std::printf("page %zu KB, %zu pages (%.2f GB)\n", pb >> 10, n, n * pb / 1e9);
  show("start");
  create();
  show("created + mapped + filled");
  release();
  show("unmapped + released");
  std::this_thread::sleep_for(std::chrono::seconds(2));
  show("2 s later");
  create();
  show("created again");
  release();
  show("released again");
  dev->ReleaseRange(range);
  show("range released");

  auto m = OpenSmall();
  if (!m) GTEST_SKIP() << "engine half: set HALO_SMALL_QUANT_DIR";
  constexpr std::uint32_t kCtx = 131072;
  auto engine = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  ASSERT_TRUE(engine) << e;
  const auto V = static_cast<std::int32_t>(m->weights->config.vocab_size);
  std::vector<std::int32_t> tokens(100000);
  for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = static_cast<std::int32_t>((i * 2654435761u) % V);
  const auto eshow = [&](const char* what) {
    const auto [used, mem] = GttUsedAndMemAvailable();
    std::printf("engine: %-26s AvailableBytes %7.3f GB  GTT used %7.3f GB  MemAvailable %7.3f GB\n", what,
                m->dev->AvailableBytes() / 1e9, used / 1e9, mem / 1e9);
  };
  std::printf("engine: KV %.1f KB a token\n", engine->KvBytesPerToken() / 1e3);
  eshow("start");
  for (int round = 0; round < 2; ++round) {
    engine::FlashNextGpu::SessionId s = 0;
    ASSERT_TRUE(engine->NewSession(&s, &e)) << e;
    ASSERT_TRUE(engine->BindSession(s, &e)) << e;
    ASSERT_TRUE(engine->Prefill(tokens, {}, &e)) << e;
    const auto b = engine->SessionMemory(s);
    std::printf("engine: session %u holds %.3f GB of KV pages\n", s, b.kv_mapped / 1e9);
    eshow(round ? "second session prefilled" : "first session prefilled");
    ASSERT_TRUE(engine->BindSession(0, &e)) << e;
    ASSERT_TRUE(engine->DropSession(s, &e)) << e;
    eshow(round ? "second session dropped" : "first session dropped");
  }
}

// ---- Serving under departures and memory pressure (audit streams/A/AUDIT.md F3, F5) -------------------------------

namespace {

/// Prefill chunks of `tokens` (and no long-prompt chunks) for the scope, restoring the defaults: prompts on the
/// small model then have several chunks, so a cancellation can land between them.
struct PrefillChunk {
  explicit PrefillChunk(std::uint32_t tokens)
      : saved(engine::FlashNextGpu::prefill_chunk_), saved_long(engine::FlashNextGpu::prefill_long_chunk_) {
    engine::FlashNextGpu::prefill_chunk_ = tokens;
    engine::FlashNextGpu::prefill_long_chunk_ = 0;
  }
  ~PrefillChunk() {
    engine::FlashNextGpu::prefill_chunk_ = saved;
    engine::FlashNextGpu::prefill_long_chunk_ = saved_long;
  }
  std::uint32_t saved, saved_long;
};

/// Params whose `cancelled` turns true at its `at`-th poll (and stays true); `polls` counts them.
struct CancelAt {
  std::size_t at = SIZE_MAX, polls = 0;
  server::TextGenerator::Params Params(server::TextGenerator::Params p) {
    p.cancelled = [this] { return ++polls >= at; };
    return p;
  }
};

}  // namespace

// Requests cancelled at any point (before the prefill, between its chunks, between decode steps) leave the cache as
// cold replays would. Exact mode: every later request's tokens equal a cold engine's, and a cancelled request's
// tokens are a prefix of its full run. Fast mode: a request stopped after k decode steps leaves the same state as
// the same request with max_tokens = k (a second fast-mode cache runs that instead), and one cancelled before its
// prefill the same as no request at all; a stop with no token emitted is ambiguous (before the first step, or
// mid-prefill), so the fast comparison ends there (an unknown is not treated as either).
TEST(SessionCache, CancelledRequestsLeaveStateEqualToAColdReplay) {
  auto m = OpenSmall();
  if (!m) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  constexpr std::uint32_t kCtx = 4096;
  PrefillChunk chunk(256);
  std::string e;
  std::vector<std::unique_ptr<engine::FlashNextGpu>> eng;
  for (int i = 0; i < 4; ++i) {
    eng.push_back(engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e));
    ASSERT_TRUE(eng.back()) << e;
  }
  const auto V = static_cast<std::int32_t>(m->weights->config.vocab_size);
  std::size_t cancels = 0, decode_cancels = 0, fast_checked = 0, before_prefill = 0;
  hegel::test([&](hegel::TestCase& tc) {
    using Mode = server::SessionCacheOptions::Mode;
    server::FlashNextGenerator a(*eng[0], std::vector<text::TokenId>{}, kCtx, {.mode = Mode::kExact, .align = kAlign});
    server::FlashNextGenerator b(*eng[1], std::vector<text::TokenId>{}, kCtx, {.mode = Mode::kOff});
    server::FlashNextGenerator c(*eng[2], std::vector<text::TokenId>{}, kCtx, {.mode = Mode::kFast});
    server::FlashNextGenerator d(*eng[3], std::vector<text::TokenId>{}, kCtx, {.mode = Mode::kFast});
    bool fast_comparable = true;
    const auto tokens = [&](std::uint32_t lo, std::uint32_t hi) {
      std::vector<std::int32_t> v(tc.draw(gs::integers<std::uint32_t>({.min_value = lo, .max_value = hi})));
      for (auto& t : v) t = tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = V - 1}));
      return v;
    };
    const auto collect = [](std::vector<std::int32_t>* out) {
      return [out](text::TokenId t) { return out->push_back(static_cast<std::int32_t>(t)), true; };
    };
    const std::vector<std::int32_t> gen_prompt = {7, 11, 13};
    std::vector<std::int32_t> prompt = tokens(200, 900);
    std::size_t stable = prompt.size();
    prompt.insert(prompt.end(), gen_prompt.begin(), gen_prompt.end());
    const int turns = tc.draw(gs::integers<int>({.min_value = 2, .max_value = 5}));
    for (int turn = 0; turn < turns && prompt.size() + 40 < kCtx; ++turn) {
      server::TextGenerator::Params p;
      p.max_tokens = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 10}));
      // The poll at which the client leaves: 1 is before the prefill; later ones fall between prefill chunks or
      // decode steps (how many of each depends on the reuse); none: it stays.
      const bool leaves = tc.draw(gs::booleans());
      const std::size_t at = leaves ? tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 14})) : SIZE_MAX;
      CancelAt ca{at}, cc{at};
      std::vector<std::int32_t> got, want, fast, ref;
      const auto ra = a.GenerateIds(prompt, stable, ca.Params(p), collect(&got));
      const auto rb = b.GenerateIds(prompt, stable, p, collect(&want));
      ASSERT_TRUE(ra.error.empty() && rb.error.empty()) << ra.error << rb.error;
      ASSERT_FALSE(rb.cancelled);
      if (ra.cancelled) {
        ++cancels;
        ASSERT_LE(got.size(), want.size());
        ASSERT_TRUE(std::equal(got.begin(), got.end(), want.begin())) << "a cancelled request's tokens";
        ASSERT_FALSE(ra.length_limited);
      } else {
        ASSERT_EQ(got, want) << "turn " << turn << ", prompt " << prompt.size() << ", reused " << ra.cached_tokens;
      }
      const auto rc = c.GenerateIds(prompt, stable, cc.Params(p), collect(&fast));
      ASSERT_TRUE(rc.error.empty()) << rc.error;
      if (fast_comparable) {
        if (!rc.cancelled) {
          ASSERT_TRUE(d.GenerateIds(prompt, stable, p, collect(&ref)).error.empty());
          ASSERT_EQ(fast, ref) << "fast mode after a cancel differs from the same history without it";
          ++fast_checked;
        } else if (cc.polls == 1) {
          ++before_prefill;  // cancelled before its prefill: as if it never ran (d runs nothing)
        } else if (!fast.empty()) {
          ++decode_cancels;
          server::TextGenerator::Params q = p;
          q.max_tokens = static_cast<std::uint32_t>(fast.size());
          ASSERT_TRUE(d.GenerateIds(prompt, stable, q, collect(&ref)).error.empty());
          ASSERT_EQ(fast, ref) << "a stop after " << fast.size() << " tokens";
        } else {
          fast_comparable = false;
        }
      }
      // The next prompt, as in ReuseIsByteIdenticalToColdPrefill (echoing the full run's tokens).
      std::vector<std::int32_t> next(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(stable));
      next.insert(next.end(), gen_prompt.begin(), gen_prompt.end());
      if (tc.draw(gs::booleans())) next.insert(next.end(), want.begin(), want.end());
      else {
        const auto other = tokens(1, 20);
        next.insert(next.end(), other.begin(), other.end());
      }
      const auto tool = tokens(1, 300);
      next.insert(next.end(), tool.begin(), tool.end());
      stable = next.size();
      next.insert(next.end(), gen_prompt.begin(), gen_prompt.end());
      prompt = std::move(next);
    }
  }, {.test_cases = 30});
  std::printf("cancelled %zu requests (fast: %zu after decode steps, %zu before the prefill); %zu fast requests compared\n",
              cancels, decode_cancels, before_prefill, fast_checked);
  EXPECT_GT(cancels, 5u) << "too few cancellations to mean anything";
  EXPECT_GT(decode_cancels, 0u);
}

// A client that leaves mid-prefill stops it at the next chunk boundary: no further chunk and no decode step runs.
// (CancelAt counts the polls: the first is before the prefill, then one before each later chunk.)
TEST(SessionCache, ACancelStopsThePrefillAtTheNextChunk) {
  auto m = OpenSmall();
  if (!m) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  constexpr std::uint32_t kCtx = 4096;
  PrefillChunk chunk(256);
  std::string e;
  auto engine = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  ASSERT_TRUE(engine) << e;
  for (const auto mode : {server::SessionCacheOptions::Mode::kExact, server::SessionCacheOptions::Mode::kFast}) {
    server::FlashNextGenerator g(*engine, std::vector<text::TokenId>{}, kCtx, {.mode = mode});
    std::vector<std::int32_t> ids(2000);  // 8 chunks
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<std::int32_t>((i * 7919) % 1000);
    server::TextGenerator::Params p;
    p.max_tokens = 4;
    for (const std::size_t at : {1u, 2u, 4u, 7u}) {
      CancelAt ca{at};
      std::vector<std::int32_t> got;
      const auto r = g.GenerateIds(ids, ids.size() - 1, ca.Params(p), [&](text::TokenId t) { return got.push_back(t), true; });
      EXPECT_TRUE(r.cancelled) << "at " << at;
      EXPECT_TRUE(r.error.empty()) << r.error;
      EXPECT_TRUE(got.empty()) << "decoded after a cancel at poll " << at;
      EXPECT_EQ(ca.polls, at) << "polled on after the cancel";
      EXPECT_EQ(g.cache().retained(), 0u) << "a partly prefilled state was retained";
    }
  }
}

// A queued request whose client leaves never runs: it gives up while waiting for the generator (the other request
// is held mid-decode), so its compute and the queue slot go at once; the held request then completes unaffected.
TEST(SessionCache, AQueuedRequestWhoseClientLeftNeverRuns) {
  auto m = OpenSmall();
  if (!m) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  constexpr std::uint32_t kCtx = 2048;
  std::string e;
  auto engine = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  auto cold = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  ASSERT_TRUE(engine && cold) << e;
  server::FlashNextGenerator g(*engine, std::vector<text::TokenId>{}, kCtx, {});
  server::FlashNextGenerator ref(*cold, std::vector<text::TokenId>{}, kCtx, {.mode = server::SessionCacheOptions::Mode::kOff});
  std::vector<std::int32_t> pa(300), pb(200);
  for (std::size_t i = 0; i < pa.size(); ++i) pa[i] = static_cast<std::int32_t>((i * 7919) % 1000);
  for (std::size_t i = 0; i < pb.size(); ++i) pb[i] = static_cast<std::int32_t>((i * 104729) % 1000);
  server::TextGenerator::Params p;
  p.max_tokens = 6;
  std::atomic<bool> in_decode{false}, release{false}, left{false};
  std::vector<std::int32_t> got_a, want_a;
  std::thread holder([&] {
    (void)g.GenerateIds(pa, pa.size() - 1, p, [&](text::TokenId t) {
      got_a.push_back(static_cast<std::int32_t>(t));
      in_decode = true;
      while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      return true;
    });
  });
  while (!in_decode) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  server::TextGenerator::Params q = p;
  q.cancelled = [&] { return left.load(); };
  std::atomic<std::size_t> ran{0};
  std::atomic<bool> returned{false};
  server::TextGenerator::Result rq;
  std::thread queued([&] {
    rq = g.GenerateIds(pb, pb.size() - 1, q, [&](text::TokenId) { return ++ran, true; });
    returned = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const auto t0 = std::chrono::steady_clock::now();
  left = true;
  // The holder stays held until the queued request returns (or 2 s pass, so a request that waits for the
  // engine fails the test instead of deadlocking it).
  while (!returned && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2))
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const auto waited = std::chrono::steady_clock::now() - t0;
  const bool returned_while_held = returned.load();
  release = true;
  queued.join();
  holder.join();
  EXPECT_TRUE(returned_while_held) << "the queued request waited for the holder";
  EXPECT_LT(waited, std::chrono::milliseconds(500));
  EXPECT_TRUE(rq.cancelled);
  EXPECT_TRUE(rq.error.empty()) << rq.error;
  EXPECT_EQ(ran.load(), 0u);
  ASSERT_TRUE(ref.GenerateIds(pa, pa.size() - 1, p, [&](text::TokenId t) {
                   return want_a.push_back(static_cast<std::int32_t>(t)), true;
                 }).error.empty());
  EXPECT_EQ(got_a, want_a);
  std::vector<std::int32_t> got_b, want_b;
  ASSERT_TRUE(g.GenerateIds(pb, pb.size() - 1, p, [&](text::TokenId t) { return got_b.push_back(t), true; }).error.empty());
  ASSERT_TRUE(ref.GenerateIds(pb, pb.size() - 1, p, [&](text::TokenId t) { return want_b.push_back(t), true; }).error.empty());
  EXPECT_EQ(got_b, want_b);
}

namespace {

/// The HIP device with a memory ledger in place of the real headroom, modelling what ReleasedPageChargeProbe
/// measured: a released page stays charged while any range it was mapped into is still reserved, a new page
/// reuses a charged one first, and creating a page past the budget fails. Buffers count while allocated.
class LedgerDevice final : public rt::Device {
 public:
  explicit LedgerDevice(rt::Device& d) : d_(d) {}
  std::size_t budget = SIZE_MAX;
  [[nodiscard]] std::size_t Used() const {
    std::lock_guard<std::mutex> lk(mu_);
    return UsedLocked();
  }
  rt::Backend backend() const noexcept override { return d_.backend(); }
  rt::Buffer Alloc(std::size_t bytes, std::string* error) override {
    rt::Buffer b = d_.Alloc(bytes, error);
    std::lock_guard<std::mutex> lk(mu_);
    if (b.device) alloc_ += (sizes_[b.device] = b.bytes);
    return b;
  }
  void Free(rt::Buffer& b) noexcept override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (auto it = sizes_.find(b.device); it != sizes_.end()) {
        alloc_ -= it->second;
        sizes_.erase(it);
      }
    }
    d_.Free(b);
  }
  bool Upload(const rt::Buffer& dst, std::size_t off, const void* src, std::size_t n, std::string* e) override {
    return d_.Upload(dst, off, src, n, e);
  }
  bool Download(const rt::Buffer& src, std::size_t off, void* dst, std::size_t n, std::string* e) override {
    return d_.Download(src, off, dst, n, e);
  }
  bool UploadAsync(const rt::Buffer& dst, std::size_t off, const void* src, std::size_t n, std::string* e) override {
    return d_.UploadAsync(dst, off, src, n, e);
  }
  bool Fill(const rt::Buffer& dst, std::uint8_t v, std::string* e) override { return d_.Fill(dst, v, e); }
  void* AllocHost(std::size_t n, std::string* e) override { return d_.AllocHost(n, e); }
  void FreeHost(void* p) noexcept override { d_.FreeHost(p); }
  std::size_t PageBytes() const noexcept override { return d_.PageBytes(); }
  rt::Buffer ReserveRange(std::size_t n, std::string* e) override {
    rt::Buffer b = d_.ReserveRange(n, e);
    std::lock_guard<std::mutex> lk(mu_);
    if (b.device) reserved_.insert(b.device);
    return b;
  }
  void ReleaseRange(rt::Buffer& r) noexcept override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      reserved_.erase(r.device);
      for (auto it = charged_.begin(); it != charged_.end();) {
        it->erase(r.device);
        it = it->empty() ? charged_.erase(it) : std::next(it);
      }
    }
    d_.ReleaseRange(r);
  }
  void* CreatePage(std::string* error) override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!charged_.empty()) {
        charged_.pop_front();  // the runtime hands back a released page: no new charge
      } else if (UsedLocked() + d_.PageBytes() > budget) {
        if (error) *error = "ledger: out of device memory";
        return nullptr;
      }
      ++live_;
    }
    return d_.CreatePage(error);
  }
  void ReleasePage(void* page) noexcept override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      --live_;
      std::set<void*> still;
      for (void* r : mapped_[page])
        if (reserved_.count(r)) still.insert(r);
      mapped_.erase(page);
      if (!still.empty()) charged_.push_back(std::move(still));
    }
    d_.ReleasePage(page);
  }
  bool MapPage(const rt::Buffer& range, std::size_t off, void* page, std::string* e) override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      mapped_[page].insert(range.device);
    }
    return d_.MapPage(range, off, page, e);
  }
  bool UnmapPage(const rt::Buffer& range, std::size_t off, std::string* e) override { return d_.UnmapPage(range, off, e); }
  std::size_t AvailableBytes() const override {
    const std::size_t u = Used();
    return budget > u ? budget - u : 0;
  }
  bool LoadModule(std::span<const std::uint8_t> h, std::string* e) override { return d_.LoadModule(h, e); }
  rt::KernelId Kernel(std::string_view n, std::string* e) override { return d_.Kernel(n, e); }
  bool ReadGlobal(std::string_view n, void* dst, std::size_t b, std::string* e) override { return d_.ReadGlobal(n, dst, b, e); }
  bool WriteGlobal(std::string_view n, const void* src, std::size_t b, std::string* e) override {
    return d_.WriteGlobal(n, src, b, e);
  }
  bool Launch(rt::KernelId k, rt::Dim3 g, rt::Dim3 b, std::span<const rt::Arg> a, std::string* e) override {
    return d_.Launch(k, g, b, a, e);
  }
  bool Synchronize(std::string* e) override { return d_.Synchronize(e); }
  bool BeginCapture(std::string* e) override { return d_.BeginCapture(e); }
  bool EndCapture(rt::Graph* g, std::string* e) override { return d_.EndCapture(g, e); }
  bool LaunchGraph(const rt::Graph& g, std::string* e) override { return d_.LaunchGraph(g, e); }
  void FreeGraph(rt::Graph& g) noexcept override { d_.FreeGraph(g); }

 private:
  std::size_t UsedLocked() const { return alloc_ + (live_ + charged_.size()) * d_.PageBytes(); }
  rt::Device& d_;
  mutable std::mutex mu_;
  std::size_t alloc_ = 0, live_ = 0;
  std::map<void*, std::size_t> sizes_;
  std::set<void*> reserved_;
  std::map<void*, std::set<void*>> mapped_;  // page -> ranges it was mapped into
  std::deque<std::set<void*>> charged_;      // released pages still charged: the reserved ranges they were in
};

}  // namespace

// Making room by eviction sees the memory it frees (audit F3, not reproduced: DropSession releases the evicted
// session's ranges, and with them its pages' charge; ReleasedPageChargeProbe). On a device ledger that keeps a
// released page charged while its range is reserved (as the HIP runtime does), a prompt that fits once two of three
// retained checkpoints are gone evicts exactly two and is admitted, and its tokens equal a cold run's; a prompt that
// cannot fit even with every checkpoint gone is refused before its prefill.
TEST(SessionCache, EvictionStopsOnceThePromptFits) {
  auto m = OpenSmall();
  if (!m) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  constexpr std::uint32_t kCtx = 8192;
  LedgerDevice ledger(*m->dev);
  std::string e;
  auto engine = engine::FlashNextGpu::Create(ledger, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  auto cold = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  ASSERT_TRUE(engine && cold) << e;
  server::FlashNextGenerator g(*engine, std::vector<text::TokenId>{}, kCtx, {});
  server::FlashNextGenerator ref(*cold, std::vector<text::TokenId>{}, kCtx, {.mode = server::SessionCacheOptions::Mode::kOff});
  const auto prompt = [](std::uint32_t seed) {
    std::vector<std::int32_t> v(1500);
    for (auto& t : v) t = static_cast<std::int32_t>((seed = seed * 1664525u + 1013904223u) >> 8) % 1000;
    return v;
  };
  server::TextGenerator::Params p;
  p.max_tokens = 4;
  const auto run = [&](server::FlashNextGenerator& gen, const std::vector<std::int32_t>& ids, std::vector<std::int32_t>* out) {
    return gen.GenerateIds(ids, ids.size() - 1, p, [out](text::TokenId t) { return out->push_back(t), true; });
  };
  for (std::uint32_t s = 1; s <= 3; ++s) {
    std::vector<std::int32_t> out;
    ASSERT_TRUE(run(g, prompt(s), &out).error.empty());
  }
  ASSERT_EQ(g.cache().retained(), 3u);
  const double each = double(g.cache().retained_bytes()) / 3;
  ASSERT_GT(each, 0);
  // The least headroom with which the next prompt is admitted without evicting, measured as Prepare checks it: with
  // a fresh session (its recurrent buffers allocated) bound. Headroom here means before that session exists.
  const auto ids = prompt(4);
  const auto positions = static_cast<std::uint32_t>(ids.size() + p.max_tokens);
  const auto headroom = [&] {
    const std::size_t before = ledger.Used();
    engine::FlashNextGpu::SessionId s = 0;
    EXPECT_TRUE(engine->NewSession(&s, &e) && engine->BindSession(s, &e)) << e;
    const std::size_t session = ledger.Used() - before;
    const auto fits = [&](std::size_t avail) {
      ledger.budget = ledger.Used() + avail;
      return engine->KvRoomFor(positions, static_cast<std::uint32_t>(ids.size() - 1), nullptr);
    };
    std::size_t lo = 0, hi = std::size_t{64} << 30;
    EXPECT_TRUE(fits(hi));
    while (hi - lo > 4096) (fits((lo + hi) / 2) ? hi : lo) = (lo + hi) / 2;
    EXPECT_TRUE(engine->BindSession(0, &e) && engine->DropSession(s, &e)) << e;
    ledger.budget = SIZE_MAX;
    return hi + session;
  };
  const std::size_t hi = headroom();
  std::printf("each checkpoint holds %.1f MB; the prompt needs %.3f GB of headroom\n", each / 1e6, hi / 1e9);
  // Short by 1.5 checkpoints: two must go, and no more.
  ledger.budget = ledger.Used() + hi - static_cast<std::size_t>(1.5 * each);
  std::vector<std::int32_t> got, want;
  const auto r = run(g, ids, &got);
  ASSERT_TRUE(r.error.empty()) << r.error << " (the eviction's gains were not seen)";
  EXPECT_EQ(g.cache().retained(), 2u) << "evicted more than it needed (or less): 1 left + this prompt's checkpoint";
  ASSERT_TRUE(run(ref, ids, &want).error.empty());
  EXPECT_EQ(got, want);
  // Short by more than everything retained holds: refused before the prefill.
  const auto ids2 = prompt(5);
  ledger.budget = ledger.Used() + hi - g.cache().retained_bytes() - static_cast<std::size_t>(each);
  std::vector<std::int32_t> none;
  const auto r2 = run(g, ids2, &none);
  EXPECT_NE(r2.error.find("not enough memory"), std::string::npos) << r2.error;
  EXPECT_TRUE(none.empty());
  // Room again: served, and equal to a cold run.
  ledger.budget = SIZE_MAX;
  std::vector<std::int32_t> got2, want2;
  ASSERT_TRUE(run(g, ids2, &got2).error.empty());
  ASSERT_TRUE(run(ref, ids2, &want2).error.empty());
  EXPECT_EQ(got2, want2);
}

// MTP in the server (docs/MTP.md "Server integration", server/decode_loop.hpp) through the generator and the exact
// prefix cache: agent conversations as above, generator `a` decoding speculatively (SpecOptions::cycle runs
// SpecStepDrafts with drafts from the cold run's own continuation or random, at the depth the loop asks for: the
// small fixture has no draft block), `b` cold without MTP. Every greedy request emits exactly b's tokens and ends
// the same way (stop token, max_tokens cut inside what a cycle could emit), and the next turn's reuse of what `a`
// ran (accepted drafts past a stop included) stays byte-identical to cold. Requests with penalties never draft;
// sampled ones without penalties reach the cycle with the request's sampler. Positive control: the cycle hook ran
// and accepted drafts (else the comparison is AR against AR).
TEST(SessionCache, SpeculativeDecodeIsGreedyIdenticalAndReuseStaysExact) {
  auto m = OpenSmall();
  if (!m) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  constexpr std::uint32_t kCtx = 1024;
  std::string e;
  auto ea = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  auto eb = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  ASSERT_TRUE(ea && eb) << e;
  ASSERT_TRUE(ea->BatchSupported()) << ea->BatchUnsupported();
  const auto V = static_cast<std::int32_t>(m->weights->config.vocab_size);
  std::uint64_t cycles = 0, accepted = 0, stops_inside = 0, cuts = 0, reused = 0;
  hegel::test([&](hegel::TestCase& tc) {
    const auto tokens = [&](std::uint32_t lo, std::uint32_t hi) {
      std::vector<std::int32_t> v(tc.draw(gs::integers<std::uint32_t>({.min_value = lo, .max_value = hi})));
      for (auto& t : v) t = tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = V - 1}));
      return v;
    };
    // Stop tokens: none, or tokens the cold run tends to produce (drawn from its first continuation below).
    const std::vector<std::int32_t> gen_prompt = {7, 11, 13};
    std::vector<std::int32_t> prompt = tokens(20, 200);
    std::size_t stable = prompt.size();
    prompt.insert(prompt.end(), gen_prompt.begin(), gen_prompt.end());
    std::vector<text::TokenId> stops;
    if (tc.draw(gs::booleans())) {
      server::FlashNextGenerator probe(*eb, std::vector<text::TokenId>{}, kCtx, {.mode = server::SessionCacheOptions::Mode::kOff});
      std::vector<std::int32_t> cont;
      server::TextGenerator::Params p;
      p.max_tokens = 12;
      ASSERT_TRUE(probe.GenerateIds(prompt, stable, p, [&](text::TokenId t) { return cont.push_back(t), true; }).error.empty());
      stops.push_back(static_cast<text::TokenId>(cont[tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = cont.size() - 1}))]));
    }
    // The cold reference of the request in flight, which the cycle's drafts are drawn from.
    std::vector<std::int32_t> want;
    std::size_t emitted = 0;  // tokens `a` emitted (cycle outputs) in this request
    bool penalties = false, sampled = false;
    server::SpecOptions spec;
    spec.enabled = true;
    spec.cycle = [&](std::int32_t token, std::uint32_t k, sampling::SamplerState* sampler,
                     engine::FlashNextGpu::SpecResult* r, std::string* err) {
      EXPECT_FALSE(penalties) << "a request with penalties drafted";
      EXPECT_EQ(sampler != nullptr, sampled) << "greedy requests verify on the device, sampled ones with their sampler";
      if (sampler) {  // no sampled verification without a draft block: an ordinary step through the sampler
        std::vector<float> l(static_cast<std::size_t>(V));
        if (!ea->Step(token, l, err)) return false;
        r->tokens.assign(1, static_cast<std::int32_t>(sampler->Sample(l)));
        r->drafted = r->accepted = 0;
        return true;
      }
      std::vector<std::int32_t> d(k);
      for (std::uint32_t j = 0; j < k; ++j) {
        const std::size_t i = emitted + j;
        d[j] = i < want.size() && tc.draw(gs::booleans()) ? want[i]
                                                         : tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = V - 1}));
      }
      if (!ea->SpecStepDrafts(token, d, r, err)) return false;
      emitted += r->tokens.size();
      ++cycles;
      accepted += r->accepted;
      return true;
    };
    server::FlashNextGenerator a(*ea, stops, kCtx, {.mode = server::SessionCacheOptions::Mode::kExact, .align = kAlign}, spec);
    server::FlashNextGenerator b(*eb, stops, kCtx, {.mode = server::SessionCacheOptions::Mode::kOff});
    const int turns = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 4}));
    for (int turn = 0; turn < turns && prompt.size() + 40 < kCtx; ++turn) {
      server::TextGenerator::Params p;
      p.max_tokens = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 16}));
      const auto mode = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}));  // 0, 1: greedy
      sampled = mode >= 2;
      penalties = mode == 3;
      if (sampled) {
        p.sampling.temperature = 0.8F;
        p.sampling.seed = tc.draw(gs::integers<std::int64_t>({.min_value = 0, .max_value = 1 << 20}));
        if (penalties) p.sampling.repeat_penalty = 1.3F;
      }
      // Cold first (its tokens, plus a margin past the budget, are the oracle drafts), then speculative.
      std::vector<std::int32_t> got;
      want.clear();
      {
        server::TextGenerator::Params q = p;
        q.max_tokens = p.max_tokens + engine::FlashNextGpu::kMaxDrafts;
        std::vector<std::int32_t> longer;
        server::FlashNextGenerator nb(*eb, std::vector<text::TokenId>{}, kCtx, {.mode = server::SessionCacheOptions::Mode::kOff});
        if (!sampled) ASSERT_TRUE(nb.GenerateIds(prompt, stable, q, [&](text::TokenId t) { return longer.push_back(t), true; }).error.empty());
        want = longer;
      }
      std::vector<std::int32_t> cold;
      const auto rb = b.GenerateIds(prompt, stable, p, [&](text::TokenId t) { return cold.push_back(t), true; });
      emitted = 0;
      const auto ra = a.GenerateIds(prompt, stable, p, [&](text::TokenId t) { return got.push_back(t), true; });
      ASSERT_TRUE(ra.error.empty() && rb.error.empty()) << ra.error << rb.error;
      reused += ra.cached_tokens;
      if (penalties || sampled) {
        ASSERT_EQ(ra.spec.cycles, 0u);
        if (penalties) ASSERT_EQ(got, cold) << "with penalties the request steps as without MTP";
      } else {
        ASSERT_EQ(got, cold) << "turn " << turn << ", prompt " << prompt.size() << ", reused " << ra.cached_tokens;
        ASSERT_EQ(ra.length_limited, rb.length_limited);
        ASSERT_EQ(ra.completion_tokens, rb.completion_tokens);
        // A stop inside an accepted run: `a` ran past the token it stopped at.
        stops_inside += !ra.length_limited && got.size() < p.max_tokens && emitted > got.size() + 1;
        cuts += ra.length_limited && ra.spec.cycles > 0;
      }
      // The next prompt: this one up to its generation prompt, then this turn echoed or rendered differently.
      std::vector<std::int32_t> next(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(stable));
      next.insert(next.end(), gen_prompt.begin(), gen_prompt.end());
      if (tc.draw(gs::booleans())) next.insert(next.end(), got.begin(), got.end());
      else {
        const auto other = tokens(1, 20);
        next.insert(next.end(), other.begin(), other.end());
      }
      const auto tool = tokens(1, 100);
      next.insert(next.end(), tool.begin(), tool.end());
      stable = next.size();
      next.insert(next.end(), gen_prompt.begin(), gen_prompt.end());
      prompt = std::move(next);
    }
  }, {.test_cases = 40});
  std::printf("speculative: %llu cycles, %llu drafts accepted, %llu stops inside a cycle's output, %llu budget cuts, "
              "%llu prompt tokens reused\n", static_cast<unsigned long long>(cycles),
              static_cast<unsigned long long>(accepted), static_cast<unsigned long long>(stops_inside),
              static_cast<unsigned long long>(cuts), static_cast<unsigned long long>(reused));
  EXPECT_GT(accepted, cycles / 4) << "positive control: hardly any draft was accepted";
  EXPECT_GT(reused, 0u) << "the cache never reused a speculative session";
}

// rt::HipDevice keeps released ranges reserved and hands them out again (the runtime reusing a freed address made
// a write to a freshly mapped page fault: docs/SESSIONS.md section 10). 32 ranges of 3 pages are used and released;
// 96 ranges of 1 or 2 pages reserved meanwhile overlap none of them (they stay reserved; returned to the runtime,
// their addresses are handed out again for other sizes); 32 new 3-page ranges are exactly the released ones. Every
// range's fresh pages hold what was written through them.
TEST(SessionCache, ReleasedRangesComeBackAndRemapCleanly) {
  std::string e;
  auto dev = rt::Device::Create(rt::Backend::kHip, &e);
  if (!dev) GTEST_SKIP() << "no GPU backend";
  const std::size_t pb = dev->PageBytes();
  if (pb == 0) GTEST_SKIP() << "no page mapping";
  // Maps a fresh page at each page of `r`, fills it with `v`, reads it back, unmaps and releases the pages.
  const auto use = [&](const rt::Buffer& r, std::uint8_t v) {
    std::vector<void*> h(r.bytes / pb);
    for (std::size_t j = 0; j < h.size(); ++j) {
      h[j] = dev->CreatePage(&e);
      ASSERT_TRUE(h[j] && dev->MapPage(r, j * pb, h[j], &e)) << e;
    }
    ASSERT_TRUE(dev->Fill(r, v, &e) && dev->Synchronize(&e)) << e;
    std::vector<std::uint8_t> back(r.bytes);
    ASSERT_TRUE(dev->Download(r, 0, back.data(), back.size(), &e)) << e;
    ASSERT_TRUE(std::all_of(back.begin(), back.end(), [&](std::uint8_t b) { return b == v; })) << int(v);
    for (std::size_t j = 0; j < h.size(); ++j) {
      ASSERT_TRUE(dev->UnmapPage(r, j * pb, &e)) << e;
      dev->ReleasePage(h[j]);
    }
  };
  std::vector<rt::Buffer> big(32), small(96);
  std::set<void*> released;
  for (auto& r : big) {
    r = dev->ReserveRange(3 * pb, &e);
    ASSERT_TRUE(r.device) << e;
  }
  for (auto& r : big) {
    ASSERT_NO_FATAL_FAILURE(use(r, 1));
    released.insert(r.device);
    dev->ReleaseRange(r);
  }
  for (std::size_t i = 0; i < small.size(); ++i) {
    small[i] = dev->ReserveRange((1 + i % 2) * pb, &e);
    ASSERT_TRUE(small[i].device) << e;
    const auto* p = static_cast<const char*>(small[i].device);
    for (void* q : released)
      ASSERT_TRUE(p + small[i].bytes <= static_cast<const char*>(q) || static_cast<const char*>(q) + 3 * pb <= p)
          << "range " << i << " overlaps a released range that should still be reserved";
    ASSERT_NO_FATAL_FAILURE(use(small[i], static_cast<std::uint8_t>(2 + i)));
  }
  for (auto& r : big) {
    r = dev->ReserveRange(3 * pb, &e);
    ASSERT_TRUE(r.device) << e;
    ASSERT_EQ(released.erase(r.device), 1u) << "a 3-page range did not come from the released ones";
    ASSERT_NO_FATAL_FAILURE(use(r, 200));
  }
  for (auto& r : big) dev->ReleaseRange(r);
  for (auto& r : small) dev->ReleaseRange(r);
}

namespace {
// Delegates to the yard provider; when armed, a revoke arrives right after a restore took its open lease (a
// restore in flight at the revoke).
struct RevokeDuringRestore final : server::SpillKeys {
  server::YardSpillKeys& y;
  const yardtest::Signer& signer;
  bool armed = false;
  server::YardSpillKeys::Answer revoke;
  RevokeDuringRestore(server::YardSpillKeys& yard, const yardtest::Signer& s) : y(yard), signer(s) {}
  std::string Describe() const override { return y.Describe(); }
  bool SealLease(std::uint64_t entry, std::string_view label, bool bound, server::SpillLease* out,
                 std::string* why) override {
    return y.SealLease(entry, label, bound, out, why);
  }
  bool OpenLease(std::string_view label, std::string_view id, server::SpillLease* out, std::string* why) override {
    const bool ok = y.OpenLease(label, id, out, why);
    if (ok && armed) {
      armed = false;
      revoke = yardtest::Register(y, signer, std::string(label), "revoke", 9, "");
    }
    return ok;
  }
  void Finish(const server::SpillLease& lease, Use use, bool ok) override { y.Finish(lease, use, ok); }
  bool MayOpen(std::string_view label, std::string_view id) override { return y.MayOpen(label, id); }
};
}  // namespace

// Phase B in the cache: an entry is sealed only under the key yard-ctl registered for that entry (the seal claim
// names it; there is no ?session=), cannot be restored until yard registers the open key at the next epoch (the
// entry waits, untouched), restores bit-exact then, needs a new seal registration for the next spill (keys are
// one-shot) and never falls back to the master once yard-bound (custody); a revocation erases exactly the entries
// at or below its fence; a restore in flight at a revoke completes and the revoke's cleanup then erases it.
TEST(SessionCache, YardKeysAuthoriseResume) {
  auto m = OpenSmall();
  if (!m) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  const char* d = std::getenv("HALO_SPILL_TEST_DIR");
  const std::string dir = (d && *d ? std::filesystem::path(d) : std::filesystem::temp_directory_path() / "halo-spill-test") /
                          ("yard-" + std::to_string(::getpid()));
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  yardtest::Signer signer("kid-a");
  yardtest::WriteFile(dir + "/jwks.json", yardtest::JwksJson({&signer}));
  std::string e;
  core::crypt::SecretBytes mk(32);
  ASSERT_TRUE(core::crypt::RandomBytes({mk.data(), 32}));
  auto master = server::MasterSpillKeys::Create("test", std::move(mk), "", &e);
  ASSERT_TRUE(master) << e;
  server::YardSpillKeys::Options yo;
  yo.jwks_path = dir + "/jwks.json";
  yo.issuer = "https://id.example.org/yard";
  yo.fallback = master.get();  // --spill-yard-fallback master: still never for a yard-bound entry
  yo.state_path = dir + "/yard-state";
  auto yard = server::YardSpillKeys::Create(yo, &e);
  ASSERT_TRUE(yard) << e;
  RevokeDuringRestore keys(*yard, signer);
  constexpr std::uint32_t kCtx = 1024;
  auto ea = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  auto eb = engine::FlashNextGpu::Create(*m->dev, *m->weights, m->ngram.get(), nullptr, kCtx, &e);
  ASSERT_TRUE(ea && eb) << e;
  std::vector<std::int32_t> prompt(200);
  for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = static_cast<std::int32_t>((i * 7919 + 3) % 1000);
  const auto collect = [](std::vector<std::int32_t>* out) {
    return [out](text::TokenId t) { return out->push_back(static_cast<std::int32_t>(t)), true; };
  };
  server::FlashNextGenerator cold(*eb, std::vector<text::TokenId>{}, kCtx, {.mode = server::SessionCacheOptions::Mode::kOff});
  server::TextGenerator::Params p;
  p.max_tokens = 4;
  std::vector<std::int32_t> want;
  ASSERT_TRUE(cold.GenerateIds(prompt, 190, p, collect(&want)).error.empty());
  server::SessionCacheOptions o{.mode = server::SessionCacheOptions::Mode::kExact, .align = 64, .max_retained = 4};
  o.spill_dir = dir + "/spill";
  o.spill_keys = &keys;
  server::FlashNextGenerator g(*ea, std::vector<text::TokenId>{}, kCtx, o);
  std::vector<std::int32_t> got;
  ASSERT_TRUE(g.GenerateIds(prompt, 190, p, collect(&got)).error.empty());
  ASSERT_EQ(got, want);
  const std::string k1 = yardtest::Key32(1), k2 = yardtest::Key32(2), k3 = yardtest::Key32(3);
  g.WithCache([&](server::SessionCache& c) {
    const auto key = c.List().front().key;
    const auto entry = static_cast<std::int64_t>(key);
    std::string why;
    bool kept = false;
    // A seal registration for another entry does not seal this one.
    EXPECT_EQ(yardtest::Register(*yard, signer, "sess-1", "seal", 3, k1, entry + 100).status, 200);
    EXPECT_TRUE(c.Spill(key, &why)) << why;  // never yard-bound: the master fallback seals it
    EXPECT_EQ(c.List().front().key_id.rfind("test:", 0), 0u) << c.List().front().key_id;
    EXPECT_TRUE(c.Restore(key, &why)) << why;
    EXPECT_EQ(yardtest::Register(*yard, signer, "sess-1", "seal", 4, k1, entry).status, 200);
    EXPECT_TRUE(c.Spill(key, &why)) << why;
    EXPECT_EQ(c.List().front().key_id.rfind("yard:", 0), 0u) << c.List().front().key_id;
    EXPECT_EQ(c.List().front().session, "sess-1");
    EXPECT_TRUE(c.List().front().yard_bound);
    EXPECT_FALSE(c.Restore(key, &why, &kept));
    EXPECT_TRUE(kept) << "an entry waiting for yard's open key was dropped: " << why;
    EXPECT_EQ(yardtest::Register(*yard, signer, "sess-1", "open", 4, k1).status, 409) << "open before the commit";
    EXPECT_EQ(yardtest::Register(*yard, signer, "sess-1", "open", 5, k1).status, 200);
    EXPECT_TRUE(c.Restore(key, &why, &kept)) << why;
    // Custody: no seal registration now; the master fallback must not take a yard-bound entry.
    EXPECT_FALSE(c.Spill(key, &why)) << "a yard-bound entry was sealed under the master (custody downgrade)";
    EXPECT_EQ(FilesIn(o.spill_dir), 0u);
    EXPECT_EQ(yardtest::Register(*yard, signer, "sess-1", "seal", 5, k2, entry).status, 200);
    EXPECT_TRUE(c.Spill(key, &why)) << why;
    // A restore in flight when a revoke (through 5) lands: it completes with the key it holds; the revoke's
    // cleanup, which the route runs under the generation lock after it, erases the restored entry.
    EXPECT_EQ(yardtest::Register(*yard, signer, "sess-1", "open", 6, k2).status, 200);
    keys.armed = true;
    EXPECT_TRUE(c.Restore(key, &why, &kept)) << why;
    EXPECT_EQ(keys.revoke.status, 200) << keys.revoke.body;
    EXPECT_EQ(keys.revoke.revoked_through, 9u);
    EXPECT_EQ(c.DropRevoked(keys.revoke.revoked_session, keys.revoke.revoked_through), 1u);
    EXPECT_EQ(c.List().size(), 0u);
    EXPECT_EQ(FilesIn(o.spill_dir), 0u);
    return true;
  });
  got.clear();
  std::size_t reused = 0;
  {
    const auto r = g.GenerateIds(prompt, 190, p, collect(&got));
    ASSERT_TRUE(r.error.empty()) << r.error;
    reused = r.cached_tokens;
  }
  EXPECT_EQ(got, want);
  EXPECT_EQ(reused, 0u) << "a revoked entry was reused";
  // The revoke's fence: a newer checkpoint (sealed at 10 > 9) survives a delayed cleanup of the old revoke.
  g.WithCache([&](server::SessionCache& c) {
    const auto key = c.List().front().key;
    std::string why;
    EXPECT_EQ(yardtest::Register(*yard, signer, "sess-1", "seal", 10, k3, static_cast<std::int64_t>(key)).status, 200);
    EXPECT_TRUE(c.Spill(key, &why)) << why;
    EXPECT_EQ(c.DropRevoked("sess-1", 9), 0u) << "a delayed revoke cleanup erased a newer checkpoint";
    EXPECT_TRUE(c.List().front().spilled);
    return true;
  });
  std::filesystem::remove_all(dir);
}
