// FlashNextGenerator (the server's engine adapter) on the tiny random
// Flash-Next, on every built backend:
//   - greedy text equals a manual loop of engine steps and argmax, stopping
//     at the same tokens;
//   - fast mode: a follow-up prompt that extends everything run reuses the
//     finished session and generates what a freshly reset engine generates;
//   - max_tokens and cancellation end generation where they should.
//   HALO_TINY_HF_DIR  tiny random Flash-Next (tools/make_tiny_hf_reference.py)

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "engine/flash_next_gpu.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"
#include "server/flash_next_backend.hpp"
#include "server/flash_next_generator.hpp"
#include "text/tokenizer.hpp"

using namespace halo;

namespace {

struct Tiny {
  std::unique_ptr<core::GgufReader> reader;
  std::optional<model::ModelWeights> weights;
  std::unique_ptr<model::NgramTable> ngram;
  std::unique_ptr<text::Tokenizer> tok;
};

Tiny* Load() {
  static Tiny* t = []() -> Tiny* {
    const char* dir = std::getenv("HALO_TINY_HF_DIR");
    if (!dir) return nullptr;
    auto* t = new Tiny;
    std::string e;
    t->reader = core::GgufReader::OpenFile(std::string(dir) + "/tiny-f32.gguf", &e);
    if (!t->reader) return nullptr;
    t->weights = model::ModelWeights::Bind(*t->reader, &e);
    t->tok = text::Tokenizer::FromGguf(*t->reader, &e);
    if (!t->weights || !t->tok) return nullptr;
    const auto& p = t->weights->ple_table;
    t->ngram = model::NgramTable::Open(t->reader->GetMappedRegions()[p.shard].file_descriptor, p.file_offset, p.rows,
                                       t->weights->config.ple_head_dim, p.type, &e);
    return t->ngram ? t : nullptr;
  }();
  return t;
}

std::string Collect(server::TextGenerator& g, const std::string& prompt, std::uint32_t max_tokens,
                    server::TextGenerator::Result* result = nullptr, std::size_t cancel_after = SIZE_MAX) {
  server::TextGenerator::Params p;
  p.prompt = prompt;
  p.max_tokens = max_tokens;  // sampling defaults: temperature 0 (greedy)
  std::string out;
  std::size_t pieces = 0;
  auto r = g.Generate(p, [&](std::string_view s) {
    out += s;
    return ++pieces < cancel_after;
  });
  if (result) *result = r;
  return out;
}

const char* kPrompt = "<|im_start|>user\nHello there<|im_end|>\n<|im_start|>assistant\n";

}  // namespace

TEST(FlashNextGenerator, GreedyMatchesEngineStepsAndReuseIsExact) {
  Tiny* t = Load();
  if (!t) GTEST_SKIP() << "set HALO_TINY_HF_DIR";
  for (auto b : {rt::Backend::kHip, rt::Backend::kHrx}) {
    std::string e;
    auto dev = rt::Device::Create(b, &e);
    if (!dev) continue;
    constexpr std::uint32_t kCtx = 256;
    auto engine = engine::FlashNextGpu::Create(*dev, *t->weights, t->ngram.get(), nullptr, kCtx, &e);
    ASSERT_TRUE(engine) << e;
    server::FlashNextGenerator gen(*engine, *t->tok, kCtx);
    server::TextGenerator::Result r;
    const std::string out = Collect(gen, kPrompt, 24, &r);
    EXPECT_TRUE(r.error.empty()) << r.error;
    EXPECT_EQ(r.cached_tokens, 0u);

    // Manual: reset, step the prompt, argmax until a stop or 24 tokens.
    auto ids = t->tok->Encode(kPrompt, true);
    ASSERT_TRUE(engine->Reset(&e)) << e;
    std::vector<float> logits(t->weights->config.vocab_size);
    for (std::size_t i = 0; i + 1 < ids.size(); ++i) ASSERT_TRUE(engine->Step(static_cast<std::int32_t>(ids[i]), {}, &e));
    std::string want;
    std::int32_t tok = static_cast<std::int32_t>(ids.back());
    const auto stop = [&](text::TokenId id) {
      return id == t->tok->Encode("<|im_end|>", true)[0] || id == t->tok->Encode("<|endoftext|>", true)[0] ||
             id == t->tok->eos();
    };
    std::uint64_t n = 0;
    for (; n < 24; ++n) {
      ASSERT_TRUE(engine->Step(tok, logits, &e)) << e;
      const auto next = static_cast<text::TokenId>(std::max_element(logits.begin(), logits.end()) - logits.begin());
      if (stop(next)) break;
      want += t->tok->TokenBytes(next);
      tok = static_cast<std::int32_t>(next);
    }
    EXPECT_EQ(out, want) << rt::BackendName(b);
    EXPECT_EQ(r.completion_tokens, n);
    EXPECT_EQ(r.length_limited, n == 24);

    // Reuse: a fresh generator (engine reset by the manual run above), then
    // a prompt extending everything it ran; compare with a fresh engine.
    // Fast mode: the finished session itself is extended (exact mode re-prefills from an aligned checkpoint;
    // tests/gpu/session_cache_gpu_test.cpp).
    server::FlashNextGenerator a(*engine, *t->tok, kCtx, {.mode = server::SessionCacheOptions::Mode::kFast});
    ASSERT_TRUE(engine->Reset(&e));
    (void)Collect(a, kPrompt, 8);
    const std::size_t ran = a.cached_tokens();
    // The text of what ran (prompt + generated but the last sampled token),
    // then a new user turn.
    auto fresh_engine = engine::FlashNextGpu::Create(*dev, *t->weights, t->ngram.get(), nullptr, kCtx, &e);
    ASSERT_TRUE(fresh_engine) << e;
    server::FlashNextGenerator f(*fresh_engine, *t->tok, kCtx);
    std::string first;
    {
      server::FlashNextGenerator probe(*fresh_engine, *t->tok, kCtx);
      first = Collect(probe, kPrompt, 8);
    }
    const std::string follow = std::string(kPrompt) + first + "<|im_end|>\n<|im_start|>user\nMore<|im_end|>\n<|im_start|>assistant\n";
    server::TextGenerator::Result ra, rf;
    const std::string got = Collect(a, follow, 16, &ra);
    ASSERT_TRUE(fresh_engine->Reset(&e));
    const std::string expect = Collect(f, follow, 16, &rf);
    EXPECT_EQ(got, expect);
    EXPECT_EQ(rf.cached_tokens, 0u);
    // This generation re-tokenizes as it was sampled, so everything that ran
    // is reused (a bookkeeping slip would silently fall back to a reset).
    EXPECT_EQ(ra.cached_tokens, ran);
    std::printf("%s: reused %llu of %llu prompt tokens (%zu had run)\n", rt::BackendName(b),
                static_cast<unsigned long long>(ra.cached_tokens), static_cast<unsigned long long>(ra.prompt_tokens), ran);

    // Limits: max_tokens 3 stops at 3 (unless a stop token comes first);
    // cancelling after the first piece stops there.
    server::TextGenerator::Result rl, rc;
    (void)Collect(gen, kPrompt, 3, &rl);
    EXPECT_LE(rl.completion_tokens, 3u);
    EXPECT_EQ(rl.length_limited, rl.completion_tokens == 3);
    const std::string one = Collect(gen, kPrompt, 24, &rc, 1);
    EXPECT_FALSE(rc.length_limited);
    EXPECT_LE(rc.completion_tokens, 4u);  // stops at the first whole-character piece
    EXPECT_TRUE(out.starts_with(one));
  }
}

TEST(FlashNextGenerator, CompleteUtf8PrefixHoldsBackCutSequences) {
  EXPECT_EQ(server::CompleteUtf8Prefix("abc"), 3u);
  EXPECT_EQ(server::CompleteUtf8Prefix("a\xc3"), 1u);           // é cut after its lead byte
  EXPECT_EQ(server::CompleteUtf8Prefix("a\xc3\xa9"), 3u);
  EXPECT_EQ(server::CompleteUtf8Prefix("\xe2\x82"), 0u);        // € cut short
  EXPECT_EQ(server::CompleteUtf8Prefix("\xf0\x9f\x98"), 0u);    // emoji cut short
  EXPECT_EQ(server::CompleteUtf8Prefix("\xf0\x9f\x98\x80"), 4u);
  EXPECT_EQ(server::CompleteUtf8Prefix("\xff"), 1u);            // invalid bytes pass
  EXPECT_EQ(server::CompleteUtf8Prefix("\x80\x80"), 2u);        // stray continuations pass
}

// Anthropic usage (server/anthropic_service.hpp): cached_tokens are the prompt tokens whose state was reused,
// cache_write_tokens the ones prefilled now that a retained state holds afterwards. Exact mode retains the
// checkpoint at the generation prompt's start rounded down to the alignment; fast mode also keeps the finished
// session, which holds the whole prompt. CountTokens is the prompt's token count.
TEST(FlashNextGenerator, CacheUsageCountsWhatIsReusedAndWhatIsRetained) {
  Tiny* t = Load();
  if (!t) GTEST_SKIP() << "set HALO_TINY_HF_DIR";
  std::string e;
  auto dev = rt::Device::Create(rt::Backend::kHip, &e);
  if (!dev) GTEST_SKIP() << e;
  constexpr std::uint32_t kCtx = 1024;
  auto engine = engine::FlashNextGpu::Create(*dev, *t->weights, t->ngram.get(), nullptr, kCtx, &e);
  ASSERT_TRUE(engine) << e;
  std::string words;
  for (int i = 0; i < 60; ++i) words += "word" + std::to_string(i) + " ";
  const std::string p1 = "<|im_start|>user\n" + words + "<|im_end|>\n<|im_start|>assistant\n";
  const text::TokenId im_start = t->tok->Encode("<|im_start|>", true)[0];
  const auto stable_of = [&](const std::string& p) {
    const auto ids = t->tok->Encode(p, true);
    std::size_t s = 0;
    for (std::size_t i = 0; i < ids.size(); ++i)
      if (ids[i] == im_start) s = i;
    return s;
  };
  {
    server::FlashNextGenerator g(*engine, *t->tok, kCtx);  // exact, align 64
    server::TextGenerator::Result r1, r2;
    const std::string a1 = Collect(g, p1, 4, &r1);
    ASSERT_TRUE(r1.error.empty()) << r1.error;
    EXPECT_EQ(g.CountTokens(p1), std::optional<std::uint64_t>(r1.prompt_tokens));
    const std::size_t ck1 = stable_of(p1) / 64 * 64;
    ASSERT_GT(ck1, 0u) << "the prompt must reach the first aligned checkpoint";
    EXPECT_EQ(r1.cached_tokens, 0u);
    EXPECT_EQ(r1.cache_write_tokens, ck1);
    const std::string p2 = p1 + a1 + "<|im_end|>\n<|im_start|>user\n" + words + "<|im_end|>\n<|im_start|>assistant\n";
    (void)Collect(g, p2, 4, &r2);
    const std::size_t ck2 = stable_of(p2) / 64 * 64;
    EXPECT_EQ(r2.cached_tokens, ck1);
    EXPECT_EQ(r2.cache_write_tokens, ck2 - ck1);
    EXPECT_LE(r2.cached_tokens + r2.cache_write_tokens, r2.prompt_tokens);
  }
  {
    ASSERT_TRUE(engine->Reset(&e)) << e;
    server::FlashNextGenerator g(*engine, *t->tok, kCtx, {.mode = server::SessionCacheOptions::Mode::kFast});
    server::TextGenerator::Result r;
    (void)Collect(g, p1, 4, &r);
    EXPECT_EQ(r.cached_tokens, 0u);
    EXPECT_EQ(r.cache_write_tokens, r.prompt_tokens) << "the finished session holds the whole prompt";
  }
  {
    // The same through the G3 scheduler (halo_serve's default): its backend's cache reports the same accounting,
    // and it counts tokens without the engine thread.
    ASSERT_TRUE(engine->Reset(&e)) << e;
    auto g = server::MakeScheduledFlashNext(*engine, t->tok.get(), {}, kCtx, {}, {}, {});
    server::TextGenerator::Result r1, r2;
    const std::string a1 = Collect(*g, p1, 4, &r1);
    ASSERT_TRUE(r1.error.empty()) << r1.error;
    EXPECT_EQ(g->CountTokens(p1), std::optional<std::uint64_t>(r1.prompt_tokens));
    const std::size_t ck1 = stable_of(p1) / 64 * 64;
    EXPECT_EQ(r1.cached_tokens, 0u);
    EXPECT_EQ(r1.cache_write_tokens, ck1);
    const std::string p2 = p1 + a1 + "<|im_end|>\n<|im_start|>user\n" + words + "<|im_end|>\n<|im_start|>assistant\n";
    (void)Collect(*g, p2, 4, &r2);
    const std::size_t ck2 = stable_of(p2) / 64 * 64;
    EXPECT_EQ(r2.cached_tokens, ck1);
    EXPECT_EQ(r2.cache_write_tokens, ck2 - ck1);
  }
}
