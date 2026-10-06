// P6 sessions (docs/SESSIONS.md) on the small quantized model (every fast decode kernel runs):
//   HALO_SMALL_QUANT_DIR  holds small-q.gguf
// Oracle: a session's logits after any history of new / fork / bind / step / prefill / reset / drop are
// bitwise equal to a fresh engine replaying that session's own operations (its steps and prefill chunks, in
// order, the forked parent's included). Exactness is the contract, so the comparison is memcmp, not a
// tolerance: the engine's state after a token sequence depends only on the sequence and how it was chunked.

#include <hegel/gtest.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "engine/flash_next_gpu.hpp"
#include "core/sampling.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"

namespace gs = hegel::generators;
using namespace halo;

namespace {

struct Small {
  std::unique_ptr<core::GgufReader> reader;
  std::optional<model::ModelWeights> weights;
  std::unique_ptr<model::NgramTable> ngram;
};

Small* LoadSmall() {
  static Small* small = []() -> Small* {
    const char* dir = std::getenv("HALO_SMALL_QUANT_DIR");
    if (!dir) return nullptr;
    auto* s = new Small;
    std::string error;
    s->reader = core::GgufReader::OpenFile(std::string(dir) + "/small-q.gguf", &error);
    if (!s->reader) return nullptr;
    s->weights = model::ModelWeights::Bind(*s->reader, &error);
    if (!s->weights) return nullptr;
    const auto& p = s->weights->ple_table;
    s->ngram = model::NgramTable::Open(s->reader->GetMappedRegions()[p.shard].file_descriptor, p.file_offset, p.rows,
                                       s->weights->config.ple_head_dim, p.type, &error);
    return s->ngram ? s : nullptr;
  }();
  return small;
}

std::vector<std::unique_ptr<rt::Device>> Devices() {
  std::vector<std::unique_ptr<rt::Device>> out;
  for (auto b : {rt::Backend::kHip, rt::Backend::kHrx}) {
    std::string error;
    if (auto d = rt::Device::Create(b, &error)) out.push_back(std::move(d));
  }
  return out;
}

// One operation of a session's history: a step (one token) or a prefill chunk.
struct Op {
  bool prefill = false;
  std::vector<std::int32_t> tokens;
};

// Replays `ops` from an empty sequence; returns the last operation's logits.
std::vector<float> Replay(engine::FlashNextGpu& oracle, const std::vector<Op>& ops, std::size_t V) {
  std::string error;
  std::vector<float> l(V);
  EXPECT_TRUE(oracle.Reset(&error)) << error;
  for (std::size_t i = 0; i < ops.size(); ++i) {
    const bool last = i + 1 == ops.size();
    const auto out = last ? std::span<float>(l) : std::span<float>{};
    if (ops[i].prefill)
      EXPECT_TRUE(oracle.Prefill(ops[i].tokens, out, &error)) << error;
    else
      EXPECT_TRUE(oracle.Step(ops[i].tokens[0], out, &error)) << error;
  }
  return l;
}

}  // namespace

TEST(Sessions, SwarmMatchesFreshReplay) {
  Small* s = LoadSmall();
  if (!s) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  const auto& c = s->weights->config;
  const std::size_t V = c.vocab_size;
  constexpr std::uint32_t kMaxContext = 64;
  // One engine per device under test, plus one oracle engine per device (replays from Reset).
  hegel::test([&](hegel::TestCase& tc) {
    auto& dev = *devices[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = devices.size() - 1}))];
    std::string error;
    auto gpu = engine::FlashNextGpu::Create(dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
    ASSERT_TRUE(gpu) << error;
    auto oracle = engine::FlashNextGpu::Create(dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
    ASSERT_TRUE(oracle) << error;
    gpu->SetUseGraphs(tc.draw(gs::booleans()));
    std::map<engine::FlashNextGpu::SessionId, std::vector<Op>> hist;  // live sessions and their histories
    hist[0] = {};
    std::map<engine::FlashNextGpu::SessionId, std::uint32_t> pos;
    pos[0] = 0;
    enum Kind { kStep, kPrefill, kNew, kFork, kBind, kDrop, kReset, kToggleGraphs, kBatch, kKinds };
    bool enabled[kKinds];
    for (bool& e : enabled) e = tc.draw(gs::booleans());  // swarm: each run uses a random subset of rules
    enabled[kStep] = true;
    const auto token = [&] {
      return tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = static_cast<std::int32_t>(V - 1)}));
    };
    const auto pick_session = [&] {
      std::vector<engine::FlashNextGpu::SessionId> ids;
      for (auto& [id, _] : hist) ids.push_back(id);
      return ids[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = ids.size() - 1}))];
    };
    std::vector<float> got(V);
    const auto check = [&](const char* what) {
      const auto id = gpu->BoundSession();
      const auto want = Replay(*oracle, hist[id], V);
      ASSERT_EQ(std::memcmp(got.data(), want.data(), V * 4), 0)
          << what << ": session " << id << " after " << hist[id].size() << " ops, position " << gpu->Position();
    };
    const int ops = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 30}));
    for (int i = 0; i < ops; ++i) {
      const auto kind = static_cast<Kind>(tc.draw(gs::integers<int>({.min_value = 0, .max_value = kKinds - 1})));
      if (!enabled[kind]) continue;
      const auto cur = gpu->BoundSession();
      switch (kind) {
        case kStep: {
          if (pos[cur] >= kMaxContext) break;
          const auto t = token();
          ASSERT_TRUE(gpu->Step(t, got, &error)) << error;
          hist[cur].push_back({false, {t}});
          ++pos[cur];
          check("step");
          break;
        }
        case kPrefill: {
          const auto n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 20}));
          if (pos[cur] + n > kMaxContext) break;
          Op op{true, std::vector<std::int32_t>(n)};
          for (auto& t : op.tokens) t = token();
          ASSERT_TRUE(gpu->Prefill(op.tokens, got, &error)) << error;
          hist[cur].push_back(op);
          pos[cur] += n;
          check("prefill");
          break;
        }
        case kNew: {
          engine::FlashNextGpu::SessionId id;
          ASSERT_TRUE(gpu->NewSession(&id, &error)) << error;
          hist[id] = {};
          pos[id] = 0;
          break;
        }
        case kFork: {
          const auto src = pick_session();
          engine::FlashNextGpu::SessionId id;
          ASSERT_TRUE(gpu->ForkSession(src, &id, &error)) << error;
          ASSERT_EQ(gpu->SessionPositionOf(id), pos[src]);
          hist[id] = hist[src];
          pos[id] = pos[src];
          break;
        }
        case kBind: {
          const auto id = pick_session();
          ASSERT_TRUE(gpu->BindSession(id, &error)) << error;
          ASSERT_EQ(gpu->Position(), pos[id]);
          break;
        }
        case kDrop: {
          const auto id = pick_session();
          if (id == cur) {
            ASSERT_FALSE(gpu->DropSession(id, &error));
            break;
          }
          ASSERT_TRUE(gpu->DropSession(id, &error)) << error;
          hist.erase(id);
          pos.erase(id);
          ASSERT_FALSE(gpu->HasSession(id));
          break;
        }
        case kReset:
          ASSERT_TRUE(gpu->Reset(&error)) << error;
          hist[cur].clear();
          pos[cur] = 0;
          break;
        case kToggleGraphs:
          gpu->SetUseGraphs(tc.draw(gs::booleans()));
          break;
        case kBatch: {  // one StepBatch over a random subset of the live sessions, in random order
          std::vector<engine::FlashNextGpu::SessionId> ids;
          for (auto& [id, _] : hist)
            if (pos[id] < kMaxContext && tc.draw(gs::booleans())) ids.push_back(id);
          if (ids.empty()) break;
          if (ids.size() > engine::FlashNextGpu::kMaxBatch) ids.resize(engine::FlashNextGpu::kMaxBatch);
          for (std::size_t j = ids.size(); j > 1; --j)
            std::swap(ids[j - 1], ids[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = j - 1}))]);
          std::vector<std::vector<float>> outs(ids.size(), std::vector<float>(V));
          std::vector<engine::FlashNextGpu::BatchEntry> entries(ids.size());
          for (std::size_t j = 0; j < ids.size(); ++j) {
            entries[j].session = ids[j];
            entries[j].token = token();
            entries[j].logits = outs[j];
          }
          ASSERT_TRUE(gpu->StepBatch(entries, &error)) << error;
          ASSERT_EQ(gpu->BoundSession(), cur);
          for (std::size_t j = 0; j < ids.size(); ++j) {
            hist[ids[j]].push_back({false, {entries[j].token}});
            ++pos[ids[j]];
            ASSERT_EQ(gpu->SessionPositionOf(ids[j]), pos[ids[j]]);
            const auto want = Replay(*oracle, hist[ids[j]], V);
            ASSERT_EQ(std::memcmp(outs[j].data(), want.data(), V * 4), 0)
                << "batch of " << ids.size() << ": entry " << j << " session " << ids[j] << " after "
                << hist[ids[j]].size() << " ops";
          }
          break;
        }
        case kKinds: break;
      }
    }
  }, {.test_cases = 60});
}

// Directed: a fork shares the parent's KV pages below the fork point; both then diverge for long enough to
// write past the shared region, and both must still equal their own replays. With a long context the
// shared pages are real (several positions per page).
TEST(Sessions, ForkedSessionsDivergeIndependently) {
  Small* s = LoadSmall();
  if (!s) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  const std::size_t V = s->weights->config.vocab_size;
  constexpr std::uint32_t kMaxContext = 1024;
  for (auto& dev : devices) {
    std::string error;
    auto gpu = engine::FlashNextGpu::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
    ASSERT_TRUE(gpu) << error;
    auto oracle = engine::FlashNextGpu::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
    ASSERT_TRUE(oracle) << error;
    std::vector<Op> parent;
    std::vector<float> got(V);
    std::uint32_t seed = 7;
    const auto next = [&] { return static_cast<std::int32_t>((seed = seed * 1103515245u + 12345u) % V); };
    for (int chunk = 0; chunk < 3; ++chunk) {  // 3 x 100 tokens prefilled
      Op op{true, std::vector<std::int32_t>(100)};
      for (auto& t : op.tokens) t = next();
      ASSERT_TRUE(gpu->Prefill(op.tokens, got, &error)) << error;
      parent.push_back(op);
    }
    engine::FlashNextGpu::SessionId child;
    ASSERT_TRUE(gpu->ForkSession(0, &child, &error)) << error;
    std::vector<Op> ch = parent;
    for (int round = 0; round < 3; ++round)
      for (auto [id, h] : {std::pair{engine::FlashNextGpu::SessionId{0}, &parent}, std::pair{child, &ch}}) {
        ASSERT_TRUE(gpu->BindSession(id, &error)) << error;
        Op op{true, std::vector<std::int32_t>(150)};
        for (auto& t : op.tokens) t = next();
        ASSERT_TRUE(gpu->Prefill(op.tokens, {}, &error)) << error;
        h->push_back(op);
        const auto t = next();
        ASSERT_TRUE(gpu->Step(t, got, &error)) << error;
        h->push_back({false, {t}});
        const auto want = Replay(*oracle, *h, V);
        ASSERT_EQ(std::memcmp(got.data(), want.data(), V * 4), 0) << "session " << id << " round " << round;
      }
    const auto m = gpu->SessionMemory(child);
    std::printf("%s: page %zu B, child recurrent %zu, kv mapped %zu, exclusive %zu\n", rt::BackendName(dev->backend()),
                dev->PageBytes(), m.recurrent, m.kv_mapped, m.kv_exclusive);
  }
}

// Directed: B sessions with different histories stepped together for many steps, every B from 2 to 8, logits
// and device sampling; each entry against its own session stepped alone on a second engine (same ops).
TEST(Sessions, BatchedStepsEqualSingleSteps) {
  Small* s = LoadSmall();
  if (!s) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  const std::size_t V = s->weights->config.vocab_size;
  constexpr std::uint32_t kMaxContext = 128;
  auto& dev = *devices[0];
  std::string error;
  auto gpu = engine::FlashNextGpu::Create(dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
  ASSERT_TRUE(gpu) << error;
  auto solo = engine::FlashNextGpu::Create(dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
  ASSERT_TRUE(solo) << error;
  ASSERT_TRUE(gpu->BatchSupported()) << "the small model must take the batched kernels: " << gpu->BatchUnsupported();
  std::uint32_t seed = 11;
  const auto next = [&] { return static_cast<std::int32_t>((seed = seed * 1103515245u + 12345u) % V); };
  for (std::uint32_t B = 2; B <= engine::FlashNextGpu::kMaxBatch; ++B)
    for (const bool sample : {false, true}) {
      // Sessions with prompts of different lengths, the same on both engines.
      std::vector<engine::FlashNextGpu::SessionId> ids(B), sids(B);
      for (std::uint32_t b = 0; b < B; ++b) {
        ASSERT_TRUE(gpu->NewSession(&ids[b], &error)) << error;
        ASSERT_TRUE(solo->NewSession(&sids[b], &error)) << error;
        std::vector<std::int32_t> prompt(3 + 5 * b);
        for (auto& t : prompt) t = next();
        ASSERT_TRUE(gpu->BindSession(ids[b], &error) && gpu->Prefill(prompt, {}, &error)) << error;
        ASSERT_TRUE(solo->BindSession(sids[b], &error) && solo->Prefill(prompt, {}, &error)) << error;
      }
      sampling::SamplingConfig cfg;
      cfg.temperature = 0.8f;
      cfg.top_p = 0.9f;
      std::vector<std::int32_t> tok(B);
      std::vector<std::uint64_t> rng(B);
      for (std::uint32_t b = 0; b < B; ++b) tok[b] = next(), rng[b] = 1000 + b;
      for (int step = 0; step < 12; ++step) {
        std::vector<std::vector<float>> outs(B, std::vector<float>(V));
        std::vector<engine::FlashNextGpu::SampleRequest> reqs(B);
        std::vector<engine::FlashNextGpu::SampleResult> res(B);
        std::vector<engine::FlashNextGpu::BatchEntry> entries(B);
        for (std::uint32_t b = 0; b < B; ++b) {
          entries[b].session = ids[b];
          entries[b].token = tok[b];
          if (sample) {
            reqs[b] = {&cfg, {}, rng[b]};
            entries[b].sample = &reqs[b];
            entries[b].result = &res[b];
          } else {
            entries[b].logits = outs[b];
          }
        }
        ASSERT_TRUE(gpu->StepBatch(entries, &error)) << error;
        for (std::uint32_t b = 0; b < B; ++b) {
          ASSERT_TRUE(solo->BindSession(sids[b], &error)) << error;
          if (sample) {
            engine::FlashNextGpu::SampleResult want;
            ASSERT_TRUE(solo->StepSample(tok[b], {&cfg, {}, rng[b]}, &want, &error)) << error;
            ASSERT_EQ(res[b].token, want.token) << "B " << B << " entry " << b << " step " << step;
            ASSERT_EQ(res[b].rng, want.rng);
            tok[b] = res[b].token;
            rng[b] = res[b].rng;
          } else {
            std::vector<float> want(V);
            ASSERT_TRUE(solo->Step(tok[b], want, &error)) << error;
            ASSERT_EQ(std::memcmp(outs[b].data(), want.data(), V * 4), 0)
                << "B " << B << " entry " << b << " step " << step;
            tok[b] = next();
          }
        }
      }
      for (std::uint32_t b = 0; b < B; ++b) {
        ASSERT_TRUE(gpu->BindSession(0, &error) && gpu->DropSession(ids[b], &error)) << error;
        ASSERT_TRUE(solo->BindSession(0, &error) && solo->DropSession(sids[b], &error)) << error;
      }
    }
}
