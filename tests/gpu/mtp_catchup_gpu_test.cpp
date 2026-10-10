// The draft block's prompt catch-up (docs/MTP.md "Batched draft prefill"): the three modes (eager full steps,
// eager state-only steps, batched columns) leave bit-identical cache rows, so drafts, acceptance and the emitted
// tokens are identical too, and greedy output equals AR. Kernel properties for the batched path's new kernels;
// an engine property on the small quantized model with a synthetic draft block (the fixture has no MTP sidecar:
// the block reuses an attention layer's weights plus host-made nextn tensors).
//   HALO_SMALL_QUANT_DIR  holds small-q.gguf

#include <hegel/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/sampling.hpp"
#include "engine/flash_next_gpu.hpp"
#include "kernels/core_hsaco.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"

namespace gs = hegel::generators;
using namespace halo;

namespace {

struct Rng {
  std::uint32_t s;
  std::uint32_t U() { return s = s * 1664525u + 1013904223u; }
  float F(float amp) { return amp * (static_cast<float>(U() >> 8) / 16777216.0f - 0.5f); }
};

void ExpectBitEqual(const std::vector<float>& want, const std::vector<float>& got, const std::string& what) {
  ASSERT_EQ(want.size(), got.size()) << what;
  for (std::size_t i = 0; i < want.size(); ++i)
    ASSERT_EQ(std::bit_cast<std::uint32_t>(got[i]), std::bit_cast<std::uint32_t>(want[i])) << what << " at " << i;
}

void ForEachDevice(const std::function<void(rt::Device&, rt::Backend)>& body) {
  std::string error;
  bool any = false;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    any = true;
    body(*dev, backend);
  }
  if (!any) GTEST_SKIP() << "no GPU backend";
}

struct Bufs {
  rt::Device& dev;
  std::vector<rt::Buffer> all;
  ~Bufs() {
    for (auto& b : all) dev.Free(b);
  }
  template <typename T>
  rt::Buffer Up(const std::vector<T>& v) {
    std::string error;
    rt::Buffer b = dev.Alloc(std::max<std::size_t>(v.size() * sizeof(T), 4), &error);
    EXPECT_TRUE(b) << error;
    if (!v.empty()) EXPECT_TRUE(dev.Upload(b, 0, v.data(), v.size() * sizeof(T), &error)) << error;
    all.push_back(b);
    return b;
  }
  rt::Buffer Fill(std::size_t floats, float v) { return Up(std::vector<float>(floats, v)); }
  std::vector<float> Down(const rt::Buffer& b, std::size_t floats, std::size_t offset = 0) {
    std::string error;
    std::vector<float> v(floats);
    EXPECT_TRUE(dev.Download(b, offset * 4, v.data(), floats * 4, &error)) << error;
    return v;
  }
  void Run(const char* name, rt::Dim3 grid, std::initializer_list<rt::Arg> args, std::uint32_t block = 256) {
    std::string error;
    const std::vector<rt::Arg> a(args);
    ASSERT_TRUE(dev.Launch(dev.Kernel(name, &error), grid, {block}, a, &error)) << name << ": " << error;
  }
};

std::uint32_t Blocks(std::size_t n, std::uint32_t per) { return static_cast<std::uint32_t>((n + per - 1) / per); }

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

// A draft block for the small model: an attention layer's weights, host-made nextn norms and eh_proj (Q8_0), and
// optionally the attention mixer's inject as F16 (then HcMix takes its unfused path, as on the real sidecar).
struct DraftBlock {
  std::vector<float> enorm, hnorm, inject32;
  std::vector<std::uint8_t> eh;
  std::vector<std::uint16_t> inject16;
  model::MtpWeights m;
};

std::unique_ptr<DraftBlock> MakeDraftBlock(const model::ModelWeights& w, std::uint32_t seed, bool f16_inject) {
  auto b = std::make_unique<DraftBlock>();
  const auto& c = w.config;
  const std::size_t H = c.hidden_size, D = c.HcDim();
  Rng r{seed};
  std::uint32_t il = 0;
  while (il < c.num_layers && w.layers[il].linear) ++il;
  if (il == c.num_layers) return nullptr;
  b->m.config = c;
  b->m.block = w.layers[il];
  b->m.block.nextn_head = w.layers[il].hc_ffn;
  b->m.block.nextn_head.inject = {};
  b->enorm.resize(H);
  b->hnorm.resize(D);
  for (auto& v : b->enorm) v = 1.0f + r.F(0.5f);
  for (auto& v : b->hnorm) v = 1.0f + r.F(0.5f);
  // eh_proj: H rows of 2H (Q8_0 blocks: fp16 scale, 32 int8).
  const std::size_t blocks = H * 2 * H / 32;
  b->eh.resize(blocks * 34);
  for (std::size_t i = 0; i < blocks; ++i) {
    const auto d = std::bit_cast<std::uint16_t>(static_cast<_Float16>(0.002f + 0.004f * (r.U() >> 8) / 16777216.0f));
    std::memcpy(&b->eh[i * 34], &d, 2);
    for (int j = 0; j < 32; ++j) b->eh[i * 34 + 2 + j] = static_cast<std::uint8_t>(r.U() >> 24);
  }
  const auto ref = [](const void* data, core::GgmlType type, std::uint64_t cols, std::uint64_t rows,
                      std::string_view name) {
    model::TensorRef t;
    t.data = data;
    t.type = type;
    t.cols = cols;
    t.rows = rows;
    t.name = name;
    return t;
  };
  b->m.block.nextn_enorm = ref(b->enorm.data(), core::GgmlType::kF32, H, 1, "test.nextn.enorm");
  b->m.block.nextn_hnorm = ref(b->hnorm.data(), core::GgmlType::kF32, D, 1, "test.nextn.hnorm");
  b->m.block.nextn_eh_proj = ref(b->eh.data(), core::GgmlType::kQ8_0, 2 * H, H, "test.nextn.eh_proj");
  if (f16_inject) {
    const auto& inj = b->m.block.hc_attn.inject;
    const auto* src = static_cast<const float*>(inj.data);
    b->inject16.resize(inj.cols * inj.rows);
    for (std::size_t i = 0; i < b->inject16.size(); ++i)
      b->inject16[i] = std::bit_cast<std::uint16_t>(static_cast<_Float16>(src[i]));
    b->m.block.hc_attn.inject = ref(b->inject16.data(), core::GgmlType::kF16, inj.cols, inj.rows, "test.inject16");
  }
  return b;
}

}  // namespace

// halo_matvec_bf16_dual_y column b == halo_matvec_bf16_dual on column b alone (both matrices, or matrix 2 only).
TEST(MtpCatchUp, Bf16DualColumnsMatchSingle) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const bool both = tc.draw(gs::booleans());
      const std::uint32_t r1 = both ? tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 300})) : 0;
      const std::uint32_t r2 = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 300}));
      const std::uint32_t cols = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 100}));
      const std::uint32_t n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 9}));
      const std::uint32_t xs = cols + 4 * tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 3}));
      const std::uint32_t y1s = r1 + tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 5}));
      const std::uint32_t y2s = r2 + tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 5}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      std::vector<std::uint16_t> w1(std::size_t{std::max(r1, 1u)} * cols), w2(std::size_t{r2} * cols);
      for (auto* w : {&w1, &w2})
        for (auto& v : *w) v = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(r.F(4.0f)) >> 16);
      std::vector<float> x(std::size_t{n} * xs);
      for (auto& v : x) v = r.F(8.0f);
      Bufs b{dev, {}};
      const auto w1b = b.Up(w1), w2b = b.Up(w2), xb = b.Up(x);
      const auto y1 = b.Fill(std::size_t{n} * std::max(y1s, 1u), -7.0f), y2 = b.Fill(std::size_t{n} * y2s, -7.0f);
      const std::uint32_t b1 = Blocks(r1, 8);
      b.Run("halo_matvec_bf16_dual_y", {b1 + Blocks(r2, 8), n},
            {rt::Arg::Buf(w1b), rt::Arg::Buf(y1), rt::Arg::Val(r1), rt::Arg::Buf(w2b), rt::Arg::Buf(y2),
             rt::Arg::Val(r2), rt::Arg::Buf(xb), rt::Arg::Val(cols), rt::Arg::Val(b1), rt::Arg::Val(xs),
             rt::Arg::Val(y1s), rt::Arg::Val(y2s)});
      const std::string be = rt::BackendName(backend);
      for (std::uint32_t col = 0; col < n; ++col) {
        const auto xc = b.Up(std::vector<float>(x.begin() + std::size_t{col} * xs, x.begin() + std::size_t{col} * xs + cols));
        const auto z1 = b.Fill(std::max(r1, 1u), 0.0f), z2 = b.Fill(r2, 0.0f);
        b.Run("halo_matvec_bf16_dual", {b1 + Blocks(r2, 8)},
              {rt::Arg::Buf(w1b), rt::Arg::Buf(z1), rt::Arg::Val(r1), rt::Arg::Buf(w2b), rt::Arg::Buf(z2),
               rt::Arg::Val(r2), rt::Arg::Buf(xc), rt::Arg::Val(cols), rt::Arg::Val(b1)});
        if (r1) ExpectBitEqual(b.Down(z1, r1), b.Down(y1, r1, std::size_t{col} * y1s), be + " matrix 1 column " + std::to_string(col));
        ExpectBitEqual(b.Down(z2, r2), b.Down(y2, r2, std::size_t{col} * y2s), be + " matrix 2 column " + std::to_string(col));
      }
    }, {.test_cases = 30});
  });
}

// halo_mtp_cat_rows: column t * hc + s = [en[t] ; hn[t][s]], exactly the vector a draft step builds per stream.
TEST(MtpCatchUp, CatRowsAreTheStepVectors) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t h = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 700}));
      const std::uint32_t hc = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4}));
      const std::uint32_t n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 40}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      std::vector<float> en(std::size_t{n} * h), hn(std::size_t{n} * hc * h), want(std::size_t{n} * hc * 2 * h);
      for (auto& v : en) v = r.F(10.0f);
      for (auto& v : hn) v = r.F(10.0f);
      for (std::uint32_t t = 0; t < n; ++t)
        for (std::uint32_t s = 0; s < hc; ++s)
          for (std::uint32_t i = 0; i < 2 * h; ++i)
            want[(std::size_t{t} * hc + s) * 2 * h + i] =
                i < h ? en[std::size_t{t} * h + i] : hn[(std::size_t{t} * hc + s) * h + i - h];
      Bufs b{dev, {}};
      const auto enb = b.Up(en), hnb = b.Up(hn), out = b.Fill(want.size(), -1.0f);
      b.Run("halo_mtp_cat_rows", {Blocks(2 * h, 256), n * hc},
            {rt::Arg::Buf(enb), rt::Arg::Buf(hnb), rt::Arg::Buf(out), rt::Arg::Val(h), rt::Arg::Val(hc)});
      ExpectBitEqual(want, b.Down(out, want.size()), rt::BackendName(backend));
    }, {.test_cases = 30});
  });
}

// The engine property. Per case: a prompt in 1-3 prefill pieces, a history window start, the batched path's
// sub-chunk size and threshold, the mixer path (fused / unfused), the prefill residual's type, graphs. For each catch-up mode on its own engine:
// the cache rows after the prefill, then greedy SpecStep cycles at drawn depths (the same depths for every mode),
// then the cache rows again. All three modes: identical rows bit for bit, identical drafts, acceptance and emitted
// tokens; the emitted tokens equal greedy AR stepping.
TEST(MtpCatchUp, ModesLeaveIdenticalStateAndDraftsOnSmallModel) {
  Small* s = LoadSmall();
  if (!s) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << "no HIP device: " << error;
  const auto& w = *s->weights;
  constexpr std::uint32_t kMaxContext = 768;
  struct Restore {
    int mode = engine::FlashNextGpu::mtp_catchup_;
    std::uint32_t min = engine::FlashNextGpu::mtp_batch_min_, rows = engine::FlashNextGpu::mtp_batch_rows_;
    int res16 = engine::FlashNextGpu::prefill_res16_;
    ~Restore() {
      engine::FlashNextGpu::prefill_res16_ = res16;
      engine::FlashNextGpu::mtp_catchup_ = mode;
      engine::FlashNextGpu::mtp_batch_min_ = min;
      engine::FlashNextGpu::mtp_batch_rows_ = rows;
    }
  } restore;
  static const sampling::SamplingConfig greedy;
  hegel::test([&](hegel::TestCase& tc) {
    const bool f16_inject = tc.draw(gs::booleans());
    auto block = MakeDraftBlock(w, tc.draw(gs::integers<std::uint32_t>()), f16_inject);
    ASSERT_TRUE(block) << "no attention layer";
    const auto V = static_cast<std::int32_t>(w.config.vocab_size);
    const auto token = [&] { return tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = V - 1})); };
    const std::uint32_t L = tc.draw(gs::integers<std::uint32_t>({.min_value = 2, .max_value = 400}));
    std::vector<std::int32_t> prompt(L);
    for (auto& t : prompt) t = token();
    const auto head = std::span<const std::int32_t>(prompt).first(L - 1);
    std::vector<std::size_t> cuts = {0};
    const std::uint32_t pieces = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3}));
    for (std::uint32_t i = 1; i < pieces; ++i)
      cuts.push_back(tc.draw(gs::integers<std::size_t>({.min_value = cuts.back(), .max_value = head.size()})));
    cuts.push_back(head.size());
    const auto start = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = L - 1}));
    engine::FlashNextGpu::mtp_batch_rows_ = 8 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 40}));
    engine::FlashNextGpu::mtp_batch_min_ = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 12}));
    const bool graphs = tc.draw(gs::booleans());
    // The prefill residual's type (fp32, fp16, bf16): the hidden rows the batched path reads.
    engine::FlashNextGpu::prefill_res16_ = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 2}));
    const std::uint32_t n = tc.draw(gs::integers<std::uint32_t>({.min_value = 4, .max_value = 40}));
    std::vector<std::uint32_t> depths(n);
    for (auto& d : depths) d = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 5}));

    tc.note("L " + std::to_string(L) + " pieces " + std::to_string(pieces) + " start " + std::to_string(start) +
            " rows " + std::to_string(engine::FlashNextGpu::mtp_batch_rows_) + " min " +
            std::to_string(engine::FlashNextGpu::mtp_batch_min_) + " f16_inject " + std::to_string(f16_inject) +
            " res16 " + std::to_string(engine::FlashNextGpu::prefill_res16_) + " graphs " + std::to_string(graphs) +
            " n " + std::to_string(n));
    // Greedy AR reference.
    auto ar = engine::FlashNextGpu::Create(*dev, w, s->ngram.get(), nullptr, kMaxContext, &error);
    ASSERT_TRUE(ar) << error;
    std::vector<std::int32_t> want;
    // The same prefill pieces: a prefill's state depends on how the sequence was chunked.
    for (std::size_t i = 0; i + 1 < cuts.size(); ++i)
      ASSERT_TRUE(ar->Prefill(head.subspan(cuts[i], cuts[i + 1] - cuts[i]), {}, &error)) << error;
    for (std::int32_t t = prompt.back(); want.size() < n;) {
      engine::FlashNextGpu::SampleResult r;
      ASSERT_TRUE(ar->StepSample(t, {&greedy, {}, 0}, &r, &error)) << error;
      want.push_back(t = r.token);
    }
    ar.reset();

    struct Run {
      std::vector<float> after_prefill, after_decode;
      std::vector<std::vector<std::int32_t>> cycles;  // per cycle: drafts, then emitted tokens
      std::vector<std::int32_t> out;
    };
    std::vector<Run> runs;
    for (const int mode : {engine::FlashNextGpu::kMtpCatchUpFull, engine::FlashNextGpu::kMtpCatchUpState,
                           engine::FlashNextGpu::kMtpCatchUpBatched}) {
      engine::FlashNextGpu::mtp_catchup_ = mode;
      auto gpu = engine::FlashNextGpu::Create(*dev, w, s->ngram.get(), nullptr, kMaxContext, &error);
      ASSERT_TRUE(gpu) << error;
      ASSERT_TRUE(gpu->LoadMtp(block->m, &error)) << error;
      if (mode == engine::FlashNextGpu::kMtpCatchUpBatched)
        ASSERT_EQ(gpu->MtpBatchUnsupported(), "") << "the batched path must run on the fixture";
      gpu->SetUseGraphs(graphs);
      ASSERT_TRUE(gpu->MtpTrackFrom(start, &error)) << error;
      for (std::size_t i = 0; i + 1 < cuts.size(); ++i)
        ASSERT_TRUE(gpu->Prefill(head.subspan(cuts[i], cuts[i + 1] - cuts[i]), {}, &error)) << error;
      Run run;
      ASSERT_TRUE(gpu->MtpStateBytes(&run.after_prefill, &error)) << error;
      std::int32_t t = prompt.back();
      for (std::size_t c = 0; run.out.size() < n; ++c) {
        const auto left = static_cast<std::uint32_t>(n - run.out.size());
        engine::FlashNextGpu::SpecResult r;
        ASSERT_TRUE(gpu->SpecStep(t, std::min(depths[c % n], left - 1), nullptr, &r, &error)) << error;
        std::vector<std::int32_t> rec = r.drafts;
        rec.push_back(-1);
        rec.insert(rec.end(), r.tokens.begin(), r.tokens.end());
        run.cycles.push_back(rec);
        for (const auto x : r.tokens) run.out.push_back(x);
        t = r.tokens.back();
      }
      run.out.resize(n);
      ASSERT_TRUE(gpu->MtpStateBytes(&run.after_decode, &error)) << error;
      runs.push_back(std::move(run));
    }
    const char* names[] = {"full", "state", "batched"};
    for (std::size_t m = 0; m < runs.size(); ++m) {
      ASSERT_EQ(runs[m].out, want) << names[m] << ": greedy output differs from AR";
      if (m == 0) continue;
      ExpectBitEqual(runs[0].after_prefill, runs[m].after_prefill, std::string(names[m]) + " cache rows after the prefill");
      ASSERT_EQ(runs[0].cycles, runs[m].cycles) << names[m] << ": drafts / acceptance differ";
      ExpectBitEqual(runs[0].after_decode, runs[m].after_decode, std::string(names[m]) + " cache rows after decoding");
    }
  }, {.test_cases = 40});
}

// The draft block following AR steps (each step's catch-up runs right after the trunk's) never changes the
// trunk: greedy tokens and the final logits equal an engine without the draft block, bit for bit.
TEST(MtpCatchUp, FollowingDraftBlockLeavesTrunkExact) {
  Small* s = LoadSmall();
  if (!s) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << "no HIP device: " << error;
  const auto& w = *s->weights;
  constexpr std::uint32_t kMaxContext = 768;
  struct Restore {
    int mode = engine::FlashNextGpu::mtp_catchup_;
    ~Restore() { engine::FlashNextGpu::mtp_catchup_ = mode; }
  } restore;
  static const sampling::SamplingConfig greedy;
  hegel::test([&](hegel::TestCase& tc) {
    const bool f16_inject = tc.draw(gs::booleans());
    auto block = MakeDraftBlock(w, tc.draw(gs::integers<std::uint32_t>()), f16_inject);
    const auto V = static_cast<std::int32_t>(w.config.vocab_size);
    const std::uint32_t L = tc.draw(gs::integers<std::uint32_t>({.min_value = 2, .max_value = 400}));
    std::vector<std::int32_t> prompt(L);
    for (auto& t : prompt) t = tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = V - 1}));
    const auto head = std::span<const std::int32_t>(prompt).first(L - 1);
    const bool graphs = tc.draw(gs::booleans());
    engine::FlashNextGpu::mtp_catchup_ = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 2}));
    const std::uint32_t n = tc.draw(gs::integers<std::uint32_t>({.min_value = 2, .max_value = 24}));
    tc.note("L " + std::to_string(L) + " graphs " + std::to_string(graphs) + " mode " +
            std::to_string(engine::FlashNextGpu::mtp_catchup_) + " f16_inject " + std::to_string(f16_inject));
    std::vector<std::vector<std::int32_t>> outs;
    std::vector<std::vector<float>> logits;
    for (const bool track : {false, true}) {
      auto gpu = engine::FlashNextGpu::Create(*dev, w, s->ngram.get(), nullptr, kMaxContext, &error);
      ASSERT_TRUE(gpu) << error;
      gpu->SetUseGraphs(graphs);
      if (track) {
        ASSERT_TRUE(gpu->LoadMtp(block->m, &error)) << error;
        ASSERT_TRUE(gpu->MtpTrackFrom(0, &error)) << error;
      }
      ASSERT_TRUE(gpu->Prefill(head, {}, &error)) << error;
      std::vector<std::int32_t> o;
      std::int32_t t = prompt.back();
      while (o.size() < n) {
        engine::FlashNextGpu::SampleResult r;
        ASSERT_TRUE(gpu->StepSample(t, {&greedy, {}, 0}, &r, &error)) << error;
        o.push_back(t = r.token);
      }
      std::vector<float> l(static_cast<std::size_t>(V));
      ASSERT_TRUE(gpu->Step(t, l, &error)) << error;
      outs.push_back(o);
      logits.push_back(std::move(l));
    }
    ASSERT_EQ(outs[0], outs[1]) << "tracking changed the greedy tokens";
    ExpectBitEqual(logits[0], logits[1], "logits after the steps");
  }, {.test_cases = 30});
}
