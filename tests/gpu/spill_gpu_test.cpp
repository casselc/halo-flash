// SR4: session state spilled to disk and restored (docs/SESSIONS.md section 10), on the small quantized model:
//   HALO_SMALL_QUANT_DIR   holds small-q.gguf
//   HALO_SPILL_TEST_DIR    where spill files go (default: a directory under the system temp directory; set it to
//                          an XFS/ext4 path so the O_DIRECT path runs: tmpfs refuses O_DIRECT and the buffered
//                          fallback runs instead)
// Oracle: logits after spill + restore are memcmp-equal to an engine that ran the same operations without the
// spill (exactness is the contract), and a refused or damaged file never yields a partly restored state.

#include <hegel/gtest.h>

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "engine/flash_next_gpu.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"

namespace gs = hegel::generators;
using namespace halo;
using E = engine::FlashNextGpu;

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

std::string SpillDir() {
  static const std::string dir = [] {
    const char* d = std::getenv("HALO_SPILL_TEST_DIR");
    std::filesystem::path p = d && *d ? std::filesystem::path(d)
                                      : std::filesystem::temp_directory_path() / "halo-spill-test";
    p /= std::to_string(::getpid());
    std::filesystem::create_directories(p);
    return p.string();
  }();
  return dir;
}

// Restores every static the tests change.
struct Statics {
  bool kv16 = E::kv_f16_only_, w8a8 = E::prefill_shexp_w8a8_;
  std::size_t piece = E::spill_piece_bytes_;
  unsigned threads = E::spill_threads_;
  int fault = E::spill_fault_;
  ~Statics() {
    E::kv_f16_only_ = kv16;
    E::prefill_shexp_w8a8_ = w8a8;
    E::spill_piece_bytes_ = piece;
    E::spill_threads_ = threads;
    E::spill_fault_ = fault;
  }
};

struct Op {
  bool prefill = false;
  std::vector<std::int32_t> tokens;
};

bool Apply(E& e, const Op& op, std::span<float> logits, std::string* error) {
  return op.prefill ? e.Prefill(op.tokens, logits, error) : e.Step(op.tokens[0], logits, error);
}

std::vector<std::uint8_t> EncodeOps(const std::vector<Op>& ops) {
  std::vector<std::uint8_t> b;
  const auto put = [&](std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
  };
  put(static_cast<std::uint32_t>(ops.size()));
  for (const auto& o : ops) {
    put(o.prefill ? 1 : 0);
    put(static_cast<std::uint32_t>(o.tokens.size()));
    for (auto t : o.tokens) put(static_cast<std::uint32_t>(t));
  }
  return b;
}

std::vector<Op> DecodeOps(const std::vector<std::uint8_t>& b) {
  std::size_t at = 0;
  const auto get = [&] {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= std::uint32_t{b.at(at++)} << (8 * i);
    return v;
  };
  std::vector<Op> ops(get());
  for (auto& o : ops) {
    o.prefill = get() != 0;
    o.tokens.resize(get());
    for (auto& t : o.tokens) t = static_cast<std::int32_t>(get());
  }
  return ops;
}

std::vector<std::uint8_t> ReadFile(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
void WriteFile(const std::string& p, const std::vector<std::uint8_t>& b) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}

std::uint32_t Crc32cSoft(const std::uint8_t* p, std::size_t n) {
  std::uint32_t c = ~0u;
  for (std::size_t i = 0; i < n; ++i) {
    c ^= p[i];
    for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1u)));
  }
  return ~c;
}
// Rewrites a header field and its checksum (a well-formed file of another identity).
void PatchHeader(std::vector<std::uint8_t>& b, std::size_t off, const void* v, std::size_t n) {
  std::memcpy(b.data() + off, v, n);
  constexpr std::size_t kHdr = 96, kCrc = 92;
  std::memset(b.data() + kCrc, 0, 4);
  const std::uint32_t c = Crc32cSoft(b.data(), kHdr);
  std::memcpy(b.data() + kCrc, &c, 4);
}

}  // namespace

// Spill a session after a random history, restore it into the same session (reset), another session of the same
// engine, or a fresh engine; the next steps' logits equal the uninterrupted engine's bitwise. Both KV kinds, I/O
// pieces from 16 KB (sections cut into many pieces) to 16 MB, 1-4 threads.
TEST(Spill, RoundTripIsBitIdentical) {
  Small* s = LoadSmall();
  if (!s) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << "no HIP device";
  Statics guard;
  const std::size_t V = s->weights->config.vocab_size;
  constexpr std::uint32_t kMaxContext = 512;
  const std::string path = SpillDir() + "/rt.spill";
  hegel::test([&](hegel::TestCase& tc) {
    E::kv_f16_only_ = tc.draw(gs::booleans());
    E::spill_piece_bytes_ = tc.draw(gs::booleans()) ? 16u << 10 : 16u << 20;
    E::spill_threads_ = tc.draw(gs::integers<unsigned>({.min_value = 1, .max_value = 4}));
    auto gpu = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
    ASSERT_TRUE(gpu) << error;
    auto oracle = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
    ASSERT_TRUE(oracle) << error;
    gpu->SetUseGraphs(tc.draw(gs::booleans()));
    const auto token = [&] {
      return tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = static_cast<std::int32_t>(V - 1)}));
    };
    // History: prefill chunks and steps.
    std::vector<Op> ops;
    std::uint32_t pos = 0;
    const int nops = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 5}));
    for (int i = 0; i < nops && pos < 380; ++i) {
      Op o;
      o.prefill = tc.draw(gs::booleans());
      const auto n = o.prefill ? tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 380 - pos})) : 1u;
      for (std::uint32_t j = 0; j < n; ++j) o.tokens.push_back(token());
      pos += n;
      ops.push_back(o);
    }
    for (const auto& o : ops)
      ASSERT_TRUE(Apply(*gpu, o, {}, &error) && Apply(*oracle, o, {}, &error)) << error;
    E::SpillInfo si;
    ASSERT_TRUE(gpu->SpillSession(path, {}, &si, &error)) << error;
    ASSERT_EQ(si.position, pos);
    ASSERT_EQ(std::filesystem::file_size(path), si.file_bytes);
    ASSERT_EQ(std::filesystem::status(path).permissions() & std::filesystem::perms::all,
              std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);
    // Something else happens to the engine in between.
    const int where = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 2}));
    E* target = gpu.get();
    std::unique_ptr<E> fresh;
    if (where == 0) {
      std::vector<std::int32_t> other(tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 100})));
      for (auto& t : other) t = token();
      ASSERT_TRUE(gpu->Reset(&error) && gpu->Prefill(other, {}, &error)) << error;  // overwritten, then restored over
    } else if (where == 1) {
      E::SessionId id;
      ASSERT_TRUE(gpu->NewSession(&id, &error) && gpu->BindSession(id, &error)) << error;
    } else {
      fresh = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
      ASSERT_TRUE(fresh) << error;
      target = fresh.get();
    }
    E::SpillInfo ri;
    ASSERT_TRUE(target->RestoreSession(path, nullptr, &ri, &error)) << error;
    ASSERT_EQ(target->Position(), pos);
    // Then steps and prefill chunks (a prefill reads the fp16 K/V copy, which a restore must re-derive).
    std::vector<float> got(V), want(V);
    const int tail = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 6}));
    for (int j = 0; j < tail && pos + 41 < kMaxContext; ++j) {
      Op o;
      o.prefill = tc.draw(gs::booleans());
      const auto n = o.prefill ? tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 40})) : 1u;
      pos += n;
      for (std::uint32_t k = 0; k < n; ++k) o.tokens.push_back(token());
      ASSERT_TRUE(Apply(*target, o, got, &error) && Apply(*oracle, o, want, &error)) << error;
      ASSERT_EQ(std::memcmp(got.data(), want.data(), V * 4), 0)
          << (o.prefill ? "prefill " : "step ") << j << " after the restore, now at " << pos << " (where " << where
          << ", kv16 " << E::kv_f16_only_ << ")";
    }
    std::filesystem::remove(path);
  }, {.test_cases = 40});
}

// Long context: 3000+ positions (many QSA blocks, the indexer-key window, multi-page KV), both kinds, a chunked
// prefill then a restore into a fresh engine and 8 decode steps; and the drop path (no spill: reset + re-prefill)
// equals the same cold prefill.
TEST(Spill, LongContextRoundTripAndDropPath) {
  Small* s = LoadSmall();
  if (!s) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << "no HIP device";
  Statics guard;
  const std::size_t V = s->weights->config.vocab_size;
  constexpr std::uint32_t kMaxContext = 4096;
  const std::string path = SpillDir() + "/long.spill";
  for (bool kv16 : {false, true}) {
    E::kv_f16_only_ = kv16;
    E::spill_piece_bytes_ = 1u << 20;
    auto gpu = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
    auto oracle = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
    ASSERT_TRUE(gpu && oracle) << error;
    std::vector<std::int32_t> prompt(3333), tail(8);
    std::uint32_t seed = kv16 ? 7 : 3;
    for (auto* v : {&prompt, &tail})
      for (auto& t : *v) {
        seed = seed * 1664525u + 1013904223u;
        t = static_cast<std::int32_t>((seed >> 8) % V);
      }
    std::vector<float> cold(V), got(V), want(V);
    ASSERT_TRUE(gpu->Prefill(prompt, cold, &error) && oracle->Prefill(prompt, {}, &error)) << error;
    ASSERT_TRUE(gpu->SpillSession(path, {}, nullptr, &error)) << error;
    auto fresh = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
    ASSERT_TRUE(fresh && fresh->RestoreSession(path, nullptr, nullptr, &error)) << error;
    for (std::size_t j = 0; j < tail.size(); ++j) {
      ASSERT_TRUE(fresh->Step(tail[j], got, &error) && oracle->Step(tail[j], want, &error)) << error;
      ASSERT_EQ(std::memcmp(got.data(), want.data(), V * 4), 0) << "kv16 " << kv16 << " step " << j;
    }
    // Drop path: the session's state is gone; a re-prefill of the prompt is the cold prefill, bit for bit.
    std::vector<float> again(V);
    ASSERT_TRUE(gpu->Reset(&error) && gpu->Prefill(prompt, again, &error)) << error;
    ASSERT_EQ(std::memcmp(again.data(), cold.data(), V * 4), 0) << "kv16 " << kv16;
    std::filesystem::remove(path);
  }
}

// Every damaged or foreign file is refused. Header-level problems are refused before the session is touched (it
// keeps its state, checked by stepping it against the oracle); payload damage is found while reading, and then the
// session is empty (reset), never partly restored.
TEST(Spill, RefusesDamagedTruncatedAndMismatchedFiles) {
  Small* s = LoadSmall();
  if (!s) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << "no HIP device";
  Statics guard;
  E::spill_piece_bytes_ = 64u << 10;
  const std::size_t V = s->weights->config.vocab_size;
  constexpr std::uint32_t kMaxContext = 512;
  const std::string good = SpillDir() + "/good.spill", bad = SpillDir() + "/bad.spill";
  auto gpu = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
  auto oracle = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
  ASSERT_TRUE(gpu && oracle) << error;
  std::vector<std::int32_t> a(150), b(60);
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<std::int32_t>((i * 7919) % V);
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::int32_t>((i * 104729 + 5) % V);
  ASSERT_TRUE(gpu->Prefill(a, {}, &error)) << error;
  ASSERT_TRUE(gpu->SpillSession(good, {}, nullptr, &error)) << error;
  // The target session holds b; the oracle holds b too, so "unchanged" is checkable.
  ASSERT_TRUE(gpu->Reset(&error) && gpu->Prefill(b, {}, &error) && oracle->Prefill(b, {}, &error)) << error;
  const auto bytes = ReadFile(good);
  ASSERT_GT(bytes.size(), 8192u);
  std::uint64_t header_bytes = 0;
  {
    std::uint32_t hb;
    std::memcpy(&hb, bytes.data() + 12, 4);
    header_bytes = hb;
  }
  int step = 0;
  const auto unchanged = [&](const std::string& what) {
    ASSERT_EQ(gpu->Position(), b.size() + step) << what;
    std::vector<float> got(V), want(V);
    const std::int32_t t = 11 + step++;
    ASSERT_TRUE(gpu->Step(t, got, &error) && oracle->Step(t, want, &error)) << error;
    ASSERT_EQ(std::memcmp(got.data(), want.data(), V * 4), 0) << what << ": the refused restore changed the session";
  };
  struct Case {
    const char* what;
    std::vector<std::uint8_t> file;
    bool payload;  // found while reading: the session is reset
  };
  std::vector<Case> cases;
  const auto with = [&](auto f) {
    auto c = bytes;
    f(c);
    return c;
  };
  cases.push_back({"truncated by one piece", with([&](auto& c) { c.resize(c.size() - 4096); }), false});
  cases.push_back({"truncated to the header", with([&](auto& c) { c.resize(header_bytes); }), false});
  cases.push_back({"empty", {}, false});
  cases.push_back({"extended", with([&](auto& c) { c.resize(c.size() + 4096, 0); }), false});
  cases.push_back({"magic", with([&](auto& c) { c[0] ^= 1; }), false});
  cases.push_back({"header bit flip (position)", with([&](auto& c) { c[40] ^= 1; }), false});
  // A field nothing else validates (draft-block blocks, no draft block in this file): only the header checksum.
  cases.push_back({"header bit flip (unused field)", with([&](auto& c) { c[60] ^= 1; }), false});
  cases.push_back({"meta bit flip (a task's checksum)", with([&](auto& c) { c[200] ^= 0x10; }), false});
  cases.push_back({"payload bit flip (first piece)", with([&](auto& c) { c[header_bytes + 5] ^= 0x40; }), true});
  cases.push_back({"payload bit flip (last piece)", with([&](auto& c) { c[c.size() - 4096 - 100] ^= 1; }), true});
  cases.push_back({"another model", with([&](auto& c) {
                     std::uint64_t fp;
                     std::memcpy(&fp, c.data() + 16, 8);
                     fp ^= 1;
                     PatchHeader(c, 16, &fp, 8);
                   }), false});
  cases.push_back({"other numerics options", with([&](auto& c) {
                     std::uint64_t fp;
                     std::memcpy(&fp, c.data() + 24, 8);
                     fp += 1;
                     PatchHeader(c, 24, &fp, 8);
                   }), false});
  cases.push_back({"other KV kind", with([&](auto& c) {
                     std::uint32_t k;
                     std::memcpy(&k, c.data() + 32, 4);
                     k ^= 1;
                     PatchHeader(c, 32, &k, 4);
                   }), false});
  cases.push_back({"wrong position (well-formed header)", with([&](auto& c) {
                     std::uint32_t p;
                     std::memcpy(&p, c.data() + 40, 4);
                     p += 1;
                     PatchHeader(c, 40, &p, 4);
                   }), false});
  cases.push_back({"version", with([&](auto& c) {
                     std::uint32_t v = 2;
                     PatchHeader(c, 8, &v, 4);
                   }), false});
  for (const auto& c : cases) {
    WriteFile(bad, c.file);
    bool reset = false;
    std::string why;
    ASSERT_FALSE(gpu->RestoreSession(bad, nullptr, nullptr, &why, &reset)) << c.what << ": restored";
    EXPECT_FALSE(why.empty()) << c.what;
    ASSERT_EQ(reset, c.payload) << c.what << ": " << why;
    if (!c.payload) {
      unchanged(std::string(c.what) + " (" + why + ")");
    } else {
      // Reset: equal to a new session; re-prefill b and catch the oracle up (the oracle stepped `step` tokens).
      ASSERT_EQ(gpu->Position(), 0u) << c.what;
      ASSERT_TRUE(gpu->Prefill(b, {}, &error)) << error;
      for (int j = 0; j < step; ++j) ASSERT_TRUE(gpu->Step(11 + j, {}, &error)) << error;
      unchanged(std::string(c.what) + " then re-prefilled");
    }
  }
  // Engines of other options refuse the good file: fp16-only KV, a precision option, a smaller max context.
  {
    E::kv_f16_only_ = true;
    auto other = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
    E::kv_f16_only_ = false;
    ASSERT_TRUE(other) << error;
    EXPECT_FALSE(other->RestoreSession(good, nullptr, nullptr, &error)) << "fp16-only engine took a default-KV file";
    E::prefill_shexp_w8a8_ = !E::prefill_shexp_w8a8_;
    EXPECT_FALSE(gpu->RestoreSession(good, nullptr, nullptr, &error)) << "restored under another precision option";
    E::prefill_shexp_w8a8_ = !E::prefill_shexp_w8a8_;
    unchanged("refused under another option");
    auto small_ctx = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, 128, &error);
    ASSERT_TRUE(small_ctx) << error;
    EXPECT_FALSE(small_ctx->RestoreSession(good, nullptr, nullptr, &error)) << "position past max_context restored";
  }
  // The good file still restores exactly.
  ASSERT_TRUE(gpu->RestoreSession(good, nullptr, nullptr, &error)) << error;
  auto ref = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
  ASSERT_TRUE(ref && ref->Prefill(a, {}, &error)) << error;
  std::vector<float> got(V), want(V);
  ASSERT_TRUE(gpu->Step(3, got, &error) && ref->Step(3, want, &error)) << error;
  ASSERT_EQ(std::memcmp(got.data(), want.data(), V * 4), 0);
  std::filesystem::remove(good);
  std::filesystem::remove(bad);
}

// A crash mid-write (fault 1: half the payload written, no rename; fault 2: written and synced, no rename) leaves
// the old file (still restoring exactly to the old state) or none, plus a temp file that SpillCleanTemps removes.
TEST(Spill, CrashMidWriteLeavesTheOldFileOrNone) {
  Small* s = LoadSmall();
  if (!s) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << "no HIP device";
  Statics guard;
  E::spill_piece_bytes_ = 16u << 10;
  const std::size_t V = s->weights->config.vocab_size;
  const std::string dir = SpillDir() + "/crash";
  std::filesystem::create_directories(dir);
  const std::string path = dir + "/s.spill";
  auto gpu = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, 256, &error);
  ASSERT_TRUE(gpu) << error;
  std::vector<std::int32_t> a(40, 5), more(30, 9);
  for (int fault : {1, 2})
    for (bool had_old : {false, true}) {
      std::filesystem::remove(path);
      ASSERT_TRUE(gpu->Reset(&error) && gpu->Prefill(a, {}, &error)) << error;
      if (had_old) ASSERT_TRUE(gpu->SpillSession(path, {}, nullptr, &error)) << error;
      ASSERT_TRUE(gpu->Prefill(more, {}, &error)) << error;
      E::spill_fault_ = fault;
      EXPECT_FALSE(gpu->SpillSession(path, {}, nullptr, &error));
      E::spill_fault_ = 0;
      std::size_t temps = 0;
      for (const auto& e : std::filesystem::directory_iterator(dir))
        temps += e.path().filename().string().find(".tmp.") != std::string::npos;
      EXPECT_EQ(temps, 1u) << "fault " << fault;
      ASSERT_EQ(std::filesystem::exists(path), had_old) << "fault " << fault;
      if (had_old) {  // the old file is whole: it restores the old state exactly
        auto fresh = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, 256, &error);
        auto ref = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, 256, &error);
        ASSERT_TRUE(fresh && ref && fresh->RestoreSession(path, nullptr, nullptr, &error)) << error;
        ASSERT_TRUE(ref->Prefill(a, {}, &error)) << error;
        std::vector<float> got(V), want(V);
        ASSERT_TRUE(fresh->Step(1, got, &error) && ref->Step(1, want, &error)) << error;
        ASSERT_EQ(std::memcmp(got.data(), want.data(), V * 4), 0);
      }
      E::SpillCleanTemps(dir);
      std::size_t left = 0;
      for (const auto& e : std::filesystem::directory_iterator(dir))
        left += e.path().filename().string().find(".tmp.") != std::string::npos;
      EXPECT_EQ(left, 0u);
    }
  std::filesystem::remove_all(dir);
}

// State machine (Hegel, swarm) over logical sessions: step / prefill / suspend (spill, then drop the engine
// session) / crash while suspending (the spill fails mid-write and the process's copy is lost) / resume (restore;
// on failure re-prefill) / evict (drop, delete the file) / corrupt the file. The caller rule under test: after a
// resume, the state is whatever the file's blob says (the ops it holds), never the caller's own record; a resume
// that fails falls back to re-prefilling the caller's record of what the file was expected to hold... and in every
// case the next step's logits equal a fresh engine replaying exactly the ops the state holds.
TEST(Spill, SuspendResumeEvictCrashMachine) {
  Small* s = LoadSmall();
  if (!s) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  const char* saved_pb = std::getenv("HALO_PAGE_BYTES");
  const std::string saved = saved_pb ? saved_pb : "";
  ::setenv("HALO_PAGE_BYTES", "65536", 1);
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (saved_pb) ::setenv("HALO_PAGE_BYTES", saved.c_str(), 1); else ::unsetenv("HALO_PAGE_BYTES");
  if (!dev) GTEST_SKIP() << "no HIP device";
  Statics guard;
  const std::size_t V = s->weights->config.vocab_size;
  constexpr std::uint32_t kMaxContext = 160;
  const std::string dir = SpillDir() + "/machine";
  hegel::test([&](hegel::TestCase& tc) {
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    E::kv_f16_only_ = tc.draw(gs::booleans());
    E::spill_piece_bytes_ = 16u << 10;
    auto gpu = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
    auto oracle = E::Create(*dev, *s->weights, s->ngram.get(), nullptr, kMaxContext, &error);
    ASSERT_TRUE(gpu && oracle) << error;
    struct Logical {
      std::optional<E::SessionId> live;  // resident engine session
      std::vector<Op> ops;               // what the resident state holds (when live)
      std::optional<std::vector<Op>> on_disk;  // what the file holds, if a complete file exists
      bool damaged = false;                    // a bit of the file was flipped since it was written
      std::uint32_t pos = 0;
    };
    std::vector<Logical> ls(3);
    const auto file = [&](std::size_t i) { return dir + "/s" + std::to_string(i) + ".spill"; };
    for (auto& l : ls) {
      E::SessionId id;
      ASSERT_TRUE(gpu->NewSession(&id, &error)) << error;
      l.live = id;
    }
    std::vector<float> got(V), want(V);
    const auto check = [&](Logical& l, const char* what) {
      // One step of a fixed token, compared with a replay; then the step is part of the history.
      ASSERT_TRUE(gpu->BindSession(*l.live, &error)) << error;
      if (l.pos + 1 >= kMaxContext) return;
      Op st{false, {static_cast<std::int32_t>(l.pos % V)}};
      ASSERT_TRUE(Apply(*gpu, st, got, &error)) << error;
      l.ops.push_back(st);
      ++l.pos;
      ASSERT_TRUE(oracle->Reset(&error)) << error;
      for (std::size_t k = 0; k < l.ops.size(); ++k)
        ASSERT_TRUE(Apply(*oracle, l.ops[k], k + 1 == l.ops.size() ? std::span<float>(want) : std::span<float>{}, &error))
            << error;
      ASSERT_EQ(std::memcmp(got.data(), want.data(), V * 4), 0) << what << " at position " << l.pos;
    };
    const auto positions = [](const std::vector<Op>& ops) {
      std::uint32_t p = 0;
      for (const auto& o : ops) p += static_cast<std::uint32_t>(o.tokens.size());
      return p;
    };
    enum Rule { kStep, kPrefill, kSuspend, kCrash, kResume, kEvict, kCorrupt, kRules };
    bool on[kRules];
    for (bool& e : on) e = tc.draw(gs::booleans());
    on[kSuspend] = on[kResume] = true;
    const int n = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 24}));
    for (int i = 0; i < n; ++i) {
      const auto r = static_cast<Rule>(tc.draw(gs::integers<int>({.min_value = 0, .max_value = kRules - 1})));
      if (!on[r]) continue;
      const auto li = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = ls.size() - 1}));
      Logical& l = ls[li];
      switch (r) {
        case kStep:
          if (l.live) check(l, "step");
          break;
        case kPrefill: {
          if (!l.live || l.pos + 40 >= kMaxContext) break;
          Op o{true, {}};
          const auto k = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 30}));
          for (std::uint32_t j = 0; j < k; ++j)
            o.tokens.push_back(tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = static_cast<std::int32_t>(V - 1)})));
          ASSERT_TRUE(gpu->BindSession(*l.live, &error) && Apply(*gpu, o, {}, &error)) << error;
          l.ops.push_back(o);
          l.pos += k;
          break;
        }
        case kSuspend:
        case kCrash: {
          if (!l.live) break;
          ASSERT_TRUE(gpu->BindSession(*l.live, &error)) << error;
          E::spill_fault_ = r == kCrash ? tc.draw(gs::integers<int>({.min_value = 1, .max_value = 2})) : 0;
          const bool ok = gpu->SpillSession(file(li), EncodeOps(l.ops), nullptr, &error);
          E::spill_fault_ = 0;
          ASSERT_EQ(ok, r == kSuspend) << error;
          if (ok) l.on_disk = l.ops, l.damaged = false;  // done: the file holds exactly this state
          // Suspended (or the process "crashed"): the resident copy goes either way.
          ASSERT_TRUE(gpu->BindSession(0, &error) && gpu->DropSession(*l.live, &error)) << error;
          l.live.reset();
          E::SpillCleanTemps(dir);
          break;
        }
        case kResume: {
          if (l.live) break;
          E::SessionId id;
          ASSERT_TRUE(gpu->NewSession(&id, &error) && gpu->BindSession(id, &error)) << error;
          l.live = id;
          std::vector<std::uint8_t> blob;
          bool reset = false;
          const bool exists = std::filesystem::exists(file(li));
          const bool restored = exists && gpu->RestoreSession(file(li), &blob, nullptr, &error, &reset);
          // A whole, undamaged file of the last done spill always restores (no needless re-prefill).
          ASSERT_TRUE(restored || !exists || l.damaged) << "an intact spill file was refused: " << error;
          if (restored) {
            // The blob says what the state holds; it must be the last completed suspend's.
            l.ops = DecodeOps(blob);
            ASSERT_TRUE(l.on_disk.has_value());
            ASSERT_EQ(EncodeOps(l.ops), EncodeOps(*l.on_disk)) << "a file held a state other than the last done spill";
          } else {
            // Not restorable (none, refused, damaged): the session must be empty; re-prefill what the caller knows
            // was last completed (or nothing).
            ASSERT_EQ(gpu->Position(), 0u);
            l.ops = l.on_disk.value_or(std::vector<Op>{});
            for (const auto& o : l.ops) ASSERT_TRUE(Apply(*gpu, o, {}, &error)) << error;
          }
          l.pos = positions(l.ops);
          ASSERT_EQ(gpu->Position(), l.pos);
          check(l, reset ? "re-prefilled after a damaged file" : "resumed");
          break;
        }
        case kEvict:
          // Dropped for good: the resident state and the file.
          if (l.live) {
            ASSERT_TRUE(gpu->BindSession(0, &error) && gpu->DropSession(*l.live, &error)) << error;
            l.live.reset();
          }
          std::filesystem::remove(file(li));
          l.on_disk.reset();
          l.damaged = false;
          l.ops.clear();
          l.pos = 0;
          break;
        case kCorrupt:
          if (std::filesystem::exists(file(li))) {
            auto b = ReadFile(file(li));
            const auto at = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = b.size() - 1}));
            b[at] ^= static_cast<std::uint8_t>(1u << tc.draw(gs::integers<int>({.min_value = 0, .max_value = 7})));
            WriteFile(file(li), b);
            l.damaged = true;
            // A flip in the padding is harmless; a flip anywhere else is refused (the resume re-prefills).
          }
          break;
        default:
          break;
      }
    }
  }, {.test_cases = 40});
  std::filesystem::remove_all(dir);
}

// Real model with the MTP draft block (HALO_FLASH_NEXT_GGUF, HALO_FLASH_NEXT_MTP): a tracked session spilled and
// restored keeps its draft history: speculative cycles after the restore draft, accept and emit exactly what the
// uninterrupted session does; a short and a long (2,600-token) context.
TEST(MtpSpec, SpillRestoreKeepsDraftHistoryOnRealModel) {
  const char* path = std::getenv("HALO_FLASH_NEXT_GGUF");
  const char* mpath = std::getenv("HALO_FLASH_NEXT_MTP");
  if (!path || !mpath) GTEST_SKIP() << "set HALO_FLASH_NEXT_GGUF and HALO_FLASH_NEXT_MTP";
  std::string error;
  auto reader = core::GgufReader::OpenFile(path, &error);
  ASSERT_TRUE(reader) << error;
  auto weights = model::ModelWeights::Bind(*reader, &error);
  ASSERT_TRUE(weights) << error;
  auto mreader = core::GgufReader::OpenFile(mpath, &error);
  ASSERT_TRUE(mreader) << error;
  auto mtp = model::MtpWeights::Bind(*mreader, weights->config, &error);
  ASSERT_TRUE(mtp) << error;
  const auto& p = weights->ple_table;
  auto ngram = model::NgramTable::Open(reader->GetMappedRegions()[p.shard].file_descriptor, p.file_offset, p.rows,
                                       weights->config.ple_head_dim, p.type, &error);
  ASSERT_TRUE(ngram) << error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << "no GPU backend";
  std::vector<rt::FileSource> sources;
  for (const auto& r : reader->GetMappedRegions()) sources.push_back({r.data, r.size, r.file_descriptor});
  auto gpu = E::Create(*dev, *weights, ngram.get(), nullptr, 4096, &error, sources);
  ASSERT_TRUE(gpu) << error;
  ASSERT_TRUE(gpu->LoadMtp(*mtp, &error)) << error;
  const std::string file = SpillDir() + "/mtp.spill";
  for (std::uint32_t n : {300u, 2600u}) {
    std::vector<std::int32_t> head(n);
    std::uint32_t seed = n;
    for (auto& t : head) {
      seed = seed * 1664525u + 1013904223u;
      t = static_cast<std::int32_t>(1000 + (seed >> 8) % 20000);  // ordinary text-range ids
    }
    ASSERT_TRUE(gpu->Reset(&error) && gpu->MtpTrack(true, &error) && gpu->Prefill(head, {}, &error)) << error;
    E::SpillInfo si;
    ASSERT_TRUE(gpu->SpillSession(file, {}, &si, &error)) << error;
    ASSERT_TRUE(si.mtp) << "the tracked session's draft history was not spilled";
    const auto run = [&](std::vector<std::int32_t>* toks, std::uint32_t* drafted, std::uint32_t* accepted) {
      std::int32_t tk = 1234;
      for (int cyc = 0; cyc < 12; ++cyc) {
        E::SpecResult r;
        ASSERT_TRUE(gpu->SpecStep(tk, 3, nullptr, &r, &error)) << error;
        toks->insert(toks->end(), r.tokens.begin(), r.tokens.end());
        toks->insert(toks->end(), r.drafts.begin(), r.drafts.end());
        *drafted += r.drafted;
        *accepted += r.accepted;
        tk = r.tokens.back();
      }
    };
    std::vector<std::int32_t> a, b;
    std::uint32_t da = 0, aa = 0, db = 0, ab = 0;
    run(&a, &da, &aa);
    // Overwrite the session (and the draft block) with something else, then restore.
    // (longer than n, so every draft-block row the restore must bring back was rewritten)
    ASSERT_TRUE(gpu->Reset(&error) && gpu->MtpTrack(true, &error) &&
                gpu->Prefill(std::vector<std::int32_t>(n + 50, 42), {}, &error))
        << error;
    E::SpillInfo ri;
    ASSERT_TRUE(gpu->RestoreSession(file, nullptr, &ri, &error)) << error;
    ASSERT_TRUE(ri.mtp);
    run(&b, &db, &ab);
    EXPECT_GT(da, 0u) << "no drafting at all: the test would not see the draft history";
    EXPECT_EQ(a, b) << "n " << n;
    EXPECT_EQ(da, db);
    EXPECT_EQ(aa, ab);
    std::printf("mtp spill n=%u: file %.1f MB, spill %.1f ms, restore %.1f ms, drafted %u accepted %u\n", n,
                si.file_bytes / 1e6, si.ms, ri.ms, da, aa);
  }
  std::filesystem::remove(file);
}
