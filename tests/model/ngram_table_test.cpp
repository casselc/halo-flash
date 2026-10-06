// NgramTable: the PLE row reader (O_DIRECT reads, a lossy row cache, a worker
// pool, or the whole table resident in memory). A Hegel state machine drives random gathers against a file of known
// rows and checks every result against a direct decode of the same bytes.

#include <hegel/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "core/quant.hpp"
#include "model/ngram.hpp"

namespace gs = hegel::generators;
using halo::core::GgmlType;
using halo::model::NgramTable;

namespace {

struct TableFile {
  std::string path;
  int fd = -1;
  std::uint64_t offset = 0;
  std::uint64_t rows = 0;
  std::uint32_t dim = 0;
  GgmlType type = GgmlType::kBF16;
  std::vector<std::uint8_t> bytes;  // the whole file

  ~TableFile() {
    if (fd >= 0) ::close(fd);
    if (!path.empty()) ::unlink(path.c_str());
  }
  std::size_t RowBytes() const {
    return type == GgmlType::kF32 ? dim * 4 : type == GgmlType::kBF16 ? dim * 2 : dim / 32 * 18;
  }
  std::vector<float> Expected(std::uint64_t row) const {
    std::vector<float> out(dim);
    const auto* src = bytes.data() + offset + row * RowBytes();
    EXPECT_TRUE(halo::quant::DequantizeRow(type, src, out.data(), dim));
    return out;
  }
};

std::unique_ptr<TableFile> MakeTable(hegel::TestCase& tc) {
  auto t = std::make_unique<TableFile>();
  t->type = tc.draw(gs::sampled_from({GgmlType::kBF16, GgmlType::kF32, GgmlType::kIQ4_NL}));
  t->dim = t->type == GgmlType::kIQ4_NL ? 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 5}))
                                        : tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 200}));
  t->rows = tc.draw(gs::integers<std::uint64_t>({.min_value = 1, .max_value = 3000}));
  // An unaligned table start, as inside a GGUF, exercises O_DIRECT windows.
  t->offset = tc.draw(gs::integers<std::uint64_t>({.min_value = 0, .max_value = 9000}));
  t->bytes.resize(t->offset + t->rows * t->RowBytes());
  std::uint32_t x = tc.draw(gs::integers<std::uint32_t>());
  for (auto& b : t->bytes) {
    x = x * 1664525u + 1013904223u;
    b = static_cast<std::uint8_t>(x >> 24);
  }
  // Keep IQ4_NL scales finite.
  if (t->type == GgmlType::kIQ4_NL) {
    for (std::uint64_t r = 0; r < t->rows; ++r)
      for (std::uint32_t b = 0; b < t->dim / 32; ++b) {
        auto* d = t->bytes.data() + t->offset + r * t->RowBytes() + b * 18;
        d[1] &= 0x3B;  // exponent well below inf/nan
      }
  } else if (t->type == GgmlType::kBF16) {
    for (std::size_t i = t->offset + 1; i < t->bytes.size(); i += 2) t->bytes[i] &= 0xBF;
  } else {
    for (std::size_t i = t->offset + 3; i < t->bytes.size(); i += 4) t->bytes[i] &= 0xBF;
  }
  // TMPDIR picks the filesystem: tmpfs exercises the buffered fallback, a disk
  // filesystem the O_DIRECT path.
  std::string tmpl = (std::filesystem::temp_directory_path() / "halo-ngram-XXXXXX").string();
  t->fd = ::mkstemp(tmpl.data());
  EXPECT_GE(t->fd, 0);
  t->path = tmpl;
  EXPECT_EQ(::write(t->fd, t->bytes.data(), t->bytes.size()), static_cast<ssize_t>(t->bytes.size()));
  return t;
}

}  // namespace

TEST(NgramTable, GathersMatchDirectDecode) {
  hegel::test([](hegel::TestCase& tc) {
    auto file = MakeTable(tc);
    std::string error;
    // Cache: off, smaller than the table (collisions) or larger (every row fits).
    const auto cache_kind = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 2}));
    const std::size_t cache_bytes = cache_kind == 0   ? 0
                                    : cache_kind == 1 ? NgramTable::kDefaultCacheBytes / 4096
                                                      : 2 * file->rows * file->RowBytes() + 4096;
    auto table = NgramTable::Open(file->fd, file->offset, file->rows, file->dim, file->type, &error, cache_bytes);
    ASSERT_TRUE(table) << error;
    const auto steps = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 12}));
    // From a drawn step on (or never), the table is resident in memory.
    const auto resident_at = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 12}));
    for (int s = 0; s < steps; ++s) {
      if (s == resident_at) {
        ASSERT_TRUE(table->LoadResident(&error)) << error;
        ASSERT_TRUE(table->Resident());
      }
      // A gather: small (decode-like, cache path) or large (prefill-like,
      // sorted batch path), with deliberate duplicates.
      const bool large = tc.draw(gs::booleans());
      const auto n = large ? tc.draw(gs::integers<std::size_t>({.min_value = 1024, .max_value = 1600}))
                           : tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 32}));
      std::vector<std::uint32_t> rows(n);
      const auto hot = tc.draw(gs::integers<std::uint64_t>({.min_value = 0, .max_value = file->rows - 1}));
      for (auto& r : rows) {
        r = tc.draw(gs::booleans()) ? static_cast<std::uint32_t>(hot)
                                    : tc.draw(gs::integers<std::uint32_t>(
                                          {.min_value = 0, .max_value = static_cast<std::uint32_t>(file->rows - 1)}));
      }
      const bool bad = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 9})) == 0;
      if (bad) rows[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = n - 1}))] =
          static_cast<std::uint32_t>(file->rows + tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 5})));
      std::vector<float> out(n * file->dim, -12345.0F);
      const bool ok = table->Read(rows, out);
      if (bad) {
        ASSERT_FALSE(ok) << "an out-of-range row must fail the gather";
        continue;
      }
      ASSERT_TRUE(ok);
      for (std::size_t i = 0; i < n; ++i) {
        const auto want = file->Expected(rows[i]);
        ASSERT_EQ(std::memcmp(out.data() + i * file->dim, want.data(), file->dim * sizeof(float)), 0)
            << "row " << rows[i] << " slot " << i;
      }
    }
  }, {.test_cases = 60});
}

// A cache larger than the table serves a repeated small gather from memory:
// the second read of the same rows counts only hits, and returns the same values.
TEST(NgramTable, CacheServesRepeatedRows) {
  hegel::test([](hegel::TestCase& tc) {
    auto file = MakeTable(tc);
    std::string error;
    auto table = NgramTable::Open(file->fd, file->offset, file->rows, file->dim, file->type, &error,
                                  4 * file->rows * file->RowBytes() + 4096);
    ASSERT_TRUE(table) << error;
    const auto n = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 32}));
    std::vector<std::uint32_t> rows(n);
    for (auto& r : rows)
      r = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = static_cast<std::uint32_t>(file->rows - 1)}));
    std::vector<float> a(n * file->dim), b(n * file->dim);
    ASSERT_TRUE(table->Read(rows, a));
    const auto before = table->Stats();
    ASSERT_TRUE(table->Read(rows, b));
    const auto after = table->Stats();
    ASSERT_EQ(after.misses, before.misses) << "every row was cached by the first read";
    ASSERT_GT(after.hits, before.hits);
    ASSERT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), 0);
  }, {.test_cases = 40});
}

TEST(NgramTable, RejectsUnsupportedFormats) {
  std::string error;
  EXPECT_FALSE(NgramTable::Open(0, 0, 10, 32, GgmlType::kQ8_0, &error));
  EXPECT_FALSE(NgramTable::Open(0, 0, 10, 33, GgmlType::kIQ4_NL, &error));
}
