// Properties of the GGUF reader and writer.
//
// 1. Round trip: anything the writer accepts, the reader returns exactly.
// 2. Hostile input: mutated or truncated images never read out of bounds
//    (enforced by ASan in the cpu-asan preset) and, when accepted, every
//    tensor extent lies inside the buffer.

#include <hegel/gtest.h>

#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include "core/gguf.hpp"
#include "core/gguf_writer.hpp"
#include "core/quant.hpp"

namespace gs = hegel::generators;
using halo::core::GgmlType;
using halo::core::GgufReader;
using halo::core::GgufWriter;

namespace {

struct DrawnTensor {
  std::string name;
  std::vector<std::uint64_t> dims;
  GgmlType type;
  std::vector<std::uint8_t> bytes;
};

DrawnTensor DrawTensor(hegel::TestCase& tc, std::size_t index) {
  DrawnTensor t;
  t.name = "blk." + std::to_string(index) + ".w";
  t.type = tc.draw(gs::sampled_from({GgmlType::kF32, GgmlType::kF16,
                                     GgmlType::kBF16, GgmlType::kQ8_0,
                                     GgmlType::kQ4_K, GgmlType::kQ6_K}));
  const std::size_t block = halo::quant::QuantizedBlockElements(t.type);
  const auto inner_blocks = tc.draw(gs::integers<std::uint64_t>({.min_value = 1, .max_value = 4}));
  const auto rank = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 3}));
  t.dims.push_back(block ? inner_blocks * block : inner_blocks);
  for (int r = 1; r < rank; ++r) {
    t.dims.push_back(tc.draw(gs::integers<std::uint64_t>({.min_value = 1, .max_value = 3})));
  }
  std::uint64_t elements = 1;
  for (auto d : t.dims) elements *= d;
  const auto size = halo::quant::EncodedSizeBytes(t.type, elements);
  // Payload contents are irrelevant to the container; draw a short seed and
  // tile it so shrinking stays cheap.
  auto seed = tc.draw(gs::binary({.min_size = 1, .max_size = 16}));
  t.bytes.resize(size);
  for (std::size_t i = 0; i < size; ++i) t.bytes[i] = seed[i % seed.size()];
  return t;
}

GgufWriter::Value DrawValue(hegel::TestCase& tc) {
  switch (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 7}))) {
    case 0: return tc.draw(gs::integers<std::uint32_t>());
    case 1: return tc.draw(gs::integers<std::int32_t>());
    case 2: return tc.draw(gs::integers<std::uint64_t>());
    case 3: return tc.draw(gs::floats<float>({.allow_nan = false}));
    case 4: return tc.draw(gs::booleans());
    case 5: return tc.draw(gs::text({.max_size = 40}));
    case 6: return tc.draw(gs::vectors(gs::text({.max_size = 8}), {.max_size = 6}));
    default: return tc.draw(gs::vectors(gs::integers<std::int32_t>(), {.max_size = 6}));
  }
}

// Compares a written value with what the reader decoded. The reader widens
// integer scalars/arrays to 64 bits and floats to double.
bool SameValue(const GgufWriter::Value& w, const halo::core::GgufMetadataValue& r) {
  return std::visit(
      [&](const auto& v) -> bool {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::uint64_t>) {
          auto* p = std::get_if<std::uint64_t>(&r.value);
          return p && *p == v;
        } else if constexpr (std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::int64_t>) {
          auto* p = std::get_if<std::int64_t>(&r.value);
          return p && *p == v;
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
          auto* p = std::get_if<double>(&r.value);
          return p && static_cast<T>(*p) == v;
        } else if constexpr (std::is_same_v<T, bool>) {
          auto* p = std::get_if<bool>(&r.value);
          return p && *p == v;
        } else if constexpr (std::is_same_v<T, std::string>) {
          auto* p = std::get_if<std::string_view>(&r.value);
          return p && *p == v;
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          auto* p = std::get_if<std::vector<std::string_view>>(&r.value);
          if (!p || p->size() != v.size()) return false;
          for (std::size_t i = 0; i < v.size(); ++i) if ((*p)[i] != v[i]) return false;
          return true;
        } else if constexpr (std::is_same_v<T, std::vector<std::int32_t>>) {
          auto* p = std::get_if<std::vector<std::int64_t>>(&r.value);
          if (!p || p->size() != v.size()) return false;
          for (std::size_t i = 0; i < v.size(); ++i) if ((*p)[i] != v[i]) return false;
          return true;
        } else {
          return false;  // not drawn by DrawValue
        }
      },
      w);
}

std::vector<std::uint8_t> DrawImage(hegel::TestCase& tc,
                                    std::vector<std::pair<std::string, GgufWriter::Value>>* meta,
                                    std::vector<DrawnTensor>* tensors) {
  const auto alignment = tc.draw(gs::sampled_from<std::uint32_t>({32, 64, 4096}));
  GgufWriter writer(alignment);
  const auto n_meta = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 6}));
  for (std::size_t i = 0; i < n_meta; ++i) {
    meta->emplace_back("k." + std::to_string(i), DrawValue(tc));
    writer.Set(meta->back().first, meta->back().second);
  }
  const auto n_tensors = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 5}));
  for (std::size_t i = 0; i < n_tensors; ++i) {
    tensors->push_back(DrawTensor(tc, i));
    const auto& t = tensors->back();
    writer.AddTensor(t.name, t.dims, t.type, t.bytes);
  }
  std::string error;
  auto image = writer.Serialize(&error);
  EXPECT_FALSE(image.empty()) << error;
  return image;
}

}  // namespace

TEST(Gguf, WriterReaderRoundTrip) {
  hegel::test([](hegel::TestCase& tc) {
    std::vector<std::pair<std::string, GgufWriter::Value>> meta;
    std::vector<DrawnTensor> tensors;
    const auto image = DrawImage(tc, &meta, &tensors);
    std::string error;
    auto reader = GgufReader::OpenMemory(image.data(), image.size(), &error);
    ASSERT_TRUE(reader) << error;
    ASSERT_EQ(reader->GetTensorCount(), tensors.size());
    for (const auto& [key, value] : meta) {
      const auto* got = reader->FindMetadata(key);
      ASSERT_NE(got, nullptr) << key;
      ASSERT_TRUE(SameValue(value, *got)) << key;
    }
    for (const auto& t : tensors) {
      const auto* got = reader->FindTensor(t.name);
      ASSERT_NE(got, nullptr) << t.name;
      ASSERT_EQ(got->type, t.type);
      ASSERT_EQ(got->dimensions, t.dims);
      ASSERT_EQ(got->size_bytes, t.bytes.size());
      ASSERT_EQ(std::memcmp(got->data, t.bytes.data(), t.bytes.size()), 0);
      const auto offset = static_cast<const std::uint8_t*>(got->data) - image.data();
      ASSERT_EQ(offset % reader->GetAlignment(), 0U);
    }
  });
}

TEST(Gguf, HostileBytesNeverEscapeTheBuffer) {
  hegel::test([](hegel::TestCase& tc) {
    std::vector<std::pair<std::string, GgufWriter::Value>> meta;
    std::vector<DrawnTensor> tensors;
    auto image = DrawImage(tc, &meta, &tensors);
    const auto n_mutations = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 8}));
    for (int i = 0; i < n_mutations && !image.empty(); ++i) {
      const auto pos = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = image.size() - 1}));
      image[pos] = tc.draw(gs::integers<std::uint8_t>());
    }
    if (tc.draw(gs::booleans())) {
      image.resize(tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = image.size()})));
    }
    // Exact-size heap copy so ASan sees any read past the end.
    std::vector<std::uint8_t> exact(image.begin(), image.end());
    std::string error;
    auto reader = GgufReader::OpenMemory(exact.data(), exact.size(), &error);
    if (!reader) return;
    const auto* base = exact.data();
    for (const auto& t : reader->GetTensors()) {
      const auto* p = static_cast<const std::uint8_t*>(t.data);
      ASSERT_GE(p, base);
      ASSERT_LE(p + t.size_bytes, base + exact.size());
    }
  }, {.test_cases = 1000});
}

TEST(Gguf, WriterRejectsDuplicatesAndBadSizes) {
  GgufWriter w;
  w.Set("a", std::uint32_t{1});
  w.Set("a", std::uint32_t{2});
  std::string error;
  EXPECT_TRUE(w.Serialize(&error).empty());
  GgufWriter t;
  std::vector<std::uint8_t> bytes(10);
  t.AddTensor("x", {4}, GgmlType::kF32, bytes);
  EXPECT_TRUE(t.Serialize(&error).empty());
}

// Regression: vocab-only GGUFs and the metadata-only first shard of a split
// model end right after the header with no alignment padding.
TEST(Gguf, MetadataOnlyFileNeedsNoPadding) {
  hegel::test([](hegel::TestCase& tc) {
    GgufWriter w;
    const auto tokens = tc.draw(gs::vectors(gs::text({.min_size = 1, .max_size = 6}),
                                            {.min_size = 1, .max_size = 20}));
    w.Set("tokenizer.ggml.tokens", tokens);
    std::string error;
    auto image = w.Serialize(&error);
    ASSERT_FALSE(image.empty()) << error;
    // Strip the writer's trailing zero padding to the exact header length.
    std::size_t header = 4 + 4 + 8 + 8 + 8 + std::string("tokenizer.ggml.tokens").size() + 4 + 4 + 8;
    for (const auto& t : tokens) header += 8 + t.size();
    ASSERT_GE(image.size(), header);
    image.resize(header);
    std::vector<std::uint8_t> exact(image);
    auto reader = GgufReader::OpenMemory(exact.data(), exact.size(), &error);
    ASSERT_TRUE(reader) << error;
    ASSERT_EQ(reader->GetMetadataStringArray("tokenizer.ggml.tokens").size(), tokens.size());
  });
}
