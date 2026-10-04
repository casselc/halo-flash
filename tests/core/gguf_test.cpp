// Properties of the GGUF reader and writer.
//
// 1. Round trip: anything the writer accepts, the reader returns exactly.
// 2. Hostile input: mutated or truncated images never read out of bounds
//    (enforced by ASan in the cpu-asan preset) and, when accepted, every
//    tensor extent lies inside the buffer.

#include <hegel/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
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
    if (!reader) {
      ASSERT_FALSE(error.empty()) << "a rejection must say why";
      return;
    }
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

// ---------------------------------------------------------------------------
// Hand-encoded images. GgufWriter emits only the value types our converters
// need, but files from other writers carry every GGUF type, so these tests
// build the bytes field by field. (Mull found the reader's handling of the
// other types, its typed getters and its rejections all untested.)

namespace {

using halo::core::GgufMetadataValue;
using VT = halo::core::GgufValueType;

struct Raw {
  std::vector<std::uint8_t> b;
  template <class T>
  Raw& Pod(T v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
    return *this;
  }
  Raw& Str(std::string_view s) {
    Pod<std::uint64_t>(s.size());
    b.insert(b.end(), s.begin(), s.end());
    return *this;
  }
  Raw& Bytes(const std::vector<std::uint8_t>& more) {
    b.insert(b.end(), more.begin(), more.end());
    return *this;
  }
};

Raw Header(std::uint64_t tensors, std::uint64_t kvs, std::uint32_t version = 3) {
  Raw r;
  r.b = {'G', 'G', 'U', 'F'};
  r.Pod(version).Pod(tensors).Pod(kvs);
  return r;
}

// What the reader must decode a value to; monostate: skipped by design
// (arrays of 8- and 16-bit integers carry nothing the engine reads).
using Expect = std::variant<std::monostate, std::uint64_t, std::int64_t, double, bool, std::string,
                            std::vector<std::string>, std::vector<std::uint64_t>,
                            std::vector<std::int64_t>, std::vector<double>>;

struct Entry {
  VT type;
  Raw bytes;  // the value's encoding, after its type field
  Expect expect;
};

template <class T, class Wide>
void DrawInt(hegel::TestCase& tc, Entry* e) {
  const auto v = tc.draw(gs::integers<T>());
  e->bytes.Pod(v);
  e->expect = static_cast<Wide>(v);
}

// One item of an array of `t`, appended to `out` (encoding) and `expect`.
void DrawItem(hegel::TestCase& tc, VT t, Raw* out, Expect* expect) {
  auto push = [&](auto v) { std::get<std::vector<decltype(v)>>(*expect).push_back(v); };
  switch (t) {
    case VT::kUint8: out->Pod(tc.draw(gs::integers<std::uint8_t>())); break;
    case VT::kInt8: out->Pod(tc.draw(gs::integers<std::int8_t>())); break;
    case VT::kUint16: out->Pod(tc.draw(gs::integers<std::uint16_t>())); break;
    case VT::kInt16: out->Pod(tc.draw(gs::integers<std::int16_t>())); break;
    case VT::kUint32: { const auto v = tc.draw(gs::integers<std::uint32_t>()); out->Pod(v); push(std::uint64_t{v}); break; }
    case VT::kUint64: { const auto v = tc.draw(gs::integers<std::uint64_t>()); out->Pod(v); push(v); break; }
    case VT::kBool: { const bool v = tc.draw(gs::booleans()); out->Pod<std::uint8_t>(v); push(std::uint64_t{v}); break; }
    case VT::kInt32: { const auto v = tc.draw(gs::integers<std::int32_t>()); out->Pod(v); push(std::int64_t{v}); break; }
    case VT::kInt64: { const auto v = tc.draw(gs::integers<std::int64_t>()); out->Pod(v); push(v); break; }
    case VT::kFloat32: { const auto v = tc.draw(gs::floats<float>({.allow_nan = false})); out->Pod(v); push(double{v}); break; }
    case VT::kFloat64: { const auto v = tc.draw(gs::floats<double>({.allow_nan = false})); out->Pod(v); push(v); break; }
    case VT::kString: { auto v = tc.draw(gs::text({.max_size = 6})); out->Str(v); push(std::move(v)); break; }
    default: break;
  }
}

Entry DrawEntry(hegel::TestCase& tc) {
  // Arrays half the time: each item type needs its own chance to appear.
  const auto type = tc.draw(gs::booleans()) ? 9 : tc.draw(gs::sampled_from({0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12}));
  Entry e{static_cast<VT>(type), {}, {}};
  switch (e.type) {
    case VT::kUint8: DrawInt<std::uint8_t, std::uint64_t>(tc, &e); break;
    case VT::kInt8: DrawInt<std::int8_t, std::int64_t>(tc, &e); break;
    case VT::kUint16: DrawInt<std::uint16_t, std::uint64_t>(tc, &e); break;
    case VT::kInt16: DrawInt<std::int16_t, std::int64_t>(tc, &e); break;
    case VT::kUint32: DrawInt<std::uint32_t, std::uint64_t>(tc, &e); break;
    case VT::kInt32: DrawInt<std::int32_t, std::int64_t>(tc, &e); break;
    case VT::kUint64: DrawInt<std::uint64_t, std::uint64_t>(tc, &e); break;
    case VT::kInt64: DrawInt<std::int64_t, std::int64_t>(tc, &e); break;
    case VT::kFloat32: { const auto v = tc.draw(gs::floats<float>({.allow_nan = false})); e.bytes.Pod(v); e.expect = double{v}; break; }
    case VT::kFloat64: { const auto v = tc.draw(gs::floats<double>({.allow_nan = false})); e.bytes.Pod(v); e.expect = v; break; }
    case VT::kBool: { const bool v = tc.draw(gs::booleans()); e.bytes.Pod<std::uint8_t>(v); e.expect = v; break; }
    case VT::kString: { auto v = tc.draw(gs::text({.max_size = 12})); e.bytes.Str(v); e.expect = std::move(v); break; }
    case VT::kArray: {
      auto item = static_cast<VT>(tc.draw(gs::sampled_from({0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12})));
      const auto n = tc.draw(gs::integers<std::uint64_t>({.min_value = 0, .max_value = 5}));
      e.bytes.Pod(static_cast<std::uint32_t>(item)).Pod(n);
      switch (item) {
        case VT::kString: e.expect = std::vector<std::string>{}; break;
        case VT::kFloat32: case VT::kFloat64: e.expect = std::vector<double>{}; break;
        case VT::kUint32: case VT::kUint64: case VT::kBool: e.expect = std::vector<std::uint64_t>{}; break;
        case VT::kInt32: case VT::kInt64: e.expect = std::vector<std::int64_t>{}; break;
        default: break;
      }
      for (std::uint64_t i = 0; i < n; ++i) {
        if (std::holds_alternative<std::monostate>(e.expect)) {
          Expect ignored = std::vector<std::uint64_t>{};
          DrawItem(tc, item, &e.bytes, &ignored);
        } else {
          DrawItem(tc, item, &e.bytes, &e.expect);
        }
      }
      break;
    }
  }
  return e;
}

bool Same(const Expect& want, const GgufMetadataValue& got) {
  return std::visit(
      [&](const auto& w) -> bool {
        using T = std::decay_t<decltype(w)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return true;
        } else if constexpr (std::is_same_v<T, std::string>) {
          auto* p = std::get_if<std::string_view>(&got.value);
          return p && *p == w;
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          auto* p = std::get_if<std::vector<std::string_view>>(&got.value);
          return p && std::equal(p->begin(), p->end(), w.begin(), w.end());
        } else {
          auto* p = std::get_if<T>(&got.value);
          return p && *p == w;
        }
      },
      want);
}

std::optional<std::uint32_t> WantU32(const Expect& e) {
  if (auto* u = std::get_if<std::uint64_t>(&e); u && *u <= UINT32_MAX) return static_cast<std::uint32_t>(*u);
  if (auto* i = std::get_if<std::int64_t>(&e); i && *i >= 0 && *i <= INT64_C(0xFFFFFFFF))
    return static_cast<std::uint32_t>(*i);
  return std::nullopt;
}

std::optional<std::uint64_t> WantU64(const Expect& e) {
  if (auto* u = std::get_if<std::uint64_t>(&e)) return *u;
  if (auto* i = std::get_if<std::int64_t>(&e); i && *i >= 0) return static_cast<std::uint64_t>(*i);
  return std::nullopt;
}

std::unique_ptr<GgufReader> Open(const std::vector<std::uint8_t>& image, std::string* error) {
  error->clear();
  // Exact-size heap copy so ASan sees any read past the end; it outlives the
  // reader only within one call, which these tests respect.
  static thread_local std::vector<std::uint8_t> exact;
  exact = image;
  return GgufReader::OpenMemory(exact.data(), exact.size(), error);
}

}  // namespace

TEST(Gguf, MetadataOfEveryTypeRoundTrips) {
  hegel::test([](hegel::TestCase& tc) {
    const auto version = tc.draw(gs::sampled_from<std::uint32_t>({2, 3}));
    const auto n = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 8}));
    std::vector<Entry> entries;
    Raw image = Header(0, n, version);
    for (std::size_t i = 0; i < n; ++i) {
      entries.push_back(DrawEntry(tc));
      image.Str("k." + std::to_string(i)).Pod(static_cast<std::uint32_t>(entries.back().type));
      image.Bytes(entries.back().bytes.b);
    }
    std::string error;
    auto reader = Open(image.b, &error);
    ASSERT_TRUE(reader) << error;
    ASSERT_EQ(reader->GetVersion(), version);
    ASSERT_EQ(reader->GetMetadataCount(), n);
    for (std::size_t i = 0; i < n; ++i) {
      const auto key = "k." + std::to_string(i);
      const auto& want = entries[i].expect;
      const auto* got = reader->FindMetadata(key);
      ASSERT_NE(got, nullptr) << key;
      ASSERT_EQ(got->type, entries[i].type) << key;
      ASSERT_TRUE(Same(want, *got)) << key << " type " << static_cast<int>(entries[i].type);
      if (std::holds_alternative<std::monostate>(want)) continue;
      ASSERT_EQ(reader->GetMetadataUint32(key), WantU32(want)) << key;
      ASSERT_EQ(reader->GetMetadataUint64(key), WantU64(want)) << key;
      const auto* d = std::get_if<double>(&want);
      ASSERT_EQ(reader->GetMetadataFloat32(key),
                d ? std::optional<float>(static_cast<float>(*d)) : std::nullopt) << key;
      const auto* b = std::get_if<bool>(&want);
      ASSERT_EQ(reader->GetMetadataBool(key), b ? std::optional<bool>(*b) : std::nullopt) << key;
      const auto* s = std::get_if<std::string>(&want);
      ASSERT_EQ(reader->GetMetadataString(key),
                s ? std::optional<std::string_view>(*s) : std::nullopt) << key;
      const auto* sa = std::get_if<std::vector<std::string>>(&want);
      const auto strings = reader->GetMetadataStringArray(key);
      ASSERT_EQ(strings.size(), sa ? sa->size() : 0U) << key;
    }
    ASSERT_EQ(reader->FindMetadata("absent"), nullptr);
    ASSERT_EQ(reader->GetMetadataUint32("absent"), std::nullopt);
    ASSERT_EQ(reader->GetMetadataUint64("absent"), std::nullopt);
    // The image ends at the last value's last byte: any truncation is caught.
    const auto cut = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = image.b.size()}));
    image.b.resize(image.b.size() - cut);
    ASSERT_FALSE(Open(image.b, &error));
    ASSERT_FALSE(error.empty());
  }, {.test_cases = 300});
}

TEST(Gguf, ElementCountGuardsOverflow) {
  auto count = [](std::vector<std::uint64_t> dims) {
    halo::core::GgufTensorInfo t;
    t.dimensions = std::move(dims);
    return t.ElementCount();
  };
  constexpr auto kMax = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(count({}), 0U);
  EXPECT_EQ(count({4, 0, 2}), 0U);
  EXPECT_EQ(count({kMax / 3, 3}), kMax);  // exactly fits (2^64 - 1 is divisible by 3)
  EXPECT_EQ(count({kMax / 3 + 1, 3}), 0U);
  EXPECT_EQ(count({3, std::uint64_t{1} << 63}), 0U);  // wraps to 2^63 if unchecked
  EXPECT_EQ(count({2, 3, 4}), 24U);
}

TEST(Gguf, SizeLimitsAreExact) {
  std::string error;
  // The smallest GGUF: a 24-byte header with nothing in it.
  ASSERT_EQ(Header(0, 0).b.size(), 24U);
  EXPECT_TRUE(Open(Header(0, 0).b, &error)) << error;
  // Keys up to 65535 bytes.
  for (std::size_t len : {65535, 65536}) {
    Raw r = Header(0, 1);
    r.Str(std::string(len, 'k')).Pod<std::uint32_t>(4).Pod<std::uint32_t>(7);
    EXPECT_EQ(static_cast<bool>(Open(r.b, &error)), len == 65535) << len;
  }
}

namespace {

// A one-tensor image: header, optional extra metadata, tensor info, padding
// to 32 and `data`.
std::vector<std::uint8_t> TensorImage(std::string_view name, const std::vector<std::uint64_t>& dims,
                                      std::uint32_t type, std::uint64_t offset,
                                      std::size_t data_bytes, std::uint32_t n_dims_field = 0) {
  Raw r = Header(1, 0);
  r.Str(name).Pod<std::uint32_t>(n_dims_field ? n_dims_field : static_cast<std::uint32_t>(dims.size()));
  for (auto d : dims) r.Pod(d);
  r.Pod(type).Pod(offset);
  r.b.resize((r.b.size() + 31) / 32 * 32, 0);
  r.b.resize(r.b.size() + data_bytes, 0x5A);
  return r.b;
}

}  // namespace

TEST(Gguf, TensorInfoBoundaries) {
  std::string error;
  // Valid: eight dimensions, and data ending at the image's last byte.
  auto ok = Open(TensorImage("t", {1, 1, 1, 1, 1, 1, 1, 2}, 0, 0, 8), &error);
  ASSERT_TRUE(ok) << error;
  const auto* t = ok->FindTensor("t");
  ASSERT_NE(t, nullptr);
  EXPECT_TRUE(ok->HasTensor("t"));
  EXPECT_FALSE(ok->HasTensor("u"));
  EXPECT_EQ(ok->FindTensor("u"), nullptr);
  EXPECT_EQ(t->size_bytes, 8U);
  EXPECT_EQ(t->ElementCount(), 2U);
  EXPECT_EQ(static_cast<const std::uint8_t*>(t->data)[7], 0x5A);
  EXPECT_EQ(ok->GetAlignment(), 32U);
  // A Q8_0 row of one block at a 32-aligned offset.
  ASSERT_TRUE(Open(TensorImage("q", {32}, 8, 32, 64 + 2), &error)) << error;

  struct Bad {
    const char* what;
    std::vector<std::uint8_t> image;
  };
  const Bad bad[] = {
      {"no dims", TensorImage("t", {}, 0, 0, 4)},
      {"nine dims", TensorImage("t", {1, 1, 1, 1, 1, 1, 1, 1, 1}, 0, 0, 4)},
      {"empty name", TensorImage("", {1}, 0, 0, 4)},
      {"data one byte short", TensorImage("t", {1, 1, 1, 1, 1, 1, 1, 2}, 0, 0, 7)},
      {"row not whole blocks", TensorImage("q", {31}, 8, 0, 64)},
      {"removed ggml type", TensorImage("t", {4}, 4, 0, 64)},
      {"type above uint16", TensorImage("t", {4}, 0x10000, 0, 64)},
      {"misaligned offset", TensorImage("t", {1}, 0, 4, 64)},
      {"offset past the end", TensorImage("t", {1}, 0, 64, 32)},
      {"zero dimension", TensorImage("t", {4, 0}, 0, 0, 64)},
  };
  for (const auto& b : bad) {
    EXPECT_FALSE(Open(b.image, &error)) << b.what;
    EXPECT_FALSE(error.empty()) << b.what;
  }
}

TEST(Gguf, RejectsMalformedHeaders) {
  auto meta = [](std::string_view key, std::uint32_t type, auto&& fill) {
    Raw r = Header(0, 1);
    r.Str(key).Pod(type);
    fill(r);
    return r.b;
  };
  auto two = [](Raw first, std::string_view key2) {
    first.b[16] = 2;  // metadata count
    first.Str(key2).Pod<std::uint32_t>(4).Pod<std::uint32_t>(1);
    return first.b;
  };
  Raw one_u32 = Header(0, 1);
  one_u32.Str("a").Pod<std::uint32_t>(4).Pod<std::uint32_t>(1);
  std::vector<std::uint8_t> bad_magic = Header(0, 0).b;
  bad_magic[0] = 'X';
  std::vector<std::uint8_t> short_header = Header(0, 0).b;
  short_header.pop_back();
  struct Bad {
    const char* what;
    std::vector<std::uint8_t> image;
  };
  const Bad bad[] = {
      {"magic", bad_magic},
      {"23 bytes", short_header},
      {"version 1", Header(0, 0, 1).b},
      {"version 4", Header(0, 0, 4).b},
      {"more entries than bytes", Header(0, 1000).b},
      {"more tensors than bytes", Header(1000, 0).b},
      {"value type 13", meta("a", 13, [](Raw& r) { r.Pod<std::uint32_t>(0); })},
      {"bool 2", meta("a", 7, [](Raw& r) { r.Pod<std::uint8_t>(2); })},
      {"bool array element 2", meta("a", 9, [](Raw& r) { r.Pod<std::uint32_t>(7).Pod<std::uint64_t>(2).Pod<std::uint8_t>(1).Pod<std::uint8_t>(2); })},
      {"nested array", meta("a", 9, [](Raw& r) { r.Pod<std::uint32_t>(9).Pod<std::uint64_t>(0); })},
      {"array item type 13", meta("a", 9, [](Raw& r) { r.Pod<std::uint32_t>(13).Pod<std::uint64_t>(0); })},
      {"array longer than file", meta("a", 9, [](Raw& r) { r.Pod<std::uint32_t>(4).Pod<std::uint64_t>(3).Pod<std::uint32_t>(1).Pod<std::uint32_t>(1); })},
      {"f32 array longer than file", meta("a", 9, [](Raw& r) { r.Pod<std::uint32_t>(6).Pod<std::uint64_t>(2).Pod<float>(1); })},
      {"u16 array longer than file", meta("a", 9, [](Raw& r) { r.Pod<std::uint32_t>(2).Pod<std::uint64_t>(3).Pod<std::uint16_t>(1).Pod<std::uint16_t>(1); })},
      {"empty key", meta("", 4, [](Raw& r) { r.Pod<std::uint32_t>(1); })},
      {"duplicate key", two(one_u32, "a")},
      {"alignment 3", meta("general.alignment", 4, [](Raw& r) { r.Pod<std::uint32_t>(3); })},
      {"alignment 2^32", meta("general.alignment", 10, [](Raw& r) { r.Pod<std::uint64_t>(std::uint64_t{1} << 32); })},
      {"alignment as text", meta("general.alignment", 8, [](Raw& r) { r.Str("32"); })},
  };
  std::string error;
  for (const auto& b : bad) {
    EXPECT_FALSE(Open(b.image, &error)) << b.what;
    EXPECT_FALSE(error.empty()) << b.what;
  }
  // Controls: the same shapes, well formed.
  EXPECT_TRUE(Open(two(one_u32, "b"), &error)) << error;
  EXPECT_TRUE(Open(meta("general.alignment", 4, [](Raw& r) { r.Pod<std::uint32_t>(64); }), &error)) << error;
  EXPECT_TRUE(Open(meta("a", 9, [](Raw& r) { r.Pod<std::uint32_t>(2).Pod<std::uint64_t>(2).Pod<std::uint16_t>(1).Pod<std::uint16_t>(1); }), &error)) << error;
  EXPECT_FALSE(GgufReader::OpenMemory(nullptr, 64, &error));
}

TEST(Gguf, QuantizationLabelWeighsElements) {
  auto label = [](std::vector<std::pair<GgmlType, std::uint64_t>> tensors) {
    GgufWriter w;
    int i = 0;
    for (auto [type, n] : tensors) {
      std::vector<std::uint8_t> bytes(halo::quant::EncodedSizeBytes(type, n));
      w.AddTensor("t" + std::to_string(i++), {n}, type, bytes);
    }
    std::string error;
    const auto image = w.Serialize(&error);
    auto r = GgufReader::OpenMemory(image.data(), image.size(), &error);
    return r ? r->GetQuantizationLabel() : "error: " + error;
  };
  EXPECT_EQ(label({}), "unknown");
  // Exactly four fifths of the elements is one format; any less is a mix.
  EXPECT_EQ(label({{GgmlType::kQ8_0, 4000}, {GgmlType::kF32, 1000}}), "Q8_0");
  EXPECT_EQ(label({{GgmlType::kQ8_0, 3968}, {GgmlType::kF32, 1000}}), "Q8_0 mixed");
  EXPECT_EQ(label({{GgmlType::kF32, 10}, {GgmlType::kQ8_0, 32}, {GgmlType::kF32, 30}}), "F32 mixed");
}

TEST(Gguf, MtpAndVisionTensorNames) {
  auto open = [](std::vector<std::string> names) {
    GgufWriter w;
    for (const auto& n : names) {
      std::vector<std::uint8_t> bytes(4);
      w.AddTensor(n, {1}, GgmlType::kF32, bytes);
    }
    std::string error;
    static std::vector<std::uint8_t> image;
    image = w.Serialize(&error);
    return GgufReader::OpenMemory(image.data(), image.size(), &error);
  };
  for (const char* n : {"mtp.w", "blk.mtp.w", "blk.0.nextn.w"}) EXPECT_TRUE(open({"a", n})->HasMtpTensors()) << n;
  for (const char* n : {"blk.0.w", "mtpx.w", "blk.nextn"}) EXPECT_FALSE(open({n})->HasMtpTensors()) << n;
  for (const char* n : {"model.visual.w", "visual.w", "v.w"}) EXPECT_TRUE(open({"a", n})->HasVisionTensors()) << n;
  for (const char* n : {"blk.v.w", "vv.w", "model.vis"}) EXPECT_FALSE(open({n})->HasVisionTensors()) << n;
}

// ---------------------------------------------------------------------------
// Split models: <name>-0000k-of-0000n.gguf shards opened as one reader.

namespace {

struct TempDir {
  std::filesystem::path path;
  TempDir() {
    static int counter = 0;
    path = std::filesystem::temp_directory_path() /
           ("halo-gguf-test-" + std::to_string(::getpid()) + "-" + std::to_string(counter++));
    std::filesystem::create_directories(path);
  }
  ~TempDir() { std::filesystem::remove_all(path); }
};

std::filesystem::path ShardPath(const TempDir& d, std::uint32_t i, std::uint32_t n) {
  char name[64];
  std::snprintf(name, sizeof(name), "m-%05u-of-%05u.gguf", i + 1, n);
  return d.path / name;
}

}  // namespace

TEST(Gguf, SplitShardsOpenAsOneModel) {
  hegel::test([](hegel::TestCase& tc) {
    const auto n = tc.draw(gs::integers<std::uint32_t>({.min_value = 2, .max_value = 4}));
    const auto opened = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = n - 1}));
    // 0: well formed. Otherwise one defect, in shard `victim`, to reject:
    // 1 file missing, 2 wrong split.no, 3 wrong split.count, 4 wrong
    // split.tensors.count, 5 another architecture, 6 a tensor name used by
    // two shards, 7 a file name without the -0000k-of-0000n pattern.
    const auto defect = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 7}));
    auto victim = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = n - 1}));
    if (defect >= 1 && defect <= 3 && victim == opened) victim = (opened + 1) % n;
    std::vector<std::uint32_t> per_shard(n);
    std::uint64_t total = 0;
    for (auto& c : per_shard) total += (c = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 2})));
    if (defect == 6) total += 2;
    TempDir dir;
    std::vector<std::vector<std::uint8_t>> payloads;
    for (std::uint32_t i = 0; i < n; ++i) {
      GgufWriter w;
      const bool bad = i == victim;
      w.Set("split.no", std::uint32_t{defect == 2 && bad ? (i + 1) % n : i});
      w.Set("split.count", std::uint32_t{defect == 3 && bad ? n + 1 : n});
      w.Set("split.tensors.count", std::uint64_t{defect == 4 ? total + 1 : total});
      w.Set("general.architecture", std::string(defect == 5 && bad ? "other" : "qwen"));
      w.Set("shard.mark", std::uint32_t{i});
      for (std::uint32_t t = 0; t < per_shard[i]; ++t) {
        payloads.emplace_back(8, static_cast<std::uint8_t>(16 * i + t));
        w.AddTensor("s" + std::to_string(i) + ".t" + std::to_string(t), {2}, GgmlType::kF32,
                    payloads.back());
      }
      if (defect == 6 && (bad || i == (victim + 1) % n)) {
        std::vector<std::uint8_t> bytes(8, 0xEE);
        w.AddTensor("dup", {2}, GgmlType::kF32, bytes);
      }
      std::string error;
      ASSERT_TRUE(w.WriteFile(ShardPath(dir, i, n), &error)) << error;
    }
    auto path = ShardPath(dir, opened, n);
    if (defect == 1) std::filesystem::remove(ShardPath(dir, victim, n));
    if (defect == 7) {
      std::filesystem::rename(path, dir.path / "model.gguf");
      path = dir.path / "model.gguf";
    }
    std::string error;
    auto reader = GgufReader::OpenFile(path, &error);
    if (defect != 0) {
      ASSERT_FALSE(reader) << "defect " << defect;
      ASSERT_FALSE(error.empty()) << "defect " << defect;
      return;
    }
    ASSERT_TRUE(reader) << error;
    ASSERT_EQ(reader->GetTensorCount(), total);
    ASSERT_EQ(reader->GetMappedRegions().size(), n);
    ASSERT_EQ(reader->GetMetadataUint32("shard.mark"), 0U);  // metadata comes from shard 1
    std::size_t k = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
      for (std::uint32_t t = 0; t < per_shard[i]; ++t, ++k) {
        const auto* got = reader->FindTensor("s" + std::to_string(i) + ".t" + std::to_string(t));
        ASSERT_NE(got, nullptr);
        ASSERT_EQ(std::memcmp(got->data, payloads[k].data(), 8), 0);
      }
    }
  }, {.test_cases = 60});
}

TEST(Gguf, SplitCountIsValidated) {
  TempDir dir;
  auto write = [&](const char* name, auto count) {
    GgufWriter w;
    w.Set("split.count", count);
    std::string error;
    EXPECT_TRUE(w.WriteFile(dir.path / name, &error)) << error;
    return dir.path / name;
  };
  std::string error;
  EXPECT_TRUE(GgufReader::OpenFile(write("one.gguf", std::uint32_t{1}), &error)) << error;
  for (auto p : {write("zero.gguf", std::uint32_t{0}), write("big.gguf", std::uint32_t{65536}),
                 write("text.gguf", std::string("2")), write("neg.gguf", std::int32_t{-2})}) {
    EXPECT_FALSE(GgufReader::OpenFile(p, &error)) << p;
    EXPECT_FALSE(error.empty()) << p;
  }
  // 65535 shards is allowed, so the reader goes on to the split metadata
  // (absent here) instead of rejecting the count.
  error.clear();
  EXPECT_FALSE(GgufReader::OpenFile(write("m-00001-of-65535.gguf", std::uint32_t{65535}), &error));
  EXPECT_NE(error.find("Invalid split GGUF metadata"), std::string::npos) << error;
  // split.no must be below split.count.
  {
    GgufWriter w;
    w.Set("split.count", std::uint32_t{2});
    w.Set("split.no", std::uint32_t{2});
    w.Set("split.tensors.count", std::uint64_t{0});
    ASSERT_TRUE(w.WriteFile(dir.path / "s-00001-of-00002.gguf", &error)) << error;
    ASSERT_TRUE(w.WriteFile(dir.path / "s-00002-of-00002.gguf", &error)) << error;
    error.clear();
    EXPECT_FALSE(GgufReader::OpenFile(dir.path / "s-00001-of-00002.gguf", &error));
    EXPECT_NE(error.find("Invalid split GGUF metadata"), std::string::npos) << error;
  }
  EXPECT_FALSE(GgufReader::OpenFile(dir.path / "missing.gguf", &error));
  EXPECT_FALSE(error.empty());
  std::filesystem::path empty = dir.path / "empty.gguf";
  std::fclose(std::fopen(empty.c_str(), "w"));
  EXPECT_FALSE(GgufReader::OpenFile(empty, &error));
  EXPECT_FALSE(error.empty());
}
