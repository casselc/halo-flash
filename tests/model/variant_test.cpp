// Variant manifests and the registry (PLAN.md P10): round trips, refusal of
// a variant built for another base, tampered or missing parts, unsafe
// paths, unknown format versions, the "base:variant" alias, and the C ABI.

#include <gtest/gtest.h>
#include <hegel/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/gguf_writer.hpp"
#include "core/sha256.hpp"
#include "halo_variant.h"
#include "model/variant.hpp"

using namespace halo;
namespace gs = hegel::generators;
namespace fs = std::filesystem;

namespace {

std::span<const std::uint8_t> Bytes(const std::vector<float>& v) {
  return {reinterpret_cast<const std::uint8_t*>(v.data()), v.size() * sizeof(float)};
}

struct Dir {
  fs::path path;
  explicit Dir(const std::string& name) : path(fs::path(testing::TempDir()) / ("halo_variant_" + name)) {
    fs::remove_all(path);
    fs::create_directories(path);
  }
  ~Dir() { fs::remove_all(path); }
  [[nodiscard]] std::string operator/(const std::string& f) const { return (path / f).string(); }
};

// A minimal qwen35 "base": one 6x4 projection.
std::string WriteBase(const Dir& d, float seed) {
  core::GgufWriter w;
  w.Set("general.architecture", std::string("qwen35"));
  std::vector<float> W(24);
  for (std::size_t i = 0; i < W.size(); ++i) W[i] = seed + 0.1F * static_cast<float>(i);
  w.AddTensor("blk.0.attn_q.weight", {4, 6}, core::GgmlType::kF32, Bytes(W));
  std::string error;
  const std::string path = d / ("base" + std::to_string(static_cast<int>(seed)) + ".gguf");
  EXPECT_TRUE(w.WriteFile(path, &error)) << error;
  return path;
}

std::string WriteLora(const Dir& d, const std::string& name, std::uint64_t rank = 2) {
  core::GgufWriter w;
  w.Set("general.architecture", std::string("qwen35"));
  w.Set("general.type", std::string("adapter"));
  w.Set("adapter.type", std::string("lora"));
  w.Set("adapter.lora.alpha", 4.0F);
  std::vector<float> A(4 * rank, 0.5F), B(rank * 6, -0.25F);
  w.AddTensor("blk.0.attn_q.weight.lora_a", {4, rank}, core::GgmlType::kF32, Bytes(A));
  w.AddTensor("blk.0.attn_q.weight.lora_b", {rank, 6}, core::GgmlType::kF32, Bytes(B));
  std::string error;
  const std::string path = d / name;
  EXPECT_TRUE(w.WriteFile(path, &error)) << error;
  return path;
}

model::VariantManifest Manifest(const std::string& name, const std::string& base_hash, const std::string& lora_path) {
  model::VariantManifest m;
  m.name = name;
  m.version = 2;
  m.base_architecture = "qwen35";
  m.base_name = "tiny";
  m.base_sha256 = base_hash;
  std::string error;
  m.parts.push_back({model::VariantPartKind::kLora, fs::path(lora_path).filename().string(),
                     *core::Sha256File(lora_path, &error), fs::file_size(lora_path)});
  m.provenance.set("trainer", json::Value::String("test"));
  m.eval.push_back({"held-out", "accuracy", std::string(64, 'a'), 0.9, 0.5, 100, 7});
  return m;
}

}  // namespace

TEST(Variant, ManifestJsonRoundTrips) {
  hegel::test([](hegel::TestCase& tc) {
    model::VariantManifest m;
    m.name = tc.draw(gs::sampled_from<std::string>({"a", "sentiment-ab", "v1.2_x", "Z"}));
    m.version = tc.draw(gs::integers<std::int64_t>({.min_value = 1, .max_value = 1000000}));
    m.base_architecture = "qwen35";
    m.base_sha256 = std::string(64, tc.draw(gs::sampled_from<char>({'0', 'f', '9'})));
    const auto nparts = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 2}));
    m.parts.push_back({model::VariantPartKind::kLora, "lora.gguf", std::string(64, 'b'),
                       tc.draw(gs::integers<std::uint64_t>({.max_value = 1ull << 40}))});
    if (nparts == 2)
      m.parts.push_back({tc.draw(gs::booleans()) ? model::VariantPartKind::kClassifierHead
                                                 : model::VariantPartKind::kPointerHead,
                         "sub/head.gguf", std::string(64, 'c'), 10});
    const auto nev = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}));
    for (int i = 0; i < nev; ++i) {
      model::VariantEval e;
      e.name = "e" + std::to_string(i);
      e.metric = "accuracy";
      e.value = tc.draw(gs::floats<double>({.min_value = -10, .max_value = 10}));
      if (tc.draw(gs::booleans())) e.baseline = 0.25;
      e.n = tc.draw(gs::integers<std::uint64_t>({.max_value = 100000}));
      e.step = tc.draw(gs::integers<std::int64_t>({.min_value = -1, .max_value = 99999}));
      m.eval.push_back(e);
    }
    std::string error;
    const auto text = json::DumpPython(m.ToJson(), 2);
    const auto v = json::Parse(text, &error);
    ASSERT_TRUE(v) << error;
    const auto back = model::VariantManifest::FromJson(*v, &error);
    ASSERT_TRUE(back) << error << "\n" << text;
    ASSERT_EQ(json::DumpPython(back->ToJson(), 2), text);
  }, {.test_cases = 200});
}

TEST(Variant, LoadsVerifiesAndUnloads) {
  Dir d("load");
  const auto base_path = WriteBase(d, 1);
  std::string error;
  auto base = core::GgufReader::OpenFile(base_path, &error);
  ASSERT_TRUE(base) << error;
  const std::vector<std::string> files = {base_path};
  const auto hash = model::BaseContentHash(files, &error);
  ASSERT_TRUE(hash) << error;
  EXPECT_EQ(*hash, *core::Sha256File(base_path, &error));
  model::VariantRegistry reg(*base, *hash);
  for (const char* name : {"one", "two"}) {
    const auto lora = WriteLora(d, std::string(name) + ".gguf");
    ASSERT_TRUE(Manifest(name, *hash, lora).WriteFile(d / (std::string(name) + ".json"), &error)) << error;
    const auto v = reg.Load(d / (std::string(name) + ".json"), &error);
    ASSERT_TRUE(v) << error;
    ASSERT_TRUE(v->lora);
    EXPECT_EQ(v->lora->pairs.size(), 1u);
    EXPECT_EQ(v->manifest.eval.at(0).baseline, 0.5);
  }
  EXPECT_EQ(reg.Names(), (std::vector<std::string>{"one", "two"}));
  EXPECT_FALSE(reg.Load(d / "one.json", &error));
  EXPECT_NE(error.find("already loaded"), std::string::npos) << error;
  const auto held = reg.Find("one");
  EXPECT_TRUE(reg.Unload("one"));
  EXPECT_FALSE(reg.Unload("one"));
  EXPECT_FALSE(reg.Find("one"));
  ASSERT_TRUE(held);  // a request holding it keeps it alive
  EXPECT_EQ(held->manifest.name, "one");
}

TEST(Variant, RefusesAnotherBaseTamperedPartsAndUnsafeManifests) {
  Dir d("refuse");
  const auto base_path = WriteBase(d, 1);
  const auto other_path = WriteBase(d, 2);
  std::string error;
  auto base = core::GgufReader::OpenFile(base_path, &error);
  const auto hash = *core::Sha256File(base_path, &error);
  const auto other = *core::Sha256File(other_path, &error);
  ASSERT_NE(hash, other);
  model::VariantRegistry reg(*base, hash);
  const auto lora = WriteLora(d, "l.gguf");

  ASSERT_TRUE(Manifest("x", other, lora).WriteFile(d / "other.json", &error));
  EXPECT_FALSE(reg.Load(d / "other.json", &error));
  EXPECT_NE(error.find("refused"), std::string::npos) << error;

  auto m = Manifest("x", hash, lora);
  m.base_architecture = "qwen4exp";
  ASSERT_TRUE(m.WriteFile(d / "arch.json", &error));
  EXPECT_FALSE(reg.Load(d / "arch.json", &error));
  EXPECT_NE(error.find("architecture"), std::string::npos) << error;

  ASSERT_TRUE(Manifest("x", hash, lora).WriteFile(d / "ok.json", &error));
  {
    std::fstream f(lora, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(-1, std::ios::end);
    f.put('\x7f');
  }
  EXPECT_FALSE(reg.Load(d / "ok.json", &error));
  EXPECT_NE(error.find("sha256"), std::string::npos) << error;

  fs::remove(lora);
  EXPECT_FALSE(reg.Load(d / "ok.json", &error));
  EXPECT_NE(error.find("missing"), std::string::npos) << error;

  // A LoRA whose shapes do not fit the base.
  const auto bad = WriteLora(d, "bad.gguf", 3);
  {
    core::GgufWriter w;
    w.Set("general.architecture", std::string("qwen35"));
    w.Set("general.type", std::string("adapter"));
    w.Set("adapter.type", std::string("lora"));
    w.Set("adapter.lora.alpha", 4.0F);
    std::vector<float> A(5 * 2, 0.5F), B(2 * 6, 0.5F);
    w.AddTensor("blk.0.attn_q.weight.lora_a", {5, 2}, core::GgmlType::kF32, Bytes(A));
    w.AddTensor("blk.0.attn_q.weight.lora_b", {2, 6}, core::GgmlType::kF32, Bytes(B));
    ASSERT_TRUE(w.WriteFile(bad, &error));
  }
  ASSERT_TRUE(Manifest("x", hash, bad).WriteFile(d / "shape.json", &error));
  EXPECT_FALSE(reg.Load(d / "shape.json", &error));
  EXPECT_NE(error.find("shape"), std::string::npos) << error;

  for (const char* file : {"../l.gguf", "/etc/passwd", "a/../../b", ""}) {
    auto v = Manifest("x", hash, bad).ToJson();
    auto parts = json::Value::MakeArray();
    auto p = v.find("parts")->items()[0];
    p.set("file", json::Value::String(file));
    parts.push(p);
    v.set("parts", parts);
    EXPECT_FALSE(model::VariantManifest::FromJson(v, &error)) << file;
  }
  auto v = Manifest("x", hash, bad).ToJson();
  v.set("format_version", json::Value::Int(2));
  EXPECT_FALSE(model::VariantManifest::FromJson(v, &error));
  EXPECT_NE(error.find("format_version 2"), std::string::npos) << error;
  v = Manifest("x", hash, bad).ToJson();
  v.set("name", json::Value::String("has:colon"));
  EXPECT_FALSE(model::VariantManifest::FromJson(v, &error));
}

TEST(Variant, ModelAlias) {
  std::string error;
  auto a = model::ParseModelAlias("qwen3.5-2b:sentiment-ab", &error);
  ASSERT_TRUE(a);
  EXPECT_EQ(a->base, "qwen3.5-2b");
  EXPECT_EQ(a->variant, "sentiment-ab");
  a = model::ParseModelAlias("plain-model", &error);
  ASSERT_TRUE(a);
  EXPECT_EQ(a->base, "plain-model");
  EXPECT_EQ(a->variant, "");
  EXPECT_FALSE(model::ParseModelAlias("base:", &error));
  EXPECT_FALSE(model::ParseModelAlias("base:bad/name", &error));
}

TEST(Variant, MergeLoraEqualsBasePlusScaledProduct) {
  std::vector<float> W(6 * 4), A(2 * 4), B(6 * 2);
  for (std::size_t i = 0; i < W.size(); ++i) W[i] = static_cast<float>(i) * 0.25F;
  for (std::size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(i) - 3.0F;
  for (std::size_t i = 0; i < B.size(); ++i) B[i] = 0.5F * static_cast<float>(i % 5);
  model::TensorRef w{W.data(), core::GgmlType::kF32, 4, 6, 1, 0, 0, {}};
  model::LoraPair p;
  p.rank = 2;
  p.a = {A.data(), core::GgmlType::kF32, 4, 2, 1, 0, 0, {}};
  p.b = {B.data(), core::GgmlType::kF32, 2, 6, 1, 0, 0, {}};
  const auto merged = model::MergeLora(w, p, 1.5F);
  for (std::size_t r = 0; r < 6; ++r)
    for (std::size_t c = 0; c < 4; ++c) {
      const float want = W[r * 4 + c] + 1.5F * (B[r * 2] * A[c] + B[r * 2 + 1] * A[4 + c]);
      EXPECT_FLOAT_EQ(merged[r * 4 + c], want);
    }
}

TEST(Variant, CApi) {
  Dir d("capi");
  const auto base_path = WriteBase(d, 1);
  char err[512] = {};
  const char* files[] = {base_path.c_str()};
  halo_variant_registry* r = halo_variant_registry_open(files, 1, err, sizeof err);
  ASSERT_TRUE(r) << err;
  std::string error;
  const std::string hash = halo_variant_registry_base_hash(r);
  EXPECT_EQ(hash, *core::Sha256File(base_path, &error));
  // The sidecar cache now exists and is reused.
  EXPECT_TRUE(fs::exists(base_path + ".halo-sha256"));
  const auto lora = WriteLora(d, "l.gguf");
  ASSERT_TRUE(Manifest("cv", hash, lora).WriteFile(d / "m.json", &error));
  char name[64] = {};
  ASSERT_EQ(halo_variant_load(r, (d / "m.json").c_str(), name, sizeof name, err, sizeof err), 0) << err;
  EXPECT_STREQ(name, "cv");
  EXPECT_EQ(halo_variant_count(r), 1u);
  char n0[64] = {};
  EXPECT_EQ(halo_variant_name(r, 0, n0, sizeof n0), 0);
  EXPECT_STREQ(n0, "cv");
  const long need = halo_variant_manifest_json(r, "cv", nullptr, 0);
  ASSERT_GT(need, 0);
  std::vector<char> buf(static_cast<std::size_t>(need) + 1);
  EXPECT_EQ(halo_variant_manifest_json(r, "cv", buf.data(), buf.size()), need);
  EXPECT_NE(std::string(buf.data()).find("\"halo-variant\""), std::string::npos);
  ASSERT_TRUE(Manifest("cw", std::string(64, '0'), lora).WriteFile(d / "w.json", &error));
  EXPECT_EQ(halo_variant_load(r, (d / "w.json").c_str(), name, sizeof name, err, sizeof err), -1);
  EXPECT_NE(std::string(err).find("refused"), std::string::npos) << err;
  char b[64], v[64];
  EXPECT_EQ(halo_variant_parse_model_alias("m:cv", b, sizeof b, v, sizeof v), 0);
  EXPECT_STREQ(b, "m");
  EXPECT_STREQ(v, "cv");
  EXPECT_EQ(halo_variant_unload(r, "cv"), 0);
  EXPECT_EQ(halo_variant_count(r), 0u);
  halo_variant_registry_close(r);
}
