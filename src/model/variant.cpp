#include "model/variant.hpp"

#include <sys/stat.h>

#include <cmath>
#include <cstdlib>
#include <set>

#include "core/atomic_file.hpp"
#include "core/sha256.hpp"
#include "model/cpu_ops.hpp"

namespace halo::model {

namespace {

using json::Value;
using Kind = json::Value::Kind;

bool Fail(std::string* error, std::string m) {
  if (error) *error = std::move(m);
  return false;
}

const std::string* Str(const Value& o, std::string_view key) {
  const Value* v = o.find(key);
  return v && v->is(Kind::kString) ? &v->str() : nullptr;
}

std::optional<std::int64_t> Int(const Value& o, std::string_view key) {
  const Value* v = o.find(key);
  if (!v || !v->is(Kind::kInt)) return std::nullopt;
  try {
    return std::stoll(v->int_digits());
  } catch (...) {
    return std::nullopt;
  }
}

bool IsHex64(std::string_view s) {
  if (s.size() != 64) return false;
  for (char c : s)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  return true;
}

bool SafeRelative(std::string_view f) {
  if (f.empty() || f.front() == '/' || f.find('\\') != std::string_view::npos) return false;
  for (const auto& part : std::filesystem::path(f))
    if (part == ".." || part == ".") return false;
  return true;
}

std::optional<VariantPartKind> ParseKind(std::string_view s) {
  if (s == "lora") return VariantPartKind::kLora;
  if (s == "classifier_head") return VariantPartKind::kClassifierHead;
  if (s == "pointer_head") return VariantPartKind::kPointerHead;
  return std::nullopt;
}

}  // namespace

const char* VariantPartKindName(VariantPartKind k) noexcept {
  switch (k) {
    case VariantPartKind::kLora: return "lora";
    case VariantPartKind::kClassifierHead: return "classifier_head";
    case VariantPartKind::kPointerHead: return "pointer_head";
  }
  return "unknown";
}

bool ValidVariantName(std::string_view name) noexcept {
  if (name.empty() || name.size() > 64 || name.front() == '.') return false;
  for (char c : name)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' ||
          c == '-'))
      return false;
  return true;
}

std::optional<ModelAlias> ParseModelAlias(std::string_view model, std::string* error) {
  const auto colon = model.rfind(':');
  if (colon == std::string_view::npos) return ModelAlias{std::string(model), ""};
  ModelAlias a{std::string(model.substr(0, colon)), std::string(model.substr(colon + 1))};
  if (!ValidVariantName(a.variant)) {
    if (error) *error = "invalid variant name in model alias \"" + std::string(model) + "\"";
    return std::nullopt;
  }
  return a;
}

std::optional<VariantManifest> VariantManifest::FromJson(const Value& v, std::string* error) {
  const auto fail = [&](std::string m) -> std::optional<VariantManifest> {
    Fail(error, "variant manifest: " + std::move(m));
    return std::nullopt;
  };
  if (!v.is(Kind::kObject)) return fail("not a JSON object");
  const std::string* format = Str(v, "format");
  if (!format || *format != "halo-variant") return fail("format is not \"halo-variant\"");
  const auto fv = Int(v, "format_version");
  if (!fv) return fail("missing format_version");
  if (*fv != kFormatVersion)
    return fail("format_version " + std::to_string(*fv) + " is not supported (this build reads " +
                std::to_string(kFormatVersion) + ")");
  VariantManifest m;
  const std::string* name = Str(v, "name");
  if (!name || !ValidVariantName(*name)) return fail("missing or invalid name");
  m.name = *name;
  const auto ver = Int(v, "version");
  if (!ver || *ver < 1) return fail("version must be a positive integer");
  m.version = *ver;
  const Value* base = v.find("base");
  if (!base || !base->is(Kind::kObject)) return fail("missing base");
  const std::string* arch = Str(*base, "architecture");
  const std::string* hash = Str(*base, "content_sha256");
  if (!arch || arch->empty()) return fail("missing base.architecture");
  if (!hash || !IsHex64(*hash)) return fail("base.content_sha256 must be 64 lowercase hex digits");
  m.base_architecture = *arch;
  m.base_sha256 = *hash;
  if (const std::string* bn = Str(*base, "name")) m.base_name = *bn;
  const Value* parts = v.find("parts");
  if (!parts || !parts->is(Kind::kArray) || parts->items().empty()) return fail("parts must be a non-empty array");
  std::set<VariantPartKind> kinds;
  for (const auto& p : parts->items()) {
    if (!p.is(Kind::kObject)) return fail("a part is not an object");
    const std::string* k = Str(p, "kind");
    const auto kind = k ? ParseKind(*k) : std::nullopt;
    if (!kind) return fail("unknown part kind " + (k ? *k : std::string("(missing)")));
    if (!kinds.insert(*kind).second) return fail(std::string("two parts of kind ") + VariantPartKindName(*kind));
    VariantPart part;
    part.kind = *kind;
    const std::string* f = Str(p, "file");
    if (!f || !SafeRelative(*f)) return fail("part file must be a relative path inside the manifest's directory");
    part.file = *f;
    const std::string* h = Str(p, "sha256");
    if (!h || !IsHex64(*h)) return fail("part " + part.file + ": sha256 must be 64 lowercase hex digits");
    part.sha256 = *h;
    const auto bytes = Int(p, "bytes");
    if (!bytes || *bytes < 0) return fail("part " + part.file + ": missing bytes");
    part.bytes = static_cast<std::uint64_t>(*bytes);
    m.parts.push_back(std::move(part));
  }
  if (kinds.contains(VariantPartKind::kClassifierHead) && kinds.contains(VariantPartKind::kPointerHead))
    return fail("a variant has at most one head");
  if (const Value* prov = v.find("provenance")) {
    if (!prov->is(Kind::kObject)) return fail("provenance must be an object");
    m.provenance = *prov;
  }
  if (const Value* ev = v.find("eval")) {
    if (!ev->is(Kind::kArray)) return fail("eval must be an array");
    for (const auto& e : ev->items()) {
      if (!e.is(Kind::kObject)) return fail("an eval entry is not an object");
      VariantEval r;
      const std::string* n = Str(e, "name");
      const std::string* metric = Str(e, "metric");
      const Value* value = e.find("value");
      if (!n || !metric || !value || !(value->is(Kind::kInt) || value->is(Kind::kFloat)))
        return fail("eval entries need name, metric and a numeric value");
      r.name = *n;
      r.metric = *metric;
      r.value = value->number();
      if (const Value* b = e.find("baseline"); b && (b->is(Kind::kInt) || b->is(Kind::kFloat))) r.baseline = b->number();
      if (const auto cnt = Int(e, "n")) r.n = static_cast<std::uint64_t>(std::max<std::int64_t>(0, *cnt));
      if (const auto step = Int(e, "step")) r.step = *step;
      if (const std::string* d = Str(e, "dataset_sha256")) r.dataset_sha256 = *d;
      m.eval.push_back(std::move(r));
    }
  }
  return m;
}

Value VariantManifest::ToJson() const {
  Value v = Value::MakeObject();
  v.set("format", Value::String("halo-variant"));
  v.set("format_version", Value::Int(kFormatVersion));
  v.set("name", Value::String(name));
  v.set("version", Value::Int(version));
  Value base = Value::MakeObject();
  base.set("architecture", Value::String(base_architecture));
  if (!base_name.empty()) base.set("name", Value::String(base_name));
  base.set("content_sha256", Value::String(base_sha256));
  v.set("base", std::move(base));
  Value ps = Value::MakeArray();
  for (const auto& p : parts) {
    Value o = Value::MakeObject();
    o.set("kind", Value::String(VariantPartKindName(p.kind)));
    o.set("file", Value::String(p.file));
    o.set("sha256", Value::String(p.sha256));
    o.set("bytes", Value::Int(static_cast<std::int64_t>(p.bytes)));
    ps.push(std::move(o));
  }
  v.set("parts", std::move(ps));
  v.set("provenance", provenance);
  Value ev = Value::MakeArray();
  for (const auto& e : eval) {
    Value o = Value::MakeObject();
    o.set("name", Value::String(e.name));
    o.set("metric", Value::String(e.metric));
    o.set("value", Value::Float(e.value));
    if (e.baseline) o.set("baseline", Value::Float(*e.baseline));
    o.set("n", Value::Int(static_cast<std::int64_t>(e.n)));
    if (e.step >= 0) o.set("step", Value::Int(e.step));
    if (!e.dataset_sha256.empty()) o.set("dataset_sha256", Value::String(e.dataset_sha256));
    ev.push(std::move(o));
  }
  v.set("eval", std::move(ev));
  return v;
}

std::optional<VariantManifest> VariantManifest::ReadFile(const std::string& path, std::string* error) {
  std::string text;
  if (!core::ReadWholeFile(path, &text, error)) return std::nullopt;
  std::string perr;
  const auto v = json::Parse(text, &perr, {.max_depth = 64, .max_bytes = 4u << 20});
  if (!v) {
    Fail(error, "variant manifest " + path + " is not valid JSON: " + perr);
    return std::nullopt;
  }
  return FromJson(*v, error);
}

bool VariantManifest::WriteFile(const std::string& path, std::string* error) const {
  return core::AtomicWriteFile(path, json::DumpPython(ToJson(), 2) + "\n", error);
}

const VariantPart* VariantManifest::Find(VariantPartKind k) const {
  for (const auto& p : parts)
    if (p.kind == k) return &p;
  return nullptr;
}

std::optional<std::string> BaseContentHash(std::span<const std::string> files, std::string* error) {
  if (files.empty()) {
    Fail(error, "no base model files");
    return std::nullopt;
  }
  if (files.size() == 1) return core::Sha256File(files[0], error);
  core::Sha256 h;
  for (const auto& f : files) {
    const auto one = core::Sha256File(f, error);
    if (!one) return std::nullopt;
    h.Update(*one + "  " + std::filesystem::path(f).filename().string() + "\n");
  }
  return core::HexDigest(h.Finish());
}

namespace {

std::optional<std::string> CachedFileHash(const std::string& path, std::string* error) {
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) {
    Fail(error, "cannot stat " + path);
    return std::nullopt;
  }
  const std::string key = std::to_string(st.st_size) + " " +
                          std::to_string(static_cast<long long>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec) +
                          " " + std::to_string(st.st_ino);
  // $HALO_HASH_CACHE_DIR keeps sidecars out of read-only or shared model directories.
  const char* cache_dir = std::getenv("HALO_HASH_CACHE_DIR");
  const std::string sidecar =
      cache_dir && *cache_dir
          ? (std::filesystem::path(cache_dir) / (core::Sha256Hex(std::filesystem::absolute(path).string()) + ".halo-sha256"))
                .string()
          : path + ".halo-sha256";
  std::string text, ignored;
  if (core::ReadWholeFile(sidecar, &text, &ignored) && text.size() > 65 && text[64] == ' ') {
    std::string rest = text.substr(65);
    while (!rest.empty() && (rest.back() == '\n' || rest.back() == ' ')) rest.pop_back();
    if (rest == key && IsHex64(text.substr(0, 64))) return text.substr(0, 64);
  }
  auto h = core::Sha256File(path, error);
  if (h) (void)core::AtomicWriteFile(sidecar, *h + " " + key + "\n", &ignored);
  return h;
}

}  // namespace

std::optional<std::string> BaseContentHashCached(std::span<const std::string> files, std::string* error) {
  if (files.empty()) {
    Fail(error, "no base model files");
    return std::nullopt;
  }
  if (files.size() == 1) return CachedFileHash(files[0], error);
  core::Sha256 h;
  for (const auto& f : files) {
    const auto one = CachedFileHash(f, error);
    if (!one) return std::nullopt;
    h.Update(*one + "  " + std::filesystem::path(f).filename().string() + "\n");
  }
  return core::HexDigest(h.Finish());
}

VariantRegistry::VariantRegistry(const core::GgufReader& base, std::string base_sha256)
    : base_(base), base_sha256_(std::move(base_sha256)) {
  base_arch_ = std::string(base.GetMetadataString("general.architecture").value_or(""));
}

std::shared_ptr<const LoadedVariant> VariantRegistry::Load(const std::string& manifest_path, std::string* error) {
  auto m = VariantManifest::ReadFile(manifest_path, error);
  if (!m) return nullptr;
  if (m->base_architecture != base_arch_) {
    Fail(error, "variant " + m->name + " is for architecture " + m->base_architecture + ", the loaded base is " +
                    base_arch_);
    return nullptr;
  }
  if (m->base_sha256 != base_sha256_) {
    Fail(error, "variant " + m->name + " was trained on base " + m->base_sha256 + "; the loaded base is " +
                    base_sha256_ + " (refused)");
    return nullptr;
  }
  auto lv = std::make_shared<LoadedVariant>();
  lv->dir = std::filesystem::absolute(manifest_path).parent_path();
  for (const auto& p : m->parts) {
    const std::string path = (lv->dir / p.file).string();
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
      Fail(error, "variant " + m->name + ": part " + p.file + " is missing");
      return nullptr;
    }
    if (size != p.bytes) {
      Fail(error, "variant " + m->name + ": part " + p.file + " has " + std::to_string(size) + " bytes, the manifest says " +
                      std::to_string(p.bytes));
      return nullptr;
    }
    const auto h = core::Sha256File(path, error);
    if (!h) return nullptr;
    if (*h != p.sha256) {
      Fail(error, "variant " + m->name + ": part " + p.file + " has sha256 " + *h + ", the manifest says " + p.sha256);
      return nullptr;
    }
    auto reader = core::GgufReader::OpenFile(path, error);
    if (!reader) return nullptr;
    if (p.kind == VariantPartKind::kLora) {
      std::string berr;
      lv->lora = LoraAdapter::Bind(*reader, base_, &berr);
      if (!lv->lora) {
        Fail(error, "variant " + m->name + ": " + berr);
        return nullptr;
      }
      lv->lora_file = std::move(reader);
    } else {
      lv->head_file = std::move(reader);
    }
  }
  lv->manifest = std::move(*m);
  std::lock_guard<std::mutex> lk(mu_);
  if (variants_.contains(lv->manifest.name)) {
    Fail(error, "a variant named " + lv->manifest.name + " is already loaded");
    return nullptr;
  }
  variants_.emplace(lv->manifest.name, lv);
  return lv;
}

bool VariantRegistry::Unload(std::string_view name) {
  std::lock_guard<std::mutex> lk(mu_);
  const auto it = variants_.find(name);
  if (it == variants_.end()) return false;
  variants_.erase(it);
  return true;
}

std::shared_ptr<const LoadedVariant> VariantRegistry::Find(std::string_view name) const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto it = variants_.find(name);
  return it == variants_.end() ? nullptr : it->second;
}

std::vector<std::string> VariantRegistry::Names() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<std::string> out;
  for (const auto& [n, _] : variants_) out.push_back(n);
  return out;
}

std::vector<float> MergeLora(const TensorRef& base, const LoraPair& pair, float scale) {
  const std::size_t rows = base.rows, cols = base.cols, r = pair.rank;
  std::vector<float> out(rows * cols);
  for (std::size_t i = 0; i < rows; ++i) cpu::DequantizeRow(base, 0, i, out.data() + i * cols);
  const auto* A = static_cast<const float*>(pair.a.data);  // [r][cols]
  const auto* B = static_cast<const float*>(pair.b.data);  // [rows][r]
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j) {
      double acc = 0;
      for (std::size_t k = 0; k < r; ++k) acc += static_cast<double>(B[i * r + k]) * A[k * cols + j];
      out[i * cols + j] += static_cast<float>(scale * acc);
    }
  return out;
}

}  // namespace halo::model
