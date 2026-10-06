// C ABI over model::VariantRegistry (include/halo_variant.h).

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "halo_variant.h"
#include "model/variant.hpp"

struct halo_variant_registry {
  std::unique_ptr<halo::core::GgufReader> base;
  std::string hash;
  std::unique_ptr<halo::model::VariantRegistry> registry;
};

namespace {

int SetErr(char* err, std::size_t errlen, const std::string& m) {
  if (err && errlen > 0) {
    const std::size_t n = std::min(errlen - 1, m.size());
    std::memcpy(err, m.data(), n);
    err[n] = '\0';
  }
  return -1;
}

bool CopyOut(const std::string& s, char* out, std::size_t len) {
  if (!out || len <= s.size()) return false;
  std::memcpy(out, s.data(), s.size());
  out[s.size()] = '\0';
  return true;
}

}  // namespace

extern "C" {

halo_variant_registry* halo_variant_registry_open(const char* const* base_files, size_t n_files, char* err,
                                                  size_t errlen) {
  if (!base_files || n_files == 0) {
    SetErr(err, errlen, "no base model files");
    return nullptr;
  }
  std::vector<std::string> files;
  for (size_t i = 0; i < n_files; ++i) {
    if (!base_files[i]) {
      SetErr(err, errlen, "null base model path");
      return nullptr;
    }
    files.emplace_back(base_files[i]);
  }
  std::string error;
  auto r = std::make_unique<halo_variant_registry>();
  r->base = halo::core::GgufReader::OpenFile(files[0], &error);
  if (!r->base) {
    SetErr(err, errlen, error);
    return nullptr;
  }
  const auto hash = halo::model::BaseContentHashCached(files, &error);
  if (!hash) {
    SetErr(err, errlen, error);
    return nullptr;
  }
  r->hash = *hash;
  r->registry = std::make_unique<halo::model::VariantRegistry>(*r->base, r->hash);
  return r.release();
}

void halo_variant_registry_close(halo_variant_registry* r) { delete r; }

const char* halo_variant_registry_base_hash(const halo_variant_registry* r) { return r ? r->hash.c_str() : ""; }

int halo_variant_load(halo_variant_registry* r, const char* manifest_path, char* name_out, size_t name_len,
                      char* err, size_t errlen) {
  if (!r || !manifest_path) return SetErr(err, errlen, "null argument");
  std::string error;
  const auto v = r->registry->Load(manifest_path, &error);
  if (!v) return SetErr(err, errlen, error);
  if (name_out && !CopyOut(v->manifest.name, name_out, name_len)) {
    r->registry->Unload(v->manifest.name);
    return SetErr(err, errlen, "name buffer too small");
  }
  return 0;
}

int halo_variant_unload(halo_variant_registry* r, const char* name) {
  return r && name && r->registry->Unload(name) ? 0 : -1;
}

size_t halo_variant_count(const halo_variant_registry* r) { return r ? r->registry->Names().size() : 0; }

int halo_variant_name(const halo_variant_registry* r, size_t i, char* out, size_t out_len) {
  if (!r) return -1;
  const auto names = r->registry->Names();
  return i < names.size() && CopyOut(names[i], out, out_len) ? 0 : -1;
}

long halo_variant_manifest_json(const halo_variant_registry* r, const char* name, char* buf, size_t buf_len) {
  if (!r || !name) return -1;
  const auto v = r->registry->Find(name);
  if (!v) return -1;
  const std::string text = halo::json::DumpPython(v->manifest.ToJson());
  if (buf && buf_len > text.size()) CopyOut(text, buf, buf_len);
  return static_cast<long>(text.size());
}

int halo_variant_parse_model_alias(const char* model, char* base_out, size_t base_len, char* variant_out,
                                   size_t variant_len) {
  if (!model) return -1;
  std::string error;
  const auto a = halo::model::ParseModelAlias(model, &error);
  if (!a) return -1;
  if (!CopyOut(a->base, base_out, base_len) || !CopyOut(a->variant, variant_out, variant_len)) return -1;
  return 0;
}

}  // extern "C"
