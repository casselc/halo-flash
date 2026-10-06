#include "train/checkpoint.hpp"

#include <cstring>

#include "core/atomic_file.hpp"
#include "core/sha256.hpp"

namespace halo::train {

namespace {

constexpr char kMagic[8] = {'H', 'A', 'L', 'O', 'C', 'K', 'P', '1'};

template <typename T>
void Put(std::vector<std::uint8_t>& out, const T& v) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
  out.insert(out.end(), p, p + sizeof(T));
}

void PutFloats(std::vector<std::uint8_t>& out, const std::vector<float>& v) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
  out.insert(out.end(), p, p + v.size() * sizeof(float));
}

const std::string* Str(const json::Value& v, std::string_view k) {
  const auto* x = v.find(k);
  return x && x->is(json::Value::Kind::kString) ? &x->str() : nullptr;
}

}  // namespace

json::Value RunBinding::ToJson() const {
  json::Value v = json::Value::MakeObject();
  v.set("base_sha256", json::Value::String(base_sha256));
  v.set("data_sha256", json::Value::String(data_sha256));
  v.set("config_sha256", json::Value::String(config_sha256));
  v.set("cache_sha256", json::Value::String(cache_sha256));
  v.set("layout_sha256", json::Value::String(layout_sha256));
  return v;
}

std::optional<RunBinding> RunBinding::FromJson(const json::Value& v) {
  RunBinding b;
  for (auto [key, field] : {std::pair{"base_sha256", &b.base_sha256}, {"data_sha256", &b.data_sha256},
                            {"config_sha256", &b.config_sha256}, {"cache_sha256", &b.cache_sha256},
                            {"layout_sha256", &b.layout_sha256}}) {
    const auto* s = Str(v, key);
    if (!s) return std::nullopt;
    *field = *s;
  }
  return b;
}

std::string RunBinding::Mismatch(const RunBinding& o) const {
  std::string out;
  const auto check = [&](const char* name, const std::string& a, const std::string& b) {
    if (a != b) out += std::string(out.empty() ? "" : ", ") + name + " " + a + " != " + b;
  };
  check("base", base_sha256, o.base_sha256);
  check("data", data_sha256, o.data_sha256);
  check("config", config_sha256, o.config_sha256);
  check("trunk cache", cache_sha256, o.cache_sha256);
  check("parameter layout", layout_sha256, o.layout_sha256);
  return out;
}

bool WriteCheckpoint(const std::string& path, const Checkpoint& c, std::string* error, std::string* sha256) {
  const std::size_t n = c.params.size();
  if (c.m.size() != n || c.v.size() != n) {
    if (error) *error = "checkpoint: moment sizes differ from the parameter count";
    return false;
  }
  json::Value h = json::Value::MakeObject();
  h.set("step", json::Value::Int(static_cast<std::int64_t>(c.step)));
  h.set("binding", c.binding.ToJson());
  const std::string header = json::DumpPython(h);
  std::vector<std::uint8_t> out;
  out.reserve(64 + header.size() + 3 * n * sizeof(float) + 32);
  out.insert(out.end(), kMagic, kMagic + 8);
  Put(out, static_cast<std::uint32_t>(header.size()));
  out.insert(out.end(), header.begin(), header.end());
  Put(out, static_cast<std::uint64_t>(n));
  PutFloats(out, c.params);
  PutFloats(out, c.m);
  PutFloats(out, c.v);
  core::Sha256 hasher;
  hasher.Update(out.data(), out.size());
  const auto digest = hasher.Finish();
  out.insert(out.end(), digest.begin(), digest.end());
  if (sha256) *sha256 = core::HexDigest(digest);
  return core::AtomicWriteFile(path, out, error);
}

std::optional<Checkpoint> ReadCheckpoint(const std::string& path, std::string* error, std::string* sha256) {
  const auto fail = [&](std::string m) -> std::optional<Checkpoint> {
    if (error) *error = "checkpoint " + path + ": " + std::move(m);
    return std::nullopt;
  };
  std::string bytes;
  if (!core::ReadWholeFile(path, &bytes, error)) return std::nullopt;
  if (bytes.size() < 8 + 4 + 8 + 32 || std::memcmp(bytes.data(), kMagic, 8) != 0) return fail("not a checkpoint");
  const std::size_t body = bytes.size() - 32;
  core::Sha256 hasher;
  hasher.Update(bytes.data(), body);
  const auto digest = hasher.Finish();
  if (std::memcmp(digest.data(), bytes.data() + body, 32) != 0) return fail("SHA-256 trailer mismatch (torn or corrupted)");
  if (sha256) *sha256 = core::HexDigest(digest);
  std::uint32_t hlen = 0;
  std::memcpy(&hlen, bytes.data() + 8, 4);
  if (12 + static_cast<std::size_t>(hlen) + 8 > body) return fail("truncated header");
  std::string perr;
  const auto h = json::Parse(std::string_view(bytes.data() + 12, hlen), &perr);
  if (!h) return fail("bad header: " + perr);
  Checkpoint c;
  const auto* step = h->find("step");
  const auto* binding = h->find("binding");
  if (!step || !step->is(json::Value::Kind::kInt) || !binding) return fail("header lacks step or binding");
  c.step = static_cast<std::uint64_t>(std::stoll(step->int_digits()));
  const auto b = RunBinding::FromJson(*binding);
  if (!b) return fail("header binding is incomplete");
  c.binding = *b;
  std::uint64_t n = 0;
  std::memcpy(&n, bytes.data() + 12 + hlen, 8);
  const std::size_t at = 12 + hlen + 8;
  if (n > (body - at) / (3 * sizeof(float)) || at + 3 * n * sizeof(float) != body) return fail("size mismatch");
  for (auto* v : {&c.params, &c.m, &c.v}) {
    v->resize(n);
    std::memcpy(v->data(), bytes.data() + at + (v == &c.params ? 0 : v == &c.m ? 1 : 2) * n * sizeof(float),
                n * sizeof(float));
  }
  return c;
}

}  // namespace halo::train
