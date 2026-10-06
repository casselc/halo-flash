#include "train/data.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "core/atomic_file.hpp"
#include "core/json.hpp"
#include "core/sha256.hpp"

namespace halo::train {

namespace {

constexpr char kShardMagic[8] = {'H', 'A', 'L', 'O', 'T', 'C', '0', '1'};

std::string ShardPath(const std::string& dir, std::size_t s) {
  char name[32];
  std::snprintf(name, sizeof name, "shard-%05zu.bin", s);
  return (std::filesystem::path(dir) / name).string();
}

// Verifies one shard and appends its floats; returns its trailer hex.
std::optional<std::string> ReadShard(const std::string& path, const std::string& key_sha, std::size_t first,
                                     std::size_t count, std::span<const TextExample> examples, std::size_t hidden,
                                     std::vector<float>* out, std::string* error) {
  const auto fail = [&](std::string m) -> std::optional<std::string> {
    if (error) *error = "trunk cache " + path + ": " + std::move(m);
    return std::nullopt;
  };
  std::string bytes, ignored;
  if (!core::ReadWholeFile(path, &bytes, &ignored)) return fail("missing");
  if (bytes.size() < 8 + 4 + 32 || std::memcmp(bytes.data(), kShardMagic, 8) != 0) return fail("not a cache shard");
  const std::size_t body = bytes.size() - 32;
  core::Sha256 h;
  h.Update(bytes.data(), body);
  const auto digest = h.Finish();
  if (std::memcmp(digest.data(), bytes.data() + body, 32) != 0) return fail("SHA-256 trailer mismatch");
  std::uint32_t hlen = 0;
  std::memcpy(&hlen, bytes.data() + 8, 4);
  if (12 + static_cast<std::size_t>(hlen) > body) return fail("truncated header");
  std::string perr;
  const auto hdr = json::Parse(std::string_view(bytes.data() + 12, hlen), &perr);
  if (!hdr) return fail("bad header: " + perr);
  const auto* k = hdr->find("key_sha256");
  const auto* f = hdr->find("first");
  const auto* lens = hdr->find("tokens");
  if (!k || k->str() != key_sha) return fail("built for another key (base, data, tokenizer, layer or engine)");
  if (!f || static_cast<std::size_t>(f->number()) != first || !lens || !lens->is(json::Value::Kind::kArray))
    return fail("header does not match its position");
  const auto& l = lens->items();
  if (l.size() != count) return fail("example count differs from the dataset");
  std::size_t floats = 0;
  for (std::size_t i = 0; i < l.size(); ++i) {
    if (first + i >= examples.size() || static_cast<std::size_t>(l[i].number()) != examples[first + i].tokens.size())
      return fail("token counts differ from the dataset");
    floats += examples[first + i].tokens.size() * hidden;
  }
  if (12 + hlen + floats * sizeof(float) != body) return fail("size mismatch");
  if (out) {
    const std::size_t at = out->size();
    out->resize(at + floats);
    std::memcpy(out->data() + at, bytes.data() + 12 + hlen, floats * sizeof(float));
  }
  return core::HexDigest(digest);
}

}  // namespace

std::optional<Dataset> LoadJsonl(const std::string& path, const text::Tokenizer& tok, HeadKind head,
                                 std::uint32_t num_classes, std::uint32_t max_tokens, std::string* error) {
  std::string text;
  if (!core::ReadWholeFile(path, &text, error)) return std::nullopt;
  Dataset d;
  d.sha256 = core::Sha256Hex(text);
  std::size_t line_no = 0, at = 0;
  const auto fail = [&](std::string m) -> std::optional<Dataset> {
    if (error) *error = path + ":" + std::to_string(line_no) + ": " + std::move(m);
    return std::nullopt;
  };
  while (at < text.size()) {
    const std::size_t end = std::min(text.find('\n', at), text.size());
    const std::string_view line(text.data() + at, end - at);
    at = end + 1;
    ++line_no;
    if (line.find_first_not_of(" \t\r") == std::string_view::npos) continue;
    std::string perr;
    const auto v = json::Parse(line, &perr);
    if (!v || !v->is(json::Value::Kind::kObject)) return fail("not a JSON object: " + perr);
    TextExample ex;
    const auto ids = [&](std::string_view s) {
      const auto t = tok.Encode(s, false);
      return std::vector<std::int32_t>(t.begin(), t.end());
    };
    if (head == HeadKind::kLm) {
      const auto* p = v->find("prompt");
      const auto* c = v->find("completion");
      if (!p || !c || !p->is(json::Value::Kind::kString) || !c->is(json::Value::Kind::kString))
        return fail("LM rows need string \"prompt\" and \"completion\"");
      auto prompt = ids(p->str());
      const auto completion = ids(c->str());
      if (prompt.empty() || completion.empty()) return fail("empty prompt or completion");
      ex.tokens = prompt;
      ex.tokens.insert(ex.tokens.end(), completion.begin(), completion.end() - 1);
      for (std::size_t i = 0; i < completion.size(); ++i)
        ex.lm_targets.emplace_back(static_cast<std::uint32_t>(prompt.size() - 1 + i), completion[i]);
    } else {
      const auto* t = v->find("text");
      const auto* l = v->find("label");
      if (!t || !l || !t->is(json::Value::Kind::kString) || !l->is(json::Value::Kind::kInt))
        return fail("classifier rows need string \"text\" and integer \"label\"");
      ex.tokens = ids(t->str());
      const double label = l->number();
      if (ex.tokens.empty() || label < 0 || label >= num_classes) return fail("empty text or label out of range");
      ex.label = static_cast<std::int32_t>(label);
      ex.label_position = static_cast<std::uint32_t>(ex.tokens.size() - 1);
    }
    if (ex.tokens.size() > max_tokens) return fail("example has " + std::to_string(ex.tokens.size()) + " tokens > " +
                                                   std::to_string(max_tokens));
    d.examples.push_back(std::move(ex));
  }
  if (d.examples.empty()) {
    if (error) *error = path + ": no examples";
    return std::nullopt;
  }
  return d;
}

std::string TrunkCache::Key::Sha256() const {
  return core::Sha256Hex("halo-trunk-cache-v1\n" + base_sha256 + "\n" + data_sha256 + "\n" + tokenizer_sha256 + "\n" +
                         engine + "\n" + std::to_string(layer) + "\n" + std::to_string(hidden) + "\n");
}

std::optional<TrunkCache::BuildStatus> TrunkCache::Build(const std::string& dir, const Key& key,
                                                         std::span<const TextExample> examples, const TrunkFn& trunk,
                                                         double max_seconds, std::string* error) {
  const auto t0 = std::chrono::steady_clock::now();
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  const std::string key_sha = key.Sha256();
  BuildStatus st;
  st.shards_total = (examples.size() + kShardExamples - 1) / kShardExamples;
  for (std::size_t s = 0; s < st.shards_total; ++s) {
    const std::size_t first = s * kShardExamples, count = std::min(kShardExamples, examples.size() - first);
    const std::string path = ShardPath(dir, s);
    std::string why;
    if (std::filesystem::exists(path) &&
        ReadShard(path, key_sha, first, count, examples, key.hidden, nullptr, &why).has_value()) {
      ++st.shards_done;
      continue;
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (elapsed >= max_seconds) break;
    json::Value h = json::Value::MakeObject();
    h.set("key_sha256", json::Value::String(key_sha));
    h.set("first", json::Value::Int(static_cast<std::int64_t>(first)));
    json::Value lens = json::Value::MakeArray();
    std::vector<float> floats;
    for (std::size_t i = first; i < first + count; ++i) {
      const auto& ex = examples[i];
      lens.push(json::Value::Int(static_cast<std::int64_t>(ex.tokens.size())));
      const std::size_t at = floats.size();
      floats.resize(at + ex.tokens.size() * key.hidden);
      if (!trunk(ex.tokens, std::span<float>(floats).subspan(at), error)) return std::nullopt;
      st.tokens_built_now += ex.tokens.size();
    }
    h.set("tokens", std::move(lens));
    const std::string header = json::DumpPython(h);
    std::vector<std::uint8_t> out(kShardMagic, kShardMagic + 8);
    const auto hlen = static_cast<std::uint32_t>(header.size());
    out.insert(out.end(), reinterpret_cast<const std::uint8_t*>(&hlen), reinterpret_cast<const std::uint8_t*>(&hlen) + 4);
    out.insert(out.end(), header.begin(), header.end());
    const auto* fb = reinterpret_cast<const std::uint8_t*>(floats.data());
    out.insert(out.end(), fb, fb + floats.size() * sizeof(float));
    core::Sha256 hs;
    hs.Update(out.data(), out.size());
    const auto digest = hs.Finish();
    out.insert(out.end(), digest.begin(), digest.end());
    if (!core::AtomicWriteFile(path, out, error)) return std::nullopt;
    ++st.shards_done;
    ++st.shards_built_now;
  }
  st.complete = st.shards_done == st.shards_total;
  st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return st;
}

std::optional<TrunkCache> TrunkCache::Load(const std::string& dir, const Key& key,
                                           std::span<const TextExample> examples, std::string* error) {
  TrunkCache c;
  const std::string key_sha = key.Sha256();
  const std::size_t shards = (examples.size() + kShardExamples - 1) / kShardExamples;
  core::Sha256 all;
  for (std::size_t s = 0; s < shards; ++s) {
    const std::size_t first = s * kShardExamples, count = std::min(kShardExamples, examples.size() - first);
    const auto trailer = ReadShard(ShardPath(dir, s), key_sha, first, count, examples, key.hidden, &c.data_, error);
    if (!trailer) return std::nullopt;
    all.Update(*trailer + "\n");
  }
  c.offsets_.push_back(0);
  for (const auto& ex : examples) c.offsets_.push_back(c.offsets_.back() + ex.tokens.size() * key.hidden);
  if (c.offsets_.back() != c.data_.size()) {
    if (error) *error = "trunk cache size mismatch";
    return std::nullopt;
  }
  c.sha256_ = core::HexDigest(all.Finish());
  return c;
}

}  // namespace halo::train
