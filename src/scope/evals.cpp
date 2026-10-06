#include "scope/evals.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <cstdio>
#include <random>

#include "scope/records.hpp"
#include "scope/sqlite_util.hpp"

// HALO_SCOPE_MUTANT selects a deliberately broken variant for the positive
// controls in CMakeLists.txt; 0 is the real code.
#ifndef HALO_SCOPE_MUTANT
#define HALO_SCOPE_MUTANT 0
#endif

namespace halo::scope {

namespace {

using json::Value;
using Kind = json::Value::Kind;
using sql::Stmt;
using sql::Tx;

void Canon(const Value& v, std::string& out) {
  switch (v.kind()) {
    case Kind::kObject: {
      std::vector<const Value::Member*> ms;
      for (const auto& m : v.members()) ms.push_back(&m);
      std::sort(ms.begin(), ms.end(), [](const auto* a, const auto* b) { return a->first < b->first; });
      out += '{';
      for (std::size_t i = 0; i < ms.size(); ++i) {
        if (i) out += ',';
        out += json::DumpPython(Value::String(ms[i]->first));
        out += ':';
        Canon(ms[i]->second, out);
      }
      out += '}';
      return;
    }
    case Kind::kArray: {
      out += '[';
      for (std::size_t i = 0; i < v.items().size(); ++i) {
        if (i) out += ',';
        Canon(v.items()[i], out);
      }
      out += ']';
      return;
    }
    default:
      out += json::DumpPython(v);
  }
}

std::string Hex(const rb::Hash16& h) {
  static const char* d = "0123456789abcdef";
  std::string s(32, '0');
  for (std::size_t i = 0; i < 16; ++i) {
    s[2 * i] = d[h[i] >> 4];
    s[2 * i + 1] = d[h[i] & 15];
  }
  return s;
}

bool IsObjectJson(const std::string& text) {
  std::string err;
  auto v = json::Parse(text, &err);
  return v && v->is(Kind::kObject);
}

DatasetItem ReadItem(Stmt& q, int c0) {
  DatasetItem it;
  it.id = q.Text(c0);
  it.ordinal = q.Int(c0 + 1);
  it.input_json = q.Text(c0 + 2);
  it.expected_json = q.Text(c0 + 3);
  it.metadata_json = q.Text(c0 + 4);
  it.source_trace_id = q.Text(c0 + 5);
  it.source_observation_id = q.Text(c0 + 6);
  it.content_hash = q.Text(c0 + 7);
  return it;
}

constexpr const char* kItemCols =
    "item_id, ordinal, input, expected_output, metadata, source_trace_id, source_observation_id, content_hash";

ScoreRow ReadScore(Stmt& q) {
  ScoreRow s;
  s.item_id = q.Text(0);
  s.name = q.Text(1);
  s.evaluator = q.Text(2);
  s.evaluator_version = q.Int(3);
  s.data_type = q.Text(4);
  if (!q.IsNull(5)) s.value = q.Double(5);
  s.string_value = q.Text(6);
  s.comment = q.Text(7);
  s.error = q.Text(8);
  s.score_id = q.Text(9);
  s.judge_model = q.Text(10);
  s.judge_prompt = q.Text(11);
  s.judge_prompt_version = q.Int(12);
  s.judge_trace_id = q.Text(13);
  s.raw_output = q.Text(14);
  return s;
}

}  // namespace

std::string CanonicalJson(const json::Value& v) {
  std::string out;
  Canon(v, out);
  return out;
}

std::optional<std::string> Canonicalize(std::string_view text, std::string* err) {
  auto v = json::Parse(text, err);
  if (!v) return std::nullopt;
  return CanonicalJson(*v);
}

std::string FramedHash(const std::vector<std::string_view>& fields) {
  std::string buf;
  for (auto f : fields) {
    const std::uint64_t n = f.size();
    for (int i = 0; i < 8; ++i) buf += static_cast<char>((n >> (8 * i)) & 0xff);
    buf.append(f);
  }
  return Hex(PayloadHash(buf));
}

std::string ItemHash(const DatasetItem& it) {
  return FramedHash({"item.v1", it.id, it.input_json, it.expected_json, it.metadata_json, it.source_trace_id,
                     it.source_observation_id});
}

std::string AttemptTraceId(std::string_view run_id, std::string_view item_id, int attempt) {
  const std::string a = std::to_string(attempt);
  return FramedHash({"trace.v1", run_id, item_id, a});
}

std::unique_ptr<EvalStore> EvalStore::Open(const std::string& path, std::string* err) {
  std::unique_ptr<EvalStore> s(new EvalStore());
  s->db_ = sql::OpenMigrated(path, MetadataStore::Migrations(), err);
  if (!s->db_) return nullptr;
  return s;
}

EvalStore::~EvalStore() {
  if (db_) sqlite3_close_v2(db_);
}

std::string EvalStore::NewId() {
  static thread_local std::mt19937_64 rng{std::random_device{}() ^
                                          (static_cast<std::uint64_t>(std::random_device{}()) << 32)};
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(rng()));
  return buf;
}

// ---------------------------------------------------------------- datasets

std::optional<std::int64_t> EvalStore::DatasetId(std::string_view name, bool* archived) {
  Stmt q(db_, "SELECT id, archived_at IS NOT NULL FROM datasets WHERE name = ?");
  q.Bind(1, name);
  if (q.Step() != SQLITE_ROW) return std::nullopt;
  if (archived) *archived = q.Int(1) != 0;
  return q.Int(0);
}

bool EvalStore::CreateDataset(std::string_view name, std::string_view description, std::string_view metadata_json,
                              std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  std::string meta = "{}";
  if (!metadata_json.empty()) {
    auto c = Canonicalize(metadata_json, err);
    if (!c || !IsObjectJson(*c)) return *err = "metadata must be a JSON object", false;
    meta = *c;
  }
  if (name.empty()) return *err = "dataset name must not be empty", false;
  Stmt q(db_, "INSERT INTO datasets (name, description, metadata) VALUES (?, ?, ?)");
  if (q.Bind(1, name).Bind(2, description).Bind(3, meta).Step() != SQLITE_DONE) {
    *err = sqlite3_extended_errcode(db_) == SQLITE_CONSTRAINT_UNIQUE ? "dataset " + std::string(name) + " exists"
                                                                       : q.Error();
    return false;
  }
  return true;
}

std::optional<DatasetInfo> EvalStore::GetDataset(std::string_view name) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt q(db_, "SELECT id, name, description, metadata, created_at, COALESCE(archived_at, '') FROM datasets "
              "WHERE name = ?");
  q.Bind(1, name);
  if (q.Step() != SQLITE_ROW) return std::nullopt;
  DatasetInfo d{q.Int(0), q.Text(1), q.Text(2), q.Text(3), q.Text(4), q.Text(5), {}};
  Stmt v(db_,
         "SELECT v.version, v.frozen_at IS NOT NULL, COALESCE(v.content_hash, ''), "
         "COALESCE(v.item_count, (SELECT COUNT(*) FROM dataset_items i WHERE i.dataset_id = v.dataset_id "
         "AND i.version = v.version)), v.created_at, COALESCE(v.frozen_at, '') "
         "FROM dataset_versions v WHERE v.dataset_id = ? ORDER BY v.version");
  v.Bind(1, d.id);
  while (v.Step() == SQLITE_ROW)
    d.versions.push_back({v.Int(0), v.Int(1) != 0, v.Text(2), v.Int(3), v.Text(4), v.Text(5)});
  return d;
}

std::vector<DatasetInfo> EvalStore::ListDatasets(bool include_archived) {
  std::vector<std::string> names;
  {
    std::lock_guard<std::mutex> lk(mu_);
    Stmt q(db_, include_archived ? "SELECT name FROM datasets ORDER BY name"
                                 : "SELECT name FROM datasets WHERE archived_at IS NULL ORDER BY name");
    while (q.Step() == SQLITE_ROW) names.push_back(q.Text(0));
  }
  std::vector<DatasetInfo> out;
  for (const auto& n : names)
    if (auto d = GetDataset(n)) out.push_back(std::move(*d));
  return out;
}

std::int64_t EvalStore::EnsureDraft(std::int64_t dataset_id, std::string* err) {
  {
    Stmt q(db_, "SELECT version FROM dataset_versions WHERE dataset_id = ? AND frozen_at IS NULL");
    q.Bind(1, dataset_id);
    if (q.Step() == SQLITE_ROW) return q.Int(0);
  }
  std::int64_t latest = 0;
  {
    Stmt q(db_, "SELECT COALESCE(MAX(version), 0) FROM dataset_versions WHERE dataset_id = ?");
    q.Bind(1, dataset_id);
    if (q.Step() == SQLITE_ROW) latest = q.Int(0);
  }
  const std::int64_t draft = latest + 1;
  Stmt ins(db_, "INSERT INTO dataset_versions (dataset_id, version) VALUES (?, ?)");
  if (ins.Bind(1, dataset_id).Bind(2, draft).Step() != SQLITE_DONE) return *err = ins.Error(), 0;
  if (latest > 0) {  // the draft starts as a copy of the latest frozen version
    Stmt cp(db_,
            "INSERT INTO dataset_items (dataset_id, version, item_id, ordinal, input, expected_output, metadata, "
            "source_trace_id, source_observation_id, content_hash) SELECT dataset_id, ?, item_id, ordinal, input, "
            "expected_output, metadata, source_trace_id, source_observation_id, content_hash FROM dataset_items "
            "WHERE dataset_id = ? AND version = ?");
    if (cp.Bind(1, draft).Bind(2, dataset_id).Bind(3, latest).Step() != SQLITE_DONE) return *err = cp.Error(), 0;
  }
  return draft;
}

bool EvalStore::UpsertLocked(std::int64_t dataset_id, std::int64_t version, std::vector<DatasetItem>& items,
                             std::string* err) {
  std::int64_t next = 0;
  {
    Stmt q(db_, "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM dataset_items WHERE dataset_id = ? AND version = ?");
    q.Bind(1, dataset_id).Bind(2, version);
    if (q.Step() == SQLITE_ROW) next = q.Int(0);
  }
  for (auto& it : items) {
    if (it.id.empty()) it.id = NewId();
    for (auto* f : {&it.input_json, &it.expected_json, &it.metadata_json}) {
      auto c = Canonicalize(f->empty() ? "null" : *f, err);
      if (!c) return *err = "item " + it.id + ": " + *err, false;
      *f = std::move(*c);
    }
    if (it.metadata_json == "null") it.metadata_json = "{}";
    if (!IsObjectJson(it.metadata_json)) return *err = "item " + it.id + ": metadata must be an object", false;
    it.content_hash = ItemHash(it);
    // A replaced item keeps its position; a new one goes last.
    std::int64_t ordinal = next;
    {
      Stmt q(db_, "SELECT ordinal FROM dataset_items WHERE dataset_id = ? AND version = ? AND item_id = ?");
      q.Bind(1, dataset_id).Bind(2, version).Bind(3, it.id);
      if (q.Step() == SQLITE_ROW) ordinal = q.Int(0);
      else ++next;
    }
    it.ordinal = ordinal;
    Stmt up(db_,
            "INSERT INTO dataset_items (dataset_id, version, item_id, ordinal, input, expected_output, metadata, "
            "source_trace_id, source_observation_id, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(dataset_id, version, item_id) DO UPDATE SET input = excluded.input, "
            "expected_output = excluded.expected_output, metadata = excluded.metadata, "
            "source_trace_id = excluded.source_trace_id, source_observation_id = excluded.source_observation_id, "
            "content_hash = excluded.content_hash");
    up.Bind(1, dataset_id).Bind(2, version).Bind(3, it.id).Bind(4, ordinal).Bind(5, it.input_json);
    up.Bind(6, it.expected_json).Bind(7, it.metadata_json).Bind(8, it.source_trace_id);
    up.Bind(9, it.source_observation_id).Bind(10, it.content_hash);
    if (up.Step() != SQLITE_DONE) return *err = up.Error(), false;
  }
  return true;
}

std::int64_t EvalStore::UpsertItems(std::string_view dataset, std::vector<DatasetItem>& items, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  bool archived = false;
  const auto id = DatasetId(dataset, &archived);
  if (!id) return *err = "no dataset named " + std::string(dataset), 0;
  if (archived) return *err = "dataset " + std::string(dataset) + " is archived", 0;
  Tx tx(db_);
  if (!tx.Begin(err)) return 0;
  const auto draft = EnsureDraft(*id, err);
  if (draft == 0 || !UpsertLocked(*id, draft, items, err)) return 0;
  return tx.Commit(err) ? draft : 0;
}

std::int64_t EvalStore::ImportJsonl(std::string_view dataset, std::string_view jsonl, std::size_t* count,
                                    std::string* err) {
  std::vector<DatasetItem> items;
  std::size_t line_no = 0;
  for (std::size_t i = 0; i < jsonl.size();) {
    const auto j = std::min(jsonl.find('\n', i), jsonl.size());
    const auto line = jsonl.substr(i, j - i);
    i = j + 1;
    ++line_no;
    if (line.find_first_not_of(" \t\r") == std::string_view::npos) continue;
    auto v = json::Parse(line, err);
    if (!v || !v->is(Kind::kObject)) return *err = "line " + std::to_string(line_no) + ": not a JSON object", 0;
    const Value* input = v->find("input");
    if (!input) return *err = "line " + std::to_string(line_no) + ": input is required", 0;
    DatasetItem it;
    if (const Value* x = v->find("id")) {
      if (!x->is(Kind::kString)) return *err = "line " + std::to_string(line_no) + ": id must be a string", 0;
      it.id = x->str();
    }
    it.input_json = CanonicalJson(*input);
    if (const Value* x = v->find("expected_output")) it.expected_json = CanonicalJson(*x);
    if (const Value* x = v->find("metadata"); x && !x->is(Kind::kNull)) it.metadata_json = CanonicalJson(*x);
    if (const Value* x = v->find("source_trace_id"); x && x->is(Kind::kString)) it.source_trace_id = x->str();
    if (const Value* x = v->find("source_observation_id"); x && x->is(Kind::kString))
      it.source_observation_id = x->str();
    items.push_back(std::move(it));
  }
  if (count) *count = items.size();
  return UpsertItems(dataset, items, err);
}

bool EvalStore::DeleteItem(std::string_view dataset, std::string_view item_id, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  const auto id = DatasetId(dataset, nullptr);
  if (!id) return *err = "no dataset named " + std::string(dataset), false;
  Stmt q(db_,
         "DELETE FROM dataset_items WHERE dataset_id = ? AND item_id = ? AND version = "
         "(SELECT version FROM dataset_versions WHERE dataset_id = ? AND frozen_at IS NULL)");
  if (q.Bind(1, *id).Bind(2, item_id).Bind(3, *id).Step() != SQLITE_DONE) return *err = q.Error(), false;
  if (sqlite3_changes(db_) == 0) return *err = "no draft item " + std::string(item_id), false;
  return true;
}

std::optional<DatasetVersionInfo> EvalStore::Freeze(std::string_view dataset, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  bool archived = false;
  const auto id = DatasetId(dataset, &archived);
  if (!id) return *err = "no dataset named " + std::string(dataset), std::nullopt;
  if (archived) return *err = "dataset " + std::string(dataset) + " is archived", std::nullopt;
  Tx tx(db_);
  if (!tx.Begin(err)) return std::nullopt;
  std::int64_t draft = 0;
  {
    Stmt q(db_, "SELECT version FROM dataset_versions WHERE dataset_id = ? AND frozen_at IS NULL");
    q.Bind(1, *id);
    if (q.Step() != SQLITE_ROW) return *err = "dataset " + std::string(dataset) + " has no draft", std::nullopt;
    draft = q.Int(0);
  }
  auto items = ItemsLocked(*id, draft);
  if (!items) return *err = "cannot read the draft", std::nullopt;
  if (items->empty()) return *err = "cannot freeze an empty draft", std::nullopt;
  // The version hash covers every item hash in dataset order.
  std::vector<std::string> fields = {"dataset.v1"};
  for (const auto& it : *items) {
    if (ItemHash(it) != it.content_hash) return *err = "item " + it.id + " hash mismatch", std::nullopt;
    fields.push_back(it.content_hash);
  }
  std::vector<std::string_view> views(fields.begin(), fields.end());
  DatasetVersionInfo info;
  info.version = draft;
  info.frozen = true;
  info.content_hash = FramedHash(views);
  info.item_count = static_cast<std::int64_t>(items->size());
  {
    Stmt u(db_,
           "UPDATE dataset_versions SET frozen_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), content_hash = ?, "
           "item_count = ? WHERE dataset_id = ? AND version = ? AND frozen_at IS NULL RETURNING frozen_at, created_at");
    u.Bind(1, info.content_hash).Bind(2, info.item_count).Bind(3, *id).Bind(4, draft);
    if (u.Step() != SQLITE_ROW) return *err = u.Error(), std::nullopt;
    info.frozen_at = u.Text(0);
    info.created_at = u.Text(1);
    u.Step();
  }
  if (!tx.Commit(err)) return std::nullopt;
  return info;
}

bool EvalStore::Archive(std::string_view dataset, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt q(db_, "UPDATE datasets SET archived_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE name = ? AND "
              "archived_at IS NULL");
  if (q.Bind(1, dataset).Step() != SQLITE_DONE) return *err = q.Error(), false;
  if (sqlite3_changes(db_) == 0) return *err = "no active dataset named " + std::string(dataset), false;
  return true;
}

std::optional<std::vector<DatasetItem>> EvalStore::ItemsLocked(std::int64_t dataset_id, std::int64_t version) {
  std::vector<DatasetItem> out;
  Stmt q(db_, (std::string("SELECT ") + kItemCols +
               " FROM dataset_items WHERE dataset_id = ? AND version = ? ORDER BY ordinal, item_id")
                  .c_str());
  q.Bind(1, dataset_id).Bind(2, version);
  int rc;
  while ((rc = q.Step()) == SQLITE_ROW) out.push_back(ReadItem(q, 0));
  if (rc != SQLITE_DONE) return std::nullopt;
  return out;
}

std::optional<std::vector<DatasetItem>> EvalStore::Items(std::string_view dataset, std::int64_t version,
                                                         std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  const auto id = DatasetId(dataset, nullptr);
  if (!id) return *err = "no dataset named " + std::string(dataset), std::nullopt;
  if (version == 0) {
    Stmt q(db_, "SELECT version FROM dataset_versions WHERE dataset_id = ? ORDER BY frozen_at IS NULL DESC, "
                "version DESC LIMIT 1");
    q.Bind(1, *id);
    if (q.Step() != SQLITE_ROW) return std::vector<DatasetItem>{};
    version = q.Int(0);
  } else {
    Stmt q(db_, "SELECT 1 FROM dataset_versions WHERE dataset_id = ? AND version = ?");
    q.Bind(1, *id).Bind(2, version);
    if (q.Step() != SQLITE_ROW) return *err = "no version " + std::to_string(version), std::nullopt;
  }
  auto items = ItemsLocked(*id, version);
  if (!items) *err = sqlite3_errmsg(db_);
  return items;
}

// -------------------------------------------------------------- evaluators

std::int64_t EvalStore::CreateEvaluatorVersion(std::string_view name, std::string_view kind,
                                               std::string_view config_json, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  if (name.empty()) return *err = "evaluator name must not be empty", 0;
  auto config = Canonicalize(config_json.empty() ? "{}" : config_json, err);
  if (!config || !IsObjectJson(*config)) return *err = "config must be a JSON object", 0;
  Tx tx(db_);
  if (!tx.Begin(err)) return 0;
  {
    Stmt a(db_, "INSERT INTO evaluators (name) VALUES (?) ON CONFLICT(name) DO NOTHING");
    if (a.Bind(1, name).Step() != SQLITE_DONE) return *err = a.Error(), 0;
  }
  Stmt v(db_,
         "INSERT INTO evaluator_versions (evaluator_id, version, kind, config) SELECT e.id, COALESCE((SELECT "
         "MAX(version) FROM evaluator_versions WHERE evaluator_id = e.id), 0) + 1, ?, ? FROM evaluators e "
         "WHERE e.name = ? RETURNING version");
  v.Bind(1, kind).Bind(2, *config).Bind(3, name);
  if (v.Step() != SQLITE_ROW) return *err = v.Error(), 0;
  const auto version = v.Int(0);
  v.Step();
  return tx.Commit(err) ? version : 0;
}

std::optional<EvaluatorDef> EvalStore::GetEvaluator(std::string_view name, std::int64_t version) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt q(db_,
         "SELECT v.version, v.kind, v.config FROM evaluators e JOIN evaluator_versions v ON v.evaluator_id = e.id "
         "WHERE e.name = ? AND (? = 0 OR v.version = ?) ORDER BY v.version DESC LIMIT 1");
  q.Bind(1, name).Bind(2, version).Bind(3, version);
  if (q.Step() != SQLITE_ROW) return std::nullopt;
  return EvaluatorDef{std::string(name), q.Int(0), q.Text(1), q.Text(2)};
}

std::vector<EvaluatorDef> EvalStore::ListEvaluators() {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt q(db_,
         "SELECT e.name, v.version, v.kind, v.config FROM evaluators e JOIN evaluator_versions v ON "
         "v.evaluator_id = e.id ORDER BY e.name, v.version");
  std::vector<EvaluatorDef> out;
  while (q.Step() == SQLITE_ROW) out.push_back({q.Text(0), q.Int(1), q.Text(2), q.Text(3)});
  return out;
}

std::optional<PromptVersion> EvalStore::PromptByVersion(std::string_view name, std::int64_t version) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt q(db_,
         "SELECT v.version, v.template, v.config FROM prompts p JOIN prompt_versions v ON v.prompt_id = p.id "
         "WHERE p.name = ? AND v.version = ?");
  q.Bind(1, name).Bind(2, version);
  if (q.Step() != SQLITE_ROW) return std::nullopt;
  return PromptVersion{std::string(name), q.Int(0), q.Text(1), q.Text(2)};
}

std::optional<PromptVersion> EvalStore::PromptByLabel(std::string_view name, std::string_view label) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt q(db_,
         "SELECT v.version, v.template, v.config FROM prompts p JOIN labels l ON l.prompt_id = p.id "
         "JOIN prompt_versions v ON v.prompt_id = p.id AND v.version = l.version WHERE p.name = ? AND l.label = ?");
  q.Bind(1, name).Bind(2, label);
  if (q.Step() != SQLITE_ROW) return std::nullopt;
  return PromptVersion{std::string(name), q.Int(0), q.Text(1), q.Text(2)};
}

// -------------------------------------------------------------------- runs

std::optional<std::string> EvalStore::BindingLocked(std::int64_t dataset_id, std::int64_t dataset_version,
                                                    const std::string& target_json,
                                                    const std::string& evaluators_json, std::string* err) {
  std::string dataset_hash;
  {
    Stmt q(db_, "SELECT content_hash FROM dataset_versions WHERE dataset_id = ? AND version = ? AND "
                "frozen_at IS NOT NULL");
    q.Bind(1, dataset_id).Bind(2, dataset_version);
    if (q.Step() != SQLITE_ROW) return *err = "dataset version " + std::to_string(dataset_version) +
                                              " is not frozen", std::nullopt;
    dataset_hash = q.Text(0);
  }
  auto prompt_text = [&](const Value* p, std::string* out) -> bool {
    if (!p || p->is(Kind::kNull)) return true;
    const Value* n = p->find("name");
    const Value* v = p->find("version");
    if (!n || !n->is(Kind::kString) || !v || !v->is(Kind::kInt)) return *err = "prompt needs name and version", false;
    Stmt q(db_,
           "SELECT v.template, v.config FROM prompts p JOIN prompt_versions v ON v.prompt_id = p.id "
           "WHERE p.name = ? AND v.version = ?");
    q.Bind(1, n->str()).Bind(2, static_cast<std::int64_t>(v->number()));
    if (q.Step() != SQLITE_ROW) return *err = "no prompt " + n->str() + " version " + v->int_digits(), false;
    *out = FramedHash({"prompt.v1", n->str(), v->int_digits(), q.Text(0), q.Text(1)});
    return true;
  };
  auto target = json::Parse(target_json, err);
  auto evals = json::Parse(evaluators_json, err);
  if (!target || !evals || !evals->is(Kind::kArray)) return *err = "bad target or evaluator list", std::nullopt;
  std::vector<std::string> fields = {"run.v1", dataset_hash, target_json};
  std::string ph;
  if (!prompt_text(target->find("prompt"), &ph)) return std::nullopt;
  fields.push_back(ph);
  for (const auto& e : evals->items()) {
    const Value* n = e.find("name");
    const Value* v = e.find("version");
    if (!n || !v) return *err = "bad evaluator reference", std::nullopt;
    Stmt q(db_,
           "SELECT v.kind, v.config FROM evaluators e JOIN evaluator_versions v ON v.evaluator_id = e.id "
           "WHERE e.name = ? AND v.version = ?");
    q.Bind(1, n->str()).Bind(2, static_cast<std::int64_t>(v->number()));
    if (q.Step() != SQLITE_ROW) return *err = "no evaluator " + n->str() + " version " + v->int_digits(), std::nullopt;
    const std::string kind = q.Text(0), config = q.Text(1);
    fields.push_back(FramedHash({"evaluator.v1", n->str(), v->int_digits(), kind, config}));
    // A judge's prompt is part of what it measures.
    auto c = json::Parse(config, err);
    std::string jh;
    if (c && kind == "llm_judge" && !prompt_text(c->find("prompt"), &jh)) return std::nullopt;
    fields.push_back(jh);
  }
  std::vector<std::string_view> views(fields.begin(), fields.end());
  return FramedHash(views);
}

std::optional<RunRecord> EvalStore::RunLocked(std::string_view run_id) {
  Stmt q(db_,
         "SELECT r.id, r.name, r.dataset_id, d.name, r.dataset_version, r.target, r.evaluators, r.binding, "
         "r.item_count, r.max_attempts, r.status, r.metadata, r.created_at, r.updated_at, "
         "COALESCE(r.completed_at, '') FROM runs r JOIN datasets d ON d.id = r.dataset_id WHERE r.id = ?");
  q.Bind(1, run_id);
  if (q.Step() != SQLITE_ROW) return std::nullopt;
  RunRecord r;
  r.id = q.Text(0);
  r.name = q.Text(1);
  r.dataset_id = q.Int(2);
  r.dataset = q.Text(3);
  r.dataset_version = q.Int(4);
  r.target_json = q.Text(5);
  r.evaluators_json = q.Text(6);
  r.binding = q.Text(7);
  r.item_count = q.Int(8);
  r.max_attempts = static_cast<int>(q.Int(9));
  r.status = q.Text(10);
  r.metadata_json = q.Text(11);
  r.created_at = q.Text(12);
  r.updated_at = q.Text(13);
  r.completed_at = q.Text(14);
  return r;
}

std::optional<RunRecord> EvalStore::CreateRun(const RunSpec& spec, bool* existed, std::string* err) {
  if (existed) *existed = false;
  if (spec.name.empty()) return *err = "a run needs a name", std::nullopt;
  if (spec.max_attempts < 1) return *err = "max_attempts must be at least 1", std::nullopt;
  auto target = json::Parse(spec.target_json.empty() ? "{}" : spec.target_json, err);
  if (!target || !target->is(Kind::kObject)) return *err = "target must be a JSON object", std::nullopt;
  auto meta = Canonicalize(spec.metadata_json.empty() ? "{}" : spec.metadata_json, err);
  if (!meta || !IsObjectJson(*meta)) return *err = "metadata must be a JSON object", std::nullopt;
  // Pin the prompt: a label resolves to the version it points at now.
  if (const Value* p = target->find("prompt"); p && !p->is(Kind::kNull)) {
    const Value* n = p->find("name");
    if (!p->is(Kind::kObject) || !n || !n->is(Kind::kString)) return *err = "target.prompt needs a name", std::nullopt;
    std::optional<PromptVersion> pv;
    if (const Value* v = p->find("version"); v && v->is(Kind::kInt)) pv = PromptByVersion(n->str(), static_cast<std::int64_t>(v->number()));
    else {
      const Value* l = p->find("label");
      pv = PromptByLabel(n->str(), l && l->is(Kind::kString) ? l->str() : "latest");
    }
    if (!pv) return *err = "no such prompt version for " + n->str(), std::nullopt;
    Value pinned = Value::MakeObject();
    pinned.set("name", Value::String(pv->name));
    pinned.set("version", Value::Int(pv->version));
    target->set("prompt", std::move(pinned));
  }
  const std::string target_json = CanonicalJson(*target);
  Value evals = Value::MakeArray();
  for (const auto& [name, version] : spec.evaluators) {
    auto def = GetEvaluator(name, version);
    if (!def) return *err = "no evaluator " + name + (version ? " version " + std::to_string(version) : ""), std::nullopt;
    Value e = Value::MakeObject();
    e.set("name", Value::String(def->name));
    e.set("version", Value::Int(def->version));
    evals.push(std::move(e));
  }
  const std::string evaluators_json = CanonicalJson(evals);

  std::lock_guard<std::mutex> lk(mu_);
  bool archived = false;
  const auto dsid = DatasetId(spec.dataset, &archived);
  if (!dsid) return *err = "no dataset named " + spec.dataset, std::nullopt;
  std::int64_t version = spec.dataset_version;
  if (version == 0) {
    Stmt q(db_, "SELECT MAX(version) FROM dataset_versions WHERE dataset_id = ? AND frozen_at IS NOT NULL");
    q.Bind(1, *dsid);
    if (q.Step() != SQLITE_ROW || q.IsNull(0)) return *err = "dataset " + spec.dataset + " has no frozen version", std::nullopt;
    version = q.Int(0);
  }
  Tx tx(db_);
  if (!tx.Begin(err)) return std::nullopt;
  auto binding = BindingLocked(*dsid, version, target_json, evaluators_json, err);
  if (!binding) return std::nullopt;
  {
    Stmt q(db_, "SELECT id, binding FROM runs WHERE dataset_id = ? AND name = ?");
    q.Bind(1, *dsid).Bind(2, spec.name);
    if (q.Step() == SQLITE_ROW) {
      if (q.Text(1) != *binding)
        return *err = "run " + spec.name + " exists with a different binding (dataset version, target or "
                      "evaluators differ); choose a new name", std::nullopt;
      if (existed) *existed = true;
      const std::string id = q.Text(0);
      return RunLocked(id);
    }
  }
  if (archived) return *err = "dataset " + spec.dataset + " is archived", std::nullopt;
  std::int64_t count = 0;
  {
    Stmt q(db_, "SELECT item_count FROM dataset_versions WHERE dataset_id = ? AND version = ?");
    q.Bind(1, *dsid).Bind(2, version);
    if (q.Step() == SQLITE_ROW) count = q.Int(0);
  }
  const std::string id = NewId();
  Stmt ins(db_,
           "INSERT INTO runs (id, name, dataset_id, dataset_version, target, evaluators, binding, item_count, "
           "max_attempts, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
  ins.Bind(1, id).Bind(2, spec.name).Bind(3, *dsid).Bind(4, version).Bind(5, target_json).Bind(6, evaluators_json);
  ins.Bind(7, *binding).Bind(8, count).Bind(9, spec.max_attempts).Bind(10, *meta);
  if (ins.Step() != SQLITE_DONE) return *err = ins.Error(), std::nullopt;
  if (!tx.Commit(err)) return std::nullopt;
  return RunLocked(id);
}

std::optional<RunRecord> EvalStore::GetRun(std::string_view run_id) {
  std::lock_guard<std::mutex> lk(mu_);
  return RunLocked(run_id);
}

std::vector<RunRecord> EvalStore::ListRuns(std::string_view dataset) {
  std::vector<std::string> ids;
  {
    std::lock_guard<std::mutex> lk(mu_);
    Stmt q(db_, "SELECT r.id FROM runs r JOIN datasets d ON d.id = r.dataset_id WHERE ? = '' OR d.name = ? "
                "ORDER BY r.created_at, r.id");
    q.Bind(1, dataset).Bind(2, dataset);
    while (q.Step() == SQLITE_ROW) ids.push_back(q.Text(0));
  }
  std::vector<RunRecord> out;
  for (const auto& id : ids)
    if (auto r = GetRun(id)) out.push_back(std::move(*r));
  return out;
}

std::optional<std::string> EvalStore::RecomputeBinding(const RunRecord& run, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  return BindingLocked(run.dataset_id, run.dataset_version, run.target_json, run.evaluators_json, err);
}

bool EvalStore::SetRunStatus(std::string_view run_id, std::string_view status, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  const bool terminal = status == "completed" || status == "failed" || status == "cancelled";
  Stmt q(db_, terminal ? "UPDATE runs SET status = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
                         "completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?"
                       : "UPDATE runs SET status = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
                         "completed_at = NULL WHERE id = ?");
  if (q.Bind(1, status).Bind(2, run_id).Step() != SQLITE_DONE) return *err = q.Error(), false;
  return sqlite3_changes(db_) == 1 || (*err = "no run " + std::string(run_id), false);
}

std::optional<std::vector<DatasetItem>> EvalStore::PendingItems(const RunRecord& run, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<DatasetItem> out;
  Stmt q(db_, (std::string("SELECT ") + kItemCols +
               " FROM dataset_items i WHERE i.dataset_id = ? AND i.version = ? AND NOT EXISTS (SELECT 1 FROM "
#if HALO_SCOPE_MUTANT == 11
               "run_items r WHERE r.run_id = ? AND r.item_id = i.item_id AND (r.attempts >= ?)) "
#else
               "run_items r WHERE r.run_id = ? AND r.item_id = i.item_id AND (r.status = 'done' OR r.attempts >= ?)) "
#endif
               "ORDER BY i.ordinal, i.item_id")
                  .c_str());
  q.Bind(1, run.dataset_id).Bind(2, run.dataset_version).Bind(3, run.id).Bind(4, run.max_attempts);
  int rc;
  while ((rc = q.Step()) == SQLITE_ROW) out.push_back(ReadItem(q, 0));
  if (rc != SQLITE_DONE) return *err = q.Error(), std::nullopt;
  return out;
}

std::optional<ItemStart> EvalStore::BeginItem(const RunRecord& run, const DatasetItem& item, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  Tx tx(db_);
  if (!tx.Begin(err)) return std::nullopt;
  ItemStart s;
  {
    Stmt q(db_, "SELECT status, attempts, item_hash FROM run_items WHERE run_id = ? AND item_id = ?");
    q.Bind(1, run.id).Bind(2, item.id);
    if (q.Step() == SQLITE_ROW) {
      if (q.Text(2) != item.content_hash)
        return *err = "item " + item.id + " changed under run " + run.id + " (content hash differs)", std::nullopt;
#if HALO_SCOPE_MUTANT != 11  // positive control: a resume redoes done items
      if (q.Text(0) == "done") return s.state = ItemStart::State::kAlreadyDone, s;
#endif
      if (q.Int(1) >= run.max_attempts) return s.state = ItemStart::State::kExhausted, s;
      s.attempt = static_cast<int>(q.Int(1)) + 1;
    } else {
      s.attempt = 1;
    }
  }
  s.trace_id = AttemptTraceId(run.id, item.id, s.attempt);
  Stmt up(db_,
          "INSERT INTO run_items (run_id, item_id, item_hash, status, attempts, trace_id) VALUES (?, ?, ?, "
          "'running', ?, ?) ON CONFLICT(run_id, item_id) DO UPDATE SET status = 'running', attempts = "
          "excluded.attempts, trace_id = excluded.trace_id, error = '', updated_at = "
          "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')");
  up.Bind(1, run.id).Bind(2, item.id).Bind(3, item.content_hash).Bind(4, s.attempt).Bind(5, s.trace_id);
  if (up.Step() != SQLITE_DONE) return *err = up.Error(), std::nullopt;
  if (!tx.Commit(err)) return std::nullopt;
  return s;
}

bool EvalStore::CompleteItem(const RunRecord& run, std::string_view item_id, int attempt, const ItemResult& r,
                             const std::vector<ScoreRow>& scores, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  Tx tx(db_);
  if (!tx.Begin(err)) return false;
  {
    Stmt q(db_, "SELECT status, attempts FROM run_items WHERE run_id = ? AND item_id = ?");
    q.Bind(1, run.id).Bind(2, item_id);
    if (q.Step() != SQLITE_ROW || q.Text(0) != "running" || q.Int(1) != attempt)
      return *err = "item " + std::string(item_id) + " is not running at attempt " + std::to_string(attempt), false;
  }
  for (const auto& s : scores) {
    Stmt ins(db_,
             "INSERT INTO run_scores (run_id, item_id, name, evaluator, evaluator_version, data_type, value, "
             "string_value, comment, error, score_id, judge_model, judge_prompt, judge_prompt_version, "
             "judge_trace_id, raw_output) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
    ins.Bind(1, run.id).Bind(2, item_id).Bind(3, s.name).Bind(4, s.evaluator).Bind(5, s.evaluator_version);
    ins.Bind(6, s.data_type);
    if (s.value) ins.Bind(7, *s.value);
    else ins.BindNull(7);
    ins.Bind(8, s.string_value).Bind(9, s.comment).Bind(10, s.error).Bind(11, s.score_id).Bind(12, s.judge_model);
    ins.Bind(13, s.judge_prompt).Bind(14, s.judge_prompt_version).Bind(15, s.judge_trace_id).Bind(16, s.raw_output);
    if (ins.Step() != SQLITE_DONE) return *err = "score " + s.name + ": " + ins.Error(), false;
  }
  Stmt u(db_,
         "UPDATE run_items SET status = 'done', observation_id = ?, output = ?, model = ?, finish_reason = ?, "
         "input_tokens = ?, output_tokens = ?, latency_ms = ?, error = '', updated_at = "
         "strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE run_id = ? AND item_id = ? AND status = 'running' AND "
         "attempts = ?");
  u.Bind(1, r.observation_id).Bind(2, r.output).Bind(3, r.model).Bind(4, r.finish_reason).Bind(5, r.input_tokens);
  u.Bind(6, r.output_tokens).Bind(7, r.latency_ms).Bind(8, run.id).Bind(9, item_id).Bind(10, attempt);
  if (u.Step() != SQLITE_DONE || sqlite3_changes(db_) != 1) return *err = "cannot mark the item done: " + u.Error(), false;
  return tx.Commit(err);
}

bool EvalStore::FailItem(const RunRecord& run, std::string_view item_id, int attempt, std::string_view error,
                         std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt u(db_,
         "UPDATE run_items SET status = 'error', error = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
         "WHERE run_id = ? AND item_id = ? AND status = 'running' AND attempts = ?");
  if (u.Bind(1, error).Bind(2, run.id).Bind(3, item_id).Bind(4, attempt).Step() != SQLITE_DONE)
    return *err = u.Error(), false;
  return sqlite3_changes(db_) == 1 || (*err = "item " + std::string(item_id) + " is not running", false);
}

RunProgress EvalStore::Progress(const RunRecord& run) {
  std::lock_guard<std::mutex> lk(mu_);
  RunProgress p;
  p.total = run.item_count;
  Stmt q(db_,
         "SELECT COALESCE(SUM(status = 'done'), 0), COALESCE(SUM(status = 'running'), 0), "
         "COALESCE(SUM(status = 'error' AND attempts < ?), 0), COALESCE(SUM(status <> 'done' AND attempts >= ? "
         "AND status <> 'running'), 0) FROM run_items WHERE run_id = ?");
  q.Bind(1, run.max_attempts).Bind(2, run.max_attempts).Bind(3, run.id);
  if (q.Step() == SQLITE_ROW) {
    p.done = q.Int(0);
    p.running = q.Int(1);
    p.error = q.Int(2);
    p.exhausted = q.Int(3);
  }
  p.pending = p.total - p.done - p.running - p.error - p.exhausted;
  return p;
}

std::vector<RunItemRow> EvalStore::RunItems(std::string_view run_id) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt q(db_,
         "SELECT r.item_id, r.item_hash, r.status, r.trace_id, r.observation_id, r.output, r.model, "
         "r.finish_reason, r.error, r.attempts, r.input_tokens, r.output_tokens, r.latency_ms FROM run_items r "
         "JOIN runs u ON u.id = r.run_id LEFT JOIN dataset_items i ON i.dataset_id = u.dataset_id AND "
         "i.version = u.dataset_version AND i.item_id = r.item_id WHERE r.run_id = ? ORDER BY i.ordinal, r.item_id");
  q.Bind(1, run_id);
  std::vector<RunItemRow> out;
  while (q.Step() == SQLITE_ROW) {
    RunItemRow r;
    r.item_id = q.Text(0);
    r.item_hash = q.Text(1);
    r.status = q.Text(2);
    r.trace_id = q.Text(3);
    r.observation_id = q.Text(4);
    r.output = q.Text(5);
    r.model = q.Text(6);
    r.finish_reason = q.Text(7);
    r.error = q.Text(8);
    r.attempts = q.Int(9);
    r.input_tokens = q.Int(10);
    r.output_tokens = q.Int(11);
    r.latency_ms = q.Double(12);
    out.push_back(std::move(r));
  }
  return out;
}

std::vector<ScoreRow> EvalStore::RunScores(std::string_view run_id) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt q(db_,
         "SELECT item_id, name, evaluator, evaluator_version, data_type, value, string_value, comment, error, "
         "score_id, judge_model, judge_prompt, judge_prompt_version, judge_trace_id, raw_output FROM run_scores "
         "WHERE run_id = ? ORDER BY item_id, name");
  q.Bind(1, run_id);
  std::vector<ScoreRow> out;
  while (q.Step() == SQLITE_ROW) out.push_back(ReadScore(q));
  return out;
}

}  // namespace halo::scope
