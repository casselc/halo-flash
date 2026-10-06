#include "scope/metadata.hpp"

#include <sqlite3.h>

#include <algorithm>

#include "scope/evals_schema.hpp"
#include "scope/sqlite_util.hpp"

namespace halo::scope {

namespace {

const Migration kPrompts{
    1,
    "prompts",
    {
        R"(CREATE TABLE prompts (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
))",
        R"(CREATE TABLE prompt_versions (
  prompt_id INTEGER NOT NULL REFERENCES prompts(id),
  version INTEGER NOT NULL CHECK (version > 0),
  template TEXT NOT NULL,
  config TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (prompt_id, version)
) WITHOUT ROWID)",
        // Versions are immutable facts; labels are the only mutable pointer.
        R"(CREATE TRIGGER prompt_versions_immutable BEFORE UPDATE ON prompt_versions
BEGIN SELECT RAISE(ABORT, 'prompt versions are immutable'); END)",
        R"(CREATE TABLE labels (
  prompt_id INTEGER NOT NULL,
  label TEXT NOT NULL CHECK (label <> ''),
  version INTEGER NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (prompt_id, label),
  FOREIGN KEY (prompt_id, version) REFERENCES prompt_versions(prompt_id, version)
) WITHOUT ROWID)",
    }};

// Stored /v1/responses (P7): the conversation through each response's own
// turn and the response object, both JSON. Pruned oldest first.
const Migration kResponses{
    2,
    "responses",
    {
        R"(CREATE TABLE responses (
  id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL,
  conversation TEXT NOT NULL,
  response TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
) WITHOUT ROWID)",
        R"(CREATE INDEX responses_seq ON responses (seq))",
    }};

using sql::Stmt;

}  // namespace

const std::vector<Migration>& MetadataStore::Migrations() {
  static const std::vector<Migration> m = [] {
    std::vector<Migration> all = {kPrompts, kResponses};
    for (const auto& e : EvalMigrations()) all.push_back(e);
    return all;
  }();
  return m;
}

std::unique_ptr<MetadataStore> MetadataStore::Open(const std::string& path, std::string* err) {
  return Open(path, Migrations(), err);
}

std::unique_ptr<MetadataStore> MetadataStore::Open(const std::string& path, const std::vector<Migration>& migrations,
                                                   std::string* err) {
  std::unique_ptr<MetadataStore> s(new MetadataStore());
  s->db_ = sql::OpenMigrated(path, migrations, err);
  if (!s->db_) return nullptr;
  return s;
}

MetadataStore::~MetadataStore() {
  if (db_) sqlite3_close_v2(db_);
}

bool MetadataStore::Exec(const char* statement, std::string* err) { return sql::Exec(db_, statement, err); }

std::int64_t MetadataStore::CreatePromptVersion(std::string_view name, std::string_view template_text,
                                                std::string_view config_json, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!Exec("BEGIN IMMEDIATE", err)) return 0;
  std::int64_t version = 0;
  bool ok = true;
  {
    Stmt a(db_, "INSERT INTO prompts (name) VALUES (?) ON CONFLICT(name) DO NOTHING");
    ok = a.Bind(1, name).Step() == SQLITE_DONE;
  }
  if (ok) {
    Stmt v(db_,
           "INSERT INTO prompt_versions (prompt_id, version, template, config) "
           "SELECT p.id, COALESCE((SELECT MAX(version) FROM prompt_versions WHERE prompt_id = p.id), 0) + 1, ?, ? "
           "FROM prompts p WHERE p.name = ? RETURNING version");
    v.Bind(1, template_text).Bind(2, config_json.empty() ? std::string_view("{}") : config_json).Bind(3, name);
    if (v.Step() == SQLITE_ROW) version = v.Int(0);
    ok = version > 0 && v.Step() == SQLITE_DONE;
  }
  if (ok) {
    Stmt l(db_,
           "INSERT INTO labels (prompt_id, label, version) SELECT id, 'latest', ? FROM prompts WHERE name = ? "
           "ON CONFLICT(prompt_id, label) DO UPDATE SET version = excluded.version, updated_at = excluded.updated_at");
    ok = l.Bind(1, version).Bind(2, name).Step() == SQLITE_DONE;
  }
  if (!ok) {
    *err = sqlite3_errmsg(db_);
    std::string ignored;
    Exec("ROLLBACK", &ignored);
    return 0;
  }
  return Exec("COMMIT", err) ? version : 0;
}

bool MetadataStore::SetLabel(std::string_view name, std::string_view label, std::int64_t version, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt l(db_,
         "INSERT INTO labels (prompt_id, label, version) SELECT id, ?, ? FROM prompts WHERE name = ? "
         "ON CONFLICT(prompt_id, label) DO UPDATE SET version = excluded.version, updated_at = excluded.updated_at");
  if (l.Bind(1, label).Bind(2, version).Bind(3, name).Step() != SQLITE_DONE) {
    *err = sqlite3_errmsg(db_);
    return false;
  }
  if (sqlite3_changes(db_) == 0) {
    *err = "no prompt named " + std::string(name);
    return false;
  }
  return true;
}

std::optional<PromptVersion> MetadataStore::GetByLabel(std::string_view name, std::string_view label) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt q(db_,
         "SELECT v.version, v.template, v.config FROM prompts p JOIN labels l ON l.prompt_id = p.id "
         "JOIN prompt_versions v ON v.prompt_id = p.id AND v.version = l.version WHERE p.name = ? AND l.label = ?");
  q.Bind(1, name).Bind(2, label);
  if (q.Step() != SQLITE_ROW) return std::nullopt;
  return PromptVersion{std::string(name), q.Int(0), q.Text(1), q.Text(2)};
}

std::optional<PromptVersion> MetadataStore::GetByVersion(std::string_view name, std::int64_t version) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt q(db_,
         "SELECT v.version, v.template, v.config FROM prompts p JOIN prompt_versions v ON v.prompt_id = p.id "
         "WHERE p.name = ? AND v.version = ?");
  q.Bind(1, name).Bind(2, version);
  if (q.Step() != SQLITE_ROW) return std::nullopt;
  return PromptVersion{std::string(name), q.Int(0), q.Text(1), q.Text(2)};
}

std::vector<std::string> MetadataStore::LabelsOf(std::string_view name, std::int64_t version) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt q(db_,
         "SELECT l.label FROM prompts p JOIN labels l ON l.prompt_id = p.id WHERE p.name = ? AND l.version = ? "
         "ORDER BY l.label");
  q.Bind(1, name).Bind(2, version);
  std::vector<std::string> out;
  while (q.Step() == SQLITE_ROW) out.push_back(q.Text(0));
  return out;
}

bool MetadataStore::PutResponse(std::string_view id, std::string_view conversation, std::string_view response,
                                std::size_t keep, std::string* err) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!Exec("BEGIN IMMEDIATE", err)) return false;
  bool ok;
  {
    Stmt p(db_,
           "INSERT INTO responses (id, seq, conversation, response) "
           "VALUES (?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM responses), ?, ?) "
           "ON CONFLICT(id) DO UPDATE SET conversation = excluded.conversation, response = excluded.response");
    ok = p.Bind(1, id).Bind(2, conversation).Bind(3, response).Step() == SQLITE_DONE;
  }
  if (ok && keep > 0) {
    Stmt d(db_,
           "DELETE FROM responses WHERE seq <= (SELECT MAX(seq) FROM responses) - ?");
    ok = d.Bind(1, static_cast<std::int64_t>(keep)).Step() == SQLITE_DONE;
  }
  if (!ok) {
    *err = sqlite3_errmsg(db_);
    std::string ignored;
    Exec("ROLLBACK", &ignored);
    return false;
  }
  return Exec("COMMIT", err);
}

std::optional<std::pair<std::string, std::string>> MetadataStore::GetResponse(std::string_view id) {
  std::lock_guard<std::mutex> lk(mu_);
  Stmt q(db_, "SELECT conversation, response FROM responses WHERE id = ?");
  q.Bind(1, id);
  if (q.Step() != SQLITE_ROW) return std::nullopt;
  return std::make_pair(q.Text(0), q.Text(1));
}

std::string MetadataStore::JournalMode() {
  Stmt q(db_, "PRAGMA journal_mode");
  return q.Step() == SQLITE_ROW ? q.Text(0) : std::string();
}

}  // namespace halo::scope
