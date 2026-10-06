// halo-scope datasets and runs on SQLite: the migrations (pinned checksums,
// upgrade from a database made before datasets existed), freeze
// immutability as a Hegel state machine against a model (every frozen
// version keeps its items and hash, and direct SQL writes to it are refused
// by the triggers), all-or-nothing imports, and run bindings.

#include <hegel/gtest.h>
#include <sqlite3.h>

#include <map>
#include <string>
#include <vector>

#include "scope/evals.hpp"
#include "scope/evals_schema.hpp"
#include "scope/metadata.hpp"
#include "scope/migrate.hpp"
#include "scope_env.hpp"

namespace gs = hegel::generators;
using hegel::stateful::Invariant;
using hegel::stateful::Rule;
using namespace halo::scope;

namespace {

std::string TempDb() { return halo::scope::testing::TempDir("halo-scope-evals") + "/meta.db"; }

// A raw connection, as a careless tool or a bug elsewhere would use.
struct Raw {
  sqlite3* db = nullptr;
  explicit Raw(const std::string& path) { sqlite3_open(path.c_str(), &db); }
  ~Raw() { sqlite3_close(db); }
  int Exec(const std::string& sql) { return sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr); }
};

}  // namespace

TEST(ScopeEvals, MigrationChecksumsArePinned) {
  // An applied migration must never change: a database migrated by an older
  // binary would refuse to open. Append a new migration instead.
  const std::map<std::uint32_t, std::string> pinned = {
      {1, "31ca3b03341e8210"}, {2, "43e8c842aa608bf4"}, {3, "9cf7d3f9e8d9f49d"}, {4, "a2d43224977b9420"}, {5, "7d079f6a2f6f76b3"},
  };
  const auto& all = MetadataStore::Migrations();
  ASSERT_EQ(all.size(), pinned.size());
  for (const auto& m : all) EXPECT_EQ(MigrationChecksum(m), pinned.at(m.version)) << "migration " << m.version;
}

TEST(ScopeEvals, UpgradesADatabaseFromBeforeDatasets) {
  const auto path = TempDb();
  std::string err;
  const auto& all = MetadataStore::Migrations();
  const std::vector<Migration> old(all.begin(), all.begin() + 2);  // prompts, responses
  {
    auto s = MetadataStore::Open(path, old, &err);
    ASSERT_TRUE(s) << err;
    ASSERT_EQ(s->CreatePromptVersion("judge", "Rate {{output}}", "", &err), 1) << err;
  }
  {
    Raw raw(path);
    EXPECT_NE(raw.Exec("SELECT * FROM datasets"), SQLITE_OK);  // not there yet
  }
  auto ev = EvalStore::Open(path, &err);
  ASSERT_TRUE(ev) << err;
  auto p = ev->PromptByVersion("judge", 1);
  ASSERT_TRUE(p);
  EXPECT_EQ(p->template_text, "Rate {{output}}");
  ASSERT_TRUE(ev->CreateDataset("d", "", "", &err)) << err;
  ev.reset();
  // A second store on the same file (the server opens both) applies nothing.
  auto meta = MetadataStore::Open(path, &err);
  ASSERT_TRUE(meta) << err;
  Raw raw(path);
  sqlite3_stmt* st = nullptr;
  sqlite3_prepare_v2(raw.db, "SELECT COUNT(*), MAX(version) FROM schema_migrations", -1, &st, nullptr);
  ASSERT_EQ(sqlite3_step(st), SQLITE_ROW);
  EXPECT_EQ(sqlite3_column_int(st, 0), 5);
  EXPECT_EQ(sqlite3_column_int(st, 1), 5);
  sqlite3_finalize(st);
  // An edited evaluation migration is refused, like any other.
  auto edited = all;
  edited[2].statements.back() = "SELECT 1";
  EXPECT_FALSE(MetadataStore::Open(path, edited, &err));
  EXPECT_NE(err.find("checksum mismatch"), std::string::npos) << err;
}

TEST(ScopeEvals, ImportIsAllOrNothing) {
  std::string err;
  auto s = EvalStore::Open(TempDb(), &err);
  ASSERT_TRUE(s) << err;
  ASSERT_TRUE(s->CreateDataset("qa", "questions", R"({"owner": "x"})", &err)) << err;
  std::size_t n = 0;
  EXPECT_EQ(s->ImportJsonl("qa", "{\"input\": \"a\"}\n{\"input\": \"b\"\n", &n, &err), 0);
  EXPECT_NE(err.find("line 2"), std::string::npos) << err;
  auto items = s->Items("qa", 0, &err);
  ASSERT_TRUE(items);
  EXPECT_TRUE(items->empty());
  EXPECT_EQ(s->ImportJsonl("qa", "{\"id\": \"q1\", \"input\": {\"q\": \"2+2\"}, \"expected_output\": \"4\"}\n\n"
                                 "{\"input\": \"b\", \"metadata\": {\"k\": 1}}\n",
                           &n, &err),
            1)
      << err;
  EXPECT_EQ(n, 2u);
  items = s->Items("qa", 0, &err);
  ASSERT_EQ(items->size(), 2u);
  EXPECT_EQ((*items)[0].id, "q1");
  EXPECT_EQ((*items)[0].input_json, R"({"q":"2+2"})");
  EXPECT_EQ((*items)[1].metadata_json, R"({"k":1})");
  EXPECT_FALSE(s->Freeze("nope", &err));
  EXPECT_FALSE(s->CreateDataset("qa", "", "", &err));
  EXPECT_NE(err.find("exists"), std::string::npos);
}

namespace {

struct DsModel {
  // version -> (id -> item content (input, expected)), in insertion order of ids
  std::map<std::int64_t, std::vector<std::pair<std::string, std::string>>> frozen;
  std::map<std::int64_t, std::string> hashes;
  std::vector<std::pair<std::string, std::string>> draft;
  bool has_draft = false;
  std::int64_t next_version = 1;
  bool archived = false;
};

struct FreezeMachine : hegel::stateful::StateMachine<FreezeMachine, DsModel> {
  std::string path;
  std::unique_ptr<EvalStore> store;
  std::unique_ptr<Raw> raw;

  FreezeMachine() : StateMachine({.initial_state = {}}) {
    path = TempDb();
    std::string err;
    store = EvalStore::Open(path, &err);
    if (!store || !store->CreateDataset("ds", "", "", &err)) throw std::runtime_error(err);
    raw = std::make_unique<Raw>(path);
  }

  void OpenDraftIfNeeded() {
    if (state.has_draft) return;
    state.has_draft = true;
    state.draft = state.frozen.empty() ? decltype(state.draft){} : state.frozen.rbegin()->second;
  }

  std::vector<Rule<FreezeMachine>> rules() {
    return {
        Rule<FreezeMachine>("upsert",
                            [](hegel::TestCase& tc, FreezeMachine& m) {
                              const auto id = tc.draw(gs::sampled_from<std::string>({"a", "b", "c", "d", "é"}));
                              const auto input = tc.draw(gs::text({.max_size = 8}));
                              std::vector<DatasetItem> items(1);
                              items[0].id = id;
                              items[0].input_json = halo::json::DumpPython(halo::json::Value::String(input));
                              std::string err;
                              const auto v = m.store->UpsertItems("ds", items, &err);
                              if (m.state.archived) {
                                if (v != 0) throw std::runtime_error("upsert into an archived dataset");
                                return;
                              }
                              m.OpenDraftIfNeeded();
                              auto& d = m.state.draft;
                              auto it = std::find_if(d.begin(), d.end(), [&](auto& p) { return p.first == id; });
                              const std::string content = items[0].input_json;
                              if (it == d.end()) d.emplace_back(id, content);
                              else it->second = content;
                              if (v != m.state.next_version) throw std::runtime_error("draft version " + std::to_string(v) + ": " + err);
                            }),
        Rule<FreezeMachine>("delete",
                            [](hegel::TestCase& tc, FreezeMachine& m) {
                              const auto id = tc.draw(gs::sampled_from<std::string>({"a", "b", "c", "d", "é"}));
                              std::string err;
                              const bool ok = m.store->DeleteItem("ds", id, &err);
                              auto& d = m.state.draft;
                              auto it = std::find_if(d.begin(), d.end(), [&](auto& p) { return p.first == id; });
                              const bool expect = m.state.has_draft && it != d.end() && !m.state.archived;
                              if (ok != expect) throw std::runtime_error("delete " + id + ": " + err);
                              if (ok) d.erase(it);
                            }),
        Rule<FreezeMachine>("freeze",
                            [](hegel::TestCase&, FreezeMachine& m) {
                              std::string err;
                              auto v = m.store->Freeze("ds", &err);
                              const bool expect = m.state.has_draft && !m.state.draft.empty() && !m.state.archived;
                              if (v.has_value() != expect) throw std::runtime_error("freeze: " + err);
                              if (!v) return;
                              if (v->version != m.state.next_version || v->item_count != static_cast<std::int64_t>(m.state.draft.size()))
                                throw std::runtime_error("frozen version/count differ");
                              m.state.frozen[v->version] = m.state.draft;
                              m.state.hashes[v->version] = v->content_hash;
                              m.state.has_draft = false;
                              m.state.draft.clear();
                              ++m.state.next_version;
                            }),
        Rule<FreezeMachine>("archive",
                            [](hegel::TestCase& tc, FreezeMachine& m) {
                              if (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4})) != 0) return;
                              std::string err;
                              const bool ok = m.store->Archive("ds", &err);
                              if (ok == m.state.archived) throw std::runtime_error("archive: " + err);
                              m.state.archived = true;
                            }),
        Rule<FreezeMachine>(
            "tamper",
            [](hegel::TestCase& tc, FreezeMachine& m) {
              // Every direct write to a frozen version must be refused.
              if (m.state.frozen.empty()) return;
              std::vector<std::int64_t> versions;
              for (const auto& [ver, items] : m.state.frozen) versions.push_back(ver);
              const auto pick = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = versions.size() - 1}));
              const auto v = std::to_string(versions[pick]);
              const auto which = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 6}));
              const std::string ds = "(SELECT id FROM datasets WHERE name = 'ds')";
              const std::vector<std::string> sql = {
                  "UPDATE dataset_items SET input = '\"x\"' WHERE dataset_id = " + ds + " AND version = " + v,
                  "DELETE FROM dataset_items WHERE dataset_id = " + ds + " AND version = " + v,
                  "INSERT INTO dataset_items (dataset_id, version, item_id, ordinal, input, content_hash) VALUES (" +
                      ds + ", " + v + ", 'zz', 99, '1', 'h')",
                  "UPDATE dataset_versions SET content_hash = 'x' WHERE dataset_id = " + ds + " AND version = " + v,
                  "UPDATE dataset_versions SET frozen_at = NULL, content_hash = NULL, item_count = NULL WHERE "
                  "dataset_id = " + ds + " AND version = " + v,
                  "DELETE FROM dataset_versions WHERE dataset_id = " + ds + " AND version = " + v,
                  "UPDATE dataset_items SET version = " + v + " WHERE dataset_id = " + ds +
                      " AND version = (SELECT version FROM dataset_versions WHERE frozen_at IS NULL AND dataset_id = " +
                      ds + ")",
              };
              const int rc = m.raw->Exec(sql[which]);
              const bool vacuous = which == 6 && (!m.state.has_draft || m.state.draft.empty());
              if (rc == SQLITE_OK && !vacuous) throw std::runtime_error("tamper accepted: " + sql[which]);
            }),
    };
  }

  std::vector<Invariant<FreezeMachine>> invariants() {
    return {
        Invariant<FreezeMachine>("frozen_versions_unchanged",
                                 [](const FreezeMachine& m) {
                                   std::string err;
                                   for (const auto& [v, items] : m.state.frozen) {
                                     auto got = m.store->Items("ds", v, &err);
                                     if (!got || got->size() != items.size())
                                       throw std::runtime_error("frozen version " + std::to_string(v) + " changed size");
                                     std::vector<std::string_view> fields = {"dataset.v1"};
                                     for (std::size_t i = 0; i < items.size(); ++i) {
                                       if ((*got)[i].id != items[i].first || (*got)[i].input_json != items[i].second)
                                         throw std::runtime_error("frozen version " + std::to_string(v) + " changed content");
                                       if (ItemHash((*got)[i]) != (*got)[i].content_hash)
                                         throw std::runtime_error("item hash stale");
                                       fields.push_back((*got)[i].content_hash);
                                     }
                                     if (FramedHash(fields) != m.state.hashes.at(v))
                                       throw std::runtime_error("version hash no longer recomputes");
                                   }
                                 }),
        Invariant<FreezeMachine>("versions_match_model",
                                 [](const FreezeMachine& m) {
                                   auto d = m.store->GetDataset("ds");
                                   std::size_t frozen = 0, drafts = 0;
                                   for (const auto& v : d->versions) {
                                     if (v.frozen) {
                                       ++frozen;
                                       if (m.state.hashes.at(v.version) != v.content_hash)
                                         throw std::runtime_error("listed hash differs");
                                     } else {
                                       ++drafts;
                                       if (v.item_count != static_cast<std::int64_t>(m.state.draft.size()))
                                         throw std::runtime_error("draft count differs");
                                     }
                                   }
                                   if (frozen != m.state.frozen.size() || drafts != (m.state.has_draft ? 1u : 0u))
                                     throw std::runtime_error("version list differs from the model");
                                   if (d->archived_at.empty() == m.state.archived) throw std::runtime_error("archive flag");
                                 }),
    };
  }
};

}  // namespace

TEST(ScopeEvals, FrozenVersionsAreImmutable) {
  hegel::test(
      [](hegel::TestCase& tc) {
        FreezeMachine m;
        hegel::stateful::run(m, tc);
      },
      {.test_cases = 60});
}

TEST(ScopeEvals, RunsBindToTheirContent) {
  std::string err;
  const auto path = TempDb();
  auto s = EvalStore::Open(path, &err);
  ASSERT_TRUE(s) << err;
  ASSERT_TRUE(s->CreateDataset("d", "", "", &err));
  std::size_t n;
  ASSERT_TRUE(s->ImportJsonl("d", "{\"id\":\"1\",\"input\":\"a\",\"expected_output\":\"A\"}\n", &n, &err)) << err;
  RunSpec spec;
  spec.name = "r1";
  spec.dataset = "d";
  spec.target_json = R"({"model": "m", "params": {"temperature": 0}})";
  EXPECT_FALSE(s->CreateRun(spec, nullptr, &err));  // only a draft so far
  EXPECT_NE(err.find("no frozen version"), std::string::npos) << err;
  ASSERT_TRUE(s->Freeze("d", &err)) << err;
  ASSERT_GT(s->CreateEvaluatorVersion("em", "exact", "{}", &err), 0) << err;
  spec.evaluators = {{"em", 0}};
  bool existed = true;
  auto r = s->CreateRun(spec, &existed, &err);
  ASSERT_TRUE(r) << err;
  EXPECT_FALSE(existed);
  EXPECT_EQ(r->dataset_version, 1);
  EXPECT_EQ(r->item_count, 1);
  EXPECT_EQ(r->evaluators_json, R"([{"name":"em","version":1}])");
  // Same name, same content: the same run (resume).
  spec.target_json = R"({"params": {"temperature": 0}, "model": "m"})";  // key order is not content
  auto again = s->CreateRun(spec, &existed, &err);
  ASSERT_TRUE(again) << err;
  EXPECT_TRUE(existed);
  EXPECT_EQ(again->id, r->id);
  EXPECT_EQ(again->binding, r->binding);
  // A new evaluator version pinned as "latest" changes the binding: refused.
  ASSERT_GT(s->CreateEvaluatorVersion("em", "exact", R"({"case_insensitive": true})", &err), 0);
  EXPECT_FALSE(s->CreateRun(spec, &existed, &err));
  EXPECT_NE(err.find("different binding"), std::string::npos) << err;
  spec.evaluators = {{"em", 1}};
  EXPECT_TRUE(s->CreateRun(spec, &existed, &err)) << err;
  EXPECT_EQ(*s->RecomputeBinding(*r, &err), r->binding);
  // The binding check is the last line of defence: with the triggers gone,
  // a change to the frozen items is noticed.
  Raw raw(path);
  ASSERT_EQ(raw.Exec("DROP TRIGGER dataset_items_frozen_update"), SQLITE_OK);
  ASSERT_EQ(raw.Exec("UPDATE dataset_items SET content_hash = 'x'"), SQLITE_OK);
  ASSERT_EQ(raw.Exec("DROP TRIGGER dataset_versions_frozen_update"), SQLITE_OK);
  ASSERT_EQ(raw.Exec("UPDATE dataset_versions SET content_hash = 'tampered'"), SQLITE_OK);
  EXPECT_NE(*s->RecomputeBinding(*r, &err), r->binding);
  // Runs need a frozen version even through raw SQL.
  EXPECT_NE(raw.Exec("INSERT INTO runs (id, name, dataset_id, dataset_version, target, evaluators, binding, "
                     "item_count, max_attempts) VALUES ('x', 'x', 1, 9, '{}', '[]', 'b', 0, 1)"),
            SQLITE_OK);
}
