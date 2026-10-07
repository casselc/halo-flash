// Stored /v1/responses in SQLite: round trip (JSON byte-identical), replace,
// pruning to the newest N, persistence across reopening, and concurrent use.

#include <gtest/gtest.h>
#include <sqlite3.h>

#include <atomic>
#include <thread>
#include <vector>

#include "scope/metadata.hpp"
#include "scope/response_store.hpp"
#include "scope_env.hpp"
#include "server/openai_response.hpp"

using namespace halo;
using namespace halo::scope;

namespace {

server::StoredResponse Make(int n) {
  std::string err;
  auto conv = json::Parse(R"([{"role": "user", "content": "hi é"}, {"role": "assistant", "content": "n)" +
                              std::to_string(n) + R"("}])",
                          &err);
  auto resp = json::Parse(R"({"id": "resp_)" + std::to_string(n) + R"(", "output": [], "temperature": 0.6})", &err);
  return {*conv, *resp};
}

}  // namespace

TEST(ScopeResponses, StoredResponsesRoundTripPruneAndPersist) {
  const std::string path = halo::scope::testing::TempDir("halo-scope-resp") + "/meta.db";
  std::string err;
  {
    auto meta = MetadataStore::Open(path, &err);
    ASSERT_TRUE(meta) << err;
    SqliteResponseStore store(*meta, 3);
    for (int i = 1; i <= 5; ++i) store.Put("resp_" + std::to_string(i), Make(i));
    EXPECT_FALSE(store.Get("resp_1"));  // pruned to the newest 3
    EXPECT_FALSE(store.Get("resp_2"));
    const auto got = store.Get("resp_5");
    ASSERT_TRUE(got);
    EXPECT_EQ(server::CompactJson(got->conversation), server::CompactJson(Make(5).conversation));
    EXPECT_EQ(server::CompactJson(got->response), server::CompactJson(Make(5).response));
    store.Put("resp_5", Make(9));  // replace
    EXPECT_EQ(server::CompactJson(store.Get("resp_5")->response), server::CompactJson(Make(9).response));
  }
  // After reopening: still there.
  auto meta = MetadataStore::Open(path, &err);
  ASSERT_TRUE(meta) << err;
  SqliteResponseStore store(*meta, 3);
  EXPECT_TRUE(store.Get("resp_3"));
  EXPECT_TRUE(store.Get("resp_4"));
  EXPECT_FALSE(store.Get("resp_unknown"));
  // Concurrent writers and readers on one store.
  std::atomic<int> found{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 4; ++t)
    ts.emplace_back([&, t] {
      for (int i = 0; i < 50; ++i) {
        const std::string id = "c" + std::to_string(t) + "_" + std::to_string(i);
        store.Put(id, Make(i));
        found += store.Get(id).has_value();
      }
    });
  for (auto& th : ts) th.join();
  EXPECT_GE(found.load(), 1);  // pruning to 3 may evict others' ids before they read
  EXPECT_TRUE(store.Get("c0_49") || store.Get("c1_49") || store.Get("c2_49") || store.Get("c3_49"));
}

// Audit F7: a write the database refuses is reported (false, with the reason), leaves nothing behind, and does not
// wedge the store: the next write succeeds. The failure is injected by a trigger from a second connection.
TEST(ScopeResponses, AFailedWriteIsReportedAndLeavesNothing) {
  const std::string path = halo::scope::testing::TempDir("halo-scope-resp-fail") + "/meta.db";
  std::string err;
  auto meta = MetadataStore::Open(path, &err);
  ASSERT_TRUE(meta) << err;
  SqliteResponseStore store(*meta, 0);
  ASSERT_TRUE(store.Put("resp_ok", Make(1), &err)) << err;
  sqlite3* db = nullptr;
  ASSERT_EQ(sqlite3_open(path.c_str(), &db), SQLITE_OK);
  const auto exec = [&](const char* sql) { return sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK; };
  ASSERT_TRUE(exec("CREATE TRIGGER refuse BEFORE INSERT ON responses BEGIN SELECT RAISE(ABORT, 'injected'); END"));
  err.clear();
  EXPECT_FALSE(store.Put("resp_bad", Make(2), &err));
  EXPECT_NE(err.find("injected"), std::string::npos) << err;
  EXPECT_FALSE(store.Get("resp_bad"));
  ASSERT_TRUE(exec("DROP TRIGGER refuse"));
  // A COMMIT that fails (a deferred foreign key, checked only at COMMIT) leaves SQLite's transaction open: unless
  // the store rolls it back, the row stays visible on its connection and every later write fails to begin.
  ASSERT_TRUE(exec("CREATE TABLE fk_parent (id INTEGER PRIMARY KEY);"
                   "CREATE TABLE fk_child (pid INTEGER REFERENCES fk_parent(id) DEFERRABLE INITIALLY DEFERRED);"
                   "CREATE TRIGGER late AFTER INSERT ON responses BEGIN INSERT INTO fk_child VALUES (42); END"));
  err.clear();
  EXPECT_FALSE(store.Put("resp_late", Make(3), &err));
  EXPECT_NE(err.find("FOREIGN KEY"), std::string::npos) << err;
  EXPECT_FALSE(store.Get("resp_late")) << "a write whose COMMIT failed is visible";
  ASSERT_TRUE(exec("DROP TRIGGER late"));
  sqlite3_close(db);
  EXPECT_TRUE(store.Put("resp_bad", Make(2), &err)) << err;  // not wedged in an open transaction
  EXPECT_TRUE(store.Get("resp_bad"));
  EXPECT_TRUE(store.Get("resp_ok"));
}
