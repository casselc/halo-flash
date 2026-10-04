// halo-scope storage: RowBinary encode -> chdb_stream_insert_n into the Null
// staging table -> materialized view -> typed MergeTree -> SELECT FORMAT
// RowBinary -> decode must reproduce the generated records exactly; plus
// the migrations and insert-token deduplication.

#include <hegel/gtest.h>

#include <bit>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "scope/records.hpp"
#include "scope/schema.hpp"
#include "scope_env.hpp"

namespace gs = hegel::generators;
using namespace halo::scope;
using halo::scope::testing::Env;
using halo::scope::testing::FreshProject;
using halo::scope::testing::Read;

namespace {

// 2026-10-01T00:00:00Z. Start times stay within a week of it (or at the
// epoch) so one insert touches few daily partitions.
constexpr std::int64_t kBaseNs = 1790812800LL * 1000000000LL;
constexpr std::int64_t kWeekNs = 7LL * 86400 * 1000000000LL;

std::string DrawString(hegel::TestCase& tc, std::size_t max = 40) {
  switch (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4}))) {
    case 0: return "";
    case 1: return tc.draw(gs::text({.max_size = max}));  // Unicode
    case 2: {                                              // any bytes, NULs included
      auto b = tc.draw(gs::binary({.max_size = max}));
      return std::string(b.begin(), b.end());
    }
    case 3: {  // long: crosses the 1-, 2- and 3-byte LEB128 length boundaries
      const auto n = tc.draw(gs::sampled_from<std::size_t>({127, 128, 255, 256, 16383, 16384, 70000}));
      return std::string(n, tc.draw(gs::sampled_from<char>({'a', '\0', '\xff'})));
    }
    default: return tc.draw(gs::sampled_from<std::string>({"GENERATION", "é", "\xe2\x80\x8b", "a\tb\nc", "'\\"}));
  }
}

std::uint64_t DrawCount(hegel::TestCase& tc) {
  return tc.draw(gs::one_of<std::uint64_t>({gs::sampled_from<std::uint64_t>({0, 1, 0xffffffffull, 0x100000000ull,
                                                                             std::numeric_limits<std::uint64_t>::max()}),
                                            gs::integers<std::uint64_t>()}));
}

template <typename T>
T DrawU(hegel::TestCase& tc) {
  return tc.draw(gs::integers<T>());
}

float DrawFloat(hegel::TestCase& tc) {
  return std::bit_cast<float>(tc.draw(gs::integers<std::uint32_t>()));  // NaN payloads and -0 included
}

rb::Hash16 DrawHash(hegel::TestCase& tc) {
  rb::Hash16 h;
  for (auto& b : h) b = tc.draw(gs::integers<std::uint8_t>());
  return h;
}

std::vector<rb::Hash16> DrawHashes(hegel::TestCase& tc) {
  std::vector<rb::Hash16> v(tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 4})));
  for (auto& h : v) h = DrawHash(tc);
  return v;
}

std::vector<std::string> DrawStrings(hegel::TestCase& tc) {
  std::vector<std::string> v(tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 3})));
  for (auto& s : v) s = DrawString(tc, 12);
  return v;
}

Observation DrawObservation(hegel::TestCase& tc, const std::string& project, std::size_t i) {
  Observation o;
  o.project = project;
  o.trace_id = DrawString(tc);
  o.span_id = DrawString(tc) + "#" + std::to_string(i);  // unique within the case
  o.parent_span_id = DrawString(tc);
  o.start_ns = tc.draw(gs::one_of<std::int64_t>(
      {gs::just<std::int64_t>(0), gs::integers<std::int64_t>({.min_value = kBaseNs, .max_value = kBaseNs + kWeekNs})}));
  o.end_ns = DrawU<std::int64_t>(tc);
  o.completion_start_ns = DrawU<std::int64_t>(tc);
  o.type = DrawString(tc);
  o.name = DrawString(tc, 400);
  o.user_id = DrawString(tc);
  o.session_id = DrawString(tc);
  o.tags = DrawStrings(tc);
  o.status_code = DrawString(tc);
  o.level = DrawString(tc);
  o.provider = DrawString(tc);
  o.model = DrawString(tc);
  o.variant_id = DrawString(tc);
  o.adapter_id = DrawString(tc);
  o.input_tokens = DrawCount(tc);
  o.output_tokens = DrawCount(tc);
  o.cached_tokens = DrawCount(tc);
  const auto keys = tc.draw(gs::sets(gs::sampled_from<std::string>({"input", "output", "cache_read", "reasoning", "",
                                                                    "\xc3\xa9t\xc3\xa9", "x"}),
                                     {.max_size = 5}));
  for (const auto& k : keys) o.usage_details.emplace_back(k, DrawCount(tc));
  o.prompt_name = DrawString(tc);
  o.prompt_version = DrawU<std::uint32_t>(tc);
  o.tool_name = DrawString(tc);
  o.tool_call_ids = DrawStrings(tc);
  o.system_ref = DrawHashes(tc);
  o.input_refs = DrawHashes(tc);
  o.output_refs = DrawHashes(tc);
  o.input_bytes = DrawCount(tc);
  o.output_bytes = DrawCount(tc);
  o.truncated = DrawU<std::uint8_t>(tc);
  o.tpot_p50_us = DrawU<std::uint32_t>(tc);
  o.tpot_p90_us = DrawU<std::uint32_t>(tc);
  o.tpot_p99_us = DrawU<std::uint32_t>(tc);
  o.mtp_drafted = DrawU<std::uint32_t>(tc);
  o.mtp_accepted = DrawU<std::uint32_t>(tc);
  o.temperature = DrawFloat(tc);
  o.top_p = DrawFloat(tc);
  o.min_p = DrawFloat(tc);
  o.top_k = DrawU<std::uint32_t>(tc);
  o.seed = DrawCount(tc);
  o.max_tokens = DrawU<std::uint32_t>(tc);
  o.finish_reason = DrawString(tc);
  o.decision_answer = DrawString(tc);
  return o;
}

// Field-wise equality with floats compared by bits (NaN, -0).
bool SameBits(const Observation& a, const Observation& b) {
  Observation x = a, y = b;
  for (auto [p, q] : {std::pair{&x.temperature, &y.temperature}, {&x.top_p, &y.top_p}, {&x.min_p, &y.min_p}}) {
    if (std::bit_cast<std::uint32_t>(*p) != std::bit_cast<std::uint32_t>(*q)) return false;
    *p = *q = 0;
  }
  return x == y;
}

std::string SelectObservations(const std::string& project) {
  return std::string("SELECT ") + kObservationColumns + " FROM halo_scope.llm_observations WHERE project = " +
         SqlQuote(project);
}

Conn& Writer() { return *Env().writer; }

Conn::Result InsertObservations(const std::string& rows, std::string_view token = {}) {
  std::string ins = std::string("INSERT INTO halo_scope.llm_observations_in (") + kObservationColumns + ")";
  if (!token.empty())
    ins += " SETTINGS deduplicate_blocks_in_dependent_materialized_views = 1, insert_deduplication_token = " +
           SqlQuote(token);
  return Writer().StreamInsert(ins, "RowBinary", rows);
}

}  // namespace

TEST(ScopeRecords, ObservationRoundTripsThroughChdb) {
  HALO_REQUIRE_CHDB();
  hegel::test([](hegel::TestCase& tc) {
    const std::string project = FreshProject("rt");
    const auto n = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 6}));
    std::map<std::string, Observation> want;
    std::string rows;
    for (std::size_t i = 0; i < n; ++i) {
      auto o = DrawObservation(tc, project, i);
      EncodeObservation(o, rows);
      want.emplace(o.span_id, std::move(o));
    }
    // Local inverse first: a mismatch here is the codec, not chDB.
    {
      rb::Reader r(rows);
      for (std::size_t i = 0; i < n; ++i) DecodeObservation(r);
      ASSERT_TRUE(r.done());
    }
    auto ins = InsertObservations(rows);
    ASSERT_TRUE(ins.ok) << ins.error;
    auto sel = Read(SelectObservations(project), "RowBinary");
    ASSERT_TRUE(sel.ok) << sel.error;
    rb::Reader r(sel.data);
    std::size_t got = 0;
    while (!r.done()) {
      const Observation o = DecodeObservation(r);
      ++got;
      auto it = want.find(o.span_id);
      ASSERT_NE(it, want.end()) << "unexpected span_id read back (" << o.span_id.size() << " bytes)";
      ASSERT_TRUE(SameBits(it->second, o)) << "row " << got << " differs after the chDB round trip";
    }
    ASSERT_EQ(got, n);
  });
}

TEST(ScopeRecords, PayloadRoundTripsThroughChdb) {
  HALO_REQUIRE_CHDB();
  hegel::test([](hegel::TestCase& tc) {
    const std::string project = FreshProject("pl");
    Payload p;
    p.project = project;
    p.hash = DrawHash(tc);
    p.content = DrawString(tc, 2000);
    p.bytes = static_cast<std::uint32_t>(p.content.size());
    std::string rows;
    EncodePayload(p, rows);
    auto ins = Writer().StreamInsert(std::string("INSERT INTO halo_scope.llm_payloads_in (") + kPayloadColumns + ")",
                                     "RowBinary", rows);
    ASSERT_TRUE(ins.ok) << ins.error;
    auto sel = Read(std::string("SELECT ") + kPayloadColumns + " FROM halo_scope.llm_payloads WHERE project = " +
                        SqlQuote(project),
                    "RowBinary");
    ASSERT_TRUE(sel.ok) << sel.error;
    rb::Reader r(sel.data);
    ASSERT_FALSE(r.done());
    ASSERT_EQ(DecodePayload(r), p);
    ASSERT_TRUE(r.done());
  });
}

TEST(ScopeRecords, PayloadHashIsBlake3Truncated) {
  // BLAKE3("") = af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262
  const rb::Hash16 want = {0xaf, 0x13, 0x49, 0xb9, 0xf5, 0xf9, 0xa1, 0xa6,
                           0xa0, 0x40, 0x4d, 0xea, 0x36, 0xdc, 0xc9, 0x49};
  EXPECT_EQ(PayloadHash(""), want);
  EXPECT_NE(PayloadHash("a"), PayloadHash("b"));
}

TEST(ScopeRecords, InsertTokenMakesReplayExactlyOnce) {
  HALO_REQUIRE_CHDB();
  const std::string project = FreshProject("dedup");
  std::string rows;
  for (int i = 0; i < 3; ++i) {
    Observation o;
    o.project = project;
    o.span_id = "s" + std::to_string(i);
    o.start_ns = kBaseNs;
    EncodeObservation(o, rows);
  }
  auto count = [&] {
    auto r = Read("SELECT count() FROM halo_scope.llm_observations WHERE project = " + SqlQuote(project), "CSV");
    EXPECT_TRUE(r.ok) << r.error;
    return std::stoi(r.data);
  };
  ASSERT_TRUE(InsertObservations(rows, project + "-b1").ok);
  ASSERT_TRUE(InsertObservations(rows, project + "-b1").ok);  // a WAL replay of the same batch
  EXPECT_EQ(count(), 3);
  ASSERT_TRUE(InsertObservations(rows, project + "-b2").ok);  // a different batch with equal bytes
  EXPECT_EQ(count(), 6);
}

TEST(ScopeSchema, MigrationsAreOrderedIdempotentAndChecksummed) {
  HALO_REQUIRE_CHDB();
  auto& e = Env();
  std::lock_guard<std::mutex> lk(e.reader_mu);
  const auto& known = ChdbMigrations();
  for (std::size_t i = 1; i < known.size(); ++i) EXPECT_LT(known[i - 1].version, known[i].version);
  std::string err;
  EXPECT_EQ(MigrateChdb(*e.reader, &err), 0) << err;  // already applied at startup
  auto r = e.reader->Query("SELECT count() FROM halo_scope.schema_migrations", "CSV");
  ASSERT_TRUE(r.ok);
  EXPECT_EQ(std::stoul(r.data), known.size());

  auto edited = known;
  edited[1].statements.back() = "SELECT 1";
  EXPECT_EQ(MigrateChdb(*e.reader, edited, &err), -1);
  EXPECT_NE(err.find("checksum mismatch"), std::string::npos) << err;

  auto fewer = known;
  fewer.pop_back();
  EXPECT_EQ(MigrateChdb(*e.reader, fewer, &err), -1);
  EXPECT_NE(err.find("newer"), std::string::npos) << err;

  r = e.reader->Query(
      "SELECT name, engine FROM system.tables WHERE database = 'halo_scope' ORDER BY name", "TabSeparated");
  ASSERT_TRUE(r.ok);
  for (const char* want :
       {"llm_observations\tMergeTree", "llm_observations_in\tNull", "llm_observations_mv\tMaterializedView",
        "llm_payloads\tReplacingMergeTree", "llm_scores\tMergeTree", "llm_traces\tMergeTree"})
    EXPECT_NE(r.data.find(want), std::string::npos) << want << "\n" << r.data;
  r = e.reader->Query(
      "SELECT name FROM system.tables WHERE database = 'halo_scope' AND engine LIKE '%MergeTree' AND name != "
      "'schema_migrations' AND create_table_query NOT LIKE '%non_replicated_deduplication_window%'",
      "TabSeparated");
  ASSERT_TRUE(r.ok);
  EXPECT_EQ(r.data, "") << "MV targets without a deduplication window";
}

TEST(ScopeSchema, ScoresAreAppendOnlyFactsResolvedAtQueryTime) {
  HALO_REQUIRE_CHDB();
  const std::string p = SqlQuote(FreshProject("sc"));
  // assert s1, correct it with s2 (supersedes s1), retract s3.
  auto& w = Writer();
  for (const std::string v :
       {"('s1', '', 'assert', 'acc', 0.5, '2026-10-01 00:00:01')", "('s2', 's1', 'assert', 'acc', 0.9, '2026-10-01 00:00:02')",
        "('s3', '', 'assert', 'ok', 1, '2026-10-01 00:00:03')", "('s4', 's3', 'retract', 'ok', 0, '2026-10-01 00:00:04')"}) {
    auto e = w.Exec("INSERT INTO halo_scope.llm_scores_in (project, score_id, supersedes, kind, name, value, event_time) "
                    "SELECT " + p + ", * FROM values('a String, b String, c String, d String, e Float64, f DateTime64(9, \\'UTC\\')', " + v + ")");
    ASSERT_EQ(e, "");
  }
  // Live facts: not superseded by any later fact; retract facts hide their target.
  auto r = Read("SELECT name, value FROM halo_scope.llm_scores WHERE project = " + p +
                    " AND kind = 'assert' AND score_id NOT IN (SELECT supersedes FROM halo_scope.llm_scores WHERE project = " +
                    p + ") ORDER BY name",
                "CSV");
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(r.data, "\"acc\",0.9\n");
}
