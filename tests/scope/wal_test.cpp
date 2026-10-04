// halo-scope WAL: frames survive a torn tail (no chDB), and replaying the
// WAL after a crash at any point writes every logged batch to chDB exactly
// once, because the insert token is derived from the frame's position.

#include <hegel/gtest.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "scope/recorder.hpp"
#include "scope/wal.hpp"
#include "scope_env.hpp"

namespace gs = hegel::generators;
using namespace halo::scope;
using halo::scope::testing::Env;
using halo::scope::testing::FreshProject;
using halo::scope::testing::Read;

namespace {

std::string TempPath(const char* name) {
  return halo::scope::testing::TempDir("halo-scope-wal") + "/" + name;
}

struct SimulatedCrash : std::runtime_error {
  SimulatedCrash() : std::runtime_error("simulated crash") {}
};

/// ChdbSink that "dies" on its `crash_at`-th Write, before or after the
/// insert reaches chDB, and remembers the span ids it did insert.
class CrashingSink : public Sink {
 public:
  CrashingSink(std::unique_ptr<Conn> c, int crash_at, bool after_insert)
      : inner_(std::move(c)), crash_at_(crash_at), after_(after_insert) {}
  bool Write(Table t, std::string_view rows, std::size_t n, std::string_view token, std::string* err) override {
    const bool crash = ++calls_ == crash_at_;
    if (crash && !after_) throw SimulatedCrash();
    const bool ok = inner_.Write(t, rows, n, token, err);
    if (!ok) throw std::runtime_error("chDB insert failed: " + *err);
    rb::Reader r(rows);
    while (!r.done()) inserted.insert(DecodeObservation(r).span_id);
    if (crash) throw SimulatedCrash();
    return ok;
  }
  std::set<std::string> inserted;

 private:
  ChdbSink inner_;
  int calls_ = 0;
  int crash_at_;
  bool after_;
};

std::unique_ptr<Conn> OpenConn() {
  std::string err;
  auto c = Conn::Open(*Env().lib, Env().path, &err);
  if (!c) throw std::runtime_error(err);
  return c;
}

}  // namespace

TEST(ScopeWal, FramesSurviveATornTail) {
  hegel::test([](hegel::TestCase& tc) {
    const auto path = TempPath("wal");
    std::vector<WalFrame> want;
    std::vector<std::size_t> ends;  // file size after each frame
    {
      Wal w;
      std::string err;
      ASSERT_TRUE(w.Open(path, false, &err)) << err;
      const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 6}));
      for (int i = 0; i < n; ++i) {
        WalFrame f;
        f.table = tc.draw(gs::integers<std::uint8_t>({.min_value = 0, .max_value = 1}));
        f.rows = tc.draw(gs::integers<std::uint32_t>());
        auto b = tc.draw(gs::binary({.max_size = 300}));
        f.body.assign(b.begin(), b.end());
        f.seq = w.Append(f.table, f.rows, f.body, &err);
        ASSERT_EQ(f.seq, static_cast<std::uint64_t>(i + 1)) << err;
        want.push_back(f);
        struct stat st;
        ASSERT_EQ(stat(path.c_str(), &st), 0);
        ends.push_back(static_cast<std::size_t>(st.st_size));
      }
    }
    const std::size_t size = ends.empty() ? 24 : ends.back();
    // a crash mid-append: cut anywhere after the header
    const auto cut = tc.draw(gs::integers<std::size_t>({.min_value = 24, .max_value = size}));
    ASSERT_EQ(truncate(path.c_str(), static_cast<off_t>(cut)), 0);
    std::uint64_t id;
    std::vector<WalFrame> got;
    std::string err;
    ASSERT_TRUE(Wal::Read(path, &id, &got, &err)) << err;
    std::size_t complete = 0;
    while (complete < ends.size() && ends[complete] <= cut) ++complete;
    ASSERT_EQ(got.size(), complete);
    for (std::size_t i = 0; i < complete; ++i) {
      EXPECT_EQ(got[i].seq, want[i].seq);
      EXPECT_EQ(got[i].table, want[i].table);
      EXPECT_EQ(got[i].rows, want[i].rows);
      EXPECT_EQ(got[i].body, want[i].body);
    }
    // Reopening cuts the torn tail and keeps numbering after the survivors.
    Wal w;
    ASSERT_TRUE(w.Open(path, false, &err)) << err;
    ASSERT_EQ(w.Pending().size(), complete);
    EXPECT_GT(w.Append(0, 1, "x", &err), complete);
  });
}

TEST(ScopeWal, ReplayIsIdempotentByWalPosition) {
  HALO_REQUIRE_CHDB();
  hegel::test(
      [](hegel::TestCase& tc) {
        const std::string project = FreshProject("wal");
        const auto path = TempPath("wal");
        const auto batches = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 5}));
        const auto crash_at = tc.draw(gs::integers<int>({.min_value = 1, .max_value = batches}));
        const bool after_insert = tc.draw(gs::booleans());
        RecorderOptions opt;
        opt.writer_thread = false;
        opt.wal_path = path;
        std::set<std::string> inserted_before_crash;
        int next = 0;
        {
          CrashingSink sink(OpenConn(), crash_at, after_insert);
          Recorder rec(sink, opt);
          ASSERT_EQ(rec.error(), "");
          Producer* p = rec.NewProducer();
          try {
            for (int b = 0; b < batches; ++b) {
              const auto n = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 20}));
              for (int i = 0; i < n; ++i) {
                Observation o;
                o.project = project;
                o.span_id = "s" + std::to_string(next++);
                o.start_ns = 1790812800LL * 1000000000LL;
                ASSERT_TRUE(p->Record(o));
              }
              rec.Flush();
            }
            FAIL() << "the sink never crashed";
          } catch (const SimulatedCrash&) {
          }
          inserted_before_crash = sink.inserted;
        }
        // Optionally tear the last frame, as a crash mid-append would.
        std::uint64_t id;
        std::vector<WalFrame> frames;
        std::string err;
        ASSERT_TRUE(Wal::Read(path, &id, &frames, &err)) << err;
        ASSERT_LE(frames.size(), 1u);  // the writer resets the log after every outcome
        if (!frames.empty() && tc.draw(gs::booleans())) {
          const auto keep = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 20}));
          ASSERT_EQ(truncate(path.c_str(), static_cast<off_t>(24 + keep)), 0);
          frames.clear();
          ASSERT_TRUE(Wal::Read(path, &id, &frames, &err)) << err;
        }
        std::set<std::string> expected = inserted_before_crash;
        for (const auto& f : frames) {
          rb::Reader r(f.body);
          while (!r.done()) expected.insert(DecodeObservation(r).span_id);
        }
        // Replay, possibly dying again after re-inserting, then replay again.
        const auto replays = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 2}));
        for (int k = 0; k < replays; ++k) {
          const bool die = k + 1 < replays;
          CrashingSink sink(OpenConn(), die ? 1 : 0, true);
          try {
            Recorder rec(sink, opt);
            ASSERT_EQ(rec.error(), "");
          } catch (const SimulatedCrash&) {
            ASSERT_TRUE(die);
          }
        }
        auto r = Read("SELECT span_id FROM halo_scope.llm_observations WHERE project = " + SqlQuote(project), "TSV");
        ASSERT_TRUE(r.ok) << r.error;
        std::multiset<std::string> got;
        for (std::size_t s = 0, i = 0; i < r.data.size(); ++i)
          if (r.data[i] == '\n') {
            got.insert(r.data.substr(s, i - s));
            s = i + 1;
          }
        for (const auto& id2 : got)
          if (got.count(id2) > 1) FAIL() << "duplicate rows after replay: " << id2 << " x" << got.count(id2);
        ASSERT_EQ(std::set<std::string>(got.begin(), got.end()), expected)
            << "read back " << got.size() << " rows, expected " << expected.size();
      },
      {.test_cases = 60});
}
