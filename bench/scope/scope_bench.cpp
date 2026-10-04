// halo-scope write-path micro-benchmark.
//
//   scope_bench [--rows N] [--threads T] [--batch B] [--payloads] [--no-token] [--transcript]
//               [--path DIR]
//
// --no-token inserts without the deduplication token (see ChdbSink).
// --transcript adds phase 4: 20 agent sessions of 40 turns where every
// request resends the whole history as payloads, with the writer's payload
// cache off and on.
//
// 1. Producer cost: one thread records N observations through its ring
//    while the writer drains into a counting sink (no chDB). Reports ns per
//    Record() (encode + ring copy) and the p50/p99 of 1000-record chunks.
// 3. End to end into chDB (libchdb from $HALO_SCOPE_LIBCHDB): T threads
//    record N observations in total, Flush() waits until all are inserted.
//    Reports rows/s, and per 10k-row batch the writer's wall time and
//    thread CPU inside chdb_stream_insert, plus the CPU of every thread
//    except the producers (writer and chDB's own threads, which parse,
//    run the materialized view and write parts).
// 2. One pre-encoded 10k-row batch inserted 30 times through ChdbSink into
//    a fresh database, with no producers running: process CPU per batch is
//    chDB's cost alone (parse, materialized view, part write). Runs before
//    phase 3, whose 100 parts start background merges.
//
// Observations carry 12 payload references (input_refs + output_refs, 192
// bytes) plus ids, model and sampler fields. --payloads also records one
// 200-byte llm_payloads row per observation.

#include <time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <string>
#include <random>
#include <thread>
#include <vector>

#include "scope/recorder.hpp"
#include "scope/schema.hpp"

using namespace halo::scope;
using Clock = std::chrono::steady_clock;

namespace {

std::uint64_t ThreadCpuNs() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<std::uint64_t>(ts.tv_nsec);
}

std::uint64_t ProcessCpuNs() {
  timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<std::uint64_t>(ts.tv_nsec);
}

Observation Sample(const std::string& project) {
  Observation o;
  o.project = project;
  o.trace_id = "4bf92f3577b34da6a3ce929d0e0e4736";
  o.span_id = "00f067aa0ba902b7";
  o.parent_span_id = "a3ce929d0e0e4736";
  o.start_ns = 1790812800LL * 1000000000LL;
  o.end_ns = o.start_ns + 850000000;
  o.completion_start_ns = o.start_ns + 42000000;
  o.type = "GENERATION";
  o.name = "chat";
  o.session_id = "session-0001";
  o.provider = "halo-flash";
  o.model = "qwen3.8-flash-next";
  o.variant_id = "base";
  o.input_tokens = 1834;
  o.output_tokens = 212;
  o.cached_tokens = 1536;
  o.usage_details = {{"input", 1834}, {"output", 212}, {"cache_read", 1536}};
  o.prompt_name = "agent-system";
  o.prompt_version = 7;
  o.input_refs.resize(8);
  o.output_refs.resize(4);
  for (std::size_t i = 0; i < o.input_refs.size(); ++i) o.input_refs[i].fill(static_cast<std::uint8_t>(i + 1));
  for (std::size_t i = 0; i < o.output_refs.size(); ++i) o.output_refs[i].fill(static_cast<std::uint8_t>(i + 101));
  o.input_bytes = 7400;
  o.output_bytes = 880;
  o.tpot_p50_us = 9100;
  o.tpot_p90_us = 11800;
  o.tpot_p99_us = 16000;
  o.mtp_drafted = 180;
  o.mtp_accepted = 131;
  o.temperature = 0.7f;
  o.top_p = 0.95f;
  o.top_k = 40;
  o.max_tokens = 1024;
  o.finish_reason = "stop";
  return o;
}

class CountingSink : public Sink {
 public:
  bool Write(Table, std::string_view, std::size_t n, std::string_view, std::string*) override {
    rows += n;
    return true;
  }
  std::atomic<std::uint64_t> rows{0};
};

}  // namespace

int main(int argc, char** argv) {
  std::size_t rows = 1000000, threads = 4, batch = 10000;
  bool payloads = false, token = true, transcript = false;
  int transcript_sessions = 20, transcript_turns = 40;
  std::string path;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--rows") rows = std::stoull(next());
    else if (a == "--threads") threads = std::stoull(next());
    else if (a == "--batch") batch = std::stoull(next());
    else if (a == "--payloads") payloads = true;
    else if (a == "--no-token") token = false;
    else if (a == "--transcript") transcript = true;
    else if (a == "--path") path = next();
    else {
      std::fprintf(stderr, "usage: %s [--rows N] [--threads T] [--batch B] [--payloads] [--no-token] [--transcript] [--path DIR]\n", argv[0]);
      return 2;
    }
  }

  {
    Observation o = Sample("bench");
    std::string enc;
    EncodeObservation(o, enc);
    std::printf("observation row: %zu bytes RowBinary (refs %zu bytes)\n", enc.size(),
                16 * (o.input_refs.size() + o.output_refs.size()));
  }

  // --- 1. producer cost ------------------------------------------------------
  {
    CountingSink sink;
    RecorderOptions opt;
    opt.ring_bytes = 64u << 20;
    opt.batch_rows = batch;
    Recorder rec(sink, opt);
    Producer* p = rec.NewProducer();
    Observation o = Sample("bench");
    Payload pl{"bench", {}, 200, std::string(200, 'p')};
    const std::size_t n = rows;
    for (std::size_t i = 0; i < 20000; ++i) p->Record(o);  // warm up
    rec.Flush();
    std::vector<double> chunk;
    chunk.reserve(n / 1000 + 1);
    const auto t0 = Clock::now();
    auto c0 = t0;
    for (std::size_t i = 1; i <= n; ++i) {
      o.span_id[0] = static_cast<char>('a' + (i & 15));
      p->Record(o);
      if (payloads) p->Record(pl);
      if (i % 1000 == 0) {
        const auto c1 = Clock::now();
        chunk.push_back(std::chrono::duration<double, std::nano>(c1 - c0).count() / 1000);
        c0 = c1;
      }
    }
    const double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / static_cast<double>(n);
    rec.Flush();
    // The ring copy alone: the same row pre-encoded, through RecordRaw.
    std::string row;
    EncodeObservation(o, row);
    const auto r0 = Clock::now();
    for (std::size_t i = 0; i < n; ++i) p->RecordRaw(Table::kObservations, row);
    const double raw_ns =
        std::chrono::duration<double, std::nano>(Clock::now() - r0).count() / static_cast<double>(n);
    rec.Flush();
    std::printf("producer: %.1f ns per pre-encoded %zu-byte row (ring copy only)\n", raw_ns, row.size());
    const auto st = rec.stats();
    std::sort(chunk.begin(), chunk.end());
    std::printf(
        "producer: %.1f ns per observation%s (1000-record chunks: p50 %.1f, p99 %.1f, max %.1f ns/record); "
        "ring totals over both loops: accepted %llu, dropped %llu\n",
        ns, payloads ? " + payload" : "", chunk[chunk.size() / 2], chunk[chunk.size() * 99 / 100], chunk.back(),
        static_cast<unsigned long long>(st.accepted), static_cast<unsigned long long>(st.dropped));
  }

  // --- 2. end to end into chDB ----------------------------------------------
  std::string err;
  const ChdbLib* lib = ChdbLib::Load("", &err);
  if (!lib) {
    std::printf("chDB: skipped (%s)\n", err.c_str());
    return 0;
  }
  if (path.empty()) {
    std::string tmpl = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/halo-scope-bench-XXXXXX";
    if (!mkdtemp(tmpl.data())) return 1;
    path = tmpl;
  }
  auto admin = Conn::Open(*lib, path, &err);
  if (!admin || MigrateChdb(*admin, &err) < 0) {
    std::fprintf(stderr, "chDB: %s\n", err.c_str());
    return 1;
  }
  std::printf("libchdb %s at %s; data %s\n", lib->version().c_str(), lib->path().c_str(), path.c_str());
  auto wconn = Conn::Open(*lib, path, &err);
  // Declared after `admin`, so destroyed first; both close before exit
  // (libchdb 26.7.0 segfaults at exit with a connection still open).
  ChdbSink sink(std::move(wconn), token);
  std::printf("insert deduplication token: %s\n", token ? "on" : "off");
  const std::string project = "bench-" + std::to_string(Clock::now().time_since_epoch().count());
  // --- 2. one pre-encoded batch, inserted repeatedly into a quiet database -----
  {
    std::string buf;
    Observation o = Sample(project + "-direct");
    for (std::size_t i = 0; i < batch; ++i) {
      o.span_id = std::to_string(i);
      EncodeObservation(o, buf);
    }
    constexpr int kReps = 30;
    std::vector<double> wall;
    const std::uint64_t c0 = ProcessCpuNs();
    for (int k = 0; k < kReps; ++k) {
      std::string e;
      const auto w0 = Clock::now();
      if (!sink.Write(Table::kObservations, buf, batch, "direct-" + std::to_string(k) + "-" + project, &e)) {
        std::fprintf(stderr, "direct insert: %s\n", e.c_str());
        return 1;
      }
      wall.push_back(std::chrono::duration<double, std::milli>(Clock::now() - w0).count());
    }
    const double c = static_cast<double>(ProcessCpuNs() - c0) / kReps / 1e6;
    std::sort(wall.begin(), wall.end());
    std::printf("direct: %zu-row batch (%zu bytes) x %d: wall p50 %.2f ms, max %.2f ms; process CPU %.2f ms per batch "
                "(%.0f ns per row) = %.0f rows/s per core\n",
                batch, buf.size(), kReps, wall[wall.size() / 2], wall.back(), c, c * 1e6 / static_cast<double>(batch),
                static_cast<double>(batch) / (c / 1e3));
  }
  RecorderOptions popt;
  popt.ring_bytes = 16u << 20;
  popt.batch_rows = batch;
  Recorder rec(sink, popt);
  std::vector<Producer*> ps;
  for (std::size_t t = 0; t < threads; ++t) ps.push_back(rec.NewProducer());
  std::atomic<std::uint64_t> producer_cpu{0};
  const std::uint64_t cpu0 = ProcessCpuNs();
  const auto t0 = Clock::now();
  std::vector<std::thread> ts;
  for (std::size_t t = 0; t < threads; ++t)
    ts.emplace_back([&, t] {
      Observation o = Sample(project);
      Payload pl{project, {}, 200, std::string(200, 'p')};
      const std::size_t per = rows / threads;
      for (std::size_t i = 0; i < per; ++i) {
        o.span_id = std::to_string(t) + "-" + std::to_string(i);
        o.start_ns += 1000;
        // Back off when the ring is full rather than measure drops: this
        // phase measures sustained throughput. Refusals are still counted.
        while (!ps[t]->Record(o)) std::this_thread::sleep_for(std::chrono::microseconds(50));
        if (payloads) {
          std::memcpy(pl.hash.data(), &i, sizeof i);
          pl.hash[15] = static_cast<std::uint8_t>(t);
          while (!ps[t]->Record(pl)) std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
      }
      producer_cpu += ThreadCpuNs();
    });
  for (auto& t : ts) t.join();
  rec.Flush();
  const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
  const std::uint64_t cpu = ProcessCpuNs() - cpu0;
  const auto st = rec.stats();
  const std::uint64_t obs = st.rows_written[0];
  auto r = admin->Query("SELECT count() FROM halo_scope.llm_observations WHERE project = " + SqlQuote(project), "CSV");
  const double batches = static_cast<double>(st.batches);
  const double rest = static_cast<double>(cpu - producer_cpu.load());
  std::printf("pipeline: %llu observations%s in %.2f s = %.0f rows/s (%zu threads, batch %zu); read back %s",
              static_cast<unsigned long long>(obs), payloads ? " + payloads" : "", secs,
              static_cast<double>(obs) / secs, threads, batch, r.ok ? r.data.c_str() : r.error.c_str());
  std::printf(
      "pipeline: %llu batches; per batch: stream-insert wall %.2f ms, writer-thread CPU in insert %.2f ms; "
      "CPU of everything but producers %.2f ms (%.0f ns per row); producers %.0f ns CPU per row; "
      "failed %llu, ring-full refusals %llu\n",
      static_cast<unsigned long long>(st.batches), st.sink_ns / batches / 1e6, st.sink_thread_cpu_ns / batches / 1e6,
      rest / batches / 1e6, rest / static_cast<double>(st.accepted),
      static_cast<double>(producer_cpu.load()) / static_cast<double>(st.accepted),
      static_cast<unsigned long long>(st.rows_failed), static_cast<unsigned long long>(st.dropped));

  rec.Stop();

  // --- 4. agent transcripts: every turn resends the history ------------------
  if (transcript) {
    for (const bool cache : {false, true}) {
      const std::string tp = project + (cache ? "-tc" : "-tn");
      RecorderOptions opt;
      opt.batch_rows = batch;
      opt.payload_cache_entries = cache ? (1u << 20) : 0;
      Recorder tr(sink, opt);
      Producer* p = tr.NewProducer();
      std::mt19937_64 rng(7);
      auto text = [&](std::size_t lo, std::size_t hi) {
        std::string s(lo + rng() % (hi - lo + 1), ' ');
        for (auto& ch : s) ch = static_cast<char>('a' + rng() % 26);
        return s;
      };
      const std::string system = text(6000, 6000);  // shared by every session
      std::uint64_t offered_rows = 0, offered_bytes = 0;
      const std::uint64_t c0 = ProcessCpuNs(), m0 = ThreadCpuNs();
      const auto w0 = Clock::now();
      for (int sidx = 0; sidx < transcript_sessions; ++sidx) {
        std::vector<std::string> history = {system};
        for (int turn = 0; turn < transcript_turns; ++turn) {
          history.push_back(text(200, 800));                             // user
          if (turn % 3 == 2) history.push_back(text(1000, 8000));        // tool result
          Observation o = Sample(tp);
          o.span_id = std::to_string(sidx) + "-" + std::to_string(turn);
          o.input_refs.clear();
          for (const auto& m : history) {  // the request carries the whole history
            Payload pl{tp, PayloadHash(m), static_cast<std::uint32_t>(m.size()), m};
            while (!p->Record(pl)) std::this_thread::sleep_for(std::chrono::microseconds(50));
            o.input_refs.push_back(pl.hash);
            ++offered_rows;
            offered_bytes += m.size();
          }
          history.push_back(text(300, 2000));  // assistant output
          Payload out{tp, PayloadHash(history.back()), static_cast<std::uint32_t>(history.back().size()), history.back()};
          while (!p->Record(out)) std::this_thread::sleep_for(std::chrono::microseconds(50));
          ++offered_rows;
          offered_bytes += history.back().size();
          o.output_refs = {out.hash};
          while (!p->Record(o)) std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
      }
      tr.Flush();
      const double secs = std::chrono::duration<double>(Clock::now() - w0).count();
      const double producer_ms = (ThreadCpuNs() - m0) / 1e6;
      const double other_ms = (ProcessCpuNs() - c0) / 1e6 - producer_ms;
      const auto st = tr.stats();
      auto q = admin->Query("SELECT count(), sum(bytes) FROM halo_scope.llm_payloads WHERE project = " + SqlQuote(tp), "TSV");
      std::printf(
          "transcript (%d sessions x %d turns, cache %s): payloads offered %llu rows / %.1f MB; sent %llu rows; "
          "deduplicated %llu; in llm_payloads now (rows, bytes) %s"
          "  %.2f s; CPU outside the producer %.0f ms; producer %.0f ms\n",
          transcript_sessions, transcript_turns, cache ? "on" : "off", static_cast<unsigned long long>(offered_rows),
          offered_bytes / 1e6, static_cast<unsigned long long>(st.rows_written[1]),
          static_cast<unsigned long long>(st.payloads_deduped), q.ok ? q.data.c_str() : q.error.c_str(), secs,
          other_ms, producer_ms);
      tr.Stop();
    }
  }
  return 0;  // the sink's and admin's connections close here, before exit
}
