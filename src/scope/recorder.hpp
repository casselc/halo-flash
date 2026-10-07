#ifndef HALO_SCOPE_RECORDER_HPP_
#define HALO_SCOPE_RECORDER_HPP_

// The halo-scope write path: per-thread bounded rings feed one writer
// thread, which batches records per table and hands each batch to a Sink
// (chDB in production, an in-memory sink in tests). Producers encode the
// staging-table RowBinary row themselves, so the writer only concatenates.
//
// Accounting, never silent: for every producer,
//   accepted + dropped == records offered,
// and for the recorder after Stop(),
//   rows_written + rows_failed + payloads_deduped == sum(accepted).
//
// With a WAL (RecorderOptions::wal_path), each batch is appended to the log
// before it is inserted, with a token derived from its log position, and
// frames left by a crash are replayed at startup (exactly once in chDB).
// With a payload cache, a payload whose (project, hash) is already durably
// written, or already in the batch being built, is not sent again.
//
// A producer leased to a thread (Recorder::Lease) is retired when the lease
// dies, at the latest when its thread exits: the writer drains what its ring
// still holds and then frees it, so threads that come and go (one per HTTP
// connection) do not each leave a ring behind. Its counts carry over into
// stats(), so the accounting above still holds.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <deque>
#include <vector>

#include "scope/chdb.hpp"
#include "scope/records.hpp"
#include "scope/ring.hpp"
#include "scope/wal.hpp"

namespace halo::scope {

/// Where batches go. Called only from the writer thread.
class Sink {
 public:
  virtual ~Sink() = default;
  /// `rows` is `nrows` concatenated staging-table RowBinary rows for
  /// `table`. `token` is unique per batch (insert deduplication token).
  /// Returns false and sets `*err` when the batch was not written.
  virtual bool Write(Table table, std::string_view rows, std::size_t nrows, std::string_view token,
                     std::string* err) = 0;
};

/// Inserts each batch into `halo_scope.<table>_in` through
/// chdb_stream_insert_n, FORMAT RowBinary. With `dedup_token`, the batch's
/// token goes in insert_deduplication_token with
/// deduplicate_blocks_in_dependent_materialized_views = 1, so replaying a
/// batch is exactly-once. Measured: on libchdb 26.7.0 the token about
/// doubled the CPU of a 10k-row insert through the staging MV; on 26.9.0
/// the difference is within noise.
class ChdbSink : public Sink {
 public:
  explicit ChdbSink(std::unique_ptr<Conn> conn, bool dedup_token = true)
      : conn_(std::move(conn)), dedup_token_(dedup_token) {}
  bool Write(Table table, std::string_view rows, std::size_t nrows, std::string_view token,
             std::string* err) override;

 private:
  std::unique_ptr<Conn> conn_;
  bool dedup_token_;
};

class Recorder;

namespace detail {
/// Whether the Recorder is still there, for leases that may outlive it.
struct RecorderLife {
  std::mutex mu;
  Recorder* rec = nullptr;  // null once the Recorder is being destroyed
};
}  // namespace detail

/// One producer thread's handle. Not thread-safe: one thread records
/// through it. Owned by the Recorder; valid until the Recorder dies.
class Producer {
 public:
  /// Never blocks. Returns false when the record was dropped (counted).
  bool Record(const Observation& o) {
    scratch_.clear();
    EncodeObservation(o, scratch_);
    return RecordRaw(Table::kObservations, scratch_);
  }
  bool Record(const Payload& p) {
    scratch_.clear();
    EncodePayload(p, scratch_);
    return RecordRaw(Table::kPayloads, scratch_);
  }
  bool Record(const Score& s) {
    scratch_.clear();
    EncodeScore(s, scratch_);
    return RecordRaw(Table::kScores, scratch_);
  }
  bool Record(const TraceFact& t) {
    scratch_.clear();
    EncodeTraceFact(t, scratch_);
    return RecordRaw(Table::kTraces, scratch_);
  }
  /// One already-encoded staging row.
  bool RecordRaw(Table t, std::string_view row);

  std::uint64_t dropped() const { return ring_.dropped(); }
  std::uint64_t accepted() const { return ring_.accepted(); }

 private:
  friend class Recorder;
  friend class ProducerLease;
  Producer(Recorder* rec, std::size_t ring_bytes) : rec_(rec), ring_(ring_bytes) { scratch_.reserve(4096); }
  Recorder* rec_;
  Ring ring_;
  std::string scratch_;
  /// Set (release) by its lease once its thread records no more; the writer frees the producer after the drain
  /// that follows its acquire load, so every record pushed before the retirement is written first.
  std::atomic<bool> retired_{false};
};

/// A thread's ownership of a producer: destroying the lease (for a
/// thread_local one, when the thread exits) retires the producer; the
/// writer drains its ring, then frees it. A lease destroyed after its
/// Recorder does nothing (the Recorder freed the producer). Movable.
class ProducerLease {
 public:
  ProducerLease() = default;
  ProducerLease(ProducerLease&& o) noexcept : life_(std::move(o.life_)), p_(o.p_) { o.p_ = nullptr; }
  ProducerLease& operator=(ProducerLease&& o) noexcept {
    if (this != &o) {
      Release();
      life_ = std::move(o.life_);
      p_ = o.p_;
      o.p_ = nullptr;
    }
    return *this;
  }
  ProducerLease(const ProducerLease&) = delete;
  ProducerLease& operator=(const ProducerLease&) = delete;
  ~ProducerLease() { Release(); }
  [[nodiscard]] Producer* get() const { return p_; }
  /// Retires the producer now (the thread must not use it afterwards).
  void Release();

 private:
  friend class Recorder;
  ProducerLease(std::shared_ptr<detail::RecorderLife> life, Producer* p) : life_(std::move(life)), p_(p) {}
  std::shared_ptr<detail::RecorderLife> life_;
  Producer* p_ = nullptr;
};

struct RecorderOptions {
  std::size_t ring_bytes = 1 << 20;  // per producer
  std::size_t batch_rows = 10000;    // per table
  /// Time bound on a batch. Per-row chDB CPU falls from ~42 us at 250-row
  /// batches to ~5 us at 10k (libchdb 26.9.0), so at engine volumes
  /// (thousands of rows/s) a 1 s bound keeps batches in the thousands.
  std::chrono::milliseconds flush_interval{1000};
  /// Write-ahead log path; empty: none (a crash loses the batch in flight).
  std::string wal_path;
  bool wal_sync = false;  // fdatasync each frame (power loss, not just a crash)
  /// Payloads remembered as durably written, (project, hash) keys, oldest
  /// evicted first; 0 turns payload deduplication off.
  std::size_t payload_cache_entries = 1 << 20;
  /// false: no writer thread; the owner drives it with Pump(), Flush() and
  /// Stop() on one thread, and batches are cut only by size, Flush() or
  /// Stop(). A deterministic mode for tests (same record sequence, same
  /// drops, same batches).
  bool writer_thread = true;
};

struct RecorderStats {
  std::array<std::uint64_t, static_cast<std::size_t>(Table::kCount)> rows_written{};
  std::uint64_t rows_failed = 0;
  std::uint64_t batches = 0;
  std::uint64_t accepted = 0;
  std::uint64_t dropped = 0;
  std::uint64_t payloads_deduped = 0;  // not sent: already durable or in the batch
  std::uint64_t rows_replayed = 0;     // re-inserted from the WAL at startup
  bool crashed = false;                // the sink threw; the writer stopped
  std::uint64_t sink_ns = 0;          // writer wall time inside Sink::Write
  std::uint64_t sink_thread_cpu_ns = 0;  // writer thread CPU inside Sink::Write
  std::uint64_t producers = 0;           // live producers (each holds a ring of ring_bytes)
  std::uint64_t producers_freed = 0;     // retired producers drained and freed
  std::string last_error;
};

class Recorder {
 public:
  /// Replays any WAL frames into `sink` before returning; check error().
  Recorder(Sink& sink, RecorderOptions opt = {});
  /// Non-empty when the WAL could not be opened or replayed.
  const std::string& error() const { return error_; }
  ~Recorder();  // Stop()s
  Recorder(const Recorder&) = delete;
  Recorder& operator=(const Recorder&) = delete;

  /// A new producer with its own ring. Takes a lock; call once per thread.
  /// It lives as long as the Recorder (see Lease for one that is freed).
  Producer* NewProducer();
  /// A new producer owned by the returned lease: when the lease dies the
  /// producer is retired, drained and freed by the writer.
  ProducerLease Lease();
  /// Returns when every record accepted before the call (in happens-before
  /// order) has been handed to the sink, written or failed.
  void Flush();
  /// Flushes and joins the writer. Producers must be quiescent: records
  /// offered afterwards are dropped (counted).
  void Stop();
  /// writer_thread == false only: one writer iteration (drain the rings,
  /// send full batches).
  void Pump();
  RecorderStats stats() const;

 private:
  friend class Producer;
  friend class ProducerLease;
  void Wake();
  /// Frees the retired producers in `freed` (writer side, after their final drain).
  void Free(const std::vector<Producer*>& freed);
  void WriterLoop();
  bool Iterate(bool send_all);
  bool DrainAll();
  void Send(std::size_t t);
  void Append(std::uint8_t table, std::string_view body);

  Sink& sink_;
  RecorderOptions opt_;
  mutable std::mutex mu_;
  std::condition_variable writer_cv_;  // the writer waits here
  std::condition_variable done_cv_;    // Flush() callers wait here
  std::vector<std::unique_ptr<Producer>> producers_;  // guarded by mu_
  std::atomic<std::uint64_t> producers_gen_{0};
  std::atomic<bool> wake_{false};
  std::atomic<bool> stop_{false};
  std::atomic<std::uint64_t> flush_req_{0};
  std::uint64_t flush_done_ = 0;  // guarded by mu_
  bool writer_done_ = false;      // guarded by mu_
  std::thread writer_;

  // writer-thread state
  std::vector<Producer*> snapshot_;
  std::uint64_t snapshot_gen_ = ~0ull;
  struct Batch {
    std::string buf;
    std::size_t rows = 0;
    std::size_t last_row_off = 0;
  };
  std::array<Batch, static_cast<std::size_t>(Table::kCount)> batches_;
  std::uint64_t batch_seq_ = 0;
  std::string scratch_;
  std::uint64_t epoch_;
  std::string error_;
  std::unique_ptr<Wal> wal_;
  bool crashed_ = false;
  // payload dedup: keys durably written (FIFO-evicted), and keys in the
  // payload batch being built (they share its fate)
  std::unordered_set<std::string> durable_;
  std::deque<std::string> durable_order_;
  std::unordered_set<std::string> in_batch_;
  std::vector<std::string> in_batch_order_;

  // published stats (guarded by mu_); the counts of freed producers live on in stats_.accepted / dropped
  RecorderStats stats_;
  std::vector<Producer*> freed_;  // writer-thread scratch
  std::shared_ptr<detail::RecorderLife> life_ = std::make_shared<detail::RecorderLife>();
};

}  // namespace halo::scope

#endif  // HALO_SCOPE_RECORDER_HPP_
