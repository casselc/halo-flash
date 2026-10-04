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
//   rows_written + rows_failed == sum(accepted).

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
#include <vector>

#include "scope/chdb.hpp"
#include "scope/records.hpp"
#include "scope/ring.hpp"

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
/// batch is exactly-once; measured on libchdb 26.7.0 this about doubles the
/// process CPU of a 10k-row insert through the staging MV.
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
  /// One already-encoded staging row.
  bool RecordRaw(Table t, std::string_view row);

  std::uint64_t dropped() const { return ring_.dropped(); }
  std::uint64_t accepted() const { return ring_.accepted(); }

 private:
  friend class Recorder;
  Producer(Recorder* rec, std::size_t ring_bytes) : rec_(rec), ring_(ring_bytes) { scratch_.reserve(4096); }
  Recorder* rec_;
  Ring ring_;
  std::string scratch_;
};

struct RecorderOptions {
  std::size_t ring_bytes = 1 << 20;  // per producer
  std::size_t batch_rows = 10000;    // per table
  std::chrono::milliseconds flush_interval{250};
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
  std::uint64_t sink_ns = 0;          // writer wall time inside Sink::Write
  std::uint64_t sink_thread_cpu_ns = 0;  // writer thread CPU inside Sink::Write
  std::string last_error;
};

class Recorder {
 public:
  Recorder(Sink& sink, RecorderOptions opt = {});
  ~Recorder();  // Stop()s
  Recorder(const Recorder&) = delete;
  Recorder& operator=(const Recorder&) = delete;

  /// A new producer with its own ring. Takes a lock; call once per thread.
  Producer* NewProducer();
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
  void Wake();
  void WriterLoop();
  bool Iterate(bool send_all);
  bool DrainAll();
  void Send(std::size_t t);

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

  // published stats (guarded by mu_)
  RecorderStats stats_;
};

}  // namespace halo::scope

#endif  // HALO_SCOPE_RECORDER_HPP_
