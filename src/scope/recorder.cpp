// Adapted from oscope (github.com/chucklehead-dev/oscope, 3a14d2c,
// spike/oscope-core/src/pipeline.rs and src/recorder.rs), originally
// EPL-2.0; relicensed under MIT for halo-flash by its author, Chuck Cassel
// (2026-10-03). See NOTICE.
//
// Write-path structure: rings -> one writer -> batched RowBinary stream
// inserts. The WAL and the payload cache are halo-flash's own.

#include "scope/recorder.hpp"

#include <time.h>

#include <random>

// HALO_SCOPE_MUTANT selects a deliberately broken writer for the positive
// controls in tests/scope (never set in a normal build):
//   3: a record whose body wraps the ring end is silently lost
//   4: after a size-triggered batch, its last row is sent again
//   5: the ring publishes its head without release ordering (ring.hpp),
//      a data race only ThreadSanitizer sees
//   6: the insert token is per process, not derived from the WAL position,
//      so a replayed batch is written twice
//   7: a payload is remembered as durable when it is batched, before its
//      insert succeeds, so a failed payload is never sent again
#ifndef HALO_SCOPE_MUTANT
#define HALO_SCOPE_MUTANT 0
#endif

namespace halo::scope {

namespace {

std::uint64_t ThreadCpuNs() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<std::uint64_t>(ts.tv_nsec);
}

const char* StagingInsert(Table t) {
  switch (t) {
    case Table::kObservations: return "INSERT INTO halo_scope.llm_observations_in";
    case Table::kPayloads: return "INSERT INTO halo_scope.llm_payloads_in";
    default: return nullptr;
  }
}

const char* Columns(Table t) {
  return t == Table::kObservations ? kObservationColumns : kPayloadColumns;
}

}  // namespace

bool ChdbSink::Write(Table table, std::string_view rows, std::size_t, std::string_view token, std::string* err) {
  // deduplicate_blocks_in_dependent_materialized_views: the Null staging
  // table keeps nothing, so deduplication has to happen at the MV target.
  std::string ins = std::string(StagingInsert(table)) + " (" + Columns(table) + ")";
  if (dedup_token_)
    ins += " SETTINGS deduplicate_blocks_in_dependent_materialized_views = 1, insert_deduplication_token = " +
           SqlQuote(token);
  auto r = conn_->StreamInsert(ins, "RowBinary", rows);
  if (!r.ok) *err = r.error;
  return r.ok;
}

bool Producer::RecordRaw(Table t, std::string_view row) {
  if (rec_->stop_.load(std::memory_order_relaxed)) {
    ring_.CountDrop();
    return false;
  }
  bool half = false;
  if (!ring_.Push(static_cast<std::uint8_t>(t), row, &half)) return false;
  if (half && !ring_.wake_sent.load(std::memory_order_relaxed)) {
    ring_.wake_sent.store(true, std::memory_order_relaxed);
    rec_->Wake();
  }
  return true;
}

namespace {

// (project, hash) of a staging payload row: its first two columns.
std::string PayloadKey(std::string_view row) {
  rb::Reader r(row);
  std::string k = r.Str();
  const auto h = r.FixedStr16();
  k.append(reinterpret_cast<const char*>(h.data()), h.size());
  return k;
}

}  // namespace

Recorder::Recorder(Sink& sink, RecorderOptions opt) : sink_(sink), opt_(opt) {
  epoch_ = std::random_device{}() ^ (static_cast<std::uint64_t>(std::random_device{}()) << 32);
  for (auto& b : batches_) b.buf.reserve(1 << 20);
  if (!opt_.wal_path.empty()) {
    wal_ = std::make_unique<Wal>();
    if (!wal_->Open(opt_.wal_path, opt_.wal_sync, &error_)) {
      wal_.reset();
    } else {
      // Replay what a crash left: same token as the first attempt, so a
      // batch that did reach chDB is deduplicated there.
      for (const WalFrame& f : wal_->Pending()) {
        if (f.table >= batches_.size()) continue;
        std::string err;
        if (sink_.Write(static_cast<Table>(f.table), f.body, f.rows, wal_->Token(f.seq), &err)) {
          stats_.rows_replayed += f.rows;
        } else {
          stats_.rows_failed += f.rows;
          stats_.last_error = error_ = "wal replay: " + err;
        }
      }
      std::string err;
      if (!wal_->Reset(&err)) error_ = err;
    }
  }
  if (opt_.writer_thread) writer_ = std::thread([this] { WriterLoop(); });
}

Recorder::~Recorder() { Stop(); }

Producer* Recorder::NewProducer() {
  std::lock_guard<std::mutex> lk(mu_);
  producers_.emplace_back(new Producer(this, opt_.ring_bytes));
  producers_gen_.fetch_add(1, std::memory_order_release);
  return producers_.back().get();
}

void Recorder::Wake() {
  // Lock-free and non-blocking for the producer: a lost wakeup only delays
  // the writer to its next interval tick.
  wake_.store(true, std::memory_order_relaxed);
  writer_cv_.notify_one();
}

void Recorder::Flush() {
  if (!opt_.writer_thread) {
    if (!stop_.load(std::memory_order_relaxed) && !crashed_) Iterate(/*send_all=*/true);
    return;
  }
  const std::uint64_t my = flush_req_.fetch_add(1, std::memory_order_acq_rel) + 1;
  std::unique_lock<std::mutex> lk(mu_);
  writer_cv_.notify_one();
  done_cv_.wait(lk, [&] { return flush_done_ >= my || writer_done_; });
}

void Recorder::Pump() {
  if (!opt_.writer_thread && !stop_.load(std::memory_order_relaxed) && !crashed_) Iterate(/*send_all=*/false);
}

void Recorder::Stop() {
  if (!opt_.writer_thread) {
    if (stop_.exchange(true) || crashed_) return;
    Iterate(/*send_all=*/true);
    return;
  }
  if (!writer_.joinable()) return;
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_.store(true, std::memory_order_release);
  }
  writer_cv_.notify_one();
  writer_.join();
}

RecorderStats Recorder::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  RecorderStats s = stats_;
  for (const auto& p : producers_) {
    s.accepted += p->accepted();
    s.dropped += p->dropped();
  }
  return s;
}

void Recorder::Send(std::size_t t) {
  Batch& b = batches_[t];
  if (b.rows == 0) return;
  std::string token = std::to_string(epoch_) + "-" + std::to_string(++batch_seq_);
  std::string err;
  std::string wal_err;
  if (wal_) {
    // Logged before the insert; the token is the frame's position.
    if (const auto seq = wal_->Append(static_cast<std::uint8_t>(t), static_cast<std::uint32_t>(b.rows), b.buf, &wal_err))
      token = wal_->Token(seq);
#if HALO_SCOPE_MUTANT == 6
    token = std::to_string(epoch_) + "-" + std::to_string(batch_seq_);
#endif
  }
  const auto w0 = std::chrono::steady_clock::now();
  const auto c0 = ThreadCpuNs();
  bool ok;
  try {
    ok = sink_.Write(static_cast<Table>(t), b.buf, b.rows, token, &err);
  } catch (...) {
    // A sink that throws models the process dying mid-insert: stop here,
    // leaving the WAL as it is for the next start to replay.
    crashed_ = true;
    std::lock_guard<std::mutex> lk(mu_);
    stats_.crashed = true;
    throw;
  }
  const auto c1 = ThreadCpuNs();
  const auto w1 = std::chrono::steady_clock::now();
  if (t == static_cast<std::size_t>(Table::kPayloads)) {
    // Only now are this batch's payloads durable (or known lost).
    if (ok)
      for (auto& k : in_batch_order_) {
        if (opt_.payload_cache_entries == 0 || durable_.count(k)) continue;
        durable_.insert(k);
        durable_order_.push_back(std::move(k));
        if (durable_order_.size() > opt_.payload_cache_entries) {
          durable_.erase(durable_order_.front());
          durable_order_.pop_front();
        }
      }
    in_batch_.clear();
    in_batch_order_.clear();
  }
  if (wal_ && !wal_->Reset(&wal_err) && wal_err.empty()) wal_err = "wal reset failed";
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (ok) {
      stats_.rows_written[t] += b.rows;
    } else {
      stats_.rows_failed += b.rows;
      stats_.last_error = err;
    }
    if (!wal_err.empty()) stats_.last_error = wal_err;
    ++stats_.batches;
    stats_.sink_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count());
    stats_.sink_thread_cpu_ns += c1 - c0;
  }
#if HALO_SCOPE_MUTANT == 4
  if (b.rows >= opt_.batch_rows && b.rows > 1) {
    b.buf.erase(0, b.last_row_off);
    b.rows = 1;
    b.last_row_off = 0;
    return;
  }
#endif
  b.buf.clear();
  b.rows = 0;
  b.last_row_off = 0;
}

bool Recorder::DrainAll() {
  if (const auto g = producers_gen_.load(std::memory_order_acquire); g != snapshot_gen_) {
    std::lock_guard<std::mutex> lk(mu_);
    snapshot_.clear();
    for (const auto& p : producers_) snapshot_.push_back(p.get());
    snapshot_gen_ = producers_gen_.load(std::memory_order_relaxed);
  }
  bool any = false;
  for (Producer* p : snapshot_) {
    Ring& ring = p->ring_;
    std::uint8_t table;
    std::string_view body;
    while (ring.Pop(&table, &body, scratch_)) {
      any = true;
      if (table < batches_.size()) {
#if HALO_SCOPE_MUTANT == 3
        const bool lost = body.data() == scratch_.data();
#else
        const bool lost = false;
#endif
        Batch& b = batches_[table];
        bool dup = false;
        if (!lost && table == static_cast<std::uint8_t>(Table::kPayloads) && opt_.payload_cache_entries) {
          std::string key = PayloadKey(body);
          dup = durable_.count(key) || in_batch_.count(key);
          if (!dup) {
#if HALO_SCOPE_MUTANT == 7
            durable_.insert(key);
            durable_order_.push_back(key);
#endif
            in_batch_.insert(key);
            in_batch_order_.push_back(std::move(key));
          } else {
            std::lock_guard<std::mutex> lk(mu_);
            ++stats_.payloads_deduped;
          }
        }
        if (!lost && !dup) {
          b.last_row_off = b.buf.size();
          b.buf.append(body);
          ++b.rows;
        }
        ring.Release();
        if (b.rows >= opt_.batch_rows) Send(table);
      } else {
        ring.Release();  // unknown table: cannot happen through Producer
      }
    }
    ring.wake_sent.store(false, std::memory_order_relaxed);
  }
  return any;
}

bool Recorder::Iterate(bool send_all) {
  const bool any = DrainAll();
  if (send_all)
    for (std::size_t t = 0; t < batches_.size(); ++t) Send(t);
  return any;
}

void Recorder::WriterLoop() {
  std::uint64_t served = 0;  // highest flush request completed
  auto last_send = std::chrono::steady_clock::now();
  while (true) {
    const bool stopping = stop_.load(std::memory_order_acquire);
    const std::uint64_t req = flush_req_.load(std::memory_order_acquire);
    wake_.store(false, std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    const bool send_all = stopping || req > served || now - last_send >= opt_.flush_interval;
    bool any = false;
    try {
      any = Iterate(send_all);
    } catch (...) {
      break;  // crashed_: see Send()
    }
    if (send_all) last_send = now;
    if (req > served) {
      served = req;
      std::lock_guard<std::mutex> lk(mu_);
      flush_done_ = served;
      done_cv_.notify_all();
    }
    if (stopping) break;
    if (!any) {
      std::unique_lock<std::mutex> lk(mu_);
      writer_cv_.wait_for(lk, opt_.flush_interval, [&] {
        return wake_.load(std::memory_order_relaxed) || stop_.load(std::memory_order_relaxed) ||
               flush_req_.load(std::memory_order_relaxed) > served;
      });
    }
  }
  std::lock_guard<std::mutex> lk(mu_);
  writer_done_ = true;
  done_cv_.notify_all();
}

}  // namespace halo::scope
