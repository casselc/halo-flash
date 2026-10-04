#ifndef HALO_SCOPE_RING_HPP_
#define HALO_SCOPE_RING_HPP_

// Single-producer, single-consumer byte ring, after oscope-core's
// src/ring.rs (oscope; see NOTICE). The producer is
// an engine thread and never blocks, allocates or takes a lock: a record
// that does not fit is dropped and counted. The consumer is the writer
// thread. Records are [u32 length][u8 table][length-1 bytes] and may wrap.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>

namespace halo::scope {

class Ring {
 public:
  /// `capacity` is rounded up to a power of two. Pages are pre-faulted so a
  /// producer never takes a first-touch fault.
  explicit Ring(std::size_t capacity) {
    std::size_t cap = 64;
    while (cap < capacity) cap <<= 1;
    mask_ = cap - 1;
    buf_ = std::make_unique<std::uint8_t[]>(cap);
    for (std::size_t i = 0; i < cap; i += 4096) reinterpret_cast<volatile std::uint8_t*>(buf_.get())[i] = 0;
  }

  std::size_t capacity() const { return mask_ + 1; }

  /// Producer: copies one record in. Returns false and counts a drop when
  /// it does not fit. `half_full` is set when the ring passed half capacity.
  bool Push(std::uint8_t table, std::string_view body, bool* half_full) {
    const std::size_t total = 5 + body.size();
    const std::size_t head = head_.load(std::memory_order_relaxed);
    const std::size_t tail = tail_.load(std::memory_order_acquire);
    if (total > capacity() - (head - tail)) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    std::uint8_t hdr[5];
    const auto len = static_cast<std::uint32_t>(body.size() + 1);
    std::memcpy(hdr, &len, 4);
    hdr[4] = table;
    CopyIn(head, hdr, 5);
    CopyIn(head + 5, body.data(), body.size());
#if defined(HALO_SCOPE_MUTANT) && HALO_SCOPE_MUTANT == 5
    head_.store(head + total, std::memory_order_relaxed);  // positive control for TSan: unpublished bytes
#else
    head_.store(head + total, std::memory_order_release);
#endif
    accepted_.fetch_add(1, std::memory_order_relaxed);
    *half_full = (head + total - tail) > capacity() / 2;
    return true;
  }

  /// Consumer: pops one record. `scratch` holds the body when it wraps.
  /// Returns false when empty.
  bool Pop(std::uint8_t* table, std::string_view* body, std::string& scratch) {
    const std::size_t tail = tail_.load(std::memory_order_relaxed);
    const std::size_t head = head_.load(std::memory_order_acquire);
    if (tail == head) return false;
    std::uint8_t hdr[5];
    CopyOut(tail, hdr, 5);
    std::uint32_t len;
    std::memcpy(&len, hdr, 4);
    *table = hdr[4];
    const std::size_t n = len - 1;
    const std::size_t off = (tail + 5) & mask_;
    if (off + n <= capacity()) {
      *body = std::string_view(reinterpret_cast<const char*>(buf_.get() + off), n);
    } else {
      scratch.resize(n);
      CopyOut(tail + 5, scratch.data(), n);
      *body = scratch;
    }
    pending_tail_ = tail + 4 + len;
    return true;
  }
  /// Consumer: releases the record returned by the last Pop (its body view
  /// is invalid afterwards).
  void Release() { tail_.store(pending_tail_, std::memory_order_release); }

  /// Producer: counts a record refused before reaching the ring.
  void CountDrop() { dropped_.fetch_add(1, std::memory_order_relaxed); }

  bool Empty() const {
    return tail_.load(std::memory_order_relaxed) == head_.load(std::memory_order_acquire);
  }
  std::uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }
  std::uint64_t accepted() const { return accepted_.load(std::memory_order_relaxed); }

  // Set by the producer when it woke the writer; cleared by the writer.
  std::atomic<bool> wake_sent{false};

 private:
  void CopyIn(std::size_t pos, const void* src, std::size_t n) {
    const std::size_t off = pos & mask_;
    const std::size_t first = n < capacity() - off ? n : capacity() - off;
    std::memcpy(buf_.get() + off, src, first);
    if (first < n) std::memcpy(buf_.get(), static_cast<const std::uint8_t*>(src) + first, n - first);
  }
  void CopyOut(std::size_t pos, void* dst, std::size_t n) const {
    const std::size_t off = pos & mask_;
    const std::size_t first = n < capacity() - off ? n : capacity() - off;
    std::memcpy(dst, buf_.get() + off, first);
    if (first < n) std::memcpy(static_cast<std::uint8_t*>(dst) + first, buf_.get(), n - first);
  }

  std::unique_ptr<std::uint8_t[]> buf_;
  std::size_t mask_ = 0;
  alignas(64) std::atomic<std::size_t> head_{0};  // producer
  alignas(64) std::atomic<std::size_t> tail_{0};  // consumer
  std::size_t pending_tail_ = 0;                  // consumer only
  alignas(64) std::atomic<std::uint64_t> dropped_{0};
  std::atomic<std::uint64_t> accepted_{0};
};

}  // namespace halo::scope

#endif  // HALO_SCOPE_RING_HPP_
