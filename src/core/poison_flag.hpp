#ifndef HALO_CORE_POISON_FLAG_HPP_
#define HALO_CORE_POISON_FLAG_HPP_

// A one-way "this engine is poisoned" flag (docs/SESSIONS.md section 10, "Partial restores"): set by the thread whose
// failed restore could not synchronize the device, read by every engine call and polled by halo_serve's main thread
// without the generation lock. Atomic, release / acquire: a reader that sees it set also sees everything the setter
// did before (the quarantine state).

#include <atomic>

namespace halo::core {

class PoisonFlag {
 public:
  void Set() noexcept { v_.store(true, std::memory_order_release); }
  [[nodiscard]] bool Get() const noexcept { return v_.load(std::memory_order_acquire); }
  explicit operator bool() const noexcept { return Get(); }

 private:
  std::atomic<bool> v_{false};
};

}  // namespace halo::core

#endif  // HALO_CORE_POISON_FLAG_HPP_
