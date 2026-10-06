#ifndef HALO_TRAIN_PARALLEL_HPP_
#define HALO_TRAIN_PARALLEL_HPP_

// A small fixed-size thread pool for the stage-1 trainer's CPU math.
// Determinism contract: ParallelFor hands out task indices dynamically, so
// callers must make every task's result independent of which thread runs it
// and of the order tasks run in (each output element written by exactly one
// task, reductions done in a fixed order). Every kernel in src/train follows
// it, which makes training bit-for-bit reproducible at any thread count.

#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace halo::train {

class ThreadPool {
 public:
  /// `threads` <= 1 runs everything on the caller.
  explicit ThreadPool(unsigned threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  /// Runs fn(i) for i in [0, n) and returns when all are done.
  void ParallelFor(std::size_t n, const std::function<void(std::size_t)>& fn);
  [[nodiscard]] unsigned size() const noexcept { return static_cast<unsigned>(workers_.size()) + 1; }

 private:
  void Worker();
  void Drain();

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable start_, done_;
  const std::function<void(std::size_t)>* fn_ = nullptr;
  std::size_t n_ = 0, next_ = 0, finished_ = 0;
  std::uint64_t generation_ = 0;
  unsigned busy_ = 0;
  bool stop_ = false;
};

}  // namespace halo::train

#endif  // HALO_TRAIN_PARALLEL_HPP_
