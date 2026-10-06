#include "train/parallel.hpp"

namespace halo::train {

ThreadPool::ThreadPool(unsigned threads) {
  for (unsigned i = 1; i < threads; ++i) workers_.emplace_back([this] { Worker(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  start_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::Drain() {
  // Called with mu_ held; claims and runs tasks until none are left.
  std::unique_lock<std::mutex> lk(mu_, std::adopt_lock);
  while (next_ < n_) {
    const std::size_t i = next_++;
    const auto* fn = fn_;
    lk.unlock();
    (*fn)(i);
    lk.lock();
    if (++finished_ == n_) done_.notify_all();
  }
  lk.release();
}

void ThreadPool::Worker() {
  std::uint64_t seen = 0;
  std::unique_lock<std::mutex> lk(mu_);
  for (;;) {
    start_.wait(lk, [&] { return stop_ || generation_ != seen; });
    if (stop_) return;
    seen = generation_;
    ++busy_;
    lk.release();
    Drain();
    lk = std::unique_lock<std::mutex>(mu_, std::adopt_lock);
    if (--busy_ == 0) done_.notify_all();
  }
}

void ThreadPool::ParallelFor(std::size_t n, const std::function<void(std::size_t)>& fn) {
  if (n == 0) return;
  if (workers_.empty() || n == 1) {
    for (std::size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  std::unique_lock<std::mutex> lk(mu_);
  // A previous call's workers may still be leaving Drain(); wait for them so
  // they never see this call's counters half set.
  done_.wait(lk, [&] { return busy_ == 0; });
  fn_ = &fn;
  n_ = n;
  next_ = 0;
  finished_ = 0;
  ++generation_;
  start_.notify_all();
  lk.release();
  Drain();
  lk = std::unique_lock<std::mutex>(mu_, std::adopt_lock);
  done_.wait(lk, [&] { return finished_ == n_ && busy_ == 0; });
  fn_ = nullptr;
  n_ = 0;
}

}  // namespace halo::train
