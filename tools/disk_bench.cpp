// Random-read benchmark for the storage the weights and n-gram table live on.
//
//   halo_disk_bench FILE [seconds per point]
//
// Uniformly random, block-aligned O_DIRECT preads over the whole of FILE, at
// 4 KiB, 64 KiB and 1 MiB, with 1, 16 and 64 reads in flight (one thread per
// read: pread blocks, so threads are the queue depth). Prints IOPS, MB/s and
// the per-read latency median and p99 per point. With O_DIRECT the page
// cache is bypassed (on ZFS 2.3+ too, when zfs_dio_enabled=1), so the file
// should not need to be larger than RAM; read-only, FILE is never written.
// The n-gram path reads one 320-byte row per lookup, which costs a 4 KiB
// aligned read: the 4 KiB rows are the ones that predict it.

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: halo_disk_bench FILE [seconds per point]\n");
    return 2;
  }
  const double seconds = argc > 2 ? std::atof(argv[2]) : 3.0;
  const int fd = ::open(argv[1], O_RDONLY | O_CLOEXEC | O_DIRECT);
  if (fd < 0) {
    std::perror("open (O_DIRECT)");
    return 1;
  }
  struct stat st{};
  if (::fstat(fd, &st) != 0 || st.st_size < (64 << 20)) {
    std::fprintf(stderr, "need a file of at least 64 MiB\n");
    return 1;
  }
  const std::uint64_t size = static_cast<std::uint64_t>(st.st_size);
  std::printf("%s: %.1f GiB, %.1f s per point\n", argv[1], static_cast<double>(size) / (1ull << 30), seconds);
  std::printf("%8s %6s %10s %10s %12s %12s\n", "block", "depth", "IOPS", "MB/s", "median us", "p99 us");
  for (const std::size_t block : {std::size_t{4096}, std::size_t{65536}, std::size_t{1} << 20}) {
    for (const int depth : {1, 16, 64}) {
      std::atomic<bool> stop{false};
      std::vector<std::vector<float>> lat(depth);
      std::atomic<int> errors{0};
      std::vector<std::thread> threads;
      for (int t = 0; t < depth; ++t)
        threads.emplace_back([&, t] {
          void* buf = nullptr;
          if (posix_memalign(&buf, 4096, block) != 0) return;
          std::mt19937_64 g(0x9E3779B97F4A7C15ull * (t + 1) + block);
          std::uniform_int_distribution<std::uint64_t> pick(0, size / block - 1);
          while (!stop.load(std::memory_order_relaxed)) {
            const auto off = static_cast<off_t>(pick(g) * block);
            const auto t0 = std::chrono::steady_clock::now();
            if (::pread(fd, buf, block, off) != static_cast<ssize_t>(block)) errors.fetch_add(1);
            lat[t].push_back(
                std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - t0).count());
          }
          std::free(buf);
        });
      const auto start = std::chrono::steady_clock::now();
      std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
      stop = true;
      for (auto& th : threads) th.join();
      const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      std::vector<float> all;
      for (auto& l : lat) all.insert(all.end(), l.begin(), l.end());
      std::sort(all.begin(), all.end());
      const double n = static_cast<double>(all.size());
      std::printf("%7zuK %6d %10.0f %10.0f %12.0f %12.0f%s\n", block / 1024, depth, n / el,
                  n * static_cast<double>(block) / el / 1e6, all.empty() ? 0.0 : all[all.size() / 2],
                  all.empty() ? 0.0 : all[std::min(all.size() - 1, static_cast<std::size_t>(n * 0.99))],
                  errors ? "  (short reads)" : "");
    }
  }
  ::close(fd);
  return 0;
}
