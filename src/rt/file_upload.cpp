#include "rt/file_upload.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <thread>

namespace halo::rt {

namespace {

constexpr std::size_t kChunk = 16u << 20;  // bytes uploaded per staging fill
constexpr std::size_t kAlign = 4096;       // O_DIRECT offset/length/buffer alignment

bool ReadFully(int fd, void* dst, std::size_t bytes, std::uint64_t offset) {
  std::size_t got = 0;
  while (got < bytes) {
    const ssize_t n = ::pread(fd, static_cast<char*>(dst) + got, bytes - got, static_cast<off_t>(offset + got));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    got += static_cast<std::size_t>(n);
  }
  return true;
}

}  // namespace

FileUploader::FileUploader(Device& device, std::span<const FileSource> sources)
    : dev_(device), sources_(sources.begin(), sources.end()) {}

bool FileUploader::Add(const Buffer& dst, std::size_t dst_offset, const void* src, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(src);
  for (std::size_t i = 0; i < sources_.size(); ++i) {
    const auto* base = static_cast<const std::uint8_t*>(sources_[i].data);
    if (sources_[i].fd < 0 || p < base || p + bytes > base + sources_[i].size) continue;
    // Split into chunk-sized tasks so threads share large tensors.
    for (std::size_t done = 0; done < bytes; done += kChunk) {
      const std::size_t n = std::min(kChunk, bytes - done);
      tasks_.push_back({dst, dst_offset + done, i, static_cast<std::uint64_t>(p - base) + done, n});
    }
    queued_ += bytes;
    return true;
  }
  return false;
}

bool FileUploader::Run(unsigned threads, std::string* error) {
  if (tasks_.empty()) return true;
  threads = std::max(1u, std::min<unsigned>(threads, static_cast<unsigned>(tasks_.size())));
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::mutex upload_mu, error_mu;
  const auto fail = [&](std::string what) {
    std::lock_guard<std::mutex> lk(error_mu);
    if (!failed.exchange(true) && error) *error = std::move(what);
  };
  const auto worker = [&] {
    // Independent O_DIRECT descriptors per thread (offsets are explicit, but
    // the flag is per open file description); buffered reads if refused.
    std::vector<int> direct(sources_.size(), -1);
    for (std::size_t i = 0; i < sources_.size(); ++i) {
      const std::string path = "/proc/self/fd/" + std::to_string(sources_[i].fd);
      direct[i] = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    }
    void* staging = dev_.AllocHost(kChunk + 2 * kAlign, nullptr);
    if (!staging) {
      fail("staging allocation failed");
    }
    auto* buf = static_cast<std::uint8_t*>(staging);
    for (std::size_t t = next++; staging && !failed && t < tasks_.size(); t = next++) {
      const Task& k = tasks_[t];
      const std::uint64_t begin = k.file_offset / kAlign * kAlign;
      const std::uint64_t end = (k.file_offset + k.bytes + kAlign - 1) / kAlign * kAlign;
      const std::size_t delta = k.file_offset - begin;
      bool ok = direct[k.source] >= 0 && ReadFully(direct[k.source], buf, end - begin, begin);
      if (!ok) {  // unsupported O_DIRECT or a short tail at EOF: buffered read of the exact range
        ok = ReadFully(sources_[k.source].fd, buf + delta, k.bytes, k.file_offset);
      }
      if (!ok) {
        fail("reading model file failed: " + std::string(std::strerror(errno)));
        break;
      }
      std::string e;
      std::lock_guard<std::mutex> lk(upload_mu);
      if (!dev_.Upload(k.dst, k.dst_offset, buf + delta, k.bytes, &e)) {
        fail("upload failed: " + e);
        break;
      }
    }
    if (staging) dev_.FreeHost(staging);
    for (int fd : direct)
      if (fd >= 0) ::close(fd);
  };
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < threads; ++i) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  tasks_.clear();
  return !failed;
}

}  // namespace halo::rt
