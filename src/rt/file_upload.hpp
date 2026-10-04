#ifndef HALO_RT_FILE_UPLOAD_HPP_
#define HALO_RT_FILE_UPLOAD_HPP_

// Parallel weight loading straight from the model files. Tensor bytes that
// live in a mapped file region are read with O_DIRECT (no page cache, no
// page faults through the mapping) by several threads into pinned staging
// buffers and uploaded from there; anything else falls back to a plain
// Upload. Loading 79 GiB through per-tensor copies of mapped (pageable)
// memory ran at ~170 MB/s; this path is bounded by the disk.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "rt/device.hpp"

namespace halo::rt {

/// A file mapped at `data` from offset 0 for `size` bytes, readable at `fd`.
struct FileSource {
  const void* data = nullptr;
  std::size_t size = 0;
  int fd = -1;
};

class FileUploader {
 public:
  FileUploader(Device& device, std::span<const FileSource> sources);

  /// Queues `bytes` at `src` for `dst` at `dst_offset`. Returns false when
  /// `src` is not inside a source file; the caller then uploads directly.
  bool Add(const Buffer& dst, std::size_t dst_offset, const void* src, std::size_t bytes);

  /// Uploads everything queued with `threads` readers; blocks until done.
  [[nodiscard]] bool Run(unsigned threads, std::string* error);

  [[nodiscard]] std::size_t QueuedBytes() const noexcept { return queued_; }

 private:
  struct Task {
    Buffer dst;
    std::size_t dst_offset;
    std::size_t source;
    std::uint64_t file_offset;
    std::size_t bytes;
  };
  Device& dev_;
  std::vector<FileSource> sources_;
  std::vector<Task> tasks_;
  std::size_t queued_ = 0;
};

}  // namespace halo::rt

#endif  // HALO_RT_FILE_UPLOAD_HPP_
