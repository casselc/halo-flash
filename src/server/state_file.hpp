#ifndef HALO_SERVER_STATE_FILE_HPP_
#define HALO_SERVER_STATE_FILE_HPP_

// Small durable state files for the spill key providers (the master's epoch, yard's per-session epochs;
// docs/SESSIONS.md section 10, "Restart"): exclusive ownership, typed publication outcomes, and checked loading.
//
//   - StateLock: an flock on "<path>.lock", taken before the state is read and held for the owner's lifetime; a
//     second owner (another halo_serve on the same spill directory) is refused instead of overwriting the first's
//     fences with a stale snapshot.
//   - WriteStateFile: temp (0600, O_EXCL) + fsync + rename + fsync of the directory. The outcome is typed: kUnpublished
//     (the old file is still in place: nothing happened), kDurable, or kUncertain (the rename happened but the
//     directory fsync failed: the new content is visible and may or may not survive a crash). Callers never treat
//     kUncertain as "not done".
//   - ReadStateFile: the directory opened O_NOFOLLOW, the file opened relative to it O_NOFOLLOW; the opened descriptor
//     must be a regular file owned by us with no group/other bits; a bounded read to EOF that must match the size.
//     "<path>.init" marks a state that was initialized once: if the state is missing but the marker is there, loading
//     refuses (recovery required: delete the marker to reset deliberately) instead of silently starting from zero.
//     Restoring an older copy of both from a backup is out of scope (no external monotonic anchor).

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace halo::server {

enum class Persist : std::uint8_t { kUnpublished, kDurable, kUncertain };
[[nodiscard]] const char* PersistName(Persist p);

[[nodiscard]] Persist WriteStateFile(const std::string& path, std::string_view bytes, std::string* error);
/// Test hook: 1 = the next write fails before its rename (unpublished); 2 = the next write fails at the directory
/// fsync after its rename (uncertain). Reset after one use.
inline int g_state_write_fault = 0;

enum class StateRead : std::uint8_t { kOk, kFresh, kError };
/// Reads `path` (at most `max_bytes`). kFresh: neither the file nor its init marker exists (first start; the caller
/// writes the initial state with InitStateFile). kError: anything else that is not a complete, safe read.
[[nodiscard]] StateRead ReadStateFile(const std::string& path, std::size_t max_bytes, std::string* out,
                                      std::string* error);
/// First initialization: writes the init marker, then the state, both durably.
[[nodiscard]] bool InitStateFile(const std::string& path, std::string_view bytes, std::string* error);

class StateLock {
 public:
  /// Takes an exclusive, non-blocking flock on "<path>.lock" (created 0600). Null with `error` if another process
  /// (or another StateLock in this one) holds it.
  static std::unique_ptr<StateLock> Acquire(const std::string& path, std::string* error);
  ~StateLock();
  StateLock(const StateLock&) = delete;
  StateLock& operator=(const StateLock&) = delete;

 private:
  explicit StateLock(int fd) : fd_(fd) {}
  int fd_ = -1;
};

/// Parses a decimal unsigned integer exactly (no sign, no junk, no overflow, <= max).
[[nodiscard]] bool ParseU64(std::string_view s, std::uint64_t max, std::uint64_t* out);

}  // namespace halo::server

#endif  // HALO_SERVER_STATE_FILE_HPP_
