#ifndef HALO_SERVER_STATE_FILE_HPP_
#define HALO_SERVER_STATE_FILE_HPP_

// Small durable state files for the spill key providers (the master's epoch, yard's per-session epochs;
// docs/SESSIONS.md section 10, "State files"): one pinned directory, exclusive ownership, typed publication outcomes
// and checked loading.
//
//   - StateDir: the directory is opened once (O_DIRECTORY | O_NOFOLLOW; ours; not group/other-writable) and every
//     later operation (lock, read, publish, marker, the directory fsync) goes through that descriptor (openat,
//     renameat, fsync of the fd). A rename or replacement of the directory's path afterwards cannot split a lock, a
//     write and its fsync across two directories. Its ancestors are trusted (same-uid / root tampering is out of
//     scope).
//   - Lock: an flock on "<name>.lock" in the pinned directory (the lock file itself checked: regular, ours, 0600),
//     taken before the state is read and held for the owner's lifetime.
//   - Write: temp (0600, O_EXCL) + fsync + renameat + fsync of the directory fd. The outcome is typed: kUnpublished
//     (the old file is still in place), kDurable, or kUncertain (the rename happened but the directory fsync failed:
//     the new content is visible and may or may not survive a crash). Callers never treat kUncertain as "not done".
//   - Read: openat O_NOFOLLOW; a regular file owned by us with no group/other bits; a bounded read to EOF that must
//     match the size. "<name>.init" marks a state that was initialized: a missing state with its marker refuses
//     (recovery required). Init writes the state first and the marker second (a crash between them leaves a state
//     without a marker, which the next load completes: never a spurious "recovery required"); EnsureMarker gives a
//     state from before the markers existed its marker on the first load.
//     Restoring an older copy of both from a backup is out of scope (no external monotonic anchor).

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace halo::server {

enum class Persist : std::uint8_t { kUnpublished, kDurable, kUncertain };
[[nodiscard]] const char* PersistName(Persist p);
enum class StateRead : std::uint8_t { kOk, kFresh, kError };

/// Test hook: 1 = the next write fails before its rename (unpublished); 2 = the next write fails at the directory
/// fsync after its rename (uncertain). Taken by exactly one write (atomic exchange).
inline std::atomic<int> g_state_write_fault{0};

class StateLock {
 public:
  ~StateLock();
  StateLock(const StateLock&) = delete;
  StateLock& operator=(const StateLock&) = delete;

 private:
  friend class StateDir;
  explicit StateLock(int fd) : fd_(fd) {}
  int fd_ = -1;
};

class StateDir {
 public:
  /// Opens and pins `path` (must exist: a directory, ours, not writable by group/others, not a symlink).
  static std::unique_ptr<StateDir> Open(const std::string& path, std::string* error);
  ~StateDir();
  StateDir(const StateDir&) = delete;
  StateDir& operator=(const StateDir&) = delete;
  [[nodiscard]] const std::string& path() const { return path_; }

  /// An exclusive, non-blocking flock on "<name>.lock". Null with `error` if another process (or another StateLock in
  /// this one) holds it, or the lock file is not ours / not 0600 / not regular.
  [[nodiscard]] std::unique_ptr<StateLock> Lock(const std::string& name, std::string* error) const;
  [[nodiscard]] Persist Write(const std::string& name, std::string_view bytes, std::string* error) const;
  /// kFresh: neither the file nor its init marker exists. kError: anything else that is not a complete, safe read.
  [[nodiscard]] StateRead Read(const std::string& name, std::size_t max_bytes, std::string* out, std::string* error) const;
  /// First initialization: the state, then the marker, both durably.
  [[nodiscard]] bool Init(const std::string& name, std::string_view bytes, std::string* error) const;
  /// A loaded state without its marker (made before markers existed, or a first init interrupted after the state):
  /// writes the marker durably. False if that is not possible (the caller refuses to start).
  [[nodiscard]] bool EnsureMarker(const std::string& name, std::string* error) const;

 private:
  StateDir(int fd, std::string path) : fd_(fd), path_(std::move(path)) {}
  [[nodiscard]] bool Exists(const std::string& name) const;
  int fd_ = -1;
  std::string path_;
};

/// Splits "dir/name" (no slash: ".").
[[nodiscard]] std::pair<std::string, std::string> SplitPath(const std::string& path);

/// Parses a decimal unsigned integer exactly (no sign, no junk, no overflow, <= max).
[[nodiscard]] bool ParseU64(std::string_view s, std::uint64_t max, std::uint64_t* out);

}  // namespace halo::server

#endif  // HALO_SERVER_STATE_FILE_HPP_
