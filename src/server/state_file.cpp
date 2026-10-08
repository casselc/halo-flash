#include "server/state_file.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace halo::server {

namespace {

bool WriteAll(int fd, std::string_view bytes) {
  std::size_t done = 0;
  while (done < bytes.size()) {
    const ssize_t w = ::write(fd, bytes.data() + done, bytes.size() - done);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) return false;
    done += static_cast<std::size_t>(w);
  }
  return true;
}

}  // namespace

std::pair<std::string, std::string> SplitPath(const std::string& path) {
  const auto slash = path.rfind('/');
  if (slash == std::string::npos) return {".", path};
  return {slash == 0 ? "/" : path.substr(0, slash), path.substr(slash + 1)};
}

const char* PersistName(Persist p) {
  switch (p) {
    case Persist::kUnpublished: return "not published";
    case Persist::kDurable: return "durable";
    case Persist::kUncertain: return "published, durability uncertain";
  }
  return "?";
}

bool ParseU64(std::string_view s, std::uint64_t max, std::uint64_t* out) {
  if (s.empty() || s.size() > 20) return false;
  std::uint64_t v = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    const std::uint64_t d = static_cast<std::uint64_t>(c - '0');
    if (v > (max - d) / 10) return false;
    v = v * 10 + d;
  }
  *out = v;
  return true;
}

std::unique_ptr<StateDir> StateDir::Open(const std::string& path, std::string* error) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) return *error = "state directory " + path + ": " + std::strerror(errno), nullptr;
  struct stat st {};
  if (::fstat(fd, &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != ::geteuid() || (st.st_mode & 022) != 0) {
    ::close(fd);
    return *error = "state directory " + path + " must be a directory of ours, not writable by group or others",
           nullptr;
  }
  return std::unique_ptr<StateDir>(new StateDir(fd, path));
}

StateDir::~StateDir() {
  if (fd_ >= 0) ::close(fd_);
}

bool StateDir::Exists(const std::string& name) const {
  struct stat st {};
  return ::fstatat(fd_, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) == 0;
}

Persist StateDir::Write(const std::string& name, std::string_view bytes, std::string* error) const {
  const int fault = g_state_write_fault.exchange(0);
  const std::string tmp = name + ".tmp." + std::to_string(::getpid());
  ::unlinkat(fd_, tmp.c_str(), 0);
  const int fd = ::openat(fd_, tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) return *error = "cannot create " + path_ + "/" + tmp + ": " + std::strerror(errno), Persist::kUnpublished;
  const bool synced = WriteAll(fd, bytes) && ::fsync(fd) == 0;
  ::close(fd);
  if (!synced || fault == 1 || ::renameat(fd_, tmp.c_str(), fd_, name.c_str()) != 0) {
    ::unlinkat(fd_, tmp.c_str(), 0);
    return *error = "cannot publish " + path_ + "/" + name + (fault == 1 ? " (injected)" : ""), Persist::kUnpublished;
  }
  // Published: from here a failure is "uncertain", never "not done". The fsync is of the pinned directory itself.
  if (fault == 2 || ::fsync(fd_) != 0)
    return *error = "fsync of " + path_ + " failed after publishing " + name, Persist::kUncertain;
  return Persist::kDurable;
}

StateRead StateDir::Read(const std::string& name, std::size_t max_bytes, std::string* out, std::string* error) const {
  out->clear();
  const std::string full = path_ + "/" + name;
  const int fd = ::openat(fd_, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) {
    const int e = errno;
    if (e != ENOENT) return *error = full + ": " + std::strerror(e), StateRead::kError;
    if (Exists(name + ".init"))
      return *error = full + " is missing but was initialized (" + name +
                      ".init exists): recovery required (docs/SESSIONS.md section 10, \"Recovery\")",
             StateRead::kError;
    return StateRead::kFresh;
  }
  struct stat st {};
  const auto fail = [&](std::string why) {
    ::close(fd);
    *error = full + ": " + why;
    out->clear();
    return StateRead::kError;
  };
  if (::fstat(fd, &st) != 0) return fail("fstat failed");
  if (!S_ISREG(st.st_mode)) return fail("not a regular file");
  if (st.st_uid != ::geteuid()) return fail("not owned by this user");
  if ((st.st_mode & 077) != 0) return fail("readable or writable by group/others (chmod 600)");
  if (static_cast<std::uint64_t>(st.st_size) > max_bytes) return fail("larger than " + std::to_string(max_bytes) + " bytes");
  out->resize(static_cast<std::size_t>(st.st_size));
  std::size_t got = 0;
  for (;;) {
    if (got == out->size()) {
      char extra;
      const ssize_t r = ::read(fd, &extra, 1);
      if (r < 0 && errno == EINTR) continue;
      if (r < 0) return fail("read error");
      if (r > 0) return fail("grew while being read");
      break;
    }
    const ssize_t r = ::read(fd, out->data() + got, out->size() - got);
    if (r < 0 && errno == EINTR) continue;
    if (r < 0) return fail("read error");
    if (r == 0) return fail("shorter than its size (truncated while read)");
    got += static_cast<std::size_t>(r);
  }
  ::close(fd);
  return StateRead::kOk;
}

bool StateDir::Init(const std::string& name, std::string_view bytes, std::string* error) const {
  // State first, marker second: an interruption between them leaves a markerless state, which EnsureMarker completes
  // on the next load (the initial state holds nothing to lose).
  if (Write(name, bytes, error) != Persist::kDurable) return false;
  return Write(name + ".init", "initialized\n", error) == Persist::kDurable;
}

bool StateDir::EnsureMarker(const std::string& name, std::string* error) const {
  if (Exists(name + ".init")) return true;
  return Write(name + ".init", "initialized\n", error) == Persist::kDurable;
}

std::unique_ptr<StateLock> StateDir::Lock(const std::string& name, std::string* error) const {
  const std::string ln = name + ".lock";
  const int fd = ::openat(fd_, ln.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) return *error = "cannot open " + path_ + "/" + ln + ": " + std::strerror(errno), nullptr;
  struct stat st {};
  if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != ::geteuid() || (st.st_mode & 077) != 0) {
    ::close(fd);
    return *error = path_ + "/" + ln + " must be a regular 0600 file of ours", nullptr;
  }
  if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
    ::close(fd);
    return *error = path_ + "/" + ln + " is held by another process (another halo_serve on this spill directory?)",
           nullptr;
  }
  return std::unique_ptr<StateLock>(new StateLock(fd));
}

StateLock::~StateLock() {
  if (fd_ >= 0) ::close(fd_);  // releases the flock; the lock file stays (removing it would race a successor)
}

}  // namespace halo::server
