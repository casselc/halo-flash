#include "server/state_file.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace halo::server {

namespace {

std::pair<std::string, std::string> Split(const std::string& path) {
  const auto slash = path.rfind('/');
  if (slash == std::string::npos) return {".", path};
  return {slash == 0 ? "/" : path.substr(0, slash), path.substr(slash + 1)};
}

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

bool Exists(const std::string& p) {
  struct stat st {};
  return ::lstat(p.c_str(), &st) == 0;
}

}  // namespace

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

Persist WriteStateFile(const std::string& path, std::string_view bytes, std::string* error) {
  const int fault = g_state_write_fault;
  g_state_write_fault = 0;
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  ::unlink(tmp.c_str());
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) return *error = "cannot create " + tmp + ": " + std::strerror(errno), Persist::kUnpublished;
  const bool synced = WriteAll(fd, bytes) && ::fsync(fd) == 0;
  ::close(fd);
  if (!synced || fault == 1 || ::rename(tmp.c_str(), path.c_str()) != 0) {
    ::unlink(tmp.c_str());
    return *error = "cannot publish " + path + (fault == 1 ? " (injected)" : ""), Persist::kUnpublished;
  }
  // Published: from here a failure is "uncertain", never "not done".
  const auto [dir, base] = Split(path);
  const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  const bool dir_ok = fault != 2 && dfd >= 0 && ::fsync(dfd) == 0;
  if (dfd >= 0) ::close(dfd);
  if (!dir_ok) return *error = "fsync of " + dir + " failed after publishing " + path, Persist::kUncertain;
  return Persist::kDurable;
}

StateRead ReadStateFile(const std::string& path, std::size_t max_bytes, std::string* out, std::string* error) {
  out->clear();
  const auto [dir, base] = Split(path);
  const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (dfd < 0) return *error = "state directory " + dir + ": " + std::strerror(errno), StateRead::kError;
  const int fd = ::openat(dfd, base.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  const int open_errno = errno;
  ::close(dfd);
  if (fd < 0) {
    if (open_errno != ENOENT) return *error = path + ": " + std::strerror(open_errno), StateRead::kError;
    if (Exists(path + ".init"))
      return *error = path + " is missing but was initialized (" + path +
                      ".init exists): recovery required, refusing to start from zero",
             StateRead::kError;
    return StateRead::kFresh;
  }
  struct stat st {};
  const auto fail = [&](std::string why) {
    ::close(fd);
    *error = path + ": " + why;
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

bool InitStateFile(const std::string& path, std::string_view bytes, std::string* error) {
  if (WriteStateFile(path + ".init", "initialized\n", error) != Persist::kDurable) return false;
  return WriteStateFile(path, bytes, error) == Persist::kDurable;
}

std::unique_ptr<StateLock> StateLock::Acquire(const std::string& path, std::string* error) {
  const std::string lp = path + ".lock";
  const int fd = ::open(lp.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) return *error = "cannot open " + lp + ": " + std::strerror(errno), nullptr;
  if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
    ::close(fd);
    return *error = lp + " is held by another process (another halo_serve on this spill directory?)", nullptr;
  }
  return std::unique_ptr<StateLock>(new StateLock(fd));
}

StateLock::~StateLock() {
  if (fd_ >= 0) ::close(fd_);  // releases the flock; the lock file stays (removing it would race a successor)
}

}  // namespace halo::server
