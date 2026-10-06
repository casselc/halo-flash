#include "core/atomic_file.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <filesystem>

namespace halo::core {

namespace {

std::atomic<int> g_fault{0};

bool Fail(std::string* error, const std::string& what) {
  if (error) *error = what + ": " + std::strerror(errno);
  return false;
}

bool WriteAll(int fd, const std::uint8_t* p, std::size_t n) {
  while (n > 0) {
    const ssize_t w = ::write(fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += w;
    n -= static_cast<std::size_t>(w);
  }
  return true;
}

bool SyncDir(const std::string& path) {
  auto dir = std::filesystem::path(path).parent_path();
  if (dir.empty()) dir = ".";
  const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return false;
  const bool ok = ::fsync(fd) == 0;
  ::close(fd);
  return ok;
}

}  // namespace

void SetAtomicWriteFaultForTest(int stage) noexcept { g_fault.store(stage); }

bool AtomicWriteFile(const std::string& path, std::span<const std::uint8_t> bytes, std::string* error) {
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) return Fail(error, "cannot create " + tmp);
  if (!WriteAll(fd, bytes.data(), bytes.size())) {
    const int e = errno;
    ::close(fd);
    ::unlink(tmp.c_str());
    errno = e;
    return Fail(error, "write " + tmp);
  }
  if (g_fault.load() == 1) {
    ::close(fd);
    if (error) *error = "injected fault after temp write";
    return false;
  }
  if (::fsync(fd) != 0) {
    const int e = errno;
    ::close(fd);
    errno = e;
    return Fail(error, "fsync " + tmp);
  }
  ::close(fd);
  if (g_fault.load() == 2) {
    if (error) *error = "injected fault before rename";
    return false;
  }
  if (::rename(tmp.c_str(), path.c_str()) != 0) return Fail(error, "rename " + tmp);
  if (!SyncDir(path)) return Fail(error, "fsync directory of " + path);
  return true;
}

bool DurableAppend(const std::string& path, std::string_view record, std::string* error) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
  if (fd < 0) return Fail(error, "cannot open " + path);
  const bool ok = WriteAll(fd, reinterpret_cast<const std::uint8_t*>(record.data()), record.size()) &&
                  ::fdatasync(fd) == 0;
  const int e = errno;
  ::close(fd);
  errno = e;
  return ok ? true : Fail(error, "append " + path);
}

bool ReadWholeFile(const std::string& path, std::string* out, std::string* error) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return Fail(error, "cannot open " + path);
  out->clear();
  char buf[1 << 16];
  for (;;) {
    const ssize_t n = ::read(fd, buf, sizeof buf);
    if (n < 0) {
      if (errno == EINTR) continue;
      const int e = errno;
      ::close(fd);
      errno = e;
      return Fail(error, "read " + path);
    }
    if (n == 0) break;
    out->append(buf, static_cast<std::size_t>(n));
  }
  ::close(fd);
  return true;
}

}  // namespace halo::core
