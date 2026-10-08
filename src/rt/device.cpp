#include "rt/device.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace halo::rt {

#if HALO_WITH_HIP
std::unique_ptr<Device> CreateHipDevice(std::string* error);
#endif
#if HALO_WITH_HRX
std::unique_ptr<Device> CreateHrxDevice(std::string* error);
#endif

void* Device::AllocHost(std::size_t bytes, std::string* error) {
  void* p = std::aligned_alloc(4096, (bytes + 4095) / 4096 * 4096);
  if (!p && error) *error = "host allocation failed";
  return p;
}
void Device::FreeHost(void* p) noexcept { std::free(p); }

bool Device::ReadGlobal(std::string_view, void*, std::size_t, std::string* error) {
  if (error) *error = "ReadGlobal: not supported by this backend";
  return false;
}

bool Device::WriteGlobal(std::string_view, const void*, std::size_t, std::string* error) {
  if (error) *error = "WriteGlobal: not supported by this backend";
  return false;
}

namespace {
bool NoPaging(std::string* error) {
  if (error) *error = "page mapping: not supported by this backend";
  return false;
}
}  // namespace
Buffer Device::ReserveRange(std::size_t, std::string* error) {
  NoPaging(error);
  return {};
}
void Device::ReleaseRange(Buffer& range) noexcept { range = {}; }
void* Device::CreatePage(std::string* error) {
  NoPaging(error);
  return nullptr;
}
void Device::ReleasePage(void*) noexcept {}
bool Device::MapPage(const Buffer&, std::size_t, void*, std::string* error) { return NoPaging(error); }
bool Device::UnmapPage(const Buffer&, std::size_t, std::string* error) { return NoPaging(error); }

const char* BackendName(Backend b) noexcept { return b == Backend::kHip ? "hip" : "hrx"; }

namespace {
// The shared-GPU rule (/data/halo-work/streams/RULES.md): every process that opens the GPU runs under
// `flock <lock> env HALO_GPU_LOCKED=1 ...`. Two model instances at once run the host out of memory (2026-10-07: a
// bench started without the lock overlapped another stream's real-model run and both failed), so a device is refused
// unless the lock is held right now AND this process was started under it. The check applies only on hosts where the
// lock file exists; HALO_GPU_LOCK_ENFORCE=0 disables it (e.g. a machine with its own GPU), HALO_GPU_LOCK_FILE moves it.
bool GpuLockOk(std::string* error) {
  const char* enforce = std::getenv("HALO_GPU_LOCK_ENFORCE");
  if (enforce && std::string_view(enforce) == "0") return true;
  const char* f = std::getenv("HALO_GPU_LOCK_FILE");
  const std::string path = f && *f ? f : "/data/halo-work/gpu.lock";
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return true;  // no shared-GPU policy on this host
  const bool free = ::flock(fd, LOCK_EX | LOCK_NB) == 0;  // nobody holds it: we are not running under it
  if (free) ::flock(fd, LOCK_UN);
  ::close(fd);
  const char* held = std::getenv("HALO_GPU_LOCKED");
  if (!free && held && std::string_view(held) == "1") return true;
  const std::string why = std::string("GPU lock not held: run under `flock ") + path +
                          " env HALO_GPU_LOCKED=1 <cmd>` (" +
                          (free ? "the lock is free" : "another process holds the lock") +
                          "; HALO_GPU_LOCK_ENFORCE=0 turns this check off)";
  std::fprintf(stderr, "halo: %s\n", why.c_str());  // loud: tests that skip on a missing device must not hide it
  if (error) *error = why;
  return false;
}
}  // namespace

std::unique_ptr<Device> Device::Create(Backend backend, std::string* error) {
  if (!GpuLockOk(error)) return nullptr;
  switch (backend) {
    case Backend::kHip:
#if HALO_WITH_HIP
      return CreateHipDevice(error);
#else
      break;
#endif
    case Backend::kHrx:
#if HALO_WITH_HRX
      return CreateHrxDevice(error);
#else
      break;
#endif
  }
  if (error) *error = std::string("backend not built: ") + BackendName(backend);
  return nullptr;
}

std::size_t Device::AvailableBytes() const { return SIZE_MAX; }

}  // namespace halo::rt
