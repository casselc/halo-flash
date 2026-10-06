#include "rt/device.hpp"

#include <cstdlib>

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

std::unique_ptr<Device> Device::Create(Backend backend, std::string* error) {
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
