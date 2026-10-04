#include "rt/device.hpp"

namespace halo::rt {

#if HALO_WITH_HIP
std::unique_ptr<Device> CreateHipDevice(std::string* error);
#endif
#if HALO_WITH_HRX
std::unique_ptr<Device> CreateHrxDevice(std::string* error);
#endif

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

}  // namespace halo::rt
