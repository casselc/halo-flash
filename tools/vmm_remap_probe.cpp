// Minimal HIP virtual-memory remap probe (docs/SESSIONS.md section 11; no engine, no model).
//
// Reserve an address range; map page A, write it, synchronize; unmap and release A; map page S at the same address
// (S is also mapped at a second address, to read it back); write 4 KB there right away; synchronize; read S.
// On ROCm 7.2.3 / gfx1151 (kernel 7.0) every such write landed in the old, released page A and S kept its pattern,
// unless the program waited >= ~60 us between the unmap and the remap or ~100 us after the remap. No HIP
// synchronization (stream, null stream, device) orders it. rt::HipDevice therefore holds GPU work until
// HALO_VMM_SETTLE_US after the last mapping change.
//
// usage: halo_vmm_remap_probe [reps] [wait_us_between_unmap_and_remap] [wait_us_after_remap]
// prints how many writes reached the newly mapped page; exit status 1 if any did not.
#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

#define CK(x)                                                                \
  do {                                                                       \
    const hipError_t e_ = (x);                                               \
    if (e_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_));           \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

constexpr std::size_t kPage = std::size_t{2} << 20;
hipMemAllocationProp g_prop{};

void Map(void* at, hipMemGenericAllocationHandle_t h) {
  CK(hipMemMap(at, kPage, 0, h, 0));
  hipMemAccessDesc a{};
  a.location = g_prop.location;
  a.flags = hipMemAccessFlagsProtReadWrite;
  CK(hipMemSetAccess(at, kPage, &a, 1));
}

void WaitUs(long us) {
  if (us > 0) std::this_thread::sleep_for(std::chrono::microseconds(us));
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 100;
  const long between = argc > 2 ? std::atol(argv[2]) : 0, after = argc > 3 ? std::atol(argv[3]) : 0;
  CK(hipSetDevice(0));
  hipStream_t s;
  CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  g_prop.type = hipMemAllocationTypePinned;
  g_prop.location.type = hipMemLocationTypeDevice;
  g_prop.location.id = 0;
  hipMemGenericAllocationHandle_t sentinel{};
  CK(hipMemCreate(&sentinel, kPage, &g_prop, 0));
  void* view = nullptr;
  CK(hipMemAddressReserve(&view, kPage, 0, nullptr, 0));
  Map(view, sentinel);
  int reached = 0;
  for (int r = 0; r < reps; ++r) {
    CK(hipMemsetD32Async(view, 0x7fc0dead, kPage / 4, s));
    void* range = nullptr;
    CK(hipMemAddressReserve(&range, 2 * kPage, 0, nullptr, 0));
    char* at = static_cast<char*>(range) + kPage;
    hipMemGenericAllocationHandle_t a{};
    CK(hipMemCreate(&a, kPage, &g_prop, 0));
    Map(at, a);
    CK(hipMemsetAsync(at, 0x11, kPage, s));
    CK(hipStreamSynchronize(s));
    CK(hipMemUnmap(at, kPage));
    CK(hipMemRelease(a));
    WaitUs(between);
    Map(at, sentinel);
    WaitUs(after);
    CK(hipMemsetAsync(at + 8192, 0x55, 4096, s));
    CK(hipStreamSynchronize(s));
    unsigned got = 0;
    CK(hipMemcpyDtoH(&got, static_cast<char*>(view) + 8192, 4));
    reached += got == 0x55555555u;
    CK(hipMemUnmap(at, kPage));
    CK(hipMemAddressFree(range, 2 * kPage));
  }
  std::printf("wait %ld us between unmap and remap, %ld us after: %d of %d writes reached the newly mapped page\n",
              between, after, reached, reps);
  return reached == reps ? 0 : 1;
}
