// ROCm HIP backend (module API). The same binary also runs on HRX's HIP
// compatibility layer via LD_PRELOAD, which bench/p0 uses.
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "rt/device.hpp"

namespace halo::rt {

std::unique_ptr<Device> CreateHipDevice(std::string* error);

namespace {

bool Ok(hipError_t e, const char* what, std::string* error) {
  if (e == hipSuccess) return true;
  if (error) *error = std::string(what) + ": " + hipGetErrorString(e);
  return false;
}

class HipDevice final : public Device {
 public:
  ~HipDevice() override {
    if (module_) (void)hipModuleUnload(module_);
    if (stream_) (void)hipStreamDestroy(stream_);
  }
  bool Init(std::string* error) {
    int n = 0;
    if (!Ok(hipGetDeviceCount(&n), "hipGetDeviceCount", error)) return false;
    if (n == 0) {
      if (error) *error = "no HIP device";
      return false;
    }
    if (!Ok(hipSetDevice(0), "hipSetDevice", error) ||
        !Ok(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreate", error))
      return false;
    // Page mapping where the device supports it (HALO_NO_PAGING=1 forces the whole-range fallback).
    int vmm = 0;
    prop_.type = hipMemAllocationTypePinned;
    prop_.location.type = hipMemLocationTypeDevice;
    prop_.location.id = 0;
    std::size_t g = 0;
    if (std::getenv("HALO_NO_PAGING") == nullptr &&
        hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0) == hipSuccess && vmm &&
        hipMemGetAllocationGranularity(&g, &prop_, hipMemAllocationGranularityMinimum) == hipSuccess)
    {
      // The granule may be 4 KB: map in 2 MB pages. HALO_PAGE_BYTES overrides (tests use small pages so that
      // short sequences span several).
      const char* env = std::getenv("HALO_PAGE_BYTES");
      const std::size_t want = env ? std::strtoull(env, nullptr, 0) : kPageTarget;
      page_bytes_ = (std::max<std::size_t>(want, 1) + g - 1) / g * g;
    }
    return true;
  }

  Backend backend() const noexcept override { return Backend::kHip; }

  Buffer Alloc(std::size_t bytes, std::string* error) override {
    Buffer b;
    void* p = nullptr;
    if (!Ok(hipMalloc(&p, bytes ? bytes : 1), "hipMalloc", error)) return b;
    b.device = p;
    b.bytes = bytes;
    return b;
  }
  void Free(Buffer& b) noexcept override {
    if (b.device) (void)hipFree(b.device);
    b = {};
  }
  // ---- P6 page mapping (HIP virtual memory management) ----
  std::size_t PageBytes() const noexcept override { return page_bytes_; }
  Buffer ReserveRange(std::size_t bytes, std::string* error) override {
    Buffer b;
    if (!page_bytes_) {
      if (error) *error = "page mapping: not supported by this device";
      return b;
    }
    const std::size_t n = (bytes + page_bytes_ - 1) / page_bytes_ * page_bytes_;
    void* p = nullptr;
    if (!Ok(hipMemAddressReserve(&p, n ? n : page_bytes_, 0, nullptr, 0), "hipMemAddressReserve", error)) return b;
    b.device = p;
    b.bytes = n ? n : page_bytes_;
    return b;
  }
  void ReleaseRange(Buffer& b) noexcept override {
    if (b.device) (void)hipMemAddressFree(b.device, b.bytes);
    b = {};
  }
  void* CreatePage(std::string* error) override {
    hipMemGenericAllocationHandle_t h{};
    if (!Ok(hipMemCreate(&h, page_bytes_, &prop_, 0), "hipMemCreate", error)) return nullptr;
    return reinterpret_cast<void*>(h);
  }
  void ReleasePage(void* page) noexcept override {
    if (page) (void)hipMemRelease(reinterpret_cast<hipMemGenericAllocationHandle_t>(page));
  }
  bool MapPage(const Buffer& range, std::size_t offset, void* page, std::string* error) override {
    if (offset % page_bytes_ != 0 || offset + page_bytes_ > range.bytes) return Bounds(error);
    void* at = static_cast<char*>(range.device) + offset;
    hipMemAccessDesc acc{};
    acc.location = prop_.location;
    acc.flags = hipMemAccessFlagsProtReadWrite;
    return Ok(hipMemMap(at, page_bytes_, 0, reinterpret_cast<hipMemGenericAllocationHandle_t>(page), 0), "hipMemMap",
              error) &&
           Ok(hipMemSetAccess(at, page_bytes_, &acc, 1), "hipMemSetAccess", error);
  }
  bool UnmapPage(const Buffer& range, std::size_t offset, std::string* error) override {
    if (offset % page_bytes_ != 0 || offset + page_bytes_ > range.bytes) return Bounds(error);
    // Work queued on the stream may still read the page: drain it first.
    return Synchronize(error) && Ok(hipMemUnmap(static_cast<char*>(range.device) + offset, page_bytes_), "hipMemUnmap", error);
  }

  bool Upload(const Buffer& dst, std::size_t off, const void* src, std::size_t n, std::string* error) override {
    if (capturing_) return Capturing(error);
    if (off + n > dst.bytes) return Bounds(error);
    return Ok(hipMemcpyHtoDAsync(static_cast<char*>(dst.device) + off, const_cast<void*>(src), n, stream_),
              "hipMemcpyHtoD", error) &&
           Synchronize(error);
  }
  bool UploadAsync(const Buffer& dst, std::size_t off, const void* src, std::size_t n, std::string* error) override {
    if (capturing_) return Capturing(error);
    if (off + n > dst.bytes) return Bounds(error);
    return Ok(hipMemcpyHtoDAsync(static_cast<char*>(dst.device) + off, const_cast<void*>(src), n, stream_),
              "hipMemcpyHtoDAsync", error);
  }
  bool Download(const Buffer& src, std::size_t off, void* dst, std::size_t n, std::string* error) override {
    if (capturing_) return Capturing(error);
    if (off + n > src.bytes) return Bounds(error);
    return Ok(hipMemcpyDtoHAsync(dst, static_cast<char*>(src.device) + off, n, stream_), "hipMemcpyDtoH", error) &&
           Synchronize(error);
  }
  bool Fill(const Buffer& dst, std::uint8_t v, std::string* error) override {
    return Ok(hipMemsetAsync(dst.device, v, dst.bytes, stream_), "hipMemset", error);
  }
  void* AllocHost(std::size_t bytes, std::string* error) override {
    void* p = nullptr;
    if (!Ok(hipHostMalloc(&p, bytes, hipHostMallocDefault), "hipHostMalloc", error)) return nullptr;
    return p;
  }
  void FreeHost(void* p) noexcept override {
    if (p) (void)hipHostFree(p);
  }
  bool LoadModule(std::span<const std::uint8_t> hsaco, std::string* error) override {
    image_.assign(hsaco.begin(), hsaco.end());
    return Ok(hipModuleLoadData(&module_, image_.data()), "hipModuleLoadData", error);
  }
  bool ReadGlobal(std::string_view name, void* dst, std::size_t bytes, std::string* error) override {
    void* p = nullptr;
    return Global(name, bytes, &p, error) && Ok(hipMemcpyDtoH(dst, p, bytes), "hipMemcpyDtoH", error);
  }
  bool WriteGlobal(std::string_view name, const void* src, std::size_t bytes, std::string* error) override {
    void* p = nullptr;
    return Global(name, bytes, &p, error) &&
           Ok(hipMemcpyHtoD(p, const_cast<void*>(src), bytes), "hipMemcpyHtoD", error);
  }
  KernelId Kernel(std::string_view name, std::string* error) override {
    hipFunction_t f = nullptr;
    if (!Ok(hipModuleGetFunction(&f, module_, std::string(name).c_str()), "hipModuleGetFunction", error)) return -1;
    functions_.push_back(f);
    return static_cast<KernelId>(functions_.size() - 1);
  }
  bool Launch(KernelId k, Dim3 grid, Dim3 block, std::span<const Arg> args, std::string* error) override {
    if (k < 0 || static_cast<std::size_t>(k) >= functions_.size()) {
      if (error) *error = "bad kernel id";
      return false;
    }
    std::vector<std::uint64_t> storage(args.size());
    std::vector<void*> params(args.size());
    for (std::size_t i = 0; i < args.size(); ++i) {
      if (args[i].kind == Arg::Kind::kBuffer) {
        void* p = static_cast<char*>(args[i].buffer.device) + args[i].offset;
        __builtin_memcpy(&storage[i], &p, sizeof(p));
      } else {
        __builtin_memcpy(&storage[i], args[i].bytes, args[i].size);
      }
      params[i] = &storage[i];
    }
    return Ok(hipModuleLaunchKernel(functions_[static_cast<std::size_t>(k)], grid.x, grid.y, grid.z, block.x,
                                    block.y, block.z, 0, stream_, params.data(), nullptr),
              "hipModuleLaunchKernel", error);
  }
  bool Synchronize(std::string* error) override {
    if (capturing_) return Capturing(error);
    return Ok(hipStreamSynchronize(stream_), "hipStreamSynchronize", error);
  }
  bool BeginCapture(std::string* error) override {
    if (!Ok(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal), "hipStreamBeginCapture", error))
      return false;
    capturing_ = true;
    return true;
  }
  bool EndCapture(Graph* out, std::string* error) override {
    capturing_ = false;
    hipGraph_t g = nullptr;
    if (!Ok(hipStreamEndCapture(stream_, &g), "hipStreamEndCapture", error)) return false;
    hipGraphExec_t exec = nullptr;
    const bool ok = Ok(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0), "hipGraphInstantiate", error);
    (void)hipGraphDestroy(g);
    if (ok) out->exec = exec;
    return ok;
  }
  bool LaunchGraph(const Graph& graph, std::string* error) override {
    return Ok(hipGraphLaunch(static_cast<hipGraphExec_t>(graph.exec), stream_), "hipGraphLaunch", error);
  }
  void FreeGraph(Graph& graph) noexcept override {
    if (graph.exec) (void)hipGraphExecDestroy(static_cast<hipGraphExec_t>(graph.exec));
    graph = {};
  }

 private:
  static bool Bounds(std::string* error) {
    if (error) *error = "copy out of buffer bounds";
    return false;
  }
  static bool Capturing(std::string* error) {
    if (error) *error = "not capturable: transfer or synchronize during graph capture";
    return false;
  }
  bool capturing_ = false;
  hipStream_t stream_ = nullptr;
  static constexpr std::size_t kPageTarget = std::size_t{2} << 20;
  hipMemAllocationProp prop_{};
  std::size_t page_bytes_ = 0;
  // The device address of a module global holding at least `bytes`.
  bool Global(std::string_view name, std::size_t bytes, void** p, std::string* error) {
    hipDeviceptr_t d = nullptr;
    std::size_t size = 0;
    if (!Ok(hipModuleGetGlobal(&d, &size, module_, std::string(name).c_str()), "hipModuleGetGlobal", error)) return false;
    if (size < bytes) {
      if (error) *error = std::string(name) + ": global holds " + std::to_string(size) + " bytes, asked " + std::to_string(bytes);
      return false;
    }
    *p = d;
    return true;
  }
  hipModule_t module_ = nullptr;
  std::vector<std::uint8_t> image_;
  std::vector<hipFunction_t> functions_;
};

}  // namespace

std::unique_ptr<Device> CreateHipDevice(std::string* error) {
  auto d = std::make_unique<HipDevice>();
  if (!d->Init(error)) return nullptr;
  return d;
}

}  // namespace halo::rt
