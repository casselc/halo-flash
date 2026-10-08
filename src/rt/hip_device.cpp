// ROCm HIP backend (module API). The same binary also runs on HRX's HIP
// compatibility layer via LD_PRELOAD, which bench/p0 uses.
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <thread>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
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
    if (range_mode_ == RangeMode::kPoison) (void)PoisonCheck("exit");
    for (const auto& [bytes, p] : range_pool_) (void)hipMemAddressFree(p, bytes);
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
    // Diagnostics (off by default; docs/SESSIONS.md section 11): HALO_DIAG_VA=1 logs every allocation, range and
    // page event and every graph launch to stderr with CLOCK_MONOTONIC microseconds (the clock of the kernel log
    // and of the HIP runtime's AMD_LOG_LEVEL lines). HALO_DIAG_RANGE picks what ReleaseRange does: "free"
    // (the default: hipMemAddressFree, the runtime reuses the address), "pool" (A/B only: kept reserved and handed
    // out again, which keeps released pages charged), "poison" (kept reserved, never reused, a sentinel page holding
    // a pattern mapped where the range had pages; checked at every ReserveRange and at exit, so a stale write into a
    // released range shows; also keeps released pages charged).
    if (const char* t = std::getenv("HALO_VMM_SETTLE_US")) settle_us_ = std::strtoul(t, nullptr, 0);
    diag_log_ = std::getenv("HALO_DIAG_VA") != nullptr;
    // HALO_DIAG_MAP_DELAY_US: sleep this long after each page mapping (is a write right after the mapping too early?)
    if (const char* d = std::getenv("HALO_DIAG_MAP_DELAY_US")) map_delay_us_ = std::strtoul(d, nullptr, 0);
    if (const char* m = std::getenv("HALO_DIAG_RANGE")) {
      if (std::strcmp(m, "pool") == 0) range_mode_ = RangeMode::kPool;
      else if (std::strcmp(m, "poison") == 0) range_mode_ = RangeMode::kPoison;
    }
    Diag("start mode %d realtime %lld", static_cast<int>(range_mode_), static_cast<long long>(std::time(nullptr)));
    return true;
  }


  Backend backend() const noexcept override { return Backend::kHip; }

  Buffer Alloc(std::size_t bytes, std::string* error) override {
    Buffer b;
    void* p = nullptr;
    if (!Ok(hipMalloc(&p, bytes ? bytes : 1), "hipMalloc", error)) return b;
    b.device = p;
    b.bytes = bytes;
    Diag("alloc %p %zu", p, bytes);
    return b;
  }
  void Free(Buffer& b) noexcept override {
    if (b.device) {
      Diag("free %p %zu", b.device, b.bytes);
      (void)hipFree(b.device);
    }
    b = {};
  }
  // ---- P6 page mapping (HIP virtual memory management) ----
  std::size_t PageBytes() const noexcept override { return page_bytes_; }
  std::size_t AvailableBytes() const override {
    // GTT headroom (amdgpu sysfs) and host MemAvailable: the same DRAM backs both on Strix Halo.
    std::size_t best = SIZE_MAX;
    for (int card = 0; card < 8; ++card) {
      const std::string d = "/sys/class/drm/card" + std::to_string(card) + "/device/mem_info_gtt_";
      std::ifstream t(d + "total"), u(d + "used");
      std::size_t total = 0, used = 0;
      if (t >> total && u >> used) {
        best = total > used ? total - used : 0;
        break;
      }
    }
    std::ifstream m("/proc/meminfo");
    std::string key, unit;
    std::size_t kb = 0;
    while (m >> key >> kb >> unit)
      if (key == "MemAvailable:") {
        best = std::min(best, kb * 1024);
        break;
      }
    return best;
  }
  Buffer ReserveRange(std::size_t bytes, std::string* error) override {
    Buffer b;
    if (!page_bytes_) {
      if (error) *error = "page mapping: not supported by this device";
      return b;
    }
    const std::size_t n = std::max((bytes + page_bytes_ - 1) / page_bytes_ * page_bytes_, page_bytes_);
    if (range_mode_ == RangeMode::kPoison) (void)PoisonCheck("reserve");
    if (range_mode_ == RangeMode::kPool) {
      std::lock_guard<std::mutex> lk(range_mu_);
      if (auto it = range_pool_.find(n); it != range_pool_.end()) {  // a released range of this size, still reserved
        b.device = it->second;
        b.bytes = n;
        range_pool_.erase(it);
        Diag("reserve %p %zu pooled", b.device, n);
        return b;
      }
    }
    void* p = nullptr;
    if (!Ok(hipMemAddressReserve(&p, n, 0, nullptr, 0), "hipMemAddressReserve", error)) return b;
    b.device = p;
    b.bytes = n;
    Diag("reserve %p %zu", p, n);
    return b;
  }
  // A released range (its pages already unmapped and released) goes back to the runtime. The memory of every page
  // ever mapped into a range stays charged until hipMemAddressFree of that range (ROCm 7.2.3: 4 GB of unmapped,
  // released pages in a kept range stayed in GTT; freeing the range returned them), so keeping released ranges
  // (HALO_DIAG_RANGE=pool, stream MS's 6240e69) grows memory with every session and ran 128K contexts out of memory.
  // The MTP-serving fault that pool was for is the mapping lag the VMM settle covers (docs/SESSIONS.md section 11).
  void ReleaseRange(Buffer& b) noexcept override {
    if (b.device) {
      Diag("release %p %zu", b.device, b.bytes);
      if (range_mode_ == RangeMode::kFree) {
        (void)hipMemAddressFree(b.device, b.bytes);
        Stamp();
      } else if (range_mode_ == RangeMode::kPoison) {
        Poison(b);
      } else {
        std::lock_guard<std::mutex> lk(range_mu_);
        range_pool_.emplace(b.bytes, b.device);
      }
    }
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
    Diag("map %p page %p", at, page);
    if (range_mode_ == RangeMode::kPoison) mapped_[range.device].insert(offset);
    const bool ok = MapAt(at, page, error);
    Stamp();
    if (map_delay_us_) std::this_thread::sleep_for(std::chrono::microseconds(map_delay_us_));
    return ok;
  }
  bool UnmapPage(const Buffer& range, std::size_t offset, std::string* error) override {
    if (offset % page_bytes_ != 0 || offset + page_bytes_ > range.bytes) return Bounds(error);
    // Work queued on the stream may still read the page: drain it first.
    Diag("unmap %p", static_cast<char*>(range.device) + offset);
    const bool ok =
        Synchronize(error) && Ok(hipMemUnmap(static_cast<char*>(range.device) + offset, page_bytes_), "hipMemUnmap", error);
    Stamp();
    return ok;
  }

  bool Upload(const Buffer& dst, std::size_t off, const void* src, std::size_t n, std::string* error) override {
    Settle();
    if (capturing_) return Capturing(error);
    if (off + n > dst.bytes) return Bounds(error);
    return Ok(hipMemcpyHtoDAsync(static_cast<char*>(dst.device) + off, const_cast<void*>(src), n, stream_),
              "hipMemcpyHtoD", error) &&
           Synchronize(error);
  }
  bool UploadAsync(const Buffer& dst, std::size_t off, const void* src, std::size_t n, std::string* error) override {
    Settle();
    if (capturing_) return Capturing(error);
    if (off + n > dst.bytes) return Bounds(error);
    return Ok(hipMemcpyHtoDAsync(static_cast<char*>(dst.device) + off, const_cast<void*>(src), n, stream_),
              "hipMemcpyHtoDAsync", error);
  }
  bool Download(const Buffer& src, std::size_t off, void* dst, std::size_t n, std::string* error) override {
    Settle();
    if (capturing_) return Capturing(error);
    if (off + n > src.bytes) return Bounds(error);
    return Ok(hipMemcpyDtoHAsync(dst, static_cast<char*>(src.device) + off, n, stream_), "hipMemcpyDtoH", error) &&
           Synchronize(error);
  }
  bool Fill(const Buffer& dst, std::uint8_t v, std::string* error) override {
    Settle();
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
    Settle();
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
    if (ok) Diag("graph-new %p", exec);
    return ok;
  }
  bool LaunchGraph(const Graph& graph, std::string* error) override {
    Settle();
    Diag("graph-launch %p", graph.exec);
    return Ok(hipGraphLaunch(static_cast<hipGraphExec_t>(graph.exec), stream_), "hipGraphLaunch", error);
  }
  void FreeGraph(Graph& graph) noexcept override {
    if (graph.exec) Diag("graph-free %p", graph.exec);
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
  bool MapAt(void* at, void* page, std::string* error) {
    hipMemAccessDesc acc{};
    acc.location = prop_.location;
    acc.flags = hipMemAccessFlagsProtReadWrite;
    return Ok(hipMemMap(at, page_bytes_, 0, reinterpret_cast<hipMemGenericAllocationHandle_t>(page), 0), "hipMemMap",
              error) &&
           Ok(hipMemSetAccess(at, page_bytes_, &acc, 1), "hipMemSetAccess", error);
  }
  // ---- VMM settle (docs/SESSIONS.md section 11) ----
  // On this stack (ROCm 7.2.3, gfx1151) the GPU's address translation lags hipMemMap / hipMemSetAccess /
  // hipMemUnmap by tens of microseconds after they return: a kernel writing a page right after it was mapped faulted
  // (MTP serving, fork copies), and one writing a VA right after its page was unmapped and another mapped there
  // wrote the OLD, released page (ana/vmm_stale in stream RC: 100 of 100; >= 60 us later: 0 of 100). No HIP
  // synchronization orders it (stream, null stream, device). So no GPU work is submitted until settle_us_ after the
  // last mapping change. HALO_VMM_SETTLE_US overrides (0: off, for the regression test's positive control).
  unsigned long settle_us_ = 1000;
  std::chrono::steady_clock::time_point vmm_changed_{};
  bool vmm_pending_ = false;
  void Stamp() noexcept {
    vmm_changed_ = std::chrono::steady_clock::now();
    vmm_pending_ = settle_us_ > 0;
  }
  void Settle() noexcept {
    if (!vmm_pending_) return;
    vmm_pending_ = false;
    const auto until = vmm_changed_ + std::chrono::microseconds(settle_us_);
    const auto now = std::chrono::steady_clock::now();
    if (now < until) std::this_thread::sleep_for(until - now);
  }
  // ---- diagnostics ----
  enum class RangeMode { kPool, kFree, kPoison };
  RangeMode range_mode_ = RangeMode::kFree;
  bool diag_log_ = false;
  unsigned long map_delay_us_ = 0;
  static constexpr std::uint32_t kPoisonWord = 0x7fc0dead;  // a NaN payload no kernel writes
  void* sentinel_ = nullptr;                                 // poison: the page mapped into released ranges
  std::map<void*, std::set<std::size_t>> mapped_;            // poison: offsets each live range ever had mapped
  template <typename... A>
  void Diag(const char* fmt, A... a) noexcept {
    if (!diag_log_) return;
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    char line[160];
    std::snprintf(line, sizeof line, fmt, a...);
    std::fprintf(stderr, "diagva %" PRId64 " %s\n", std::int64_t{ts.tv_sec} * 1000000 + ts.tv_nsec / 1000, line);
  }
  // Released range in poison mode: never reused; the sentinel page is mapped (read-write) at each offset that held
  // a page, so a stale write lands in the sentinel instead of faulting or reaching another session.
  void Poison(const Buffer& b) noexcept {
    std::string e;
    if (!sentinel_) {
      sentinel_ = CreatePage(&e);
      void* at = nullptr;
      if (!sentinel_ || !Ok(hipMemAddressReserve(&at, page_bytes_, 0, nullptr, 0), "reserve", &e) ||
          !MapAt(at, sentinel_, &e) || hipMemsetD32Async(at, kPoisonWord, page_bytes_ / 4, stream_) != hipSuccess ||
          hipStreamSynchronize(stream_) != hipSuccess) {
        std::fprintf(stderr, "diagva poison setup failed: %s\n", e.c_str());
        std::abort();
      }
      sentinel_view_ = at;
    }
    const auto it = mapped_.find(b.device);
    if (it == mapped_.end()) return;
    for (std::size_t off : it->second)
      if (!MapAt(static_cast<char*>(b.device) + off, sentinel_, &e))
        std::fprintf(stderr, "diagva poison map %p+%zu failed: %s\n", b.device, off, e.c_str());
    Stamp();
    Diag("poisoned %p %zu pages", b.device, it->second.size());
    mapped_.erase(it);
  }
  void* sentinel_view_ = nullptr;
  // Compares the sentinel with the pattern; reports and repairs a change. Returns false on a stale write.
  bool PoisonCheck(const char* when) noexcept {
    if (!sentinel_view_) return true;
    Settle();
    std::vector<std::uint32_t> h(page_bytes_ / 4);
    if (hipStreamSynchronize(stream_) != hipSuccess ||
        hipMemcpyDtoH(h.data(), sentinel_view_, page_bytes_) != hipSuccess) {
      std::fprintf(stderr, "diagva poison check (%s): device error\n", when);
      return false;
    }
    std::size_t first = SIZE_MAX, n = 0;
    for (std::size_t i = 0; i < h.size(); ++i)
      if (h[i] != kPoisonWord) {
        if (first == SIZE_MAX) first = i;
        ++n;
      }
    if (n == 0) return true;
    std::fprintf(stderr, "diagva POISON-HIT (%s): %zu words changed, first at byte %zu (0x%08x)\n", when, n, first * 4,
                 h[first]);
    (void)hipMemsetD32Async(sentinel_view_, kPoisonWord, page_bytes_ / 4, stream_);
    (void)hipStreamSynchronize(stream_);
    return false;
  }
  bool capturing_ = false;
  std::mutex range_mu_;
  std::multimap<std::size_t, void*> range_pool_;  // released ranges by size, still reserved (guarded by range_mu_)
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
