#ifndef HALO_RT_DEVICE_HPP_
#define HALO_RT_DEVICE_HPP_

// One device interface over two runtimes, so every kernel and model runs (and
// is tested) on both: ROCm HIP and native HRX (libhrx). Kernels are compiled
// once to a gfx1151 HSACO (src/kernels/*.hip) and loaded by either runtime.
//
// Kernel ABI rule shared by both backends: pointer parameters are buffers
// (passed as Arg::Buf), everything else is a by-value scalar. HIP receives the
// kernarg list as written; HRX receives pointers as bindings in order and the
// scalars concatenated tightly, which is how libhrx maps HSACO metadata.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

namespace halo::rt {

enum class Backend { kHip, kHrx };
const char* BackendName(Backend b) noexcept;

/// Device memory. `device` is the address kernels see; `handle` is the
/// backend's buffer object (hrx_buffer_t on HRX, null on HIP).
struct Buffer {
  void* device = nullptr;
  std::size_t bytes = 0;
  void* handle = nullptr;
  explicit operator bool() const noexcept { return device != nullptr; }
};

struct Arg {
  enum class Kind : std::uint8_t { kBuffer, kScalar };
  Kind kind = Kind::kScalar;
  Buffer buffer;
  std::size_t offset = 0;
  std::uint8_t bytes[8] = {};
  std::uint8_t size = 0;

  static Arg Buf(const Buffer& b, std::size_t offset_bytes = 0) {
    Arg a;
    a.kind = Kind::kBuffer;
    a.buffer = b;
    a.offset = offset_bytes;
    return a;
  }
  template <typename T>
  static Arg Val(T v) {
    static_assert(sizeof(T) <= 8 && std::is_trivially_copyable_v<T>);
    Arg a;
    a.kind = Kind::kScalar;
    a.size = sizeof(T);
    __builtin_memcpy(a.bytes, &v, sizeof(T));
    return a;
  }
};

struct Dim3 {
  std::uint32_t x = 1, y = 1, z = 1;
};

/// An opaque kernel handle from LoadModule/Kernel.
using KernelId = std::int32_t;

/// An instantiated, replayable launch sequence (HIP graph exec or HRX graph
/// exec), recorded by stream capture. Arguments, including buffer bindings
/// and scalar values, are frozen at capture; per-replay values must live in
/// device memory.
struct Graph {
  void* exec = nullptr;
  explicit operator bool() const noexcept { return exec != nullptr; }
};

class Device {
 public:
  virtual ~Device() = default;
  [[nodiscard]] static std::unique_ptr<Device> Create(Backend backend, std::string* error);

  [[nodiscard]] virtual Backend backend() const noexcept = 0;
  [[nodiscard]] virtual Buffer Alloc(std::size_t bytes, std::string* error) = 0;
  virtual void Free(Buffer& buffer) noexcept = 0;
  [[nodiscard]] virtual bool Upload(const Buffer& dst, std::size_t offset, const void* src, std::size_t bytes,
                                    std::string* error) = 0;
  [[nodiscard]] virtual bool Download(const Buffer& src, std::size_t offset, void* dst, std::size_t bytes,
                                      std::string* error) = 0;
  /// Stream-ordered upload that may return before the copy is done: the source must stay unchanged until the
  /// next Synchronize / Download. Default: the synchronous Upload.
  [[nodiscard]] virtual bool UploadAsync(const Buffer& dst, std::size_t offset, const void* src, std::size_t bytes,
                                         std::string* error) {
    return Upload(dst, offset, src, bytes, error);
  }
  [[nodiscard]] virtual bool Fill(const Buffer& dst, std::uint8_t value, std::string* error) = 0;
  /// Host memory the backend can copy from at full speed (pinned where the
  /// runtime supports it), 4096-byte aligned. Default: plain aligned memory.
  [[nodiscard]] virtual void* AllocHost(std::size_t bytes, std::string* error);
  virtual void FreeHost(void* p) noexcept;

  /// Loads one HSACO code object; kernels are then looked up by name.
  [[nodiscard]] virtual bool LoadModule(std::span<const std::uint8_t> hsaco, std::string* error) = 0;
  [[nodiscard]] virtual KernelId Kernel(std::string_view name, std::string* error) = 0;
  /// Copies a module-scope __device__ variable of the loaded module to or from
  /// the host (the span tracer's records: src/kernels/span_trace.h). HIP only;
  /// other backends report unsupported.
  [[nodiscard]] virtual bool ReadGlobal(std::string_view name, void* dst, std::size_t bytes, std::string* error);
  [[nodiscard]] virtual bool WriteGlobal(std::string_view name, const void* src, std::size_t bytes,
                                         std::string* error);
  [[nodiscard]] virtual bool Launch(KernelId kernel, Dim3 grid, Dim3 block, std::span<const Arg> args,
                                    std::string* error) = 0;
  [[nodiscard]] virtual bool Synchronize(std::string* error) = 0;

  /// Launches and fills between Begin/EndCapture are recorded instead of run.
  /// Uploads, downloads and Synchronize are not capturable and fail inside.
  [[nodiscard]] virtual bool BeginCapture(std::string* error) = 0;
  [[nodiscard]] virtual bool EndCapture(Graph* out, std::string* error) = 0;
  [[nodiscard]] virtual bool LaunchGraph(const Graph& graph, std::string* error) = 0;
  virtual void FreeGraph(Graph& graph) noexcept = 0;
};

}  // namespace halo::rt

#endif  // HALO_RT_DEVICE_HPP_
