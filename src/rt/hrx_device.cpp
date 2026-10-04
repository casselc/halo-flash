// Native HRX backend (libhrx C API). Needs an HSA runtime that provides
// hsa_amd_queue_create (TheRock; see docs/P0-BASELINES.md).
#include <hrx_runtime.h>

#include <mutex>
#include <string>
#include <vector>

#include "rt/device.hpp"

namespace halo::rt {

std::unique_ptr<Device> CreateHrxDevice(std::string* error);

namespace {

bool Ok(hrx_status_t s, const char* what, std::string* error) {
  if (hrx_status_is_ok(s)) return true;
  if (error) {
    char* m = nullptr;
    size_t n = 0;
    hrx_status_to_string(s, &m, &n);
    *error = std::string(what) + ": " + (m ? std::string(m, n) : "?");
    if (m) hrx_status_free_message(m);
  }
  hrx_status_ignore(s);
  return false;
}

class HrxDevice final : public Device {
 public:
  ~HrxDevice() override {
    if (stream_) hrx_stream_release(stream_);
    if (exe_) hrx_executable_release(exe_);
  }
  bool Init(std::string* error) {
    // libhrx initializes the GPU driver once per process.
    static std::once_flag once;
    static std::string init_error;
    std::call_once(once, [] { Ok(hrx_gpu_initialize(0), "hrx_gpu_initialize", &init_error); });
    if (!init_error.empty()) {
      if (error) *error = init_error;
      return false;
    }
    if (!Ok(hrx_gpu_device_get(0, &dev_), "hrx_gpu_device_get", error)) return false;
    char arch[64] = {};
    if (!Ok(hrx_device_get_property(dev_, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof(arch)), "arch", error))
      return false;
    arch_ = arch;
    return Ok(hrx_stream_create(dev_, 0, &stream_), "hrx_stream_create", error);
  }

  Backend backend() const noexcept override { return Backend::kHrx; }

  Buffer Alloc(std::size_t bytes, std::string* error) override {
    Buffer b;
    hrx_buffer_t h = nullptr;
    if (!Ok(hrx_buffer_allocate(stream_, bytes ? bytes : 1, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT,
                                &h),
            "hrx_buffer_allocate", error))
      return b;
    void* p = nullptr;
    if (!Ok(hrx_stream_synchronize(stream_), "sync", error) ||
        !Ok(hrx_buffer_get_device_ptr(h, &p), "hrx_buffer_get_device_ptr", error)) {
      hrx_buffer_release(h);
      return b;
    }
    b.device = p;
    b.bytes = bytes;
    b.handle = h;
    return b;
  }
  void Free(Buffer& b) noexcept override {
    if (b.handle) hrx_buffer_release(static_cast<hrx_buffer_t>(b.handle));
    b = {};
  }
  bool Upload(const Buffer& dst, std::size_t off, const void* src, std::size_t n, std::string* error) override {
    if (capturing_) return Capturing(error);
    if (off + n > dst.bytes) return Bounds(error);
    return Ok(hrx_stream_synchronize(stream_), "sync", error) &&
           Ok(hrx_synchronous_h2d(dev_, src, static_cast<hrx_buffer_t>(dst.handle), off, n), "h2d", error);
  }
  bool Download(const Buffer& src, std::size_t off, void* dst, std::size_t n, std::string* error) override {
    if (capturing_) return Capturing(error);
    if (off + n > src.bytes) return Bounds(error);
    return Ok(hrx_stream_synchronize(stream_), "sync", error) &&
           Ok(hrx_synchronous_d2h(dev_, static_cast<hrx_buffer_t>(src.handle), off, dst, n), "d2h", error);
  }
  bool Fill(const Buffer& dst, std::uint8_t v, std::string* error) override {
    const std::uint32_t pattern = v * 0x01010101u;
    if (capturing_) {
      const hrx_graph_fill_buffer_node_attrs_t attrs = {
          {static_cast<hrx_buffer_t>(dst.handle), 0, dst.bytes}, pattern, 4};
      return Chain(
          hrx_graph_add_fill_buffer_node(graph_, has_last_ ? &last_ : nullptr, has_last_ ? 1 : 0, &attrs, &last_),
          "hrx_graph_add_fill_buffer_node", error);
    }
    return Ok(hrx_stream_fill_buffer(stream_, static_cast<hrx_buffer_t>(dst.handle), 0, dst.bytes, &pattern, 4),
              "fill", error);
  }
  bool LoadModule(std::span<const std::uint8_t> hsaco, std::string* error) override {
    return Ok(hrx_executable_load_data(dev_, hsaco.data(), hsaco.size(), "amdgpu", arch_.c_str(), &exe_),
              "hrx_executable_load_data", error);
  }
  KernelId Kernel(std::string_view name, std::string* error) override {
    uint32_t ordinal = 0;
    if (!Ok(hrx_executable_lookup_export_by_name(exe_, std::string(name).c_str(), &ordinal), "lookup", error))
      return -1;
    return static_cast<KernelId>(ordinal);
  }
  bool Launch(KernelId k, Dim3 grid, Dim3 block, std::span<const Arg> args, std::string* error) override {
    std::vector<hrx_buffer_ref_t> bindings;
    std::vector<std::uint8_t> constants;
    for (const auto& a : args) {
      if (a.kind == Arg::Kind::kBuffer) {
        bindings.push_back({static_cast<hrx_buffer_t>(a.buffer.handle), a.offset, a.buffer.bytes - a.offset});
      } else {
        constants.insert(constants.end(), a.bytes, a.bytes + a.size);
      }
    }
    const hrx_dispatch_config_t cfg = {{grid.x, grid.y, grid.z}, {block.x, block.y, block.z}, 0};
    if (capturing_) {
      const hrx_graph_kernel_node_attrs_t attrs = {exe_,           static_cast<uint32_t>(k), cfg,
                                                   constants.data(), constants.size(),         bindings.data(),
                                                   bindings.size(),  HRX_DISPATCH_FLAG_NONE};
      return Chain(hrx_graph_add_kernel_node(graph_, has_last_ ? &last_ : nullptr, has_last_ ? 1 : 0, &attrs, &last_),
                   "hrx_graph_add_kernel_node", error);
    }
    return Ok(hrx_stream_dispatch(stream_, exe_, static_cast<uint32_t>(k), &cfg, constants.data(), constants.size(),
                                  bindings.data(), bindings.size(), HRX_DISPATCH_FLAG_NONE),
              "hrx_stream_dispatch", error);
  }
  bool Synchronize(std::string* error) override {
    if (capturing_) return Capturing(error);
    return Ok(hrx_stream_synchronize(stream_), "sync", error);
  }
  // libhrx v0.1.0 does not implement stream capture, so capture builds the
  // graph explicitly: each launch or fill becomes a node depending on the
  // previous one, the same order the stream would execute.
  bool BeginCapture(std::string* error) override {
    if (!Ok(hrx_graph_create(dev_, 0, &graph_), "hrx_graph_create", error)) return false;
    capturing_ = true;
    has_last_ = false;
    return true;
  }
  bool EndCapture(Graph* out, std::string* error) override {
    capturing_ = false;
    hrx_graph_exec_t exec = nullptr;
    const bool ok = capture_ok_ && Ok(hrx_graph_instantiate(graph_, 0, &exec), "hrx_graph_instantiate", error);
    if (!capture_ok_ && error) *error = "a launch failed during capture";
    hrx_graph_release(graph_);
    graph_ = nullptr;
    capture_ok_ = true;
    if (ok) out->exec = exec;
    return ok;
  }
  bool LaunchGraph(const Graph& graph, std::string* error) override {
    return Ok(hrx_graph_exec_launch(static_cast<hrx_graph_exec_t>(graph.exec), stream_), "hrx_graph_exec_launch",
              error);
  }
  void FreeGraph(Graph& graph) noexcept override {
    if (graph.exec) hrx_graph_exec_release(static_cast<hrx_graph_exec_t>(graph.exec));
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
  bool Chain(hrx_status_t s, const char* what, std::string* error) {
    const bool ok = Ok(s, what, error);
    has_last_ = has_last_ || ok;
    capture_ok_ = capture_ok_ && ok;
    return ok;
  }
  bool capturing_ = false, has_last_ = false, capture_ok_ = true;
  hrx_graph_t graph_ = nullptr;
  hrx_graph_node_t last_{};
  hrx_device_t dev_ = nullptr;
  hrx_stream_t stream_ = nullptr;
  hrx_executable_t exe_ = nullptr;
  std::string arch_;
};

}  // namespace

std::unique_ptr<Device> CreateHrxDevice(std::string* error) {
  auto d = std::make_unique<HrxDevice>();
  if (!d->Init(error)) return nullptr;
  return d;
}

}  // namespace halo::rt
