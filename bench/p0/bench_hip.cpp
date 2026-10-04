// P0 microbenchmarks through the HIP module API. Run unchanged against ROCm's
// libamdhip64 and, via LD_PRELOAD, against HRX's HIP compatibility layer.
#include <hip/hip_runtime.h>

#include <cstring>
#include <string>
#include <vector>

#include "bench_common.hpp"

#define CHECK(x)                                                              \
  do {                                                                        \
    hipError_t e_ = (x);                                                      \
    if (e_ != hipSuccess) {                                                   \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,          \
                   hipGetErrorString(e_));                                    \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

const char* kBackend = "hip";
hipFunction_t g_noop, g_read, g_write, g_store;
int g_cus = 0;

void Launch(hipFunction_t f, unsigned grid, unsigned block, void** args,
            hipStream_t s) {
  CHECK(hipModuleLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, s, args, nullptr));
}

void Noop(hipStream_t s) { Launch(g_noop, 1, 64, nullptr, s); }

void LaunchLatency(const bench::Options& o, hipStream_t s) {
  for (int i = 0; i < 1000; ++i) Noop(s);
  CHECK(hipStreamSynchronize(s));
  // Submission throughput: back-to-back launches, one sync at the end.
  double t0 = bench::NowUs();
  for (int i = 0; i < o.launches; ++i) Noop(s);
  CHECK(hipStreamSynchronize(s));
  bench::Emit(kBackend, "noop_async_per_launch", (bench::NowUs() - t0) * 1000.0 / o.launches, "ns");
  // Round trip: launch then wait, as a decode step's host sync would.
  std::vector<double> rt;
  for (int i = 0; i < o.roundtrips; ++i) {
    double a = bench::NowUs();
    Noop(s);
    CHECK(hipStreamSynchronize(s));
    rt.push_back(bench::NowUs() - a);
  }
  bench::Emit(kBackend, "noop_sync_roundtrip_p50", bench::Median(rt), "us");
  bench::Emit(kBackend, "noop_sync_roundtrip_p99", bench::Percentile(rt, 0.99), "us");
}

void GraphReplay(hipStream_t s) {
  for (int nodes : {1, 100, 1000}) {
    hipGraph_t g;
    hipGraphExec_t ge;
    CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    for (int i = 0; i < nodes; ++i) Noop(s);
    CHECK(hipStreamEndCapture(s, &g));
    CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    for (int i = 0; i < 5; ++i) CHECK(hipGraphLaunch(ge, s));
    CHECK(hipStreamSynchronize(s));
    std::vector<double> t;
    for (int i = 0; i < 50; ++i) {
      double a = bench::NowUs();
      CHECK(hipGraphLaunch(ge, s));
      CHECK(hipStreamSynchronize(s));
      t.push_back(bench::NowUs() - a);
    }
    const double p50 = bench::Median(t);
    bench::Emit(kBackend, "graph_launch_sync_p50", p50, "us", "\"nodes\":" + std::to_string(nodes));
    bench::Emit(kBackend, "graph_per_node", p50 * 1000.0 / nodes, "ns", "\"nodes\":" + std::to_string(nodes));
    CHECK(hipGraphExecDestroy(ge));
    CHECK(hipGraphDestroy(g));
  }
  // Independent nodes: no inter-kernel barrier, isolates per-dispatch cost.
  // HALO_BENCH_SKIP_INDEPENDENT=1 skips it (HRX's HIP shim rejects these
  // graphs at launch as of hrx-system 1618350).
  if (std::getenv("HALO_BENCH_SKIP_INDEPENDENT")) return;
  for (int nodes : {100, 1000}) {
    hipGraph_t g;
    CHECK(hipGraphCreate(&g, 0));
    hipKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(g_noop);
    p.gridDim = dim3(1);
    p.blockDim = dim3(64);
    for (int i = 0; i < nodes; ++i) {
      hipGraphNode_t n;
      CHECK(hipGraphAddKernelNode(&n, g, nullptr, 0, &p));
    }
    hipGraphExec_t ge;
    CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    for (int i = 0; i < 5; ++i) CHECK(hipGraphLaunch(ge, s));
    CHECK(hipStreamSynchronize(s));
    std::vector<double> t;
    for (int i = 0; i < 50; ++i) {
      double a = bench::NowUs();
      CHECK(hipGraphLaunch(ge, s));
      CHECK(hipStreamSynchronize(s));
      t.push_back(bench::NowUs() - a);
    }
    bench::Emit(kBackend, "graph_independent_per_node", bench::Median(t) * 1000.0 / nodes, "ns",
                "\"nodes\":" + std::to_string(nodes));
    CHECK(hipGraphExecDestroy(ge));
    CHECK(hipGraphDestroy(g));
  }
}

// Returns best GB/s over `iters` runs of `f` at each grid size.
void Bandwidth(const char* name, hipFunction_t f, void* buf, std::size_t bytes,
               hipStream_t s) {
  std::uint64_t n4 = bytes / 16;
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  std::vector<char> partial_host(1);
  void* partial = nullptr;
  CHECK(hipMalloc(&partial, 1 << 20));
  for (int per_cu : {4, 8, 16, 32}) {
    unsigned grid = g_cus * per_cu;
    void* read_args[] = {&buf, &partial, &n4};
    void* write_args[] = {&buf, &n4};
    void** args = f == g_read ? read_args : write_args;
    Launch(f, grid, 256, args, s);
    CHECK(hipStreamSynchronize(s));
    double best = 0;
    for (int i = 0; i < 8; ++i) {
      CHECK(hipEventRecord(e0, s));
      Launch(f, grid, 256, args, s);
      CHECK(hipEventRecord(e1, s));
      CHECK(hipEventSynchronize(e1));
      float ms = 0;
      CHECK(hipEventElapsedTime(&ms, e0, e1));
      best = std::max(best, bytes / (ms * 1e6));
    }
    bench::Emit(kBackend, name, best, "GB/s",
                "\"bytes\":" + std::to_string(bytes) + ",\"workgroups\":" + std::to_string(grid));
  }
  CHECK(hipFree(partial));
  CHECK(hipEventDestroy(e0));
  CHECK(hipEventDestroy(e1));
}

}  // namespace

int main(int argc, char** argv) {
  const auto o = bench::ParseOptions(argc, argv);
  if (const char* b = std::getenv("HALO_BENCH_BACKEND")) kBackend = b;
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  g_cus = prop.multiProcessorCount;
  std::fprintf(stderr, "device: %s %s, %d CUs\n", prop.name, prop.gcnArchName, g_cus);

  const auto image = bench::ReadFile(o.hsaco);
  hipModule_t mod;
  CHECK(hipModuleLoadData(&mod, image.data()));
  CHECK(hipModuleGetFunction(&g_noop, mod, "halo_noop"));
  CHECK(hipModuleGetFunction(&g_read, mod, "halo_read_bw"));
  CHECK(hipModuleGetFunction(&g_write, mod, "halo_write_bw"));
  CHECK(hipModuleGetFunction(&g_store, mod, "halo_store"));

  hipStream_t s;
  CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  // Argument-passing check before trusting any timing.
  unsigned* out = nullptr;
  CHECK(hipMalloc(&out, 4));
  unsigned value = 0xC0FFEE11u, got = 0;
  void* store_args[] = {&out, &value};
  Launch(g_store, 1, 64, store_args, s);
  CHECK(hipStreamSynchronize(s));
  CHECK(hipMemcpy(&got, out, 4, hipMemcpyDeviceToHost));
  if (got != value) {
    std::fprintf(stderr, "store check failed: %x\n", got);
    return 1;
  }
  CHECK(hipFree(out));

  LaunchLatency(o, s);
  GraphReplay(s);

  void* buf = nullptr;
  CHECK(hipMalloc(&buf, o.bw_bytes));
  Bandwidth("write_bw", g_write, buf, o.bw_bytes, s);
  Bandwidth("read_bw", g_read, buf, o.bw_bytes, s);
  CHECK(hipFree(buf));

  if (o.big_gib) {
    void* big = nullptr;
    const std::size_t bytes = o.big_gib << 30;
    double a = bench::NowUs();
    hipError_t e = hipMalloc(&big, bytes);
    bench::Emit(kBackend, "big_alloc_ok", e == hipSuccess, "bool",
                "\"gib\":" + std::to_string(o.big_gib) + ",\"alloc_ms\":" +
                    std::to_string((bench::NowUs() - a) / 1000.0));
    if (e == hipSuccess) {
      Bandwidth("big_write_bw", g_write, big, bytes, s);
      Bandwidth("big_read_bw", g_read, big, bytes, s);
      CHECK(hipFree(big));
    }
  }
  return 0;
}
