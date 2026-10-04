// P0 microbenchmarks through the native libhrx C API, on the same HSACO the
// HIP benchmark loads. Pointer kernel arguments are buffer bindings, by-value
// arguments are packed constants.
#include <hrx_runtime.h>

#include <cstring>
#include <string>
#include <vector>

#include "bench_common.hpp"

#define CHECK(x)                                                              \
  do {                                                                        \
    hrx_status_t s_ = (x);                                                    \
    if (!hrx_status_is_ok(s_)) {                                              \
      char* m_ = nullptr;                                                     \
      size_t n_ = 0;                                                          \
      hrx_status_to_string(s_, &m_, &n_);                                     \
      std::fprintf(stderr, "%s:%d %s: %.*s\n", __FILE__, __LINE__, #x,        \
                   (int)n_, m_ ? m_ : "?");                                   \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

const char* kBackend = "hrx-native";
hrx_device_t g_dev;
hrx_executable_t g_exe;
uint32_t g_noop, g_read, g_write, g_store;
uint32_t g_cus = 0;

hrx_dispatch_config_t Config(uint32_t grid, uint32_t block) {
  return {{grid, 1, 1}, {block, 1, 1}, 0};
}

void Noop(hrx_stream_t s, uint32_t flags) {
  const auto cfg = Config(1, 64);
  CHECK(hrx_stream_dispatch(s, g_exe, g_noop, &cfg, nullptr, 0, nullptr, 0, flags));
}

void LaunchLatency(const bench::Options& o, hrx_stream_t s) {
  for (uint32_t flags : {uint32_t{HRX_DISPATCH_FLAG_NONE},
                         uint32_t{HRX_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION}}) {
    const std::string tag = std::string("\"flags\":") + std::to_string(flags);
    for (int i = 0; i < 1000; ++i) Noop(s, flags);
    CHECK(hrx_stream_synchronize(s));
    double t0 = bench::NowUs();
    for (int i = 0; i < o.launches; ++i) Noop(s, flags);
    CHECK(hrx_stream_synchronize(s));
    bench::Emit(kBackend, "noop_async_per_launch", (bench::NowUs() - t0) * 1000.0 / o.launches, "ns", tag);
    std::vector<double> rt;
    for (int i = 0; i < o.roundtrips; ++i) {
      double a = bench::NowUs();
      Noop(s, flags);
      CHECK(hrx_stream_synchronize(s));
      rt.push_back(bench::NowUs() - a);
    }
    bench::Emit(kBackend, "noop_sync_roundtrip_p50", bench::Median(rt), "us", tag);
    bench::Emit(kBackend, "noop_sync_roundtrip_p99", bench::Percentile(rt, 0.99), "us", tag);
  }
}

void GraphReplay(hrx_stream_t s) {
  for (int nodes : {1, 100, 1000}) {
    hrx_graph_t g;
    CHECK(hrx_graph_create(g_dev, 0, &g));
    hrx_graph_kernel_node_attrs_t attrs{};
    attrs.executable = g_exe;
    attrs.export_ordinal = g_noop;
    attrs.config = Config(1, 64);
    hrx_graph_node_t prev = nullptr;
    for (int i = 0; i < nodes; ++i) {
      hrx_graph_node_t node;
      CHECK(hrx_graph_add_kernel_node(g, prev ? &prev : nullptr, prev ? 1 : 0, &attrs, &node));
      prev = node;
    }
    hrx_graph_exec_t ge;
    CHECK(hrx_graph_instantiate(g, 0, &ge));
    for (int i = 0; i < 5; ++i) CHECK(hrx_graph_exec_launch(ge, s));
    CHECK(hrx_stream_synchronize(s));
    std::vector<double> t;
    for (int i = 0; i < 50; ++i) {
      double a = bench::NowUs();
      CHECK(hrx_graph_exec_launch(ge, s));
      CHECK(hrx_stream_synchronize(s));
      t.push_back(bench::NowUs() - a);
    }
    const double p50 = bench::Median(t);
    bench::Emit(kBackend, "graph_launch_sync_p50", p50, "us", "\"nodes\":" + std::to_string(nodes));
    bench::Emit(kBackend, "graph_per_node", p50 * 1000.0 / nodes, "ns", "\"nodes\":" + std::to_string(nodes));
    hrx_graph_exec_release(ge);
    hrx_graph_release(g);
  }
  // Independent nodes: no inter-kernel barrier, isolates per-dispatch cost.
  for (int nodes : {100, 1000}) {
    hrx_graph_t g;
    CHECK(hrx_graph_create(g_dev, 0, &g));
    hrx_graph_kernel_node_attrs_t attrs{};
    attrs.executable = g_exe;
    attrs.export_ordinal = g_noop;
    attrs.config = Config(1, 64);
    for (int i = 0; i < nodes; ++i) {
      hrx_graph_node_t node;
      CHECK(hrx_graph_add_kernel_node(g, nullptr, 0, &attrs, &node));
    }
    hrx_graph_exec_t ge;
    CHECK(hrx_graph_instantiate(g, 0, &ge));
    for (int i = 0; i < 5; ++i) CHECK(hrx_graph_exec_launch(ge, s));
    CHECK(hrx_stream_synchronize(s));
    std::vector<double> t;
    for (int i = 0; i < 50; ++i) {
      double a = bench::NowUs();
      CHECK(hrx_graph_exec_launch(ge, s));
      CHECK(hrx_stream_synchronize(s));
      t.push_back(bench::NowUs() - a);
    }
    bench::Emit(kBackend, "graph_independent_per_node", bench::Median(t) * 1000.0 / nodes, "ns",
                "\"nodes\":" + std::to_string(nodes));
    hrx_graph_exec_release(ge);
    hrx_graph_release(g);
  }
}

void Bandwidth(const char* name, uint32_t ordinal, hrx_buffer_t buf, std::size_t bytes,
               hrx_buffer_t partial, hrx_stream_t s) {
  const uint64_t n4 = bytes / 16;
  hrx_event_t e0, e1;
  CHECK(hrx_event_create(g_dev, HRX_EVENT_FLAG_NONE, &e0));
  CHECK(hrx_event_create(g_dev, HRX_EVENT_FLAG_NONE, &e1));
  const bool is_read = ordinal == g_read;
  hrx_buffer_ref_t bindings[2] = {{buf, 0, bytes}, {partial, 0, 1 << 20}};
  for (uint32_t per_cu : {4u, 8u, 16u, 32u}) {
    const auto cfg = Config(g_cus * per_cu, 256);
    auto run = [&] {
      CHECK(hrx_stream_dispatch(s, g_exe, ordinal, &cfg, &n4, sizeof(n4), bindings,
                                is_read ? 2 : 1, HRX_DISPATCH_FLAG_NONE));
    };
    run();
    CHECK(hrx_stream_synchronize(s));
    double best = 0, best_wall = 0;
    for (int i = 0; i < 8; ++i) {
      double a = bench::NowUs();
      CHECK(hrx_event_record(e0, s));
      run();
      CHECK(hrx_event_record(e1, s));
      CHECK(hrx_event_synchronize(e1));
      const double wall_us = bench::NowUs() - a;
      float ms = 0;
      CHECK(hrx_event_elapsed_time(e0, e1, &ms));
      best = std::max(best, bytes / (ms * 1e6));
      best_wall = std::max(best_wall, bytes / (wall_us * 1e3));
    }
    bench::Emit(kBackend, name, best, "GB/s",
                "\"bytes\":" + std::to_string(bytes) + ",\"workgroups\":" +
                    std::to_string(g_cus * per_cu) + ",\"wall_gbps\":" + std::to_string(best_wall));
  }
  hrx_event_release(e0);
  hrx_event_release(e1);
}

hrx_buffer_t Alloc(hrx_stream_t s, std::size_t bytes) {
  hrx_buffer_t b = nullptr;
  CHECK(hrx_buffer_allocate(s, bytes, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &b));
  return b;
}

}  // namespace

int main(int argc, char** argv) {
  const auto o = bench::ParseOptions(argc, argv);
  CHECK(hrx_gpu_initialize(0));
  CHECK(hrx_gpu_device_get(0, &g_dev));
  char arch[64] = {}, name[128] = {};
  CHECK(hrx_device_get_property(g_dev, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof(arch)));
  CHECK(hrx_device_get_property(g_dev, HRX_DEVICE_PROPERTY_NAME, name, sizeof(name)));
  CHECK(hrx_device_get_property(g_dev, HRX_DEVICE_PROPERTY_COMPUTE_UNITS, &g_cus, sizeof(g_cus)));
  // HRX v0.1.0 reports 0 here on gfx1151; use the 20 units HIP reports
  // (multiProcessorCount) so both backends launch identical grids.
  if (g_cus == 0) g_cus = 20;
  std::fprintf(stderr, "device: %s %s, %u CUs (grid unit)\n", name, arch, g_cus);

  const auto image = bench::ReadFile(o.hsaco);
  CHECK(hrx_executable_load_data(g_dev, image.data(), image.size(), "amdgpu", arch, &g_exe));
  CHECK(hrx_executable_lookup_export_by_name(g_exe, "halo_noop", &g_noop));
  CHECK(hrx_executable_lookup_export_by_name(g_exe, "halo_read_bw", &g_read));
  CHECK(hrx_executable_lookup_export_by_name(g_exe, "halo_write_bw", &g_write));
  CHECK(hrx_executable_lookup_export_by_name(g_exe, "halo_store", &g_store));
  for (uint32_t ord : {g_read, g_store}) {
    hrx_executable_export_info_t info{};
    CHECK(hrx_executable_export_info(g_exe, ord, &info));
    std::fprintf(stderr, "export %s: constants=%u bindings=%u\n", info.name,
                 (unsigned)info.constant_byte_length, (unsigned)info.binding_count);
  }

  hrx_stream_t s;
  CHECK(hrx_stream_create(g_dev, 0, &s));

  // Argument-passing check before trusting any timing.
  hrx_buffer_t out = Alloc(s, 4);
  const uint32_t value = 0xC0FFEE11u;
  uint32_t got = 0;
  hrx_buffer_ref_t store_binding = {out, 0, 4};
  const auto one = Config(1, 64);
  CHECK(hrx_stream_dispatch(s, g_exe, g_store, &one, &value, sizeof(value), &store_binding, 1,
                            HRX_DISPATCH_FLAG_NONE));
  CHECK(hrx_stream_synchronize(s));
  CHECK(hrx_synchronous_d2h(g_dev, out, 0, &got, sizeof(got)));
  if (got != value) {
    std::fprintf(stderr, "store check failed: %x\n", got);
    return 1;
  }
  hrx_buffer_release(out);

  LaunchLatency(o, s);
  GraphReplay(s);

  hrx_buffer_t partial = Alloc(s, 1 << 20);
  hrx_buffer_t buf = Alloc(s, o.bw_bytes);
  CHECK(hrx_stream_synchronize(s));
  Bandwidth("write_bw", g_write, buf, o.bw_bytes, partial, s);
  Bandwidth("read_bw", g_read, buf, o.bw_bytes, partial, s);
  hrx_buffer_release(buf);

  if (o.big_gib) {
    const std::size_t bytes = o.big_gib << 30;
    hrx_buffer_t big = nullptr;
    double a = bench::NowUs();
    hrx_status_t st = hrx_buffer_allocate(s, bytes, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                          HRX_BUFFER_USAGE_DEFAULT, &big);
    if (hrx_status_is_ok(st)) st = hrx_stream_synchronize(s);
    const bool ok = hrx_status_is_ok(st);
    if (!ok) hrx_status_ignore(st);
    bench::Emit(kBackend, "big_alloc_ok", ok, "bool",
                "\"gib\":" + std::to_string(o.big_gib) + ",\"alloc_ms\":" +
                    std::to_string((bench::NowUs() - a) / 1000.0));
    if (ok) {
      Bandwidth("big_write_bw", g_write, big, bytes, partial, s);
      Bandwidth("big_read_bw", g_read, big, bytes, partial, s);
      hrx_buffer_release(big);
    }
  }
  hrx_buffer_release(partial);
  hrx_stream_release(s);
  hrx_executable_release(g_exe);
  return 0;
}
