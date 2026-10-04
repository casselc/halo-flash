// Graph capture, instantiation and replay cost against node count, on every
// backend: N dependent halo_axpy launches recorded with rt::Device capture.
#include <chrono>
#include <cstdio>
#include "kernels/core_hsaco.hpp"
#include "rt/device.hpp"
using namespace halo;
int main() {
  std::string e;
  for (auto be : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto d = rt::Device::Create(be, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) continue;
    auto x = d->Alloc(4096 * 4, &e), y = d->Alloc(4096 * 4, &e);
    const auto k = d->Kernel("halo_axpy", &e);
    const rt::Arg args[] = {rt::Arg::Buf(y), rt::Arg::Buf(x), rt::Arg::Val(1.0f), rt::Arg::Val(4096u)};
    for (int n : {500, 1000, 2000, 4000, 8000}) {
      using clk = std::chrono::steady_clock;
      const auto t0 = clk::now();
      if (!d->BeginCapture(&e)) return std::printf("capture: %s\n", e.c_str()), 1;
      for (int i = 0; i < n; ++i) (void)d->Launch(k, {16}, {256}, args, &e);
      rt::Graph g;
      const auto t1 = clk::now();
      if (!d->EndCapture(&g, &e)) return std::printf("end: %s\n", e.c_str()), 1;
      const auto t2 = clk::now();
      (void)d->LaunchGraph(g, &e);
      (void)d->Synchronize(&e);
      const auto t3 = clk::now();
      for (int r = 0; r < 5; ++r) (void)d->LaunchGraph(g, &e);
      (void)d->Synchronize(&e);
      const auto t4 = clk::now();
      const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
      std::printf("%s n=%5d  record %8.1f ms  end+instantiate %9.1f ms  first replay %7.2f ms  replay %7.2f ms (%.2f us/node)\n",
                  rt::BackendName(be), n, ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4) / 5, ms(t3, t4) / 5 * 1000 / n);
      d->FreeGraph(g);
    }
    d->Free(x);
    d->Free(y);
  }
}
