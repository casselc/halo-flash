// Shared helpers for the P0 runtime microbenchmarks. Each backend binary
// prints one JSON object per measurement so runs can be diffed and tabulated.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace bench {

inline std::vector<char> ReadFile(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", path);
    std::exit(2);
  }
  return {std::istreambuf_iterator<char>(f), {}};
}

inline double NowUs() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

inline double Percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  if (v.empty()) return 0.0;
  return v[std::min(v.size() - 1, static_cast<std::size_t>(p * v.size()))];
}

inline void Emit(const char* backend, const char* metric, double value,
                 const char* unit, const std::string& extra = "") {
  std::printf("{\"backend\":\"%s\",\"metric\":\"%s\",\"value\":%.4f,\"unit\":\"%s\"%s%s}\n",
              backend, metric, value, unit, extra.empty() ? "" : ",", extra.c_str());
  std::fflush(stdout);
}

struct Options {
  const char* hsaco = "kernels.hsaco";
  std::size_t bw_bytes = std::size_t{8} << 30;  // bandwidth buffer
  std::size_t big_gib = 0;                      // large-allocation probe (0 = skip)
  int launches = 20000;
  int roundtrips = 2000;
};

inline Options ParseOptions(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&] { return i + 1 < argc ? argv[++i] : (std::exit(2), nullptr); };
    if (a == "--hsaco") o.hsaco = next();
    else if (a == "--bw-gib") o.bw_bytes = std::size_t(std::atoll(next())) << 30;
    else if (a == "--big-gib") o.big_gib = std::size_t(std::atoll(next()));
    else if (a == "--launches") o.launches = std::atoi(next());
    else if (a == "--roundtrips") o.roundtrips = std::atoi(next());
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); std::exit(2); }
  }
  return o;
}

}  // namespace bench
