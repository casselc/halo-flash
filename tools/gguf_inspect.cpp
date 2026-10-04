// Prints a GGUF file's metadata keys and a per-type tensor summary, and
// dequantizes the first row of every tensor whose type has a CPU dequantizer
// (a cheap end-to-end check of the reader and the quant oracle on real files).
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/quant.hpp"

int main(int argc, char** argv) {
  int status = 0;
  for (int a = 1; a < argc; ++a) {
    std::string error;
    auto r = halo::core::GgufReader::OpenFile(argv[a], &error);
    if (!r) {
      std::printf("%s: FAIL %s\n", argv[a], error.c_str());
      status = 1;
      continue;
    }
    std::map<std::string, std::pair<std::size_t, std::uint64_t>> by_type;
    std::size_t deq_ok = 0, deq_skip = 0, nonfinite = 0;
    for (const auto& t : r->GetTensors()) {
      auto& e = by_type[std::string(halo::core::ToString(t.type))];
      ++e.first;
      e.second += t.size_bytes;
      const std::size_t row = t.dimensions.front();
      std::vector<float> out(row);
      if (halo::quant::DequantizeRow(t.type, t.data, out.data(), row)) {
        ++deq_ok;
        for (float v : out) nonfinite += !std::isfinite(v);
      } else {
        ++deq_skip;
      }
    }
    std::printf("%s: v%u arch=%s kv=%llu tensors=%llu align=%llu\n", argv[a], r->GetVersion(),
                std::string(r->GetMetadataString("general.architecture").value_or("?")).c_str(),
                (unsigned long long)r->GetMetadataCount(), (unsigned long long)r->GetTensorCount(),
                (unsigned long long)r->GetAlignment());
    for (const auto& [type, e] : by_type) {
      std::printf("  %-8s %5zu tensors %10.1f MiB\n", type.c_str(), e.first, e.second / 1048576.0);
    }
    std::printf("  first-row dequant: %zu ok, %zu unsupported, %zu non-finite values\n", deq_ok,
                deq_skip, nonfinite);
  }
  return status;
}
