// libFuzzer target: the GGUF header parser consumes untrusted bytes.
#include <cstdint>
#include <string>

#include "core/gguf.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  std::string error;
  auto reader = halo::core::GgufReader::OpenMemory(data, size, &error);
  if (reader) {
    volatile std::uint8_t sink = 0;
    for (const auto& t : reader->GetTensors()) {
      if (t.size_bytes) sink ^= static_cast<const std::uint8_t*>(t.data)[t.size_bytes - 1];
    }
    (void)sink;
  }
  return 0;
}
