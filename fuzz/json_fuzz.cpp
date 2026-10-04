// libFuzzer target: the JSON parser takes untrusted HTTP bodies. Anything it
// accepts must dump and re-parse to an equal value.
#include <cstdint>
#include <cstdlib>
#include <string>

#include "core/json.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view s(reinterpret_cast<const char*>(data), size);
  std::string error;
  const auto v = halo::json::Parse(s, &error, {.max_depth = 64, .max_bytes = 1 << 20});
  if (!v) return 0;
  const auto text = halo::json::DumpPython(*v, 2);
  const auto back = halo::json::Parse(text, &error);
  if (!back || !(*back == *v)) std::abort();
  return 0;
}
