#ifndef HALO_CORE_GGUF_WRITER_HPP_
#define HALO_CORE_GGUF_WRITER_HPP_

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "core/gguf.hpp"

namespace halo::core {

/// Builds a GGUF v3 image in memory. Used by tools/make_tiny (synthetic
/// models), tools/repack, and the reader round-trip properties. Keys and tensor
/// names must be unique; Serialize() reports the first violation instead of
/// writing a file the reader would reject.
class GgufWriter {
 public:
  using Value =
      std::variant<std::uint32_t, std::int32_t, std::uint64_t, std::int64_t,
                   float, double, bool, std::string, std::vector<std::string>,
                   std::vector<std::uint32_t>, std::vector<std::int32_t>,
                   std::vector<std::int64_t>, std::vector<float>>;

  explicit GgufWriter(std::uint32_t alignment = 32) : alignment_(alignment) {}

  void Set(std::string key, Value value);

  /// `dims` are GGUF order (innermost first). `bytes` must be exactly the
  /// encoded size for `type` and `dims`.
  void AddTensor(std::string name, std::vector<std::uint64_t> dims,
                 GgmlType type, std::span<const std::uint8_t> bytes);

  /// Returns the full file image, or an empty vector with `error` set.
  [[nodiscard]] std::vector<std::uint8_t> Serialize(std::string* error) const;

  /// Writes atomically (temp file + rename).
  [[nodiscard]] bool WriteFile(const std::filesystem::path& path,
                               std::string* error) const;

 private:
  struct Tensor {
    std::string name;
    std::vector<std::uint64_t> dims;
    GgmlType type;
    std::vector<std::uint8_t> bytes;
  };
  std::uint32_t alignment_;
  std::vector<std::pair<std::string, Value>> metadata_;
  std::vector<Tensor> tensors_;
};

}  // namespace halo::core

#endif  // HALO_CORE_GGUF_WRITER_HPP_
