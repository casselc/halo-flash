#ifndef HALO_CORE_SHA256_HPP_
#define HALO_CORE_SHA256_HPP_

// Streaming SHA-256 (FIPS 180-4) for content hashes: variant manifests,
// training data, trunk caches and checkpoints. Uses the x86 SHA extensions
// when the CPU has them (about 2 GB/s per core) and a portable path
// otherwise; both give identical digests (tests/core/sha256_test.cpp).

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace halo::core {

class Sha256 {
 public:
  using Digest = std::array<std::uint8_t, 32>;
  Sha256() { Reset(); }
  void Reset();
  void Update(const void* data, std::size_t size);
  void Update(std::string_view s) { Update(s.data(), s.size()); }
  [[nodiscard]] Digest Finish();

  /// For tests: forces the portable path even when SHA-NI is available.
  static void ForcePortable(bool on) noexcept;
  [[nodiscard]] static bool HardwareAccelerated() noexcept;

 private:
  void Blocks(const std::uint8_t* p, std::size_t nblocks);
  std::uint32_t h_[8];
  std::uint8_t buf_[64];
  std::size_t buf_len_ = 0;
  std::uint64_t total_ = 0;
};

[[nodiscard]] std::string HexDigest(const Sha256::Digest& d);
[[nodiscard]] std::string Sha256Hex(std::string_view data);
/// SHA-256 of a file's bytes (read in 8 MiB blocks), or nullopt with `error`.
[[nodiscard]] std::optional<std::string> Sha256File(const std::string& path, std::string* error);

}  // namespace halo::core

#endif  // HALO_CORE_SHA256_HPP_
