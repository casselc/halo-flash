#ifndef HALO_CORE_ATOMIC_FILE_HPP_
#define HALO_CORE_ATOMIC_FILE_HPP_

// Crash-safe file writes for checkpoints, manifests and progress logs.
// AtomicWriteFile: write to "<path>.tmp.<pid>", fsync it, rename over
// <path>, fsync the directory. A crash at any point leaves either the old
// file or the new one, never a torn one.
// DurableAppend: append one record and fsync before returning.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace halo::core {

/// Test hook: when set, AtomicWriteFile stops (returns false with an error)
/// after this stage, simulating a crash: 1 = temp written but not synced,
/// 2 = temp synced but not renamed. 0 (default) = normal.
void SetAtomicWriteFaultForTest(int stage) noexcept;

[[nodiscard]] bool AtomicWriteFile(const std::string& path, std::span<const std::uint8_t> bytes,
                                   std::string* error);
[[nodiscard]] inline bool AtomicWriteFile(const std::string& path, std::string_view text, std::string* error) {
  return AtomicWriteFile(path, std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(text.data()),
                                                             text.size()),
                         error);
}
[[nodiscard]] bool DurableAppend(const std::string& path, std::string_view record, std::string* error);
/// Reads a whole file; false with `error` if it cannot.
[[nodiscard]] bool ReadWholeFile(const std::string& path, std::string* out, std::string* error);

}  // namespace halo::core

#endif  // HALO_CORE_ATOMIC_FILE_HPP_
