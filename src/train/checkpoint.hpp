#ifndef HALO_TRAIN_CHECKPOINT_HPP_
#define HALO_TRAIN_CHECKPOINT_HPP_

// Resumable, content-bound training checkpoints (the standing rule for long
// runs): the file holds the step, the FP32 master weights and both AdamW
// moments, and a header binding it to the exact inputs (base model, training
// data, config, trunk cache, parameter layout). Written atomically (temp
// file, fsync, rename, fsync dir) with a trailing SHA-256 over every
// preceding byte; a reader refuses a torn, corrupted or foreign checkpoint.
// Because the data order is a pure function of (seed, step) and the math is
// deterministic, resuming from a checkpoint continues bit for bit.
//
// Layout: "HALOCKP1" | u32 header length | header JSON | u64 n |
//         params[n] f32 | m[n] f32 | v[n] f32 | SHA-256 (32 bytes)

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/json.hpp"

namespace halo::train {

/// What a run's state is bound to; every field must match on resume.
struct RunBinding {
  std::string base_sha256, data_sha256, config_sha256, cache_sha256, layout_sha256;
  [[nodiscard]] json::Value ToJson() const;
  [[nodiscard]] static std::optional<RunBinding> FromJson(const json::Value& v);
  [[nodiscard]] std::string Mismatch(const RunBinding& other) const;  ///< "" when equal
};

struct Checkpoint {
  RunBinding binding;
  std::uint64_t step = 0;
  std::vector<float> params, m, v;
};

/// The checkpoint's SHA-256 trailer is returned in `sha256` when non-null.
[[nodiscard]] bool WriteCheckpoint(const std::string& path, const Checkpoint& c, std::string* error,
                                   std::string* sha256 = nullptr);
[[nodiscard]] std::optional<Checkpoint> ReadCheckpoint(const std::string& path, std::string* error,
                                                       std::string* sha256 = nullptr);

}  // namespace halo::train

#endif  // HALO_TRAIN_CHECKPOINT_HPP_
