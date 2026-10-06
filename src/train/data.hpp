#ifndef HALO_TRAIN_DATA_HPP_
#define HALO_TRAIN_DATA_HPP_

// Training data for stage 1, and the frozen trunk's cache.
//
// JSONL rows, one example each:
//   LM head:          {"prompt": "...", "completion": "..."}
//     prompt and completion are tokenized separately and concatenated; the
//     loss covers every completion token (teacher forcing).
//   classifier head:  {"text": "...", "label": 2}
//     the label is read from the final hidden state of the last token.
//
// The trunk cache holds, per example, the residual stream entering layer
// L-k for every position, computed once by the inference engine (the trunk
// is frozen, so it never changes during training). It is written in shards
// of kShardExamples, each atomically with a SHA-256 trailer and bound to a
// key (base model, data, tokenizer, layer, engine), so building it resumes
// after a crash or a lock hand-off and a stale cache is never read.

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "text/tokenizer.hpp"
#include "train/last_k.hpp"

namespace halo::train {

struct TextExample {
  std::vector<std::int32_t> tokens;
  std::vector<std::pair<std::uint32_t, std::int32_t>> lm_targets;
  std::int32_t label = -1;
  std::uint32_t label_position = 0;
  [[nodiscard]] std::uint32_t LossCount() const {
    return label >= 0 ? 1 : static_cast<std::uint32_t>(lm_targets.size());
  }
};

struct Dataset {
  std::vector<TextExample> examples;
  std::string sha256;  ///< of the JSONL file's bytes
};

/// Reads and tokenizes a JSONL file. Examples longer than `max_tokens` are
/// an error (never silently truncated).
[[nodiscard]] std::optional<Dataset> LoadJsonl(const std::string& path, const text::Tokenizer& tok, HeadKind head,
                                               std::uint32_t num_classes, std::uint32_t max_tokens,
                                               std::string* error);

/// Computes the residual entering the cached layer for `tokens`
/// ([tokens.size()][hidden]); the engine is fresh for each call.
using TrunkFn = std::function<bool(std::span<const std::int32_t> tokens, std::span<float> out, std::string* error)>;

class TrunkCache {
 public:
  static constexpr std::size_t kShardExamples = 64;
  struct Key {
    std::string base_sha256, data_sha256, tokenizer_sha256, engine;
    std::uint32_t layer = 0, hidden = 0;
    [[nodiscard]] std::string Sha256() const;
  };
  struct BuildStatus {
    bool complete = false;
    std::size_t shards_done = 0, shards_total = 0, shards_built_now = 0;
    double seconds = 0;
    std::uint64_t tokens_built_now = 0;
  };
  /// Builds missing shards in `dir` until done or `max_seconds` has passed
  /// (checked between shards); existing shards are verified, not rebuilt.
  [[nodiscard]] static std::optional<BuildStatus> Build(const std::string& dir, const Key& key,
                                                        std::span<const TextExample> examples, const TrunkFn& trunk,
                                                        double max_seconds, std::string* error);
  /// Loads a complete cache (every shard verified against `key`).
  [[nodiscard]] static std::optional<TrunkCache> Load(const std::string& dir, const Key& key,
                                                      std::span<const TextExample> examples, std::string* error);

  [[nodiscard]] std::span<const float> Boundary(std::size_t i) const {
    return std::span<const float>(data_).subspan(offsets_[i], offsets_[i + 1] - offsets_[i]);
  }
  [[nodiscard]] const std::string& sha256() const noexcept { return sha256_; }  ///< over the shard trailers

 private:
  std::vector<float> data_;
  std::vector<std::size_t> offsets_;
  std::string sha256_;
};

}  // namespace halo::train

#endif  // HALO_TRAIN_DATA_HPP_
