#ifndef HALO_TRAIN_TRAINER_HPP_
#define HALO_TRAIN_TRAINER_HPP_

// The stage-1 training loop: mini-batches of cached trunk boundaries through
// the CPU tail (LastK), AdamW, durable progress and content-bound checkpoints.
//
// Determinism and resume: batch b of step s is a pure function of (seed, s)
// (one Fisher-Yates permutation per epoch), the tail's math is deterministic
// at any thread count, and a checkpoint stores the master weights and both
// moments bit for bit, so stopping at any point and resuming from the last
// checkpoint gives the same weights as an uninterrupted run (tested by
// killing the process; tools/train_resume_test.sh).
//
// Files in the run directory:
//   checkpoint.bin   the latest checkpoint (atomic replace)
//   progress.jsonl   one fsync'd line per step: step, loss, tokens, grad
//                    norm, learning rates, seconds; on resume, lines past the
//                    checkpoint are dropped (atomically) and a resume event
//                    is appended, so the log always matches the weights
//   evals.jsonl      held-out evaluations (JsonlEvalSink)

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "train/adamw.hpp"
#include "train/checkpoint.hpp"
#include "train/data.hpp"
#include "train/eval_report.hpp"
#include "train/last_k.hpp"

namespace halo::train {

struct TrainOptions {
  AdamWConfig opt;
  std::uint64_t steps = 100;
  std::uint32_t batch_size = 8;
  std::uint64_t checkpoint_every = 10;
  std::uint64_t eval_every = 0;  ///< 0: evaluate at the start and the end only
  std::uint64_t seed = 0;
};

struct EvalSet {
  std::string name;
  const Dataset* data = nullptr;
  const TrunkCache* cache = nullptr;
};

struct TailEval {
  double loss = 0;      ///< mean cross-entropy per loss position
  double accuracy = 0;  ///< argmax == target, per loss position
  double exact = 0;     ///< every loss position of the example correct
  std::uint64_t examples = 0, positions = 0;
};
[[nodiscard]] TailEval EvaluateTail(const LastK& model, const EvalSet& set, ThreadPool& pool);

/// SHA-256 of the parameter layout (slice names, offsets, shapes, groups).
[[nodiscard]] std::string LayoutSha256(const LastK& model);

/// Example indices of step `step` (0-based): batch_size draws from the
/// concatenation of per-epoch permutations of [0, n).
[[nodiscard]] std::vector<std::size_t> BatchIndices(std::uint64_t seed, std::size_t n, std::uint64_t step,
                                                    std::uint32_t batch_size);

class Trainer {
 public:
  Trainer(LastK& model, TrainOptions options, const Dataset& train, const TrunkCache& cache, RunBinding binding,
          std::string run_dir, ThreadPool& pool);
  /// Held-out evaluation with the CPU tail, reported to `sink` (base fields
  /// of each report from `report_template`).
  void SetEval(EvalSet set, EvalSink* sink, EvalReport report_template);

  struct Status {
    bool done = false;  ///< all steps trained
    std::uint64_t step = 0;
    double seconds = 0;
    std::string checkpoint_sha256;
    double last_loss = 0;
  };
  /// Resumes from the run directory's checkpoint if there is one (refusing
  /// a checkpoint bound to other inputs), then trains until all steps are
  /// done or `max_seconds` has passed, checkpointing every
  /// checkpoint_every steps and at the end of the chunk.
  [[nodiscard]] std::optional<Status> Run(double max_seconds, std::string* error);

 private:
  bool Checkpoint(std::uint64_t step, std::string* sha, std::string* error);
  bool Evaluate(std::uint64_t step, std::string* error);

  LastK& model_;
  TrainOptions o_;
  const Dataset& train_;
  const TrunkCache& cache_;
  RunBinding binding_;
  std::string dir_;
  ThreadPool& pool_;
  AdamWState opt_state_;
  EvalSet eval_;
  EvalSink* sink_ = nullptr;
  EvalReport tmpl_;
};

}  // namespace halo::train

#endif  // HALO_TRAIN_TRAINER_HPP_
