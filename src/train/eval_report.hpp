#ifndef HALO_TRAIN_EVAL_REPORT_HPP_
#define HALO_TRAIN_EVAL_REPORT_HPP_

// How training runs and variant evaluations report results (P11 "eval hooks
// into P12"). The trainer and halo_train's eval command call an EvalSink;
// this file holds the interface and two sinks that need nothing else:
//   - JsonlEvalSink: one durable JSON line per result (fsync'd);
//   - the variant manifest's eval summary (ToVariantEval).
// halo-scope (stream X) is expected to add a sink that records each report
// as an experiment run item with a score; nothing here depends on it.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "model/variant.hpp"

namespace halo::train {

struct EvalReport {
  std::string run_id;           ///< training run (its directory name), or "" for a standalone eval
  std::string variant;          ///< variant name ("" = the base model)
  std::int64_t variant_version = 0;
  std::string base_sha256;
  std::string dataset;          ///< e.g. "sentiment-heldout"
  std::string dataset_sha256;
  std::string metric;           ///< e.g. "accuracy", "loss", "exact_match"
  double value = 0;
  std::optional<double> baseline;  ///< the base model's value on the same set, when known
  std::uint64_t n = 0;              ///< examples (or tokens, as the metric says)
  std::int64_t step = -1;           ///< training step, -1 for a final / standalone eval
  std::string engine;               ///< "cpu-tail" (trainer) or "hip" (inference engine)
  std::int64_t unix_ms = 0;

  [[nodiscard]] json::Value ToJson() const;
};

class EvalSink {
 public:
  virtual ~EvalSink() = default;
  [[nodiscard]] virtual bool Report(const EvalReport& r, std::string* error) = 0;
};

class JsonlEvalSink : public EvalSink {
 public:
  explicit JsonlEvalSink(std::string path) : path_(std::move(path)) {}
  bool Report(const EvalReport& r, std::string* error) override;

 private:
  std::string path_;
};

class FanOutEvalSink : public EvalSink {
 public:
  void Add(EvalSink* s) { sinks_.push_back(s); }
  bool Report(const EvalReport& r, std::string* error) override;

 private:
  std::vector<EvalSink*> sinks_;
};

[[nodiscard]] model::VariantEval ToVariantEval(const EvalReport& r);

}  // namespace halo::train

#endif  // HALO_TRAIN_EVAL_REPORT_HPP_
