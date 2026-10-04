#ifndef HALO_SCOPE_ANNOTATE_HPP_
#define HALO_SCOPE_ANNOTATE_HPP_

// The harness annotation API (PLAN.md P12): an embedding harness attaches
// its own steps, tool results, scores and trace context to engine-owned
// traces through a few typed calls. Not a generic tracing API: every call
// becomes one typed row (llm_observations, llm_scores or llm_traces) through
// the same rings and writer as the engine's own records, and output content
// follows the same capture policy.
//
// Thread-safe: any thread may call any method; a step may end on a different
// thread from the one that began it. Each calling thread gets its own ring
// on first use, kept for the recorder's life, so harnesses should annotate
// from a bounded set of threads. Records never block: a full ring drops the
// record, counts it, and the call returns kDropped.

#include <cstdint>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>

#include "scope/recorder.hpp"

namespace halo::scope {

/// What output content is kept: nothing; its BLAKE3-128 hash and size
/// (default); or the content itself as a deduplicated payload row.
enum class Capture : std::uint8_t { kOff, kDigest, kFull };

enum class StepKind : std::uint8_t { kTool, kRetry, kPlan, kMemory, kCustom };
const char* StepKindName(StepKind k);

enum class Status : std::uint8_t { kOk, kError };

enum class Outcome : std::uint8_t {
  kOk,
  kDropped,  // a ring was full; counted in the recorder's stats
  kInvalid,  // bad argument (empty id, unknown step); nothing recorded
};

/// What a score is about: a trace, optionally one observation in it, or a
/// whole session (trace empty).
struct ScoreTarget {
  std::string trace_id;
  std::string observation_id;
  std::string session_id;
};

class Annotator {
 public:
  using Clock = std::function<std::int64_t()>;  // ns since the Unix epoch
  /// `clock` defaults to the system clock; tests pass a fake.
  Annotator(Recorder& rec, std::string project, Capture capture, Clock clock = {});

  /// Opens a step (kind tool, retry, plan, memory or custom) under `parent`
  /// (a span id, or empty for the trace root). Writes `*step_id`, a fresh
  /// 16-hex-digit span id. Nothing is recorded until StepEnd.
  Outcome StepBegin(std::string_view trace_id, std::string_view parent, StepKind kind, std::string_view name,
                    std::string* step_id);
  /// Closes an open step: one SPAN observation from begin to now.
  Outcome StepEnd(std::string_view step_id, Status status, std::string_view output);

  /// A tool execution's result for the call the engine parsed as
  /// `tool_call_id`: a TOOL observation ending now, starting `latency_ns`
  /// earlier, linked to its generation through tool_call_ids.
  Outcome ToolResult(std::string_view trace_id, std::string_view tool_call_id, std::string_view tool_name,
                     Status status, std::string_view output, std::uint64_t latency_ns);

  /// Score facts. `*score_id` receives the new fact's id. `supersedes`
  /// (optional) names a fact this one corrects.
  Outcome ScoreNumeric(const ScoreTarget& target, std::string_view name, double value, std::string_view source,
                       std::string_view comment, std::string_view supersedes, std::string* score_id);
  Outcome ScoreCategorical(const ScoreTarget& target, std::string_view name, std::string_view value,
                           std::string_view source, std::string_view comment, std::string_view supersedes,
                           std::string* score_id);
  Outcome ScoreBoolean(const ScoreTarget& target, std::string_view name, bool value, std::string_view source,
                       std::string_view comment, std::string_view supersedes, std::string* score_id);
  /// Withdraws a score fact.
  Outcome Retract(std::string_view score_id, std::string_view source, std::string_view comment);

  /// Trace context. Keys session_id, user_id, release, name and tag set those
  /// fields (tag appends one tag); any other key is a metadata entry.
  Outcome TraceMeta(std::string_view trace_id, std::string_view key, std::string_view value);

  /// An engine-side observation (a generation) with its input and output
  /// content under this annotator's capture policy; fills project and the
  /// content references.
  Outcome RecordObservation(Observation o, std::string_view input, std::string_view output);

  /// Steps begun and not yet ended.
  std::size_t OpenSteps() const;

 private:
  struct OpenStep {
    std::string trace_id, parent, name;
    StepKind kind;
    std::int64_t start_ns;
  };
  Producer* ThisThread();
  std::string NewId();
  std::int64_t Now() const;
  /// Applies the capture policy to `output`; returns false when a payload row was dropped.
  bool CaptureOutput(Producer* p, std::string_view output, Observation& o);
  /// The same for input content; false when a payload row was dropped.
  bool CaptureContent(Producer* p, std::string_view content, std::vector<rb::Hash16>& refs, std::uint64_t& bytes);
  Outcome Score(const ScoreTarget& target, std::string_view name, std::string_view data_type, double value,
                std::string_view string_value, std::string_view source, std::string_view comment,
                std::string_view supersedes, std::string* score_id);

  Recorder& rec_;
  const std::string project_;
  const Capture capture_;
  const Clock clock_;
  mutable std::mutex mu_;
  const std::uint64_t serial_;  // distinguishes annotators in thread-local caches
  std::unordered_map<std::string, OpenStep> open_;             // guarded by mu_
  std::mt19937_64 rng_;                                         // guarded by mu_
};

}  // namespace halo::scope

#endif  // HALO_SCOPE_ANNOTATE_HPP_
