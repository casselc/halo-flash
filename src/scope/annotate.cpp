#include "scope/annotate.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <vector>

namespace halo::scope {

namespace {

std::atomic<std::uint64_t> g_next_serial{1};

std::string_view StatusName(Status s) { return s == Status::kOk ? "OK" : "ERROR"; }

std::string Hex16(std::uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

}  // namespace

const char* StepKindName(StepKind k) {
  switch (k) {
    case StepKind::kTool: return "tool";
    case StepKind::kRetry: return "retry";
    case StepKind::kPlan: return "plan";
    case StepKind::kMemory: return "memory";
    case StepKind::kCustom: return "custom";
  }
  return "custom";
}

Annotator::Annotator(Recorder& rec, std::string project, Capture capture, Clock clock)
    : rec_(rec),
      project_(std::move(project)),
      capture_(capture),
      clock_(std::move(clock)),
      serial_(g_next_serial.fetch_add(1)),
      rng_(std::random_device{}()) {}

Producer* Annotator::ThisThread() {
  // Serial-keyed, so an annotator allocated where a dead one was never
  // inherits its producers.
  thread_local std::vector<std::pair<std::uint64_t, Producer*>> mine;
  for (const auto& [serial, p] : mine)
    if (serial == serial_) return p;
  Producer* p = rec_.NewProducer();
  mine.emplace_back(serial_, p);
  return p;
}

std::string Annotator::NewId() {
  std::lock_guard<std::mutex> lk(mu_);
  return Hex16(rng_());
}

std::int64_t Annotator::Now() const {
  if (clock_) return clock_();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

bool Annotator::CaptureContent(Producer* p, std::string_view content, std::vector<rb::Hash16>& refs, std::uint64_t& bytes) {
  if (capture_ == Capture::kOff || content.empty()) return true;
  const auto h = PayloadHash(content);
  refs = {h};
  bytes = content.size();
  if (capture_ != Capture::kFull) return true;
  Payload pl;
  pl.project = project_;
  pl.hash = h;
  pl.bytes = static_cast<std::uint32_t>(content.size());
  pl.content = std::string(content);
  return p->Record(pl);
}

bool Annotator::CaptureOutput(Producer* p, std::string_view output, Observation& o) {
  return CaptureContent(p, output, o.output_refs, o.output_bytes);
}

Outcome Annotator::RecordObservation(Observation o, std::string_view input, std::string_view output) {
  if (o.trace_id.empty() || o.span_id.empty()) return Outcome::kInvalid;
  o.project = project_;
  Producer* p = ThisThread();
  const bool in_ok = CaptureContent(p, input, o.input_refs, o.input_bytes);
  const bool out_ok = CaptureContent(p, output, o.output_refs, o.output_bytes);
  const bool row_ok = p->Record(o);
  return in_ok && out_ok && row_ok ? Outcome::kOk : Outcome::kDropped;
}

Outcome Annotator::StepBegin(std::string_view trace_id, std::string_view parent, StepKind kind,
                             std::string_view name, std::string* step_id) {
  if (trace_id.empty() || step_id == nullptr || static_cast<std::uint8_t>(kind) > static_cast<std::uint8_t>(StepKind::kCustom))
    return Outcome::kInvalid;
  const auto start = Now();
  std::lock_guard<std::mutex> lk(mu_);
  std::string id;
  do id = Hex16(rng_());
  while (open_.contains(id));
  open_.emplace(id, OpenStep{std::string(trace_id), std::string(parent), std::string(name), kind, start});
  *step_id = std::move(id);
  return Outcome::kOk;
}

Outcome Annotator::StepEnd(std::string_view step_id, Status status, std::string_view output) {
  OpenStep step;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = open_.find(std::string(step_id));
    if (it == open_.end()) return Outcome::kInvalid;
    step = std::move(it->second);
    open_.erase(it);
  }
  Observation o;
  o.project = project_;
  o.trace_id = std::move(step.trace_id);
  o.span_id = std::string(step_id);
  o.parent_span_id = std::move(step.parent);
  o.start_ns = step.start_ns;
  o.end_ns = Now();
  o.type = "SPAN";
  o.name = std::move(step.name);
  o.tags = {std::string("step:") + StepKindName(step.kind)};
  o.status_code = StatusName(status);
  o.level = status == Status::kOk ? "DEFAULT" : "ERROR";
  Producer* p = ThisThread();
  const bool payload_ok = CaptureOutput(p, output, o);
  const bool row_ok = p->Record(o);
  return payload_ok && row_ok ? Outcome::kOk : Outcome::kDropped;
}

Outcome Annotator::ToolResult(std::string_view trace_id, std::string_view tool_call_id, std::string_view tool_name,
                              Status status, std::string_view output, std::uint64_t latency_ns) {
  if (trace_id.empty() || tool_call_id.empty()) return Outcome::kInvalid;
  Observation o;
  o.project = project_;
  o.trace_id = std::string(trace_id);
  o.span_id = NewId();
  o.end_ns = Now();
  o.start_ns = o.end_ns - static_cast<std::int64_t>(std::min<std::uint64_t>(latency_ns, static_cast<std::uint64_t>(o.end_ns)));
  o.type = "TOOL";
  o.name = std::string(tool_name);
  o.tool_name = std::string(tool_name);
  o.tool_call_ids = {std::string(tool_call_id)};
  o.status_code = StatusName(status);
  o.level = status == Status::kOk ? "DEFAULT" : "ERROR";
  Producer* p = ThisThread();
  const bool payload_ok = CaptureOutput(p, output, o);
  const bool row_ok = p->Record(o);
  return payload_ok && row_ok ? Outcome::kOk : Outcome::kDropped;
}

Outcome Annotator::Score(const ScoreTarget& target, std::string_view name, std::string_view data_type, double value,
                         std::string_view string_value, std::string_view source, std::string_view comment,
                         std::string_view supersedes, std::string* score_id) {
  if (name.empty() || (target.trace_id.empty() && target.session_id.empty()) ||
      (!target.observation_id.empty() && target.trace_id.empty()))
    return Outcome::kInvalid;
  struct Score s;
  s.project = project_;
  s.score_id = NewId();
  s.supersedes = std::string(supersedes);
  s.kind = "assert";
  s.trace_id = target.trace_id;
  s.observation_id = target.observation_id;
  s.session_id = target.session_id;
  s.name = std::string(name);
  s.source = std::string(source);
  s.data_type = std::string(data_type);
  s.value = value;
  s.string_value = std::string(string_value);
  s.comment = std::string(comment);
  s.event_ns = Now();
  if (score_id) *score_id = s.score_id;
  return ThisThread()->Record(s) ? Outcome::kOk : Outcome::kDropped;
}

Outcome Annotator::ScoreNumeric(const ScoreTarget& target, std::string_view name, double value, std::string_view source,
                                std::string_view comment, std::string_view supersedes, std::string* score_id) {
  return Score(target, name, "NUMERIC", value, {}, source, comment, supersedes, score_id);
}

Outcome Annotator::ScoreCategorical(const ScoreTarget& target, std::string_view name, std::string_view value,
                                    std::string_view source, std::string_view comment, std::string_view supersedes,
                                    std::string* score_id) {
  return Score(target, name, "CATEGORICAL", 0, value, source, comment, supersedes, score_id);
}

Outcome Annotator::ScoreBoolean(const ScoreTarget& target, std::string_view name, bool value, std::string_view source,
                                std::string_view comment, std::string_view supersedes, std::string* score_id) {
  return Score(target, name, "BOOLEAN", value ? 1.0 : 0.0, {}, source, comment, supersedes, score_id);
}

Outcome Annotator::Retract(std::string_view score_id, std::string_view source, std::string_view comment) {
  if (score_id.empty()) return Outcome::kInvalid;
  struct Score s;
  s.project = project_;
  s.score_id = NewId();
  s.supersedes = std::string(score_id);
  s.kind = "retract";
  s.source = std::string(source);
  s.comment = std::string(comment);
  s.event_ns = Now();
  return ThisThread()->Record(s) ? Outcome::kOk : Outcome::kDropped;
}

Outcome Annotator::TraceMeta(std::string_view trace_id, std::string_view key, std::string_view value) {
  if (trace_id.empty() || key.empty()) return Outcome::kInvalid;
  TraceFact t;
  t.project = project_;
  t.trace_id = std::string(trace_id);
  t.start_ns = t.end_ns = Now();
  if (key == "session_id") t.session_id = std::string(value);
  else if (key == "user_id") t.user_id = std::string(value);
  else if (key == "release") t.release = std::string(value);
  else if (key == "name") t.name = std::string(value);
  else if (key == "tag") t.tags = {std::string(value)};
  else t.metadata = {{std::string(key), std::string(value)}};
  return ThisThread()->Record(t) ? Outcome::kOk : Outcome::kDropped;
}

Outcome Annotator::RecordScore(struct Score s) {
  if (s.score_id.empty() || s.name.empty() || (s.trace_id.empty() && s.session_id.empty())) return Outcome::kInvalid;
  s.project = project_;
  if (s.kind.empty()) s.kind = "assert";
  if (s.event_ns == 0) s.event_ns = Now();
  return ThisThread()->Record(s) ? Outcome::kOk : Outcome::kDropped;
}

Outcome Annotator::RecordTrace(TraceFact t) {
  if (t.trace_id.empty()) return Outcome::kInvalid;
  t.project = project_;
  if (t.start_ns == 0) t.start_ns = Now();
  if (t.end_ns == 0) t.end_ns = t.start_ns;
  return ThisThread()->Record(t) ? Outcome::kOk : Outcome::kDropped;
}

std::size_t Annotator::OpenSteps() const {
  std::lock_guard<std::mutex> lk(mu_);
  return open_.size();
}

}  // namespace halo::scope
