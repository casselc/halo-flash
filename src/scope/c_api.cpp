// The halo-scope annotation C ABI (include/halo_scope.h) over Annotator.

#include "halo_scope.h"

#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#include "scope/annotate.hpp"
#include "scope/c_api.hpp"
#include "scope/chdb.hpp"
#include "scope/recorder.hpp"
#include "scope/schema.hpp"

struct halo_scope {
  // Owned when opened through halo_scope_open; tests wrap their own.
  std::unique_ptr<halo::scope::Sink> sink;
  std::unique_ptr<halo::scope::Recorder> owned_recorder;
  halo::scope::Recorder* recorder = nullptr;
  std::unique_ptr<halo::scope::Annotator> annotator;
};

namespace {

using halo::scope::Outcome;

std::string_view Sv(const char* s) { return s ? std::string_view(s) : std::string_view(); }

int Code(Outcome o) {
  switch (o) {
    case Outcome::kOk: return HALO_SCOPE_OK;
    case Outcome::kDropped: return HALO_SCOPE_DROPPED;
    case Outcome::kInvalid: return HALO_SCOPE_INVALID;
  }
  return HALO_SCOPE_INVALID;
}

void CopyId(const std::string& id, char* out) {
  if (!out) return;
  const auto n = std::min<std::size_t>(id.size(), HALO_SCOPE_ID_SIZE - 1);
  std::memcpy(out, id.data(), n);
  out[n] = '\0';
}

void SetError(char* error, std::size_t size, const std::string& msg) {
  if (!error || size == 0) return;
  const auto n = std::min(msg.size(), size - 1);
  std::memcpy(error, msg.data(), n);
  error[n] = '\0';
}

bool ValidStatus(halo_scope_status s) { return s == HALO_SCOPE_STATUS_OK || s == HALO_SCOPE_STATUS_ERROR; }

halo::scope::Status ToStatus(halo_scope_status s) {
  return s == HALO_SCOPE_STATUS_OK ? halo::scope::Status::kOk : halo::scope::Status::kError;
}

halo::scope::ScoreTarget ToTarget(const halo_scope_target* t) {
  return {std::string(Sv(t->trace_id)), std::string(Sv(t->observation_id)), std::string(Sv(t->session_id))};
}

}  // namespace

namespace halo::scope {

halo_scope* WrapRecorder(Recorder& rec, std::string project, Capture capture, Annotator::Clock clock) {
  auto* h = new halo_scope;
  h->recorder = &rec;
  h->annotator = std::make_unique<Annotator>(rec, std::move(project), capture, std::move(clock));
  return h;
}

}  // namespace halo::scope

extern "C" {

halo_scope* halo_scope_open(const char* data_path, const char* project, halo_scope_capture capture,
                            const char* wal_path, char* error, size_t error_size) {
  using namespace halo::scope;
  if (!data_path || !*data_path || capture < HALO_SCOPE_CAPTURE_OFF || capture > HALO_SCOPE_CAPTURE_FULL) {
    SetError(error, error_size, "halo_scope_open: data_path required and capture must be OFF, DIGEST or FULL");
    return nullptr;
  }
  std::string err;
  const ChdbLib* lib = ChdbLib::Load("", &err);
  if (!lib) return SetError(error, error_size, err), nullptr;
  auto conn = Conn::Open(*lib, data_path, &err);
  if (!conn) return SetError(error, error_size, err), nullptr;
  if (MigrateChdb(*conn, &err) < 0) return SetError(error, error_size, err), nullptr;
  auto h = std::make_unique<halo_scope>();
  h->sink = std::make_unique<ChdbSink>(std::move(conn));
  RecorderOptions opt;
  if (wal_path) opt.wal_path = wal_path;
  h->owned_recorder = std::make_unique<Recorder>(*h->sink, opt);
  if (!h->owned_recorder->error().empty()) return SetError(error, error_size, h->owned_recorder->error()), nullptr;
  h->recorder = h->owned_recorder.get();
  h->annotator = std::make_unique<Annotator>(*h->recorder, std::string(Sv(project)), static_cast<Capture>(capture));
  return h.release();
}

void halo_scope_close(halo_scope* scope) {
  if (!scope) return;
  scope->annotator.reset();
  if (scope->owned_recorder) scope->owned_recorder->Stop();
  scope->owned_recorder.reset();
  scope->sink.reset();
  delete scope;
}

void halo_scope_flush(halo_scope* scope) {
  if (scope) scope->recorder->Flush();
}

int halo_scope_step_begin(halo_scope* scope, const char* trace_id, const char* parent, halo_scope_step_kind kind,
                          const char* name, char step_id[HALO_SCOPE_ID_SIZE]) {
  if (!scope || !step_id || kind < HALO_SCOPE_STEP_TOOL || kind > HALO_SCOPE_STEP_CUSTOM) return HALO_SCOPE_INVALID;
  std::string id;
  const auto o = scope->annotator->StepBegin(Sv(trace_id), Sv(parent), static_cast<halo::scope::StepKind>(kind),
                                             Sv(name), &id);
  if (o == Outcome::kOk) CopyId(id, step_id);
  return Code(o);
}

int halo_scope_step_end(halo_scope* scope, const char* step_id, halo_scope_status status, const char* output,
                        size_t output_size) {
  if (!scope || !ValidStatus(status) || (!output && output_size)) return HALO_SCOPE_INVALID;
  return Code(scope->annotator->StepEnd(Sv(step_id), ToStatus(status), std::string_view(output ? output : "", output_size)));
}

int halo_scope_tool_result(halo_scope* scope, const char* trace_id, const char* tool_call_id, const char* tool_name,
                           halo_scope_status status, const char* output, size_t output_size, uint64_t latency_ns) {
  if (!scope || !ValidStatus(status) || (!output && output_size)) return HALO_SCOPE_INVALID;
  return Code(scope->annotator->ToolResult(Sv(trace_id), Sv(tool_call_id), Sv(tool_name), ToStatus(status),
                                           std::string_view(output ? output : "", output_size), latency_ns));
}

int halo_scope_score_numeric(halo_scope* scope, const halo_scope_target* target, const char* name, double value,
                             const char* source, const char* comment, const char* supersedes,
                             char score_id[HALO_SCOPE_ID_SIZE]) {
  if (!scope || !target) return HALO_SCOPE_INVALID;
  std::string id;
  const auto o = scope->annotator->ScoreNumeric(ToTarget(target), Sv(name), value, Sv(source), Sv(comment),
                                                Sv(supersedes), &id);
  if (o != Outcome::kInvalid) CopyId(id, score_id);
  return Code(o);
}

int halo_scope_score_categorical(halo_scope* scope, const halo_scope_target* target, const char* name,
                                 const char* value, const char* source, const char* comment, const char* supersedes,
                                 char score_id[HALO_SCOPE_ID_SIZE]) {
  if (!scope || !target) return HALO_SCOPE_INVALID;
  std::string id;
  const auto o = scope->annotator->ScoreCategorical(ToTarget(target), Sv(name), Sv(value), Sv(source), Sv(comment),
                                                    Sv(supersedes), &id);
  if (o != Outcome::kInvalid) CopyId(id, score_id);
  return Code(o);
}

int halo_scope_score_boolean(halo_scope* scope, const halo_scope_target* target, const char* name, int value,
                             const char* source, const char* comment, const char* supersedes,
                             char score_id[HALO_SCOPE_ID_SIZE]) {
  if (!scope || !target) return HALO_SCOPE_INVALID;
  std::string id;
  const auto o = scope->annotator->ScoreBoolean(ToTarget(target), Sv(name), value != 0, Sv(source), Sv(comment),
                                                Sv(supersedes), &id);
  if (o != Outcome::kInvalid) CopyId(id, score_id);
  return Code(o);
}

int halo_scope_score_retract(halo_scope* scope, const char* score_id, const char* source, const char* comment) {
  if (!scope) return HALO_SCOPE_INVALID;
  return Code(scope->annotator->Retract(Sv(score_id), Sv(source), Sv(comment)));
}

int halo_scope_trace_meta(halo_scope* scope, const char* trace_id, const char* key, const char* value) {
  if (!scope) return HALO_SCOPE_INVALID;
  return Code(scope->annotator->TraceMeta(Sv(trace_id), Sv(key), Sv(value)));
}

void halo_scope_get_stats(const halo_scope* scope, halo_scope_stats* stats) {
  if (!scope || !stats) return;
  const auto s = scope->recorder->stats();
  stats->accepted = s.accepted;
  stats->dropped = s.dropped;
  stats->rows_written = 0;
  for (auto n : s.rows_written) stats->rows_written += n;
  stats->rows_failed = s.rows_failed;
}

}  // extern "C"
