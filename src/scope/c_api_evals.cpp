// The halo-scope evaluation C ABI (include/halo_scope.h, second half) over
// EvalStore, the evaluators and the runner.

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#include "halo_scope.h"
#include "scope/c_api.hpp"
#include "scope/eval_json.hpp"
#include "scope/eval_stats.hpp"
#include "scope/evals.hpp"
#include "scope/evaluators.hpp"
#include "scope/runner.hpp"

struct halo_scope_evals {
  std::unique_ptr<halo::scope::EvalStore> store;
  halo::scope::Annotator* annotator = nullptr;
  halo::scope::CodeHooks hooks;
  halo::scope::HttpPoster poster;
};

struct halo_scope_eval_result {
  halo::scope::EvalOutcome outcome = halo::scope::EvalOutcome::Failed("NUMERIC", "the code hook set no result");
};

struct halo_scope_target_reply {
  halo::scope::ChatReply reply;
};

namespace {

using namespace halo::scope;

std::string Sv(const char* s) { return s ? std::string(s) : std::string(); }

int Fail(char* error, std::size_t size, const std::string& msg) {
  if (error && size) {
    const auto n = std::min(msg.size(), size - 1);
    std::memcpy(error, msg.data(), n);
    error[n] = '\0';
  }
  return HALO_SCOPE_INVALID;
}

void CopyOut(const std::string& s, char* out, std::size_t size) {
  if (!out || size == 0) return;
  const auto n = std::min(s.size(), size - 1);
  std::memcpy(out, s.data(), n);
  out[n] = '\0';
}

}  // namespace

extern "C" {

halo_scope_evals* halo_scope_evals_open(const char* metadata_path, halo_scope* scope, char* error, size_t error_size) {
  if (!metadata_path || !*metadata_path) return Fail(error, error_size, "metadata_path required"), nullptr;
  std::string err;
  auto store = EvalStore::Open(metadata_path, &err);
  if (!store) return Fail(error, error_size, err), nullptr;
  auto* h = new halo_scope_evals;
  h->store = std::move(store);
  if (scope) h->annotator = &AnnotatorOf(scope);
  return h;
}

void halo_scope_evals_close(halo_scope_evals* evals) { delete evals; }

int halo_scope_dataset_create(halo_scope_evals* ev, const char* name, const char* description,
                              const char* metadata_json, char* error, size_t error_size) {
  if (!ev) return HALO_SCOPE_INVALID;
  std::string err;
  return ev->store->CreateDataset(Sv(name), Sv(description), Sv(metadata_json), &err) ? HALO_SCOPE_OK
                                                                                       : Fail(error, error_size, err);
}

int halo_scope_dataset_add_item(halo_scope_evals* ev, const char* dataset, const char* item_id,
                                const char* input_json, const char* expected_json, const char* metadata_json,
                                char new_id[HALO_SCOPE_ID_SIZE], char* error, size_t error_size) {
  if (!ev) return HALO_SCOPE_INVALID;
  std::vector<DatasetItem> items(1);
  items[0].id = Sv(item_id);
  items[0].input_json = input_json ? input_json : "null";
  items[0].expected_json = expected_json ? expected_json : "null";
  items[0].metadata_json = metadata_json ? metadata_json : "{}";
  std::string err;
  if (ev->store->UpsertItems(Sv(dataset), items, &err) == 0) return Fail(error, error_size, err);
  if (!item_id) CopyOut(items[0].id, new_id, HALO_SCOPE_ID_SIZE);
  return HALO_SCOPE_OK;
}

int halo_scope_dataset_import_jsonl(halo_scope_evals* ev, const char* dataset, const char* data, size_t size,
                                    size_t* count, char* error, size_t error_size) {
  if (!ev || (!data && size)) return HALO_SCOPE_INVALID;
  std::string err;
  std::size_t n = 0;
  if (ev->store->ImportJsonl(Sv(dataset), std::string_view(data ? data : "", size), &n, &err) == 0)
    return Fail(error, error_size, err);
  if (count) *count = n;
  return HALO_SCOPE_OK;
}

int halo_scope_dataset_freeze(halo_scope_evals* ev, const char* dataset, int64_t* version, char content_hash[33],
                              char* error, size_t error_size) {
  if (!ev) return HALO_SCOPE_INVALID;
  std::string err;
  auto v = ev->store->Freeze(Sv(dataset), &err);
  if (!v) return Fail(error, error_size, err);
  if (version) *version = v->version;
  CopyOut(v->content_hash, content_hash, 33);
  return HALO_SCOPE_OK;
}

int halo_scope_dataset_archive(halo_scope_evals* ev, const char* dataset, char* error, size_t error_size) {
  if (!ev) return HALO_SCOPE_INVALID;
  std::string err;
  return ev->store->Archive(Sv(dataset), &err) ? HALO_SCOPE_OK : Fail(error, error_size, err);
}

int halo_scope_evaluator_create(halo_scope_evals* ev, const char* name, const char* kind, const char* config_json,
                                int64_t* version, char* error, size_t error_size) {
  if (!ev) return HALO_SCOPE_INVALID;
  std::string err;
  auto config = PrepareEvaluatorConfig(*ev->store, Sv(kind), Sv(config_json), &err);
  if (!config) return Fail(error, error_size, err);
  const auto v = ev->store->CreateEvaluatorVersion(Sv(name), Sv(kind), *config, &err);
  if (v == 0) return Fail(error, error_size, err);
  if (version) *version = v;
  return HALO_SCOPE_OK;
}

void halo_scope_eval_set_numeric(halo_scope_eval_result* r, double value, const char* comment) {
  if (r) r->outcome = std::isfinite(value) ? EvalOutcome::Numeric(value, Sv(comment))
                                           : EvalOutcome::Failed("NUMERIC", "the code hook returned a non-finite value");
}
void halo_scope_eval_set_boolean(halo_scope_eval_result* r, int value, const char* comment) {
  if (r) r->outcome = EvalOutcome::Boolean(value != 0, Sv(comment));
}
void halo_scope_eval_set_categorical(halo_scope_eval_result* r, const char* value, const char* comment) {
  if (r) r->outcome = EvalOutcome::Categorical(Sv(value), Sv(comment));
}
void halo_scope_eval_set_failed(halo_scope_eval_result* r, const char* error) {
  if (r) r->outcome = EvalOutcome::Failed("NUMERIC", Sv(error));
}
void halo_scope_eval_set_retry(halo_scope_eval_result* r, const char* error) {
  if (r) r->outcome = EvalOutcome::Transient(Sv(error));
}

int halo_scope_evals_register_hook(halo_scope_evals* ev, const char* name, halo_scope_eval_fn fn, void* user) {
  if (!ev || !name || !*name || !fn) return HALO_SCOPE_INVALID;
  ev->hooks.Register(name, [fn, user](const EvalContext& ctx) {
    halo_scope_eval_result r;
    halo_scope_eval_input in{};
    in.run_id = ctx.run_id.c_str();
    in.item_id = ctx.item->id.c_str();
    in.attempt = ctx.attempt;
    in.input_json = ctx.item->input_json.c_str();
    in.expected_json = ctx.item->expected_json.c_str();
    in.metadata_json = ctx.item->metadata_json.c_str();
    in.output = ctx.output.data();
    in.output_size = ctx.output.size();
    fn(user, &in, &r);
    return r.outcome;
  });
  return HALO_SCOPE_OK;
}

void halo_scope_target_set_output(halo_scope_target_reply* r, const char* text, size_t text_size, const char* model,
                                  const char* finish_reason, uint64_t input_tokens, uint64_t output_tokens) {
  if (!r || (!text && text_size)) return;
  r->reply.status = ChatReply::Status::kOk;
  r->reply.text.assign(text ? text : "", text_size);
  r->reply.model = Sv(model);
  r->reply.finish_reason = Sv(finish_reason);
  r->reply.prompt_tokens = input_tokens;
  r->reply.completion_tokens = output_tokens;
  r->reply.error.clear();
}

void halo_scope_target_set_error(halo_scope_target_reply* r, int transient, const char* error) {
  if (!r) return;
  r->reply.status = transient ? ChatReply::Status::kTransient : ChatReply::Status::kFailed;
  r->reply.error = Sv(error);
}

int halo_scope_run_experiment(halo_scope_evals* ev, const halo_scope_run_options* o, char run_id[HALO_SCOPE_ID_SIZE],
                              char* error, size_t error_size) {
  if (!ev || !o) return HALO_SCOPE_INVALID;
  RunSpec spec;
  spec.name = Sv(o->name);
  spec.dataset = Sv(o->dataset);
  spec.dataset_version = o->dataset_version;
  spec.target_json = o->target_json ? o->target_json : "{}";
  spec.max_attempts = o->max_attempts > 0 ? o->max_attempts : 3;
  const std::string list = Sv(o->evaluators);
  for (std::size_t i = 0; i < list.size();) {
    const auto j = std::min(list.find(',', i), list.size());
    std::string ref = list.substr(i, j - i);
    i = j + 1;
    while (!ref.empty() && ref.front() == ' ') ref.erase(0, 1);
    while (!ref.empty() && ref.back() == ' ') ref.pop_back();
    if (ref.empty()) continue;
    const auto at = ref.find('@');
    const std::int64_t v = at == std::string::npos ? 0 : std::strtoll(ref.c_str() + at + 1, nullptr, 10);
    spec.evaluators.emplace_back(ref.substr(0, at), v);
  }
  std::string err;
  bool existed = false;
  auto run = ev->store->CreateRun(spec, &existed, &err);
  if (!run) return Fail(error, error_size, err);
  CopyOut(run->id, run_id, HALO_SCOPE_ID_SIZE);
  RunnerOptions opt;
  opt.concurrency = o->concurrency > 0 ? o->concurrency : 1;
  opt.target_endpoint = Sv(o->endpoint);
  opt.env.poster = &ev->poster;
  opt.env.default_endpoint = Sv(o->endpoint);
  opt.env.decider_endpoint = o->decider_endpoint ? o->decider_endpoint : Sv(o->endpoint);
  opt.env.store = ev->store.get();
  opt.env.hooks = &ev->hooks;
  opt.annotator = ev->annotator;
  if (o->target_fn) {
    opt.target = [fn = o->target_fn, user = o->target_user](const TargetCall& c) {
      halo_scope_target_reply r;
      r.reply.status = ChatReply::Status::kFailed;
      r.reply.error = "the target set no output";
      const std::string body = halo::json::DumpPython(c.body);
      halo_scope_target_call call{c.run_id.c_str(), c.item_id.c_str(), c.trace_id.c_str(), c.attempt, body.c_str()};
      fn(user, &call, &r);
      return r.reply;
    };
  }
  if (o->progress) {
    opt.on_progress = [fn = o->progress, user = o->progress_user](const RunProgress& p) {
      fn(user, p.done, p.total, p.error + p.exhausted);
    };
  }
  const RunOutcome out = ExecuteRun(*ev->store, ev->poster, run->id, opt);
  if (out.status == "completed") return HALO_SCOPE_OK;
  if (out.status == "refused") return Fail(error, error_size, out.error);
  Fail(error, error_size, out.error);
  return HALO_SCOPE_INCOMPLETE;
}

int halo_scope_run_compare(halo_scope_evals* ev, const char* run_a, const char* run_b, const char* score,
                           halo_scope_comparison* out, char* error, size_t error_size) {
  if (!ev || !out) return HALO_SCOPE_INVALID;
  auto a = ev->store->GetRun(Sv(run_a));
  auto b = ev->store->GetRun(Sv(run_b));
  if (!a || !b) return Fail(error, error_size, "no such run");
  if (a->dataset_id != b->dataset_id) return Fail(error, error_size, "runs are on different datasets");
  std::string err;
  auto c = ComparePaired(ev->store->RunScores(a->id), ev->store->RunScores(b->id), Sv(score), &err);
  if (!c) return Fail(error, error_size, err);
  const double nan = std::nan("");
  *out = {c->n, c->only_a, c->only_b, c->wins, c->losses, c->ties, c->mean_a.value_or(nan), c->mean_b.value_or(nan),
          c->mean_delta.value_or(nan), c->se_delta.value_or(nan)};
  return HALO_SCOPE_OK;
}

int halo_scope_run_summary_json(halo_scope_evals* ev, const char* run_id, char* buffer, size_t size, size_t* needed) {
  if (!ev) return HALO_SCOPE_INVALID;
  auto r = ev->store->GetRun(Sv(run_id));
  if (!r) return HALO_SCOPE_INVALID;
  const std::string s = halo::json::DumpPython(RunJson(*ev->store, *r, true));
  if (needed) *needed = s.size();
  CopyOut(s, buffer, size);
  return HALO_SCOPE_OK;
}

}  // extern "C"
