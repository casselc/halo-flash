#include "server/scheduled_generator.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>


namespace halo::server {

namespace {
using Clock = std::chrono::steady_clock;
const bool kTrace = std::getenv("HALO_SCHED_TRACE") != nullptr;  // every admission, unit and outcome on stderr
double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
}  // namespace

struct ScheduledGenerator::Job {
  Scheduler::Id id = 0;
  std::vector<std::int32_t> ids;
  std::size_t stable = 0;
  sampling::SamplingConfig sampling;
  std::uint32_t budget = 0;
  bool spec_ok = false;  // MTP would apply (enabled, no penalties) if it starts alone
  std::function<void()> work;  // a decider evaluation (Scheduler kDecide): runs on the engine thread
  std::int64_t admitted_ms = 0;
  bool waited = false;  // its queue wait (admission to its first unit) was observed (engine thread)
  // engine thread only
  bool begun = false, spec = false, finished = false;
  std::unique_ptr<sampling::SamplerState> state;
  std::unique_ptr<DecodeRun> run;
  SpecDecodeStats spec_stats;
  std::string pending;  // a trailing incomplete UTF-8 sequence
  // shared with the waiting HTTP thread (mu)
  std::mutex mu;
  std::condition_variable cv;
  std::string text;
  std::vector<std::int32_t> tokens;
  bool done = false, consumer_stop = false;
  std::uint64_t gen = 0;  // attachment: an adopting retry bumps it, and the earlier reader leaves
  Result result;
  bool keyed = false;  // has an idempotency key: a departed reader leaves it running (keyed grace)
  // its completion tokens were counted (by the reader that took the whole answer, or a replay of it)
  std::shared_ptr<std::atomic<bool>> counted = std::make_shared<std::atomic<bool>>(false);
};

namespace {
struct Ticket {
  std::shared_ptr<void> job;  // ScheduledGenerator::Job
  std::uint64_t gen = 0;
  bool replay = false, dropped = false;
  std::string text;
  std::vector<std::int32_t> tokens;
  TextGenerator::Result result;
};
}  // namespace

ScheduledGenerator::ScheduledGenerator(std::unique_ptr<ScheduledBackend> backend, std::uint32_t max_context,
                                       engine::MtpDepthOptions depth, ScheduledOptions options)
    : backend_(std::move(backend)), max_context_(max_context), opt_(std::move(options)),
      chooser_(engine::MtpMeasuredCosts(), depth), sched_(opt_.sched, {}) {
  // The scheduler's engine hooks run on the engine thread inside Next() (its lock held: they must not call it).
  Scheduler::Hooks h;
  h.park = [this](Scheduler::Id id) { return backend_->Park(id); };
  h.evict = [this](Scheduler::Key k) -> std::uint64_t {
    backend_->Evict(k);
    return 0;
  };
  sched_.SetHooks(std::move(h));
  epoch_ = Clock::now();
  bytes_per_token_ = backend_->BytesPerToken();
  thread_ = std::thread([this] { Loop(); });
}

ScheduledGenerator::~ScheduledGenerator() {
  stop_ = true;
  wake_.notify_all();
  if (thread_.joinable()) thread_.join();
}

std::int64_t ScheduledGenerator::NowMs() const {
  return opt_.now_ms ? opt_.now_ms() : static_cast<std::int64_t>(Ms(epoch_, Clock::now()));
}

std::shared_ptr<ScheduledGenerator::Job> ScheduledGenerator::FindShared(Job& job) {
  auto j = Find(job.id);
  return j ? j : std::shared_ptr<Job>(&job, [](Job*) {});
}

std::shared_ptr<ScheduledGenerator::Job> ScheduledGenerator::Find(Scheduler::Id id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = jobs_.find(id);
  return it == jobs_.end() ? nullptr : it->second;
}

void ScheduledGenerator::Post(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    posts_.push_back(std::move(task));
  }
  wake_.notify_all();
}

// ---------------------------------------------------------------- HTTP threads

TextGenerator::Admission ScheduledGenerator::Admit(const Params& params) {
  std::vector<std::int32_t> ids;
  std::size_t stable = 0;
  if (!params.variant.empty() || !backend_->Encode(params.prompt, &ids, &stable)) return {};  // Generate answers
  return AdmitIds(ids, stable, params);
}

TextGenerator::Admission ScheduledGenerator::AdmitIds(std::span<const std::int32_t> prompt, std::size_t stable,
                                                      const Params& params) {
  Admission a;
  if (prompt.empty() || prompt.size() >= max_context_) return a;  // GenerateIds answers the error
  auto job = std::make_shared<Job>();
  job->ids.assign(prompt.begin(), prompt.end());
  job->stable = stable;
  job->sampling = params.sampling;
  job->budget = std::min<std::uint32_t>(params.max_tokens, max_context_ - static_cast<std::uint32_t>(prompt.size()));
  job->spec_ok = backend_->CanSpeculate() && !params.sampling.penalties_enabled();
  job->keyed = !params.idempotency_key.empty();
  job->result.prompt_tokens = prompt.size();
  Scheduler::Turn t;
  t.principal = params.principal;
  t.root = params.chain_root;
  t.key = params.idempotency_key;
  t.prompt_tokens = prompt.size() - 1;  // prefilled; the last token is the first decode step's
  t.max_tokens = job->budget + (job->spec_ok ? backend_->MaxDrafts() : 0);
  t.prompt = std::make_shared<const std::vector<std::int32_t>>(job->ids);
  t.bytes_per_token = bytes_per_token_;
  t.fixed_bytes = opt_.fixed_bytes;
  const bool connected = !(params.cancelled && params.cancelled());
  std::lock_guard<std::mutex> lk(mu_);  // the job is registered before the engine thread can pick it
  const auto adm = sched_.Arrive(t, NowMs(), connected);
  using O = Scheduler::Outcome;
  if (kTrace)
    std::fprintf(stderr, "sched: arrive %s/%s key %s -> outcome %d id %llu target %llu\n", t.principal.c_str(),
                 t.root.c_str(), t.key.c_str(), static_cast<int>(adm.outcome), static_cast<unsigned long long>(adm.id),
                 static_cast<unsigned long long>(adm.target));
  auto ticket = std::make_shared<Ticket>();
  switch (adm.outcome) {
    case O::kQueued:
      job->id = adm.id;
      job->admitted_ms = NowMs();
      jobs_[adm.id] = job;
      backend_->Admitted(adm.id, job->ids);
      ticket->job = job;
      a.answer = "generated";
      kick_ = true;
      wake_.notify_all();
      break;
    case O::kAdopted: {
      auto it = jobs_.find(adm.target);
      if (it != jobs_.end()) {
        std::lock_guard<std::mutex> jl(it->second->mu);
        ticket->gen = ++it->second->gen;
        ticket->job = it->second;
        it->second->cv.notify_all();  // the earlier reader leaves
        a.answer = "adopted";
        break;
      }
      [[fallthrough]];  // it finished meanwhile: its answer is stored
    }
    case O::kReplayed: {
      auto it = replay_.find(adm.target);
      if (it == replay_.end()) {
        ticket->dropped = true;  // forgotten (kept as long as the scheduler keeps the key: should not happen)
        break;
      }
      ticket->replay = true;
      a.answer = "replayed";
      ticket->text = it->second.text;
      ticket->tokens = it->second.tokens;
      ticket->result = it->second.result;
      ticket->result.replayed = true;
      ticket->result.uncounted = it->second.counted->exchange(true);
      break;
    }
    case O::kDuplicate:
      ticket->dropped = true;
      break;
    case O::kLimit:
    case O::kMemory:
      a.refused = true;
      a.retry_after_s = adm.retry_after_s;
      a.why = adm.outcome == O::kLimit ? "limit" : "memory";
      return a;
  }
  a.ticket = ticket;
  return a;
}

TextGenerator::Result ScheduledGenerator::Generate(const Params& params,
                                                   const std::function<bool(std::string_view)>& on_text) {
  Result r;
  if (!params.variant.empty()) return r.error = "Flash-Next variants are not supported yet (P11 stage 2)", r;
  if (!backend_->HasTokenizer()) return r.error = "no tokenizer", r;
  Params p = params;
  if (!p.ticket) {
    const auto a = Admit(p);
    if (a.refused) {
      r.refused = true;
      r.retry_after_s = a.retry_after_s;
      r.refused_why = a.why;
      return r;
    }
    p.ticket = a.ticket;
  }
  if (!p.ticket) {  // not admitted: an empty or too long prompt
    std::vector<std::int32_t> ids;
    std::size_t stable = 0;
    (void)backend_->Encode(params.prompt, &ids, &stable);
    r.prompt_tokens = ids.size();
    return r.error = ids.empty() ? "empty prompt"
                                 : "prompt of " + std::to_string(ids.size()) + " tokens exceeds the context of " +
                                       std::to_string(max_context_),
           r;
  }
  auto* t = static_cast<Ticket*>(p.ticket.get());
  if (t->dropped) return r.cancelled = true, r;
  if (t->replay) {
    if (!t->text.empty()) (void)on_text(t->text);
    return t->result;
  }
  return Wait(std::static_pointer_cast<Job>(t->job), t->gen, p, nullptr, &on_text);
}

std::optional<std::uint64_t> ScheduledGenerator::CountTokens(std::string_view prompt) const {
  if (!backend_->HasTokenizer()) return std::nullopt;
  std::vector<std::int32_t> ids;
  std::size_t stable = 0;
  if (!backend_->Encode(std::string(prompt), &ids, &stable)) return std::nullopt;
  return ids.size();
}

TextGenerator::Result ScheduledGenerator::GenerateIds(std::span<const std::int32_t> prompt, std::size_t stable,
                                                      const Params& params,
                                                      const std::function<bool(text::TokenId)>& on_token) {
  Result r;
  r.prompt_tokens = prompt.size();
  if (prompt.empty()) return r.error = "empty prompt", r;
  if (prompt.size() >= max_context_)
    return r.error = "prompt of " + std::to_string(prompt.size()) + " tokens exceeds the context of " +
                     std::to_string(max_context_), r;
  Params p = params;
  if (!p.ticket) {
    const auto a = AdmitIds(prompt, stable, p);
    if (a.refused) {
      r.refused = true;
      r.retry_after_s = a.retry_after_s;
      r.refused_why = a.why;
      return r;
    }
    p.ticket = a.ticket;
  }
  auto* t = static_cast<Ticket*>(p.ticket.get());
  if (t->dropped) return r.cancelled = true, r;
  if (t->replay) {
    for (auto tok : t->tokens)
      if (!on_token(static_cast<text::TokenId>(tok))) break;
    return t->result;
  }
  return Wait(std::static_pointer_cast<Job>(t->job), t->gen, p, &on_token, nullptr);
}

// Delivers the job's output as the engine thread produces it, polling the client's departure (and with it the SSE
// keepalive) every 20 ms.
TextGenerator::Result ScheduledGenerator::Wait(const std::shared_ptr<Job>& job, std::uint64_t gen, const Params& params,
                                               const std::function<bool(text::TokenId)>* on_token,
                                               const std::function<bool(std::string_view)>* on_text) {
  const auto gone = [&] { return params.cancelled && params.cancelled(); };
  std::size_t off_text = 0, off_tok = 0;
  for (;;) {
    std::string chunk;
    std::vector<std::int32_t> toks;
    bool done = false;
    Result res;
    {
      std::unique_lock<std::mutex> lk(job->mu);
      job->cv.wait_for(lk, std::chrono::milliseconds(20), [&] {
        return job->text.size() > off_text || job->tokens.size() > off_tok || job->done || job->gen != gen;
      });
      if (job->gen != gen) {  // a retry with the same key took this request over
        res = job->result;
        res.cancelled = res.detached = res.uncounted = true;
        return res;
      }
      chunk = job->text.substr(off_text);
      off_text = job->text.size();
      toks.assign(job->tokens.begin() + static_cast<std::ptrdiff_t>(off_tok), job->tokens.end());
      off_tok = job->tokens.size();
      done = job->done;
      res = job->result;
    }
    bool more = true;
    if (on_token) {
      for (auto t : toks)
        if (more) more = (*on_token)(static_cast<text::TokenId>(t));
    } else if (on_text && !chunk.empty()) {
      more = (*on_text)(chunk);
    }
    if (!more && gone()) more = true;  // the callback found the client gone: a departure (below), not a stop
    if (!more) {
      // The consumer stopped (a stop string): the answer ends here, complete. The engine thread finishes the request
      // at its next token. (A client that left is a departure, never this: its key would replay a truncated answer
      // as finished.)
      std::lock_guard<std::mutex> lk(job->mu);
      job->consumer_stop = true;
      Result r = job->result;
      r.completion_tokens = off_tok;
      r.cancelled = false;
      return r;
    }
    if (done) {
      res.uncounted = job->counted->exchange(true);
      return res;
    }
    if (gone()) {
      sched_.Depart(job->id, NowMs());
      res.cancelled = true;
      // A keyed request runs on through its grace for a retry to adopt or replay; whoever takes the answer counts it.
      if (job->keyed && opt_.sched.keyed_grace_s > 0) res.detached = res.uncounted = true;
      return res;
    }
  }
}

bool ScheduledGenerator::RunDecider(const std::function<void()>& work, std::string* error) {
  auto job = std::make_shared<Job>();
  job->work = work;
  {
    Scheduler::Turn t;
    t.decider = true;
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_) return *error = "the scheduler stopped", false;
    const auto adm = sched_.Arrive(t, NowMs());
    job->id = adm.id;
    job->admitted_ms = NowMs();
    jobs_[adm.id] = job;
    kick_ = true;
  }
  wake_.notify_all();
  std::unique_lock<std::mutex> lk(job->mu);
  job->cv.wait(lk, [&] { return job->done; });
  if (!job->result.error.empty() || job->result.cancelled)
    return *error = job->result.error.empty() ? "the scheduler stopped" : job->result.error, false;
  return true;
}

void ScheduledGenerator::RenderMetrics(std::string& out, const std::string& labels) const {
  std::uint64_t queued = 0, prefilling = 0, decoding = 0, deciders = 0;
  for (const auto& v : sched_.Requests()) {
    if (v.decider) ++deciders;
    else if (!v.started) ++queued;
    else if (!v.prefill_done) ++prefilling;
    else ++decoding;
  }
  const auto c = sched_.counters();
  const auto l = sched_.ledger();
  const auto gauge = [&](const char* name, const char* help, double v) {
    out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " gauge\n";
    out += std::string(name) + "{" + labels + "} " + std::to_string(v) + "\n";
  };
  const auto counter = [&](const char* name, const char* help, std::uint64_t v) {
    out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " counter\n";
    out += std::string(name) + "{" + labels + "} " + std::to_string(v) + "\n";
  };
  gauge("halo:sched_requests_queued", "Admitted agent requests not started.", static_cast<double>(queued));
  gauge("halo:sched_requests_prefilling", "Started agent requests still prefilling.", static_cast<double>(prefilling));
  gauge("halo:sched_requests_decoding", "Agent requests decoding.", static_cast<double>(decoding));
  gauge("halo:sched_deciders_active", "Decider evaluations admitted and not finished.", static_cast<double>(deciders));
  gauge("halo:sched_ledger_capacity_bytes", "Reservation ledger capacity.", static_cast<double>(l.capacity));
  gauge("halo:sched_ledger_retained_bytes", "Ledger bytes held by retained prefixes.", static_cast<double>(l.retained));
  gauge("halo:sched_ledger_reserved_bytes", "Ledger bytes reserved by admitted requests.", static_cast<double>(l.reserved));
  gauge("halo:sched_decider_credit", "Decider evaluations banked (D4).", sched_.credit(NowMs()));
  counter("halo:sched_admitted_total", "Requests admitted (deciders included).", c.admitted);
  counter("halo:sched_replayed_total", "Idempotency keys replayed from a finished turn.", c.replayed);
  counter("halo:sched_adopted_total", "Idempotency keys adopted while in flight.", c.adopted);
  counter("halo:sched_duplicates_total", "Late duplicates of an in-flight key dropped.", c.duplicates);
  counter("halo:sched_refused_limit_total", "429s: the principal's in-flight limit.", c.limit429);
  counter("halo:sched_refused_memory_total", "429s: no ledger room.", c.memory429);
  counter("halo:sched_evictions_total", "Retained prefixes evicted by admission.", c.evictions);
  counter("halo:sched_fenced_total", "Requests fenced after their client left.", c.fenced);
  counter("halo:sched_grace_finished_total", "Requests that finished while their client was away (a keyed one is replayable).",
          c.grace_finished);
  counter("halo:sched_parked_total", "Fenced prefills whose progress was parked.", c.parked);
  counter("halo:sched_fenced_decoded_tokens_total",
          "Tokens decoded by requests that were then fenced (in no answer's usage).", fenced_decoded_.load());
  counter("halo:sched_units_total", "Scheduler units run.", c.units);
  counter("halo:sched_slices_total", "Prefill slices.", c.slices);
  counter("halo:sched_batches_total", "Decode units.", c.batches);
  counter("halo:sched_batched_total", "Decode units with more than one request.", c.batched);
  counter("halo:sched_decides_total", "Decider units.", c.decides);
  counter("halo:sched_preemptions_total", "A started prefill's next slice yielded to another request.", c.preemptions);
  counter("halo:sched_decider_out_of_credit_total", "Deciders admitted without credit.", c.decider_out_of_credit);
  UnitStats s;
  {
    std::lock_guard<std::mutex> lk(stats_mu_);
    s = stats_;
  }
  RenderHistogram(out, "halo:sched_unit_seconds", "Scheduler unit length, seconds.", labels + ",kind=\"slice\"", s.slice);
  RenderHistogram(out, "halo:sched_unit_seconds", "Scheduler unit length, seconds.", labels + ",kind=\"batch\"", s.batch, false);
  RenderHistogram(out, "halo:sched_unit_seconds", "Scheduler unit length, seconds.", labels + ",kind=\"decide\"", s.decide, false);
  RenderHistogram(out, "halo:sched_queue_wait_seconds", "Admission to a request's first unit, seconds.", labels,
                  s.queue_wait);
}

// ---------------------------------------------------------------- the engine thread

void ScheduledGenerator::Publish(Job& job) {
  {
    std::lock_guard<std::mutex> lk(job.mu);
    job.done = true;
  }
  job.cv.notify_all();
}

void ScheduledGenerator::PublishFinished() {
  for (Job* j : unpublished_) Publish(*j);
  unpublished_.clear();
}

void ScheduledGenerator::FinishJob(Job& job, const std::string& error) {
  if (job.finished) return;
  job.finished = true;
  if (job.spec) spec_active_ = false;
  bool completed = false;
  {
    std::lock_guard<std::mutex> lk(job.mu);
    if (error.empty() && !job.result.cancelled && !job.pending.empty()) job.text += job.pending;
    job.pending.clear();
    job.result.error = error;
    job.result.spec = job.spec_stats;
    completed = error.empty() && !job.result.cancelled;
    if (kTrace)
      std::fprintf(stderr, "sched: finish %llu tokens %zu error '%s' cancelled %d\n",
                   static_cast<unsigned long long>(job.id), job.tokens.size(), error.c_str(),
                   job.result.cancelled ? 1 : 0);
  }
  if (completed && !job.work) {
    Stored s;
    {
      std::lock_guard<std::mutex> jl(job.mu);
      s = {job.text, job.tokens, job.result, job.counted};
    }
    std::lock_guard<std::mutex> lk(mu_);
    replay_[job.id] = std::move(s);
    replay_order_.push_back(job.id);
    while (replay_order_.size() > opt_.sched.max_keys) {
      replay_.erase(replay_order_.front());
      replay_order_.pop_front();
    }
  }
  // Answered once the scheduler has let the request go (after Done): a client's next turn must not find its previous
  // one still holding its principal's slot.
  unpublished_.push_back(&job);
}

bool ScheduledGenerator::Begin(Job& job, std::string* error) {
  // MTP only for a request that starts alone (docs/SCHEDULER.md 5: B x MTP is a follow-up).
  bool alone = true;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& [id, j] : jobs_)
      if (id != job.id && !j->finished) alone = false;
  }
  job.spec = job.spec_ok && alone && !spec_active_;
  const std::uint32_t positions = std::min<std::uint32_t>(
      max_context_, static_cast<std::uint32_t>(job.ids.size()) + job.budget + (job.spec ? backend_->MaxDrafts() : 0));
  std::size_t reused = 0;
  std::uint64_t bound = 0;
  if (!backend_->Begin(job.id, job.ids, job.stable, positions, job.spec, spec_active_, &job.spec_stats, &reused, &bound,
                       error))
    return false;
  job.begun = true;
  if (job.spec) spec_active_ = true;
  {
    std::lock_guard<std::mutex> lk(job.mu);
    job.result.cached_tokens = reused;
  }
  sched_.Bind(job.id, bound, reused);
  return true;
}

void ScheduledGenerator::RunSlice(Job& job, std::uint32_t tokens, Scheduler::Progress* p) {
  std::string error;
  if (!job.begun && !Begin(job, &error)) {
    FinishJob(job, error);
    p->finished = true;
    return;
  }
  std::size_t ran = 0;
  bool done = false;
  if (!backend_->PrefillSlice(job.id, tokens, &ran, &done, &error)) {
    backend_->Abandon(job.id);
    FinishJob(job, error);
    p->finished = true;
    return;
  }
  p->prefilled = ran;
  p->prefill_done = done;
  if (done) {
    std::vector<text::TokenId> as_tok(job.ids.begin(), job.ids.end());
    job.state = std::make_unique<sampling::SamplerState>(job.sampling, as_tok);
    if (job.spec) chooser_.Reset();
    job.run = std::make_unique<DecodeRun>(job.ids.back(), job.budget, job.spec ? &chooser_ : nullptr,
                                          job.spec ? &job.spec_stats : nullptr);
    // The first token belongs to the prefill's unit (~40 ms): TTFT is then the request's own prefill and the slices
    // fair queuing ordered before it, not every other request's prefill too.
    if (!opt_.first_token_in_slice) return;
    std::vector<Scheduler::Progress> first;
    RunDecodeUnit({FindShared(job)}, 1, &first);
    p->decoded = first.empty() ? 0 : first[0].decoded;
    p->finished = !first.empty() && first[0].finished;
    p->completed = !first.empty() && first[0].completed;
  }
}

void ScheduledGenerator::RunDecodeUnit(const std::vector<std::shared_ptr<Job>>& jobs, std::uint32_t steps,
                                       std::vector<Scheduler::Progress>* ps) {
  const auto emit = [this](Job& job, text::TokenId t) {
    job.state->Accept(t);
    bool more = true;
    {
      std::lock_guard<std::mutex> lk(job.mu);
      job.tokens.push_back(static_cast<std::int32_t>(t));
      job.pending += backend_->TokenBytes(t);
      const std::size_t n = CompleteUtf8Prefix(job.pending);
      job.text.append(job.pending, 0, n);
      job.pending.erase(0, n);
      job.result.completion_tokens = job.tokens.size();
      more = !job.consumer_stop;
    }
    job.cv.notify_all();
    return more;
  };
  const auto io_for = [&](Job& job) {
    DecodeIo io;
    io.ran = [this, &job](std::int32_t t) { backend_->Ran(job.id, t); };
    io.is_stop = [this](text::TokenId t) { return backend_->IsStop(t); };
    io.emit = [&emit, &job](text::TokenId t) { return emit(job, t); };
    return io;
  };
  std::vector<std::uint32_t> before(jobs.size());
  for (std::size_t i = 0; i < jobs.size(); ++i) before[i] = jobs[i]->run ? jobs[i]->run->outcome().completion : 0;
  std::string error;
  if (jobs.size() == 1) {
    Job& job = *jobs[0];
    DecodeIo io = io_for(job);
    io.step = [&](std::int32_t token, std::uint32_t drafts, DecodeCycle* c, std::string* err) {
      return backend_->Step(job.id, token, drafts, job.sampling, job.state.get(), c, err);
    };
    if (job.run)
      for (std::uint32_t s = 0; s < steps && !job.run->done(); ++s) job.run->Cycle(io, job.spec);
  } else {
    for (std::uint32_t s = 0; s < steps; ++s) {
      std::vector<Job*> live;
      for (const auto& j : jobs)
        if (!j->finished && j->run && !j->run->done()) live.push_back(j.get());
      if (live.empty()) break;
      std::vector<ScheduledBackend::BatchItem> items(live.size());
      for (std::size_t i = 0; i < live.size(); ++i)
        items[i] = {live[i]->id, live[i]->run->token(), &live[i]->sampling, live[i]->state.get(), 0};
      if (!backend_->StepBatch(items, &error)) {
        for (Job* j : live) {
          backend_->Abandon(j->id);
          FinishJob(*j, error);
        }
        break;
      }
      for (std::size_t i = 0; i < live.size(); ++i) {
        DecodeIo io = io_for(*live[i]);
        io.step = [tok = items[i].out](std::int32_t, std::uint32_t, DecodeCycle* c, std::string*) {
          c->tokens.assign(1, tok);
          return true;
        };
        live[i]->run->Cycle(io, false);
      }
    }
  }
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    Job& job = *jobs[i];
    Scheduler::Progress p{job.id};
    if (job.run) p.decoded = job.run->outcome().completion - before[i];
    if (!job.finished && job.run && job.run->done()) {
      const DecodeOutcome out = job.run->Finish();
      if (!out.error.empty()) {
        backend_->Abandon(job.id);
        FinishJob(job, out.error);
      } else {
        {
          std::lock_guard<std::mutex> lk(job.mu);
          job.result.completion_tokens = out.completion;
          job.result.length_limited = out.length_limited;
        }
        backend_->Finish(job.id);
        {
          const std::size_t retained = backend_->RetainedPrefix(job.ids);
          std::lock_guard<std::mutex> lk(job.mu);
          job.result.cache_write_tokens = retained > job.result.cached_tokens ? retained - job.result.cached_tokens : 0;
        }
        FinishJob(job, "");
      }
    }
    p.finished = job.finished;
    p.completed = job.finished && job.result.error.empty() && !job.result.cancelled;
    ps->push_back(p);
  }
}

void ScheduledGenerator::Sync() {
  if (backend_->Generation() == synced_generation_) return;
  synced_generation_ = backend_->Generation();
  sched_.SyncRetained(backend_->Retained());
}

void ScheduledGenerator::Fence(const std::vector<Scheduler::Id>& ids) {
  for (auto id : ids) {
    auto job = Find(id);
    if (!job) continue;
    if (backend_->HasLease(id)) backend_->Abandon(id);
    backend_->Fenced(id);
    // Its prefill progress may be parked for a retry; its decoded tokens are discarded.
    if (job->run) fenced_decoded_ += job->run->outcome().completion;
    {
      std::lock_guard<std::mutex> lk(job->mu);
      job->result.cancelled = true;
    }
    FinishJob(*job, "");
    {
      std::lock_guard<std::mutex> lk(mu_);
      held_.push_back(job);
      jobs_.erase(id);
    }
  }
}

void ScheduledGenerator::Loop() {
  while (!stop_) {
    std::deque<std::function<void()>> posts;
    {
      std::lock_guard<std::mutex> lk(mu_);
      posts.swap(posts_);
    }
    for (auto& f : posts) f();
    Sync();
    // The cache's own memory loop must not evict a prefix an admitted request plans to reuse.
    std::set<std::uint64_t> pins;
    for (const auto& v : sched_.Requests())
      if (!v.started && v.planned) pins.insert(v.planned);
    backend_->SetPins(pins);
    if (opt_.before_next) {
      while (!stop_ && !opt_.before_next()) {
        Fence(sched_.Boundary(NowMs()));
        PublishFinished();
        held_.clear();
        Sync();
      }
      if (stop_) break;
    }
    auto unit = sched_.Next(NowMs());
    Fence(unit.fenced);
    PublishFinished();  // the scheduler already let them go
    held_.clear();
    Sync();
    if (unit.kind == Scheduler::Kind::kNone) {
      std::unique_lock<std::mutex> lk(mu_);
      // An admission wakes the engine at once (without kick_ the predicate ignored it: up to 20 ms added to every
      // request that found the engine idle, found by D2's deciders: 175 ms scheduled vs 153 ms concurrent).
      wake_.wait_for(lk, std::chrono::milliseconds(20), [&] { return stop_.load() || !posts_.empty() || kick_; });
      kick_ = false;
      continue;
    }
    const auto t0_ms = NowMs();  // the test clock's unit start (it may move while before_unit holds the unit)
    if (opt_.before_unit) opt_.before_unit(unit);
    if (kTrace) {
      std::fprintf(stderr, "sched %lld: unit %d [", static_cast<long long>(NowMs()), static_cast<int>(unit.kind));
      for (auto id : unit.ids) std::fprintf(stderr, " %llu", static_cast<unsigned long long>(id));
      std::fprintf(stderr, " ] slice %u steps %u\n", unit.slice_tokens, unit.steps);
    }
    const auto t0 = Clock::now();
    std::vector<Scheduler::Progress> ps;
    std::vector<std::shared_ptr<Job>> jobs;
    for (auto id : unit.ids)
      if (auto j = Find(id)) jobs.push_back(j);
      else ps.push_back({id, 0, 0, false, true, false});  // vanished (cannot happen): let it go
    if (unit.kind == Scheduler::Kind::kSlice && !jobs.empty()) {
      Scheduler::Progress p{jobs[0]->id};
      RunSlice(*jobs[0], unit.slice_tokens, &p);
      ps.push_back(p);
    } else if (unit.kind == Scheduler::Kind::kBatch && !jobs.empty()) {
      RunDecodeUnit(jobs, unit.steps, &ps);
    } else {
      for (const auto& j : jobs) {
        if (j->work) j->work();
        FinishJob(*j, j->work ? "" : "a decider unit without an evaluation");
        ps.push_back({j->id, 0, 0, false, true, static_cast<bool>(j->work)});
      }
    }
    const auto elapsed = opt_.now_ms ? NowMs() - t0_ms : static_cast<std::int64_t>(Ms(t0, Clock::now()));
    {
      std::lock_guard<std::mutex> lk(stats_mu_);
      const double sec = static_cast<double>(elapsed) * 1e-3;
      (unit.kind == Scheduler::Kind::kSlice   ? stats_.slice
       : unit.kind == Scheduler::Kind::kBatch ? stats_.batch
                                              : stats_.decide)
          .Observe(sec);
      const auto start_ms = NowMs() - elapsed;
      for (const auto& j : jobs)
        if (!j->waited && !j->work) {
          j->waited = true;
          stats_.queue_wait.Observe(static_cast<double>(std::max<std::int64_t>(0, start_ms - j->admitted_ms)) * 1e-3);
        }
    }
    sched_.Done(unit, ps, NowMs(), std::max<std::int64_t>(1, elapsed));
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (const auto& p : ps)
        if (p.finished) {
          if (auto it = jobs_.find(p.id); it != jobs_.end()) held_.push_back(it->second);  // alive until published
          jobs_.erase(p.id);
        }
    }
    PublishFinished();
    held_.clear();
    Sync();
  }
  // Stopping: every request still held ends cancelled.
  std::map<Scheduler::Id, std::shared_ptr<Job>> left;
  {
    std::lock_guard<std::mutex> lk(mu_);
    left.swap(jobs_);
  }
  for (auto& [id, job] : left) {
    if (backend_->HasLease(id)) backend_->Abandon(id);
    {
      std::lock_guard<std::mutex> lk(job->mu);
      job->result.cancelled = true;
    }
    FinishJob(*job, "");
  }
  PublishFinished();
}

}  // namespace halo::server
