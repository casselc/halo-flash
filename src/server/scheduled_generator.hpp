#ifndef HALO_SERVER_SCHEDULED_GENERATOR_HPP_
#define HALO_SERVER_SCHEDULED_GENERATOR_HPP_

// G3 M3 (docs/SCHEDULER.md): the TextGenerator behind the fair scheduler (server/scheduler.hpp).
//
// One engine thread owns the backend (the engine and its session cache) and runs one unit at a time: a prefill slice
// of one request (a session-cache lease prefilled in one engine call), or a decode unit over every decoding request
// (B = 1: a few steps or MTP cycles of server/decode_loop; B >= 2: StepBatch steps, each entry byte-identical to
// stepping it alone, so batching never changes a token). HTTP threads admit (Admit: idempotency, the principal's
// limit, the memory ledger; a refusal is a 429 before anything is sent) and then wait for their request's output,
// polling the client's departure (and with it the SSE keepalive) every 20 ms. A request whose client left is fenced
// at the next unit boundary: a prefill's slice-aligned progress is parked as an exact checkpoint for its retry.
// A finished request is answered only after the scheduler let it go (Scheduler::Done), so a client's next turn never
// finds its previous one still holding the principal's slot.
//
// MTP runs only when a request starts alone and only while it decodes alone (B = 1; docs/SCHEDULER.md 5).

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "core/sampling.hpp"
#include "engine/mtp_depth.hpp"
#include "server/decode_loop.hpp"
#include "server/metrics.hpp"
#include "server/openai_service.hpp"
#include "server/scheduler.hpp"
#include "text/tokenizer.hpp"

namespace halo::server {

/// The engine side of a ScheduledGenerator. Every method but the token ones runs on the engine thread only.
class ScheduledBackend {
 public:
  using Id = Scheduler::Id;
  virtual ~ScheduledBackend() = default;
  // ---- tokens (any thread) ----
  /// The prompt's token ids and its stable prefix (where the generation prompt starts); false: no tokenizer.
  virtual bool Encode(const std::string& prompt, std::vector<std::int32_t>* ids, std::size_t* stable) const = 0;
  [[nodiscard]] virtual std::string TokenBytes(text::TokenId t) const = 0;
  [[nodiscard]] virtual bool IsStop(text::TokenId t) const = 0;
  [[nodiscard]] virtual bool HasTokenizer() const = 0;
  [[nodiscard]] virtual double BytesPerToken() const = 0;  ///< ledger bytes a position costs
  [[nodiscard]] virtual bool CanSpeculate() const { return false; }
  /// Positions a speculative cycle may write past the last emitted one (reserved for a speculating request).
  [[nodiscard]] virtual std::uint32_t MaxDrafts() const { return 0; }
  // ---- engine thread ----
  /// Binds request `id`'s live state for `prompt` (reuse, memory); nothing prefilled yet. `spec`: it will decode
  /// speculatively (the draft block follows it); `keep_tracking`: another request does, leave the draft block alone.
  virtual bool Begin(Id id, std::span<const std::int32_t> prompt, std::size_t stable, std::uint32_t positions, bool spec,
                     bool keep_tracking, SpecDecodeStats* spec_stats, std::size_t* reused, std::uint64_t* bound_key,
                     std::string* error) = 0;
  /// One prefill slice: at most `max_tokens` prompt tokens in one engine call.
  virtual bool PrefillSlice(Id id, std::uint32_t max_tokens, std::size_t* ran, bool* done, std::string* error) = 0;
  /// B = 1: one step of `id` (`drafts` 0) or a speculative cycle (greedy when `state` is null).
  virtual bool Step(Id id, std::int32_t token, std::uint32_t drafts, const sampling::SamplingConfig& sampling,
                    sampling::SamplerState* state, DecodeCycle* cycle, std::string* error) = 0;
  struct BatchItem {
    Id id = 0;
    std::int32_t token = 0;
    const sampling::SamplingConfig* sampling = nullptr;
    sampling::SamplerState* state = nullptr;
    std::int32_t out = 0;  ///< the sampled next token
  };
  /// B >= 2: one ordinary step of each, every entry byte-identical to its Step alone.
  virtual bool StepBatch(std::vector<BatchItem>& items, std::string* error) = 0;
  virtual void Ran(Id id, std::int32_t token) = 0;
  virtual void Finish(Id id) = 0;
  virtual void Abandon(Id id) = 0;
  /// The client left: the live state goes; a prefill's exact progress is kept (returned) for a retry.
  virtual std::optional<Scheduler::Retained> Park(Id id) = 0;
  virtual void Evict(Scheduler::Key key) = 0;
  [[nodiscard]] virtual bool HasLease(Id id) const = 0;
  [[nodiscard]] virtual std::uint64_t Generation() const = 0;  ///< bumped when the retained set changes
  /// The longest prefix of `prompt` a retained state holds whole (SessionCache::RetainedPrefix); after Finish, minus
  /// the request's reused tokens, what it added to the cache (Anthropic's cache_creation_input_tokens).
  [[nodiscard]] virtual std::size_t RetainedPrefix(std::span<const std::int32_t> prompt) const {
    (void)prompt;
    return 0;
  }
  [[nodiscard]] virtual std::vector<Scheduler::Retained> Retained() const = 0;
  virtual void SetPins(const std::set<std::uint64_t>& keys) = 0;
  /// Notifications (optional): `id` was admitted (an HTTP thread, under the generator's lock); `id` was fenced (its
  /// client left; the engine thread, after the backend's Park when it had started).
  virtual void Admitted(Id id, std::span<const std::int32_t> prompt) {
    (void)id;
    (void)prompt;
  }
  virtual void Fenced(Id id) { (void)id; }
};

struct ScheduledOptions {
  Scheduler::Options sched;
  /// Ledger bytes a request costs besides its positions: recurrent state and a fork's own pages (SESSIONS 2, 7:
  /// 118 MB + 151 MB), rounded up.
  std::uint64_t fixed_bytes = 320ull << 20;
  /// Test hook: the engine thread calls this before each unit.
  std::function<void(const Scheduler::Unit&)> before_unit;
};

class ScheduledGenerator : public TextGenerator {
 public:
  /// Starts the engine thread over `backend` (server/flash_next_backend.hpp: MakeScheduledFlashNext; tests: a fake).
  /// `depth`: the MTP depth chooser's options (when the backend can speculate).
  ScheduledGenerator(std::unique_ptr<ScheduledBackend> backend, std::uint32_t max_context,
                     engine::MtpDepthOptions depth = {}, ScheduledOptions options = {});
  ~ScheduledGenerator() override;

  Admission Admit(const Params& params) override;
  Result Generate(const Params& params, const std::function<bool(std::string_view)>& on_text) override;
  [[nodiscard]] std::optional<std::uint64_t> CountTokens(std::string_view prompt) const override;
  /// Token level: admits (unless `params.ticket` holds an admission) and generates; `on_token` gets each token.
  Result GenerateIds(std::span<const std::int32_t> prompt, std::size_t stable, const Params& params,
                     const std::function<bool(text::TokenId)>& on_token);
  Admission AdmitIds(std::span<const std::int32_t> prompt, std::size_t stable, const Params& params);

  /// Runs `f()` on the engine thread between units (SR4 admin over the backend's session cache).
  template <class F>
  auto OnEngineThread(F&& f) {
    using R = decltype(f());
    auto task = std::make_shared<std::packaged_task<R()>>([&] { return f(); });
    auto fut = task->get_future();
    Post([task] { (*task)(); });
    return fut.get();
  }
  /// G3 M4: runs `work` (a decider evaluation, on its own model) as a scheduler unit on the engine thread and
  /// returns once it ran; false with `error` when it did not (the generator stopped). DeciderGate's kScheduled.
  bool RunDecider(const std::function<void()>& work, std::string* error);
  /// The scheduler's series for /metrics (Metrics::AddSection): halo:sched_* gauges (queued, prefilling, decoding,
  /// ledger, decider credit), counters (Scheduler::Counters) and unit / queue-wait histograms.
  void RenderMetrics(std::string& out, const std::string& labels) const;
  [[nodiscard]] ScheduledBackend& backend() { return *backend_; }
  [[nodiscard]] const Scheduler& scheduler() const { return sched_; }

 private:
  struct Job;
  void Post(std::function<void()> task);
  void Loop();
  bool Begin(Job& job, std::string* error);
  void RunSlice(Job& job, std::uint32_t tokens, Scheduler::Progress* p);
  void RunDecodeUnit(const std::vector<std::shared_ptr<Job>>& jobs, std::uint32_t steps,
                     std::vector<Scheduler::Progress>* ps);
  void FinishJob(Job& job, const std::string& error);
  void Publish(Job& job);
  void PublishFinished();
  void Sync();
  std::shared_ptr<Job> Find(Scheduler::Id id);
  std::shared_ptr<Job> FindShared(Job& job);
  Result Wait(const std::shared_ptr<Job>& job, std::uint64_t gen, const Params& params,
              const std::function<bool(text::TokenId)>* on_token, const std::function<bool(std::string_view)>* on_text);
  std::int64_t NowMs() const;

  std::unique_ptr<ScheduledBackend> backend_;
  const std::uint32_t max_context_;
  ScheduledOptions opt_;
  engine::MtpDepthChooser chooser_;
  Scheduler sched_;

  std::mutex mu_;  // jobs_, posts_, replay_ (lock order: mu_, then a job's mu or the scheduler's)
  std::condition_variable wake_;
  bool kick_ = false;  // an admission since the engine thread last looked (guarded by mu_)
  std::map<Scheduler::Id, std::shared_ptr<Job>> jobs_;
  std::deque<std::function<void()>> posts_;
  struct Stored {
    std::string text;
    std::vector<std::int32_t> tokens;
    Result result;
    std::shared_ptr<std::atomic<bool>> counted;  ///< the job's: whether its completion tokens were counted
  };
  std::map<Scheduler::Id, Stored> replay_;  // finished answers (kept while their keys may replay)
  /// Tokens decoded by fenced attempts, by idempotency key, until a new generation for the key is admitted
  /// (Halo-Discarded-Tokens). Bounded: past kMaxDiscardedKeys the oldest key is forgotten.
  std::map<std::string, std::uint64_t> discarded_;
  std::deque<std::string> discarded_order_;
  static constexpr std::size_t kMaxDiscardedKeys = 4096;
  std::deque<Scheduler::Id> replay_order_;
  std::uint64_t synced_generation_ = ~0ull;
  std::chrono::steady_clock::time_point epoch_;
  double bytes_per_token_ = 0;
  bool spec_active_ = false;                // a request decodes with MTP (engine thread)
  std::vector<Job*> unpublished_;           // finished, answered after the scheduler's Done (engine thread)
  std::vector<std::shared_ptr<Job>> held_;  // ... kept alive until then
  struct UnitStats {
    Metrics::Histogram slice{{0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0}, {}};
    Metrics::Histogram batch{{0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0}, {}};
    Metrics::Histogram decide{{0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5}, {}};
    Metrics::Histogram queue_wait{{0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0}, {}};
  };
  mutable std::mutex stats_mu_;
  UnitStats stats_;  // guarded by stats_mu_
  std::atomic<bool> stop_{false};
  /// Tokens decoded by requests that were then fenced: generated work no answer's usage reports.
  std::atomic<std::uint64_t> fenced_decoded_{0};
  std::thread thread_;  // last: started once everything else is constructed
};

}  // namespace halo::server

#endif  // HALO_SERVER_SCHEDULED_GENERATOR_HPP_
