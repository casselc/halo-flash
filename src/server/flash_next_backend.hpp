#ifndef HALO_SERVER_FLASH_NEXT_BACKEND_HPP_
#define HALO_SERVER_FLASH_NEXT_BACKEND_HPP_

// G3 M3: the ScheduledGenerator's backend on Flash-Next (server/scheduled_generator.hpp): the engine and its session
// cache, one cache lease per scheduled request.

#include <memory>
#include <vector>

#include "engine/flash_next_gpu.hpp"
#include "server/flash_next_generator.hpp"
#include "server/scheduled_generator.hpp"
#include "server/session_cache.hpp"
#include "text/tokenizer.hpp"

namespace halo::server {

class FlashNextBackend : public ScheduledBackend {
 public:
  /// `engine` (and `tokenizer`, when given) must outlive the backend. Token-level only without a tokenizer:
  /// generation stops at `stops`.
  FlashNextBackend(engine::FlashNextGpu& engine, const text::Tokenizer* tokenizer, std::vector<text::TokenId> stops,
                   SessionCacheOptions cache, SpecOptions spec);
  bool Encode(const std::string& prompt, std::vector<std::int32_t>* ids, std::size_t* stable) const override;
  [[nodiscard]] std::string TokenBytes(text::TokenId t) const override { return tok_ ? std::string(tok_->TokenBytes(t)) : std::string(); }
  [[nodiscard]] bool IsStop(text::TokenId t) const override;
  [[nodiscard]] bool HasTokenizer() const override { return tok_ != nullptr; }
  [[nodiscard]] double BytesPerToken() const override { return bytes_per_token_; }
  [[nodiscard]] bool CanSpeculate() const override { return spec_.enabled; }
  [[nodiscard]] std::uint32_t MaxDrafts() const override { return spec_.enabled ? engine::FlashNextGpu::kMaxDrafts : 0; }
  bool Begin(Id id, std::span<const std::int32_t> prompt, std::size_t stable, std::uint32_t positions, bool spec,
             bool keep_tracking, SpecDecodeStats* spec_stats, std::size_t* reused, std::uint64_t* bound_key,
             std::string* error) override;
  bool PrefillSlice(Id id, std::uint32_t max_tokens, std::size_t* ran, bool* done, std::string* error) override;
  bool Step(Id id, std::int32_t token, std::uint32_t drafts, const sampling::SamplingConfig& sampling,
            sampling::SamplerState* state, DecodeCycle* cycle, std::string* error) override;
  bool StepBatch(std::vector<BatchItem>& items, std::string* error) override;
  void Ran(Id id, std::int32_t token) override { cache_.Ran(id, token); }
  void Finish(Id id) override { cache_.Finish(id); }
  void Abandon(Id id) override { cache_.Abandon(id); }
  std::optional<Scheduler::Retained> Park(Id id) override;
  void Evict(Scheduler::Key key) override { (void)cache_.EvictKey(key); }
  [[nodiscard]] bool HasLease(Id id) const override { return cache_.HasLease(id); }
  [[nodiscard]] std::uint64_t Generation() const override { return cache_.generation(); }
  [[nodiscard]] std::vector<Scheduler::Retained> Retained() const override;
  void SetPins(const std::set<std::uint64_t>& keys) override { cache_.SetPins(keys); }
  /// The session cache (SR4 admin; on the engine thread only: ScheduledGenerator::OnEngineThread).
  SessionCache& cache() { return cache_; }

 private:
  engine::FlashNextGpu& engine_;
  const text::Tokenizer* tok_;
  std::vector<text::TokenId> stops_;
  text::TokenId im_start_ = text::kNoToken;
  SessionCache cache_;
  SpecOptions spec_;
  double bytes_per_token_ = 0;
};

/// halo_serve's generator: Flash-Next behind the G3 scheduler. `backend_out` (optional) gets the backend (owned by
/// the generator), for the SR4 admin routes.
std::unique_ptr<ScheduledGenerator> MakeScheduledFlashNext(engine::FlashNextGpu& engine, const text::Tokenizer* tokenizer,
                                                           std::vector<text::TokenId> stops, std::uint32_t max_context,
                                                           SessionCacheOptions cache, SpecOptions spec,
                                                           ScheduledOptions options,
                                                           FlashNextBackend** backend_out = nullptr);

}  // namespace halo::server

#endif  // HALO_SERVER_FLASH_NEXT_BACKEND_HPP_
