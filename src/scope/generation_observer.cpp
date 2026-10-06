#include "scope/generation_observer.hpp"

#include <algorithm>
#include <cmath>

namespace halo::scope {

namespace {

// Nearest-rank percentile of `v` (sorted in place).
std::uint32_t Percentile(std::vector<std::uint32_t>& v, double p) {
  if (v.empty()) return 0;
  const auto rank = static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(v.size())));
  const std::size_t i = std::min(v.size(), std::max<std::size_t>(rank, 1)) - 1;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i), v.end());
  return v[i];
}

}  // namespace

Observation GenerationObserver::ToObservation(const server::GenerationRecord& r) {
  Observation o;
  o.trace_id = r.trace_id.empty() ? r.request_id : r.trace_id;
  o.span_id = r.request_id;
  o.start_ns = r.start_ns;
  o.end_ns = r.end_ns;
  o.completion_start_ns = r.first_text_ns;
  o.type = "GENERATION";
  o.name = r.api;
  o.user_id = r.user;
  o.session_id = r.session_id;
  o.status_code = r.error.empty() ? "OK" : "ERROR";
  o.level = r.error.empty() ? "DEFAULT" : "ERROR";
  o.provider = "halo-flash";
  o.model = r.model;
  o.variant_id = r.variant_id;
  o.prompt_name = r.prompt_name;
  o.prompt_version = r.prompt_version;
  o.input_tokens = r.prompt_tokens;
  o.output_tokens = r.completion_tokens;
  o.cached_tokens = r.cached_tokens;
  o.usage_details = {{"input", r.prompt_tokens}, {"output", r.completion_tokens}, {"cache_read", r.cached_tokens}};
  o.tool_name = r.tool_names.empty() ? std::string() : r.tool_names.front();
  o.tool_call_ids = r.tool_call_ids;
  std::vector<std::uint32_t> gaps = r.gaps_us;
  o.tpot_p50_us = Percentile(gaps, 50);
  o.tpot_p90_us = Percentile(gaps, 90);
  o.tpot_p99_us = Percentile(gaps, 99);
  o.temperature = r.sampling.temperature;
  o.top_p = r.sampling.top_p;
  o.min_p = r.sampling.min_p;
  o.top_k = static_cast<std::uint32_t>(std::max(r.sampling.top_k, 0));
  o.seed = r.sampling.seed < 0 ? 0 : static_cast<std::uint64_t>(r.sampling.seed);
  o.max_tokens = r.max_tokens;
  o.finish_reason = r.finish_reason;
  return o;
}

void GenerationObserver::OnGeneration(const server::GenerationRecord& r) noexcept {
  try {
    (void)a_.RecordObservation(ToObservation(r), r.input, r.output);
  } catch (...) {
    // Recording must never take the server down; a lost record is a gap,
    // not an error the client should see.
  }
}

}  // namespace halo::scope
