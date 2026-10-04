#ifndef HALO_SCOPE_RECORDS_HPP_
#define HALO_SCOPE_RECORDS_HPP_

// Typed halo-scope records and their RowBinary encoding. Each Encode*
// writes exactly the column order of the Null-engine staging table
// (`<table>_in` in schema.cpp) with plain types; a materialized view casts
// each block into the typed MergeTree table. Each Decode* reads the typed
// table's columns back in the same order (LowCardinality(String) reads as
// String in RowBinary), so Decode(Select(Insert(Encode(x)))) == x.

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "scope/rowbinary.hpp"

namespace halo::scope {

/// The tables a ring record can target.
enum class Table : std::uint8_t { kObservations = 0, kPayloads = 1, kCount = 2 };

/// One row of llm_observations: casselc/observability's typed llm_spans
/// columns plus the engine-only columns of PLAN.md P12.
struct Observation {
  // identity and time (nanoseconds since the Unix epoch, UTC)
  std::string project;
  std::string trace_id;
  std::string span_id;
  std::string parent_span_id;
  std::int64_t start_ns = 0;
  std::int64_t end_ns = 0;
  std::int64_t completion_start_ns = 0;  // first output token (TTFT)
  // classification
  std::string type;  // GENERATION, SPAN, EVENT, DECISION
  std::string name;
  std::string user_id;
  std::string session_id;
  std::vector<std::string> tags;
  std::string status_code;
  std::string level;
  // model
  std::string provider;
  std::string model;
  std::string variant_id;
  std::string adapter_id;
  // usage
  std::uint64_t input_tokens = 0;
  std::uint64_t output_tokens = 0;
  std::uint64_t cached_tokens = 0;  // prefix-cache hit tokens
  std::vector<std::pair<std::string, std::uint64_t>> usage_details;
  // prompt and tools
  std::string prompt_name;
  std::uint32_t prompt_version = 0;
  std::string tool_name;
  std::vector<std::string> tool_call_ids;
  // content references into llm_payloads
  std::vector<rb::Hash16> system_ref;
  std::vector<rb::Hash16> input_refs;
  std::vector<rb::Hash16> output_refs;
  std::uint64_t input_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint8_t truncated = 0;
  // decode timing and speculation
  std::uint32_t tpot_p50_us = 0;
  std::uint32_t tpot_p90_us = 0;
  std::uint32_t tpot_p99_us = 0;
  std::uint32_t mtp_drafted = 0;
  std::uint32_t mtp_accepted = 0;
  // sampler
  float temperature = 0;
  float top_p = 0;
  float min_p = 0;
  std::uint32_t top_k = 0;
  std::uint64_t seed = 0;
  std::uint32_t max_tokens = 0;
  std::string finish_reason;
  std::string decision_answer;  // /v1/systemone answer, empty otherwise

  bool operator==(const Observation&) const = default;
};

/// One content-addressed payload row of llm_payloads.
struct Payload {
  std::string project;
  rb::Hash16 hash{};
  std::uint32_t bytes = 0;
  std::string content;
  bool operator==(const Payload&) const = default;
};

void EncodeObservation(const Observation& o, std::string& out);
Observation DecodeObservation(rb::Reader& r);
void EncodePayload(const Payload& p, std::string& out);
Payload DecodePayload(rb::Reader& r);

/// Column lists in staging/typed order (comma-separated, for INSERT/SELECT).
extern const char* const kObservationColumns;
extern const char* const kPayloadColumns;

}  // namespace halo::scope

#endif  // HALO_SCOPE_RECORDS_HPP_
