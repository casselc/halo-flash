#include "scope/records.hpp"

#include <blake3.h>

#include <algorithm>

// HALO_SCOPE_MUTANT selects a deliberately broken encoder for the positive
// controls in tests/scope (never set in a normal build):
//   1: `name` is cut to 255 bytes (a large-string bug)
//   2: usage_details values are narrowed to 32 bits (a large-count bug)
#ifndef HALO_SCOPE_MUTANT
#define HALO_SCOPE_MUTANT 0
#endif

namespace halo::scope {

const char* const kObservationColumns =
    "project, trace_id, span_id, parent_span_id, start_time, end_time, completion_start_time, "
    "type, name, user_id, session_id, tags, status_code, level, "
    "provider, model, variant_id, adapter_id, "
    "input_tokens, output_tokens, cached_tokens, usage_details, "
    "prompt_name, prompt_version, tool_name, tool_call_ids, "
    "system_ref, input_refs, output_refs, input_bytes, output_bytes, truncated, "
    "tpot_p50_us, tpot_p90_us, tpot_p99_us, mtp_drafted, mtp_accepted, "
    "temperature, top_p, min_p, top_k, seed, max_tokens, finish_reason, decision_answer";

const char* const kPayloadColumns = "project, hash, bytes, content";

namespace {

void StrArray(std::string& out, const std::vector<std::string>& v) {
  rb::Varint(out, v.size());
  for (const auto& s : v) rb::Str(out, s);
}

void HashArray(std::string& out, const std::vector<rb::Hash16>& v) {
  rb::Varint(out, v.size());
  for (const auto& h : v) rb::FixedStr16(out, h);
}

std::vector<std::string> ReadStrArray(rb::Reader& r) {
  std::vector<std::string> v(r.Varint());
  for (auto& s : v) s = r.Str();
  return v;
}

std::vector<rb::Hash16> ReadHashArray(rb::Reader& r) {
  std::vector<rb::Hash16> v(r.Varint());
  for (auto& h : v) h = r.FixedStr16();
  return v;
}

}  // namespace

void EncodeObservation(const Observation& o, std::string& out) {
  rb::Str(out, o.project);
  rb::Str(out, o.trace_id);
  rb::Str(out, o.span_id);
  rb::Str(out, o.parent_span_id);
  rb::Fixed<std::int64_t>(out, o.start_ns);  // DateTime64(9) is Int64 ns
  rb::Fixed<std::int64_t>(out, o.end_ns);
  rb::Fixed<std::int64_t>(out, o.completion_start_ns);
  rb::Str(out, o.type);
#if HALO_SCOPE_MUTANT == 1
  rb::Str(out, std::string_view(o.name).substr(0, 255));
#else
  rb::Str(out, o.name);
#endif
  rb::Str(out, o.user_id);
  rb::Str(out, o.session_id);
  StrArray(out, o.tags);
  rb::Str(out, o.status_code);
  rb::Str(out, o.level);
  rb::Str(out, o.provider);
  rb::Str(out, o.model);
  rb::Str(out, o.variant_id);
  rb::Str(out, o.adapter_id);
  rb::Fixed<std::uint64_t>(out, o.input_tokens);
  rb::Fixed<std::uint64_t>(out, o.output_tokens);
  rb::Fixed<std::uint64_t>(out, o.cached_tokens);
  rb::Varint(out, o.usage_details.size());  // Map = Array(Tuple(K, V))
  for (const auto& [k, v] : o.usage_details) {
    rb::Str(out, k);
#if HALO_SCOPE_MUTANT == 2
    rb::Fixed<std::uint64_t>(out, static_cast<std::uint32_t>(v));
#else
    rb::Fixed<std::uint64_t>(out, v);
#endif
  }
  rb::Str(out, o.prompt_name);
  rb::Fixed<std::uint32_t>(out, o.prompt_version);
  rb::Str(out, o.tool_name);
  StrArray(out, o.tool_call_ids);
  HashArray(out, o.system_ref);
  HashArray(out, o.input_refs);
  HashArray(out, o.output_refs);
  rb::Fixed<std::uint64_t>(out, o.input_bytes);
  rb::Fixed<std::uint64_t>(out, o.output_bytes);
  rb::Fixed<std::uint8_t>(out, o.truncated);
  rb::Fixed<std::uint32_t>(out, o.tpot_p50_us);
  rb::Fixed<std::uint32_t>(out, o.tpot_p90_us);
  rb::Fixed<std::uint32_t>(out, o.tpot_p99_us);
  rb::Fixed<std::uint32_t>(out, o.mtp_drafted);
  rb::Fixed<std::uint32_t>(out, o.mtp_accepted);
  rb::Fixed<float>(out, o.temperature);
  rb::Fixed<float>(out, o.top_p);
  rb::Fixed<float>(out, o.min_p);
  rb::Fixed<std::uint32_t>(out, o.top_k);
  rb::Fixed<std::uint64_t>(out, o.seed);
  rb::Fixed<std::uint32_t>(out, o.max_tokens);
  rb::Str(out, o.finish_reason);
  rb::Str(out, o.decision_answer);
}

Observation DecodeObservation(rb::Reader& r) {
  Observation o;
  o.project = r.Str();
  o.trace_id = r.Str();
  o.span_id = r.Str();
  o.parent_span_id = r.Str();
  o.start_ns = r.Fixed<std::int64_t>();
  o.end_ns = r.Fixed<std::int64_t>();
  o.completion_start_ns = r.Fixed<std::int64_t>();
  o.type = r.Str();
  o.name = r.Str();
  o.user_id = r.Str();
  o.session_id = r.Str();
  o.tags = ReadStrArray(r);
  o.status_code = r.Str();
  o.level = r.Str();
  o.provider = r.Str();
  o.model = r.Str();
  o.variant_id = r.Str();
  o.adapter_id = r.Str();
  o.input_tokens = r.Fixed<std::uint64_t>();
  o.output_tokens = r.Fixed<std::uint64_t>();
  o.cached_tokens = r.Fixed<std::uint64_t>();
  o.usage_details.resize(r.Varint());
  for (auto& [k, v] : o.usage_details) {
    k = r.Str();
    v = r.Fixed<std::uint64_t>();
  }
  o.prompt_name = r.Str();
  o.prompt_version = r.Fixed<std::uint32_t>();
  o.tool_name = r.Str();
  o.tool_call_ids = ReadStrArray(r);
  o.system_ref = ReadHashArray(r);
  o.input_refs = ReadHashArray(r);
  o.output_refs = ReadHashArray(r);
  o.input_bytes = r.Fixed<std::uint64_t>();
  o.output_bytes = r.Fixed<std::uint64_t>();
  o.truncated = r.Fixed<std::uint8_t>();
  o.tpot_p50_us = r.Fixed<std::uint32_t>();
  o.tpot_p90_us = r.Fixed<std::uint32_t>();
  o.tpot_p99_us = r.Fixed<std::uint32_t>();
  o.mtp_drafted = r.Fixed<std::uint32_t>();
  o.mtp_accepted = r.Fixed<std::uint32_t>();
  o.temperature = r.Fixed<float>();
  o.top_p = r.Fixed<float>();
  o.min_p = r.Fixed<float>();
  o.top_k = r.Fixed<std::uint32_t>();
  o.seed = r.Fixed<std::uint64_t>();
  o.max_tokens = r.Fixed<std::uint32_t>();
  o.finish_reason = r.Str();
  o.decision_answer = r.Str();
  return o;
}

rb::Hash16 PayloadHash(std::string_view content) {
  blake3_hasher h;
  blake3_hasher_init(&h);
  blake3_hasher_update(&h, content.data(), content.size());
  rb::Hash16 out;
  blake3_hasher_finalize(&h, out.data(), out.size());
  return out;
}

void EncodePayload(const Payload& p, std::string& out) {
  rb::Str(out, p.project);
  rb::FixedStr16(out, p.hash);
  rb::Fixed<std::uint32_t>(out, p.bytes);
  rb::Str(out, p.content);
}

Payload DecodePayload(rb::Reader& r) {
  Payload p;
  p.project = r.Str();
  p.hash = r.FixedStr16();
  p.bytes = r.Fixed<std::uint32_t>();
  p.content = r.Str();
  return p;
}

}  // namespace halo::scope
