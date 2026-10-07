// halo_serve's KV-kind choice (docs/SESSIONS.md section 7): fp16-only KV (26.2 KB a token) or the default kinds
// (75.3 KB). Default KV reliably holds ~150K positions of one session; the model's full 262,144 is servable only
// with fp16-only KV. So when the command line names no kind, halo_serve picks fp16-only KV for a --max-context
// above a threshold; an explicit --kv-f16 0|1 always wins. halo_run and the library default stay default KV.
#pragma once

#include <cstdint>
#include <string>

namespace halo::server {

struct KvChoiceOptions {
  int explicit_kv_f16 = -1;            ///< --kv-f16: -1 not given, 0 default kinds, 1 fp16-only
  std::uint32_t max_context = 0;       ///< --max-context
  std::uint32_t f16_above = 150000;    ///< --kv-f16-above: auto fp16 when max_context > this; 0 disables auto
};

struct KvChoice {
  bool f16 = false;
  std::string reason;  ///< one line for the startup log, e.g. "auto: --max-context 262144 > --kv-f16-above 150000"
};

KvChoice ResolveKvKind(const KvChoiceOptions& o);

}  // namespace halo::server
