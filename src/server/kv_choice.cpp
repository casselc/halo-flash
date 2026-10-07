#include "server/kv_choice.hpp"

namespace halo::server {

KvChoice ResolveKvKind(const KvChoiceOptions& o) {
  if (o.explicit_kv_f16 >= 0) return {o.explicit_kv_f16 != 0, "explicit --kv-f16 " + std::to_string(o.explicit_kv_f16)};
  const std::string ctx = "--max-context " + std::to_string(o.max_context);
  if (o.f16_above == 0) return {false, "auto disabled (--kv-f16-above 0)"};
  const std::string thr = "--kv-f16-above " + std::to_string(o.f16_above);
  if (o.max_context > o.f16_above) return {true, "auto: " + ctx + " > " + thr};
  return {false, "auto: " + ctx + " <= " + thr};
}

}  // namespace halo::server
