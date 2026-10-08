// halo_serve: the OpenAI-compatible server on Flash-Next.
//   halo_serve --model <shard-00001.gguf> [--host 127.0.0.1] [--port 8080]
//              [--device hip|hrx] [--max-context 32768] [--kv-f16 0|1] [--kv-f16-above 150000] [--api-key KEY]
//              [--model-id NAME] [--decider-dir DIR [--decider-device hip|hrx|cpu]] [--send-timeout 30]
// --send-timeout S: a write to a client that makes no progress for S seconds fails, and the request is cancelled
// as if the client had left (a client that stops reading cannot hold the engine for longer).
// --sse-keepalive S (default 15; 0: off): a streamed request writes an SSE comment after S seconds without a write
// (queued, prefilling), so a client's idle-read timeout does not fire before its first token.
// With halo-scope built in (HALO_SCOPE=ON): --scope-dir DIR records every
// generation as a GENERATION observation in DIR/chdb and keeps stored
// /v1/responses in DIR/metadata.sqlite, so they survive restarts; harnesses
// annotate traces through POST /v1/scope/... (scope/http_annotations.hpp)
// and read them back with GET /v1/scope/traces[/{id}] (scope/trace_reader.hpp)
// KV kind (docs/SESSIONS.md section 7): without --kv-f16, halo_serve uses fp16-only KV (26.2 KB a token against
// 75.3) when --max-context is above --kv-f16-above (default 150000, about what default KV holds; 0 disables the
// automatic choice), the default kinds otherwise. --kv-f16 1 forces fp16-only KV, --kv-f16 0 the default kinds.
// The choice is logged at startup ("kv: ..."). halo_run and the library default stay default KV.
// [--scope-project NAME] [--scope-capture off|digest|full] [--scope-wal PATH].
// MTP speculative decoding (docs/MTP.md, off by default): --mtp-model SIDECAR (the shared-* MTP gguf) loads the
// draft block; --mtp-vocab IDS (int32 token ids, e.g. a 32K trimmed draft head) restricts its head; --mtp-window W
// (default 128, measured: docs/MTP.md) is how many of a request's last prefilled prompt positions the draft block
// catches up on; --mtp-depth N (default 4) the deepest draft. Greedy requests, and sampled ones without penalties,
// decode speculatively (greedy output identical to MTP off); requests with penalties decode as without it.
// Datasets, evaluators and experiments are served under /v1/scope/
// (scope/eval_routes.hpp); runs call this server's own OpenAI endpoints
// (and /v1/systemone for decider evaluators) unless
// --scope-eval-endpoint URL points them elsewhere.
// --decider-dir serves POST /v1/systemone with strands-decider-2B (the
// directory holds Qwen3.5-2B-Base-BF16.gguf and the strands-decider-*.gguf
// files, as the decider tests expect).
// Endpoints: GET /health, /metrics (Prometheus, vLLM names), /v1/models, /v1/responses/{id};
//            POST /v1/completions, /v1/chat/completions, /v1/responses.
// P10 variants: --qwen35-model GGUF --qwen35-tokenizer GGUF serves the OpenAI
// routes from a Qwen3.5 dense model instead of Flash-Next (no --model);
// --variant MANIFEST (repeatable) loads a variant of the Qwen3.5 base (the
// --qwen35-model, else the decider torso): its LoRA becomes selectable on
// the OpenAI routes ("variant" or model "<id>:<name>") and, with a
// pointer_head part, on /v1/systemone. A variant built for another base is
// refused at startup.

#include <algorithm>
#include <chrono>
#include <csignal>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "core/gguf.hpp"
#include "decide/decider.hpp"
#include "engine/qwen35_gpu.hpp"
#include "model/qwen35.hpp"
#include "model/variant.hpp"
#include "engine/flash_next_gpu.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"
#include "server/flash_next_generator.hpp"
#include "server/http.hpp"
#include "server/kv_choice.hpp"
#include "server/metrics.hpp"
#include "server/openai_service.hpp"
#include "server/qwen35_generator.hpp"
#include "server/spill_keys.hpp"
#include "server/yard_keys.hpp"
#include "server/systemone_service.hpp"
#include "text/tokenizer.hpp"
#if HALO_WITH_SCOPE
#include "halo_scope.h"
#include "scope/c_api.hpp"
#include "scope/generation_observer.hpp"
#include "scope/chdb.hpp"
#include "scope/eval_routes.hpp"
#include "scope/evals.hpp"
#include "scope/http_annotations.hpp"
#include "scope/prompt_routes.hpp"
#include "scope/trace_reader.hpp"
#include "scope/metadata.hpp"
#include "scope/response_store.hpp"
#endif

using namespace halo;

namespace {
volatile std::sig_atomic_t g_stop = 0;
void OnSignal(int) { g_stop = 1; }

class StrandsEngine : public server::DecisionEngine {
 public:
  explicit StrandsEngine(std::unique_ptr<decide::StrandsDecider> d) : d_(std::move(d)) {}
  std::optional<json::Value> Evaluate(const decide::Request& r, std::string* error) override {
    return d_->Evaluate(r, error);
  }
  decide::StrandsDecider& decider() { return *d_; }

 private:
  std::unique_ptr<decide::StrandsDecider> d_;
};
}  // namespace


// Printable ASCII without quotes or backslashes (labels and key ids are; anything else is dropped).
static std::string JsonSafe(const std::string& s) {
  std::string o;
  for (char ch : s)
    if (ch >= 0x20 && ch < 0x7f && ch != '"' && ch != '\\') o += ch;
  return o;
}

// SR4 admin routes over the session cache (docs/SESSIONS.md section 10): list retained states, spill one to the
// spill directory, restore one. Each runs between generations (the generation lock). `key_routes`: also the yard key
// route (POST /admin/keys/{session}, GET /admin/keys); by default only on the admin Unix socket.
// Lock order: the generation lock (WithCache) may take the key provider's lock (leases), never the reverse: a key
// registration runs outside the generation lock and only its revoke cleanup enters it afterwards.
static void RegisterSpillRoutes(halo::server::HttpServer& http, halo::server::FlashNextGenerator& g,
                                halo::server::MasterSpillKeys* master, halo::server::YardSpillKeys* yard,
                                bool key_routes) {
  using halo::server::HttpRequest;
  using halo::server::HttpResponse;
  using halo::server::SessionCache;
  http.Route("GET", "/admin/sessions", [&g](const HttpRequest&) {
    std::string body = "{\"sessions\":[";
    g.WithCache([&](SessionCache& c) {
      bool first = true;
      for (const auto& e : c.List()) {
        char b[700];  // labels and key ids are printable ASCII without quotes (spill_crypt.cpp, yard session ids)
        std::snprintf(b, sizeof(b),
                      "%s{\"key\":%llu,\"tokens\":%zu,\"checkpoint\":%s,\"canonical\":%s,\"spilled\":%s,"
                      "\"device_bytes\":%zu,\"file_bytes\":%llu,\"session\":\"%s\",\"key_id\":\"%s\",\"yard_bound\":%s,"
                      "\"entry_id\":\"%s\"}",
                      first ? "" : ",", static_cast<unsigned long long>(e.key), e.tokens, e.checkpoint ? "true" : "false",
                      e.canonical ? "true" : "false", e.spilled ? "true" : "false", e.device_bytes,
                      static_cast<unsigned long long>(e.file_bytes), JsonSafe(e.session).c_str(),
                      JsonSafe(e.key_id).c_str(), e.yard_bound ? "true" : "false", JsonSafe(e.entry_id).c_str());
        body += b;
        first = false;
      }
      return true;
    });
    return HttpResponse::Json(200, body + "]}");
  });
  http.RoutePrefix("POST", "/admin/sessions/", [&g](const HttpRequest& r) {
    const std::string rest = r.path.substr(std::string_view("/admin/sessions/").size());
    const auto slash = rest.find('/');
    const std::string op = slash == std::string::npos ? "" : rest.substr(slash + 1);
    std::uint64_t key = 0;
    if (slash == std::string::npos || !halo::server::ParseU64(std::string_view(rest).substr(0, slash), ~0ull, &key) ||
        (op != "spill" && op != "restore"))
      return HttpResponse::Json(404, "{\"error\":\"POST /admin/sessions/{key}/spill or /restore\"}");
    // No ?session=: which yard session (and key) seals an entry is named by the seal registration itself
    // (halo_spill.entry), so an admin caller cannot relabel an entry or spend another session's registration.
    if (!r.query.empty())
      return HttpResponse::Json(400, "{\"error\":\"no query parameters (the seal registration names the entry)\"}");
    std::string why;
    const bool ok = g.WithCache([&](SessionCache& c) { return op == "spill" ? c.Spill(key, &why) : c.Restore(key, &why); });
    return ok ? HttpResponse::Json(200, "{\"ok\":true}")
              : HttpResponse::Json(409, "{\"ok\":false,\"error\":\"" + JsonSafe(why) + "\"}");
  });
  // Spill encryption, phase B: yard-ctl registers per-checkpoint keys (docs/SESSIONS.md section 10, "The key route").
  if (yard && key_routes) {
    http.Route("GET", "/admin/keys", [yard](const HttpRequest&) { return HttpResponse::Json(200, yard->ListJson()); });
    http.RoutePrefix("POST", "/admin/keys/", [&g, yard](const HttpRequest& r) {
      const std::string session = r.path.substr(std::string_view("/admin/keys/").size());
      // The body holds the key and the token: Register overwrites it (the request is this connection's own copy).
      auto a = yard->Register(session, const_cast<std::string&>(r.body));
      std::size_t dropped = 0;
      // A revoke's cleanup carries its fence: only that session's entries at key epochs <= through go.
      if (a.status == 200 && !a.revoked_session.empty())
        g.WithCache([&](SessionCache& c) { return dropped = c.DropRevoked(a.revoked_session, a.revoked_through), true; });
      if (dropped) a.body.insert(a.body.size() - 1, ",\"dropped\":" + std::to_string(dropped));
      return HttpResponse::Json(a.status, a.body);
    });
  }
  // Spill encryption, phase A: fence the master's epoch. Files sealed before no longer open through this server and
  // their entries are erased and files deleted. This is an access fence, not crypto-shredding: the master still
  // derives the old KEKs (docs/SESSIONS.md section 10, "Epochs"). /admin/spill/rotate is the old name.
  const auto fence = [&g, master](const HttpRequest&) {
    std::string why;
    if (!master) return HttpResponse::Json(409, "{\"ok\":false,\"error\":\"no master spill key provider\"}");
    std::size_t dropped = 0;
    const bool ok = g.WithCache([&](SessionCache& c) {
      if (!master->FenceEpoch(&why)) return false;
      dropped = c.DropUnreadable();
      return true;
    });
    if (!ok) return HttpResponse::Json(409, "{\"ok\":false,\"error\":\"" + JsonSafe(why) + "\"}");
    return HttpResponse::Json(200, "{\"ok\":true,\"keys\":\"" + JsonSafe(master->Describe()) +
                                       "\",\"dropped\":" + std::to_string(dropped) + "}");
  };
  http.Route("POST", "/admin/spill/fence-epoch", fence);
  http.Route("POST", "/admin/spill/rotate", fence);
}

int main(int argc, char** argv) {
  std::string model, device = "hip", host = "127.0.0.1", api_key, model_id = "qwen3.8-flash-next", decider_dir,
                    decider_device = "hip", scope_dir, scope_project = "halo-flash", scope_capture = "digest",
                    scope_wal, scope_eval_endpoint, qwen35_model, qwen35_tokenizer;
  std::vector<std::string> variant_paths;
  int port = 8080;
  double send_timeout_s = 30;
  double sse_keepalive_s = 15;
  std::uint32_t moe_topk = 0;  // --moe-topk N: experts per token (0: the file's count; CODESIGN arm B)
  double ngram_cache_gb = 4.0;  // cache of n-gram rows read from disk, GiB (0: off)
  std::uint32_t max_context = 32768;
  server::SessionCacheOptions cache;  // prompt-prefix reuse across requests (server/session_cache.hpp)
  server::KvChoiceOptions kv;         // --kv-f16 / --kv-f16-above (server/kv_choice.hpp)
  std::string mtp_model, mtp_vocab;   // --mtp-model / --mtp-vocab (docs/MTP.md)
  server::SpecOptions spec;           // --mtp-window / --mtp-depth
  std::string spill_key = "none";     // --spill-key (server/spill_keys.hpp)
  std::string admin_socket;           // --admin-socket (default $XDG_RUNTIME_DIR/halo-flash/admin.sock)
  bool yard_keys_tcp = false;         // --yard-keys-tcp
  server::YardSpillKeys::Options yard_opt;  // --yard-* (phase B)
  bool yard_fallback_master = false;  // --spill-yard-fallback
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::printf(
          "usage: halo_serve --model <shard-00001.gguf> | --qwen35-model <gguf> --qwen35-tokenizer <gguf> [options]\n"
          "  --host 127.0.0.1 --port 8080 --device hip|hrx --max-context 32768 --api-key KEY --model-id NAME\n"
          "  --send-timeout S      a client write with no progress for S seconds (default 30) cancels the request\n"
          "  --sse-keepalive S     streams: an SSE comment after S idle seconds while queued or prefilling (default 15; 0: off)\n"
          "  --kv-f16 0|1          KV kind: 1 fp16-only (26.2 KB/token), 0 default kinds (75.3 KB/token).\n"
          "                        Not given: fp16-only when --max-context > --kv-f16-above, else default.\n"
          "  --kv-f16-above N      threshold for the automatic fp16-only choice (default 150000; 0: never).\n"
          "                        Default KV reliably holds ~150K positions; 262144 needs fp16-only KV.\n"
          "  --prefix-cache exact|fast|off  prompt-prefix reuse (default exact: byte-identical to a cold prefill)\n"
          "  --prefix-cache-sessions N --prefix-cache-gb G --prefix-cache-align N\n"
          "  --spill-dir DIR       session spill files (default /data/halo-work/spill; 0700 dir, 0600 files)\n"
          "  --spill-evict 0|1     cache eviction spills to DIR instead of dropping (default 0). Admin routes:\n"
          "                        GET /admin/sessions, POST /admin/sessions/{key}/spill | /restore,\n"
          "                        POST /admin/spill/fence-epoch (master epoch + 1, persisted: older files are\n"
          "                        fenced and deleted; an access fence, not erasure; old name /admin/spill/rotate)\n"
          "  --spill-key tpm[:CRED]|file:PATH|none  spill encryption key (format v2, AES-256-GCM). tpm: the\n"
          "                        TPM2-sealed systemd credential halo-spill-master (default CRED\n"
          "                        $XDG_CONFIG_HOME/halo-flash/spill-master.cred; $CREDENTIALS_DIRECTORY first).\n"
          "                        Default none: spills are REFUSED (eviction drops) unless --spill-plaintext-ok 1\n"
          "  --spill-plaintext-ok 0|1  with no --spill-key, write plaintext v1 files (default 0)\n"
          "  --spill-accept-v1 0|1 restore plaintext v1 files this process wrote as v1 (default 0; logged per file;\n"
          "                        a migration aid only, to be removed with format v1)\n"
          "                        The master's epoch is persisted in DIR/master-<key name>.epoch (never goes back)\n"
          "  --yard-jwks PATH      phase B: yard-ctl registers per-checkpoint keys (POST /admin/keys/{session}, RS256 JWT\n"
          "                        verified against this JWKS file, re-read when it changes and at least every 60 s);\n"
          "                        --yard-issuer ISS (required) --yard-audience AUD (default halo_serve, non-empty)\n"
          "                        --yard-key-ttl S (1..3600, default 600) --yard-state PATH (default DIR/yard-state)\n"
          "                        --spill-yard-fallback master|refuse (default refuse: an entry without a seal\n"
          "                        registration is not spilled; master: sealed under --spill-key unless yard-bound)\n"
          "  --admin-socket PATH   admin routes on a Unix socket (dir 0700, socket 0600, peer uid checked); the yard\n"
          "                        key route is served only there (default $XDG_RUNTIME_DIR/halo-flash/admin.sock)\n"
          "                        Same uid only: a caller of another uid (yard-ctl) uses the tailnet route\n"
          "  --yard-keys-tcp 0|1   also serve the key route on --host:--port (only behind an authenticated, encrypted\n"
          "                        transport such as the tailnet's WireGuard: the key is in the body; default 0)\n"
          "  --mtp-model SIDECAR [--mtp-vocab IDS] [--mtp-window W (128)] [--mtp-depth N (4)]\n"
          "                        MTP speculative decoding (off without --mtp-model)\n"
          "  --ngram-cache-gb G --moe-topk N --decider-dir DIR --decider-device hip|hrx|cpu --variant MANIFEST\n"
          "  --scope-dir DIR --scope-project NAME --scope-capture off|digest|full --scope-wal PATH\n"
          "  --scope-eval-endpoint URL\n");
      return 0;
    }
  }
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--model") model = v;
    else if (a == "--device") device = v;
    else if (a == "--host") host = v;
    else if (a == "--port") port = std::atoi(v.c_str());
    else if (a == "--send-timeout") {
      send_timeout_s = std::atof(v.c_str());
      if (!(send_timeout_s > 0)) return std::fprintf(stderr, "--send-timeout must be positive seconds\n"), 2;
    } else if (a == "--sse-keepalive") {
      sse_keepalive_s = std::atof(v.c_str());
      if (!(sse_keepalive_s >= 0)) return std::fprintf(stderr, "--sse-keepalive must be seconds >= 0\n"), 2;
    }
    else if (a == "--max-context") max_context = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--ngram-cache-gb") ngram_cache_gb = std::atof(v.c_str());
    else if (a == "--kv-f16") {  // fp16-only KV (docs/SESSIONS.md); not given: by --kv-f16-above
      if (v != "0" && v != "1") return std::fprintf(stderr, "--kv-f16 must be 0 or 1\n"), 2;
      kv.explicit_kv_f16 = v == "1";
    } else if (a == "--kv-f16-above") kv.f16_above = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--moe-topk") moe_topk = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--api-key") api_key = v;
    else if (a == "--prefix-cache") {  // exact (default) | fast | off
      if (v == "exact") cache.mode = server::SessionCacheOptions::Mode::kExact;
      else if (v == "fast") cache.mode = server::SessionCacheOptions::Mode::kFast;
      else if (v == "off") cache.mode = server::SessionCacheOptions::Mode::kOff;
      else return std::fprintf(stderr, "--prefix-cache must be exact, fast or off\n"), 2;
    } else if (a == "--prefix-cache-sessions") cache.max_retained = std::strtoul(v.c_str(), nullptr, 10);
    else if (a == "--prefix-cache-gb") cache.budget_bytes = static_cast<std::size_t>(std::atof(v.c_str()) * (1ull << 30));
    else if (a == "--prefix-cache-align") cache.align = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--spill-dir") cache.spill_dir = v;
    else if (a == "--spill-evict") cache.spill_on_evict = v == "1";
    else if (a == "--spill-key") spill_key = v;
    else if (a == "--spill-plaintext-ok") cache.spill_plaintext_ok = v == "1";
    else if (a == "--spill-accept-v1") {
      if (v != "0" && v != "1") return std::fprintf(stderr, "--spill-accept-v1 must be 0 or 1\n"), 2;
      cache.spill_accept_v1 = v == "1";
    } else if (a == "--spill-epoch") {
      return std::fprintf(stderr, "--spill-epoch was removed: the master's epoch is persisted next to the spill files "
                                  "and moves only by POST /admin/spill/fence-epoch\n"), 2;
    }
    else if (a == "--yard-jwks") yard_opt.jwks_path = v;
    else if (a == "--yard-issuer") yard_opt.issuer = v;
    else if (a == "--yard-audience") yard_opt.audience = v;
    else if (a == "--yard-state") yard_opt.state_path = v;
    else if (a == "--yard-key-ttl") {
      std::uint64_t t = 0;
      if (!server::ParseU64(v, 3600, &t) || t < 1) return std::fprintf(stderr, "--yard-key-ttl must be 1..3600 s\n"), 2;
      yard_opt.key_ttl_s = static_cast<std::int64_t>(t);
    } else if (a == "--admin-socket") admin_socket = v;
    else if (a == "--admin-socket-peer-uid") {
      return std::fprintf(stderr, "--admin-socket-peer-uid was removed: the admin socket (0700 dir, 0600 socket) only "
                                  "admits halo_serve's own uid; see IDENTITY.md \"Transport\"\n"), 2;
    } else if (a == "--yard-keys-tcp") {
      if (v != "0" && v != "1") return std::fprintf(stderr, "--yard-keys-tcp must be 0 or 1\n"), 2;
      yard_keys_tcp = v == "1";
    }
    else if (a == "--spill-yard-fallback") {
      if (v != "master" && v != "refuse") return std::fprintf(stderr, "--spill-yard-fallback must be master or refuse\n"), 2;
      yard_fallback_master = v == "master";
    }
    else if (a == "--model-id") model_id = v;
    else if (a == "--mtp-model") mtp_model = v;
    else if (a == "--mtp-vocab") mtp_vocab = v;
    else if (a == "--mtp-window") spec.window = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--mtp-depth") {
      spec.depth.limit = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
      if (spec.depth.limit < 1 || spec.depth.limit > engine::FlashNextGpu::kMaxDrafts)
        return std::fprintf(stderr, "--mtp-depth must be 1..%u\n", engine::FlashNextGpu::kMaxDrafts), 2;
    }
    else if (a == "--decider-dir") decider_dir = v;
    else if (a == "--decider-device") decider_device = v;
    else if (a == "--scope-dir") scope_dir = v;
    else if (a == "--scope-project") scope_project = v;
    else if (a == "--scope-capture") scope_capture = v;
    else if (a == "--scope-wal") scope_wal = v;
    else if (a == "--scope-eval-endpoint") scope_eval_endpoint = v;
    else if (a == "--variant") variant_paths.push_back(v);
    else if (a == "--qwen35-model") qwen35_model = v;
    else if (a == "--qwen35-tokenizer") qwen35_tokenizer = v;
    else return std::fprintf(stderr, "unknown option %s\n", a.c_str()), 2;
  }
  if (model.empty() == qwen35_model.empty())
    return std::fprintf(stderr, "usage: halo_serve --model <gguf> | --qwen35-model <gguf> --qwen35-tokenizer <gguf> [options]\n"), 2;
  if (cache.spill_dir.empty()) cache.spill_dir = "/data/halo-work/spill";  // SR4 default: not /tmp (docs/SESSIONS.md 10)
  if (!mtp_model.empty() && model.empty()) return std::fprintf(stderr, "--mtp-model needs --model (Flash-Next)\n"), 2;
  std::string error;
  // Spill encryption keys: loaded before the cache exists and destroyed after it (masters zeroed on exit).
  // The spill directory holds the key state (master epoch, yard epochs): created 0700 here, before any key loads.
  if (spill_key != "none" || !yard_opt.jwks_path.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(cache.spill_dir, ec);
    if (!ec) std::filesystem::permissions(cache.spill_dir, std::filesystem::perms::owner_all, ec);
    if (ec) return std::fprintf(stderr, "--spill-dir %s: %s\n", cache.spill_dir.c_str(), ec.message().c_str()), 1;
  }
  std::unique_ptr<server::MasterSpillKeys> spill_keys;
  if (!server::MakeSpillKeys(spill_key, cache.spill_dir, &spill_keys, &error))
    return std::fprintf(stderr, "--spill-key %s: %s\n", spill_key.c_str(), error.c_str()), 1;
  cache.spill_keys = spill_keys.get();
  std::unique_ptr<server::YardSpillKeys> yard_keys;
  if (!yard_opt.jwks_path.empty()) {
    if (yard_fallback_master && !spill_keys)
      return std::fprintf(stderr, "--spill-yard-fallback master needs --spill-key tpm|file:PATH\n"), 2;
    yard_opt.fallback = yard_fallback_master ? spill_keys.get() : nullptr;
    if (yard_opt.state_path.empty()) yard_opt.state_path = cache.spill_dir + "/yard-state";
    yard_keys = server::YardSpillKeys::Create(yard_opt, &error);  // refuses an empty issuer / audience, bad TTLs
    if (!yard_keys) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    cache.spill_keys = yard_keys.get();
    cache.incarnation = yard_keys->incarnation();  // entry ids "<incarnation>.<key>" (the seal claim names one)
    std::fprintf(stderr, "spill: encrypted (v2), keys %s\n", yard_keys->Describe().c_str());
  } else if (spill_keys) std::fprintf(stderr, "spill: encrypted (v2), key %s\n", spill_keys->Describe().c_str());
  else if (cache.spill_plaintext_ok) std::fprintf(stderr, "spill: PLAINTEXT v1 files (--spill-plaintext-ok 1)\n");
  else std::fprintf(stderr, "spill: refused (no --spill-key; eviction drops)\n");
  std::unique_ptr<server::TextGenerator> generator;
  server::FlashNextGenerator* flash_generator = nullptr;  // SR4 admin routes
  // Flash-Next (--model) or a Qwen3.5 dense model with variants (--qwen35-model).
  std::unique_ptr<core::GgufReader> reader, q35_reader, q35_tok_reader;
  std::optional<model::ModelWeights> weights;
  std::optional<model::Qwen35Weights> q35_weights;
  std::unique_ptr<text::Tokenizer> tok;
  std::unique_ptr<model::NgramTable> ngram;
  std::unique_ptr<rt::Device> dev;
  std::unique_ptr<core::GgufReader> mtp_reader;  // the draft block's weights outlive the engine's use of them
  std::optional<model::MtpWeights> mtp_weights;
  std::unique_ptr<engine::FlashNextGpu> engine;
  std::unique_ptr<engine::Qwen35Gpu> q35_engine;
  server::Qwen35Generator* q35_generator = nullptr;
  if (!model.empty()) {
    reader = core::GgufReader::OpenFile(model, &error);
    if (!reader) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    weights = model::ModelWeights::Bind(*reader, &error);
    tok = text::Tokenizer::FromGguf(*reader, &error);
    if (!weights || !tok) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    if (moe_topk && !weights->SetExpertsUsed(moe_topk, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 2;
    if (weights->config.context_length > 0 && max_context > weights->config.context_length)
      return std::fprintf(stderr, "--max-context %u exceeds the model's context length %u\n", max_context,
                          weights->config.context_length), 2;
    const auto& t = weights->ple_table;
    ngram = model::NgramTable::Open(reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset, t.rows,
                                    weights->config.ple_head_dim, t.type, &error,
                                    static_cast<std::size_t>(ngram_cache_gb * (1ull << 30)));
    if (!ngram) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    dev = rt::Device::Create(device == "hrx" ? rt::Backend::kHrx : rt::Backend::kHip, &error);
    if (!dev) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    std::vector<rt::FileSource> sources;
    for (const auto& r : reader->GetMappedRegions()) sources.push_back({r.data, r.size, r.file_descriptor});
    kv.max_context = max_context;
    const server::KvChoice kv_choice = server::ResolveKvKind(kv);
    engine::FlashNextGpu::kv_f16_only_ = kv_choice.f16;
    std::fprintf(stderr, "kv: %s (%s)\n", kv_choice.f16 ? "fp16-only" : "default", kv_choice.reason.c_str());
    std::fprintf(stderr, "loading weights...\n");
    engine = engine::FlashNextGpu::Create(*dev, *weights, ngram.get(), nullptr, max_context, &error, sources);
    if (!engine) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    if (!mtp_model.empty()) {  // the MTP draft block (and its trimmed head)
      using clk = std::chrono::steady_clock;
      const auto t0 = clk::now();
      const std::size_t free0 = dev->AvailableBytes();
      mtp_reader = core::GgufReader::OpenFile(mtp_model, &error);
      if (mtp_reader) mtp_weights = model::MtpWeights::Bind(*mtp_reader, weights->config, &error);
      if (!mtp_reader || !mtp_weights || !engine->LoadMtp(*mtp_weights, &error))
        return std::fprintf(stderr, "--mtp-model %s: %s\n", mtp_model.c_str(), error.c_str()), 1;
      const std::size_t file_bytes = std::filesystem::file_size(mtp_model);
      const auto t1 = clk::now();
      const std::size_t free1 = dev->AvailableBytes();
      std::size_t ids_n = 0;
      if (!mtp_vocab.empty()) {
        std::ifstream f(mtp_vocab, std::ios::binary | std::ios::ate);
        if (!f) return std::fprintf(stderr, "--mtp-vocab %s: cannot read\n", mtp_vocab.c_str()), 1;
        std::vector<std::int32_t> ids(static_cast<std::size_t>(f.tellg()) / 4);
        f.seekg(0);
        f.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(ids.size() * 4));
        for (const auto id : ids)
          if (id < 0 || static_cast<std::uint32_t>(id) >= weights->config.vocab_size)
            return std::fprintf(stderr, "--mtp-vocab %s: id %d out of range\n", mtp_vocab.c_str(), id), 1;
        if (!engine->SetDraftVocab(ids, &error)) return std::fprintf(stderr, "--mtp-vocab: %s\n", error.c_str()), 1;
        ids_n = ids.size();
      }
      const auto t2 = clk::now();
      const std::size_t free2 = dev->AvailableBytes();
      const auto s = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
      const auto gb = [](std::size_t a, std::size_t b) { return a > b ? double(a - b) / (1ull << 30) : 0.0; };
      std::fprintf(stderr,
                   "mtp: draft block %s (%.2f GB file) loaded in %.1f s, %.2f GB device (weights and its %u-position "
                   "cache); draft head %s in %.1f s, %.3f GB device; window %u, depth limit %u\n",
                   mtp_model.c_str(), double(file_bytes) / (1ull << 30), s(t0, t1), gb(free0, free1), max_context,
                   ids_n ? (std::to_string(ids_n) + " ids (" + mtp_vocab + ")").c_str() : "full vocabulary", s(t1, t2),
                   gb(free1, free2), spec.window, spec.depth.limit);
      spec.enabled = true;
    }
    auto fg = std::make_unique<server::FlashNextGenerator>(*engine, *tok, max_context, cache, spec);
    flash_generator = fg.get();
    generator = std::move(fg);
  } else {
    q35_reader = core::GgufReader::OpenFile(qwen35_model, &error);
    if (!q35_reader) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    q35_weights = model::Qwen35Weights::Bind(*q35_reader, &error);
    q35_tok_reader = core::GgufReader::OpenFile(qwen35_tokenizer.empty() ? qwen35_model : qwen35_tokenizer, &error);
    if (!q35_weights || !q35_tok_reader) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    tok = text::Tokenizer::FromGguf(*q35_tok_reader, &error);
    if (!tok) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    dev = rt::Device::Create(device == "hrx" ? rt::Backend::kHrx : rt::Backend::kHip, &error);
    if (!dev) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    q35_engine = engine::Qwen35Gpu::Create(*dev, *q35_weights, nullptr, max_context, &error);
    if (!q35_engine) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    auto g = std::make_unique<server::Qwen35Generator>(*q35_engine, *tok, max_context);
    q35_generator = g.get();
    generator = std::move(g);
  }

  server::ServiceOptions so;
  so.model_id = model_id;
  so.sse_keepalive_ms = static_cast<int>(std::min(sse_keepalive_s * 1000, 3.6e6));
  server::Metrics metrics(model_id);
  server::FanOut observers;
  observers.Add(&metrics);
  so.observer = &observers;
  so.on_activity = [&metrics](bool started) { started ? metrics.Started() : metrics.Finished(); };
#if HALO_WITH_SCOPE
  halo_scope* scope = nullptr;
  std::unique_ptr<scope::GenerationObserver> observed;
  std::unique_ptr<scope::MetadataStore> metadata;
  std::unique_ptr<scope::SqliteResponseStore> responses;
  std::unique_ptr<scope::TraceReader> traces;
  std::unique_ptr<scope::EvalStore> evals;
  std::unique_ptr<scope::EvalService> eval_service;
  if (!scope_dir.empty()) {
    const auto capture = scope_capture == "off"    ? HALO_SCOPE_CAPTURE_OFF
                         : scope_capture == "full" ? HALO_SCOPE_CAPTURE_FULL
                                                   : HALO_SCOPE_CAPTURE_DIGEST;
    char err[512] = {};
    std::filesystem::create_directories(scope_dir);
    metadata = scope::MetadataStore::Open(scope_dir + "/metadata.sqlite", &error);
    if (!metadata) return std::fprintf(stderr, "halo-scope metadata: %s\n", error.c_str()), 1;
    responses = std::make_unique<scope::SqliteResponseStore>(*metadata, 100000);
    so.store = responses.get();
    scope = halo_scope_open((scope_dir + "/chdb").c_str(), scope_project.c_str(), capture,
                            scope_wal.empty() ? nullptr : scope_wal.c_str(), err, sizeof err);
    if (!scope) return std::fprintf(stderr, "halo-scope: %s\n", err), 1;
    observed = std::make_unique<scope::GenerationObserver>(scope::AnnotatorOf(scope));
    observers.Add(observed.get());
    const scope::ChdbLib* lib = scope::ChdbLib::Load("", &error);
    auto reader_conn = lib ? scope::Conn::Open(*lib, scope_dir + "/chdb", &error) : nullptr;
    if (!reader_conn) return std::fprintf(stderr, "halo-scope reader: %s\n", error.c_str()), 1;
    traces = std::make_unique<scope::TraceReader>(std::move(reader_conn), scope_project);
    evals = scope::EvalStore::Open(scope_dir + "/metadata.sqlite", &error);
    if (!evals) return std::fprintf(stderr, "halo-scope evals: %s\n", error.c_str()), 1;
    scope::EvalRoutesOptions eo;
    eo.endpoint = scope_eval_endpoint.empty() ? "http://127.0.0.1:" + std::to_string(port) : scope_eval_endpoint;
    if (!api_key.empty()) eo.headers.emplace_back("Authorization", "Bearer " + api_key);
    eo.annotator = &scope::AnnotatorOf(scope);
    eo.traces = traces.get();
    eval_service = std::make_unique<scope::EvalService>(*evals, eo);
  }
#else
  if (!scope_dir.empty()) return std::fprintf(stderr, "--scope-dir needs a build with HALO_SCOPE=ON\n"), 2;
#endif
  if (!qwen35_model.empty()) so.template_kind = text::ChatTemplateKind::kQwen35;
  // Variants of the Qwen3.5 text model, loaded before the routes go live.
  std::unique_ptr<model::VariantRegistry> q35_variants;
  if (q35_generator && !variant_paths.empty()) {
    const std::vector<std::string> files = {qwen35_model};
    const auto hash = model::BaseContentHashCached(files, &error);
    if (!hash) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    q35_variants = std::make_unique<model::VariantRegistry>(*q35_reader, *hash);
    for (const auto& path : variant_paths) {
      const auto v = q35_variants->Load(path, &error);
      if (!v) return std::fprintf(stderr, "variant %s: %s\n", path.c_str(), error.c_str()), 1;
      if (v->lora && !q35_generator->AddVariant(*v, &error))
        return std::fprintf(stderr, "variant %s: %s\n", path.c_str(), error.c_str()), 1;
      std::fprintf(stderr, "variant %s v%lld loaded\n", v->manifest.name.c_str(),
                   static_cast<long long>(v->manifest.version));
    }
  }
  if (q35_generator) so.variants = [q35_generator] { return q35_generator->Variants(); };
  server::OpenAiService service(*generator, so);
  server::HttpServer http({.host = host,
                           .port = port,
                           .send_timeout_ms = static_cast<int>(std::min(send_timeout_s * 1000, 3.6e6)),
                           .api_key = api_key});
  service.Register(http);
  metrics.Register(http);
  if (flash_generator) RegisterSpillRoutes(http, *flash_generator, spill_keys.get(), yard_keys.get(), yard_keys_tcp);
  // The admin Unix socket: the same admin routes, and the yard key route (its default and only same-host transport).
  std::unique_ptr<server::HttpServer> admin;
  if (flash_generator && (yard_keys || !admin_socket.empty())) {
    if (admin_socket.empty()) {
      const char* rt = std::getenv("XDG_RUNTIME_DIR");
      if (!rt || !*rt) return std::fprintf(stderr, "--admin-socket: no $XDG_RUNTIME_DIR; give a path\n"), 2;
      const std::string dir = std::string(rt) + "/halo-flash";
      std::error_code ec;
      std::filesystem::create_directories(dir, ec);
      std::filesystem::permissions(dir, std::filesystem::perms::owner_all, ec);
      admin_socket = dir + "/admin.sock";
    }
    server::HttpServerOptions ao;
    ao.max_connections = 4;
    ao.api_key = api_key;
    ao.unix_path = admin_socket;
    admin = std::make_unique<server::HttpServer>(ao);
    RegisterSpillRoutes(*admin, *flash_generator, spill_keys.get(), yard_keys.get(), true);
    if (!admin->Start(&error)) return std::fprintf(stderr, "--admin-socket: %s\n", error.c_str()), 1;
    std::fprintf(stderr, "admin routes on unix:%s%s\n", admin_socket.c_str(), yard_keys ? " (yard key route)" : "");
  }
#if HALO_WITH_SCOPE
  if (scope) {
    scope::RegisterAnnotationRoutes(http, scope::AnnotatorOf(scope));  // harnesses in other processes
    traces->Register(http);                                            // GET /v1/scope/traces[/{id}]
    scope::RegisterPromptRoutes(http, *metadata);                      // /v1/scope/prompts
    eval_service->Register(http);  // datasets, evaluators, runs, compare
  }
#endif
  std::unique_ptr<StrandsEngine> decider;
  if (!decider_dir.empty()) {
    const std::string& d = decider_dir;
    auto sd = decide::StrandsDecider::Open({d + "/Qwen3.5-2B-Base-BF16.gguf", d + "/strands-decider-lora-f32.gguf",
                                            d + "/strands-decider-head.gguf", d + "/strands-decider-tokenizer.gguf",
                                            decider_device},
                                           &error);
    if (!sd) return std::fprintf(stderr, "decider: %s\n", error.c_str()), 1;
    // Decider variants (a pointer_head part) of the torso's base model.
    std::unique_ptr<model::VariantRegistry> reg;
    for (const auto& path : variant_paths) {
      if (!reg) {
        const std::vector<std::string> files = {d + "/Qwen3.5-2B-Base-BF16.gguf"};
        const auto hash = model::BaseContentHashCached(files, &error);
        if (!hash) return std::fprintf(stderr, "decider: %s\n", error.c_str()), 1;
        reg = std::make_unique<model::VariantRegistry>(sd->torso_file(), *hash);
      }
      const auto v = reg->Load(path, &error);
      if (!v) return std::fprintf(stderr, "decider variant %s: %s\n", path.c_str(), error.c_str()), 1;
      if (!v->manifest.Find(model::VariantPartKind::kPointerHead)) continue;  // a text-model variant
      if (!sd->AddVariant(v, &error))
        return std::fprintf(stderr, "decider variant %s: %s\n", path.c_str(), error.c_str()), 1;
      std::fprintf(stderr, "decider variant %s loaded\n", v->manifest.name.c_str());
    }
    decider = std::make_unique<StrandsEngine>(std::move(sd));
    server::RegisterSystemOne(http, *decider);
  }
  if (!http.Start(&error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  std::fprintf(stderr, "serving %s on http://%s:%d\n", model_id.c_str(), host.c_str(), http.port());
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);
  while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
#if HALO_WITH_SCOPE
  if (eval_service) eval_service->Shutdown();  // running experiments stop after their current item
#endif
  http.Stop();
  if (admin) admin->Stop();
#if HALO_WITH_SCOPE
  if (scope) {
    halo_scope_flush(scope);
    halo_scope_stats st{};
    halo_scope_get_stats(scope, &st);
    std::fprintf(stderr, "halo-scope: %llu rows written, %llu failed, %llu dropped\n",
                 static_cast<unsigned long long>(st.rows_written), static_cast<unsigned long long>(st.rows_failed),
                 static_cast<unsigned long long>(st.dropped));
    halo_scope_close(scope);
  }
#endif
}
