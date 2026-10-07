// halo_serve: the OpenAI-compatible server on Flash-Next.
//   halo_serve --model <shard-00001.gguf> [--host 127.0.0.1] [--port 8080]
//              [--device hip|hrx] [--max-context 32768] [--kv-f16 0|1] [--api-key KEY]
//              [--model-id NAME] [--decider-dir DIR [--decider-device hip|hrx|cpu]]
// With halo-scope built in (HALO_SCOPE=ON): --scope-dir DIR records every
// generation as a GENERATION observation in DIR/chdb and keeps stored
// /v1/responses in DIR/metadata.sqlite, so they survive restarts; harnesses
// annotate traces through POST /v1/scope/... (scope/http_annotations.hpp)
// and read them back with GET /v1/scope/traces[/{id}] (scope/trace_reader.hpp)
// [--scope-project NAME] [--scope-capture off|digest|full] [--scope-wal PATH].
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

#include <csignal>
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
#include "server/metrics.hpp"
#include "server/openai_service.hpp"
#include "server/qwen35_generator.hpp"
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

int main(int argc, char** argv) {
  std::string model, device = "hip", host = "127.0.0.1", api_key, model_id = "qwen3.8-flash-next", decider_dir,
                    decider_device = "hip", scope_dir, scope_project = "halo-flash", scope_capture = "digest",
                    scope_wal, scope_eval_endpoint, qwen35_model, qwen35_tokenizer;
  std::vector<std::string> variant_paths;
  int port = 8080;
  std::uint32_t moe_topk = 0;  // --moe-topk N: experts per token (0: the file's count; CODESIGN arm B)
  double ngram_cache_gb = 4.0;  // cache of n-gram rows read from disk, GiB (0: off)
  std::uint32_t max_context = 32768;
  server::SessionCacheOptions cache;  // prompt-prefix reuse across requests (server/session_cache.hpp)
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--model") model = v;
    else if (a == "--device") device = v;
    else if (a == "--host") host = v;
    else if (a == "--port") port = std::atoi(v.c_str());
    else if (a == "--max-context") max_context = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--ngram-cache-gb") ngram_cache_gb = std::atof(v.c_str());
    else if (a == "--kv-f16") engine::FlashNextGpu::kv_f16_only_ = v == "1";  // fp16-only KV (docs/SESSIONS.md)
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
    else if (a == "--model-id") model_id = v;
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
  std::string error;
  std::unique_ptr<server::TextGenerator> generator;
  // Flash-Next (--model) or a Qwen3.5 dense model with variants (--qwen35-model).
  std::unique_ptr<core::GgufReader> reader, q35_reader, q35_tok_reader;
  std::optional<model::ModelWeights> weights;
  std::optional<model::Qwen35Weights> q35_weights;
  std::unique_ptr<text::Tokenizer> tok;
  std::unique_ptr<model::NgramTable> ngram;
  std::unique_ptr<rt::Device> dev;
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
    std::fprintf(stderr, "loading weights...\n");
    engine = engine::FlashNextGpu::Create(*dev, *weights, ngram.get(), nullptr, max_context, &error, sources);
    if (!engine) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    generator = std::make_unique<server::FlashNextGenerator>(*engine, *tok, max_context, cache);
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
  server::HttpServer http({.host = host, .port = port, .api_key = api_key});
  service.Register(http);
  metrics.Register(http);
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
