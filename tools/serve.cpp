// halo_serve: the OpenAI-compatible server on Flash-Next.
//   halo_serve --model <shard-00001.gguf> [--host 127.0.0.1] [--port 8080]
//              [--device hip|hrx] [--max-context 32768] [--api-key KEY]
//              [--model-id NAME] [--decider-dir DIR [--decider-device hip|hrx|cpu]]
// --decider-dir serves POST /v1/systemone with strands-decider-2B (the
// directory holds Qwen3.5-2B-Base-BF16.gguf and the strands-decider-*.gguf
// files, as the decider tests expect).
// Endpoints: GET /health, /v1/models, /v1/responses/{id};
//            POST /v1/completions, /v1/chat/completions, /v1/responses.

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "core/gguf.hpp"
#include "decide/decider.hpp"
#include "engine/flash_next_gpu.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"
#include "server/flash_next_generator.hpp"
#include "server/http.hpp"
#include "server/openai_service.hpp"
#include "server/systemone_service.hpp"
#include "text/tokenizer.hpp"

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

 private:
  std::unique_ptr<decide::StrandsDecider> d_;
};
}  // namespace

int main(int argc, char** argv) {
  std::string model, device = "hip", host = "127.0.0.1", api_key, model_id = "qwen3.8-flash-next", decider_dir,
                    decider_device = "hip";
  int port = 8080;
  std::uint32_t max_context = 32768;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--model") model = v;
    else if (a == "--device") device = v;
    else if (a == "--host") host = v;
    else if (a == "--port") port = std::atoi(v.c_str());
    else if (a == "--max-context") max_context = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--api-key") api_key = v;
    else if (a == "--model-id") model_id = v;
    else if (a == "--decider-dir") decider_dir = v;
    else if (a == "--decider-device") decider_device = v;
    else return std::fprintf(stderr, "unknown option %s\n", a.c_str()), 2;
  }
  if (model.empty()) return std::fprintf(stderr, "usage: halo_serve --model <gguf> [options]\n"), 2;
  std::string error;
  auto reader = core::GgufReader::OpenFile(model, &error);
  if (!reader) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  auto weights = model::ModelWeights::Bind(*reader, &error);
  auto tok = text::Tokenizer::FromGguf(*reader, &error);
  if (!weights || !tok) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  const auto& t = weights->ple_table;
  auto ngram = model::NgramTable::Open(reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset, t.rows,
                                       weights->config.ple_head_dim, t.type, &error);
  if (!ngram) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  auto dev = rt::Device::Create(device == "hrx" ? rt::Backend::kHrx : rt::Backend::kHip, &error);
  if (!dev) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  std::vector<rt::FileSource> sources;
  for (const auto& r : reader->GetMappedRegions()) sources.push_back({r.data, r.size, r.file_descriptor});
  std::fprintf(stderr, "loading weights...\n");
  auto engine = engine::FlashNextGpu::Create(*dev, *weights, ngram.get(), nullptr, max_context, &error, sources);
  if (!engine) return std::fprintf(stderr, "%s\n", error.c_str()), 1;

  server::FlashNextGenerator generator(*engine, *tok, max_context);
  server::ServiceOptions so;
  so.model_id = model_id;
  server::OpenAiService service(generator, so);
  server::HttpServer http({.host = host, .port = port, .api_key = api_key});
  service.Register(http);
  std::unique_ptr<StrandsEngine> decider;
  if (!decider_dir.empty()) {
    const std::string& d = decider_dir;
    auto sd = decide::StrandsDecider::Open({d + "/Qwen3.5-2B-Base-BF16.gguf", d + "/strands-decider-lora-f32.gguf",
                                            d + "/strands-decider-head.gguf", d + "/strands-decider-tokenizer.gguf",
                                            decider_device},
                                           &error);
    if (!sd) return std::fprintf(stderr, "decider: %s\n", error.c_str()), 1;
    decider = std::make_unique<StrandsEngine>(std::move(sd));
    server::RegisterSystemOne(http, *decider);
  }
  if (!http.Start(&error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  std::fprintf(stderr, "serving %s on http://%s:%d\n", model_id.c_str(), host.c_str(), http.port());
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);
  while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
  http.Stop();
}
