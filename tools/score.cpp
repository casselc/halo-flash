// halo_score: teacher-forced quality score on ds4's hosted-API fixtures
// (gguf-tools/quality-testing/data/qwen38-flash-alibaba-*), for the P1/P2
// quality gate. Prompts are rendered as ds4 renders Qwen3.8 non-thinking
// chat (no system turn, empty think block):
//   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
// The continuation (the API's tokens) is tokenized locally and fed token by
// token. Reported: mean NLL over the continuation tokens, and API top-token
// agreement where local tokens align byte for byte with the API's.
//
//   halo_score --model <shard-00001.gguf> --fixtures <dir> [--device hip|hrx|cpu] [--limit N]
//              [--prefill 1] [--prefill-int8 0|1]   (prompts through the batched prefill)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/gguf.hpp"
#include "core/json.hpp"
#include "engine/flash_next_gpu.hpp"
#include "model/ngram.hpp"
#include "model/reference.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"
#include "text/tokenizer.hpp"

using namespace halo;

namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string Basename(const std::string& p) { return p.substr(p.find_last_of('/') + 1); }

double LogSoftmaxAt(const std::vector<float>& l, std::uint32_t t) {
  float mx = l[0];
  for (float v : l) mx = std::max(mx, v);
  double s = 0;
  for (float v : l) s += std::exp(double(v) - mx);
  return double(l[t]) - mx - std::log(s);
}

std::uint32_t Argmax(const std::vector<float>& l) {
  std::uint32_t b = 0;
  for (std::uint32_t i = 1; i < l.size(); ++i)
    if (l[i] > l[b]) b = i;
  return b;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, fixtures, device = "hip";
  int limit = 1 << 30;
  bool eager = true;  // graph replay is the default once instantiation is fast on the real model
  bool prefill = false;
  bool ngram_resident = false;  // --ngram-resident 1: the n-gram table in RAM (~29 GB)
  setvbuf(stdout, nullptr, _IOLBF, 0);
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--model") model = argv[i + 1];
    else if (a == "--fixtures") fixtures = argv[i + 1];
    else if (a == "--device") device = argv[i + 1];
    else if (a == "--limit") limit = std::atoi(argv[i + 1]);
    else if (a == "--eager") eager = std::string(argv[i + 1]) == "1";
    else if (a == "--int8-moe") engine::FlashNextGpu::int8_act_moe_ = std::string(argv[i + 1]) == "1";
    else if (a == "--prefill") prefill = std::string(argv[i + 1]) == "1";
    else if (a == "--ngram-resident") ngram_resident = std::string(argv[i + 1]) == "1";
    else if (a == "--prefill-int8") engine::FlashNextGpu::prefill_int8_gemm_ = std::string(argv[i + 1]) == "1";
    else return std::fprintf(stderr, "unknown option %s\n", a.c_str()), 2;
  }
  if (model.empty() || fixtures.empty())
    return std::fprintf(stderr, "usage: halo_score --model <gguf> --fixtures <dir> [--device hip|hrx|cpu]\n"), 2;
  std::string error;
  auto reader = core::GgufReader::OpenFile(model, &error);
  if (!reader) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  auto weights = model::ModelWeights::Bind(*reader, &error);
  auto tok = text::Tokenizer::FromGguf(*reader, &error);
  if (!weights || !tok) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  const auto& c = weights->config;
  const auto& t = weights->ple_table;
  auto ngram = model::NgramTable::Open(reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset, t.rows,
                                       c.ple_head_dim, t.type, &error);
  if (!ngram) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  if (ngram_resident) {
    const auto tr = std::chrono::steady_clock::now();
    if (!ngram->LoadResident(&error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    std::fprintf(stderr, "n-gram table resident (%.1f GB) in %.1f s\n", ngram->Rows() * ngram->RowBytes() / 1e9,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - tr).count());
  }
  std::unordered_map<std::string, text::TokenId> by_bytes;
  for (text::TokenId id = 0; id < c.vocab_size; ++id) by_bytes.emplace(std::string(tok->TokenBytes(id)), id);

  constexpr std::uint32_t kMaxContext = 32768;
  std::unique_ptr<rt::Device> dev;
  std::unique_ptr<engine::FlashNextGpu> gpu;
  std::unique_ptr<model::ReferenceModel> cpu;
  if (device == "cpu") {
    cpu = std::make_unique<model::ReferenceModel>(*weights, ngram.get(), kMaxContext);
  } else {
    dev = rt::Device::Create(device == "hrx" ? rt::Backend::kHrx : rt::Backend::kHip, &error);
    std::vector<rt::FileSource> sources;
    for (const auto& r : reader->GetMappedRegions()) sources.push_back({r.data, r.size, r.file_descriptor});
    if (dev) gpu = engine::FlashNextGpu::Create(*dev, *weights, ngram.get(), nullptr, kMaxContext, &error, sources);
    if (!gpu) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    gpu->SetUseGraphs(!eager);
  }
  const auto step = [&](std::int32_t token, std::span<float> logits) {
    return cpu ? cpu->Step(token, logits, {}, &error) : gpu->Step(token, logits, &error);
  };

  std::ifstream manifest(fixtures + "/manifest-aligned.tsv");
  std::string line;
  double nll = 0;
  std::size_t targets = 0, aligned = 0, agree = 0, cases = 0, misaligned_cases = 0;
  std::vector<float> logits(c.vocab_size);
  const auto t0 = std::chrono::steady_clock::now();
  while (std::getline(manifest, line) && static_cast<int>(cases) < limit) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ls(line);
    std::string id, pf, cf, rf;
    std::getline(ls, id, '\t');
    std::getline(ls, pf, '\t');
    std::getline(ls, cf, '\t');
    std::getline(ls, rf, '\t');
    const std::string prompt = ReadFile(fixtures + "/prompts/" + Basename(pf));
    const std::string cont = ReadFile(fixtures + "/continuations/" + Basename(cf));
    auto resp = json::Parse(ReadFile(fixtures + "/responses/" + Basename(rf)), &error);
    if (!resp) return std::fprintf(stderr, "%s: %s\n", id.c_str(), error.c_str()), 1;

    // Prompt, piecewise as ds4 renders it.
    std::vector<text::TokenId> ids;
    for (const auto& piece : {std::string("<|im_start|>user\n"), prompt, std::string("<|im_end|>\n"),
                              std::string("<|im_start|>assistant\n<think>\n\n</think>\n\n")}) {
      const bool special = &piece != &prompt;
      const auto p = tok->Encode(piece, special);
      ids.insert(ids.end(), p.begin(), p.end());
    }
    const auto target = tok->Encode(cont, false);

    // API tokens (bytes) and their top token, to align with local tokens.
    std::vector<std::string> api_tokens, api_top;
    const auto* content = resp->find("choices");
    if (content && content->is(json::Value::Kind::kArray) && !content->items().empty()) {
      const auto* lp = content->items()[0].find("logprobs");
      const auto* arr = lp ? lp->find("content") : nullptr;
      if (arr && arr->is(json::Value::Kind::kArray))
        for (const auto& e : arr->items()) {
          const auto bytes = [](const json::Value* b) {
            std::string s;
            if (b && b->is(json::Value::Kind::kArray))
              for (const auto& x : b->items()) s.push_back(static_cast<char>(static_cast<int>(x.number())));
            return s;
          };
          api_tokens.push_back(bytes(e.find("bytes")));
          const auto* top = e.find("top_logprobs");
          api_top.push_back(top && top->is(json::Value::Kind::kArray) && !top->items().empty() ? bytes(top->items()[0].find("bytes"))
                                                                             : std::string());
        }
    }
    bool is_aligned = api_tokens.size() == target.size();
    for (std::size_t i = 0; is_aligned && i < target.size(); ++i)
      is_aligned = api_tokens[i] == tok->TokenBytes(target[i]);
    misaligned_cases += !is_aligned;

    if (cpu) cpu->Reset();
    else if (!gpu->Reset(&error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    if (prefill && gpu) {  // P4: the prompt through Prefill (batched chunks), the targets stepped
      const std::vector<std::int32_t> head(ids.begin(), ids.end() - 1);
      if (!head.empty() && !gpu->Prefill(head, {}, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    } else {
      for (std::size_t i = 0; i + 1 < ids.size(); ++i)
        if (!step(static_cast<std::int32_t>(ids[i]), {})) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    }
    std::int32_t prev = static_cast<std::int32_t>(ids.back());
    double case_nll = 0;
    for (std::size_t j = 0; j < target.size(); ++j) {
      if (!step(prev, logits)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
      case_nll -= LogSoftmaxAt(logits, target[j]);
      if (is_aligned) {
        ++aligned;
        const auto top = by_bytes.find(api_top[j]);
        agree += top != by_bytes.end() && top->second == Argmax(logits);
      }
      prev = static_cast<std::int32_t>(target[j]);
    }
    nll += case_nll;
    targets += target.size();
    ++cases;
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s  prompt %4zu  targets %3zu  nll %.4f%s   | running nll %.6f  agreement %.2f%%  (%.0f s)\n",
                id.c_str(), ids.size(), target.size(), case_nll / target.size(), is_aligned ? "" : "  (misaligned)",
                nll / targets, aligned ? 100.0 * agree / aligned : 0.0, secs);
    std::fflush(stdout);
  }
  std::printf("cases %zu, target tokens %zu, aligned %zu (%zu misaligned cases)\nNLL %.6f   API top-token agreement %.2f%%\n",
              cases, targets, aligned, misaligned_cases, nll / targets, aligned ? 100.0 * agree / aligned : 0.0);
}
