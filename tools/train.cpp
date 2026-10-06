// halo_train: P11 stage 1 (last-k LoRA and task heads on Qwen3.5 dense
// models) from a JSON run config. docs/TRAINING.md describes the design.
//
//   halo_train cache  --config run.json [--max-seconds S]
//       Runs the frozen trunk (layers 0 .. L-k-1) over the train and eval
//       data on the inference engine and writes the boundary cache
//       (resumable shards).
//   halo_train train  --config run.json [--max-seconds S] [--threads N]
//       Trains on the CPU from the cache; resumes from the run's checkpoint;
//       stops after S seconds with a checkpoint (run it again to continue).
//   halo_train export --config run.json [--code-commit SHA]
//       Writes the variant (lora.gguf / head.gguf / manifest.json).
//   halo_train eval   --config run.json [--variant manifest.json] [--update-manifest]
//       Evaluates the base model and the variant on the held-out data with
//       the inference engine (GPU prefill through every layer, the variant
//       loaded through its manifest), the head on the CPU.
//   halo_train bench  --model base.gguf --k K --tokens T [--targets N] [--threads N]
//       Times the CPU tail's forward + backward at the model's real shapes.
//
// run.json (paths relative to the file):
//   {"base": "Qwen3.5-2B-Base-BF16.gguf", "tokenizer": "tokenizer.gguf",
//    "train_data": "train.jsonl", "eval_data": "heldout.jsonl", "work_dir": "run1",
//    "engine": "hip" | "cpu", "last_k": 2, "head": "lm" | "classifier", "num_classes": 0,
//    "lora": {"rank": 16, "alpha": 32, "targets": ["attn_q", ...]},
//    "steps": 200, "batch_size": 8, "lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0.01,
//    "warmup_steps": 10, "max_grad_norm": 1.0, "seed": 0, "checkpoint_every": 10,
//    "eval_every": 50, "max_tokens": 512,
//    "variant": {"name": "sentiment-ab", "version": 1, "base_name": "Qwen3.5-2B-Base"}}

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <thread>

#include "core/atomic_file.hpp"
#include "core/gguf.hpp"
#include "core/json.hpp"
#include "core/sha256.hpp"
#include "model/qwen35.hpp"
#include "model/variant.hpp"
#include "text/tokenizer.hpp"
#include "train/data.hpp"
#include "train/eval_report.hpp"
#include "train/export.hpp"
#include "train/last_k.hpp"
#include "train/trainer.hpp"
#if HALO_WITH_HIP
#include "engine/qwen35_gpu.hpp"
#include "rt/device.hpp"
#endif

using namespace halo;
namespace fs = std::filesystem;

namespace {

struct RunConfig {
  std::string path, sha256, dir;
  json::Value raw;
  std::string base, tokenizer, train_data, eval_data, work_dir, engine = "hip";
  train::TailConfig tail;
  train::TrainOptions opt;
  std::uint32_t max_tokens = 512;
  std::string variant_name, base_name;
  std::int64_t variant_version = 1;
};

int Die(const std::string& m) {
  std::fprintf(stderr, "halo_train: %s\n", m.c_str());
  return 1;
}

double Num(const json::Value& o, const char* k, double def) {
  const auto* v = o.find(k);
  return v && (v->is(json::Value::Kind::kInt) || v->is(json::Value::Kind::kFloat)) ? v->number() : def;
}
std::string Str(const json::Value& o, const char* k, const std::string& def = "") {
  const auto* v = o.find(k);
  return v && v->is(json::Value::Kind::kString) ? v->str() : def;
}

std::optional<RunConfig> LoadConfig(const std::string& path, std::string* error) {
  RunConfig c;
  c.path = path;
  std::string text;
  if (!core::ReadWholeFile(path, &text, error)) return std::nullopt;
  c.sha256 = core::Sha256Hex(text);
  auto v = json::Parse(text, error);
  if (!v || !v->is(json::Value::Kind::kObject)) return std::nullopt;
  c.raw = *v;
  c.dir = fs::absolute(path).parent_path().string();
  const auto rel = [&](const std::string& p) { return p.empty() || p[0] == '/' ? p : (fs::path(c.dir) / p).string(); };
  c.base = rel(Str(*v, "base"));
  c.tokenizer = rel(Str(*v, "tokenizer"));
  c.train_data = rel(Str(*v, "train_data"));
  c.eval_data = rel(Str(*v, "eval_data"));
  c.work_dir = rel(Str(*v, "work_dir"));
  c.engine = Str(*v, "engine", "hip");
  if (c.base.empty() || c.train_data.empty() || c.work_dir.empty()) {
    if (error) *error = "config needs base, train_data and work_dir";
    return std::nullopt;
  }
  auto& t = c.tail;
  t.last_k = static_cast<std::uint32_t>(Num(*v, "last_k", 2));
  const std::string head = Str(*v, "head", "lm");
  t.head = head == "classifier" ? train::HeadKind::kClassifier : train::HeadKind::kLm;
  t.num_classes = static_cast<std::uint32_t>(Num(*v, "num_classes", 0));
  if (const auto* l = v->find("lora")) {
    t.lora_rank = static_cast<std::uint32_t>(Num(*l, "rank", 0));
    t.lora_alpha = static_cast<float>(Num(*l, "alpha", 0));
    if (const auto* ts = l->find("targets"))
      for (const auto& x : ts->items()) t.lora_targets.push_back(x.str());
  }
  auto& o = c.opt;
  o.steps = static_cast<std::uint64_t>(Num(*v, "steps", 100));
  o.batch_size = static_cast<std::uint32_t>(Num(*v, "batch_size", 8));
  o.checkpoint_every = static_cast<std::uint64_t>(Num(*v, "checkpoint_every", 10));
  o.eval_every = static_cast<std::uint64_t>(Num(*v, "eval_every", 0));
  o.seed = static_cast<std::uint64_t>(Num(*v, "seed", 0));
  o.opt.lr = static_cast<float>(Num(*v, "lr", 1e-4));
  o.opt.head_lr = static_cast<float>(Num(*v, "head_lr", 1e-3));
  o.opt.weight_decay = static_cast<float>(Num(*v, "weight_decay", 0.01));
  o.opt.max_grad_norm = static_cast<float>(Num(*v, "max_grad_norm", 1.0));
  o.opt.warmup_steps = static_cast<std::uint64_t>(Num(*v, "warmup_steps", 0));
  o.opt.total_steps = o.steps;
  c.max_tokens = static_cast<std::uint32_t>(Num(*v, "max_tokens", 512));
  if (const auto* var = v->find("variant")) {
    c.variant_name = Str(*var, "name");
    c.variant_version = static_cast<std::int64_t>(Num(*var, "version", 1));
    c.base_name = Str(*var, "base_name");
  }
  return c;
}

// Everything a command needs from the base model.
struct Base {
  std::unique_ptr<core::GgufReader> reader, tok_reader;
  std::optional<model::Qwen35Weights> w;
  std::unique_ptr<text::Tokenizer> tok;
  std::string sha256, tokenizer_sha256;
};

std::optional<Base> OpenBase(const RunConfig& c, std::string* error) {
  Base b;
  b.reader = core::GgufReader::OpenFile(c.base, error);
  if (!b.reader) return std::nullopt;
  b.w = model::Qwen35Weights::Bind(*b.reader, error);
  if (!b.w) return std::nullopt;
  const std::vector<std::string> files = {c.base};
  const auto h = model::BaseContentHashCached(files, error);
  if (!h) return std::nullopt;
  b.sha256 = *h;
  const std::string tp = c.tokenizer.empty() ? c.base : c.tokenizer;
  b.tok_reader = core::GgufReader::OpenFile(tp, error);
  if (!b.tok_reader) return std::nullopt;
  b.tok = text::Tokenizer::FromGguf(*b.tok_reader, error);
  if (!b.tok) return std::nullopt;
  if (c.tokenizer.empty()) {
    b.tokenizer_sha256 = b.sha256;
  } else {
    const auto th = core::Sha256File(c.tokenizer, error);
    if (!th) return std::nullopt;
    b.tokenizer_sha256 = *th;
  }
  return b;
}

train::TrunkCache::Key CacheKey(const RunConfig& c, const Base& b, const std::string& data_sha) {
  return {b.sha256, data_sha, b.tokenizer_sha256, c.engine == "cpu" ? "cpu-reference" : "hip-prefill",
          b.w->config.num_layers - c.tail.last_k, b.w->config.hidden_size};
}

std::string Arg(int argc, char** argv, const char* name, const std::string& def = "") {
  for (int i = 2; i + 1 < argc; ++i)
    if (std::string(argv[i]) == name) return argv[i + 1];
  return def;
}
bool Flag(int argc, char** argv, const char* name) {
  for (int i = 2; i < argc; ++i)
    if (std::string(argv[i]) == name) return true;
  return false;
}

unsigned Threads(int argc, char** argv) {
  const auto s = Arg(argc, argv, "--threads");
  if (!s.empty()) return static_cast<unsigned>(std::atoi(s.c_str()));
  return std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));
}

// A trunk function on the configured engine; `engine_holder` keeps it alive.
struct Trunk {
  train::TrunkFn fn;
#if HALO_WITH_HIP
  std::unique_ptr<rt::Device> dev;
  std::unique_ptr<engine::Qwen35Gpu> gpu;
#endif
};

std::optional<Trunk> MakeTrunk(const RunConfig& c, const Base& b, std::uint32_t layer, std::string* error) {
  Trunk t;
  const auto& w = *b.w;
  if (c.engine == "cpu") {
    t.fn = [&w, layer](std::span<const std::int32_t> tokens, std::span<float> out, std::string* err) {
      model::Qwen35Reference ref(w, static_cast<std::uint32_t>(tokens.size()) + 1);
      const std::size_t H = w.config.hidden_size;
      for (std::size_t p = 0; p < tokens.size(); ++p)
        if (!ref.StepToLayer(tokens[p], layer, out.subspan(p * H, H), err)) return false;
      return true;
    };
    return t;
  }
#if HALO_WITH_HIP
  t.dev = rt::Device::Create(rt::Backend::kHip, error);
  if (!t.dev) return std::nullopt;
  t.gpu = engine::Qwen35Gpu::Create(*t.dev, w, nullptr, c.max_tokens + 8, error);
  if (!t.gpu) return std::nullopt;
  auto* gpu = t.gpu.get();
  t.fn = [gpu, layer](std::span<const std::int32_t> tokens, std::span<float> out, std::string* err) {
    return gpu->Reset(err) && gpu->PrefillToLayer(tokens, layer, out, err);
  };
  return t;
#else
  if (error) *error = "engine hip needs a GPU build";
  return std::nullopt;
#endif
}

int CmdCache(int argc, char** argv) {
  std::string error;
  auto c = LoadConfig(Arg(argc, argv, "--config"), &error);
  if (!c) return Die(error);
  auto b = OpenBase(*c, &error);
  if (!b) return Die(error);
  const double max_s = std::atof(Arg(argc, argv, "--max-seconds", "1e9").c_str());
  const std::uint32_t layer = b->w->config.num_layers - c->tail.last_k;
  auto trunk = MakeTrunk(*c, *b, layer, &error);
  if (!trunk) return Die(error);
  const auto t0 = std::chrono::steady_clock::now();
  bool all = true;
  for (const auto& [name, path] : {std::pair{"train", c->train_data}, std::pair{"eval", c->eval_data}}) {
    if (path.empty()) continue;
    auto d = train::LoadJsonl(path, *b->tok, c->tail.head, c->tail.num_classes, c->max_tokens, &error);
    if (!d) return Die(error);
    const double left = max_s - std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto st = train::TrunkCache::Build(c->work_dir + "/cache-" + name, CacheKey(*c, *b, d->sha256), d->examples,
                                             trunk->fn, left, &error);
    if (!st) return Die(error);
    std::fprintf(stderr, "halo_train cache %s: %zu/%zu shards (%zu built now, %llu tokens in %.1f s, %.0f tok/s)\n", name,
                 st->shards_done, st->shards_total, st->shards_built_now,
                 static_cast<unsigned long long>(st->tokens_built_now), st->seconds,
                 st->seconds > 0 ? static_cast<double>(st->tokens_built_now) / st->seconds : 0.0);
    all = all && st->complete;
  }
  std::printf("%s\n", all ? "complete" : "incomplete");
  return all ? 0 : 3;
}

struct Loaded {
  RunConfig c;
  Base b;
  train::Dataset train, eval;
  std::optional<train::TrunkCache> train_cache, eval_cache;
  std::unique_ptr<train::LastK> tail;
};

// Heap-allocated: the tail keeps a pointer to the bound weights.
std::unique_ptr<Loaded> LoadForTraining(const std::string& config, std::string* error) {
  auto owned = std::make_unique<Loaded>();
  Loaded& L = *owned;
  auto c = LoadConfig(config, error);
  if (!c) return nullptr;
  L.c = std::move(*c);
  auto b = OpenBase(L.c, error);
  if (!b) return nullptr;
  L.b = std::move(*b);
  auto d = train::LoadJsonl(L.c.train_data, *L.b.tok, L.c.tail.head, L.c.tail.num_classes, L.c.max_tokens, error);
  if (!d) return nullptr;
  L.train = std::move(*d);
  L.train_cache = train::TrunkCache::Load(L.c.work_dir + "/cache-train", CacheKey(L.c, L.b, L.train.sha256),
                                          L.train.examples, error);
  if (!L.train_cache) return nullptr;
  if (!L.c.eval_data.empty()) {
    auto e = train::LoadJsonl(L.c.eval_data, *L.b.tok, L.c.tail.head, L.c.tail.num_classes, L.c.max_tokens, error);
    if (!e) return nullptr;
    L.eval = std::move(*e);
    L.eval_cache = train::TrunkCache::Load(L.c.work_dir + "/cache-eval", CacheKey(L.c, L.b, L.eval.sha256),
                                           L.eval.examples, error);
    if (!L.eval_cache) return nullptr;
  }
  L.tail = train::LastK::Create(*L.b.w, L.c.tail, error);
  if (!L.tail) return nullptr;
  L.tail->InitParams(L.c.opt.seed);
  return owned;
}

train::RunBinding Binding(const Loaded& L) {
  return {L.b.sha256, L.train.sha256, L.c.sha256, L.train_cache->sha256(), train::LayoutSha256(*L.tail)};
}

int CmdTrain(int argc, char** argv) {
  std::string error;
  auto L = LoadForTraining(Arg(argc, argv, "--config"), &error);
  if (!L) return Die(error);
  const double max_s = std::atof(Arg(argc, argv, "--max-seconds", "1e9").c_str());
  train::ThreadPool pool(Threads(argc, argv));
  const std::string run_dir = L->c.work_dir + "/run";
  train::Trainer trainer(*L->tail, L->c.opt, L->train, *L->train_cache, Binding(*L), run_dir, pool);
  std::error_code ec;
  fs::create_directories(run_dir, ec);
  train::JsonlEvalSink sink(run_dir + "/evals.jsonl");
  if (L->eval_cache) {
    train::EvalReport tmpl;
    tmpl.run_id = fs::path(L->c.work_dir).filename().string();
    tmpl.variant = L->c.variant_name;
    tmpl.variant_version = L->c.variant_version;
    tmpl.base_sha256 = L->b.sha256;
    trainer.SetEval({"heldout", &L->eval, &*L->eval_cache}, &sink, tmpl);
  }
  const auto st = trainer.Run(max_s, &error);
  if (!st) return Die(error);
  std::fprintf(stderr, "halo_train: step %llu/%llu in %.1f s, last loss %.4f, checkpoint %s\n",
               static_cast<unsigned long long>(st->step), static_cast<unsigned long long>(L->c.opt.steps), st->seconds,
               st->last_loss, st->checkpoint_sha256.substr(0, 16).c_str());
  std::printf("%s\n", st->done ? "done" : "incomplete");
  return st->done ? 0 : 3;
}

int CmdExport(int argc, char** argv) {
  std::string error;
  auto L = LoadForTraining(Arg(argc, argv, "--config"), &error);
  if (!L) return Die(error);
  const std::string run_dir = L->c.work_dir + "/run";
  std::string ckpt_sha;
  auto ck = train::ReadCheckpoint(run_dir + "/checkpoint.bin", &error, &ckpt_sha);
  if (!ck) return Die(error);
  if (const auto why = ck->binding.Mismatch(Binding(*L)); !why.empty()) return Die("checkpoint mismatch: " + why);
  if (ck->step < L->c.opt.steps) return Die("training is not finished (step " + std::to_string(ck->step) + ")");
  L->tail->params() = ck->params;
  train::ExportInfo info;
  info.name = L->c.variant_name;
  info.version = L->c.variant_version;
  info.out_dir = L->c.work_dir + "/variant";
  info.base_name = L->c.base_name;
  info.base_sha256 = L->b.sha256;
  json::Value prov = json::Value::MakeObject();
  prov.set("trainer", json::Value::String("halo_train stage1 (CPU tail, last " + std::to_string(L->c.tail.last_k) +
                                          " layers)"));
  prov.set("code_commit", json::Value::String(Arg(argc, argv, "--code-commit", "unknown")));
  prov.set("data_sha256", json::Value::String(L->train.sha256));
  prov.set("eval_data_sha256", json::Value::String(L->eval.sha256));
  prov.set("config", L->c.raw);
  prov.set("config_sha256", json::Value::String(L->c.sha256));
  prov.set("trunk_cache_sha256", json::Value::String(L->train_cache->sha256()));
  prov.set("checkpoint_sha256", json::Value::String(ckpt_sha));
  prov.set("steps", json::Value::Int(static_cast<std::int64_t>(ck->step)));
  prov.set("tokenizer_sha256", json::Value::String(L->b.tokenizer_sha256));
  info.provenance = prov;
  // The trainer's own held-out evals (CPU tail), first and last, go into the summary.
  std::string text, ignored;
  if (core::ReadWholeFile(run_dir + "/evals.jsonl", &text, &ignored)) {
    std::map<std::string, std::pair<double, double>> first_last;  // metric -> (step 0, final)
    std::map<std::string, model::VariantEval> last;
    std::size_t at = 0;
    while (at < text.size()) {
      const auto end = text.find('\n', at);
      if (end == std::string::npos) break;
      std::string perr;
      const auto v = json::Parse(std::string_view(text.data() + at, end - at), &perr);
      at = end + 1;
      if (!v) continue;
      const std::string metric = Str(*v, "metric");
      const auto step = static_cast<std::int64_t>(Num(*v, "step", -1));
      if (step == 0) first_last[metric].first = Num(*v, "value", 0);
      if (step == static_cast<std::int64_t>(ck->step)) {
        model::VariantEval e;
        e.name = Str(*v, "dataset") + " (cpu tail)";
        e.metric = metric;
        e.value = Num(*v, "value", 0);
        e.n = static_cast<std::uint64_t>(Num(*v, "n", 0));
        e.step = step;
        e.dataset_sha256 = Str(*v, "dataset_sha256");
        last[metric] = e;
      }
    }
    for (auto& [metric, e] : last) {
      if (first_last.contains(metric) && L->c.tail.head == train::HeadKind::kLm)
        e.baseline = first_last[metric].first;  // step 0 of an LM-head LoRA is the base model
      info.eval.push_back(e);
    }
  }
  const auto path = train::ExportVariant(*L->tail, info, &error);
  if (!path) return Die(error);
  std::printf("%s\n", path->c_str());
  return 0;
}

int CmdEval(int argc, char** argv) {
  std::string error;
  auto c = LoadConfig(Arg(argc, argv, "--config"), &error);
  if (!c) return Die(error);
  auto b = OpenBase(*c, &error);
  if (!b) return Die(error);
  const std::string data_path = Arg(argc, argv, "--data", c->eval_data);
  auto d = train::LoadJsonl(data_path, *b->tok, c->tail.head, c->tail.num_classes, c->max_tokens, &error);
  if (!d) return Die(error);
  const std::string manifest = Arg(argc, argv, "--variant", c->work_dir + "/variant/manifest.json");
  model::VariantRegistry reg(*b->reader, b->sha256);
  const auto variant = reg.Load(manifest, &error);
  if (!variant) return Die(error);
  const std::size_t H = b->w->config.hidden_size, L = b->w->config.num_layers;
  // Residual entering the final norm, every position, with the adapter
  // ("" = base) applied: the inference engine (GPU prefill), or the CPU
  // reference for engine "cpu" (models the GPU prefill cannot run).
  std::function<bool(const std::string&, std::span<const std::int32_t>, std::span<float>)> residual;
#if HALO_WITH_HIP
  std::unique_ptr<rt::Device> dev;
  std::unique_ptr<engine::Qwen35Gpu> gpu;
#endif
  if (c->engine == "cpu") {
    residual = [&](const std::string& adapter, std::span<const std::int32_t> tokens, std::span<float> out) {
      model::Qwen35Reference ref(*b->w, static_cast<std::uint32_t>(tokens.size()) + 1);
      ref.SetAdapter(adapter.empty() || !variant->lora ? nullptr : &*variant->lora);
      for (std::size_t p = 0; p < tokens.size(); ++p)
        if (!ref.StepToLayer(tokens[p], static_cast<std::uint32_t>(L), out.subspan(p * H, H), &error)) return false;
      return true;
    };
  } else {
#if HALO_WITH_HIP
    dev = rt::Device::Create(rt::Backend::kHip, &error);
    if (!dev) return Die(error);
    gpu = engine::Qwen35Gpu::Create(*dev, *b->w, nullptr, c->max_tokens + 8, &error);
    if (!gpu) return Die(error);
    if (variant->lora && !gpu->AddAdapter(variant->manifest.name, *variant->lora, &error)) return Die(error);
    residual = [&](const std::string& adapter, std::span<const std::int32_t> tokens, std::span<float> out) {
      return gpu->SelectAdapter(variant->lora ? adapter : "", &error) && gpu->Reset(&error) &&
             gpu->PrefillToLayer(tokens, static_cast<std::uint32_t>(L), out, &error);
    };
#else
    return Die("engine hip needs a GPU build");
#endif
  }
  // The head on the CPU: final norm + LM head, or the variant's classifier.
  train::TailConfig hc;
  hc.last_k = 0;
  hc.head = c->tail.head;
  hc.num_classes = c->tail.num_classes;
  auto head = train::LastK::Create(*b->w, hc, &error);
  if (!head) return Die(error);
  if (hc.head == train::HeadKind::kClassifier) {
    if (!variant->head() || !train::LoadClassifierHead(*variant->head(), *head, &error))
      return Die(error.empty() ? "the variant has no classifier head" : error);
  }
  train::ThreadPool pool(Threads(argc, argv));
  struct Result {
    double loss = 0;
    std::uint64_t correct = 0, exact = 0, positions = 0;
    double seconds = 0;
  };
  const auto run = [&](const std::string& adapter, bool classifier_ok) -> std::optional<Result> {
    Result r;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> res;
    for (const auto& ex : d->examples) {
      res.resize(ex.tokens.size() * H);
      if (!residual(adapter, ex.tokens, res)) return std::nullopt;
      train::Example e{res, static_cast<std::uint32_t>(ex.tokens.size()), ex.lm_targets, ex.label, ex.label_position};
      const auto l = head->Forward(e, pool);
      if (l.count != ex.LossCount()) {
        if (!classifier_ok) return r;
        error = "head failed";
        return std::nullopt;
      }
      r.loss += l.sum;
      r.correct += l.correct;
      r.exact += l.correct == l.count;
      r.positions += l.count;
    }
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
  };
  const auto var = run(variant->manifest.name, true);
  if (!var) return Die(error);
  std::optional<Result> base;
  if (hc.head == train::HeadKind::kLm) {  // a classifier head has no base-model counterpart
    base = run("", true);
    if (!base) return Die(error);
  }
  const double n = static_cast<double>(d->examples.size());
  train::JsonlEvalSink sink(Arg(argc, argv, "--out", c->work_dir + "/evals-engine.jsonl"));
  std::vector<model::VariantEval> summary;
  const auto now =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  for (const auto& [metric, vv, bv, count] :
       {std::tuple{"accuracy", var->correct / double(var->positions), base ? base->correct / double(base->positions) : NAN,
                   var->positions},
        std::tuple{"exact_match", var->exact / n, base ? base->exact / n : NAN, std::uint64_t(n)},
        std::tuple{"loss", var->loss / double(var->positions), base ? base->loss / double(base->positions) : NAN,
                   var->positions}}) {
    train::EvalReport r;
    r.run_id = fs::path(c->work_dir).filename().string();
    r.variant = variant->manifest.name;
    r.variant_version = variant->manifest.version;
    r.base_sha256 = b->sha256;
    r.dataset = fs::path(data_path).stem().string();
    r.dataset_sha256 = d->sha256;
    r.metric = metric;
    r.value = vv;
    if (!std::isnan(bv)) r.baseline = bv;
    r.n = count;
    r.engine = "hip";
    r.unix_ms = now;
    if (!sink.Report(r, &error)) return Die(error);
    summary.push_back(train::ToVariantEval(r));
    std::printf("%-12s variant %.4f%s\n", metric, vv,
                std::isnan(bv) ? "" : (std::string("  base ") + std::to_string(bv)).c_str());
  }
  std::printf("engine eval: %zu examples, variant %.1f s%s\n", d->examples.size(), var->seconds,
              base ? (", base " + std::to_string(base->seconds) + " s").c_str() : "");
  if (Flag(argc, argv, "--update-manifest")) {
    auto m = variant->manifest;
    for (auto& e : summary) e.name += " (engine)";
    std::erase_if(m.eval, [](const model::VariantEval& e) { return e.name.ends_with(" (engine)"); });
    m.eval.insert(m.eval.end(), summary.begin(), summary.end());
    if (!m.WriteFile(manifest, &error)) return Die(error);
  }
  return 0;
}

int CmdBench(int argc, char** argv) {
  std::string error;
  auto reader = core::GgufReader::OpenFile(Arg(argc, argv, "--model"), &error);
  if (!reader) return Die(error);
  auto w = model::Qwen35Weights::Bind(*reader, &error);
  if (!w) return Die(error);
  const auto T = static_cast<std::uint32_t>(std::atoi(Arg(argc, argv, "--tokens", "256").c_str()));
  const auto targets = static_cast<std::uint32_t>(std::atoi(Arg(argc, argv, "--targets", "16").c_str()));
  train::ThreadPool pool(Threads(argc, argv));
  const std::size_t H = w->config.hidden_size;
  std::vector<float> boundary(T * H);
  std::mt19937 g(1);
  std::normal_distribution<float> nd(0.0F, 1.0F);
  for (auto& x : boundary) x = nd(g);
  for (const auto& kstr : {Arg(argc, argv, "--k", "2")}) {
    train::TailConfig tc;
    tc.last_k = static_cast<std::uint32_t>(std::atoi(kstr.c_str()));
    tc.lora_rank = 16;
    tc.lora_alpha = 32;
    tc.lora_targets = {"attn_q", "attn_k",    "attn_v",   "attn_output", "attn_qkv", "attn_gate",
                       "ssm_alpha", "ssm_beta", "ssm_out", "ffn_gate",    "ffn_up",   "ffn_down"};
    const auto t0 = std::chrono::steady_clock::now();
    auto tail = train::LastK::Create(*w, tc, &error);
    if (!tail) return Die(error);
    const double setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    train::Example ex{boundary, T, {}, -1, 0};
    for (std::uint32_t i = 0; i < targets; ++i) ex.lm_targets.emplace_back(T - targets + i - 1, 1000 + i);
    std::vector<float> grad(tail->num_params());
    double fwd = 1e9, both = 1e9;
    for (int r = 0; r < 3; ++r) {
      auto a = std::chrono::steady_clock::now();
      (void)tail->Forward(ex, pool);
      auto bb = std::chrono::steady_clock::now();
      (void)tail->ForwardBackward(ex, 1.0F / targets, grad, pool);
      auto cc = std::chrono::steady_clock::now();
      fwd = std::min(fwd, std::chrono::duration<double>(bb - a).count());
      both = std::min(both, std::chrono::duration<double>(cc - bb).count());
    }
    std::printf("k=%u T=%u targets=%u threads=%u params=%zu: setup %.2f s, forward %.1f ms, forward+backward %.1f ms "
                "(%.2f ms/token)\n",
                tc.last_k, T, targets, pool.size(), tail->num_params(), setup, fwd * 1e3, both * 1e3, both * 1e3 / T);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: halo_train cache|train|export|eval|bench [options] (see tools/train.cpp)\n");
    return 2;
  }
  const std::string cmd = argv[1];
  if (cmd == "cache") return CmdCache(argc, argv);
  if (cmd == "train") return CmdTrain(argc, argv);
  if (cmd == "export") return CmdExport(argc, argv);
  if (cmd == "eval") return CmdEval(argc, argv);
  if (cmd == "bench") return CmdBench(argc, argv);
  std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
  return 2;
}
