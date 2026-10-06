#include "train/trainer.hpp"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>

#include "core/atomic_file.hpp"
#include "core/sha256.hpp"

namespace halo::train {

namespace {

std::uint64_t Mix(std::uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

std::vector<std::size_t> Permutation(std::uint64_t seed, std::size_t n, std::uint64_t epoch) {
  std::vector<std::size_t> p(n);
  for (std::size_t i = 0; i < n; ++i) p[i] = i;
  std::uint64_t s = Mix(seed ^ Mix(epoch + 1));
  for (std::size_t i = n; i > 1; --i) {
    s = Mix(s);
    std::swap(p[i - 1], p[s % i]);
  }
  return p;
}

std::uint64_t EnvStep(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::strtoull(v, nullptr, 10) : 0;
}

std::int64_t NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace

std::vector<std::size_t> BatchIndices(std::uint64_t seed, std::size_t n, std::uint64_t step, std::uint32_t batch_size) {
  std::vector<std::size_t> out;
  std::uint64_t cached_epoch = UINT64_MAX;
  std::vector<std::size_t> perm;
  for (std::uint32_t j = 0; j < batch_size; ++j) {
    const std::uint64_t k = step * batch_size + j, epoch = k / n;
    if (epoch != cached_epoch) {
      perm = Permutation(seed, n, epoch);
      cached_epoch = epoch;
    }
    out.push_back(perm[k % n]);
  }
  return out;
}

std::string LayoutSha256(const LastK& model) {
  std::string text = "halo-tail-layout-v1\n";
  const auto& c = model.config();
  text += "k=" + std::to_string(c.last_k) + " rank=" + std::to_string(c.lora_rank) +
          " alpha=" + std::to_string(c.lora_alpha) + " head=" + std::to_string(static_cast<int>(c.head)) +
          " classes=" + std::to_string(c.num_classes) + "\n";
  for (const auto& s : model.slices())
    text += s.name + " " + std::to_string(s.offset) + " " + std::to_string(s.rows) + " " + std::to_string(s.cols) +
            " " + std::to_string(s.is_head) + std::to_string(s.decay) + "\n";
  return core::Sha256Hex(text);
}

TailEval EvaluateTail(const LastK& model, const EvalSet& set, ThreadPool& pool) {
  TailEval e;
  double loss = 0;
  std::uint64_t correct = 0, exact = 0;
  for (std::size_t i = 0; i < set.data->examples.size(); ++i) {
    const auto& tx = set.data->examples[i];
    Example ex{set.cache->Boundary(i), static_cast<std::uint32_t>(tx.tokens.size()), tx.lm_targets, tx.label,
               tx.label_position};
    const auto r = model.Forward(ex, pool);
    loss += r.sum;
    correct += r.correct;
    exact += r.correct == r.count && r.count > 0;
    e.positions += r.count;
    ++e.examples;
  }
  e.loss = e.positions ? loss / static_cast<double>(e.positions) : 0;
  e.accuracy = e.positions ? static_cast<double>(correct) / static_cast<double>(e.positions) : 0;
  e.exact = e.examples ? static_cast<double>(exact) / static_cast<double>(e.examples) : 0;
  return e;
}

Trainer::Trainer(LastK& model, TrainOptions options, const Dataset& train, const TrunkCache& cache, RunBinding binding,
                 std::string run_dir, ThreadPool& pool)
    : model_(model),
      o_(std::move(options)),
      train_(train),
      cache_(cache),
      binding_(std::move(binding)),
      dir_(std::move(run_dir)),
      pool_(pool) {}

void Trainer::SetEval(EvalSet set, EvalSink* sink, EvalReport report_template) {
  eval_ = std::move(set);
  sink_ = sink;
  tmpl_ = std::move(report_template);
}

bool Trainer::Checkpoint(std::uint64_t step, std::string* sha, std::string* error) {
  train::Checkpoint c;
  c.binding = binding_;
  c.step = step;
  c.params = model_.params();
  c.m = opt_state_.m;
  c.v = opt_state_.v;
  const std::uint64_t kill = EnvStep("HALO_TRAIN_KILL_IN_CHECKPOINT");
  if (kill != 0 && kill == step) {
    // Crash test: the temp file is written and synced, the rename never happens.
    core::SetAtomicWriteFaultForTest(2);
    (void)WriteCheckpoint(dir_ + "/checkpoint.bin", c, error, sha);
    std::fprintf(stderr, "halo_train: injected kill inside the checkpoint write at step %llu\n",
                 static_cast<unsigned long long>(step));
    std::raise(SIGKILL);
  }
  return WriteCheckpoint(dir_ + "/checkpoint.bin", c, error, sha);
}

bool Trainer::Evaluate(std::uint64_t step, std::string* error) {
  if (!eval_.data || !sink_) return true;
  const auto e = EvaluateTail(model_, eval_, pool_);
  for (const auto& [metric, value, n] : {std::tuple{"loss", e.loss, e.positions}, {"accuracy", e.accuracy, e.positions},
                                         {"exact_match", e.exact, e.examples}}) {
    EvalReport r = tmpl_;
    r.dataset = eval_.name;
    r.dataset_sha256 = eval_.data->sha256;
    r.metric = metric;
    r.value = value;
    r.n = n;
    r.step = static_cast<std::int64_t>(step);
    r.engine = "cpu-tail";
    r.unix_ms = NowMs();
    if (!sink_->Report(r, error)) return false;
  }
  std::fprintf(stderr, "halo_train: eval %s at step %llu: loss %.4f accuracy %.4f exact %.4f (%llu examples)\n",
               eval_.name.c_str(), static_cast<unsigned long long>(step), e.loss, e.accuracy, e.exact,
               static_cast<unsigned long long>(e.examples));
  return true;
}

std::optional<Trainer::Status> Trainer::Run(double max_seconds, std::string* error) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string ckpt = dir_ + "/checkpoint.bin", progress = dir_ + "/progress.jsonl";
  std::error_code ec;
  std::filesystem::create_directories(dir_, ec);
  Status st;
  if (std::filesystem::exists(ckpt)) {
    auto c = ReadCheckpoint(ckpt, error, &st.checkpoint_sha256);
    if (!c) return std::nullopt;
    if (const auto why = c->binding.Mismatch(binding_); !why.empty()) {
      if (error) *error = "checkpoint " + ckpt + " belongs to another run (" + why + "); refusing to resume";
      return std::nullopt;
    }
    if (c->params.size() != model_.num_params()) {
      if (error) *error = "checkpoint parameter count differs";
      return std::nullopt;
    }
    model_.params() = std::move(c->params);
    opt_state_.m = std::move(c->m);
    opt_state_.v = std::move(c->v);
    opt_state_.step = c->step;
    st.step = c->step;
    // Keep the progress log consistent with the restored weights.
    std::string text, kept, ignored;
    if (core::ReadWholeFile(progress, &text, &ignored)) {
      std::size_t at = 0;
      while (at < text.size()) {
        const std::size_t end = text.find('\n', at);
        if (end == std::string::npos) break;  // a torn last line is dropped
        const std::string_view line(text.data() + at, end - at);
        at = end + 1;
        std::string perr;
        const auto v = json::Parse(line, &perr);
        if (!v) continue;
        const auto* s = v->find("step");
        if (s && s->number() > static_cast<double>(c->step) && !v->find("event")) continue;
        kept.append(line).push_back('\n');
      }
      if (!core::AtomicWriteFile(progress, kept, error)) return std::nullopt;
    }
    json::Value ev = json::Value::MakeObject();
    ev.set("event", json::Value::String("resume"));
    ev.set("step", json::Value::Int(static_cast<std::int64_t>(c->step)));
    ev.set("checkpoint_sha256", json::Value::String(st.checkpoint_sha256));
    if (!core::DurableAppend(progress, json::DumpPython(ev) + "\n", error)) return std::nullopt;
    std::fprintf(stderr, "halo_train: resumed at step %llu (checkpoint %s)\n",
                 static_cast<unsigned long long>(c->step), st.checkpoint_sha256.substr(0, 16).c_str());
  } else {
    opt_state_ = AdamWState{};
    json::Value ev = json::Value::MakeObject();
    ev.set("event", json::Value::String("start"));
    ev.set("step", json::Value::Int(0));
    ev.set("binding", binding_.ToJson());
    if (!core::DurableAppend(progress, json::DumpPython(ev) + "\n", error)) return std::nullopt;
    if (!Evaluate(0, error)) return std::nullopt;
  }
  const std::uint64_t kill_at = EnvStep("HALO_TRAIN_KILL_AT_STEP");
  std::vector<float> grad(model_.num_params());
  const std::size_t n = train_.examples.size();
  bool dirty = false;
  while (st.step < o_.steps) {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (elapsed >= max_seconds) break;
    const auto ts = std::chrono::steady_clock::now();
    const auto batch = BatchIndices(o_.seed, n, st.step, o_.batch_size);
    std::uint32_t count = 0;
    for (auto i : batch) count += train_.examples[i].LossCount();
    std::fill(grad.begin(), grad.end(), 0.0F);
    double loss = 0;
    std::uint64_t tokens = 0;
    for (auto i : batch) {
      const auto& tx = train_.examples[i];
      Example ex{cache_.Boundary(i), static_cast<std::uint32_t>(tx.tokens.size()), tx.lm_targets, tx.label,
                 tx.label_position};
      const auto r = model_.ForwardBackward(ex, 1.0F / static_cast<float>(count), grad, pool_);
      if (r.count != tx.LossCount()) {
        if (error) *error = "example " + std::to_string(i) + " failed in the tail";
        return std::nullopt;
      }
      loss += r.sum;
      tokens += tx.tokens.size();
    }
    const auto info = AdamWStep(o_.opt, model_.slices(), model_.params(), grad, opt_state_);
    ++st.step;
    st.last_loss = loss / count;
    dirty = true;
    json::Value p = json::Value::MakeObject();
    p.set("step", json::Value::Int(static_cast<std::int64_t>(st.step)));
    p.set("loss", json::Value::Float(st.last_loss));
    p.set("positions", json::Value::Int(count));
    p.set("tokens", json::Value::Int(static_cast<std::int64_t>(tokens)));
    p.set("grad_norm", json::Value::Float(info.grad_norm));
    p.set("lr", json::Value::Float(info.lr));
    p.set("head_lr", json::Value::Float(info.head_lr));
    p.set("seconds", json::Value::Float(std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count()));
    if (!core::DurableAppend(progress, json::DumpPython(p) + "\n", error)) return std::nullopt;
    if (kill_at != 0 && st.step == kill_at) {
      std::fprintf(stderr, "halo_train: injected kill after step %llu\n", static_cast<unsigned long long>(st.step));
      std::raise(SIGKILL);
    }
    if (o_.checkpoint_every && st.step % o_.checkpoint_every == 0) {
      if (!Checkpoint(st.step, &st.checkpoint_sha256, error)) return std::nullopt;
      dirty = false;
    }
    if (o_.eval_every && st.step % o_.eval_every == 0 && st.step < o_.steps && !Evaluate(st.step, error))
      return std::nullopt;
  }
  if (dirty && !Checkpoint(st.step, &st.checkpoint_sha256, error)) return std::nullopt;
  st.done = st.step >= o_.steps;
  // The final evaluation: after the chunk that finished training, or again
  // when a finished run is started (its final eval may have been lost to a
  // crash after the last checkpoint).
  if (st.done && !Evaluate(st.step, error)) return std::nullopt;
  st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return st;
}

}  // namespace halo::train
