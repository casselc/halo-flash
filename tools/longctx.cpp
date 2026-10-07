// halo_longctx: long-context checks of Flash-Next on the GPU engine (stream C, docs/SESSIONS.md "Full context").
// Every mode appends one JSON line per finished item to --out and skips items already there, so an interrupted
// run resumes where it stopped (each GPU hold can be one process).
//
//   halo_longctx --model <shard-00001.gguf> --mode MODE --text FILE --out FILE.jsonl [--max-context N]
//                [--kv-f16 0|1] [--prefill-long-chunk N] [--ngram-cache-gb G]
//   --mode nll    --context N --targets M --windows W1,W2,.. [--offset O]
//       Scores tokens [O+N, O+N+M) of FILE stepping through them after a prefill of the W-1 tokens before
//       O+N-1 (W = N: the full preceding context). One line per window: mean NLL and every target's log-prob.
//   --mode needle --lengths L1,.. --depths D1,.. (percent) [--seed S] [--max-new 24]
//       A "special magic number" sentence planted at each depth of an L-token chat prompt made of FILE's
//       text, then the question; greedy answer. One line per (length, depth).
//   --mode split  --length N --steps K
//       One prefill of N tokens against: a prefill of N-K tokens then K steps; a prefill of N-K then one of
//       K; and the same N tokens with --prefill-long-chunk 0 (2048-token chunks). Bitwise / max |d| / argmax.
//   --mode fork   --length N --steps K [--snapshot 1]
//       Prefill N tokens, fork, step K tokens in the fork; then the parent reset and replayed fresh (prefill N,
//       the same K steps): every step's logits compared bitwise. --snapshot 1 also saves the state at N, runs
//       the K steps, restores and runs them again.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/sampling.hpp"
#include "engine/flash_next_gpu.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"
#include "text/tokenizer.hpp"

using namespace halo;
using clk = std::chrono::steady_clock;

namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::vector<std::uint32_t> ParseList(const std::string& s) {
  std::vector<std::uint32_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back(static_cast<std::uint32_t>(std::strtoul(item.c_str(), nullptr, 10)));
  return out;
}

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

double Secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

// MemAvailable and the GPU's GTT use, GB (unified memory: both come from the same DRAM).
std::string Mem() {
  double avail = -1, gtt = -1;
  std::ifstream m("/proc/meminfo");
  std::string k;
  double v;
  std::string unit;
  while (m >> k >> v >> unit)
    if (k == "MemAvailable:") avail = v * 1024 / 1e9;
  for (int card = 0; card < 4 && gtt < 0; ++card) {
    std::ifstream g("/sys/class/drm/card" + std::to_string(card) + "/device/mem_info_gtt_used");
    double b;
    if (g >> b) gtt = b / 1e9;
  }
  char buf[96];
  std::snprintf(buf, sizeof(buf), "\"mem_available_gb\": %.2f, \"gtt_used_gb\": %.2f", avail, gtt);
  return buf;
}

std::string JsonEscape(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') o += '\\', o += static_cast<char>(c);
    else if (c == '\n') o += "\\n";
    else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof(b), "\\u%04x", c);
      o += b;
    } else o += static_cast<char>(c);
  }
  return o;
}

struct Cmp {
  bool identical = true;
  double max_abs = 0, rel_l2 = 0;
  bool argmax_equal = true;
};
Cmp Compare(const std::vector<float>& a, const std::vector<float>& b) {
  Cmp c;
  c.identical = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
  double num = 0, den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = double(a[i]) - b[i];
    c.max_abs = std::max(c.max_abs, std::fabs(d));
    num += d * d;
    den += double(a[i]) * a[i];
  }
  c.rel_l2 = std::sqrt(num / std::max(den, 1e-30));
  c.argmax_equal = Argmax(a) == Argmax(b);
  return c;
}
std::string CmpJson(const Cmp& c) {
  char buf[200];
  std::snprintf(buf, sizeof(buf), "{\"identical\": %s, \"max_abs\": %.6g, \"rel_l2\": %.6g, \"argmax_equal\": %s}",
                c.identical ? "true" : "false", c.max_abs, c.rel_l2, c.argmax_equal ? "true" : "false");
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, mode, text_path, out_path, windows_s, lengths_s, depths_s;
  std::uint32_t max_context = 262144, context = 0, targets = 0, offset = 0, length = 0, steps = 8, seed = 1,
                max_new = 24;
  bool snapshot = false;
  double ngram_cache_gb = 4.0;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--model") model = v;
    else if (a == "--mode") mode = v;
    else if (a == "--text") text_path = v;
    else if (a == "--out") out_path = v;
    else if (a == "--max-context") max_context = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--context") context = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--targets") targets = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--offset") offset = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--windows") windows_s = v;
    else if (a == "--lengths") lengths_s = v;
    else if (a == "--depths") depths_s = v;
    else if (a == "--length") length = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--steps") steps = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--seed") seed = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--max-new") max_new = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else if (a == "--snapshot") snapshot = v == "1";
    else if (a == "--ngram-cache-gb") ngram_cache_gb = std::atof(v.c_str());
    else if (a == "--kv-f16") engine::FlashNextGpu::kv_f16_only_ = v == "1";
    else if (a == "--prefill-long-chunk")
      engine::FlashNextGpu::prefill_long_chunk_ = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    else return std::fprintf(stderr, "unknown option %s\n", a.c_str()), 2;
  }
  if (model.empty() || mode.empty() || text_path.empty() || out_path.empty())
    return std::fprintf(stderr, "usage: halo_longctx --model M --mode nll|needle|split|fork --text F --out F.jsonl ...\n"), 2;
  // Items already done (resume).
  std::set<std::string> done;
  {
    std::ifstream f(out_path);
    std::string line;
    while (std::getline(f, line)) {
      const auto p = line.find("\"key\": \"");
      if (p == std::string::npos) continue;
      const auto q = line.find('"', p + 8);
      done.insert(line.substr(p + 8, q - p - 8));
    }
  }
  std::ofstream out(out_path, std::ios::app);
  const auto emit = [&](const std::string& json) {
    out << json << "\n";
    out.flush();
    std::printf("%s\n", json.substr(0, 400).c_str());
    std::fflush(stdout);
  };

  std::string error;
  auto reader = core::GgufReader::OpenFile(model, &error);
  if (!reader) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  auto weights = model::ModelWeights::Bind(*reader, &error);
  auto tok = text::Tokenizer::FromGguf(*reader, &error);
  if (!weights || !tok) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  const auto& c = weights->config;
  const auto& t = weights->ple_table;
  auto ngram = model::NgramTable::Open(reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset, t.rows,
                                       c.ple_head_dim, t.type, &error,
                                       static_cast<std::size_t>(ngram_cache_gb * (1ull << 30)));
  if (!ngram) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  std::printf("{\"event\": \"start\", %s}\n", Mem().c_str());
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  std::vector<rt::FileSource> sources;
  for (const auto& r : reader->GetMappedRegions()) sources.push_back({r.data, r.size, r.file_descriptor});
  const auto tl = clk::now();
  auto gpu = engine::FlashNextGpu::Create(*dev, *weights, ngram.get(), nullptr, max_context, &error, sources);
  if (!gpu) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  std::printf("{\"event\": \"loaded\", \"secs\": %.1f, \"max_context\": %u, \"kv_f16\": %d, \"long_chunk\": %u, %s}\n",
              Secs(tl), max_context, engine::FlashNextGpu::kv_f16_only_ ? 1 : 0,
              engine::FlashNextGpu::prefill_long_chunk_, Mem().c_str());
  std::fflush(stdout);
  const std::string tag = std::string("\"kv_f16\": ") + (engine::FlashNextGpu::kv_f16_only_ ? "1" : "0");

  const std::string text = ReadFile(text_path);
  const auto enc = tok->Encode(text, false);
  const std::vector<std::int32_t> toks(enc.begin(), enc.end());
  std::printf("{\"event\": \"text\", \"tokens\": %zu}\n", toks.size());
  const std::size_t V = c.vocab_size;
  std::vector<float> logits(V);
  const auto bytes = [&](engine::FlashNextGpu::SessionId id) {
    const auto m = gpu->SessionMemory(id);
    char b[128];
    std::snprintf(b, sizeof(b), "\"recurrent_mb\": %.1f, \"kv_mapped_mb\": %.1f", m.recurrent / 1e6, m.kv_mapped / 1e6);
    return std::string(b);
  };

  if (mode == "nll") {
    if (std::size_t{offset} + context + targets > toks.size() || context < 2)
      return std::fprintf(stderr, "text too short: %zu tokens\n", toks.size()), 2;
    const std::size_t p0 = std::size_t{offset} + context;  // first target
    for (std::uint32_t w : ParseList(windows_s)) {
      w = std::min(w, context);
      const std::string key = "nll/" + text_path.substr(text_path.find_last_of('/') + 1) + "/o" +
                              std::to_string(offset) + "/n" + std::to_string(context) + "/m" + std::to_string(targets) +
                              "/w" + std::to_string(w) + "/f16" + (engine::FlashNextGpu::kv_f16_only_ ? "1" : "0");
      if (done.contains(key)) continue;
      if (!gpu->Reset(&error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
      const std::span<const std::int32_t> head(toks.data() + p0 - w, w - 1);
      auto tp = clk::now();
      if (!head.empty() && (!gpu->Prefill(head, {}, &error) || !dev->Synchronize(&error)))
        return std::fprintf(stderr, "prefill: %s\n", error.c_str()), 1;
      const double prefill_s = Secs(tp);
      const std::string mem_after = Mem();
      std::vector<double> lp(targets);
      std::size_t correct = 0;
      std::int32_t prev = toks[p0 - 1];
      std::vector<double> step_ms;
      for (std::uint32_t j = 0; j < targets; ++j) {
        const auto ts = clk::now();
        if (!gpu->Step(prev, logits, &error)) return std::fprintf(stderr, "step: %s\n", error.c_str()), 1;
        step_ms.push_back(Secs(ts) * 1e3);
        const auto target = static_cast<std::uint32_t>(toks[p0 + j]);
        lp[j] = LogSoftmaxAt(logits, target);
        correct += Argmax(logits) == target;
        prev = toks[p0 + j];
      }
      double nll = 0;
      for (double x : lp) nll -= x;
      std::sort(step_ms.begin(), step_ms.end());
      std::ostringstream js;
      js << "{\"key\": \"" << key << "\", \"mode\": \"nll\", " << tag << ", \"window\": " << w << ", \"context\": "
         << context << ", \"targets\": " << targets << ", \"nll\": " << nll / targets
         << ", \"acc\": " << double(correct) / targets << ", \"prefill_s\": " << prefill_s
         << ", \"step_ms_median\": " << step_ms[step_ms.size() / 2] << ", " << mem_after << ", " << bytes(0)
         << ", \"lp\": [";
      for (std::size_t j = 0; j < lp.size(); ++j) js << (j ? "," : "") << lp[j];
      js << "]}";
      emit(js.str());
    }
  } else if (mode == "needle") {
    static const char* kKeys[] = {"Lisbon", "Kyoto", "Valparaiso", "Tromso", "Windhoek", "Hobart", "Cusco",
                                  "Tbilisi", "Ushuaia", "Zanzibar", "Reykjavik", "Mandalay", "Quito"};
    sampling::SamplingConfig greedy;
    greedy.temperature = 0.0f;
    std::vector<std::int32_t> stops;
    for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
      const auto ids = tok->Encode(s, true);
      if (ids.size() == 1) stops.push_back(ids[0]);
    }
    const auto encode = [&](const std::string& s, bool special) {
      const auto e = tok->Encode(s, special);
      return std::vector<std::int32_t>(e.begin(), e.end());
    };
    const auto pre = encode("<|im_start|>user\n", true);
    const auto post = encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", true);
    for (std::uint32_t L : ParseList(lengths_s))
      for (std::uint32_t d : ParseList(depths_s)) {
        const std::string key = "needle/" + text_path.substr(text_path.find_last_of('/') + 1) + "/L" +
                                std::to_string(L) + "/d" + std::to_string(d) + "/s" + std::to_string(seed) + "/f16" +
                                (engine::FlashNextGpu::kv_f16_only_ ? "1" : "0");
        if (done.contains(key)) continue;
        // The needle: a key and a 7-digit number drawn from (seed, L, d).
        std::uint64_t h = (std::uint64_t{seed} * 1000003u + L) * 1000003u + d;
        h ^= h >> 31;
        h *= 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
        const std::string k = kKeys[h % (sizeof(kKeys) / sizeof(kKeys[0]))];
        const std::string num = std::to_string(1000000 + (h >> 20) % 9000000);
        const std::string needle_s = "\nThe special magic number for " + k + " is " + num + ".\n";
        const std::string question_s = "\n\nWhat is the special magic number for " + k +
                                       " mentioned in the text above? Reply with the number only.";
        const std::size_t overhead = pre.size() + post.size() + encode(needle_s, false).size() +
                                     encode(question_s, false).size();
        if (L <= overhead + 16 || L - overhead > toks.size())
          return std::fprintf(stderr, "length %u: text has %zu tokens\n", L, toks.size()), 2;
        const std::size_t hay = L - overhead;
        // Planted after the first token at or past depth whose text ends a line or a sentence.
        std::size_t at = hay * d / 100;
        while (at < hay && at > 0) {
          const auto piece = tok->TokenBytes(static_cast<text::TokenId>(toks[at - 1]));
          if (!piece.empty() && (piece.back() == '\n' || piece.back() == '.')) break;
          ++at;
        }
        // The user turn as text (written beside --out for other engines), encoded as halo_score renders chat.
        const auto span_text = [&](std::size_t a, std::size_t b) {
          return tok->Decode(std::vector<text::TokenId>(toks.begin() + static_cast<std::ptrdiff_t>(a),
                                                        toks.begin() + static_cast<std::ptrdiff_t>(b)));
        };
        const std::string before = span_text(0, at);
        const std::string content = before + needle_s + span_text(at, hay) + question_s;
        {
          std::ofstream pf(out_path + "." + std::to_string(L) + "_" + std::to_string(d) + "_s" + std::to_string(seed) +
                               ".txt",
                           std::ios::binary);
          pf << content;
        }
        std::vector<std::int32_t> prompt(pre);
        const auto body = encode(content, false);
        prompt.insert(prompt.end(), body.begin(), body.end());
        prompt.insert(prompt.end(), post.begin(), post.end());
        at = pre.size() + encode(before, false).size();
        if (prompt.size() + max_new >= max_context)
          return std::fprintf(stderr, "prompt of %zu tokens does not fit --max-context\n", prompt.size()), 2;
        if (!gpu->Reset(&error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
        auto tp = clk::now();
        if (!gpu->Prefill(std::span<const std::int32_t>(prompt).first(prompt.size() - 1), {}, &error) ||
            !dev->Synchronize(&error))
          return std::fprintf(stderr, "prefill: %s\n", error.c_str()), 1;
        const double prefill_s = Secs(tp);
        const std::string mem_after = Mem();
        std::int32_t cur = prompt.back();
        std::vector<text::TokenId> gen;
        std::vector<double> step_ms;
        for (std::uint32_t i = 0; i < max_new; ++i) {
          engine::FlashNextGpu::SampleResult r;
          const auto ts = clk::now();
          if (!gpu->StepSample(cur, {&greedy, {}, 0}, &r, &error))
            return std::fprintf(stderr, "step: %s\n", error.c_str()), 1;
          step_ms.push_back(Secs(ts) * 1e3);
          if (std::find(stops.begin(), stops.end(), r.token) != stops.end()) break;
          gen.push_back(static_cast<text::TokenId>(r.token));
          cur = r.token;
        }
        const std::string answer = tok->Decode(gen);
        const bool ok = answer.find(num) != std::string::npos;
        std::sort(step_ms.begin(), step_ms.end());
        std::ostringstream js;
        js << "{\"key\": \"" << key << "\", \"mode\": \"needle\", " << tag << ", \"length\": " << L
           << ", \"prompt_tokens\": " << prompt.size() << ", \"depth\": " << d << ", \"needle_at\": " << at
           << ", \"key_word\": \"" << k << "\", \"number\": \"" << num << "\", \"correct\": " << (ok ? "true" : "false")
           << ", \"answer\": \"" << JsonEscape(answer) << "\", \"prefill_s\": " << prefill_s
           << ", \"prefill_tok_s\": " << (prompt.size() - 1) / prefill_s
           << ", \"step_ms_median\": " << (step_ms.empty() ? 0.0 : step_ms[step_ms.size() / 2]) << ", " << mem_after
           << ", " << bytes(0) << "}";
        emit(js.str());
      }
  } else if (mode == "split") {
    const std::string key = "split/n" + std::to_string(length) + "/k" + std::to_string(steps) + "/f16" +
                            (engine::FlashNextGpu::kv_f16_only_ ? "1" : "0");
    if (done.contains(key)) return 0;
    if (length > toks.size() || steps == 0 || steps >= length) return std::fprintf(stderr, "bad --length/--steps\n"), 2;
    const std::span<const std::int32_t> all(toks.data(), length);
    std::vector<float> a(V), b(V), cc(V), e(V);
    auto tp = clk::now();
    if (!gpu->Reset(&error) || !gpu->Prefill(all, a, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    const double one_s = Secs(tp);
    // N-K prefilled, K stepped.
    if (!gpu->Reset(&error) || !gpu->Prefill(all.first(length - steps), {}, &error))
      return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    for (std::uint32_t j = length - steps; j < length; ++j)
      if (!gpu->Step(all[j], j + 1 == length ? std::span<float>(b) : std::span<float>{}, &error))
        return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    // N-K prefilled, then K prefilled.
    if (!gpu->Reset(&error) || !gpu->Prefill(all.first(length - steps), {}, &error) ||
        !gpu->Prefill(all.subspan(length - steps), cc, &error))
      return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    // 2048-token chunks.
    const auto saved = engine::FlashNextGpu::prefill_long_chunk_;
    engine::FlashNextGpu::prefill_long_chunk_ = saved ? 0 : 8192;
    tp = clk::now();
    if (!gpu->Reset(&error) || !gpu->Prefill(all, e, &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    const double other_s = Secs(tp);
    engine::FlashNextGpu::prefill_long_chunk_ = saved;
    std::ostringstream js;
    js << "{\"key\": \"" << key << "\", \"mode\": \"split\", " << tag << ", \"length\": " << length
       << ", \"steps\": " << steps << ", \"long_chunk\": " << saved << ", \"prefill_s\": " << one_s
       << ", \"other_chunk_prefill_s\": " << other_s << ", \"prefill_then_steps\": " << CmpJson(Compare(a, b))
       << ", \"prefill_then_prefill\": " << CmpJson(Compare(a, cc)) << ", \"other_long_chunk\": " << CmpJson(Compare(a, e))
       << ", " << Mem() << "}";
    emit(js.str());
  } else if (mode == "fork") {
    const std::string key = "fork/n" + std::to_string(length) + "/k" + std::to_string(steps) + "/snap" +
                            (snapshot ? "1" : "0") + "/f16" + (engine::FlashNextGpu::kv_f16_only_ ? "1" : "0");
    if (done.contains(key)) return 0;
    if (length + steps > toks.size()) return std::fprintf(stderr, "text too short\n"), 2;
    const std::span<const std::int32_t> all(toks.data(), length + steps);
    std::vector<std::vector<float>> forked(steps, std::vector<float>(V)), fresh = forked, restored = forked;
    if (!gpu->Reset(&error) || !gpu->Prefill(all.first(length), {}, &error))
      return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    const std::string parent_mem = bytes(0);
    engine::FlashNextGpu::SessionId f;
    auto tf = clk::now();
    if (!gpu->ForkSession(0, &f, &error)) return std::fprintf(stderr, "fork: %s\n", error.c_str()), 1;
    const double fork_ms = Secs(tf) * 1e3;
    const std::string fork_mem = bytes(f);
    if (!gpu->BindSession(f, &error)) return 1;
    for (std::uint32_t j = 0; j < steps; ++j)
      if (!gpu->Step(all[length + j], forked[j], &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    if (!gpu->BindSession(0, &error) || !gpu->DropSession(f, &error)) return 1;
    // Snapshot of the parent at N (before its own reset): restore must replay the same K steps.
    double save_ms = 0, restore_ms = 0;
    std::size_t snap_same = 0;
    if (snapshot) {
      auto snap = gpu->NewSnapshot(&error);
      tf = clk::now();
      if (!snap || !gpu->Save(*snap, &error)) return std::fprintf(stderr, "save: %s\n", error.c_str()), 1;
      save_ms = Secs(tf) * 1e3;
      std::vector<float> l(V);
      for (std::uint32_t j = 0; j < steps; ++j)  // move the live state on (different tokens)
        if (!gpu->Step(all[length - 1 - j], {}, &error)) return 1;
      tf = clk::now();
      if (!gpu->Restore(*snap, &error)) return std::fprintf(stderr, "restore: %s\n", error.c_str()), 1;
      restore_ms = Secs(tf) * 1e3;
      for (std::uint32_t j = 0; j < steps; ++j)
        if (!gpu->Step(all[length + j], restored[j], &error)) return 1;
    }
    if (!gpu->Reset(&error) || !gpu->Prefill(all.first(length), {}, &error))
      return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    for (std::uint32_t j = 0; j < steps; ++j)
      if (!gpu->Step(all[length + j], fresh[j], &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    std::size_t same = 0;
    for (std::uint32_t j = 0; j < steps; ++j) {
      same += std::memcmp(forked[j].data(), fresh[j].data(), V * 4) == 0;
      if (snapshot) snap_same += std::memcmp(restored[j].data(), fresh[j].data(), V * 4) == 0;
    }
    std::ostringstream js;
    js << "{\"key\": \"" << key << "\", \"mode\": \"fork\", " << tag << ", \"length\": " << length
       << ", \"steps\": " << steps << ", \"fork_ms\": " << fork_ms << ", \"fork_equal_fresh\": " << same
       << ", \"snapshot_equal_fresh\": " << snap_same << ", \"save_ms\": " << save_ms << ", \"restore_ms\": "
       << restore_ms << ", \"parent\": {" << parent_mem << "}, \"fork\": {" << fork_mem << "}, \"last_step\": "
       << CmpJson(Compare(fresh.back(), forked.back())) << ", " << Mem() << "}";
    emit(js.str());
  } else {
    return std::fprintf(stderr, "unknown --mode %s\n", mode.c_str()), 2;
  }
  return 0;
}
