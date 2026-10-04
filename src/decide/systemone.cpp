#include "decide/systemone.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>

#include "text/unicode.hpp"

namespace halo::decide {

namespace {

using json::Value;

std::vector<char32_t> Decode(std::string_view s) {
  std::vector<char32_t> out;
  for (std::size_t i = 0; i < s.size();) {
    const auto u = text::DecodeUtf8(s, i);
    out.push_back(u.value);
    i += u.length;
  }
  return out;
}

std::string Encode(const std::vector<char32_t>& cps, std::size_t a, std::size_t b) {
  std::string out;
  for (std::size_t i = a; i < b; ++i) text::AppendUtf8(out, cps[i]);
  return out;
}

std::string PyStrip(std::string_view s) {
  const auto cps = Decode(s);
  std::size_t a = 0, b = cps.size();
  while (a < b && PyIsSpace(cps[a])) ++a;
  while (b > a && PyIsSpace(cps[b - 1])) --b;
  return Encode(cps, a, b);
}

// " ".join(s.split())
std::string PyCollapse(std::string_view s) {
  const auto cps = Decode(s);
  std::string out;
  std::size_t i = 0;
  while (i < cps.size()) {
    while (i < cps.size() && PyIsSpace(cps[i])) ++i;
    const std::size_t start = i;
    while (i < cps.size() && !PyIsSpace(cps[i])) ++i;
    if (i > start) {
      if (!out.empty()) out.push_back(' ');
      out += Encode(cps, start, i);
    }
  }
  return out;
}

bool IsContent(const Value& v) {
  return v.is(Value::Kind::kString) || v.is(Value::Kind::kObject) || v.is(Value::Kind::kArray);
}

// Validates one question against one kind; returns an error message or "".
std::string TryKind(const Value& q, Kind kind, Question& out) {
  const Value* type = q.find("type");
  if (type && !(type->is(Value::Kind::kString) && type->str() == KindName(kind))) return "type mismatch";
  const Value* ins = q.find("instructions");
  if (!ins) return "instructions is required";
  if (!IsContent(*ins)) return "instructions must be a string, object or array";
  out.kind = kind;
  out.instructions = *ins;
  out.criteria.clear();
  const Value* crit = q.find("criteria");
  switch (kind) {
    case Kind::kNoul:
      if (!crit || crit->is(Value::Kind::kNull)) return "";
      if (!crit->is(Value::Kind::kObject)) return "noul criteria must be an object";
      for (const auto& [k, v] : crit->members()) {
        if (k != "true" && k != "false") return "noul criteria keys must be a subset of {'true', 'false'}";
        if (!v.is(Value::Kind::kNull) && !IsContent(v)) return "noul criterion must be content or null";
        out.criteria.emplace_back(k, v.is(Value::Kind::kNull) ? std::nullopt : std::optional<Value>(v));
      }
      return "";
    case Kind::kChoice:
      if (!crit || !crit->is(Value::Kind::kObject)) return "choice criteria must be an object";
      for (const auto& [k, v] : crit->members()) {
        if (!v.is(Value::Kind::kNull) && !IsContent(v)) return "choice option must be content or null";
        out.criteria.emplace_back(k, v.is(Value::Kind::kNull) ? std::nullopt : std::optional<Value>(v));
      }
      if (out.criteria.size() < 2) return "choice requires at least 2 options";
      if (out.criteria.size() > kMaxChoiceOptions) return "choice allows at most 255 options";
      return "";
    case Kind::kScore:
      if (!crit || !crit->is(Value::Kind::kArray)) return "score criteria must be an array of strings";
      for (const auto& v : crit->items()) {
        if (!v.is(Value::Kind::kString)) return "score levels must be strings";
        out.criteria.emplace_back("", v);
      }
      if (out.criteria.size() < kMinScoreLevels || out.criteria.size() > kMaxScoreLevels)
        return "score requires between 2 and 10 levels";
      return "";
  }
  return "unknown kind";
}

float LogSumExpF32(std::span<const float> x) {
  float m = -INFINITY;
  for (float v : x) m = std::max(m, v);
  float s = 0;
  for (float v : x) s += std::exp(v - m);
  return m + std::log(s);
}

}  // namespace

const char* KindName(Kind k) noexcept {
  switch (k) {
    case Kind::kNoul: return "noul";
    case Kind::kChoice: return "choice";
    case Kind::kScore: return "score";
  }
  return "?";
}

bool PyIsSpace(char32_t c) noexcept {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 ||
         c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
         c == 0x205F || c == 0x3000;
}

std::size_t CodepointLength(std::string_view s) { return Decode(s).size(); }

std::optional<Request> ParseRequest(const Value& body, std::string* error) {
  const auto fail = [&](std::string m) -> std::optional<Request> {
    if (error) *error = std::move(m);
    return std::nullopt;
  };
  if (!body.is(Value::Kind::kObject)) return fail("request must be an object");
  Request r;
  const Value* state = body.find("state");
  if (!state) return fail("state is required");
  if (!IsContent(*state)) return fail("state must be a string, object or array");
  r.state = *state;
  const Value* qs = body.find("questions");
  if (!qs || !qs->is(Value::Kind::kObject)) return fail("questions must be an object");
  if (qs->members().empty()) return fail("questions must not be empty");
  for (const auto& [name, q] : qs->members()) {
    if (!q.is(Value::Kind::kObject)) return fail("question " + name + " must be an object");
    Question out;
    out.name = name;
    std::string why;
    bool ok = false;
    for (Kind k : {Kind::kNoul, Kind::kChoice, Kind::kScore}) {
      const auto e = TryKind(q, k, out);
      if (e.empty()) { ok = true; break; }
      if (e != "type mismatch") why = e;
    }
    if (!ok) return fail("question " + name + ": " + (why.empty() ? "unknown question type" : why));
    r.questions.push_back(std::move(out));
  }
  if (const Value* m = body.find("model")) {
    if (!m->is(Value::Kind::kString)) return fail("model must be a string");
    r.model = m->str();
  }
  if (const Value* im = body.find("images")) {
    if (!im->is(Value::Kind::kArray)) return fail("images must be an array of strings");
    for (const auto& v : im->items()) {
      if (!v.is(Value::Kind::kString)) return fail("images must be an array of strings");
      r.images.push_back(v.str());
    }
  }
  return r;
}

std::string RenderContent(const Value* c) {
  if (!c || c->is(Value::Kind::kNull)) return "";
  if (c->is(Value::Kind::kString)) return PyStrip(c->str());
  return json::DumpPython(*c, 2);
}

std::string RenderState(const Value& state) { return "<state>\n" + RenderContent(&state) + "\n</state>\n"; }

RenderedQuestion RenderQuestion(const Question& q, const std::vector<std::size_t>* order) {
  RenderedQuestion rq;
  rq.kind = q.kind;
  std::vector<std::pair<std::string, std::string>> pairs;  // name, rendered description
  const char* header = "";
  switch (q.kind) {
    case Kind::kNoul: {
      std::optional<Value> desc[2];
      bool given[2] = {false, false};
      for (const auto& [k, v] : q.criteria) {
        const int i = k == "true" ? 1 : 0;
        desc[i] = v;
        given[i] = true;
      }
      const char* def[2] = {"the statement does not hold for this state", "the statement holds for this state"};
      const char* labels[2] = {"false", "true"};
      for (int i = 0; i < 2; ++i) {
        const Value d = given[i] ? (desc[i] ? *desc[i] : Value::Null()) : Value::String(def[i]);
        pairs.emplace_back(labels[i], RenderContent(&d));
      }
      header = "Decide whether the statement is true of the state.";
      break;
    }
    case Kind::kChoice:
      for (const auto& [k, v] : q.criteria) pairs.emplace_back(k, v ? RenderContent(&*v) : std::string());
      header = "Select exactly one option.";
      break;
    case Kind::kScore:
      for (std::size_t i = 0; i < q.criteria.size(); ++i) pairs.emplace_back(std::to_string(i), q.criteria[i].second->str());
      header = "Rate the state against the ordered levels below (lowest first).";
      break;
  }
  if (order) {
    std::vector<std::pair<std::string, std::string>> permuted;
    for (std::size_t i : *order) permuted.push_back(pairs.at(i));
    pairs = std::move(permuted);
  }
  const std::string prefix = std::string("<question type=\"") + KindName(q.kind) + "\">\n" + header + "\n" +
                             RenderContent(&q.instructions) + "\n<options>\n";
  std::string block;
  std::size_t cursor = 0;
  const std::size_t base = CodepointLength(prefix);
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const std::string desc = PyCollapse(pairs[i].second);
    std::string line = std::to_string(i + 1) + ". " + pairs[i].first;
    if (!desc.empty()) line += " — " + desc;
    if (i) block.push_back('\n');
    block += line;
    const std::size_t len = CodepointLength(line);
    rq.option_spans.emplace_back(base + cursor, base + cursor + len);
    cursor += len + 1;
    rq.slot_labels.push_back(pairs[i].first);
    rq.slot_descriptions.push_back(PyStrip(pairs[i].second));
  }
  rq.text = prefix + block + "\n</options>\n</question>\n<answer>";
  return rq;
}

std::vector<float> SlotProbabilities(std::span<const float> logits, float temperature) {
  std::vector<float> scaled(logits.size());
  const float t = std::max(temperature, 1e-6F);
  for (std::size_t i = 0; i < logits.size(); ++i) scaled[i] = logits[i] / t;
  const float lse = LogSumExpF32(scaled);
  std::vector<float> p(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) p[i] = std::exp(scaled[i] - lse);
  return p;
}

double PyRound(double x, int nd) {
  if (!std::isfinite(x)) return x;
  // Python rounds the exact binary value to nd decimals (ties to even); a
  // fixed-precision to_chars is the same correctly rounded conversion.
  char buf[512];
  const auto r = std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::fixed, nd);
  double out = 0;
  std::from_chars(buf, r.ptr, out);
  return out;
}

double DeriveConfidence(std::span<const double> p) {
  const std::size_t n = p.size();
  if (n <= 1) return 1.0;
  const double pmax = *std::max_element(p.begin(), p.end());
  return std::max(0.0, std::min(1.0, (n * pmax - 1.0) / (n - 1.0)));
}

double DeriveScoreConfidence(std::span<const double> probs, double smoothing) {
  const std::size_t n = probs.size();
  if (n <= 1) return 1.0;
  double total = 0;
  for (double x : probs) total += x;
  if (total <= 0) return 0.0;
  double mean = 0, var = 0;
  for (std::size_t i = 0; i < n; ++i) mean += static_cast<double>(i) * (probs[i] / total);
  for (std::size_t i = 0; i < n; ++i) var += (probs[i] / total) * (i - mean) * (i - mean);
  const double sigma = std::sqrt(var);
  const double sigma_max = (n - 1) / 2.0;
  const double floor = smoothing > 0 ? std::sqrt(smoothing) : 0.0;
  if (sigma_max <= floor) return sigma <= floor ? 1.0 : 0.0;
  return std::max(0.0, std::min(1.0, (sigma_max - sigma) / (sigma_max - floor)));
}

Answer ReadAnswer(const RenderedQuestion& rq, std::span<const float> probs_f32, double smoothing) {
  std::vector<double> probs(probs_f32.begin(), probs_f32.end());  // torch .tolist(): float32 -> double
  Answer a;
  a.kind = rq.kind;
  const auto n = rq.slot_labels.size();
  switch (rq.kind) {
    case Kind::kNoul: {
      const auto it = std::find(rq.slot_labels.begin(), rq.slot_labels.end(), "true");
      a.noul = PyRound(probs[static_cast<std::size_t>(it - rq.slot_labels.begin())], 4);
      break;
    }
    case Kind::kChoice: {
      std::size_t best = 0;
      for (std::size_t i = 1; i < n; ++i)
        if (probs[i] > probs[best]) best = i;
      a.choice = rq.slot_labels[best];
      for (std::size_t i = 0; i < n; ++i) a.probabilities.emplace_back(rq.slot_labels[i], PyRound(probs[i], 4));
      a.confidence = PyRound(DeriveConfidence(probs), 4);
      break;
    }
    case Kind::kScore: {
      std::vector<double> ordered(n);
      double expected = 0;
      for (std::size_t i = 0; i < n; ++i) {
        const auto level = static_cast<std::size_t>(std::stoul(rq.slot_labels[i]));
        ordered.at(level) = probs[i];
        expected += static_cast<double>(level) * probs[i];
      }
      std::vector<std::pair<std::size_t, std::string>> legend;
      for (std::size_t i = 0; i < n; ++i)
        legend.emplace_back(std::stoul(rq.slot_labels[i]), rq.slot_descriptions[i]);
      std::sort(legend.begin(), legend.end());
      for (auto& [lvl, d] : legend) a.legend.emplace_back(std::to_string(lvl), d);
      a.score = PyRound(expected, 4);
      for (std::size_t i = 0; i < n; ++i) a.probabilities.emplace_back(std::to_string(i), PyRound(ordered[i], 4));
      a.confidence = PyRound(DeriveScoreConfidence(ordered, smoothing), 4);
      break;
    }
  }
  return a;
}

std::optional<FittedPrompt> FitTokens(const text::Tokenizer& tok, std::string_view state_text,
                                      std::span<const RenderedQuestion> questions, const FitOptions& opt,
                                      std::string* error) {
  FittedPrompt out;
  std::vector<text::Tokenizer::Encoded> enc;
  std::size_t longest = 0;
  for (const auto& rq : questions) {
    enc.push_back(tok.EncodeWithOffsets(rq.text));
    longest = std::max(longest, enc.back().ids.size());
  }
  const auto frac_cap = static_cast<std::size_t>(static_cast<double>(opt.max_length) * opt.max_question_fraction);
  const std::size_t reserve = std::min(longest, std::max<std::size_t>(1, frac_cap));
  for (std::size_t q = 0; q < questions.size(); ++q) {
    auto& e = enc[q];
    const std::size_t cut = e.ids.size() > reserve ? e.ids.size() - reserve : 0;
    std::vector<text::TokenId> ids(e.ids.begin() + static_cast<std::ptrdiff_t>(cut), e.ids.end());
    std::vector<std::pair<std::size_t, std::size_t>> offs(e.offsets.begin() + static_cast<std::ptrdiff_t>(cut),
                                                          e.offsets.end());
    std::vector<std::size_t> option_index;
    for (const auto& [a, b] : questions[q].option_spans) {
      std::ptrdiff_t last = -1;
      for (std::size_t j = 0; j < offs.size(); ++j) {
        const auto [lo, hi] = offs[j];
        if (hi <= lo) continue;
        if (lo >= a && hi <= b) last = static_cast<std::ptrdiff_t>(j);
      }
      if (last < 0) {
        if (error) *error = "option span has no tokens left; the prompt was truncated through its option list";
        return std::nullopt;
      }
      option_index.push_back(static_cast<std::size_t>(last));
    }
    out.question_ids.push_back(std::move(ids));
    out.option_index.push_back(std::move(option_index));
  }
  const std::size_t state_budget = std::max<std::size_t>(1, opt.max_length > reserve ? opt.max_length - reserve : 1);
  out.state_ids = tok.Encode(state_text);
  if (out.state_ids.size() > state_budget) out.state_ids.resize(state_budget);
  return out;
}

json::Value AnswerToJson(const Answer& a) {
  Value o = Value::MakeObject();
  o.set("type", Value::String(KindName(a.kind)));
  if (a.kind == Kind::kNoul) {
    o.set("noul", Value::Float(a.noul));
    return o;
  }
  if (a.kind == Kind::kChoice) {
    o.set("choice", Value::String(a.choice));
  } else {
    o.set("score", Value::Float(a.score));
    Value legend = Value::MakeObject();
    for (const auto& [k, v] : a.legend) legend.set(k, Value::String(v));
    o.set("legend", std::move(legend));
  }
  Value probs = Value::MakeObject();
  for (const auto& [k, v] : a.probabilities) probs.set(k, Value::Float(v));
  o.set("probabilities", std::move(probs));
  o.set("confidence", Value::Float(a.confidence));
  return o;
}

}  // namespace halo::decide
