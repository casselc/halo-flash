#include "scope/evaluators.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <regex>

#include "scope/json_schema.hpp"

namespace halo::scope {

namespace {

using json::Value;
using Kind = json::Value::Kind;

bool Flag(const Value& c, const char* key, bool dflt) {
  const Value* v = c.find(key);
  return v && v->is(Kind::kBool) ? v->boolean() : dflt;
}

double Num(const Value& c, const char* key, double dflt) {
  const Value* v = c.find(key);
  return v && (v->is(Kind::kInt) || v->is(Kind::kFloat)) ? v->number() : dflt;
}

std::string Str(const Value& c, const char* key, const std::string& dflt = {}) {
  const Value* v = c.find(key);
  return v && v->is(Kind::kString) ? v->str() : dflt;
}

std::string AsText(const Value& v) { return v.is(Kind::kString) ? v.str() : CanonicalJson(v); }

std::string Trim(std::string s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return {};
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::string Normalize(std::string s, const Value& c) {
  if (Flag(c, "first_line", false)) {
    s = Trim(std::move(s));
    s = s.substr(0, s.find('\n'));
  }
  if (Flag(c, "collapse_whitespace", false)) {
    std::string o;
    bool space = false;
    for (char ch : s) {
      if (std::isspace(static_cast<unsigned char>(ch))) {
        space = true;
        continue;
      }
      if (space && !o.empty()) o += ' ';
      space = false;
      o += ch;
    }
    s = o;
  }
  if (Flag(c, "trim", true)) s = Trim(std::move(s));
  if (Flag(c, "case_insensitive", false))
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

std::string StripFences(const std::string& s) {
  std::string t = Trim(s);
  if (t.rfind("```", 0) == 0) {
    const auto nl = t.find('\n');
    const auto end = t.rfind("```");
    if (nl != std::string::npos && end != std::string::npos && end > nl) t = Trim(t.substr(nl + 1, end - nl - 1));
  }
  return t;
}

std::string StripThink(const std::string& s) {
  const auto e = s.rfind("</think>");
  return e == std::string::npos ? s : s.substr(e + 8);
}

// Numbers in text: optional sign, digits with optional thousands commas,
// optional fraction and exponent.
std::vector<double> NumbersIn(const std::string& s) {
  static const std::regex re(R"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+)");
  std::vector<double> out;
  for (auto it = std::sregex_iterator(s.begin(), s.end(), re); it != std::sregex_iterator(); ++it) {
    std::string m = it->str();
    m.erase(std::remove(m.begin(), m.end(), ','), m.end());
    out.push_back(std::strtod(m.c_str(), nullptr));
  }
  return out;
}

std::optional<double> ExpectedNumber(const Value& e) {
  if (e.is(Kind::kInt) || e.is(Kind::kFloat)) return e.number();
  if (e.is(Kind::kString)) {
    auto ns = NumbersIn(e.str());
    if (ns.size() == 1) return ns[0];
  }
  return std::nullopt;
}

EvalOutcome Exact(const Value& c, const EvalContext& x) {
  if (x.expected.is(Kind::kNull)) return EvalOutcome::Failed("BOOLEAN", "the item has no expected output");
  if (Flag(c, "json", false)) {
    std::string err;
    auto got = json::Parse(StripFences(x.output), &err);
    if (!got) return EvalOutcome::Boolean(false, "output is not JSON: " + err);
    const bool eq = CanonicalJson(*got) == CanonicalJson(x.expected);
    return EvalOutcome::Boolean(eq);
  }
  if (!x.expected.is(Kind::kString)) return EvalOutcome::Failed("BOOLEAN", "expected output is not a string");
  const bool eq = Normalize(x.output, c) == Normalize(x.expected.str(), c);
  return EvalOutcome::Boolean(eq);
}

EvalOutcome Regex(const Value& c, const EvalContext& x) {
  std::string pattern = Str(c, "pattern");
  if (Flag(c, "from_expected", false)) {
    if (!x.expected.is(Kind::kString)) return EvalOutcome::Failed("BOOLEAN", "expected output is not a pattern string");
    pattern = x.expected.str();
  }
  try {
    auto flags = std::regex::ECMAScript;
    if (Flag(c, "case_insensitive", false)) flags |= std::regex::icase;
    std::regex re(pattern, flags);
    bool m = Flag(c, "full_match", false) ? std::regex_match(x.output, re) : std::regex_search(x.output, re);
    if (Flag(c, "negate", false)) m = !m;
    return EvalOutcome::Boolean(m);
  } catch (const std::exception& e) {
    return EvalOutcome::Failed("BOOLEAN", std::string("bad pattern: ") + e.what());
  }
}

EvalOutcome Schema(const Value& c, const EvalContext& x) {
  const Value* schema = c.find("schema");
  std::string err;
  auto doc = json::Parse(Flag(c, "strip_fences", true) ? StripFences(x.output) : x.output, &err);
  if (!doc) return EvalOutcome::Boolean(false, "output is not JSON: " + err);
  std::string why;
  const bool ok = ValidateJsonSchema(*schema, *doc, &why);
  return EvalOutcome::Boolean(ok, why);
}

EvalOutcome Numeric(const Value& c, const EvalContext& x) {
  const auto e = ExpectedNumber(x.expected);
  if (!e) return EvalOutcome::Failed("BOOLEAN", "expected output is not a number");
  const auto ns = NumbersIn(StripThink(x.output));
  if (ns.empty()) return EvalOutcome::Boolean(false, "no number in the output");
  const double got = Str(c, "pick", "last") == "first" ? ns.front() : ns.back();
  const double tol = std::max(Num(c, "abs_tol", 0), Num(c, "rel_tol", 0) * std::fabs(*e));
  const bool ok = std::fabs(got - *e) <= tol;
  char buf[128];
  std::snprintf(buf, sizeof buf, "got %.17g, expected %.17g, tolerance %.17g", got, *e, tol);
  return EvalOutcome::Boolean(ok, buf);
}

// Validates a judge verdict's "score" against the evaluator's score config.
EvalOutcome Verdict(const Value& score_cfg, const Value& verdict) {
  const std::string type = Str(score_cfg, "type", "numeric");
  const Value* s = verdict.find("score");
  const std::string reasoning = Str(verdict, "reasoning");
  if (type == "boolean") {
    if (s->is(Kind::kBool)) return EvalOutcome::Boolean(s->boolean(), reasoning);
    if (s->is(Kind::kString) && (s->str() == "true" || s->str() == "false"))
      return EvalOutcome::Boolean(s->str() == "true", reasoning);
    if (s->is(Kind::kInt) && (s->number() == 0 || s->number() == 1)) return EvalOutcome::Boolean(s->number() == 1, reasoning);
    return EvalOutcome::Failed("BOOLEAN", "judge score is not a boolean");
  }
  if (type == "categorical") {
    if (!s->is(Kind::kString)) return EvalOutcome::Failed("CATEGORICAL", "judge score is not a string");
    if (const Value* cats = score_cfg.find("categories")) {
      bool ok = false;
      for (const auto& k : cats->items()) ok = ok || (k.is(Kind::kString) && k.str() == s->str());
      if (!ok) return EvalOutcome::Failed("CATEGORICAL", "judge category " + s->str() + " is not allowed");
    }
    return EvalOutcome::Categorical(s->str(), reasoning);
  }
  double v;
  if (s->is(Kind::kInt) || s->is(Kind::kFloat)) v = s->number();
  else if (s->is(Kind::kString) && NumbersIn(s->str()).size() == 1) v = NumbersIn(s->str())[0];
  else return EvalOutcome::Failed("NUMERIC", "judge score is not a number");
  if (!std::isfinite(v)) return EvalOutcome::Failed("NUMERIC", "judge score is not finite");
  if (const Value* lo = score_cfg.find("min"); lo && v < lo->number())
    return EvalOutcome::Failed("NUMERIC", "judge score below min");
  if (const Value* hi = score_cfg.find("max"); hi && v > hi->number())
    return EvalOutcome::Failed("NUMERIC", "judge score above max");
  return EvalOutcome::Numeric(v, reasoning);
}

std::string JudgeDataType(const Value& score_cfg) {
  const std::string t = Str(score_cfg, "type", "numeric");
  return t == "boolean" ? "BOOLEAN" : t == "categorical" ? "CATEGORICAL" : "NUMERIC";
}

EvalOutcome Judge(const EvaluatorDef& def, const Value& c, const EvalContext& x, const EvalEnv& env) {
  const Value* p = c.find("prompt");
  const Value& score_cfg = *c.find("score");
  const std::string dt = JudgeDataType(score_cfg);
  const std::string pname = Str(*p, "name");
  const auto pversion = static_cast<std::int64_t>(Num(*p, "version", 0));
  auto prompt = env.store ? env.store->PromptByVersion(pname, pversion) : std::nullopt;
  auto with_judge = [&](EvalOutcome o, const std::string& trace, const std::string& model, const std::string& raw) {
    o.row.judge_model = model;
    o.row.judge_prompt = pname;
    o.row.judge_prompt_version = pversion;
    o.row.judge_trace_id = trace;
    o.row.raw_output = raw;
    return o;
  };
  if (!prompt) return EvalOutcome::Failed(dt, "judge prompt " + pname + " version " + std::to_string(pversion) + " not found");
  std::string err;
  auto rendered = RenderTemplate(prompt->template_text, TemplateVars(x, true), &err);
  if (!rendered) return EvalOutcome::Failed(dt, "judge prompt: " + err);
  Value messages = Value::MakeArray();
  auto pconfig = json::Parse(prompt->config_json, &err);
  if (pconfig && pconfig->is(Kind::kObject)) {
    if (const Value* sys = pconfig->find("system"); sys && sys->is(Kind::kString)) {
      Value m = Value::MakeObject();
      m.set("role", Value::String("system"));
      m.set("content", Value::String(sys->str()));
      messages.push(std::move(m));
    }
  }
  Value m = Value::MakeObject();
  m.set("role", Value::String("user"));
  m.set("content", Value::String(*rendered));
  messages.push(std::move(m));
  Value body = Value::MakeObject();
  if (const Value* req = c.find("request"); req && req->is(Kind::kObject))
    for (const auto& [k, v] : req->members()) body.set(k, v);
  const std::string model = Str(c, "model");
  body.set("model", Value::String(model));
  body.set("messages", std::move(messages));
  body.set("temperature", Value::Float(Num(c, "temperature", 0)));
  body.set("max_tokens", Value::Int(static_cast<std::int64_t>(Num(c, "max_tokens", 512))));
  body.set("stream", Value::Bool(false));
  const std::string trace = EvaluatorTraceId(x.run_id, x.item->id, x.attempt, def.name);
  HttpHeaders headers = env.headers;
  headers.emplace_back("X-Halo-Trace-Id", trace);
  headers.emplace_back("X-Halo-Session-Id", x.run_id);
  headers.emplace_back("X-Halo-Prompt", pname + "@" + std::to_string(pversion));
  const std::string url = Str(c, "endpoint", env.default_endpoint);
  if (url.empty()) return EvalOutcome::Failed(dt, "no judge endpoint configured");
  const ChatReply r = CallChat(*env.poster, url, body, headers);
  if (r.status == ChatReply::Status::kTransient) return EvalOutcome::Transient("judge " + def.name + ": " + r.error);
  if (r.status == ChatReply::Status::kFailed)
    return with_judge(EvalOutcome::Failed(dt, "judge call failed: " + r.error), trace, model, r.raw);
  const std::string judge_model = r.model.empty() ? model : r.model;
  auto verdict = ExtractVerdict(r.text);
  if (!verdict) return with_judge(EvalOutcome::Failed(dt, "no JSON verdict with a score in the judge reply"), trace,
                                  judge_model, r.text);
  return with_judge(Verdict(score_cfg, *verdict), trace, judge_model, r.text);
}

EvalOutcome Decider(const EvaluatorDef& def, const Value& c, const EvalContext& x, const EvalEnv& env) {
  const Value& q = *c.find("question");
  const std::string qname = Str(q, "name");
  const std::string qtype = Str(q, "type", "noul");
  const bool thresholded = c.find("threshold") != nullptr;
  const std::string dt = qtype == "choice" ? "CATEGORICAL" : (qtype == "noul" && thresholded) ? "BOOLEAN" : "NUMERIC";
  Value state;
  std::string err;
  if (const Value* st = c.find("state"); st && st->is(Kind::kString)) {
    auto r = RenderTemplate(st->str(), TemplateVars(x, true), &err);
    if (!r) return EvalOutcome::Failed(dt, "decider state: " + err);
    state = Value::String(*r);
  } else {
    state = Value::MakeObject();
    state.set("input", x.input);
    state.set("output", Value::String(x.output));
    state.set("expected", x.expected);
  }
  Value body = Value::MakeObject();
  body.set("state", std::move(state));
  // The strands-decider schema keys questions by name.
  Value spec = Value::MakeObject();
  for (const auto& [k, v] : q.members())
    if (k != "name") spec.set(k, v);
  Value qs = Value::MakeObject();
  qs.set(qname, std::move(spec));
  body.set("questions", std::move(qs));
  if (const std::string model = Str(c, "model"); !model.empty()) body.set("model", Value::String(model));
  const std::string trace = EvaluatorTraceId(x.run_id, x.item->id, x.attempt, def.name);
  HttpHeaders headers = env.headers;
  headers.emplace_back("X-Halo-Trace-Id", trace);
  const std::string url = Str(c, "endpoint", env.decider_endpoint.empty() ? env.default_endpoint : env.decider_endpoint);
  if (url.empty()) return EvalOutcome::Failed(dt, "no decider endpoint configured");
  const HttpResult h = env.poster->Post(url, "/v1/systemone", json::DumpPython(body), headers);
  auto with_raw = [&](EvalOutcome o, const std::string& model) {
    o.row.judge_model = model;
    o.row.judge_trace_id = trace;
    o.row.raw_output = h.body;
    return o;
  };
  if (!h.transport_ok) return EvalOutcome::Transient("decider " + def.name + ": " + h.error);
  if (h.status == 429 || h.status >= 500)
    return EvalOutcome::Transient("decider " + def.name + ": HTTP " + std::to_string(h.status));
  if (h.status != 200) return with_raw(EvalOutcome::Failed(dt, "decider HTTP " + std::to_string(h.status)), "");
  auto v = json::Parse(h.body, &err);
  const Value* ans = v ? v->find("answers") : nullptr;
  const Value* a = ans ? ans->find(qname) : nullptr;
  const std::string model = v ? Str(*v, "model", Str(c, "model", "strands-decider")) : "";
  if (!a) return with_raw(EvalOutcome::Failed(dt, "no answer for question " + qname), model);
  char comment[96];
  std::snprintf(comment, sizeof comment, "confidence %.4f", Num(*a, "confidence", 0));
  if (qtype == "choice") return with_raw(EvalOutcome::Categorical(Str(*a, "choice"), comment), model);
  if (qtype == "score") return with_raw(EvalOutcome::Numeric(Num(*a, "score", 0), comment), model);
  const double p = Num(*a, "noul", 0);
  if (thresholded) return with_raw(EvalOutcome::Boolean(p >= Num(c, "threshold", 0.5), comment), model);
  return with_raw(EvalOutcome::Numeric(p, comment), model);
}

}  // namespace

EvalOutcome EvalOutcome::Numeric(double v, std::string comment) {
  EvalOutcome o;
  o.status = Status::kScored;
  o.row.data_type = "NUMERIC";
  o.row.value = v;
  o.row.comment = std::move(comment);
  return o;
}

EvalOutcome EvalOutcome::Boolean(bool v, std::string comment) {
  EvalOutcome o = Numeric(v ? 1.0 : 0.0, std::move(comment));
  o.row.data_type = "BOOLEAN";
  o.row.string_value = v ? "true" : "false";
  return o;
}

EvalOutcome EvalOutcome::Categorical(std::string v, std::string comment) {
  EvalOutcome o = Numeric(0, std::move(comment));
  o.row.data_type = "CATEGORICAL";
  o.row.string_value = std::move(v);
  return o;
}

EvalOutcome EvalOutcome::Failed(std::string data_type, std::string error) {
  EvalOutcome o;
  o.status = Status::kFailed;
  o.row.data_type = std::move(data_type);
  o.row.error = std::move(error);
  return o;
}

EvalOutcome EvalOutcome::Transient(std::string error) {
  EvalOutcome o;
  o.status = Status::kTransient;
  o.transient_error = std::move(error);
  return o;
}

void CodeHooks::Register(const std::string& name, Fn fn) {
  std::lock_guard<std::mutex> lk(mu_);
  hooks_[name] = std::move(fn);
}

std::optional<CodeHooks::Fn> CodeHooks::Find(const std::string& name) const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto it = hooks_.find(name);
  if (it == hooks_.end()) return std::nullopt;
  return it->second;
}

std::string EvaluatorTraceId(std::string_view run_id, std::string_view item_id, int attempt,
                             std::string_view evaluator) {
  const std::string a = std::to_string(attempt);
  return FramedHash({"evaluator-trace.v1", run_id, item_id, a, evaluator});
}

std::optional<std::string> RenderTemplate(const std::string& tmpl, const std::map<std::string, std::string>& vars,
                                          std::string* err) {
  std::string out;
  std::size_t i = 0;
  while (true) {
    const auto open = tmpl.find("{{", i);
    if (open == std::string::npos) {
      out.append(tmpl, i);
      return out;
    }
    const auto close = tmpl.find("}}", open + 2);
    if (close == std::string::npos) return *err = "unclosed {{ in template", std::nullopt;
    out.append(tmpl, i, open - i);
    const std::string name = Trim(tmpl.substr(open + 2, close - open - 2));
    const auto it = vars.find(name);
    if (it == vars.end()) return *err = "unknown template variable {{" + name + "}}", std::nullopt;
    out += it->second;
    i = close + 2;
  }
}

std::map<std::string, std::string> TemplateVars(const EvalContext& x, bool with_output) {
  std::map<std::string, std::string> v;
  if (x.input.is(Kind::kObject))
    for (const auto& [k, val] : x.input.members()) v[k] = AsText(val);
  v["input"] = AsText(x.input);
  v["expected"] = x.expected.is(Kind::kNull) ? std::string() : AsText(x.expected);
  v["metadata"] = CanonicalJson(x.metadata);
  if (with_output) v["output"] = x.output;
  return v;
}

std::optional<json::Value> ExtractVerdict(const std::string& reply) {
  const std::string text = StripThink(reply);
  // Scan candidate objects from the last '{' backwards: the verdict usually
  // ends the reply, after any reasoning that may itself contain braces.
  for (std::size_t pos = text.rfind('{'); pos != std::string::npos; pos = pos == 0 ? std::string::npos : text.rfind('{', pos - 1)) {
    int depth = 0;
    bool in_str = false, esc = false;
    for (std::size_t j = pos; j < text.size(); ++j) {
      const char ch = text[j];
      if (in_str) {
        if (esc) esc = false;
        else if (ch == '\\') esc = true;
        else if (ch == '"') in_str = false;
        continue;
      }
      if (ch == '"') in_str = true;
      else if (ch == '{') ++depth;
      else if (ch == '}' && --depth == 0) {
        std::string err;
        auto v = json::Parse(std::string_view(text).substr(pos, j - pos + 1), &err);
        if (v && v->is(Kind::kObject) && v->find("score")) return v;
        break;
      }
    }
  }
  return std::nullopt;
}

std::optional<std::string> PrepareEvaluatorConfig(EvalStore& store, const std::string& kind,
                                                  const std::string& config_json, std::string* err) {
  auto c = json::Parse(config_json.empty() ? "{}" : config_json, err);
  if (!c || !c->is(Kind::kObject)) return *err = "config must be a JSON object", std::nullopt;
  auto need_string = [&](const Value& o, const char* k) {
    const Value* v = o.find(k);
    return v && v->is(Kind::kString) && !v->str().empty();
  };
  if (kind == "exact" || kind == "numeric") {
    // all options optional
  } else if (kind == "regex") {
    if (!Flag(*c, "from_expected", false)) {
      if (!need_string(*c, "pattern")) return *err = "regex needs a pattern (or from_expected)", std::nullopt;
      try {
        std::regex re(Str(*c, "pattern"), std::regex::ECMAScript);
      } catch (const std::exception& e) {
        return *err = std::string("bad pattern: ") + e.what(), std::nullopt;
      }
    }
  } else if (kind == "json_schema") {
    const Value* s = c->find("schema");
    if (!s) return *err = "json_schema needs a schema", std::nullopt;
    if (!CheckJsonSchema(*s, err)) return std::nullopt;
  } else if (kind == "code") {
    if (!need_string(*c, "hook")) return *err = "code needs a hook name", std::nullopt;
  } else if (kind == "llm_judge") {
    const Value* p = c->find("prompt");
    if (!p || !p->is(Kind::kObject) || !need_string(*p, "name")) return *err = "llm_judge needs prompt.name", std::nullopt;
    std::optional<PromptVersion> pv;
    if (const Value* v = p->find("version"); v && v->is(Kind::kInt)) pv = store.PromptByVersion(p->find("name")->str(), static_cast<std::int64_t>(v->number()));
    else pv = store.PromptByLabel(p->find("name")->str(), Str(*p, "label", "latest"));
    if (!pv) return *err = "no such judge prompt version", std::nullopt;
    Value pinned = Value::MakeObject();
    pinned.set("name", Value::String(pv->name));
    pinned.set("version", Value::Int(pv->version));
    c->set("prompt", std::move(pinned));
    if (!need_string(*c, "model")) return *err = "llm_judge needs a model", std::nullopt;
    const Value* s = c->find("score");
    if (!s || !s->is(Kind::kObject)) return *err = "llm_judge needs a score config", std::nullopt;
    const std::string t = Str(*s, "type", "numeric");
    if (t != "numeric" && t != "boolean" && t != "categorical") return *err = "score.type must be numeric, boolean or categorical", std::nullopt;
    if (t == "categorical") {
      const Value* cats = s->find("categories");
      if (cats && !cats->is(Kind::kArray)) return *err = "score.categories must be an array", std::nullopt;
    }
  } else if (kind == "decider") {
    const Value* q = c->find("question");
    if (!q || !q->is(Kind::kObject) || !need_string(*q, "name")) return *err = "decider needs question.name", std::nullopt;
    const std::string t = Str(*q, "type", "noul");
    if (t != "noul" && t != "choice" && t != "score") return *err = "question.type must be noul, choice or score", std::nullopt;
  } else {
    return *err = "unknown evaluator kind " + kind, std::nullopt;
  }
  return CanonicalJson(*c);
}

EvalOutcome RunEvaluator(const EvaluatorDef& def, const EvalContext& ctx, const EvalEnv& env) {
  std::string err;
  auto c = json::Parse(def.config_json, &err);
  EvalOutcome o;
  if (!c) o = EvalOutcome::Failed("NUMERIC", "bad evaluator config: " + err);
  else if (def.kind == "exact") o = Exact(*c, ctx);
  else if (def.kind == "regex") o = Regex(*c, ctx);
  else if (def.kind == "json_schema") o = Schema(*c, ctx);
  else if (def.kind == "numeric") o = Numeric(*c, ctx);
  else if (def.kind == "llm_judge") o = Judge(def, *c, ctx, env);
  else if (def.kind == "decider") o = Decider(def, *c, ctx, env);
  else if (def.kind == "code") {
    const auto hook = env.hooks ? env.hooks->Find(Str(*c, "hook")) : std::nullopt;
    if (!hook) o = EvalOutcome::Transient("code hook " + Str(*c, "hook") + " is not registered");
    else {
      try {
        o = (*hook)(ctx);
      } catch (const std::exception& e) {
        o = EvalOutcome::Failed("NUMERIC", std::string("code hook threw: ") + e.what());
      }
    }
  } else {
    o = EvalOutcome::Failed("NUMERIC", "unknown evaluator kind " + def.kind);
  }
  o.row.item_id = ctx.item->id;
  o.row.name = def.name;
  o.row.evaluator = def.name;
  o.row.evaluator_version = def.version;
  if (o.row.data_type.empty()) o.row.data_type = "NUMERIC";
  return o;
}

}  // namespace halo::scope
