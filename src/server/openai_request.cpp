#include "server/openai_request.hpp"

#include "model/variant.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace halo::server {

namespace {

using json::Value;
using Kind = json::Value::Kind;

// Parsing state: the first error wins and every helper returns false after it.
class Parser {
 public:
  Parser(const Value& body, RequestError* error) : body_(body), error_(error) {}

  bool Fail(std::string param, std::string message) {
    if (ok_) {
      ok_ = false;
      if (error_) *error_ = {std::move(message), std::move(param)};
    }
    return false;
  }
  bool ok() const { return ok_; }
  const Value& body() const { return body_; }

  /// A present, non-null member of the body.
  const Value* Get(const char* key) const {
    const Value* v = body_.find(key);
    return v && !v->is(Kind::kNull) ? v : nullptr;
  }

  bool Number(const Value& v, const std::string& param, double lo, double hi, double* out) {
    if (!v.is(Kind::kInt) && !v.is(Kind::kFloat)) return Fail(param, param + " must be a number");
    const double x = v.number();
    if (!std::isfinite(x) || x < lo || x > hi)
      return Fail(param, param + " must be between " + Fmt(lo) + " and " + Fmt(hi));
    *out = x;
    return true;
  }
  bool Integer(const Value& v, const std::string& param, std::int64_t lo, std::int64_t hi, std::int64_t* out) {
    if (!v.is(Kind::kInt)) return Fail(param, param + " must be an integer");
    const auto& d = v.int_digits();
    // Out-of-range digit strings are rejected without overflowing.
    if (d.size() > 19 || (d.size() == 19 && d[0] != '-' && d > "9223372036854775807"))
      return Fail(param, param + " is out of range");
    std::int64_t x = 0;
    try {
      x = std::stoll(d);
    } catch (const std::exception&) {
      return Fail(param, param + " is out of range");
    }
    if (x < lo || x > hi) return Fail(param, param + " must be between " + std::to_string(lo) + " and " + std::to_string(hi));
    *out = x;
    return true;
  }
  bool Bool(const Value& v, const std::string& param, bool* out) {
    if (!v.is(Kind::kBool)) return Fail(param, param + " must be a boolean");
    *out = v.boolean();
    return true;
  }
  bool String(const Value& v, const std::string& param, std::string* out) {
    if (!v.is(Kind::kString)) return Fail(param, param + " must be a string");
    *out = v.str();
    return true;
  }

 private:
  static std::string Fmt(double x) {
    std::string s = json::PythonFloatRepr(x);
    if (s.size() > 2 && s.ends_with(".0")) s.resize(s.size() - 2);
    return s;
  }
  const Value& body_;
  RequestError* error_;
  bool ok_ = true;
};

// --- shared settings ----------------------------------------------------------

bool ParseSampling(Parser& p, const RequestDefaults& d, GenerationRequest& r) {
  r.sampling = d.sampling;
  double x = 0;
  std::int64_t i = 0;
  if (const Value* v = p.Get("temperature")) {
    if (!p.Number(*v, "temperature", 0, 2, &x)) return false;
    r.sampling.temperature = static_cast<float>(x);
  }
  if (const Value* v = p.Get("top_p")) {
    if (!p.Number(*v, "top_p", 0, 1, &x)) return false;
    r.sampling.top_p = static_cast<float>(x);
  }
  // Extensions (vLLM / llama.cpp names).
  if (const Value* v = p.Get("top_k")) {
    if (!p.Integer(*v, "top_k", -1, 1 << 30, &i)) return false;
    r.sampling.top_k = i < 0 ? 0 : static_cast<std::int32_t>(i);
  }
  if (const Value* v = p.Get("min_p")) {
    if (!p.Number(*v, "min_p", 0, 1, &x)) return false;
    r.sampling.min_p = static_cast<float>(x);
  }
  if (const Value* v = p.Get("repetition_penalty")) {
    if (!p.Number(*v, "repetition_penalty", 0.01, 100, &x)) return false;
    r.sampling.repeat_penalty = static_cast<float>(x);
  }
  if (const Value* v = p.Get("presence_penalty")) {
    if (!p.Number(*v, "presence_penalty", -2, 2, &x)) return false;
    r.sampling.presence_penalty = static_cast<float>(x);
  }
  if (const Value* v = p.Get("frequency_penalty")) {
    if (!p.Number(*v, "frequency_penalty", -2, 2, &x)) return false;
    r.sampling.frequency_penalty = static_cast<float>(x);
  }
  if (const Value* v = p.Get("seed")) {
    if (!p.Integer(*v, "seed", std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max(), &i))
      return false;
    // The sampler treats a negative seed as "random"; keep the request's
    // determinism by folding it into the non-negative range.
    r.sampling.seed = i < 0 ? static_cast<std::int64_t>(static_cast<std::uint64_t>(i) >> 1) : i;
  }
  try {
    r.sampling.Validate();
  } catch (const std::exception& e) {
    return p.Fail("", e.what());
  }
  return true;
}

bool ParseMaxTokens(Parser& p, const RequestDefaults& d, std::initializer_list<const char*> keys,
                    std::uint32_t fallback, GenerationRequest& r) {
  r.max_tokens = fallback;
  for (const char* key : keys)
    if (const Value* v = p.Get(key)) {
      std::int64_t n = 0;
      if (!p.Integer(*v, key, 1, d.max_tokens_limit, &n)) return false;
      r.max_tokens = static_cast<std::uint32_t>(n);
      return true;
    }
  return true;
}

bool ParseStop(Parser& p, GenerationRequest& r) {
  const Value* v = p.Get("stop");
  if (!v) return true;
  const auto one = [&](const Value& s, const std::string& param) {
    if (!s.is(Kind::kString) || s.str().empty()) return p.Fail(param, "stop sequences must be non-empty strings");
    r.stop.push_back(s.str());
    return true;
  };
  if (v->is(Kind::kString)) return one(*v, "stop");
  if (!v->is(Kind::kArray)) return p.Fail("stop", "stop must be a string or an array of strings");
  if (v->items().size() > 4) return p.Fail("stop", "at most 4 stop sequences are supported");
  for (std::size_t i = 0; i < v->items().size(); ++i)
    if (!one(v->items()[i], "stop[" + std::to_string(i) + "]")) return false;
  return true;
}

bool ParseCommon(Parser& p, GenerationRequest& r) {
  if (const Value* v = p.Get("model"); v && !p.String(*v, "model", &r.model)) return false;
  if (const Value* v = p.Get("variant"); v && !v->is(Kind::kNull) && !p.String(*v, "variant", &r.variant)) return false;
  if (!r.variant.empty() && !model::ValidVariantName(r.variant))
    return p.Fail("variant", "variant names are 1-64 of [A-Za-z0-9._-]");
  if (r.model.find(':') != std::string::npos) {
    std::string why;
    const auto alias = model::ParseModelAlias(r.model, &why);
    if (!alias) return p.Fail("model", why);
    if (!r.variant.empty() && r.variant != alias->variant)
      return p.Fail("variant", "variant \"" + r.variant + "\" differs from the model alias \"" + r.model + "\"");
    r.variant = alias->variant;
  }
  if (const Value* v = p.Get("stream"); v && !p.Bool(*v, "stream", &r.stream)) return false;
  if (const Value* v = p.Get("user"); v && !p.String(*v, "user", &r.user)) return false;
  if (const Value* v = p.Get("parallel_tool_calls"); v && !p.Bool(*v, "parallel_tool_calls", &r.parallel_tool_calls))
    return false;
  if (const Value* v = p.Get("n")) {
    std::int64_t n = 0;
    if (!p.Integer(*v, "n", 1, 128, &n)) return false;
    if (n != 1) return p.Fail("n", "only n = 1 is supported");
  }
  return true;
}

/// "none" turns thinking off; "minimal" is the lowest effort Flash-Next has.
bool ApplyEffort(Parser& p, const Value& v, const std::string& param, text::ChatTemplateOptions& o) {
  std::string effort;
  if (!p.String(v, param, &effort)) return false;
  if (effort == "none") {
    o.enable_thinking = false;
    return true;
  }
  if (effort == "minimal") effort = "low";
  if (effort != "low" && effort != "medium" && effort != "high" && effort != "xhigh")
    return p.Fail(param, param + " must be one of none, minimal, low, medium, high, xhigh");
  o.enable_thinking = true;
  o.reasoning_effort = effort;
  return true;
}

/// tool_choice: "none" | "auto" | "required" | a named function, in the
/// Chat ({"type": "function", "function": {"name"}}) or Responses
/// ({"type": "function", "name"}) shape.
bool ParseToolChoice(Parser& p, GenerationRequest& r) {
  const Value* v = p.Get("tool_choice");
  if (!v) return true;
  auto& c = r.tool_choice;
  if (v->is(Kind::kString)) {
    const auto& s = v->str();
    if (s == "auto") c.mode = ToolChoice::Mode::kAuto;
    else if (s == "none") c.mode = ToolChoice::Mode::kNone;
    else if (s == "required") c.mode = ToolChoice::Mode::kRequired;
    else return p.Fail("tool_choice", "tool_choice must be none, auto, required or a function");
    return true;
  }
  const Value* type = v->is(Kind::kObject) ? v->find("type") : nullptr;
  if (!type || !type->is(Kind::kString) || type->str() != "function")
    return p.Fail("tool_choice", "only function tool choices are supported");
  const Value* fn = v->find("function");
  const Value* name = fn && fn->is(Kind::kObject) ? fn->find("name") : v->find("name");
  if (!name || !name->is(Kind::kString) || name->str().empty())
    return p.Fail("tool_choice", "tool_choice names no function");
  c.mode = ToolChoice::Mode::kFunction;
  c.function = name->str();
  bool found = false;
  if (r.tools.is(Kind::kArray))
    for (const auto& t : r.tools.items())
      if (const Value* f = t.find("function"); f && f->find("name") && f->find("name")->str() == c.function) found = true;
  if (!found) return p.Fail("tool_choice", "tool_choice names a function that is not in tools: " + c.function);
  return true;
}

/// json_schema formats: {"name", "schema", "strict"} (Chat nests it under
/// "json_schema"; Responses puts it beside "type").
bool ParseFormat(Parser& p, const Value& v, const std::string& param, bool nested, ResponseFormat& f) {
  const Value* type = v.is(Kind::kObject) ? v.find("type") : nullptr;
  if (!type || !type->is(Kind::kString)) return p.Fail(param, param + ".type is required");
  if (type->str() == "text") f.kind = ResponseFormat::Kind::kText;
  else if (type->str() == "json_object") f.kind = ResponseFormat::Kind::kJsonObject;
  else if (type->str() == "json_schema") {
    f.kind = ResponseFormat::Kind::kJsonSchema;
    const Value* spec = nested ? v.find("json_schema") : &v;
    const std::string sp = nested ? param + ".json_schema" : param;
    if (!spec || !spec->is(Kind::kObject)) return p.Fail(sp, sp + " is required");
    const Value* name = spec->find("name");
    if (!name || !name->is(Kind::kString) || name->str().empty()) return p.Fail(sp + ".name", "a json_schema format needs a name");
    f.name = name->str();
    const Value* schema = spec->find("schema");
    if (schema && !schema->is(Kind::kObject)) return p.Fail(sp + ".schema", "schema must be an object");
    f.schema = schema ? *schema : Value::MakeObject();
    if (const Value* s = spec->find("strict"); s && !s->is(Kind::kNull) && !p.Bool(*s, sp + ".strict", &f.strict))
      return false;
  } else {
    return p.Fail(param + ".type", "format type must be text, json_object or json_schema");
  }
  return true;
}

/// A template tool from a function's name, description and parameters.
bool TemplateTool(Parser& p, const Value& fn, const std::string& param, Value* out) {
  const Value* name = fn.find("name");
  if (!name || !name->is(Kind::kString) || name->str().empty()) return p.Fail(param + ".name", "a function tool needs a name");
  Value f = Value::MakeObject();
  f.set("name", *name);
  if (const Value* d = fn.find("description"); d && !d->is(Kind::kNull)) {
    if (!d->is(Kind::kString)) return p.Fail(param + ".description", "description must be a string");
    f.set("description", *d);
  }
  if (const Value* s = fn.find("parameters"); s && !s->is(Kind::kNull)) {
    if (!s->is(Kind::kObject)) return p.Fail(param + ".parameters", "parameters must be an object");
    f.set("parameters", *s);
  }
  *out = Value::MakeObject();
  out->set("type", Value::String("function"));
  out->set("function", std::move(f));
  return true;
}

/// A Responses namespace tool: each member function becomes the function tool "<namespace>.<name>", its
/// description prefixed with the namespace's (the model sees what the namespace is for).
bool NamespaceTool(Parser& p, const Value& t, const std::string& param, GenerationRequest& r) {
  const Value* ns = t.find("name");
  if (!ns || !ns->is(Kind::kString) || ns->str().empty()) return p.Fail(param + ".name", "a namespace tool needs a name");
  const Value* members = t.find("tools");
  if (!members || !members->is(Kind::kArray) || members->items().empty())
    return p.Fail(param + ".tools", "a namespace tool needs a non-empty tools array");
  std::string ns_description;
  if (const Value* d = t.find("description"); d && !d->is(Kind::kNull)) {
    if (!d->is(Kind::kString)) return p.Fail(param + ".description", "description must be a string");
    ns_description = d->str();
  }
  for (std::size_t j = 0; j < members->items().size(); ++j) {
    const std::string mp = param + ".tools[" + std::to_string(j) + "]";
    const Value& m = members->items()[j];
    const Value* type = m.is(Kind::kObject) ? m.find("type") : nullptr;
    if (!type || !type->is(Kind::kString) || type->str() != "function")
      return p.Fail(mp + ".type", "only function tools are supported in a namespace");
    Value tool;
    if (!TemplateTool(p, m, mp, &tool)) return false;
    Value* fn = const_cast<Value*>(tool.find("function"));
    const std::string name = fn->find("name")->str();
    const std::string flat_name = ns->str() + "." + name;
    fn->set("name", Value::String(flat_name));
    if (!ns_description.empty()) {
      const Value* d = fn->find("description");
      fn->set("description", Value::String("[" + ns->str() + ": " + ns_description + "] " + (d ? d->str() : "")));
    }
    r.namespaced_tools.push_back({flat_name, ns->str(), name});
    r.tools.push(std::move(tool));
  }
  return true;
}

bool ParseTools(Parser& p, bool flat, GenerationRequest& r) {
  const Value* v = p.Get("tools");
  if (!v) return true;
  if (!v->is(Kind::kArray)) return p.Fail("tools", "tools must be an array");
  if (v->items().empty()) return true;
  r.tools = Value::MakeArray();
  for (std::size_t i = 0; i < v->items().size(); ++i) {
    const std::string param = "tools[" + std::to_string(i) + "]";
    const Value& t = v->items()[i];
    const Value* type = t.is(Kind::kObject) ? t.find("type") : nullptr;
    if (flat && type && type->is(Kind::kString) && type->str() == "namespace") {
      if (!NamespaceTool(p, t, param, r)) return false;
      continue;
    }
    if (!type || !type->is(Kind::kString) || type->str() != "function")
      return p.Fail(param + ".type", "only function and namespace tools are supported");
    const Value* fn = flat ? &t : t.find("function");
    if (!fn || !fn->is(Kind::kObject)) return p.Fail(param + ".function", "a function tool needs a function object");
    Value tool;
    if (!TemplateTool(p, *fn, flat ? param : param + ".function", &tool)) return false;
    r.tools.push(std::move(tool));
  }
  // Flattened names must not collide with each other or with plain functions (a call could not round-trip).
  for (std::size_t i = 0; i < r.tools.items().size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (r.tools.items()[i].find("function")->find("name")->str() == r.tools.items()[j].find("function")->find("name")->str())
        return p.Fail("tools", "duplicate tool name: " + r.tools.items()[i].find("function")->find("name")->str());
  return true;
}

/// Tool-call arguments as the template wants them: an object, parsed from
/// the JSON string OpenAI clients send (an empty string is no arguments).
bool Arguments(Parser& p, const Value* v, const std::string& param, Value* out) {
  if (!v || v->is(Kind::kNull)) {
    *out = Value::MakeObject();
    return true;
  }
  if (v->is(Kind::kObject)) {
    *out = *v;
    return true;
  }
  if (!v->is(Kind::kString)) return p.Fail(param, "arguments must be a JSON string");
  if (text::PyStrip(v->str()).empty()) {
    *out = Value::MakeObject();
    return true;
  }
  std::string err;
  auto parsed = json::Parse(v->str(), &err);
  if (!parsed || !parsed->is(Kind::kObject)) return p.Fail(param, "arguments must be a JSON object: " + err);
  *out = std::move(*parsed);
  return true;
}

Value ToolCall(std::string id, Value name, Value arguments) {
  Value fn = Value::MakeObject();
  fn.set("name", std::move(name));
  fn.set("arguments", std::move(arguments));
  Value call = Value::MakeObject();
  if (!id.empty()) call.set("id", Value::String(std::move(id)));
  call.set("type", Value::String("function"));
  call.set("function", std::move(fn));
  return call;
}

/// Text content: a string, or text parts of the given types joined in
/// template form ([{"type": "text", "text"}]).
bool Content(Parser& p, const Value& v, const std::string& param, std::initializer_list<const char*> text_types,
             Value* out) {
  if (v.is(Kind::kString) || v.is(Kind::kNull)) {
    *out = v;
    return true;
  }
  if (!v.is(Kind::kArray)) return p.Fail(param, "content must be a string or an array of parts");
  Value parts = Value::MakeArray();
  for (std::size_t i = 0; i < v.items().size(); ++i) {
    const std::string pp = param + "[" + std::to_string(i) + "]";
    const Value& part = v.items()[i];
    const Value* type = part.is(Kind::kObject) ? part.find("type") : nullptr;
    if (!type || !type->is(Kind::kString)) return p.Fail(pp + ".type", "a content part needs a type");
    bool is_text = false;
    for (const char* t : text_types) is_text |= type->str() == t;
    if (!is_text) return p.Fail(pp + ".type", "unsupported content part type: " + type->str());
    const Value* text = part.find("text");
    if (!text || !text->is(Kind::kString)) return p.Fail(pp + ".text", "a text part needs text");
    Value q = Value::MakeObject();
    q.set("type", Value::String("text"));
    q.set("text", *text);
    parts.push(std::move(q));
  }
  *out = std::move(parts);
  return true;
}

bool ValidRole(const std::string& role) {
  return role == "system" || role == "developer" || role == "user" || role == "assistant" || role == "tool";
}

// --- Chat Completions -----------------------------------------------------------

bool ChatMessages(Parser& p, GenerationRequest& r) {
  const Value* v = p.Get("messages");
  if (!v || !v->is(Kind::kArray) || v->items().empty()) return p.Fail("messages", "messages must be a non-empty array");
  for (std::size_t i = 0; i < v->items().size(); ++i) {
    const std::string param = "messages[" + std::to_string(i) + "]";
    const Value& m = v->items()[i];
    if (!m.is(Kind::kObject)) return p.Fail(param, "a message must be an object");
    const Value* role = m.find("role");
    if (!role || !role->is(Kind::kString) || !ValidRole(role->str()))
      return p.Fail(param + ".role", "role must be system, developer, user, assistant or tool");
    Value out = Value::MakeObject();
    out.set("role", *role);
    if (const Value* c = m.find("content")) {
      Value content;
      if (!Content(p, *c, param + ".content", {"text"}, &content)) return false;
      out.set("content", std::move(content));
    } else if (role->str() != "assistant") {
      return p.Fail(param + ".content", "content is required");
    }
    if (role->str() == "assistant") {
      // vLLM's newer "reasoning" field and the older "reasoning_content".
      for (const char* key : {"reasoning_content", "reasoning"})
        if (const Value* rc = m.find(key); rc && !rc->is(Kind::kNull)) {
          if (!rc->is(Kind::kString)) return p.Fail(param + "." + key, std::string(key) + " must be a string");
          out.set("reasoning_content", *rc);
        }
      if (const Value* calls = m.find("tool_calls"); calls && !calls->is(Kind::kNull)) {
        if (!calls->is(Kind::kArray)) return p.Fail(param + ".tool_calls", "tool_calls must be an array");
        Value list = Value::MakeArray();
        for (std::size_t j = 0; j < calls->items().size(); ++j) {
          const std::string cp = param + ".tool_calls[" + std::to_string(j) + "]";
          const Value& c = calls->items()[j];
          const Value* fn = c.is(Kind::kObject) ? c.find("function") : nullptr;
          const Value* name = fn && fn->is(Kind::kObject) ? fn->find("name") : nullptr;
          if (!name || !name->is(Kind::kString) || name->str().empty())
            return p.Fail(cp + ".function.name", "a tool call needs a function name");
          Value args;
          if (!Arguments(p, fn->find("arguments"), cp + ".function.arguments", &args)) return false;
          const Value* id = c.find("id");
          list.push(ToolCall(id && id->is(Kind::kString) ? id->str() : "", *name, std::move(args)));
        }
        out.set("tool_calls", std::move(list));
      }
    }
    if (role->str() == "tool")
      if (const Value* id = m.find("tool_call_id"); id && id->is(Kind::kString)) out.set("tool_call_id", *id);
    r.messages.push(std::move(out));
  }
  return true;
}

// --- Responses --------------------------------------------------------------------

// Builds template messages from Responses items, merging each run of
// assistant-side items into one assistant message.
class ResponsesInput {
 public:
  explicit ResponsesInput(Parser& p) : p_(p) {}

  bool Instructions(const Value& v) {
    std::string s;
    if (!p_.String(v, "instructions", &s)) return false;
    Value m = Value::MakeObject();
    m.set("role", Value::String("system"));
    m.set("content", Value::String(std::move(s)));
    Push(std::move(m));
    return true;
  }

  bool Item(const Value& item, const std::string& param) {
    if (!item.is(Kind::kObject)) return p_.Fail(param, "an input item must be an object");
    const Value* type = item.find("type");
    const std::string t = type && type->is(Kind::kString) ? type->str() : "message";
    if (t == "message") return Message(item, param);
    if (t == "reasoning") return Reasoning(item, param);
    if (t == "function_call") return FunctionCall(item, param);
    if (t == "function_call_output") return FunctionCallOutput(item, param);
    if (t == "item_reference") return p_.Fail(param + ".type", "item references need stored responses, not supported yet");
    return p_.Fail(param + ".type", "unsupported input item type: " + t);
  }

  Value Finish() {
    Close();
    return std::move(messages_);
  }

 private:
  bool Message(const Value& item, const std::string& param) {
    const Value* role = item.find("role");
    if (!role || !role->is(Kind::kString) || role->str() == "tool" || !ValidRole(role->str()))
      return p_.Fail(param + ".role", "role must be user, system, developer or assistant");
    const Value* c = item.find("content");
    if (!c) return p_.Fail(param + ".content", "content is required");
    Value content;
    if (!Content(p_, *c, param + ".content", {"input_text", "output_text", "text"}, &content)) return false;
    if (role->str() == "assistant") {
      Value& a = Assistant();
      if (a.find("content")) {
        // A second assistant message in a run: its own turn.
        Close();
        Assistant();
      }
      open_->set("content", std::move(content));
      return true;
    }
    Close();
    Value m = Value::MakeObject();
    m.set("role", *role);
    m.set("content", std::move(content));
    Push(std::move(m));
    return true;
  }

  bool Reasoning(const Value& item, const std::string& param) {
    // The full reasoning text when given, else its summary.
    std::string text;
    bool any = false;
    for (const char* key : {"content", "summary"}) {
      const Value* parts = item.find(key);
      if (!parts || parts->is(Kind::kNull)) continue;
      if (!parts->is(Kind::kArray)) return p_.Fail(param + "." + key, std::string(key) + " must be an array");
      if (parts->items().empty()) continue;
      for (std::size_t i = 0; i < parts->items().size(); ++i) {
        const Value* t = parts->items()[i].is(Kind::kObject) ? parts->items()[i].find("text") : nullptr;
        if (!t || !t->is(Kind::kString))
          return p_.Fail(param + "." + key + "[" + std::to_string(i) + "].text", "a reasoning part needs text");
        if (i) text += "\n\n";
        text += t->str();
      }
      any = true;
      break;
    }
    if (const Value* enc = item.find("encrypted_content"); !any && enc && !enc->is(Kind::kNull))
      return p_.Fail(param, "encrypted reasoning is not supported");
    // encrypted_content: null (Codex always sends the field) is no encrypted reasoning.
    Value& a = Assistant();
    if (a.find("content") || a.find("tool_calls")) {
      // Reasoning after output starts the next turn.
      Close();
      Assistant();
    }
    if (const Value* prev = open_->find("reasoning_content")) text = prev->str() + "\n\n" + text;
    open_->set("reasoning_content", Value::String(std::move(text)));
    return true;
  }

  bool FunctionCall(const Value& item, const std::string& param) {
    const Value* name = item.find("name");
    if (!name || !name->is(Kind::kString) || name->str().empty()) return p_.Fail(param + ".name", "a function call needs a name");
    const Value* call_id = item.find("call_id");
    if (!call_id || !call_id->is(Kind::kString) || call_id->str().empty())
      return p_.Fail(param + ".call_id", "a function call needs a call_id");
    Value args;
    if (!Arguments(p_, item.find("arguments"), param + ".arguments", &args)) return false;
    // A namespaced call (Codex): rendered under the flat name its tool was given.
    Value rendered_name = *name;
    if (const Value* ns = item.find("namespace"); ns && !ns->is(Kind::kNull)) {
      if (!ns->is(Kind::kString)) return p_.Fail(param + ".namespace", "namespace must be a string");
      if (!ns->str().empty()) rendered_name = Value::String(ns->str() + "." + name->str());
    }
    Assistant();
    Value* calls = const_cast<Value*>(open_->find("tool_calls"));
    if (!calls) {
      open_->set("tool_calls", Value::MakeArray());
      calls = const_cast<Value*>(open_->find("tool_calls"));
    }
    calls->push(ToolCall(call_id->str(), std::move(rendered_name), std::move(args)));
    return true;
  }

  bool FunctionCallOutput(const Value& item, const std::string& param) {
    const Value* call_id = item.find("call_id");
    if (!call_id || !call_id->is(Kind::kString) || call_id->str().empty())
      return p_.Fail(param + ".call_id", "a function call output needs a call_id");
    const Value* out = item.find("output");
    if (!out) return p_.Fail(param + ".output", "output is required");
    Value content;
    if (!Content(p_, *out, param + ".output", {"input_text", "output_text", "text"}, &content)) return false;
    Close();
    Value m = Value::MakeObject();
    m.set("role", Value::String("tool"));
    m.set("content", std::move(content));
    m.set("tool_call_id", *call_id);
    Push(std::move(m));
    return true;
  }

  Value& Assistant() {
    if (!open_) {
      Value m = Value::MakeObject();
      m.set("role", Value::String("assistant"));
      pending_ = std::move(m);
      open_ = &pending_;
    }
    return *open_;
  }
  void Close() {
    if (!open_) return;
    // A turn of only reasoning or only calls renders with empty content.
    if (!pending_.find("content")) pending_.set("content", Value::String(""));
    messages_.push(std::move(pending_));
    open_ = nullptr;
  }
  void Push(Value m) { messages_.push(std::move(m)); }

  Parser& p_;
  Value messages_ = Value::MakeArray();
  Value pending_;
  Value* open_ = nullptr;
};

}  // namespace

const GenerationRequest::NamespacedTool* GenerationRequest::FindNamespaced(std::string_view emitted) const {
  if (namespaced_tools.empty()) return nullptr;
  for (const auto& t : namespaced_tools)
    if (t.flat == emitted) return &t;
  // A bare member name: only when no plain tool has it and exactly one namespace defines it.
  if (tools.is(Kind::kArray))
    for (const auto& t : tools.items())
      if (t.find("function")->find("name")->str() == emitted) return nullptr;
  const NamespacedTool* found = nullptr;
  for (const auto& t : namespaced_tools)
    if (t.name == emitted) {
      if (found) return nullptr;
      found = &t;
    }
  return found;
}

std::optional<GenerationRequest> ParseChatRequest(const json::Value& body, const RequestDefaults& d,
                                                  RequestError* error) {
  Parser p(body, error);
  if (!body.is(Kind::kObject)) return p.Fail("", "the request body must be a JSON object"), std::nullopt;
  GenerationRequest r;
  r.api = Api::kChat;
  r.template_options.add_generation_prompt = true;
  if (!ParseCommon(p, r) || !ParseSampling(p, d, r) ||
      !ParseMaxTokens(p, d, {"max_completion_tokens", "max_tokens"}, d.max_tokens, r) || !ParseStop(p, r) ||
      !ChatMessages(p, r) || !ParseTools(p, /*flat=*/false, r) || !ParseToolChoice(p, r))
    return std::nullopt;
  if (const Value* v = p.Get("chat_template_kwargs")) {
    std::string err;
    if (!v->is(Kind::kObject) || !text::ChatOptionsFromJson(*v, &r.template_options, &err))
      return p.Fail("chat_template_kwargs", "invalid chat_template_kwargs: " + err), std::nullopt;
    r.template_options.add_generation_prompt = true;
  }
  if (const Value* v = p.Get("reasoning_effort"); v && !ApplyEffort(p, *v, "reasoning_effort", r.template_options))
    return std::nullopt;
  if (const Value* v = p.Get("response_format"); v && !ParseFormat(p, *v, "response_format", true, r.format))
    return std::nullopt;
  if (const Value* v = p.Get("logprobs"); v && !p.Bool(*v, "logprobs", &r.logprobs)) return std::nullopt;
  if (const Value* v = p.Get("top_logprobs")) {
    std::int64_t n = 0;
    if (!p.Integer(*v, "top_logprobs", 0, 20, &n)) return std::nullopt;
    if (!r.logprobs && n > 0) return p.Fail("top_logprobs", "top_logprobs requires logprobs"), std::nullopt;
    r.top_logprobs = static_cast<std::uint32_t>(n);
  }
  if (const Value* so = p.Get("stream_options")) {
    const Value* u = so->is(Kind::kObject) ? so->find("include_usage") : nullptr;
    if (u && !u->is(Kind::kNull) && !p.Bool(*u, "stream_options.include_usage", &r.include_usage)) return std::nullopt;
  }
  return r;
}

std::optional<GenerationRequest> ParseResponsesRequest(const json::Value& body, const RequestDefaults& d,
                                                       RequestError* error) {
  Parser p(body, error);
  if (!body.is(Kind::kObject)) return p.Fail("", "the request body must be a JSON object"), std::nullopt;
  GenerationRequest r;
  r.api = Api::kResponses;
  r.template_options.add_generation_prompt = true;
  if (!ParseCommon(p, r) || !ParseSampling(p, d, r) ||
      !ParseMaxTokens(p, d, {"max_output_tokens"}, d.max_tokens, r) || !ParseTools(p, /*flat=*/true, r) ||
      !ParseToolChoice(p, r))
    return std::nullopt;
  ResponsesInput in(p);
  if (const Value* v = p.Get("instructions")) {
    if (!in.Instructions(*v)) return std::nullopt;
    r.has_instructions = true;
  }
  const Value* input = p.Get("input");
  if (!input) return p.Fail("input", "input is required"), std::nullopt;
  if (input->is(Kind::kString)) {
    Value item = Value::MakeObject();
    item.set("role", Value::String("user"));
    item.set("content", *input);
    if (!in.Item(item, "input")) return std::nullopt;
  } else if (input->is(Kind::kArray)) {
    for (std::size_t i = 0; i < input->items().size(); ++i)
      if (!in.Item(input->items()[i], "input[" + std::to_string(i) + "]")) return std::nullopt;
  } else {
    return p.Fail("input", "input must be a string or an array of items"), std::nullopt;
  }
  r.messages = in.Finish();
  if (r.messages.items().empty()) return p.Fail("input", "input has no items"), std::nullopt;
  if (const Value* v = p.Get("reasoning")) {
    if (!v->is(Kind::kObject)) return p.Fail("reasoning", "reasoning must be an object"), std::nullopt;
    if (const Value* e = v->find("effort"); e && !e->is(Kind::kNull) && !ApplyEffort(p, *e, "reasoning.effort", r.template_options))
      return std::nullopt;
  }
  if (const Value* v = p.Get("text")) {
    if (!v->is(Kind::kObject)) return p.Fail("text", "text must be an object"), std::nullopt;
    if (const Value* f = v->find("format"); f && !f->is(Kind::kNull) && !ParseFormat(p, *f, "text.format", false, r.format))
      return std::nullopt;
  }
  if (const Value* v = p.Get("store"); v && !p.Bool(*v, "store", &r.store)) return std::nullopt;
  if (const Value* v = p.Get("previous_response_id"); v && !p.String(*v, "previous_response_id", &r.previous_response_id))
    return std::nullopt;
  if (const Value* v = p.Get("truncation")) {
    std::string t;
    if (!p.String(*v, "truncation", &t)) return std::nullopt;
    if (t != "auto" && t != "disabled") return p.Fail("truncation", "truncation must be auto or disabled"), std::nullopt;
    r.truncation_auto = t == "auto";
  }
  if (const Value* v = p.Get("metadata")) {
    if (!v->is(Kind::kObject) || v->members().size() > 16)
      return p.Fail("metadata", "metadata must be an object of at most 16 strings"), std::nullopt;
    for (const auto& [k, val] : v->members())
      if (!val.is(Kind::kString) || k.size() > 64 || val.str().size() > 512)
        return p.Fail("metadata." + k, "metadata values must be strings (keys <= 64, values <= 512 chars)"), std::nullopt;
    r.metadata = *v;
  }
  return r;
}

std::optional<GenerationRequest> ParseCompletionsRequest(const json::Value& body, const RequestDefaults& d,
                                                         RequestError* error) {
  Parser p(body, error);
  if (!body.is(Kind::kObject)) return p.Fail("", "the request body must be a JSON object"), std::nullopt;
  GenerationRequest r;
  r.api = Api::kCompletions;
  // OpenAI's legacy default is 16 tokens.
  if (!ParseCommon(p, r) || !ParseSampling(p, d, r) || !ParseMaxTokens(p, d, {"max_tokens"}, 16, r) ||
      !ParseStop(p, r))
    return std::nullopt;
  const Value* prompt = p.Get("prompt");
  if (prompt && prompt->is(Kind::kArray) && prompt->items().size() == 1 && prompt->items()[0].is(Kind::kString))
    prompt = &prompt->items()[0];
  if (!prompt || !prompt->is(Kind::kString))
    return p.Fail("prompt", "prompt must be a string (batches and token arrays are not supported)"), std::nullopt;
  r.prompt = prompt->str();
  for (const char* unsupported : {"suffix", "echo"})
    if (const Value* v = p.Get(unsupported); v && !(v->is(Kind::kBool) && !v->boolean()))
      return p.Fail(unsupported, std::string(unsupported) + " is not supported"), std::nullopt;
  if (const Value* v = p.Get("best_of")) {
    std::int64_t n = 0;
    if (!p.Integer(*v, "best_of", 1, 1, &n)) return std::nullopt;
  }
  if (const Value* v = p.Get("logprobs")) {
    std::int64_t n = 0;
    if (!p.Integer(*v, "logprobs", 0, 5, &n)) return std::nullopt;
    r.logprobs = true;
    r.top_logprobs = static_cast<std::uint32_t>(n);
  }
  return r;
}

json::Value ErrorBody(const RequestError& e) {
  Value err = Value::MakeObject();
  err.set("message", Value::String(e.message));
  err.set("type", Value::String("invalid_request_error"));
  err.set("param", e.param.empty() ? Value::Null() : Value::String(e.param));
  err.set("code", Value::Null());
  Value body = Value::MakeObject();
  body.set("error", std::move(err));
  return body;
}

}  // namespace halo::server
