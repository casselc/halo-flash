// Qwen chat templates compiled to C++. Each function below mirrors a block of
// the Jinja source (tests/text/data/chat/*.jinja); comments quote the Jinja
// where the semantics are subtle. Adapted in part from gufo's
// src/models/qwen/chat_template.cpp (MIT, see NOTICE).
#include "text/chat_template.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "text/unicode.hpp"

namespace halo::text {

namespace {

using json::Value;
using Kind = Value::Kind;

constexpr std::string_view kFlashNextSha256 =
    "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce";
constexpr std::string_view kQwen35Sha256 =
    "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80";

constexpr std::string_view kXHighInstruction =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";
constexpr std::string_view kLowInstruction =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
    "conclusion without unnecessary elaboration.";
constexpr std::string_view kToolsIntro =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr std::string_view kToolsOutro =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- "
    "Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n- If there is no "
    "function call available, answer the question like normal with your current knowledge and do "
    "not tell the user about function calls\n</IMPORTANT>";

// ---- SHA-256 (FIPS 180-4), for pinning template sources --------------------

std::array<std::uint8_t, 32> Sha256(std::string_view data) {
  static constexpr std::uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
      0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
      0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
      0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
      0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
      0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
      0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};
  std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::string msg(data);
  const std::uint64_t bits = static_cast<std::uint64_t>(data.size()) * 8;
  msg.push_back(static_cast<char>(0x80));
  while (msg.size() % 64 != 56) msg.push_back(0);
  for (int i = 7; i >= 0; --i) msg.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
  const auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  for (std::size_t off = 0; off < msg.size(); off += 64) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = 0;
      for (int b = 0; b < 4; ++b)
        w[i] = (w[i] << 8) | static_cast<std::uint8_t>(msg[off + static_cast<std::size_t>(4 * i + b)]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  std::array<std::uint8_t, 32> out{};
  for (int i = 0; i < 32; ++i) out[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(h[i / 4] >> (24 - 8 * (i % 4)));
  return out;
}

// ---- Python/Jinja value semantics -------------------------------------------

bool IsPySpace(char32_t c) {
  return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 ||
         c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 ||
         c == 0x202f || c == 0x205f || c == 0x3000;
}

const Value* Get(const Value* obj, std::string_view key) {
  return obj && obj->is(Kind::kObject) ? obj->find(key) : nullptr;
}

bool IsString(const Value* v) { return v && v->is(Kind::kString); }

// `x == 'literal'` on a possibly undefined or non-string value.
bool Equals(const Value* v, std::string_view s) { return IsString(v) && v->str() == s; }

// Jinja truthiness (Python bool()) of a defined value; undefined is false.
bool Truthy(const Value* v) {
  if (!v) return false;
  switch (v->kind()) {
    case Kind::kNull: return false;
    case Kind::kBool: return v->boolean();
    case Kind::kInt: return v->int_digits() != "0" && v->int_digits() != "-0";
    case Kind::kFloat: return v->number() != 0;
    case Kind::kString: return !v->str().empty();
    case Kind::kArray: return !v->items().empty();
    case Kind::kObject: return !v->members().empty();
  }
  return false;
}

// Python str() of a scalar, as `{{ x }}` or `x | string` prints it. Lists and
// objects would print Python repr; unsupported (returns false).
bool PyStr(const Value& v, std::string& out) {
  switch (v.kind()) {
    case Kind::kNull: out += "None"; return true;
    case Kind::kBool: out += v.boolean() ? "True" : "False"; return true;
    case Kind::kInt: out += v.int_digits(); return true;
    case Kind::kFloat: {
      const double d = v.number();
      if (std::isnan(d)) out += "nan";
      else if (std::isinf(d)) out += d > 0 ? "inf" : "-inf";
      else out += json::PythonFloatRepr(d);
      return true;
    }
    case Kind::kString: out += v.str(); return true;
    default: return false;
  }
}

std::string_view LStripChar(std::string_view s, char c) {
  while (!s.empty() && s.front() == c) s.remove_prefix(1);
  return s;
}
std::string_view RStripChar(std::string_view s, char c) {
  while (!s.empty() && s.back() == c) s.remove_suffix(1);
  return s;
}

class Renderer {
 public:
  Renderer(ChatTemplateKind kind, const ChatTemplateOptions& options, std::string* error)
      : kind_(kind), opt_(options), error_(error) {}

  std::optional<std::string> Run(const Value& messages, const Value* tools) {
    if (!messages.is(Kind::kArray)) return Fail("messages must be an array");
    // HF rejects an empty conversation before the template runs; the
    // Flash-Next template raises 'No messages provided.' itself.
    if (messages.items().empty()) return Fail("No messages provided.");
    if (tools && !tools->is(Kind::kNull)) {
      if (!tools->is(Kind::kArray)) return Fail("tools must be an array");
      for (const auto& t : tools->items())
        if (!t.is(Kind::kObject)) return Fail("Tools should either be a JSON schema");
    }
    const bool ok = kind_ == ChatTemplateKind::kFlashNext ? FlashNext(messages.items(), tools)
                                                          : Qwen35(messages.items(), tools);
    if (!ok) return std::nullopt;
    if (opt_.add_generation_prompt) out_ += GenerationPrompt(kind_, opt_);
    return std::move(out_);
  }

 private:
  using Messages = Value::Array;

  std::nullopt_t Fail(std::string message) {
    if (error_) *error_ = std::move(message);
    return std::nullopt;
  }
  bool Error(std::string message) {
    Fail(std::move(message));
    return false;
  }

  // macro render_content(content, do_vision_count, is_system_content=false)
  bool RenderContent(const Value* content, bool count, bool system, std::string& out) {
    if (!content || content->is(Kind::kNull)) return true;
    if (content->is(Kind::kString)) {
      out += content->str();
      return true;
    }
    if (!content->is(Kind::kArray)) return Error("Unexpected content type.");
    for (const auto& item : content->items()) {
      // 'image' in item: a key test on objects, a substring test on strings,
      // membership on lists; anything else is a TypeError.
      const auto contains = [&](std::string_view word) {
        switch (item.kind()) {
          case Kind::kObject: return item.find(word) != nullptr;
          case Kind::kString: return item.str().find(word) != std::string::npos;
          case Kind::kArray:
            for (const auto& x : item.items())
              if (Equals(&x, word)) return true;
            return false;
          default: return false;
        }
      };
      if (!item.is(Kind::kObject) && !item.is(Kind::kString) && !item.is(Kind::kArray))
        return Error("argument of type is not iterable");
      const Value* type = Get(&item, "type");
      if (contains("image") || contains("image_url") || Equals(type, "image")) {
        if (system) return Error("System message cannot contain images.");
        if (count) ++images_;
        if (opt_.add_vision_id) out += "Picture " + std::to_string(images_) + ": ";
        out += "<|vision_start|><|image_pad|><|vision_end|>";
      } else if (contains("video") || Equals(type, "video")) {
        if (system) return Error("System message cannot contain videos.");
        if (count) ++videos_;
        if (opt_.add_vision_id) out += "Video " + std::to_string(videos_) + ": ";
        out += "<|vision_start|><|video_pad|><|vision_end|>";
      } else if (contains("text")) {
        // {{- item.text }}: undefined on strings and lists renders nothing.
        if (const Value* text = Get(&item, "text"); text && !PyStr(*text, out))
          return Error("content part text must be a string or scalar");
      } else {
        return Error("Unexpected item type in content.");
      }
    }
    return true;
  }

  // render_content(...)|trim
  bool Trimmed(const Value* message, bool count, bool system, std::string& out) {
    std::string raw;
    if (!RenderContent(Get(message, "content"), count, system, raw)) return false;
    out = std::string(PyStrip(raw));
    return true;
  }

  // The reversed scan for the last user turn that is not a bare tool response.
  // Returns the index, or nullopt when every user turn is a tool response.
  bool LastQueryIndex(const Messages& messages, std::optional<std::size_t>* index) {
    *index = std::nullopt;
    for (std::size_t i = messages.size(); i-- > 0;) {
      if (!Equals(Get(&messages[i], "role"), "user")) continue;
      std::string content;
      if (!Trimmed(&messages[i], false, false, content)) return false;
      if (!(content.starts_with("<tool_response>") && content.ends_with("</tool_response>"))) {
        *index = i;
        return true;
      }
    }
    return true;
  }

  void Tools(const Value* tools) {
    for (const auto& tool : tools->items()) {
      out_ += '\n';
      out_ += json::DumpPython(tool);
    }
    out_ += kToolsOutro;
  }

  // Tool calls of one assistant message; `content` is what was printed before.
  bool ToolCalls(const Value& message, std::string_view content) {
    const Value* calls = Get(&message, "tool_calls");
    // message.tool_calls and message.tool_calls is iterable and not mapping
    if (!Truthy(calls) || !(calls->is(Kind::kArray) || calls->is(Kind::kString))) return true;
    if (calls->is(Kind::kString)) return Error("Tool call is missing a function name.");
    bool first = true;
    for (const auto& item : calls->items()) {
      const Value* call = &item;
      if (const Value* fn = Get(call, "function")) call = fn;
      const Value* name = Get(call, "name");
      if (!name || name->is(Kind::kNull)) return Error("Tool call is missing a function name.");
      if (!name->is(Kind::kString)) return Error("can only concatenate str to str");
      if (first)
        out_ += PyStrip(content).empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=";
      else
        out_ += "\n<tool_call>\n<function=";
      first = false;
      out_ += name->str();
      out_ += ">\n";
      const Value* args = Get(call, "arguments");
      if (kind_ == ChatTemplateKind::kFlashNext) {
        if (args && args->is(Kind::kObject)) {
          for (const auto& [key, value] : args->members()) {
            out_ += "<parameter=" + key + ">\n";
            // args_value | string if args_value is string else args_value | tojson
            out_ += value.is(Kind::kString) ? value.str() : json::DumpPython(value);
            out_ += "\n</parameter>\n";
          }
        } else if (args && args->is(Kind::kString)) {
          if (!PyStrip(args->str()).empty())
            return Error("Tool call arguments for function \"" + name->str() +
                         "\" were passed as a JSON string.");
        } else if (args && !args->is(Kind::kNull)) {
          return Error("Tool call arguments for function \"" + name->str() +
                       "\" must be an object/mapping or a JSON string.");
        }
      } else if (args) {
        // {% for k, v in tool_call.arguments|items %}: mappings only.
        if (!args->is(Kind::kObject)) return Error("Can only get item pairs from a mapping.");
        for (const auto& [key, value] : args->members()) {
          out_ += "<parameter=" + key + ">\n";
          // tojson if mapping or (sequence and not string) else | string
          if (value.is(Kind::kObject) || value.is(Kind::kArray)) out_ += json::DumpPython(value);
          else PyStr(value, out_);
          out_ += "\n</parameter>\n";
        }
      }
      out_ += "</function>\n</tool_call>";
    }
    return true;
  }

  bool ToolMessage(const Messages& messages, std::size_t i, const std::string& content) {
    // loop.previtem and loop.previtem.role != "tool"
    if (i > 0 && Truthy(&messages[i - 1]) && !Equals(Get(&messages[i - 1], "role"), "tool"))
      out_ += "<|im_start|>user";
    out_ += "\n<tool_response>\n";
    out_ += content;
    out_ += "\n</tool_response>";
    if (i + 1 == messages.size() || !Equals(Get(&messages[i + 1], "role"), "tool"))
      out_ += "<|im_end|>\n";
    return true;
  }

  bool FlashNext(const Messages& messages, const Value* tools) {
    // Leading system/developer messages merge into one system turn.
    std::size_t num_sys = 0;
    std::string merged;
    for (const auto& m : messages) {
      const Value* role = Get(&m, "role");
      if (!Equals(role, "system") && !Equals(role, "developer")) break;
      std::string content;
      if (!Trimmed(&m, false, true, content)) return false;
      if (!content.empty()) {
        if (!merged.empty()) merged += '\n';
        merged += content;
      }
      ++num_sys;
    }
    std::string_view instructions;
    if (!opt_.enable_thinking.has_value() || *opt_.enable_thinking) {
      std::string effort = opt_.reasoning_effort.value_or("xhigh");
      if (effort == "high") effort = "xhigh";
      if (effort != "xhigh" && effort != "medium" && effort != "low")
        return Error("Unexpected reasoning effort " + *opt_.reasoning_effort +
                     ". Supported types are xhigh (default), medium, and low.");
      if (effort == "xhigh") instructions = kXHighInstruction;
      if (effort == "low") instructions = kLowInstruction;
    }
    if (tools && tools->is(Kind::kArray) && !tools->items().empty()) {
      out_ += "<|im_start|>system\n";
      if (!instructions.empty()) (out_ += instructions) += "\n\n";
      out_ += kToolsIntro;
      Tools(tools);
      if (!merged.empty()) (out_ += "\n\n") += merged;
      out_ += "<|im_end|>\n";
    } else if (!merged.empty()) {
      out_ += "<|im_start|>system\n";
      if (!instructions.empty()) (out_ += instructions) += "\n\n";
      (out_ += merged) += "<|im_end|>\n";
    } else if (!instructions.empty()) {
      ((out_ += "<|im_start|>system\n") += instructions) += "<|im_end|>\n";
    }
    std::optional<std::size_t> last_query;
    if (!LastQueryIndex(messages, &last_query)) return false;
    const std::size_t lqi = last_query.value_or(messages.size() - 1);

    for (std::size_t i = num_sys; i < messages.size(); ++i) {
      const Value& m = messages[i];
      std::string content;
      if (!Trimmed(&m, true, false, content)) return false;
      const Value* role = Get(&m, "role");
      if (Equals(role, "system") || Equals(role, "developer")) {
        return Error("System message must be at the beginning.");
      } else if (Equals(role, "user")) {
        out_ += "<|im_start|>user\n" + content + "<|im_end|>\n";
      } else if (Equals(role, "assistant")) {
        const Value* rc = Get(&m, "reasoning_content");
        const std::string_view reasoning = IsString(rc) ? PyStrip(rc->str()) : std::string_view{};
        if (!opt_.preserve_thinking.has_value() || *opt_.preserve_thinking || i > lqi) {
          out_ += "<|im_start|>assistant\n<think>\n";
          (out_ += reasoning) += "\n</think>\n\n";
        } else {
          out_ += "<|im_start|>assistant\n";
        }
        out_ += content;
        if (!ToolCalls(m, content)) return false;
        out_ += "<|im_end|>\n";
      } else if (Equals(role, "tool")) {
        ToolMessage(messages, i, content);
      } else {
        return Error("Unexpected message role.");
      }
    }
    return true;
  }

  bool Qwen35(const Messages& messages, const Value* tools) {
    const bool first_is_system = Equals(Get(&messages[0], "role"), "system");
    std::string system;
    if (first_is_system && !Trimmed(&messages[0], false, true, system)) return false;
    if (tools && tools->is(Kind::kArray) && !tools->items().empty()) {
      out_ += "<|im_start|>system\n";
      out_ += kToolsIntro;
      Tools(tools);
      if (!system.empty()) (out_ += "\n\n") += system;
      out_ += "<|im_end|>\n";
    } else if (first_is_system) {
      out_ += "<|im_start|>system\n" + system + "<|im_end|>\n";
    }
    std::optional<std::size_t> last_query;
    if (!LastQueryIndex(messages, &last_query)) return false;
    if (!last_query) return Error("No user query found in messages.");

    for (std::size_t i = 0; i < messages.size(); ++i) {
      const Value& m = messages[i];
      std::string content;
      if (!Trimmed(&m, true, false, content)) return false;
      const Value* role = Get(&m, "role");
      if (Equals(role, "system")) {
        if (i != 0) return Error("System message must be at the beginning.");
      } else if (Equals(role, "user")) {
        out_ += "<|im_start|>user\n" + content + "<|im_end|>\n";
      } else if (Equals(role, "assistant")) {
        std::string reasoning;
        std::string body = content;
        if (const Value* rc = Get(&m, "reasoning_content"); IsString(rc)) {
          reasoning = rc->str();
        } else if (const auto close = content.find("</think>"); close != std::string::npos) {
          // content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
          std::string_view head = RStripChar(std::string_view(content).substr(0, close), '\n');
          if (const auto open = head.rfind("<think>"); open != std::string_view::npos)
            head.remove_prefix(open + 7);
          reasoning = std::string(LStripChar(head, '\n'));
          // content.split('</think>')[-1].lstrip('\n')
          body = std::string(LStripChar(std::string_view(content).substr(content.rfind("</think>") + 8), '\n'));
        }
        const std::string_view r = PyStrip(reasoning);
        if (i > *last_query) {
          out_ += "<|im_start|>assistant\n<think>\n";
          (((out_ += r) += "\n</think>\n\n") += body);
        } else {
          (out_ += "<|im_start|>assistant\n") += body;
        }
        if (!ToolCalls(m, body)) return false;
        out_ += "<|im_end|>\n";
      } else if (Equals(role, "tool")) {
        ToolMessage(messages, i, content);
      } else {
        return Error("Unexpected message role.");
      }
    }
    return true;
  }

  ChatTemplateKind kind_;
  const ChatTemplateOptions& opt_;
  std::string* error_;
  std::string out_;
  int images_ = 0;
  int videos_ = 0;
};

}  // namespace

std::string_view PyStrip(std::string_view s) {
  std::size_t begin = 0;
  while (begin < s.size()) {
    const Utf8Unit u = DecodeUtf8(s, begin);
    if (!u.valid || !IsPySpace(u.value)) break;
    begin += u.length;
  }
  std::size_t end = s.size();
  while (end > begin) {
    // Step back to the start of the last code point (at most 4 bytes).
    std::size_t start = end - 1;
    while (start > begin && end - start < 4 && (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) --start;
    const Utf8Unit u = DecodeUtf8(s, start);
    if (!u.valid || start + u.length != end || !IsPySpace(u.value)) break;
    end = start;
  }
  return s.substr(begin, end - begin);
}

std::string ChatTemplateSha256(std::string_view source) {
  static const char* hex = "0123456789abcdef";
  std::string out;
  for (const auto b : Sha256(source)) {
    out.push_back(hex[b >> 4]);
    out.push_back(hex[b & 15]);
  }
  return out;
}

std::optional<ChatTemplateKind> IdentifyChatTemplate(std::string_view source) {
  const std::string h = ChatTemplateSha256(source);
  if (h == kFlashNextSha256) return ChatTemplateKind::kFlashNext;
  if (h == kQwen35Sha256) return ChatTemplateKind::kQwen35;
  return std::nullopt;
}

bool ChatOptionsFromJson(const json::Value& kwargs, ChatTemplateOptions* options, std::string* error) {
  if (!kwargs.is(Kind::kObject)) {
    if (error) *error = "chat template kwargs must be an object";
    return false;
  }
  ChatTemplateOptions o;
  for (const auto& [key, v] : kwargs.members()) {
    const auto want_bool = [&](auto assign) {
      if (!v.is(Kind::kBool)) {
        if (error) *error = "'" + key + "' must be a boolean";
        return false;
      }
      assign(v.boolean());
      return true;
    };
    bool ok = true;
    if (key == "add_generation_prompt") ok = want_bool([&](bool b) { o.add_generation_prompt = b; });
    else if (key == "enable_thinking") ok = want_bool([&](bool b) { o.enable_thinking = b; });
    else if (key == "preserve_thinking") ok = want_bool([&](bool b) { o.preserve_thinking = b; });
    else if (key == "add_vision_id") ok = want_bool([&](bool b) { o.add_vision_id = b; });
    else if (key == "reasoning_effort") {
      if (!v.is(Kind::kString)) {
        if (error) *error = "'reasoning_effort' must be a string";
        return false;
      }
      o.reasoning_effort = v.str();
    }
    if (!ok) return false;
  }
  *options = std::move(o);
  return true;
}

std::optional<std::string> RenderChat(ChatTemplateKind kind, const json::Value& messages,
                                      const json::Value* tools, const ChatTemplateOptions& options,
                                      std::string* error) {
  return Renderer(kind, options, error).Run(messages, tools);
}

std::string_view GenerationPrompt(ChatTemplateKind kind, const ChatTemplateOptions& options) {
  // Flash-Next: closed only when enable_thinking is defined and false.
  // Qwen3.5: open only when enable_thinking is defined and true.
  const bool open = kind == ChatTemplateKind::kFlashNext ? options.enable_thinking.value_or(true)
                                                         : options.enable_thinking.value_or(false);
  return open ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

}  // namespace halo::text
