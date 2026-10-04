// Streaming tool-call parser tests.
//
// Directed: the call formats, framing, partial tags at every split, malformed
// and truncated calls, the size limit.
// Properties (Hegel, run under the sanitizer presets):
//   * render then parse: an assistant turn rendered by either chat template
//     parses back to its reasoning, content and tool calls;
//   * chunking invariance: any split of the same text gives the same events
//     (adjacent text events of one kind merged);
//   * accounting: on generated, malformed and truncated streams nothing
//     crashes and the events' bytes concatenate to the input, with no tool
//     markup inside content or reasoning;
//   * a stateful (swarm) machine interleaving pushes, split tags, raw bytes
//     and stream ends against the one-shot parse.

#include <hegel/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "text/chat_template.hpp"
#include "text/tool_call_parser.hpp"

namespace gs = hegel::generators;
namespace hs = hegel::stateful;
using halo::json::Value;
using namespace halo::text;
using EK = StreamEventKind;

namespace {

void Check(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}

const char* KindName(EK k) {
  switch (k) {
    case EK::kContent: return "content";
    case EK::kReasoning: return "reasoning";
    case EK::kToolCall: return "tool_call";
    case EK::kFraming: return "framing";
    case EK::kError: return "error";
  }
  return "?";
}

std::string Describe(const std::vector<StreamEvent>& events) {
  std::string out;
  for (const auto& e : events) {
    out += KindName(e.kind);
    out += "[";
    for (char c : e.text) out += c == '\n' ? std::string("\\n") : std::string(1, c);
    out += "]";
    if (e.kind == EK::kToolCall) out += " " + e.call.name + halo::json::DumpPython(e.call.arguments);
    if (e.kind == EK::kError) out += " (" + e.error + ")";
    out += "\n";
  }
  return out;
}

// Adjacent content/reasoning/framing events merged: what chunking may change.
std::vector<StreamEvent> Normalize(const std::vector<StreamEvent>& events) {
  std::vector<StreamEvent> out;
  for (const auto& e : events) {
    const bool text_kind = e.kind == EK::kContent || e.kind == EK::kReasoning || e.kind == EK::kFraming;
    if (text_kind && !out.empty() && out.back().kind == e.kind) out.back().text += e.text;
    else out.push_back(e);
  }
  return out;
}

std::string Joined(const std::vector<StreamEvent>& events, EK kind) {
  std::string out;
  for (const auto& e : events)
    if (e.kind == kind) out += e.text;
  return out;
}

std::string AllBytes(const std::vector<StreamEvent>& events) {
  std::string out;
  for (const auto& e : events) out += e.text;
  return out;
}

std::vector<ParsedToolCall> Calls(const std::vector<StreamEvent>& events) {
  std::vector<ParsedToolCall> out;
  for (const auto& e : events)
    if (e.kind == EK::kToolCall) out.push_back(e.call);
  return out;
}

std::vector<StreamEvent> ParseChunks(const std::vector<std::string>& chunks, const ToolCallParserOptions& o) {
  ToolCallStreamParser p(o);
  std::vector<StreamEvent> events;
  for (const auto& c : chunks) p.Push(c, &events);
  p.Finish(&events);
  return events;
}

std::vector<std::string> SplitAt(std::string_view s, std::vector<std::size_t> cuts) {
  std::sort(cuts.begin(), cuts.end());
  std::vector<std::string> out;
  std::size_t at = 0;
  for (std::size_t c : cuts) {
    c = std::min(c, s.size());
    out.emplace_back(s.substr(at, c - at));
    at = c;
  }
  out.emplace_back(s.substr(at));
  return out;
}

// Structural checks that hold for every input.
void CheckEvents(std::string_view input, const std::vector<StreamEvent>& events, const ToolCallParserOptions& o) {
  Check(AllBytes(events) == input, "bytes not accounted for:\n" + Describe(events));
  for (const auto& e : events) {
    Check(!e.text.empty(), "empty event");
    switch (e.kind) {
      case EK::kContent:
        Check(e.text.find("<tool_call>") == std::string::npos, "call opener in content");
        break;
      case EK::kReasoning:
        Check(e.text.find("</think>") == std::string::npos, "</think> in reasoning");
        if (o.tool_calls_in_reasoning) Check(e.text.find("<tool_call>") == std::string::npos, "opener in reasoning");
        break;
      case EK::kToolCall:
        Check(e.text.starts_with("<tool_call>") && e.text.ends_with("</tool_call>"), "call bytes");
        Check(!e.call.name.empty() && e.call.arguments.is(Value::Kind::kObject), "call fields");
        break;
      case EK::kError:
        Check(e.text.starts_with("<tool_call>") && !e.error.empty(), "error event");
        break;
      case EK::kFraming:
        break;
    }
  }
}

ToolCallParserOptions Opts(StreamStart start) {
  ToolCallParserOptions o;
  o.start = start;
  return o;
}

Value Obj(std::initializer_list<std::pair<const char*, Value>> members) {
  Value v = Value::MakeObject();
  for (const auto& [k, x] : members) v.set(k, x);
  return v;
}

}  // namespace

// ---- directed ---------------------------------------------------------------

TEST(ToolCallParser, ReasoningContentAndXmlCall) {
  const std::string text =
      "Need the weather.\n</think>\n\nLet me check.\n\n<tool_call>\n<function=get_weather>\n"
      "<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>";
  const auto events = ParseAssistantText(text, Opts(StreamStart::kReasoning));
  ASSERT_EQ(AllBytes(events), text);
  EXPECT_EQ(Joined(events, EK::kReasoning), "Need the weather.");
  EXPECT_EQ(Joined(events, EK::kContent), "Let me check.");
  ASSERT_EQ(Calls(events).size(), 1u) << Describe(events);
  EXPECT_EQ(Calls(events)[0].name, "get_weather");
  EXPECT_EQ(Calls(events)[0].arguments, Obj({{"city", Value::String("Paris")}, {"days", Value::Int(2)}}));
}

TEST(ToolCallParser, SchemaKeepsStringsAndTypesTheRest) {
  const std::string text =
      "<tool_call>\n<function=f>\n<parameter=s>\n42\n</parameter>\n<parameter=n>\n42\n</parameter>\n"
      "<parameter=o>\n{\"a\": [1.5, null]}\n</parameter>\n<parameter=u>\ntrue\n</parameter>\n"
      "<parameter=q>\n\"quoted\"\n</parameter>\n</function>\n</tool_call>";
  ToolCallParserOptions o = Opts(StreamStart::kContent);
  std::string error;
  o.tools = *halo::json::Parse(
      R"([{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{)"
      R"("s":{"type":"string"},"n":{"type":["integer","null"]},"o":{"type":"object"}}}}}])",
      &error);
  const auto events = ParseAssistantText(text, o);
  ASSERT_EQ(Calls(events).size(), 1u) << Describe(events);
  const auto calls = Calls(events);
  const Value& a = calls[0].arguments;
  for (const char* key : {"s", "n", "o", "u", "q"}) ASSERT_TRUE(a.find(key)) << key << "\n" << Describe(events);
  EXPECT_EQ(*a.find("s"), Value::String("42"));
  EXPECT_EQ(*a.find("n"), Value::Int(42));
  EXPECT_TRUE(a.find("o")->is(Value::Kind::kObject));
  EXPECT_EQ(*a.find("u"), Value::Bool(true));                // no schema: JSON
  EXPECT_EQ(*a.find("q"), Value::String("\"quoted\""));      // a JSON string is a raw string
}

TEST(ToolCallParser, JsonFormAndMultipleCalls) {
  const std::string text =
      "<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": \"}\"}}\n</tool_call>\n"
      "<tool_call>\n{\"name\": \"b\", \"arguments\": \"{\\\"y\\\": 1}\"}\n</tool_call>";
  const auto events = ParseAssistantText(text, Opts(StreamStart::kContent));
  ASSERT_EQ(Calls(events).size(), 2u) << Describe(events);
  EXPECT_EQ(Calls(events)[0].arguments, Obj({{"x", Value::String("}")}}));
  EXPECT_EQ(Calls(events)[1].arguments, Obj({{"y", Value::Int(1)}}));
  EXPECT_EQ(Joined(events, EK::kContent), "");
}

TEST(ToolCallParser, ParameterFraming) {
  // Exactly one newline each side is framing; the rest is the value.
  const std::string text =
      "<tool_call>\n<function=f>\n<parameter=a>\n\n  two\n\n</parameter>\n<parameter=b>\n\n</parameter>\n"
      "<parameter=c>\n</parameter>\n<parameter=d>inline</parameter>\n</function>\n</tool_call>";
  const auto events = ParseAssistantText(text, Opts(StreamStart::kContent));
  ASSERT_EQ(Calls(events).size(), 1u) << Describe(events);
  const auto calls = Calls(events);
  const Value& a = calls[0].arguments;
  for (const char* key : {"a", "b", "c", "d"}) ASSERT_TRUE(a.find(key)) << key << "\n" << Describe(events);
  EXPECT_EQ(*a.find("a"), Value::String("\n  two\n"));
  EXPECT_EQ(*a.find("b"), Value::String(""));
  EXPECT_EQ(*a.find("c"), Value::String(""));
  EXPECT_EQ(*a.find("d"), Value::String("inline"));
}

TEST(ToolCallParser, DetectStartAndThinkingOff) {
  auto events = ParseAssistantText("\n<think>\nhm\n</think>\n\nanswer", Opts(StreamStart::kDetect));
  EXPECT_EQ(Joined(events, EK::kReasoning), "hm");
  EXPECT_EQ(Joined(events, EK::kContent), "answer");
  events = ParseAssistantText("plain </think> text", Opts(StreamStart::kDetect));
  EXPECT_EQ(Joined(events, EK::kContent), "plain </think> text");
  EXPECT_EQ(Joined(events, EK::kReasoning), "");
  events = ParseAssistantText("<thin", Opts(StreamStart::kDetect));
  EXPECT_EQ(Joined(events, EK::kContent), "<thin");
}

TEST(ToolCallParser, CallInsideReasoning) {
  const std::string text = "thinking <tool_call>\n<function=f>\n</function>\n</tool_call>";
  auto o = Opts(StreamStart::kReasoning);
  auto events = ParseAssistantText(text, o);
  ASSERT_EQ(Calls(events).size(), 1u) << Describe(events);
  EXPECT_EQ(Joined(events, EK::kReasoning), "thinking");
  o.tool_calls_in_reasoning = false;
  events = ParseAssistantText(text, o);
  EXPECT_EQ(Calls(events).size(), 0u);
  EXPECT_EQ(Joined(events, EK::kReasoning), text);
}

TEST(ToolCallParser, MalformedAndTruncatedCallsAreErrorsWithTheirBytes) {
  const auto o = Opts(StreamStart::kContent);
  for (const std::string& text : {
           std::string("<tool_call>\n<function=f>\n<parameter=a>\nunterminated"),
           std::string("<tool_call>\nnot a call\n</tool_call>after"),
           std::string("<tool_call>\n<function=f\nx>\n</function>\n</tool_call>"),
           std::string("<tool_call>\n{\"name\": 5}\n</tool_call>"),
           std::string("<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n<parameter=a>\n2\n</parameter>\n</function>\n</tool_call>"),
           std::string("<tool_call>"),
       }) {
    const auto events = ParseAssistantText(text, o);
    CheckEvents(text, events, o);
    EXPECT_TRUE(std::ranges::any_of(events, [](const auto& e) { return e.kind == EK::kError; })) << Describe(events);
    EXPECT_TRUE(Calls(events).empty()) << Describe(events);
  }
  // A broken call does not swallow the valid call after it.
  const std::string text = "<tool_call>\n<function=f>\n<oops>\n<tool_call>\n<function=g>\n</function>\n</tool_call>";
  const auto events = ParseAssistantText(text, o);
  ASSERT_EQ(Calls(events).size(), 1u) << Describe(events);
  EXPECT_EQ(Calls(events)[0].name, "g");
}

TEST(ToolCallParser, NoPartialTagIsEverEmittedAsContent) {
  const std::string text =
      "Hi <tool_call>\n<function=f>\n<parameter=a>\nv\n</parameter>\n</function>\n</tool_call>\n tail <tool";
  const auto o = Opts(StreamStart::kContent);
  const auto whole = Normalize(ParseAssistantText(text, o));
  for (std::size_t i = 0; i <= text.size(); ++i) {
    for (std::size_t j = i; j <= text.size(); ++j) {
      ToolCallStreamParser p(o);
      std::vector<StreamEvent> events;
      p.Push(text.substr(0, i), &events);
      // Mid-stream content never ends inside "<tool_call>".
      const std::string content = Joined(events, EK::kContent);
      ASSERT_FALSE(content.ends_with("<") || content.ends_with("<tool")) << i;
      p.Push(text.substr(i, j - i), &events);
      p.Push(text.substr(j), &events);
      p.Finish(&events);
      ASSERT_EQ(Normalize(events), whole) << i << "," << j << "\n" << Describe(events);
    }
  }
  EXPECT_EQ(Joined(whole, EK::kContent), "Hitail <tool");
}

TEST(ToolCallParser, Utf8TailsAreHeldBack) {
  const std::string text = "caf\xc3\xa9 \xf0\x9f\xa6\x99!";
  for (std::size_t i = 0; i <= text.size(); ++i) {
    ToolCallStreamParser p(Opts(StreamStart::kContent));
    std::vector<StreamEvent> events;
    p.Push(text.substr(0, i), &events);
    const std::string content = Joined(events, EK::kContent);
    // A prefix of the text that ends on a code point boundary.
    ASSERT_EQ(content, text.substr(0, content.size())) << i;
    ASSERT_TRUE(content.size() == text.size() || (static_cast<unsigned char>(text[content.size()]) & 0xC0) != 0x80)
        << i << " " << content;
    ASSERT_GE(content.size() + 4, i) << i;
  }
}

TEST(ToolCallParser, SizeLimitCutsDeterministically) {
  ToolCallParserOptions o = Opts(StreamStart::kContent);
  o.max_tool_call_bytes = 40;
  const std::string text = "<tool_call>\n<function=f>\n<parameter=a>\n" + std::string(100, 'x') +
                           "\n</parameter>\n</function>\n</tool_call>ok";
  const auto whole = ParseAssistantText(text, o);
  CheckEvents(text, whole, o);
  ASSERT_EQ(whole[0].kind, EK::kError);
  EXPECT_EQ(whole[0].text.size(), 40u);
  for (std::size_t i = 0; i <= text.size(); i += 7)
    EXPECT_EQ(Normalize(ParseChunks(SplitAt(text, {i}), o)), Normalize(whole)) << i;
}

// ---- generators -------------------------------------------------------------

namespace {

const std::vector<std::string> kWords = {"Hello", "world", "a", "é", "日本", "🦙", "x=1", "{", "}", "\"",
                                         "<", ">", "</", "(ok)", "1.5", "true", "null", "\\", "think",
                                         "　", " ", "tab\there"};
const std::vector<std::string> kSpace = {" ", "  ", "\n", "\n\n", "\t", "\r\n"};

std::string DrawProse(hegel::TestCase& tc, std::size_t max_words = 6) {
  const auto n = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = max_words}));
  std::string s;
  for (std::size_t i = 0; i < n; ++i) {
    if (i && tc.draw(gs::booleans())) s += tc.draw(gs::sampled_from(kSpace));
    s += tc.draw(gs::sampled_from(kWords));
  }
  if (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3})) == 0) s = tc.draw(gs::sampled_from(kSpace)) + s;
  if (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3})) == 0) s += tc.draw(gs::sampled_from(kSpace));
  return s;
}

std::string DrawIdent(hegel::TestCase& tc) {
  return tc.draw(gs::sampled_from<std::string>({"f", "get_weather", "search", "a.b", "名前", "x-y", "q", "data"}));
}

Value DrawArgValue(hegel::TestCase& tc, int depth) {
  switch (tc.draw(gs::integers<int>({.min_value = 0, .max_value = depth > 1 ? 4 : 6}))) {
    case 0: return Value::String(DrawProse(tc));
    case 1: return Value::Int(tc.draw(gs::integers<std::int64_t>()));
    case 2: {
      const double d = tc.draw(gs::floats<double>({.allow_nan = false, .allow_infinity = false}));
      return Value::Float(d);
    }
    case 3: return Value::Bool(tc.draw(gs::booleans()));
    case 4: return Value::Null();
    case 5: {
      Value a = Value::MakeArray();
      const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}));
      for (int i = 0; i < n; ++i) a.push(DrawArgValue(tc, depth + 1));
      return a;
    }
    default: {
      Value o = Value::MakeObject();
      const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}));
      for (int i = 0; i < n; ++i) o.set(DrawIdent(tc), DrawArgValue(tc, depth + 1));
      return o;
    }
  }
}

const char* SchemaType(const Value& v) {
  switch (v.kind()) {
    case Value::Kind::kNull: return "null";
    case Value::Kind::kBool: return "boolean";
    case Value::Kind::kInt: return "integer";
    case Value::Kind::kFloat: return "number";
    case Value::Kind::kString: return "string";
    case Value::Kind::kArray: return "array";
    case Value::Kind::kObject: return "object";
  }
  return "string";
}

// One serialized call in the template's own layout, or a JSON-form call.
std::string DrawCallText(hegel::TestCase& tc) {
  if (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4})) == 0) {
    Value body = Value::MakeObject();
    body.set("name", Value::String(DrawIdent(tc)));
    body.set("arguments", DrawArgValue(tc, 1));
    return "<tool_call>\n" + halo::json::DumpPython(body) + "\n</tool_call>";
  }
  std::string s = "<tool_call>\n<function=" + DrawIdent(tc) + ">\n";
  const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}));
  for (int i = 0; i < n; ++i) {
    const Value v = DrawArgValue(tc, 1);
    s += "<parameter=" + DrawIdent(tc) + ">\n" + (v.is(Value::Kind::kString) ? v.str() : halo::json::DumpPython(v)) +
         "\n</parameter>\n";
  }
  return s + "</function>\n</tool_call>";
}

const std::vector<std::string> kMarkup = {
    "<think>", "</think>", "<tool_call>", "</tool_call>", "<function=f>", "<function=", "</function>",
    "<parameter=a>", "<parameter=", "</parameter>", "\n</parameter>", "<tool_", "</thi", "<", "{\"name\": \"f\"}",
    "<|im_end|>"};

// Text that looks like model output: prose, markup, whole calls, raw bytes.
std::string DrawPiece(hegel::TestCase& tc) {
  switch (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 5}))) {
    case 0:
    case 1: return DrawProse(tc);
    case 2: return tc.draw(gs::sampled_from(kMarkup));
    case 3: return DrawCallText(tc);
    case 4: return tc.draw(gs::sampled_from(kSpace));
    default: {
      const auto b = tc.draw(gs::binary({.max_size = 6}));
      return std::string(b.begin(), b.end());
    }
  }
}

std::string DrawStream(hegel::TestCase& tc) {
  std::string s;
  const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 10}));
  for (int i = 0; i < n; ++i) s += DrawPiece(tc);
  return s;
}

ToolCallParserOptions DrawOptions(hegel::TestCase& tc) {
  ToolCallParserOptions o;
  o.start = tc.draw(gs::sampled_from<StreamStart>({StreamStart::kReasoning, StreamStart::kContent, StreamStart::kDetect}));
  o.tool_calls_in_reasoning = tc.draw(gs::booleans());
  if (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3})) == 0)
    o.max_tool_call_bytes = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 120}));
  return o;
}

std::vector<std::size_t> DrawCuts(hegel::TestCase& tc, std::size_t size) {
  return tc.draw(gs::vectors(gs::integers<std::size_t>({.min_value = 0, .max_value = size}), {.max_size = 12}));
}

}  // namespace

// ---- properties -------------------------------------------------------------

// More than Hegel's default 100: the generators are cheap and the space is wide.
constexpr std::uint64_t kCases = 400;

TEST(ToolCallParser, RenderThenParseRoundTrips) {
  hegel::test([](hegel::TestCase& tc) {
    const auto kind = tc.draw(gs::sampled_from<ChatTemplateKind>({ChatTemplateKind::kFlashNext, ChatTemplateKind::kQwen35}));
    const std::string reasoning = DrawProse(tc), content = DrawProse(tc);
    // Distinct names per call; the schema types each parameter by its value.
    Value tools = Value::MakeArray(), calls = Value::MakeArray();
    std::vector<ParsedToolCall> expected;
    const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}));
    for (int i = 0; i < n; ++i) {
      ParsedToolCall call;
      call.name = DrawIdent(tc) + std::to_string(i);
      Value props = Value::MakeObject();
      const auto nargs = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4}));
      for (int j = 0; j < nargs; ++j) {
        const std::string key = DrawIdent(tc) + "_" + std::to_string(j);
        const Value v = DrawArgValue(tc, 0);
        call.arguments.set(key, v);
        props.set(key, Obj({{"type", Value::String(SchemaType(v))}}));
      }
      tools.push(Obj({{"type", Value::String("function")},
                      {"function", Obj({{"name", Value::String(call.name)},
                                        {"parameters", Obj({{"type", Value::String("object")}, {"properties", props}})}})}}));
      calls.push(Obj({{"type", Value::String("function")},
                      {"function", Obj({{"name", Value::String(call.name)}, {"arguments", call.arguments}})}}));
      expected.push_back(std::move(call));
    }
    Value assistant = Obj({{"role", Value::String("assistant")}, {"content", Value::String(content)},
                           {"reasoning_content", Value::String(reasoning)}});
    if (n) assistant.set("tool_calls", calls);
    Value messages = Value::MakeArray();
    messages.push(Obj({{"role", Value::String("user")}, {"content", Value::String("q")}}));
    messages.push(assistant);
    std::string error;
    const auto rendered = RenderChat(kind, messages, n ? &tools : nullptr, {}, &error);
    Check(rendered.has_value(), error);
    // The assistant turn: after its header, up to its <|im_end|>.
    const std::string_view header = "<|im_start|>assistant\n";
    const std::size_t begin = rendered->rfind(header) + header.size();
    const std::size_t end = rendered->rfind("<|im_end|>");
    std::string_view turn = std::string_view(*rendered).substr(begin, end - begin);
    ToolCallParserOptions o;
    o.tools = tools;
    if (tc.draw(gs::booleans())) {
      o.start = StreamStart::kDetect;
    } else {
      o.start = StreamStart::kReasoning;  // the generation prompt consumed "<think>\n"
      Check(turn.starts_with("<think>\n"), "think block");
      turn.remove_prefix(8);
    }
    const auto events = ParseChunks(SplitAt(turn, DrawCuts(tc, turn.size())), o);
    CheckEvents(turn, events, o);
    Check(Joined(events, EK::kReasoning) == PyStrip(reasoning), "reasoning:\n" + Describe(events));
    Check(Joined(events, EK::kContent) == PyStrip(content), "content:\n" + Describe(events));
    Check(Calls(events) == expected, "calls:\n" + Describe(events));
    Check(Joined(events, EK::kError).empty(), "errors:\n" + Describe(events));
  }, {.test_cases = kCases});
}

TEST(ToolCallParser, ChunkingInvariance) {
  hegel::test([](hegel::TestCase& tc) {
    const auto o = DrawOptions(tc);
    const std::string text = DrawStream(tc);
    const auto whole = ParseAssistantText(text, o);
    const auto chunked = ParseChunks(SplitAt(text, DrawCuts(tc, text.size())), o);
    Check(Normalize(chunked) == Normalize(whole), "chunked:\n" + Describe(chunked) + "whole:\n" + Describe(whole));
    // Byte by byte, the hardest split.
    std::vector<std::string> bytes;
    for (char c : text) bytes.emplace_back(1, c);
    Check(Normalize(ParseChunks(bytes, o)) == Normalize(whole), "bytewise");
  }, {.test_cases = kCases});
}

TEST(ToolCallParser, MalformedStreamsLoseNoBytes) {
  hegel::test([](hegel::TestCase& tc) {
    const auto o = DrawOptions(tc);
    std::string text = DrawStream(tc);
    // Damage it: truncate, delete, or overwrite bytes.
    const auto edits = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}));
    for (int i = 0; i < edits && !text.empty(); ++i) {
      const auto at = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = text.size() - 1}));
      switch (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 2}))) {
        case 0: text.resize(at); break;
        case 1: text.erase(at, 1); break;
        default: text[at] = static_cast<char>(tc.draw(gs::integers<std::uint8_t>()));
      }
    }
    const auto events = ParseChunks(SplitAt(text, DrawCuts(tc, text.size())), o);
    CheckEvents(text, events, o);
  }, {.test_cases = kCases});
}

// ---- stateful ---------------------------------------------------------------

namespace {

// State: the chunks pushed in the current stream.
struct ParserMachine : hs::StateMachine<ParserMachine, std::vector<std::string>> {
  explicit ParserMachine(ToolCallParserOptions o) : StateMachine({.initial_state = {}}), options(o), parser(o) {}

  ToolCallParserOptions options;
  ToolCallStreamParser parser;
  std::vector<StreamEvent> events;  // this stream's events so far

  std::string Pushed() const {
    std::string s;
    for (const auto& c : state) s += c;
    return s;
  }

  void Push(std::string chunk) {
    parser.Push(chunk, &events);
    state.push_back(std::move(chunk));
  }

  std::vector<hs::Rule<ParserMachine>> rules() {
    return {
        hs::Rule<ParserMachine>("push_piece", [](hegel::TestCase& tc, ParserMachine& m) { m.Push(DrawPiece(tc)); }),
        hs::Rule<ParserMachine>("push_call_split",
                                [](hegel::TestCase& tc, ParserMachine& m) {
                                  const std::string call = DrawCallText(tc);
                                  for (auto& part : SplitAt(call, DrawCuts(tc, call.size()))) m.Push(std::move(part));
                                }),
        hs::Rule<ParserMachine>("push_tag_split",
                                [](hegel::TestCase& tc, ParserMachine& m) {
                                  const std::string tag = tc.draw(gs::sampled_from(kMarkup));
                                  const auto at = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = tag.size()}));
                                  m.Push(tag.substr(0, at));
                                  m.Push(tag.substr(at));
                                }),
        hs::Rule<ParserMachine>("push_bytes",
                                [](hegel::TestCase& tc, ParserMachine& m) {
                                  const auto b = tc.draw(gs::binary({.max_size = 8}));
                                  m.Push(std::string(b.begin(), b.end()));
                                }),
        hs::Rule<ParserMachine>("finish",
                                [](hegel::TestCase&, ParserMachine& m) {
                                  m.parser.Finish(&m.events);
                                  const std::string text = m.Pushed();
                                  CheckEvents(text, m.events, m.options);
                                  const auto whole = ParseAssistantText(text, m.options);
                                  Check(Normalize(m.events) == Normalize(whole),
                                        "stream:\n" + Describe(m.events) + "one shot:\n" + Describe(whole));
                                  Check(m.parser.pending_bytes() == 0, "pending after finish");
                                  m.events.clear();
                                  m.state.clear();
                                }),
    };
  }

  std::vector<hs::Invariant<ParserMachine>> invariants() {
    return {
        hs::Invariant<ParserMachine>("bytes_accounted",
                                     [](const ParserMachine& m) {
                                       const std::string pushed = m.Pushed();
                                       const std::string emitted = AllBytes(m.events);
                                       Check(emitted.size() + m.parser.pending_bytes() == pushed.size(),
                                             "emitted + pending != pushed");
                                       Check(pushed.starts_with(emitted), "emitted bytes are not a prefix");
                                     }),
    };
  }
};

}  // namespace

TEST(ToolCallParser, StatefulStreams) {
  hegel::test([](hegel::TestCase& tc) {
    ParserMachine machine(DrawOptions(tc));
    hs::run(machine, tc, {.print_state = true});
  }, {.test_cases = kCases});
}
