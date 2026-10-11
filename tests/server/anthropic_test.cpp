// The Anthropic Messages API (docs/ANTHROPIC.md):
//   - request parsing: content blocks onto template messages (system blocks, Claude Code's billing line, tool
//     results, thinking signatures, mid-conversation system messages), the refusals, tool_choice, thinking;
//   - response building: golden event sequences, and a property that every stream follows the grammar
//     (message_start, contiguous blocks each started / deltas / stopped, message_delta, exactly one
//     message_stop) and that its deltas add up to the final Message;
//   - the service over loopback HTTP with a scripted generator: whole and streamed messages, the tool-use round
//     trip, stop sequences, max_tokens, cache usage, count_tokens, model aliases, x-api-key auth, errors, pings.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <hegel/gtest.h>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "core/json.hpp"
#include "server/anthropic_request.hpp"
#include "server/anthropic_response.hpp"
#include "server/anthropic_service.hpp"
#include "server/http.hpp"
#include "server/openai_response.hpp"
#include "text/chat_template.hpp"

namespace gs = hegel::generators;
using namespace halo;
using namespace halo::server;
using json::Value;
using Kind = json::Value::Kind;

namespace {

Value J(std::string_view text) {
  std::string err;
  auto v = json::Parse(text, &err);
  if (!v) throw std::runtime_error("bad JSON: " + err + " in " + std::string(text.substr(0, 200)));
  return *v;
}

MessagesRequest Parse(const std::string& body, bool count_tokens = false) {
  MessagesError e;
  auto r = ParseMessagesRequest(J(body), RequestDefaults{}, count_tokens, &e);
  if (!r) throw std::runtime_error("refused: " + e.message);
  return std::move(*r);
}

MessagesError Refusal(const std::string& body) {
  MessagesError e;
  EXPECT_FALSE(ParseMessagesRequest(J(body), RequestDefaults{}, false, &e).has_value()) << body;
  return e;
}

std::string Render(const Value& messages, const Value* tools = nullptr, text::ChatTemplateOptions o = {}) {
  o.add_generation_prompt = true;
  std::string err;
  auto s = text::RenderChat(text::ChatTemplateKind::kFlashNext, messages, tools, o, &err);
  if (!s) throw std::runtime_error(err);
  return *s;
}

}  // namespace

// ---- request parsing -------------------------------------------------------------------------------------------

TEST(AnthropicRequest, ContentBlocksMapOntoTemplateMessages) {
  const auto r = Parse(R"({
    "model": "claude-sonnet-5-5", "max_tokens": 100,
    "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.293.2a8;"},
               {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}},
               {"type": "text", "text": "Be kind."}],
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "Weather?"}]},
      {"role": "system", "content": [{"type": "text", "text": "<note>", "cache_control": {"type": "ephemeral"}}]},
      {"role": "assistant", "content": [
         {"type": "thinking", "thinking": "", "signature": ")" + EncodeThinkingSignature("I should call it.") + R"("},
         {"type": "text", "text": "Checking."},
         {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Paris"}}]},
      {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"},
                                   {"type": "tool_result", "tool_use_id": "toolu_2", "is_error": true,
                                    "content": [{"type": "text", "text": "timeout"}]}]},
      {"role": "system", "content": "<budget>"}
    ],
    "tools": [{"name": "weather", "description": "Weather", "input_schema": {"type": "object"}}]})");
  const Value want = J(R"([
    {"role": "system", "content": "You are terse.\n\nBe kind."},
    {"role": "user", "content": [{"type": "text", "text": "Weather?"}, {"type": "text", "text": "\n\n<note>"}]},
    {"role": "assistant", "reasoning_content": "I should call it.", "content": "Checking.",
     "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}]},
    {"role": "tool", "content": "18C", "tool_call_id": "toolu_1"},
    {"role": "tool", "content": "[error] timeout", "tool_call_id": "toolu_2"},
    {"role": "user", "content": "<budget>"}])");
  EXPECT_EQ(CompactJson(r.gen.messages), CompactJson(want));
  EXPECT_EQ(CompactJson(r.gen.tools),
            R"([{"type":"function","function":{"name":"weather","description":"Weather","parameters":{"type":"object"}}}])");
  EXPECT_EQ(r.cache_breakpoints, 2u);
  EXPECT_EQ(r.gen.max_tokens, 100u);
  EXPECT_EQ(r.gen.api, Api::kMessages);
  // The template renders it (tool messages after an assistant call, the system note as a user turn).
  EXPECT_NO_THROW((void)Render(r.gen.messages, &r.gen.tools));
}

TEST(AnthropicRequest, SettingsAndIgnoredFields) {
  const auto r = Parse(R"({"model": "m", "max_tokens": 5, "stream": true, "temperature": 0.2, "top_p": 0.5,
    "top_k": 7, "stop_sequences": ["END", "##"], "metadata": {"user_id": "u1"},
    "thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "max"},
    "context_management": {"edits": []}, "safeguards": [], "service_tier": "auto",
    "messages": [{"role": "user", "content": "hi"}]})");
  EXPECT_TRUE(r.gen.stream);
  EXPECT_FLOAT_EQ(r.gen.sampling.temperature, 0.2F);
  EXPECT_FLOAT_EQ(r.gen.sampling.top_p, 0.5F);
  EXPECT_EQ(r.gen.sampling.top_k, 7);
  EXPECT_EQ(r.gen.stop, (std::vector<std::string>{"END", "##"}));
  EXPECT_EQ(r.gen.user, "u1");
  EXPECT_TRUE(r.omit_thinking);
  EXPECT_EQ(r.gen.template_options.enable_thinking, std::optional<bool>(true));
  EXPECT_EQ(r.gen.template_options.reasoning_effort, std::optional<std::string>("xhigh"));
  EXPECT_EQ(r.ignored, (std::vector<std::string>{"context_management", "safeguards", "service_tier"}));
  // Unset sampling keeps the server's defaults (the model's recommended settings).
  const auto d = Parse(R"({"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]})");
  EXPECT_FLOAT_EQ(d.gen.sampling.temperature, RequestDefaults{}.sampling.temperature);
  EXPECT_FALSE(d.gen.template_options.enable_thinking.has_value());
  const auto off = Parse(R"({"model": "m", "max_tokens": 5, "thinking": {"type": "disabled"},
                             "messages": [{"role": "user", "content": "hi"}]})");
  EXPECT_EQ(off.gen.template_options.enable_thinking, std::optional<bool>(false));
  // count_tokens: no max_tokens needed.
  EXPECT_NO_THROW((void)Parse(R"({"model": "m", "messages": [{"role": "user", "content": "hi"}]})", true));
}

TEST(AnthropicRequest, RefusesWhatItCannotHonour) {
  const std::string user = R"("messages": [{"role": "user", "content": "hi"}])";
  const auto refused = [&](const std::string& extra, const std::string& msgs, const char* needle) {
    const auto e = Refusal("{\"model\": \"m\"" + extra + ", " + msgs + "}");
    EXPECT_EQ(e.status, 400);
    EXPECT_EQ(e.type, "invalid_request_error");
    EXPECT_NE(e.message.find(needle), std::string::npos) << e.message;
  };
  refused("", user, "max_tokens");
  refused(R"(, "max_tokens": 0)", user, "max_tokens");
  refused(R"(, "max_tokens": 5)",
          R"("messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64"}}]}])",
          "messages.0.content.0.type: image input is not supported");
  refused(R"(, "max_tokens": 5)",
          R"("messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
              "content": [{"type": "image"}]}]}])",
          "image input is not supported");
  refused(R"(, "max_tokens": 5)",
          R"("messages": [{"role": "user", "content": [{"type": "document"}]}])", "document input");
  refused(R"(, "max_tokens": 5, "tools": [{"type": "web_search_20260209", "name": "web_search"}])", user,
          "tools.0.type: Anthropic-defined tool type");
  refused(R"(, "max_tokens": 5, "output_config": {"format": {"type": "json_schema"}})", user, "output_config.format");
  refused(R"(, "max_tokens": 5, "mcp_servers": [{}])", user, "mcp_servers");
  refused(R"(, "max_tokens": 5)",
          R"("messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Sure,"}])",
          "prefill is not supported");
  refused(R"(, "max_tokens": 5, "temperature": 1.5)", user, "temperature");
  refused(R"(, "max_tokens": 5, "thinking": {"type": "sometimes"})", user, "thinking.type");
  refused(R"(, "max_tokens": 5, "tools": [{"name": "a", "input_schema": {}}], "tool_choice": {"type": "tool", "name": "b"})",
          user, "no tool named b");
  refused(R"(, "max_tokens": 5, "thinking": {"type": "adaptive"}, "tools": [{"name": "a", "input_schema": {}}],
             "tool_choice": {"type": "any"})", user, "Thinking may not be enabled");
  refused(R"(, "max_tokens": 5)", R"("messages": [])", "messages");
}

TEST(AnthropicRequest, ForcedToolChoiceOpensTheCallWithThinkingOff) {
  const auto r = Parse(R"({"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}],
    "tools": [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {}}],
    "tool_choice": {"type": "tool", "name": "b", "disable_parallel_tool_use": true}})");
  EXPECT_EQ(r.forced_prefix, "<tool_call>\n<function=b>\n");
  EXPECT_EQ(r.gen.template_options.enable_thinking, std::optional<bool>(false));
  EXPECT_TRUE(r.disable_parallel_tool_use);
  const auto any = Parse(R"({"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}],
    "tools": [{"name": "a", "input_schema": {}}], "tool_choice": {"type": "any"}})");
  EXPECT_EQ(any.forced_prefix, "<tool_call>\n<function=");
}

TEST(AnthropicRequest, ThinkingSignaturesRoundTripAnyBytes) {
  hegel::test([](hegel::TestCase& tc) {
    const auto s = tc.draw(gs::binary({.max_size = 64}));
    const std::string text(s.begin(), s.end());
    const auto back = DecodeThinkingSignature(EncodeThinkingSignature(text));
    ASSERT_TRUE(back.has_value());
    ASSERT_EQ(*back, text);
  });
  EXPECT_FALSE(DecodeThinkingSignature("EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds").has_value());
  EXPECT_FALSE(DecodeThinkingSignature("halo1.abc").has_value());
  EXPECT_FALSE(DecodeThinkingSignature("halo1.a=bc").has_value());
}

TEST(AnthropicRequest, ModelAliases) {
  const std::vector<ModelAlias> aliases = {{"claude-haiku-*", "q:small"}, {"claude-*", ""}, {"local", ""}};
  EXPECT_EQ(ResolveModel("q", "q", aliases), std::optional<std::string>("q"));
  EXPECT_EQ(ResolveModel("claude-sonnet-5-5", "q", aliases), std::optional<std::string>("q"));
  EXPECT_EQ(ResolveModel("claude-haiku-4-5", "q", aliases), std::optional<std::string>("q:small"));
  EXPECT_EQ(ResolveModel("local", "q", aliases), std::optional<std::string>("q"));
  EXPECT_FALSE(ResolveModel("gpt-5", "q", aliases).has_value());
  EXPECT_FALSE(ResolveModel("claude-x", "q", {}).has_value());
  EXPECT_TRUE(ParseModelAliasSpec("claude-*=").has_value());
  EXPECT_EQ(ParseModelAliasSpec("a=b")->target, "b");
  EXPECT_FALSE(ParseModelAliasSpec("=b").has_value());
  EXPECT_FALSE(ParseModelAliasSpec("cl*ude=b").has_value());
  EXPECT_FALSE(ParseModelAliasSpec("nothing").has_value());
}

// ---- response building -----------------------------------------------------------------------------------------

namespace {

text::StreamEvent Ev(text::StreamEventKind k, std::string t) {
  text::StreamEvent e;
  e.kind = k;
  e.text = std::move(t);
  return e;
}
text::StreamEvent Call(std::string name, Value args) {
  text::StreamEvent e;
  e.kind = text::StreamEventKind::kToolCall;
  e.text = "<tool_call>...</tool_call>";
  e.call.name = std::move(name);
  e.call.arguments = std::move(args);
  return e;
}

MessageBuilder Builder(bool omit = false, bool tools = true) {
  auto n = std::make_shared<int>(0);
  return MessageBuilder({"msg_1", "qwen", omit, tools, [n] { return std::to_string(++*n); }});
}

std::string Sse(const std::vector<Value>& events) {
  std::string s;
  for (const auto& e : events) s += MessagesFrame(e);
  return s;
}

}  // namespace

TEST(AnthropicResponse, GoldenTextStream) {
  auto b = Builder();
  std::vector<Value> out;
  b.Start(25, &out);
  b.Ping(&out);
  b.Add({Ev(text::StreamEventKind::kContent, "Hel"), Ev(text::StreamEventKind::kFraming, "\n"),
         Ev(text::StreamEventKind::kContent, "lo!")},
        &out);
  b.Finish(StopReason::kEndTurn, "", {25, 0, 0, 3}, &out);
  EXPECT_EQ(Sse(out),
            "event: message_start\n"
            R"(data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],)"
            R"("model":"qwen","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":0}}})"
            "\n\nevent: ping\ndata: {\"type\":\"ping\"}\n\n"
            "event: content_block_start\n"
            R"(data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})"
            "\n\nevent: content_block_delta\n"
            R"(data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}})"
            "\n\nevent: content_block_delta\n"
            R"(data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo!"}})"
            "\n\nevent: content_block_stop\n"
            R"(data: {"type":"content_block_stop","index":0})"
            "\n\nevent: message_delta\n"
            R"(data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},)"
            R"("usage":{"input_tokens":25,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":3}})"
            "\n\nevent: message_stop\n"
            R"(data: {"type":"message_stop"})"
            "\n\n");
  EXPECT_EQ(CompactJson(b.Final()),
            R"({"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],)"
            R"("model":"qwen","stop_reason":"end_turn","stop_sequence":null,)"
            R"("usage":{"input_tokens":25,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":3}})");
}

TEST(AnthropicResponse, GoldenThinkingTextAndToolUse) {
  for (const bool omit : {false, true}) {
    auto b = Builder(omit);
    std::vector<Value> out;
    b.Start(10, &out);
    b.Add({Ev(text::StreamEventKind::kReasoning, "Plan"), Ev(text::StreamEventKind::kFraming, "</think>"),
           Ev(text::StreamEventKind::kContent, "Calling."), Call("weather", J(R"({"city":"Paris","n":2})"))},
          &out);
    b.Finish(StopReason::kToolUse, "", {100, 60, 30, 9}, &out);
    std::vector<std::string> kinds;
    for (const auto& e : out) {
      std::string k = e.find("type")->str();
      if (const Value* d = e.find("delta"); d && d->find("type")) k += ":" + d->find("type")->str();
      if (const Value* cb = e.find("content_block")) k += ":" + cb->find("type")->str();
      kinds.push_back(k);
    }
    const std::vector<std::string> want = {
        "message_start", "content_block_start:thinking", "content_block_delta:thinking_delta",
        "content_block_delta:signature_delta", "content_block_stop", "content_block_start:text",
        "content_block_delta:text_delta", "content_block_stop", "content_block_start:tool_use",
        "content_block_delta:input_json_delta", "content_block_stop", "message_delta", "message_stop"};
    EXPECT_EQ(kinds, want);
    EXPECT_EQ(out[2].find("delta")->find("thinking")->str(), omit ? "" : "Plan");
    EXPECT_EQ(out[3].find("delta")->find("signature")->str(), EncodeThinkingSignature("Plan"));
    EXPECT_EQ(out[9].find("delta")->find("partial_json")->str(), R"({"city":"Paris","n":2})");
    EXPECT_EQ(CompactJson(*out[11].find("usage")),
              R"({"input_tokens":10,"cache_creation_input_tokens":30,"cache_read_input_tokens":60,"output_tokens":9})");
    const Value f = b.Final();
    EXPECT_EQ(CompactJson(*f.find("content")),
              std::string(R"([{"type":"thinking","thinking":")") + (omit ? "" : "Plan") + R"(","signature":")" +
                  EncodeThinkingSignature("Plan") +
                  R"("},{"type":"text","text":"Calling."},{"type":"tool_use","id":"toolu_1","name":"weather",)"
                  R"("input":{"city":"Paris","n":2}}])");
    EXPECT_EQ(f.find("stop_reason")->str(), "tool_use");
  }
  // tool_choice none: a call the model writes anyway is text.
  auto b = Builder(false, /*tools=*/false);
  b.Add({Call("weather", Value::MakeObject())}, nullptr);
  b.Finish(StopReason::kEndTurn, "", {}, nullptr);
  EXPECT_EQ(b.tool_uses(), 0u);
  EXPECT_EQ(b.Final().find("content")->items()[0].find("type")->str(), "text");
}

TEST(AnthropicResponse, StopReasons) {
  EXPECT_STREQ(StopReasonName(ChooseStopReason(false, false, false, false)), "end_turn");
  EXPECT_STREQ(StopReasonName(ChooseStopReason(false, false, false, true)), "tool_use");
  EXPECT_STREQ(StopReasonName(ChooseStopReason(false, false, true, true)), "stop_sequence");
  EXPECT_STREQ(StopReasonName(ChooseStopReason(true, false, true, true)), "max_tokens");
  EXPECT_STREQ(StopReasonName(ChooseStopReason(true, true, false, false)), "model_context_window_exceeded");
  auto b = Builder();
  std::vector<Value> out;
  b.Finish(StopReason::kStopSequence, "END", {}, &out);
  EXPECT_EQ(out.front().find("type")->str(), "message_start");  // Finish starts a stream that never started
  EXPECT_EQ(out[1].find("delta")->find("stop_sequence")->str(), "END");
  EXPECT_EQ(b.Final().find("stop_sequence")->str(), "END");
  auto c = Builder();
  c.Finish(StopReason::kEndTurn, "END", {}, nullptr);  // only a stop_sequence stop names one
  EXPECT_TRUE(c.Final().find("stop_sequence")->is(Kind::kNull));
}

namespace {

/// Checks the stream grammar and rebuilds the Message from the events (as the SDKs accumulate them).
Value Accumulate(const std::vector<Value>& events) {
  EXPECT_FALSE(events.empty());
  EXPECT_EQ(events.front().find("type")->str(), "message_start");
  Value msg = *events.front().find("message");
  Value content = Value::MakeArray();
  std::vector<std::string> partial;
  std::optional<std::size_t> open;
  bool delta_seen = false, stopped = false;
  for (std::size_t i = 1; i < events.size(); ++i) {
    const Value& e = events[i];
    const std::string type = e.find("type")->str();
    EXPECT_FALSE(stopped) << "an event after message_stop: " << type;
    if (type == "ping") continue;
    EXPECT_NE(type, "message_start") << "a second message_start";
    if (type == "content_block_start") {
      EXPECT_FALSE(open) << "a block started inside another";
      EXPECT_FALSE(delta_seen) << "a block after message_delta";
      const std::size_t index = static_cast<std::size_t>(e.find("index")->number());
      EXPECT_EQ(index, content.items().size()) << "block indices not contiguous";
      content.push(*e.find("content_block"));
      partial.emplace_back();
      open = index;
    } else if (type == "content_block_delta") {
      const std::size_t index = static_cast<std::size_t>(e.find("index")->number());
      EXPECT_EQ(open, std::optional<std::size_t>(index)) << "a delta for a block that is not open";
      if (!open || *open != index) continue;
      Value& block = content.items()[index];
      const std::string bt = block.find("type")->str();
      const Value& d = *e.find("delta");
      const std::string dt = d.find("type")->str();
      if (bt == "text") {
        EXPECT_EQ(dt, "text_delta");
        block.set("text", Value::String(block.find("text")->str() + d.find("text")->str()));
      } else if (bt == "thinking") {
        EXPECT_TRUE(dt == "thinking_delta" || dt == "signature_delta") << dt;
        EXPECT_TRUE(block.find("signature")->str().empty()) << "a delta after the signature";
        if (dt == "thinking_delta") block.set("thinking", Value::String(block.find("thinking")->str() + d.find("thinking")->str()));
        else block.set("signature", *d.find("signature"));
      } else {
        EXPECT_EQ(dt, "input_json_delta");
        partial[index] += d.find("partial_json")->str();
      }
    } else if (type == "content_block_stop") {
      const std::size_t index = static_cast<std::size_t>(e.find("index")->number());
      EXPECT_EQ(open, std::optional<std::size_t>(index));
      Value& block = content.items()[index];
      if (block.find("type")->str() == "tool_use") block.set("input", J(partial[index]));
      if (block.find("type")->str() == "thinking") EXPECT_FALSE(block.find("signature")->str().empty()) << "no signature";
      open.reset();
    } else if (type == "message_delta") {
      EXPECT_FALSE(open) << "message_delta inside a block";
      EXPECT_FALSE(delta_seen) << "a second message_delta";
      delta_seen = true;
      msg.set("stop_reason", *e.find("delta")->find("stop_reason"));
      msg.set("stop_sequence", *e.find("delta")->find("stop_sequence"));
      msg.set("usage", *e.find("usage"));  // cumulative
    } else if (type == "message_stop") {
      EXPECT_TRUE(delta_seen) << "message_stop without message_delta";
      stopped = true;
    } else {
      ADD_FAILURE() << "unknown event " << type;
    }
  }
  EXPECT_TRUE(stopped) << "no message_stop";
  msg.set("content", std::move(content));
  return msg;
}

}  // namespace

// Any sequence of parser events, pings anywhere after the start, any finish: the stream is grammatical and
// accumulates to exactly the final Message (thinking text and all, when not omitted).
TEST(AnthropicResponse, EveryStreamFollowsTheGrammarAndAccumulatesToTheMessage) {
  hegel::test([](hegel::TestCase& tc) {
    const bool omit = tc.draw(gs::booleans());
    const bool tools = tc.draw(gs::booleans());
    auto b = Builder(omit, tools);
    std::vector<Value> out;
    b.Start(tc.draw(gs::integers<std::uint64_t>({.max_value = 1000})), &out);
    const int n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 12}));
    for (int i = 0; i < n; ++i) {
      const int k = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 5}));
      const std::string t = tc.draw(gs::sampled_from<std::string>({"a", "b c", "", "é", "\n"}));
      std::vector<text::StreamEvent> evs;
      if (k == 0) evs.push_back(Ev(text::StreamEventKind::kContent, t));
      if (k == 1) evs.push_back(Ev(text::StreamEventKind::kReasoning, t));
      if (k == 2) evs.push_back(Ev(text::StreamEventKind::kFraming, t));
      if (k == 3) evs.push_back(Ev(text::StreamEventKind::kError, t));
      if (k == 4) evs.push_back(Call(tc.draw(gs::sampled_from<std::string>({"f", "g"})), J(R"({"x": [1, "y"]})")));
      if (k == 5) b.Ping(&out);
      b.Add(evs, &out);
    }
    const auto reason = static_cast<StopReason>(tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4})));
    const MessagesUsage u{tc.draw(gs::integers<std::uint64_t>({.max_value = 100})), 5, 7,
                          tc.draw(gs::integers<std::uint64_t>({.max_value = 100}))};
    b.Finish(reason, "S", u, &out);
    b.Ping(&out);  // after the end: nothing
    const Value got = Accumulate(out);
    if (::testing::Test::HasFailure()) return;
    ASSERT_EQ(CompactJson(got), CompactJson(b.Final()));
    ASSERT_EQ(std::count_if(out.begin(), out.end(), [](const Value& e) { return e.find("type")->str() == "message_stop"; }), 1);
  });
}

// ---- the service over HTTP -------------------------------------------------------------------------------------

namespace {

class ScriptedGenerator : public TextGenerator {
 public:
  Result Generate(const Params& p, const std::function<bool(std::string_view)>& on_text) override {
    std::string script;
    {
      std::lock_guard<std::mutex> lk(mu);
      prompts.push_back(p.prompt);
      max_tokens.push_back(p.max_tokens);
      script = scripts.empty() ? "ok" : scripts.front();
      if (!scripts.empty()) scripts.erase(scripts.begin());
    }
    // "Prefill", polling Params::cancelled as the engine's generator does between chunks (it writes the pings).
    for (const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(hold_ms);
         std::chrono::steady_clock::now() < until;) {
      if (p.cancelled) (void)p.cancelled();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Result r;
    {
      std::lock_guard<std::mutex> lk(mu);
      seen.push_back({p.principal, p.chain_root, p.idempotency_key, p.ticket != nullptr});
    }
    r.detached = detached;
    r.uncounted = uncounted;
    r.prompt_tokens = p.prompt.size();
    r.cached_tokens = cached;
    r.cache_write_tokens = written;
    for (std::size_t i = 0; i < script.size(); i += piece) {
      if (poll_cancel && p.cancelled && p.cancelled()) {
        r.cancelled = stopped_early = true;
        break;
      }
      if (r.completion_tokens == p.max_tokens || r.completion_tokens == context_cap) {
        r.length_limited = true;
        break;
      }
      ++r.completion_tokens;
      if (!on_text(std::string_view(script).substr(i, piece))) {
        stopped_early = true;
        // G3: a generator tells a departure (cancelled) from a stop string (not) after the callback said stop.
        cancelled_at_stop = p.cancelled && p.cancelled();
        break;
      }
      if (delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
    return r;
  }
  std::optional<std::uint64_t> CountTokens(std::string_view prompt) const override { return prompt.size(); }
  Admission Admit(const Params& p) override {
    std::lock_guard<std::mutex> lk(mu);
    admitted.push_back({p.principal, p.chain_root, p.idempotency_key, false});
    if (refuse_retry_s >= 0) return {true, refuse_retry_s, "limit", nullptr};
    return {false, 0, "", std::make_shared<int>(1)};
  }
  struct Seen {
    std::string principal, chain_root, key;
    bool ticket = false;
  };
  std::vector<Seen> admitted, seen;
  std::int64_t refuse_retry_s = -1;  // >= 0: Admit refuses with this Retry-After
  bool detached = false, uncounted = false;
  bool poll_cancel = true;  // false: only the text callback can find the client gone
  std::atomic<bool> cancelled_at_stop{false};
  std::mutex mu;
  std::vector<std::string> scripts, prompts;
  std::vector<std::uint32_t> max_tokens;
  std::size_t piece = 3;
  int delay_ms = 0, hold_ms = 0;
  std::uint64_t cached = 0, written = 0, context_cap = UINT64_MAX;
  std::atomic<bool> stopped_early{false};
};

struct Recording : GenerationObserver {
  void OnGeneration(const GenerationRecord& r) noexcept override {
    std::lock_guard<std::mutex> lk(mu);
    records.push_back(r);
  }
  std::mutex mu;
  std::vector<GenerationRecord> records;
};

struct Rig {
  explicit Rig(std::string api_key = "", int keepalive_ms = 0) : server({.api_key = std::move(api_key)}) {
    ServiceOptions o;
    o.observer = &observed;
    o.sse_keepalive_ms = keepalive_ms;
    int n = 0;
    o.new_id = [n]() mutable { return std::to_string(++n); };
    service = std::make_unique<AnthropicService>(gen, o, std::vector<ModelAlias>{{"claude-*", ""}, {"tiny", "qwen3.8-flash-next:tiny"}});
    service->Register(server);
    std::string err;
    if (!server.Start(&err)) throw std::runtime_error(err);
  }
  ~Rig() { server.Stop(); }
  ScriptedGenerator gen;
  Recording observed;
  HttpServer server;
  std::unique_ptr<AnthropicService> service;
};

struct Reply {
  int status = 0;
  std::string head, body;
};

Reply Post(int port, const std::string& path, const std::string& body, const std::string& headers = "",
           bool read_all = true) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) throw std::runtime_error("connect");
  const std::string req = "POST " + path + " HTTP/1.1\r\nConnection: close\r\n" + headers +
                          "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
  ::send(fd, req.data(), req.size(), MSG_NOSIGNAL);
  std::string raw;
  char b[8192];
  for (ssize_t n; (n = ::recv(fd, b, sizeof b, 0)) > 0;) {
    raw.append(b, static_cast<std::size_t>(n));
    if (!read_all) break;
  }
  ::close(fd);
  Reply r;
  const auto end = raw.find("\r\n\r\n");
  r.status = std::stoi(raw.substr(9, 3));
  r.head = raw.substr(0, end);
  const std::string rest = raw.substr(end + 4);
  if (r.head.find("Transfer-Encoding: chunked") == std::string::npos) return r.body = rest, r;
  for (std::size_t i = 0;;) {
    const auto eol = rest.find("\r\n", i);
    if (eol == std::string::npos) break;
    const std::size_t n = std::stoul(rest.substr(i, eol - i), nullptr, 16);
    if (n == 0) break;
    r.body += rest.substr(eol + 2, n);
    i = eol + 2 + n + 2;
  }
  return r;
}

/// The events of an SSE body; every frame names its event and the name matches the data's type.
std::vector<Value> SseEvents(const std::string& sse) {
  std::vector<Value> out;
  std::size_t i = 0;
  while (i < sse.size()) {
    const auto end = sse.find("\n\n", i);
    if (end == std::string::npos) break;
    const std::string frame = sse.substr(i, end - i);
    i = end + 2;
    if (frame.starts_with(":")) continue;
    EXPECT_TRUE(frame.starts_with("event: ")) << frame;
    const auto nl = frame.find('\n');
    const std::string name = frame.substr(7, nl - 7);
    EXPECT_TRUE(frame.substr(nl + 1).starts_with("data: ")) << frame;
    Value v = J(frame.substr(nl + 7));
    EXPECT_EQ(v.find("type")->str(), name);
    out.push_back(std::move(v));
  }
  return out;
}

const std::string kTools = R"([{"name": "weather", "description": "Weather in a city",
  "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}])";

}  // namespace

TEST(AnthropicService, WholeAndStreamedMessages) {
  Rig rig;
  const std::string answer = "Thinking hard.\n</think>\n\nThe answer is 4.";
  rig.gen.scripts = {answer, answer};
  const std::string body = R"({"model": "claude-sonnet-5-5", "max_tokens": 64, "system": "Be terse.",
    "messages": [{"role": "user", "content": "2+2?"}]})";
  const auto r = Post(rig.server.port(), "/v1/messages", body);
  ASSERT_EQ(r.status, 200) << r.body;
  const Value v = J(r.body);
  EXPECT_EQ(v.find("id")->str(), "msg_1");
  EXPECT_EQ(v.find("model")->str(), "qwen3.8-flash-next");  // the served model, as the API reports a resolved alias
  EXPECT_EQ(v.find("stop_reason")->str(), "end_turn");
  ASSERT_EQ(v.find("content")->items().size(), 2u);
  EXPECT_EQ(v.find("content")->items()[0].find("thinking")->str(), "Thinking hard.");
  EXPECT_EQ(v.find("content")->items()[1].find("text")->str(), "The answer is 4.");
  EXPECT_NE(r.head.find("request-id: req_1"), std::string::npos);
  EXPECT_EQ(rig.gen.prompts[0],
            Render(J(R"([{"role": "system", "content": "Be terse."}, {"role": "user", "content": "2+2?"}])")));
  // Streamed: the same message, accumulated from grammatical events.
  const auto s = Post(rig.server.port(), "/v1/messages",
                      body.substr(0, body.size() - 1) + R"(, "stream": true})");
  ASSERT_EQ(s.status, 200);
  EXPECT_NE(s.head.find("text/event-stream"), std::string::npos);
  Value got = Accumulate(SseEvents(s.body));
  EXPECT_EQ(got.find("content")->items()[1].find("text")->str(), "The answer is 4.");
  EXPECT_EQ(got.find("stop_reason")->str(), "end_turn");
  // message_start carried the provisional count (the whole prompt), message_delta the final usage.
  EXPECT_EQ(SseEvents(s.body).front().find("message")->find("usage")->find("input_tokens")->number(),
            static_cast<double>(rig.gen.prompts[1].size()));
  std::lock_guard<std::mutex> lk(rig.observed.mu);
  ASSERT_EQ(rig.observed.records.size(), 2u);
  EXPECT_EQ(rig.observed.records[0].api, "messages");
  EXPECT_EQ(rig.observed.records[0].finish_reason, "stop");
  EXPECT_EQ(rig.observed.records[0].model, "claude-sonnet-5-5");
}

// A tool call comes back as a tool_use block; sending it back with its tool_result renders the same conversation
// the template would render for the equivalent chat messages (the round trip a coding agent makes every turn).
TEST(AnthropicService, ToolUseRoundTrip) {
  Rig rig;
  rig.gen.scripts = {"I need the weather.\n</think>\n\n<tool_call>\n<function=weather>\n<parameter=city>\nParis\n"
                     "</parameter>\n</function>\n</tool_call>",
                     "Done.\n</think>\n\nIt is 18C."};
  const std::string first = R"({"model": "claude-x", "max_tokens": 64, "tools": )" + kTools +
                            R"(, "messages": [{"role": "user", "content": "Weather in Paris?"}]})";
  const auto r1 = Post(rig.server.port(), "/v1/messages", first);
  ASSERT_EQ(r1.status, 200) << r1.body;
  const Value m1 = J(r1.body);
  EXPECT_EQ(m1.find("stop_reason")->str(), "tool_use");
  const Value& use = m1.find("content")->items().back();
  ASSERT_EQ(use.find("type")->str(), "tool_use");
  EXPECT_EQ(use.find("name")->str(), "weather");
  EXPECT_EQ(CompactJson(*use.find("input")), R"({"city":"Paris"})");
  const std::string id = use.find("id")->str();
  EXPECT_TRUE(id.starts_with("toolu_"));
  // The next turn: the assistant content as returned, then the result.
  const std::string second = R"({"model": "claude-x", "max_tokens": 64, "tools": )" + kTools +
                             R"(, "messages": [{"role": "user", "content": "Weather in Paris?"},
       {"role": "assistant", "content": )" + CompactJson(*m1.find("content")) + R"(},
       {"role": "user", "content": [{"type": "tool_result", "tool_use_id": ")" + id + R"(", "content": "18C"}]}]})";
  const auto r2 = Post(rig.server.port(), "/v1/messages", second);
  ASSERT_EQ(r2.status, 200) << r2.body;
  EXPECT_EQ(J(r2.body).find("content")->items().back().find("text")->str(), "It is 18C.");
  const Value tools = J(R"([{"type": "function", "function": {"name": "weather", "description": "Weather in a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}])");
  const Value want = J(R"([{"role": "user", "content": "Weather in Paris?"},
      {"role": "assistant", "reasoning_content": "I need the weather.", "content": "",
       "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}]},
      {"role": "tool", "content": "18C"}])");
  ASSERT_EQ(rig.gen.prompts.size(), 2u);
  EXPECT_EQ(rig.gen.prompts[1], Render(want, &tools));
  // The first turn's prompt plus its generation is a prefix of the second prompt: the session cache can extend it.
  EXPECT_TRUE(rig.gen.prompts[1].starts_with(rig.gen.prompts[0]));
  // Streamed tool use: an input_json_delta that parses to the input.
  rig.gen.scripts = {"\n</think>\n\n<tool_call>\n<function=weather>\n<parameter=city>\nRome\n</parameter>\n"
                     "</function>\n</tool_call>"};
  const auto s = Post(rig.server.port(), "/v1/messages", first.substr(0, first.size() - 1) + R"(, "stream": true})");
  const Value got = Accumulate(SseEvents(s.body));
  EXPECT_EQ(got.find("stop_reason")->str(), "tool_use");
  EXPECT_EQ(CompactJson(*got.find("content")->items().back().find("input")), R"({"city":"Rome"})");
}

TEST(AnthropicService, StopSequencesLimitsAndCacheUsage) {
  Rig rig;
  rig.gen.piece = 2;
  rig.gen.cached = 40;
  rig.gen.written = 25;
  rig.gen.scripts = {"\n</think>\n\nHello STOP world", "\n</think>\n\nabcdefghij", "\n</think>\n\nabcdefghij",
                     "\n</think>\n\nHello ST"};
  const auto msg = [&](const std::string& extra) {
    return Post(rig.server.port(), "/v1/messages",
                R"({"model": "claude-x", "messages": [{"role": "user", "content": "hi"}])" + extra + "}");
  };
  const auto r1 = msg(R"(, "max_tokens": 50, "stop_sequences": ["STOP", "zzz"])");
  ASSERT_EQ(r1.status, 200) << r1.body;
  const Value v1 = J(r1.body);
  EXPECT_EQ(v1.find("stop_reason")->str(), "stop_sequence");
  EXPECT_EQ(v1.find("stop_sequence")->str(), "STOP");
  EXPECT_EQ(v1.find("content")->items()[0].find("text")->str(), "Hello");  // trailing space trimmed (template |trim)
  const std::uint64_t prompt = rig.gen.prompts[0].size();
  EXPECT_EQ(CompactJson(*v1.find("usage")), R"({"input_tokens":)" + std::to_string(prompt - 65) +
                                                R"(,"cache_creation_input_tokens":25,"cache_read_input_tokens":40,)"
                                                R"("output_tokens":)" + v1.find("usage")->find("output_tokens")->int_digits() + "}");
  const auto r2 = msg(R"(, "max_tokens": 4)");
  const Value v2 = J(r2.body);
  EXPECT_EQ(v2.find("stop_reason")->str(), "max_tokens");
  EXPECT_EQ(v2.find("usage")->find("output_tokens")->number(), 4);
  rig.gen.context_cap = 3;  // the context ran out before max_tokens
  const auto r3 = msg(R"(, "max_tokens": 40)");
  EXPECT_EQ(J(r3.body).find("stop_reason")->str(), "model_context_window_exceeded");
  // Ending on a prefix of a stop sequence: the text held back for it is delivered, and it is no stop.
  rig.gen.context_cap = UINT64_MAX;
  const Value v4 = J(msg(R"(, "max_tokens": 50, "stop_sequences": ["STOP"])").body);
  EXPECT_EQ(v4.find("content")->items()[0].find("text")->str(), "Hello ST");
  EXPECT_EQ(v4.find("stop_reason")->str(), "end_turn");
}

TEST(AnthropicService, DisableParallelToolUseStopsAfterTheFirstCall) {
  Rig rig;
  rig.gen.scripts = {"\n</think>\n\n<tool_call>\n<function=weather>\n<parameter=city>\nA\n</parameter>\n</function>\n"
                     "</tool_call>\n<tool_call>\n<function=weather>\n<parameter=city>\nB\n</parameter>\n</function>\n"
                     "</tool_call>"};
  const auto r = Post(rig.server.port(), "/v1/messages",
                      R"({"model": "claude-x", "max_tokens": 200, "tools": )" + kTools +
                          R"(, "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
                          "messages": [{"role": "user", "content": "go"}]})");
  ASSERT_EQ(r.status, 200) << r.body;
  const Value v = J(r.body);
  EXPECT_EQ(v.find("stop_reason")->str(), "tool_use");
  ASSERT_EQ(v.find("content")->items().size(), 1u);
  EXPECT_EQ(CompactJson(*v.find("content")->items()[0].find("input")), R"({"city":"A"})");
  EXPECT_TRUE(rig.gen.stopped_early.load());
}

TEST(AnthropicService, ForcedToolChoiceFeedsTheOpenedCallToTheParser) {
  Rig rig;
  // The model continues after "<tool_call>\n<function=weather>\n" (the prompt opened it).
  rig.gen.scripts = {"<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>"};
  const auto r = Post(rig.server.port(), "/v1/messages",
                      R"({"model": "claude-x", "max_tokens": 200, "tools": )" + kTools +
                          R"(, "tool_choice": {"type": "tool", "name": "weather"},
                          "messages": [{"role": "user", "content": "go"}]})");
  ASSERT_EQ(r.status, 200) << r.body;
  const Value v = J(r.body);
  ASSERT_EQ(v.find("content")->items().size(), 1u);
  EXPECT_EQ(v.find("content")->items()[0].find("name")->str(), "weather");
  EXPECT_EQ(CompactJson(*v.find("content")->items()[0].find("input")), R"({"city":"Oslo"})");
  EXPECT_TRUE(rig.gen.prompts[0].ends_with("</think>\n\n<tool_call>\n<function=weather>\n")) << rig.gen.prompts[0];
}

TEST(AnthropicService, ErrorsAuthAliasesAndCountTokens) {
  Rig rig("sekret");
  const int port = rig.server.port();
  const std::string ok = R"({"model": "claude-x", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})";
  // Auth: x-api-key or Bearer; a missing key is an Anthropic-shaped 401 on these routes.
  const auto noauth = Post(port, "/v1/messages", ok);
  EXPECT_EQ(noauth.status, 401);
  EXPECT_EQ(CompactJson(J(noauth.body)), R"({"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}})");
  EXPECT_EQ(Post(port, "/v1/messages", ok, "x-api-key: wrong\r\n").status, 401);
  EXPECT_EQ(Post(port, "/v1/messages", ok, "x-api-key: sekret\r\n").status, 200);
  EXPECT_EQ(Post(port, "/v1/messages", ok, "Authorization: Bearer sekret\r\n").status, 200);
  const std::string key = "x-api-key: sekret\r\nanthropic-version: 2023-06-01\r\n";
  // Errors in the API's shape.
  const auto bad = Post(port, "/v1/messages", "{nope", key);
  EXPECT_EQ(bad.status, 400);
  EXPECT_EQ(J(bad.body).find("error")->find("type")->str(), "invalid_request_error");
  const auto img = Post(port, "/v1/messages",
                        R"({"model": "claude-x", "max_tokens": 8, "messages": [{"role": "user", "content":
                            [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}}]}]})", key);
  EXPECT_EQ(img.status, 400);
  EXPECT_NE(J(img.body).find("error")->find("message")->str().find("image input is not supported"), std::string::npos);
  const auto unknown = Post(port, "/v1/messages", R"({"model": "gpt-9", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})", key);
  EXPECT_EQ(unknown.status, 404);
  EXPECT_EQ(J(unknown.body).find("error")->find("type")->str(), "not_found_error");
  // An alias to a variant the generator does not serve.
  EXPECT_EQ(Post(port, "/v1/messages", R"({"model": "tiny", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})", key).status, 404);
  // The served model's own id needs no alias.
  EXPECT_EQ(Post(port, "/v1/messages", R"({"model": "qwen3.8-flash-next", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})", key).status, 200);
  // Fields steering only Anthropic's service: accepted and named.
  const auto ign = Post(port, "/v1/messages",
                        R"({"model": "claude-x", "max_tokens": 8, "safeguards": [], "context_management": {"edits": []},
                            "messages": [{"role": "user", "content": "hi"}]})", key);
  EXPECT_EQ(ign.status, 200);
  EXPECT_NE(ign.head.find("x-halo-ignored: context_management,safeguards"), std::string::npos) << ign.head;
  // count_tokens: the rendered prompt's tokens (the scripted generator counts bytes).
  const auto ct = Post(port, "/v1/messages/count_tokens",
                       R"({"model": "claude-x", "system": "S", "messages": [{"role": "user", "content": "hi"}]})", key);
  ASSERT_EQ(ct.status, 200) << ct.body;
  const std::string rendered = Render(J(R"([{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}])"));
  EXPECT_EQ(CompactJson(J(ct.body)), R"({"input_tokens":)" + std::to_string(rendered.size()) + "}");
  EXPECT_EQ(Post(port, "/v1/messages/count_tokens", R"({"model": "claude-x"})", key).status, 400);
}

TEST(AnthropicService, PingsWhileAStreamWaitsAndAGenerationErrorEndsTheStream) {
  Rig rig("", /*keepalive_ms=*/20);
  rig.gen.hold_ms = 150;
  rig.gen.scripts = {"\n</think>\n\nhi"};
  const auto s = Post(rig.server.port(), "/v1/messages",
                      R"({"model": "claude-x", "max_tokens": 8, "stream": true, "messages": [{"role": "user", "content": "x"}]})");
  const auto events = SseEvents(s.body);
  std::size_t pings = 0;
  for (const auto& e : events) pings += e.find("type")->str() == "ping";
  EXPECT_GE(pings, 3u) << "about one every 20 ms over 150 ms";
  EXPECT_EQ(events.front().find("type")->str(), "message_start");  // pings follow message_start
  EXPECT_EQ(Accumulate(events).find("content")->items()[0].find("text")->str(), "hi");
}

namespace {
struct FailingGenerator : TextGenerator {
  Result Generate(const Params&, const std::function<bool(std::string_view)>& on_text) override {
    on_text("\n</think>\n\npartial");
    Result r;
    r.error = "device lost";
    return r;
  }
};
}  // namespace

TEST(AnthropicService, AGenerationErrorIsAnErrorEventOrA500) {
  FailingGenerator gen;
  HttpServer server({});
  AnthropicService service(gen, ServiceOptions{});
  service.Register(server);
  std::string err;
  ASSERT_TRUE(server.Start(&err)) << err;
  const std::string body = R"({"model": "claude-x", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}])";
  const auto whole = Post(server.port(), "/v1/messages", body + "}");
  EXPECT_EQ(whole.status, 500);
  EXPECT_EQ(J(whole.body).find("error")->find("type")->str(), "api_error");
  const auto s = Post(server.port(), "/v1/messages", body + R"(, "stream": true})");
  const auto events = SseEvents(s.body);
  ASSERT_FALSE(events.empty());
  EXPECT_EQ(events.front().find("type")->str(), "message_start");
  EXPECT_EQ(events.back().find("type")->str(), "error");
  EXPECT_NE(events.back().find("error")->find("message")->str().find("device lost"), std::string::npos);
  for (const auto& e : events) EXPECT_NE(e.find("type")->str(), "message_stop");
  server.Stop();
}

TEST(AnthropicService, ADepartingStreamStopsGenerationAndIsRecordedCancelled) {
  Rig rig;
  rig.gen.scripts = {std::string(30000, 'a')};
  rig.gen.piece = 1;
  rig.gen.delay_ms = 1;
  (void)Post(rig.server.port(), "/v1/messages",
             R"({"model": "claude-x", "max_tokens": 30000, "stream": true, "messages": [{"role": "user", "content": "x"}]})",
             "", /*read_all=*/false);
  for (int i = 0; i < 500 && !rig.gen.stopped_early.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_TRUE(rig.gen.stopped_early.load());
  for (int i = 0; i < 500; ++i) {
    {
      std::lock_guard<std::mutex> lk(rig.observed.mu);
      if (!rig.observed.records.empty()) break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::lock_guard<std::mutex> lk(rig.observed.mu);
  ASSERT_EQ(rig.observed.records.size(), 1u);
  EXPECT_EQ(rig.observed.records[0].finish_reason, "cancelled");
}

// G3 through /v1/messages: the scheduling headers reach the generator with the admission's ticket; a request a broker
// stamped with a principal and no key gets the same default key on a retry; a refused admission is a 429 with
// Retry-After in the API's error shape, before anything streams, and never generates.
TEST(AnthropicService, SchedulingHeadersKeysAndRefusal) {
  Rig rig;
  const std::string body = R"({"model": "claude-x", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})";
  auto r = Post(rig.server.port(), "/v1/messages", body,
                "X-Halo-Principal: sb\r\nX-Halo-Chain-Id: root\r\nIdempotency-Key: k1\r\n");
  ASSERT_EQ(r.status, 200) << r.body;
  r = Post(rig.server.port(), "/v1/messages", body, "X-Halo-Principal: sb\r\n");
  ASSERT_EQ(r.status, 200) << r.body;
  r = Post(rig.server.port(), "/v1/messages", body, "X-Halo-Principal: sb\r\n");
  ASSERT_EQ(r.status, 200) << r.body;
  r = Post(rig.server.port(), "/v1/messages", body);  // anonymous: no default key
  ASSERT_EQ(r.status, 200) << r.body;
  {
    std::lock_guard<std::mutex> lk(rig.gen.mu);
    ASSERT_EQ(rig.gen.seen.size(), 4u);
    EXPECT_EQ(rig.gen.seen[0].principal, "sb");
    EXPECT_EQ(rig.gen.seen[0].chain_root, "root");
    EXPECT_EQ(rig.gen.seen[0].key, "k1");
    EXPECT_TRUE(rig.gen.seen[0].ticket);
    EXPECT_TRUE(rig.gen.seen[1].key.starts_with("d5:")) << rig.gen.seen[1].key;
    EXPECT_EQ(rig.gen.seen[1].key, rig.gen.seen[2].key);
    EXPECT_EQ(rig.gen.seen[3].key, "");
    EXPECT_EQ(rig.gen.admitted.size(), 4u);
    EXPECT_EQ(rig.gen.admitted[0].key, "k1");
  }
  rig.gen.refuse_retry_s = 7;
  for (const bool stream : {false, true}) {
    const auto x = Post(rig.server.port(), "/v1/messages",
                        stream ? body.substr(0, body.size() - 1) + R"(, "stream": true})" : body,
                        "X-Halo-Principal: sb\r\n");
    EXPECT_EQ(x.status, 429) << x.body;
    EXPECT_NE(x.head.find("Retry-After: 7"), std::string::npos) << x.head;
    const Value v = J(x.body);
    EXPECT_EQ(v.find("type")->str(), "error");
    EXPECT_EQ(v.find("error")->find("type")->str(), "rate_limit_error");
  }
  std::lock_guard<std::mutex> lk(rig.gen.mu);
  EXPECT_EQ(rig.gen.seen.size(), 4u);  // the refused requests never generated
}

// G3: a client that left is reported to the generator through Params::cancelled (a stop string is not), so a keyed
// request is not completed truncated; a reader that detached from a generation running on is recorded "detached"
// and, when another reader counts the generation, with no completion tokens of its own.
TEST(AnthropicService, ADepartureIsNotAStopAndADetachedReaderIsRecordedSo) {
  {
    Rig rig;
    rig.gen.scripts = {"one two END three"};
    rig.gen.piece = 1;
    const auto r = Post(rig.server.port(), "/v1/messages",
                        R"({"model": "claude-x", "max_tokens": 64, "stop_sequences": ["END"],
                            "messages": [{"role": "user", "content": "x"}]})");
    ASSERT_EQ(r.status, 200) << r.body;
    EXPECT_TRUE(rig.gen.stopped_early.load());
    EXPECT_FALSE(rig.gen.cancelled_at_stop.load());
  }
  {
    Rig rig;
    rig.gen.scripts = {std::string(30000, 'a')};
    rig.gen.piece = 1;
    rig.gen.delay_ms = 1;
    rig.gen.detached = rig.gen.uncounted = true;
    rig.gen.poll_cancel = false;
    (void)Post(rig.server.port(), "/v1/messages",
               R"({"model": "claude-x", "max_tokens": 30000, "stream": true, "messages": [{"role": "user", "content": "x"}]})",
               "", /*read_all=*/false);
    for (int i = 0; i < 500 && !rig.gen.stopped_early.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ASSERT_TRUE(rig.gen.stopped_early.load());
    EXPECT_TRUE(rig.gen.cancelled_at_stop.load());
    for (int i = 0; i < 500; ++i) {
      {
        std::lock_guard<std::mutex> lk(rig.observed.mu);
        if (!rig.observed.records.empty()) break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::lock_guard<std::mutex> lk(rig.observed.mu);
    ASSERT_EQ(rig.observed.records.size(), 1u);
    EXPECT_EQ(rig.observed.records[0].finish_reason, "detached");
    EXPECT_EQ(rig.observed.records[0].completion_tokens, 0u);
  }
}
