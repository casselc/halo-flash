// OpenAI request normalization (server/openai_request):
//   - directed: a Responses conversation maps onto the expected template
//     messages; errors name their parameter;
//   - property: the same conversation sent as Chat Completions and as
//     Responses renders to the identical Flash-Next prompt (both APIs are
//     views of one template);
//   - property: arbitrary JSON bodies never crash a parser, and any request
//     a parser accepts renders or fails with an error, never a crash.

#include <hegel/gtest.h>

#include <functional>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "server/openai_request.hpp"
#include "text/chat_template.hpp"

namespace gs = hegel::generators;
using namespace halo;
using namespace halo::server;
using json::Value;

namespace {

Value J(std::string_view text) {
  std::string err;
  auto v = json::Parse(text, &err);
  if (!v) throw std::runtime_error("bad test JSON: " + err + " in " + std::string(text));
  return *v;
}

std::string Render(const GenerationRequest& r) {
  std::string err;
  auto s = text::RenderChat(text::ChatTemplateKind::kFlashNext, r.messages, r.tools.is(Value::Kind::kNull) ? nullptr : &r.tools,
                            r.template_options, &err);
  if (!s) throw std::runtime_error("render failed: " + err);
  return *s;
}

}  // namespace

TEST(OpenAiRequest, ResponsesItemsBecomeTemplateTurns) {
  RequestError e;
  const auto r = ParseResponsesRequest(J(R"({
    "model": "flash", "instructions": "Be brief.",
    "input": [
      {"role": "user", "content": "Weather in Paris and Rome?"},
      {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Need two lookups."}]},
      {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{\"city\": \"Paris\"}"},
      {"type": "function_call", "call_id": "c2", "name": "weather", "arguments": "{\"city\": \"Rome\"}"},
      {"type": "function_call_output", "call_id": "c1", "output": "18C"},
      {"type": "function_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "21C"}]},
      {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Paris 18C, Rome 21C."}]},
      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Thanks"}]}
    ],
    "tools": [{"type": "function", "name": "weather", "description": "Look up weather",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}],
    "tool_choice": {"type": "function", "name": "weather"},
    "reasoning": {"effort": "high"}, "max_output_tokens": 100, "temperature": 0.2,
    "text": {"format": {"type": "json_schema", "name": "w", "schema": {"type": "object"}, "strict": true}},
    "store": false, "truncation": "auto", "metadata": {"run": "7"}
  })"), RequestDefaults{}, &e);
  ASSERT_TRUE(r) << e.param << ": " << e.message;
  EXPECT_EQ(json::DumpPython(r->messages), json::DumpPython(J(R"([
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Weather in Paris and Rome?"},
    {"role": "assistant", "reasoning_content": "Need two lookups.", "tool_calls": [
      {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}},
      {"id": "c2", "type": "function", "function": {"name": "weather", "arguments": {"city": "Rome"}}}],
     "content": ""},
    {"role": "tool", "content": "18C", "tool_call_id": "c1"},
    {"role": "tool", "content": [{"type": "text", "text": "21C"}], "tool_call_id": "c2"},
    {"role": "assistant", "content": [{"type": "text", "text": "Paris 18C, Rome 21C."}]},
    {"role": "user", "content": [{"type": "text", "text": "Thanks"}]}
  ])")));
  EXPECT_EQ(json::DumpPython(r->tools), json::DumpPython(J(R"([{"type": "function", "function": {
    "name": "weather", "description": "Look up weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}])")));
  EXPECT_EQ(r->tool_choice.mode, ToolChoice::Mode::kFunction);
  EXPECT_EQ(r->tool_choice.function, "weather");
  EXPECT_EQ(r->template_options.reasoning_effort, "high");
  EXPECT_EQ(r->max_tokens, 100u);
  EXPECT_FLOAT_EQ(r->sampling.temperature, 0.2f);
  EXPECT_EQ(r->format.kind, ResponseFormat::Kind::kJsonSchema);
  EXPECT_TRUE(r->format.strict);
  EXPECT_FALSE(r->store);
  EXPECT_TRUE(r->truncation_auto);
  const std::string prompt = Render(*r);
  EXPECT_NE(prompt.find("<parameter=city>\nRome\n</parameter>"), std::string::npos);
  EXPECT_TRUE(prompt.ends_with("<|im_start|>assistant\n<think>\n"));
}

// Interleaved thinking within one response (reasoning, call, reasoning,
// call): the template has one reasoning block per assistant turn, so a
// reasoning item after output opens the next turn; consecutive reasoning
// items before any output join.
TEST(OpenAiRequest, ReasoningAfterOutputOpensTheNextAssistantTurn) {
  RequestError e;
  const auto r = ParseResponsesRequest(J(R"({"input": [
      {"role": "user", "content": "go"},
      {"type": "reasoning", "summary": [{"type": "summary_text", "text": "a"}]},
      {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "b"}], "summary": []},
      {"type": "function_call", "call_id": "c1", "name": "f", "arguments": ""},
      {"type": "reasoning", "summary": [{"type": "summary_text", "text": "c"}]},
      {"type": "function_call", "call_id": "c2", "name": "f", "arguments": "{}"}]})"),
                                       RequestDefaults{}, &e);
  ASSERT_TRUE(r) << e.param << ": " << e.message;
  EXPECT_EQ(json::DumpPython(r->messages), json::DumpPython(J(R"([
    {"role": "user", "content": "go"},
    {"role": "assistant", "reasoning_content": "a\n\nb", "tool_calls": [
      {"id": "c1", "type": "function", "function": {"name": "f", "arguments": {}}}], "content": ""},
    {"role": "assistant", "reasoning_content": "c", "tool_calls": [
      {"id": "c2", "type": "function", "function": {"name": "f", "arguments": {}}}], "content": ""}])")));
}

TEST(OpenAiRequest, UnsupportedAndInvalidFieldsNameTheirParameter) {
  const RequestDefaults d;
  struct Case {
    const char* body;
    bool responses;
    const char* param;
  };
  for (const auto& c : std::vector<Case>{
           {R"({"messages": [{"role": "user", "content": "x"}], "n": 2})", false, "n"},
           {R"({"messages": [{"role": "user", "content": "x"}], "temperature": 2.5})", false, "temperature"},
           {R"({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}]})", false,
            "messages[0].content[0].type"},
           {R"({"messages": [{"role": "robot", "content": "x"}]})", false, "messages[0].role"},
           {R"({"messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}]}]})",
            false, "messages[0].tool_calls[0].function.arguments"},
           {R"({"messages": [{"role": "user", "content": "x"}], "tool_choice": {"type": "function", "function": {"name": "g"}}})",
            false, "tool_choice"},
           {R"({"messages": [{"role": "user", "content": "x"}], "top_logprobs": 3})", false, "top_logprobs"},
           {R"({"messages": [{"role": "user", "content": "x"}], "stop": ["a", "b", "c", "d", "e"]})", false, "stop"},
           {R"({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0})", false, "max_tokens"},
           {R"({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "max"})", false, "reasoning_effort"},
           {R"({"input": "x", "tools": [{"type": "web_search"}]})", true, "tools[0].type"},
           {R"({"input": [{"type": "item_reference", "id": "msg_1"}]})", true, "input[0].type"},
           {R"({"input": [{"type": "function_call", "name": "f", "arguments": "{}"}]})", true, "input[0].call_id"},
           {R"({"input": [{"type": "reasoning", "encrypted_content": "abc"}]})", true, "input[0]"},
           {R"({"input": "x", "truncation": "middle"})", true, "truncation"},
           {R"({"input": 3})", true, "input"},
           {R"({"model": "m"})", true, "input"},
       }) {
    RequestError e;
    const auto body = J(c.body);
    const bool ok = c.responses ? ParseResponsesRequest(body, d, &e).has_value() : ParseChatRequest(body, d, &e).has_value();
    EXPECT_FALSE(ok) << c.body;
    EXPECT_EQ(e.param, c.param) << c.body << " -> " << e.message;
  }
  RequestError e{"bad", "temperature"};
  EXPECT_EQ(json::DumpPython(ErrorBody(e)),
            R"({"error": {"message": "bad", "type": "invalid_request_error", "param": "temperature", "code": null}})");
}

TEST(OpenAiRequest, ChatSettingsAndDefaults) {
  RequestDefaults d;
  RequestError e;
  auto r = ParseChatRequest(J(R"({"messages": [{"role": "user", "content": "hi"}]})"), d, &e);
  ASSERT_TRUE(r) << e.message;
  EXPECT_FLOAT_EQ(r->sampling.temperature, d.sampling.temperature);
  EXPECT_EQ(r->sampling.top_k, d.sampling.top_k);
  EXPECT_EQ(r->max_tokens, d.max_tokens);
  EXPECT_TRUE(r->template_options.add_generation_prompt);
  r = ParseChatRequest(J(R"({"messages": [{"role": "user", "content": "hi"}], "temperature": 0, "top_k": -1,
    "min_p": 0.1, "max_completion_tokens": 9, "max_tokens": 3, "stop": "END", "seed": -5, "logprobs": true,
    "top_logprobs": 2, "stream": true, "stream_options": {"include_usage": true}, "reasoning_effort": "none",
    "response_format": {"type": "json_object"}})"), d, &e);
  ASSERT_TRUE(r) << e.message;
  EXPECT_EQ(r->sampling.temperature, 0.0f);
  EXPECT_EQ(r->sampling.top_k, 0);
  EXPECT_EQ(r->max_tokens, 9u);  // max_completion_tokens wins
  EXPECT_EQ(r->stop, std::vector<std::string>{"END"});
  EXPECT_GE(r->sampling.seed, 0);
  EXPECT_EQ(r->top_logprobs, 2u);
  EXPECT_TRUE(r->stream && r->include_usage);
  EXPECT_EQ(r->template_options.enable_thinking, false);
  EXPECT_EQ(r->format.kind, ResponseFormat::Kind::kJsonObject);
  // Completions: OpenAI's 16-token default, single prompts only.
  r = ParseCompletionsRequest(J(R"({"prompt": ["Once upon"]})"), d, &e);
  ASSERT_TRUE(r) << e.message;
  EXPECT_EQ(r->prompt, "Once upon");
  EXPECT_EQ(r->max_tokens, 16u);
  EXPECT_FALSE(ParseCompletionsRequest(J(R"({"prompt": [1, 2]})"), d, &e));
  EXPECT_EQ(e.param, "prompt");
  EXPECT_FALSE(ParseCompletionsRequest(J(R"({"prompt": "x", "echo": true})"), d, &e));
  EXPECT_EQ(e.param, "echo");
}

// One conversation, both wire formats: the renders must be identical.
TEST(OpenAiRequest, ChatAndResponsesRenderTheSameConversationIdentically) {
  hegel::test([](hegel::TestCase& tc) {
    const auto text = [&] {
      return tc.draw(gs::sampled_from<std::string>({"hi", "Paris?", "a\nb", "", " spaced ", "{\"x\": 1}", "é"}));
    };
    Value chat = Value::MakeArray(), items = Value::MakeArray();
    const bool instructions = tc.draw(gs::booleans());
    std::string instr;
    if (instructions) {
      instr = text();
      chat.push(J(R"({"role": "system"})"));
      chat.items().back().set("content", Value::String(instr));
    }
    const auto user = [&] {
      const std::string t = text();
      Value m = J(R"({"role": "user"})");
      m.set("content", Value::String(t));
      chat.push(m);
      items.push(m);
    };
    user();
    int call_seq = 0;
    const auto turns = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4}));
    for (int turn = 0; turn < turns; ++turn) {
      // An assistant turn: optional reasoning, then calls or content.
      Value a = J(R"({"role": "assistant"})");
      if (tc.draw(gs::booleans())) {
        const std::string r = text();
        a.set("reasoning_content", Value::String(r));
        Value item = J(R"({"type": "reasoning", "summary": [{"type": "summary_text"}]})");
        const_cast<Value&>(item.find("summary")->items()[0]).set("text", Value::String(r));
        items.push(item);
      }
      const auto ncalls = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 2}));
      if (ncalls == 0) {
        const std::string c = text();
        a.set("content", Value::String(c));
        Value item = J(R"({"type": "message", "role": "assistant"})");
        item.set("content", Value::String(c));
        items.push(item);
        chat.push(a);
      } else {
        Value calls = Value::MakeArray();
        std::vector<std::string> ids;
        for (int k = 0; k < ncalls; ++k) {
          const std::string id = "call_" + std::to_string(++call_seq);
          const std::string args = tc.draw(gs::sampled_from<std::string>({"{}", "{\"q\": \"x\"}", "{\"n\": 3, \"l\": [1, 2]}", ""}));
          Value call = J(R"({"type": "function", "function": {"name": "f"}})");
          call.set("id", Value::String(id));
          const_cast<Value&>(*call.find("function")).set("arguments", Value::String(args));
          calls.push(call);
          Value item = J(R"({"type": "function_call", "name": "f"})");
          item.set("call_id", Value::String(id));
          item.set("arguments", Value::String(args));
          items.push(item);
          ids.push_back(id);
        }
        a.set("tool_calls", calls);
        a.set("content", Value::String(""));
        chat.push(a);
        for (const auto& id : ids) {
          const std::string out = text();
          Value t = J(R"({"role": "tool"})");
          t.set("content", Value::String(out));
          t.set("tool_call_id", Value::String(id));
          chat.push(t);
          Value item = J(R"({"type": "function_call_output"})");
          item.set("call_id", Value::String(id));
          item.set("output", Value::String(out));
          items.push(item);
        }
      }
      // Responses items carry no turn boundaries: an assistant message and a
      // following function call are one turn, as the model emits them. A
      // final answer is therefore always followed by the user's next message.
      if (ncalls == 0 || tc.draw(gs::booleans())) user();
    }
    const bool tools = tc.draw(gs::booleans());
    const std::string effort = tc.draw(gs::sampled_from<std::string>({"", "none", "low", "high"}));
    Value chat_body = Value::MakeObject(), resp_body = Value::MakeObject();
    chat_body.set("messages", chat);
    resp_body.set("input", items);
    if (instructions) resp_body.set("instructions", Value::String(instr));
    if (tools) {
      chat_body.set("tools", J(R"([{"type": "function", "function": {"name": "f", "description": "d",
                                   "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}])"));
      resp_body.set("tools", J(R"([{"type": "function", "name": "f", "description": "d",
                                   "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}])"));
    }
    if (!effort.empty()) {
      chat_body.set("reasoning_effort", Value::String(effort));
      Value r = Value::MakeObject();
      r.set("effort", Value::String(effort));
      resp_body.set("reasoning", r);
    }
    RequestError e1, e2;
    const auto c = ParseChatRequest(chat_body, RequestDefaults{}, &e1);
    const auto r = ParseResponsesRequest(resp_body, RequestDefaults{}, &e2);
    ASSERT_TRUE(c) << e1.param << ": " << e1.message;
    ASSERT_TRUE(r) << e2.param << ": " << e2.message;
    ASSERT_EQ(Render(*c), Render(*r));
  }, {.test_cases = 300});
}

// Bodies drawn from the fields these APIs read, with values of every JSON
// kind: parsing never crashes, and an accepted request renders or reports an
// error.
TEST(OpenAiRequest, ArbitraryBodiesNeverCrash) {
  hegel::test([](hegel::TestCase& tc) {
    std::function<Value(int)> any = [&](int depth) -> Value {
      switch (tc.draw(gs::integers<int>({.min_value = 0, .max_value = depth > 0 ? 7 : 4}))) {
        case 0: return Value::Null();
        case 1: return Value::Bool(tc.draw(gs::booleans()));
        case 2: return Value::Int(tc.draw(gs::integers<std::int64_t>({.min_value = -3, .max_value = 300})));
        case 3: return Value::Float(tc.draw(gs::sampled_from<double>({0.0, 0.5, 1.0, 2.0, -1.0, 1e300})));
        case 4:
          return Value::String(tc.draw(gs::sampled_from<std::string>(
              {"", "user", "assistant", "tool", "system", "function", "text", "input_text", "message", "reasoning",
               "function_call", "function_call_output", "auto", "none", "required", "{}", "{\"a\": 1}", "low", "json_schema"})));
        default: {
          const bool obj = tc.draw(gs::booleans());
          Value v = obj ? Value::MakeObject() : Value::MakeArray();
          const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4}));
          for (int i = 0; i < n; ++i) {
            if (obj)
              v.set(tc.draw(gs::sampled_from<std::string>({"role", "content", "type", "text", "name", "arguments", "call_id",
                                                           "output", "function", "tool_calls", "summary", "id", "parameters",
                                                           "json_schema", "schema", "effort", "format"})),
                    any(depth - 1));
            else
              v.push(any(depth - 1));
          }
          return v;
        }
      }
    };
    Value body = Value::MakeObject();
    const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 8}));
    for (int i = 0; i < n; ++i)
      body.set(tc.draw(gs::sampled_from<std::string>(
                   {"messages", "input", "instructions", "tools", "tool_choice", "temperature", "top_p", "top_k", "n",
                    "stop", "max_tokens", "max_output_tokens", "reasoning", "reasoning_effort", "text",
                    "response_format", "logprobs", "top_logprobs", "seed", "prompt", "metadata", "truncation",
                    "chat_template_kwargs", "stream_options"})),
               any(3));
    RequestError e;
    for (auto parse : {&ParseChatRequest, &ParseResponsesRequest, &ParseCompletionsRequest}) {
      const auto r = parse(body, RequestDefaults{}, &e);
      if (r && r->api != Api::kCompletions) {
        std::string err;
        (void)text::RenderChat(text::ChatTemplateKind::kFlashNext, r->messages,
                               r->tools.is(Value::Kind::kNull) ? nullptr : &r->tools, r->template_options, &err);
      }
    }
  }, {.test_cases = 2000});
}
