// OpenAI output shapes (server/openai_response) built from parser events:
//   - directed: a reasoning + content + tool-call generation in each API;
//   - property: for any generation, chunked any way, the streamed deltas
//     concatenate to the final object; Responses events are numbered from
//     0, every item added is done in order, and the final output is the
//     done items; streaming and non-streaming builds give the same object;
//   - property: a Responses output fed back as input items becomes the
//     assistant turn the Chat API returns for the same generation.

#include <hegel/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "server/openai_request.hpp"
#include "server/openai_response.hpp"
#include "text/tool_call_parser.hpp"

namespace gs = hegel::generators;
using namespace halo;
using namespace halo::server;
using json::Value;

namespace {

Value J(std::string_view text) {
  std::string err;
  auto v = json::Parse(text, &err);
  if (!v) throw std::runtime_error("bad test JSON: " + err);
  return *v;
}

GenerationRequest Request(Api api) {
  RequestError e;
  std::optional<GenerationRequest> r;
  const Value tools = J(R"([{"type": "function", "function": {"name": "weather", "parameters": {"type": "object",
      "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}])");
  if (api == Api::kChat) {
    Value b = J(R"({"model": "flash", "messages": [{"role": "user", "content": "hi"}], "stream_options": {"include_usage": true}})");
    b.set("tools", tools);
    r = ParseChatRequest(b, RequestDefaults{}, &e);
  } else if (api == Api::kResponses) {
    r = ParseResponsesRequest(J(R"({"model": "flash", "input": "hi", "tools": [{"type": "function", "name": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}]})"),
                              RequestDefaults{}, &e);
  } else {
    r = ParseCompletionsRequest(J(R"({"model": "flash", "prompt": "hi"})"), RequestDefaults{}, &e);
  }
  if (!r) throw std::runtime_error(e.message);
  return *r;
}

OutputBuilder::IdSource Counter() {
  auto n = std::make_shared<int>(0);
  return [n] { return std::to_string(++*n); };
}

text::ToolCallParserOptions ParserOptions(const GenerationRequest& r) {
  text::ToolCallParserOptions o;
  o.start = text::StreamStart::kReasoning;
  o.tools = r.tools;
  return o;
}

const std::string kGeneration =
    "The user wants weather.\n</think>\n\nChecking.\n\n<tool_call>\n<function=weather>\n<parameter=city>\nParis\n"
    "</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";

}  // namespace

TEST(OpenAiResponse, ChatCompletionCarriesReasoningContentAndCalls) {
  const auto req = Request(Api::kChat);
  OutputBuilder b(req, "chatcmpl-1", 1700000000, Counter());
  b.Add(text::ParseAssistantText(kGeneration, ParserOptions(req)), nullptr);
  b.Finish(FinishReason::kStop, Usage{10, 4, 20, 6}, nullptr);
  const Value v = b.Final();
  const Value& m = *v.find("choices")->items()[0].find("message");
  EXPECT_EQ(m.find("reasoning_content")->str(), "The user wants weather.");
  EXPECT_EQ(m.find("content")->str(), "Checking.");
  const Value& call = m.find("tool_calls")->items()[0];
  EXPECT_EQ(call.find("function")->find("name")->str(), "weather");
  EXPECT_EQ(call.find("function")->find("arguments")->str(), R"({"city":"Paris","days":3})");
  EXPECT_EQ(v.find("choices")->items()[0].find("finish_reason")->str(), "tool_calls");
  EXPECT_EQ(v.find("usage")->find("total_tokens")->number(), 30);
  EXPECT_EQ(v.find("usage")->find("prompt_tokens_details")->find("cached_tokens")->number(), 4);
}

TEST(OpenAiResponse, ResponsesOutputIsReasoningMessageAndFunctionCall) {
  const auto req = Request(Api::kResponses);
  OutputBuilder b(req, "resp_1", 1700000000, Counter());
  std::vector<Value> events;
  b.Add(text::ParseAssistantText(kGeneration, ParserOptions(req)), &events);
  b.Finish(FinishReason::kStop, Usage{10, 0, 20, 6}, &events);
  std::vector<std::string> types;
  for (const auto& e : events) types.push_back(e.find("type")->str());
  EXPECT_EQ(types, (std::vector<std::string>{
                       "response.created", "response.in_progress",
                       "response.output_item.added", "response.reasoning_text.delta", "response.reasoning_text.done",
                       "response.output_item.done",
                       "response.output_item.added", "response.content_part.added", "response.output_text.delta",
                       "response.output_text.done", "response.content_part.done", "response.output_item.done",
                       "response.output_item.added", "response.function_call_arguments.delta",
                       "response.function_call_arguments.done", "response.output_item.done", "response.completed"}));
  const Value v = b.Final();
  EXPECT_EQ(json::DumpPython(*v.find("output")), json::DumpPython(J(R"([
    {"type": "reasoning", "id": "rs_1", "summary": [], "content": [{"type": "reasoning_text", "text": "The user wants weather."}]},
    {"type": "message", "id": "msg_2", "status": "completed", "role": "assistant",
     "content": [{"type": "output_text", "text": "Checking.", "annotations": []}]},
    {"type": "function_call", "id": "fc_3", "call_id": "call_4", "name": "weather",
     "arguments": "{\"city\":\"Paris\",\"days\":3}", "status": "completed"}])")));
  EXPECT_EQ(v.find("status")->str(), "completed");
  EXPECT_EQ(v.find("usage")->find("output_tokens_details")->find("reasoning_tokens")->number(), 6);
  EXPECT_EQ(json::DumpPython(*events.back().find("response")), json::DumpPython(v));
  // Length-limited: incomplete, with its reason.
  OutputBuilder c(req, "resp_2", 1700000000, Counter());
  c.Add(text::ParseAssistantText("partial thought", ParserOptions(req)), nullptr);
  c.Finish(FinishReason::kLength, Usage{}, nullptr);
  EXPECT_EQ(c.Final().find("status")->str(), "incomplete");
  EXPECT_EQ(c.Final().find("incomplete_details")->find("reason")->str(), "max_output_tokens");
}

// A call the parser rejects is not dropped: its bytes are content.
TEST(OpenAiResponse, MalformedCallStaysVisibleAsContent) {
  const std::string bad = "<tool_call>\n<function=weather>\n<parameter=city>Paris</oops>\n</tool_call>";
  for (const Api api : {Api::kChat, Api::kResponses}) {
    const auto req = Request(api);
    OutputBuilder b(req, "id", 1, Counter());
    b.Add(text::ParseAssistantText("r\n</think>\n\nSee: " + bad, ParserOptions(req)), nullptr);
    b.Finish(FinishReason::kStop, Usage{}, nullptr);
    const Value v = b.Final();
    const std::string content = api == Api::kChat
        ? v.find("choices")->items()[0].find("message")->find("content")->str()
        : v.find("output")->items().back().find("content")->items()[0].find("text")->str();
    EXPECT_NE(content.find("</oops>"), std::string::npos) << content;
    EXPECT_FALSE(b.has_tool_calls());
  }
}

TEST(OpenAiResponse, CompletionsAreRawText) {
  const auto req = Request(Api::kCompletions);
  OutputBuilder b(req, "cmpl-1", 1, Counter());
  std::vector<Value> chunks;
  b.AddText("Once ", &chunks);
  b.AddText("upon <tool_call>", &chunks);
  b.Finish(FinishReason::kLength, Usage{2, 0, 3, 0}, &chunks);
  ASSERT_EQ(chunks.size(), 3u);
  EXPECT_EQ(chunks[1].find("choices")->items()[0].find("text")->str(), "upon <tool_call>");
  EXPECT_EQ(chunks[2].find("choices")->items()[0].find("finish_reason")->str(), "length");
  const Value v = b.Final();
  EXPECT_EQ(v.find("object")->str(), "text_completion");
  EXPECT_EQ(v.find("choices")->items()[0].find("text")->str(), "Once upon <tool_call>");
}

namespace {

// A generation: reasoning, then content and calls interleaved, possibly a
// malformed call, possibly cut off.
std::string DrawGeneration(hegel::TestCase& tc) {
  const auto pick = [&](std::vector<std::string> v) { return tc.draw(gs::sampled_from(v)); };
  std::string g = pick({"", "think", "a\nb", "Let me see: <b>x</b>"});
  if (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 5})) == 0) return g;  // cut inside reasoning
  g += "\n</think>\n\n";
  const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4}));
  for (int i = 0; i < n; ++i) {
    switch (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}))) {
      case 0: g += pick({"Hello", "é ü", "a < b", "x\n\ny", "{\"k\": 1}"}); break;
      case 1:
        g += "\n<tool_call>\n<function=weather>\n<parameter=city>\n" + pick({"Paris", "New York", "1"}) +
             "\n</parameter>\n</function>\n</tool_call>\n";
        break;
      case 2: g += "\n<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"days\": 2}}\n</tool_call>\n"; break;
      default: g += "<tool_call>\n<function=weather>\n<parameter=city>Paris</oops>\n</tool_call>"; break;  // malformed
    }
  }
  return g;
}

// Feeds `text` to a parser in drawn chunks.
std::vector<std::vector<text::StreamEvent>> Chunked(hegel::TestCase& tc, const std::string& text,
                                                    const text::ToolCallParserOptions& opt) {
  text::ToolCallStreamParser p(opt);
  std::vector<std::vector<text::StreamEvent>> out;
  std::size_t i = 0;
  while (i < text.size()) {
    const auto n = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 7}));
    out.emplace_back();
    p.Push(std::string_view(text).substr(i, n), &out.back());
    i += n;
  }
  out.emplace_back();
  p.Finish(&out.back());
  return out;
}

}  // namespace

TEST(OpenAiResponse, StreamsConcatenateToTheFinalObject) {
  hegel::test([](hegel::TestCase& tc) {
    const std::string gen = DrawGeneration(tc);
    const auto reason = tc.draw(gs::booleans()) ? FinishReason::kStop : FinishReason::kLength;
    for (const Api api : {Api::kChat, Api::kResponses}) {
      const auto req = Request(api);
      const auto chunks = Chunked(tc, gen, ParserOptions(req));
      OutputBuilder streamed(req, "id", 7, Counter()), whole(req, "id", 7, Counter());
      std::vector<Value> events;
      for (const auto& c : chunks) {
        streamed.Add(c, &events);
        whole.Add(c, nullptr);
      }
      streamed.Finish(reason, Usage{1, 0, 2, 0}, &events);
      whole.Finish(reason, Usage{1, 0, 2, 0}, nullptr);
      const Value final = whole.Final();
      ASSERT_EQ(json::DumpPython(streamed.Final()), json::DumpPython(final));
      if (api == Api::kChat) {
        std::string content, reasoning;
        Value calls = Value::MakeArray();
        std::string finish;
        for (const auto& e : events) {
          const auto& ch = e.find("choices")->items();
          if (ch.empty()) continue;  // usage chunk
          const Value& d = *ch[0].find("delta");
          if (const Value* c = d.find("content")) content += c->str();
          if (const Value* r = d.find("reasoning_content")) reasoning += r->str();
          if (const Value* t = d.find("tool_calls")) {
            Value call = t->items()[0];
            ASSERT_EQ(call.find("index")->number(), static_cast<double>(calls.items().size()));
            Value plain = Value::MakeObject();
            for (const auto& [k, v] : call.members())
              if (k != "index") plain.set(k, v);
            calls.push(plain);
          }
          if (const Value* f = ch[0].find("finish_reason"); f && f->is(Value::Kind::kString)) finish = f->str();
        }
        const Value& m = *final.find("choices")->items()[0].find("message");
        ASSERT_EQ(content, m.find("content")->is(Value::Kind::kNull) ? "" : m.find("content")->str());
        ASSERT_EQ(reasoning, m.find("reasoning_content") ? m.find("reasoning_content")->str() : "");
        ASSERT_EQ(json::DumpPython(calls), m.find("tool_calls") ? json::DumpPython(*m.find("tool_calls")) : "[]");
        ASSERT_EQ(finish, final.find("choices")->items()[0].find("finish_reason")->str());
        ASSERT_TRUE(events.back().find("usage"));
      } else {
        std::map<std::uint64_t, std::string> text;  // output_index -> streamed text or arguments
        std::vector<Value> done;
        std::int64_t open = -1;
        for (std::size_t i = 0; i < events.size(); ++i) {
          const Value& e = events[i];
          ASSERT_EQ(e.find("sequence_number")->number(), static_cast<double>(i));
          const std::string type = e.find("type")->str();
          if (type == "response.output_item.added") {
            ASSERT_EQ(open, -1) << "an item was added while another is open";
            open = static_cast<std::int64_t>(e.find("output_index")->number());
            ASSERT_EQ(open, static_cast<std::int64_t>(done.size()));
          } else if (type.ends_with(".delta")) {
            ASSERT_EQ(static_cast<std::int64_t>(e.find("output_index")->number()), open);
            text[static_cast<std::uint64_t>(open)] += e.find("delta")->str();
          } else if (type == "response.output_item.done") {
            ASSERT_EQ(static_cast<std::int64_t>(e.find("output_index")->number()), open);
            done.push_back(*e.find("item"));
            open = -1;
          }
        }
        ASSERT_EQ(open, -1);
        const auto& out = final.find("output")->items();
        ASSERT_EQ(out.size(), done.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
          ASSERT_EQ(json::DumpPython(out[i]), json::DumpPython(done[i]));
          const std::string type = out[i].find("type")->str();
          const std::string whole_text = type == "function_call" ? out[i].find("arguments")->str()
                                                                 : out[i].find("content")->items()[0].find("text")->str();
          ASSERT_EQ(text[i], whole_text) << "item " << i;
        }
        ASSERT_EQ(events.back().find("type")->str(),
                  reason == FinishReason::kStop ? "response.completed" : "response.incomplete");
      }
    }
  }, {.test_cases = 300});
}

// Stateful conversations depend on this: a Responses output sent back as
// input items is the assistant turn Chat reports for the same generation.
TEST(OpenAiResponse, ResponsesOutputRoundTripsAsInputItems) {
  hegel::test([](hegel::TestCase& tc) {
    const std::string gen = DrawGeneration(tc);
    const auto rreq = Request(Api::kResponses), creq = Request(Api::kChat);
    const auto events = text::ParseAssistantText(gen, ParserOptions(rreq));
    OutputBuilder r(rreq, "resp", 1, Counter()), c(creq, "chat", 1, Counter());
    r.Add(events, nullptr);
    c.Add(events, nullptr);
    r.Finish(FinishReason::kStop, Usage{}, nullptr);
    c.Finish(FinishReason::kStop, Usage{}, nullptr);
    const Value rfinal = r.Final(), cfinal = c.Final();  // the loops below reference into these
    Value input = J(R"([{"role": "user", "content": "hi"}])");
    for (const auto& item : rfinal.find("output")->items()) input.push(item);
    Value body = Value::MakeObject();
    body.set("input", input);
    RequestError e;
    const auto parsed = ParseResponsesRequest(body, RequestDefaults{}, &e);
    ASSERT_TRUE(parsed) << e.param << ": " << e.message;
    const auto& msgs = parsed->messages.items();
    const Value& chat = *cfinal.find("choices")->items()[0].find("message");
    if (msgs.size() == 1) {  // nothing generated
      ASSERT_TRUE(chat.find("content")->str().empty() && !chat.find("reasoning_content") && !chat.find("tool_calls"));
      return;
    }
    // Content after a call starts a new assistant turn on the Responses side
    // (items keep their order); Chat concatenates a generation's content.
    // Joined across the assistant turns, both say the same.
    std::string reasoning, content;
    std::vector<const Value*> calls;
    for (std::size_t i = 1; i < msgs.size(); ++i) {
      const Value& a = msgs[i];
      ASSERT_EQ(a.find("role")->str(), "assistant");
      if (const Value* rc = a.find("reasoning_content")) reasoning += rc->str();
      const Value* c = a.find("content");
      content += c->is(Value::Kind::kString) ? c->str() : c->items()[0].find("text")->str();
      if (const Value* t = a.find("tool_calls"))
        for (const auto& call : t->items()) calls.push_back(&call);
    }
    ASSERT_EQ(reasoning, chat.find("reasoning_content") ? chat.find("reasoning_content")->str() : "");
    ASSERT_EQ(content, chat.find("content")->is(Value::Kind::kNull) ? "" : chat.find("content")->str());
    const Value* want = chat.find("tool_calls");
    ASSERT_EQ(calls.size(), want ? want->items().size() : 0u);
    for (std::size_t i = 0; i < calls.size(); ++i) {
      const Value& fn = *calls[i]->find("function");
      const Value& wf = *want->items()[i].find("function");
      ASSERT_EQ(fn.find("name")->str(), wf.find("name")->str());
      ASSERT_EQ(CompactJson(*fn.find("arguments")), wf.find("arguments")->str());
    }
  }, {.test_cases = 300});
}
