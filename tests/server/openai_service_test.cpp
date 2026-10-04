// The OpenAI service end to end over loopback HTTP, with a scripted
// generator in place of the engine:
//   - chat (whole and streamed), completions, responses: the generator sees
//     the template's rendering of the request, and the reply carries the
//     parsed reasoning, content and tool calls;
//   - stop sequences end the output wherever the generator's pieces split
//     them (property over chunkings), and a client leaving mid-stream stops
//     generation;
//   - stored responses: previous_response_id continues the conversation
//     (instructions do not carry over), GET returns the stored object, an
//     unknown id is a 404.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <hegel/gtest.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/json.hpp"
#include "server/http.hpp"
#include "server/openai_response.hpp"
#include "server/openai_service.hpp"
#include "text/chat_template.hpp"

namespace gs = hegel::generators;
using namespace halo;
using namespace halo::server;
using json::Value;

namespace {

Value J(std::string_view text) {
  std::string err;
  auto v = json::Parse(text, &err);
  if (!v) throw std::runtime_error("bad JSON: " + err + " in " + std::string(text.substr(0, 200)));
  return *v;
}

/// Emits a script in fixed-size pieces, one "token" each.
class ScriptedGenerator : public TextGenerator {
 public:
  Result Generate(const Params& p, const std::function<bool(std::string_view)>& on_text) override {
    std::string script;
    {
      std::lock_guard<std::mutex> lk(mu);
      prompts.push_back(p.prompt);
      script = scripts.empty() ? "ok" : scripts.front();
      if (!scripts.empty()) scripts.erase(scripts.begin());
    }
    Result r;
    r.prompt_tokens = p.prompt.size();
    for (std::size_t i = 0; i < script.size(); i += piece) {
      if (r.completion_tokens == p.max_tokens) {
        r.length_limited = true;
        break;
      }
      ++r.completion_tokens;
      if (!on_text(std::string_view(script).substr(i, piece))) {
        stopped_early = true;
        break;
      }
      if (delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
    return r;
  }
  std::mutex mu;
  std::vector<std::string> scripts, prompts;
  std::size_t piece = 3;
  int delay_ms = 0;
  std::atomic<bool> stopped_early{false};
};

struct Rig {
  Rig() : server({}) {
    ServiceOptions o;
    int n = 0;
    o.new_id = [n]() mutable { return std::to_string(++n); };
    o.now_s = [] { return std::int64_t{1700000000}; };
    service = std::make_unique<OpenAiService>(gen, o);
    service->Register(server);
    std::string err;
    if (!server.Start(&err)) throw std::runtime_error(err);
  }
  ScriptedGenerator gen;
  HttpServer server;
  std::unique_ptr<OpenAiService> service;
};

/// One request on a fresh connection; returns (status, body), dechunked.
std::pair<int, std::string> Post(int port, const std::string& method, const std::string& path, const std::string& body,
                                 bool read_all = true) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) throw std::runtime_error("connect");
  const std::string req = method + " " + path + " HTTP/1.1\r\nConnection: close\r\nContent-Length: " +
                          std::to_string(body.size()) + "\r\n\r\n" + body;
  ::send(fd, req.data(), req.size(), MSG_NOSIGNAL);
  std::string raw;
  char b[8192];
  for (ssize_t n; (n = ::recv(fd, b, sizeof b, 0)) > 0;) {
    raw.append(b, static_cast<std::size_t>(n));
    if (!read_all) break;
  }
  ::close(fd);
  const auto end = raw.find("\r\n\r\n");
  const int status = std::stoi(raw.substr(9, 3));
  std::string head = raw.substr(0, end), rest = raw.substr(end + 4), out;
  if (head.find("Transfer-Encoding: chunked") == std::string::npos) return {status, rest};
  for (std::size_t i = 0;;) {
    const auto eol = rest.find("\r\n", i);
    if (eol == std::string::npos) break;
    const std::size_t n = std::stoul(rest.substr(i, eol - i), nullptr, 16);
    if (n == 0) break;
    out += rest.substr(eol + 2, n);
    i = eol + 2 + n + 2;
  }
  return {status, out};
}

/// The JSON payloads of an SSE body (skipping [DONE]).
std::vector<Value> Events(const std::string& sse, bool* done) {
  std::vector<Value> out;
  *done = false;
  std::size_t i = 0;
  while ((i = sse.find("data: ", i)) != std::string::npos) {
    const auto end = sse.find("\n\n", i);
    const std::string data = sse.substr(i + 6, end - i - 6);
    if (data == "[DONE]") *done = true;
    else out.push_back(J(data));
    i = end;
  }
  return out;
}

std::string Render(const char* messages, const char* tools = nullptr) {
  text::ChatTemplateOptions o;
  o.add_generation_prompt = true;
  std::string err;
  const Value m = J(messages);
  const Value t = tools ? J(tools) : Value();
  auto s = text::RenderChat(text::ChatTemplateKind::kFlashNext, m, tools ? &t : nullptr, o, &err);
  if (!s) throw std::runtime_error(err);
  return *s;
}

}  // namespace

TEST(OpenAiService, ChatWholeAndStreamed) {
  Rig rig;
  const std::string answer = "Thinking hard.\n</think>\n\nThe answer is 4.";
  rig.gen.scripts = {answer, answer};
  const std::string body = R"({"model": "m", "messages": [{"role": "user", "content": "2+2?"}]})";
  auto [status, text] = Post(rig.server.port(), "POST", "/v1/chat/completions", body);
  ASSERT_EQ(status, 200) << text;
  const Value v = J(text);
  const Value& msg = *v.find("choices")->items()[0].find("message");
  EXPECT_EQ(msg.find("content")->str(), "The answer is 4.");
  EXPECT_EQ(msg.find("reasoning_content")->str(), "Thinking hard.");
  EXPECT_EQ(v.find("id")->str(), "chatcmpl-1");
  EXPECT_EQ(rig.gen.prompts[0], Render(R"([{"role": "user", "content": "2+2?"}])"));
  // Streamed: deltas add up to the same, then [DONE].
  auto [s2, sse] = Post(rig.server.port(), "POST", "/v1/chat/completions",
                        R"({"messages": [{"role": "user", "content": "2+2?"}], "stream": true})");
  ASSERT_EQ(s2, 200);
  bool done = false;
  std::string content;
  for (const auto& e : Events(sse, &done))
    if (const Value* c = e.find("choices")->items()[0].find("delta")->find("content")) content += c->str();
  EXPECT_TRUE(done);
  EXPECT_EQ(content, "The answer is 4.");
  // Errors come back in OpenAI's shape.
  auto [s3, err] = Post(rig.server.port(), "POST", "/v1/chat/completions", R"({"messages": []})");
  EXPECT_EQ(s3, 400);
  EXPECT_EQ(J(err).find("error")->find("param")->str(), "messages");
  EXPECT_EQ(Post(rig.server.port(), "POST", "/v1/chat/completions", "{not json").first, 400);
  EXPECT_EQ(Post(rig.server.port(), "GET", "/v1/models", "").first, 200);
}

TEST(OpenAiService, StopSequencesAndLengthLimits) {
  Rig rig;
  rig.gen.piece = 2;  // "STOP" arrives split across pieces
  rig.gen.scripts = {"Hello STOP world", "abcdefghij"};
  auto [s1, t1] = Post(rig.server.port(), "POST", "/v1/completions", R"({"prompt": "x", "stop": ["STOP"], "max_tokens": 50})");
  ASSERT_EQ(s1, 200) << t1;
  const Value v1 = J(t1);
  EXPECT_EQ(v1.find("choices")->items()[0].find("text")->str(), "Hello ");
  EXPECT_EQ(v1.find("choices")->items()[0].find("finish_reason")->str(), "stop");
  EXPECT_EQ(rig.gen.prompts[0], "x");  // completions: no template
  auto [s2, t2] = Post(rig.server.port(), "POST", "/v1/completions", R"({"prompt": "x", "max_tokens": 3})");
  const Value v2 = J(t2);
  EXPECT_EQ(v2.find("choices")->items()[0].find("text")->str(), "abcdef");
  EXPECT_EQ(v2.find("choices")->items()[0].find("finish_reason")->str(), "length");
}

TEST(OpenAiService, StoredResponsesContinueTheConversation) {
  Rig rig;
  rig.gen.scripts = {"\n</think>\n\n<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
                     "Plan.\n</think>\n\nIt is 18C."};
  const char* tools = R"([{"type": "function", "name": "weather", "parameters": {"type": "object",
                          "properties": {"city": {"type": "string"}}}}])";
  auto [s1, t1] = Post(rig.server.port(), "POST", "/v1/responses",
                       std::string(R"({"instructions": "Be terse.", "input": "Weather in Paris?", "tools": )") + tools + "}");
  ASSERT_EQ(s1, 200) << t1;
  const Value r1 = J(t1);
  ASSERT_EQ(r1.find("id")->str(), "resp_1");
  const Value& call = r1.find("output")->items().back();
  ASSERT_EQ(call.find("type")->str(), "function_call");
  const std::string call_id = call.find("call_id")->str();
  // Continue: the tool's result, new instructions (the old ones do not carry over).
  auto [s2, t2] = Post(rig.server.port(), "POST", "/v1/responses",
                       std::string(R"({"previous_response_id": "resp_1", "instructions": "Use Celsius.", "tools": )") + tools +
                           R"(, "input": [{"type": "function_call_output", "call_id": ")" + call_id +
                           R"(", "output": "18"}]})");
  ASSERT_EQ(s2, 200) << t2;
  const std::string chat_tools = R"([{"type": "function", "function": {"name": "weather", "parameters": {"type": "object",
                                    "properties": {"city": {"type": "string"}}}}}])";
  const std::string want = Render((R"([{"role": "system", "content": "Use Celsius."},
      {"role": "user", "content": "Weather in Paris?"},
      {"role": "assistant", "content": "", "tool_calls": [{"id": ")" + call_id +
                                   R"(", "type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}]},
      {"role": "tool", "content": "18", "tool_call_id": ")" + call_id + R"("}])").c_str(),
                                  chat_tools.c_str());
  ASSERT_EQ(rig.gen.prompts.size(), 2u);
  EXPECT_EQ(rig.gen.prompts[1], want);
  const Value r2 = J(t2);
  EXPECT_EQ(r2.find("output")->items().back().find("content")->items()[0].find("text")->str(), "It is 18C.");
  EXPECT_EQ(r2.find("previous_response_id")->str(), "resp_1");
  // Stored objects come back; unknown ids do not.
  auto [s3, t3] = Post(rig.server.port(), "GET", "/v1/responses/resp_1", "");
  EXPECT_EQ(s3, 200);
  EXPECT_EQ(J(t3).find("id")->str(), "resp_1");
  EXPECT_EQ(Post(rig.server.port(), "GET", "/v1/responses/resp_9", "").first, 404);
  auto [s4, t4] = Post(rig.server.port(), "POST", "/v1/responses", R"({"previous_response_id": "resp_9", "input": "x"})");
  EXPECT_EQ(s4, 404);
  EXPECT_EQ(J(t4).find("error")->find("param")->str(), "previous_response_id");
  // store: false is not kept.
  rig.gen.scripts = {"\n</think>\n\nx"};
  auto [s5, t5] = Post(rig.server.port(), "POST", "/v1/responses", R"({"input": "x", "store": false})");
  ASSERT_EQ(s5, 200);
  EXPECT_EQ(Post(rig.server.port(), "GET", "/v1/responses/" + J(t5).find("id")->str(), "").first, 404);
}

TEST(OpenAiService, ResponsesStreamEndsWithCompletedAndADepartingClientStopsGeneration) {
  Rig rig;
  rig.gen.scripts = {"r\n</think>\n\nHello there"};
  auto [s, sse] = Post(rig.server.port(), "POST", "/v1/responses", R"({"input": "hi", "stream": true})");
  ASSERT_EQ(s, 200);
  EXPECT_NE(sse.find("event: response.created\n"), std::string::npos);
  bool done = false;
  const auto events = Events(sse, &done);
  ASSERT_FALSE(events.empty());
  EXPECT_EQ(events.back().find("type")->str(), "response.completed");
  EXPECT_FALSE(done);  // Responses streams end with response.completed, not [DONE]
  // A client that leaves: the generator is told to stop.
  rig.gen.scripts = {std::string(30000, 'a')};
  rig.gen.piece = 1;
  rig.gen.delay_ms = 1;
  (void)Post(rig.server.port(), "POST", "/v1/completions", R"({"prompt": "x", "stream": true, "max_tokens": 30000})",
             /*read_all=*/false);
  for (int i = 0; i < 500 && !rig.gen.stopped_early.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_TRUE(rig.gen.stopped_early.load());
}

// Output up to where the first stop sequence completes (byte at a time),
// however the text arrives. (Found by this property: "earliest occurrence
// in the whole text" is not a streaming semantics; with stops "a" and "bab",
// "baba" would give "" whole and "b" streamed.)
TEST(OpenAiService, StopMatcherIsChunkingIndependent) {
  hegel::test([](hegel::TestCase& tc) {
    const auto alphabet = [&] { return tc.draw(gs::sampled_from<std::string>({"a", "b", "ab", "ba", "\n", "</"})); };
    std::string text;
    const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 20}));
    for (int i = 0; i < n; ++i) text += alphabet();
    std::vector<std::string> stops;
    const auto ns = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}));
    for (int i = 0; i < ns; ++i) {
      std::string s;
      const auto len = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 3}));
      for (int k = 0; k < len; ++k) s += alphabet();
      stops.push_back(s);
    }
    // Byte-at-a-time: the first end position where any stop completes.
    std::size_t cut = text.size();
    bool hit = false;
    for (std::size_t end = 1; end <= text.size() && !hit; ++end)
      for (const auto& s : stops)
        if (s.size() <= end && text.compare(end - s.size(), s.size(), s) == 0) {
          cut = hit ? std::min(cut, end - s.size()) : end - s.size();
          hit = true;
        }
    StopMatcher m(stops);
    std::string out;
    for (std::size_t i = 0; i < text.size();) {
      const auto k = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 4}));
      out += m.Push(std::string_view(text).substr(i, k));
      i += k;
    }
    out += m.Flush();
    ASSERT_EQ(out, text.substr(0, cut));
    ASSERT_EQ(m.stopped(), hit);
  }, {.test_cases = 1000});
}
