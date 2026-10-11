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
#include <cctype>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <chrono>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "core/json.hpp"
#include "engine/mtp_depth.hpp"
#include "server/decode_loop.hpp"
#include "server/http.hpp"
#include "server/openai_response.hpp"
#include "server/openai_service.hpp"
#include "server/scheduled_generator.hpp"
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
      variants.push_back(p.variant);
      script = scripts.empty() ? "ok" : scripts.front();
      if (!scripts.empty()) scripts.erase(scripts.begin());
    }
    Result r;
    r.prompt_tokens = p.prompt.size();
    for (std::size_t i = 0; i < script.size(); i += piece) {
      if (p.cancelled && p.cancelled()) {  // as the engine's generator polls before each step
        r.cancelled = stopped_early = true;
        break;
      }
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
  std::vector<std::string> scripts, prompts, variants;
  std::size_t piece = 3;
  int delay_ms = 0;
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
  explicit Rig(std::vector<std::string> variants = {}) : server({}) {
    ServiceOptions o;
    if (!variants.empty()) o.variants = [variants] { return variants; };
    o.observer = &observed;
    int n = 0;
    o.new_id = [n]() mutable { return std::to_string(++n); };
    o.now_s = [] { return std::int64_t{1700000000}; };
    service = std::make_unique<OpenAiService>(gen, o);
    service->Register(server);
    std::string err;
    if (!server.Start(&err)) throw std::runtime_error(err);
  }
  // Workers may still be streaming through the service: stop them first
  // (members are destroyed in reverse order, the service before the server).
  ~Rig() { server.Stop(); }
  ScriptedGenerator gen;
  Recording observed;
  HttpServer server;
  std::unique_ptr<OpenAiService> service;
};

/// One request on a fresh connection; returns (status, body), dechunked.
std::pair<int, std::string> Post(int port, const std::string& method, const std::string& path, const std::string& body,
                                 bool read_all = true, const std::string& headers = "") {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) throw std::runtime_error("connect");
  const std::string req = method + " " + path + " HTTP/1.1\r\nConnection: close\r\n" + headers +
                          "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
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
  EXPECT_EQ(v.find("model")->str(), "m");
  EXPECT_EQ(rig.gen.prompts[0], Render(R"([{"role": "user", "content": "2+2?"}])"));
  // Streamed: deltas add up to the same, then [DONE].
  auto [s2, sse] = Post(rig.server.port(), "POST", "/v1/chat/completions",
                        R"({"messages": [{"role": "user", "content": "2+2?"}], "stream": true})");
  ASSERT_EQ(s2, 200);
  bool done = false;
  std::string content;
  EXPECT_EQ(Events(sse, &done).front().find("model")->str(), "qwen3.8-flash-next");  // the service's id when omitted
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

// Every generation is reported once, with the harness's trace and session
// ids, usage, timing and the tool-call ids the reply carries.
TEST(OpenAiService, ReportsEachGenerationToTheObserver) {
  Rig rig;
  const std::string script =
      "\n</think>\n\n<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
  rig.gen.scripts = {script};
  auto [status, text] = Post(rig.server.port(), "POST", "/v1/chat/completions",
                             R"({"model": "m", "user": "u7", "temperature": 0.3, "max_tokens": 77,
                                 "messages": [{"role": "user", "content": "Weather?"}],
                                 "tools": [{"type": "function", "function": {"name": "weather"}}]})",
                             true, "X-Halo-Trace-Id: tr-1\r\nX-Halo-Session-Id: sess-9\r\n");
  ASSERT_EQ(status, 200) << text;
  ASSERT_EQ(rig.observed.records.size(), 1u);
  const auto& r = rig.observed.records[0];
  const auto reply = J(text);
  EXPECT_EQ(r.api, "chat");
  EXPECT_EQ(r.request_id, reply.find("id")->str());
  EXPECT_EQ(r.trace_id, "tr-1");
  EXPECT_EQ(r.session_id, "sess-9");
  EXPECT_EQ(r.user, "u7");
  EXPECT_EQ(r.model, "m");
  EXPECT_EQ(r.finish_reason, "tool_calls");
  EXPECT_FLOAT_EQ(r.sampling.temperature, 0.3f);
  EXPECT_EQ(r.max_tokens, 77u);
  EXPECT_EQ(r.prompt_tokens, rig.gen.prompts[0].size());  // the scripted generator counts bytes
  EXPECT_GT(r.completion_tokens, 0u);
  EXPECT_LE(r.start_ns, r.first_text_ns);
  EXPECT_LE(r.first_text_ns, r.end_ns);
  EXPECT_EQ(r.gaps_us.size() + 1, r.completion_tokens);  // one piece per scripted token
  EXPECT_EQ(r.input, rig.gen.prompts[0]);
  EXPECT_EQ(r.output, script);
  const auto& call = reply.find("choices")->items()[0].find("message")->find("tool_calls")->items()[0];
  ASSERT_EQ(r.tool_call_ids.size(), 1u);
  EXPECT_EQ(r.tool_call_ids[0], call.find("id")->str());
  EXPECT_EQ(r.tool_names[0], "weather");
}


TEST(OpenAiService, VariantsAreSelectedPerRequestAndUnknownOnesRefused) {
  Rig rig({"sentiment-ab", "tools-v2"});
  const auto chat = [&](const std::string& extra) {
    return Post(rig.server.port(), "POST", "/v1/chat/completions",
                R"({"messages": [{"role": "user", "content": "hi"}])" + extra + "}");
  };
  // The "variant" field, the "base:variant" alias (echoed as sent), and both agreeing.
  auto [s1, t1] = chat(R"(, "variant": "sentiment-ab")");
  ASSERT_EQ(s1, 200) << t1;
  auto [s2, t2] = chat(R"(, "model": "qwen3.8-flash-next:tools-v2")");
  ASSERT_EQ(s2, 200) << t2;
  EXPECT_EQ(J(t2).find("model")->str(), "qwen3.8-flash-next:tools-v2");
  auto [s3, t3] = chat(R"(, "model": "m:tools-v2", "variant": "tools-v2")");
  ASSERT_EQ(s3, 200) << t3;
  auto [s4, t4] = chat("");
  ASSERT_EQ(s4, 200) << t4;
  EXPECT_EQ(rig.gen.variants, (std::vector<std::string>{"sentiment-ab", "tools-v2", "tools-v2", ""}));
  // The resolved variant reaches the generation record (halo-scope's variant_id);
  // the X-Halo-Variant-Id header labels only requests without one.
  {
    std::lock_guard<std::mutex> lk(rig.observed.mu);
    ASSERT_EQ(rig.observed.records.size(), 4u);
    EXPECT_EQ(rig.observed.records[0].variant_id, "sentiment-ab");
    EXPECT_EQ(rig.observed.records[1].variant_id, "tools-v2");
    EXPECT_EQ(rig.observed.records[3].variant_id, "");
  }
  const auto with_header = [&](const std::string& extra) {
    return Post(rig.server.port(), "POST", "/v1/chat/completions",
                R"({"messages": [{"role": "user", "content": "hi"}])" + extra + "}", true,
                "X-Halo-Variant-Id: harness-label\r\n");
  };
  ASSERT_EQ(with_header(R"(, "variant": "tools-v2")").first, 200);
  ASSERT_EQ(with_header("").first, 200);
  {
    std::lock_guard<std::mutex> lk(rig.observed.mu);
    ASSERT_EQ(rig.observed.records.size(), 6u);
    EXPECT_EQ(rig.observed.records[4].variant_id, "tools-v2");
    EXPECT_EQ(rig.observed.records[5].variant_id, "harness-label");
  }
  rig.gen.variants.resize(4);
  // Unknown, conflicting and malformed variants never reach the generator.
  auto [s5, t5] = chat(R"(, "variant": "missing")");
  EXPECT_EQ(s5, 404) << t5;
  EXPECT_EQ(J(t5).find("error")->find("param")->str(), "model");
  EXPECT_EQ(chat(R"(, "model": "m:a", "variant": "b")").first, 400);
  EXPECT_EQ(chat(R"(, "variant": "bad/name")").first, 400);
  EXPECT_EQ(chat(R"(, "model": "m:")").first, 400);
  auto [s6, t6] = Post(rig.server.port(), "POST", "/v1/completions", R"({"prompt": "x", "model": "m:missing"})");
  EXPECT_EQ(s6, 404) << t6;
  EXPECT_EQ(rig.gen.variants.size(), 4u);
  // /v1/models lists the base and "<id>:<variant>".
  const Value models = J(Post(rig.server.port(), "GET", "/v1/models", "").second);
  std::vector<std::string> ids;
  for (const auto& m : models.find("data")->items()) ids.push_back(m.find("id")->str());
  EXPECT_EQ(ids, (std::vector<std::string>{"qwen3.8-flash-next", "qwen3.8-flash-next:sentiment-ab",
                                           "qwen3.8-flash-next:tools-v2"}));
  // Without a variant catalog every variant is refused.
  Rig plain;
  EXPECT_EQ(Post(plain.server.port(), "POST", "/v1/chat/completions",
                 R"({"messages": [{"role": "user", "content": "hi"}], "variant": "sentiment-ab"})")
                .first,
            404);
}

namespace {

/// Sends a request and reads the reply until `until` appears in it (or the
/// connection ends); returns the bytes read and leaves the socket open in *fd.
std::string SendAndReadUntil(int port, const std::string& path, const std::string& body, const std::string& until,
                             int* fd) {
  *fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (::connect(*fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) throw std::runtime_error("connect");
  const std::string req = "POST " + path + " HTTP/1.1\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
  ::send(*fd, req.data(), req.size(), MSG_NOSIGNAL);
  std::string raw;
  char b[4096];
  while (!until.empty() && raw.find(until) == std::string::npos) {
    const ssize_t n = ::recv(*fd, b, sizeof b, 0);
    if (n <= 0) break;
    raw.append(b, static_cast<std::size_t>(n));
  }
  return raw;
}

/// Waits until the observer holds `n` records; returns a copy of record `n - 1`.
std::optional<GenerationRecord> AwaitRecord(Recording& r, std::size_t n, int ms = 10000) {
  for (int i = 0; i < ms / 5; ++i) {
    {
      std::lock_guard<std::mutex> lk(r.mu);
      if (r.records.size() >= n) return r.records[n - 1];
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return std::nullopt;
}

}  // namespace

// A client that leaves a /v1/responses stream partway: the generation is not a completed response (audit F1). The
// observer records it as cancelled (never "stop": a truncated generation must not look like a finished one), and
// the store does not keep the truncated turn as a completed response.
TEST(OpenAiService, ADepartedStreamIsRecordedAndStoredAsCancelled) {
  Rig rig;
  rig.gen.scripts = {"r\n</think>\n\n" + std::string(3000, 'a')};
  rig.gen.piece = 1;
  rig.gen.delay_ms = 1;
  int fd = -1;
  const std::string got = SendAndReadUntil(rig.server.port(), "/v1/responses", R"({"input": "hi", "stream": true})",
                                           "response.output_text.delta", &fd);
  ASSERT_NE(got.find("response.created"), std::string::npos);
  ::close(fd);
  const auto rec = AwaitRecord(rig.observed, 1);
  ASSERT_TRUE(rec);
  EXPECT_EQ(rec->finish_reason, "cancelled");
  EXPECT_LT(rec->completion_tokens, 3000u);
  // GET answers with the cancelled object (the client holds its id from response.created); its turn is not kept,
  // so continuing from it is refused rather than building on output the client never got in full.
  auto [status, body] = Post(rig.server.port(), "GET", "/v1/responses/resp_1", "");
  ASSERT_EQ(status, 200) << body;
  EXPECT_EQ(J(body).find("status")->str(), "cancelled") << body;
  rig.gen.scripts = {"\n</think>\n\nok"};
  rig.gen.delay_ms = 0;
  auto [s2, t2] = Post(rig.server.port(), "POST", "/v1/responses", R"({"previous_response_id": "resp_1", "input": "go on"})");
  EXPECT_EQ(s2, 400) << t2;
  EXPECT_EQ(J(t2).find("error")->find("param")->str(), "previous_response_id");
  EXPECT_EQ(rig.gen.prompts.size(), 1u) << "a refused continuation reached the generator";
}

// A non-streaming request whose client leaves: the generation stops (audit F5); before the fix it ran every token.
TEST(OpenAiService, ADepartedNonStreamingClientStopsGeneration) {
  Rig rig;
  rig.gen.scripts = {std::string(3000, 'a')};
  rig.gen.piece = 1;
  rig.gen.delay_ms = 1;
  int fd = -1;
  (void)SendAndReadUntil(rig.server.port(), "/v1/completions", R"({"prompt": "x", "max_tokens": 5000})", "", &fd);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ::close(fd);
  const auto rec = AwaitRecord(rig.observed, 1);
  ASSERT_TRUE(rec);
  EXPECT_EQ(rec->finish_reason, "cancelled");
  EXPECT_LT(rec->completion_tokens, 1500u) << "the generation ran on after its client left";
}

// ---- The serving path as a state machine: clients that complete, leave, stall and retry ---------------------------

namespace {

/// The engine's generator as a model: one generation at a time (a timed lock, polling Params::cancelled while
/// queued, as FlashNextGenerator does), a prefill of kChunks units with a poll between them, then decode steps with a
/// poll before each; every chunk or step is one unit of work. A run can be held before a given unit, so that a
/// client's departure lands at a definite point (queued, prefill or decode); a held run keeps polling.
/// The request's own marker ("m<n>x"): the last one in its prompt (a continuation's prompt holds earlier ones).
std::string LastMarker(const std::string& prompt) {
  std::string last;
  for (std::size_t i = prompt.find('m'); i != std::string::npos; i = prompt.find('m', i + 1)) {
    std::size_t j = i + 1;
    while (j < prompt.size() && std::isdigit(static_cast<unsigned char>(prompt[j]))) ++j;
    if (j > i + 1 && j < prompt.size() && prompt[j] == 'x') last = prompt.substr(i, j + 1 - i);
  }
  return last;
}

class SteppedGenerator : public TextGenerator {
 public:
  static constexpr int kChunks = 3;
  enum Phase : int { kPlanned, kQueued, kPrefill, kDecode, kDone };
  struct Run {
    int hold_at = -1;  // the unit to wait before (-1: none)
    int words = 3;
    std::size_t big = 0;  // bytes per decode piece (0: short words)
    // Speculative decoding (server/decode_loop.hpp, as FlashNextGenerator with --mtp-model): a decode unit is a
    // cycle that emits 1 + accepted pieces (accepted drawn from `seed`); the departure is checked between them.
    bool spec = false;
    std::uint64_t seed = 0;
    std::atomic<bool> release{false}, held{false}, cancelled{false};
    std::atomic<int> phase{kPlanned}, work{0}, work_at_hold{-1};
  };
  bool spec = false;          // plans decode speculatively
  std::uint64_t spec_seed = 0;
  /// The next request whose prompt contains `marker` runs under this plan.
  std::shared_ptr<Run> Plan(const std::string& marker, int hold_at, int words, std::size_t big = 0) {
    auto run = std::make_shared<Run>();
    run->hold_at = hold_at;
    run->words = words;
    run->big = big;
    run->spec = spec;
    run->seed = spec_seed++;
    std::lock_guard<std::mutex> lk(mu);
    plans[marker].push_back(run);
    return run;
  }
  Result Generate(const Params& p, const std::function<bool(std::string_view)>& on_text) override {
    std::shared_ptr<Run> run;
    {
      std::lock_guard<std::mutex> lk(mu);
      prompts.push_back(p.prompt);
      if (auto it = plans.find(LastMarker(p.prompt)); it != plans.end() && !it->second.empty()) {
        run = it->second.front();
        it->second.pop_front();
      }
    }
    if (!run) run = std::make_shared<Run>();
    Result r;
    r.prompt_tokens = p.prompt.size();
    const auto gone = [&] { return p.cancelled && p.cancelled(); };
    const auto end = [&](bool cancelled) {
      run->cancelled = r.cancelled = cancelled;
      run->phase = kDone;
      return r;
    };
    run->phase = kQueued;
    std::unique_lock<std::timed_mutex> lk(engine, std::defer_lock);
    while (!lk.try_lock_for(std::chrono::milliseconds(1)))
      if (gone()) return end(true);
    const auto hold = [&] {
      run->work_at_hold = run->work.load();
      run->held = true;
      while (!run->release && !gone()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      run->held = false;
    };
    const auto unit = [&](int u) {
      if (u == run->hold_at) hold();
      if (gone()) return false;
      std::this_thread::sleep_for(std::chrono::microseconds(300));
      ++run->work;
      return true;
    };
    run->phase = kPrefill;
    for (int c = 0; c < kChunks; ++c)
      if (!unit(c)) return end(true);
    run->phase = kDecode;
    const auto piece = [&](int w) {
      return w == 0     ? std::string("r\n</think>\n\n")
             : run->big ? std::string(run->big, static_cast<char>('a' + w % 26))
                        : "w" + std::to_string(w) + " ";
    };
    if (run->spec) {
      // Piece w is token w; token words + 1 is the stop (it can sit inside an accepted run). A hold at unit
      // kChunks + w waits after piece w - 1 went out, so the leave lands between a cycle's tokens or between cycles.
      engine::MtpDepthChooser chooser(engine::MtpMeasuredCosts(), {.limit = 4});
      std::mt19937_64 rng(run->seed);
      std::int32_t next = 0;
      DecodeIo io;
      io.step = [&](std::int32_t, std::uint32_t drafts, DecodeCycle* c, std::string*) {
        std::this_thread::sleep_for(std::chrono::microseconds(300));
        ++run->work;
        c->drafted = drafts;
        c->accepted = drafts ? static_cast<std::uint32_t>(rng() % (drafts + 1)) : 0;
        for (std::uint32_t i = 0; i <= c->accepted; ++i) c->tokens.push_back(next++);
        return true;
      };
      io.ran = [](std::int32_t) {};
      io.gone = gone;
      io.is_stop = [&](text::TokenId t) { return t > run->words; };
      io.emit = [&](text::TokenId t) {
        const bool more = on_text(piece(t));
        if (run->hold_at == kChunks + t + 1) hold();
        return more;
      };
      const DecodeOutcome out = RunDecode(0, static_cast<std::uint32_t>(p.max_tokens), &chooser, io, &r.spec);
      r.completion_tokens = out.completion;
      r.length_limited = out.length_limited;
      return end(out.cancelled);
    }
    for (int w = 0; w <= run->words; ++w) {
      if (r.completion_tokens == p.max_tokens) {
        r.length_limited = true;
        break;
      }
      if (!unit(kChunks + w)) return end(true);
      ++r.completion_tokens;
      if (!on_text(piece(w))) break;
    }
    return end(false);
  }
  std::mutex mu;
  std::map<std::string, std::deque<std::shared_ptr<Run>>> plans;
  std::vector<std::string> prompts;
  std::timed_mutex engine;
};


/// G3: SteppedGenerator's plans served through the real ScheduledGenerator (server/scheduled_generator.hpp: the
/// scheduler, its engine thread, admission, fences, publication) on a fake engine. A prompt's bytes are its tokens;
/// a request prefills in kChunks slices and then decodes one piece a step (token kWord + w, with its big size in the
/// high bits; kStop ends it); each unit sleeps 300 us and counts as work; a planned hold blocks the engine thread
/// at its unit until released or until the scheduler knows the client is gone (then that unit does no work).
class FakeBackend : public ScheduledBackend {
 public:
  static constexpr std::int32_t kWord = 1 << 20, kStop = kWord - 1;
  explicit FakeBackend(SteppedGenerator& plans) : g_(plans) {}
  std::function<bool(Id)> departed;  // whether the scheduler knows `id`'s client left (set once the generator exists)

  bool Encode(const std::string& prompt, std::vector<std::int32_t>* ids, std::size_t* stable) const override {
    ids->clear();
    for (unsigned char ch : prompt) ids->push_back(ch);
    *stable = ids->empty() ? 0 : ids->size() - 1;
    return true;
  }
  [[nodiscard]] std::string TokenBytes(text::TokenId t) const override {
    if (static_cast<std::int32_t>(t) < kWord) return {};
    const std::uint32_t v = static_cast<std::uint32_t>(t) - kWord, w = v & 0xffff, big = (v >> 16) * 1024;
    return w == 0 ? std::string("r\n</think>\n\n") : big ? std::string(big, static_cast<char>('a' + w % 26))
                                                         : "w" + std::to_string(w) + " ";
  }
  [[nodiscard]] bool IsStop(text::TokenId t) const override { return static_cast<std::int32_t>(t) == kStop; }
  [[nodiscard]] bool HasTokenizer() const override { return true; }
  [[nodiscard]] double BytesPerToken() const override { return 1; }
  void Admitted(Id id, std::span<const std::int32_t> prompt) override {
    std::string text(prompt.begin(), prompt.end());
    std::shared_ptr<SteppedGenerator::Run> run;
    {
      std::lock_guard<std::mutex> lk(g_.mu);
      g_.prompts.push_back(text);
      if (auto it = g_.plans.find(LastMarker(text)); it != g_.plans.end() && !it->second.empty()) {
        run = it->second.front();
        it->second.pop_front();
      }
    }
    if (!run) run = std::make_shared<SteppedGenerator::Run>();
    run->phase = SteppedGenerator::kQueued;
    std::lock_guard<std::mutex> lk(mu_);
    runs_[id] = {run};
  }
  bool Begin(Id id, std::span<const std::int32_t>, std::size_t, std::uint32_t, bool, bool, SpecDecodeStats*,
             std::size_t* reused, std::uint64_t* bound, std::string*) override {
    Of(id).run->phase = SteppedGenerator::kPrefill;
    *reused = 0;
    *bound = 0;
    return true;
  }
  bool PrefillSlice(Id id, std::uint32_t, std::size_t* ran, bool* done, std::string*) override {
    auto& r = Of(id);
    *ran = 0;
    *done = false;
    if (!Unit(id, *r.run, r.chunk)) return true;  // its client left during a hold: nothing done
    *ran = 1;
    *done = ++r.chunk >= SteppedGenerator::kChunks;
    if (*done) r.run->phase = SteppedGenerator::kDecode;
    return true;
  }
  bool Step(Id id, std::int32_t, std::uint32_t, const sampling::SamplingConfig&, sampling::SamplerState*,
            DecodeCycle* c, std::string*) override {
    c->tokens.assign(1, Next(id));
    return true;
  }
  bool StepBatch(std::vector<BatchItem>& items, std::string*) override {
    for (auto& it : items) it.out = Next(it.id);
    return true;
  }
  void Ran(Id, std::int32_t) override {}
  void Finish(Id id) override { End(id, false); }
  void Abandon(Id id) override { End(id, true); }
  std::optional<Scheduler::Retained> Park(Id id) override {
    End(id, true);
    return std::nullopt;
  }
  void Fenced(Id id) override { End(id, true); }
  void Evict(Scheduler::Key) override {}
  [[nodiscard]] bool HasLease(Id id) const override {
    std::lock_guard<std::mutex> lk(mu_);
    return runs_.count(id) > 0 && runs_.at(id).run->phase.load() >= SteppedGenerator::kPrefill;
  }
  [[nodiscard]] std::uint64_t Generation() const override { return 0; }
  [[nodiscard]] std::vector<Scheduler::Retained> Retained() const override { return {}; }
  void SetPins(const std::set<std::uint64_t>&) override {}

 private:
  struct R {
    std::shared_ptr<SteppedGenerator::Run> run;
    int chunk = 0, word = 0;
  };
  R& Of(Id id) {
    std::lock_guard<std::mutex> lk(mu_);
    return runs_.at(id);
  }
  // One unit of `id`: a planned hold first; false (no work) when its client left during it.
  bool Unit(Id id, SteppedGenerator::Run& run, int u) {
    if (u == run.hold_at) {
      run.work_at_hold = run.work.load();
      run.held = true;
      while (!run.release && !(departed && departed(id))) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      run.held = false;
      if (departed && departed(id)) return false;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(300));
    ++run.work;
    return true;
  }
  std::int32_t Next(Id id) {
    auto& r = Of(id);
    if (r.word > r.run->words || !Unit(id, *r.run, SteppedGenerator::kChunks + r.word)) return kStop;
    const int w = r.word++;
    return kWord + w + static_cast<std::int32_t>((r.run->big / 1024) << 16);
  }
  void End(Id id, bool cancelled) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = runs_.find(id);
    if (it == runs_.end()) return;
    if (cancelled) it->second.run->cancelled = true;
    it->second.run->phase = SteppedGenerator::kDone;
    runs_.erase(it);
  }
  SteppedGenerator& g_;
  mutable std::mutex mu_;
  std::map<Id, R> runs_;
};

/// The in-memory store, with an injectable write failure (a full disk, a locked SQLite file).
struct FailableStore : InMemoryResponseStore {
  using InMemoryResponseStore::InMemoryResponseStore;
  bool Put(const std::string& id, const StoredResponse& r, std::string* error) override {
    if (fail.load()) {
      if (error) *error = "injected store failure";
      return false;
    }
    return InMemoryResponseStore::Put(id, r, error);
  }
  std::atomic<bool> fail{false};
};

struct ServingRig {
  /// `scheduled`: the service runs on the G3 ScheduledGenerator over a FakeBackend serving `gen`'s plans.
  explicit ServingRig(int sse_keepalive_ms = 0, bool scheduled = false) : server({.send_timeout_ms = 200}) {
    ServiceOptions o;
    o.sse_keepalive_ms = sse_keepalive_ms;
    o.observer = &observed;
    o.store = &store;
    o.now_s = [] { return std::int64_t{1700000000}; };
    o.on_refused = [this](const std::string& why) {
      std::lock_guard<std::mutex> lk(observed.mu);
      refused.push_back(why);
    };
    if (scheduled) {
      auto fb = std::make_unique<FakeBackend>(gen);
      fake = fb.get();
      sched = std::make_unique<ScheduledGenerator>(std::move(fb), 1u << 20);
      fake->departed = [s = sched.get()](Scheduler::Id id) {
        const auto v = s->scheduler().Request(id);
        return !v || v->departed;
      };
    }
    service = std::make_unique<OpenAiService>(scheduled ? static_cast<TextGenerator&>(*sched) : gen, o);
    service->Register(server);
    std::string err;
    if (!server.Start(&err)) throw std::runtime_error(err);
  }
  ~ServingRig() { server.Stop(); }
  SteppedGenerator gen;  // (before `sched`: the fake backend serves its plans)
  std::unique_ptr<ScheduledGenerator> sched;
  FakeBackend* fake = nullptr;
  FailableStore store{100000};
  Recording observed;
  std::vector<std::string> refused;  // 429s, by why (under observed.mu)
  HttpServer server;
  std::unique_ptr<OpenAiService> service;
};

/// One client connection ("Connection: close": the server closes after its reply).
struct Conn {
  int fd = -1;
  std::string raw;
  Conn(int port, const std::string& body, int rcvbuf = 0, const std::string& headers = "") {
    fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (rcvbuf) ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof rcvbuf);
    const timeval tv{10, 0};  // a reply that never comes fails the test instead of hanging it
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<std::uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) throw std::runtime_error("connect");
    const std::string req = "POST /v1/responses HTTP/1.1\r\nConnection: close\r\n" + headers + "Content-Length: " +
                            std::to_string(body.size()) + "\r\n\r\n" + body;
    ::send(fd, req.data(), req.size(), MSG_NOSIGNAL);
  }
  ~Conn() { Close(); }
  void Close() {
    if (fd >= 0) ::close(fd);
    fd = -1;
  }
  void ReadToEnd() {
    char b[65536];
    for (ssize_t n; (n = ::recv(fd, b, sizeof b, 0)) > 0;) raw.append(b, static_cast<std::size_t>(n));
  }
  [[nodiscard]] int Status() const { return raw.size() >= 12 ? std::stoi(raw.substr(9, 3)) : 0; }
  /// The body, dechunked (as far as it arrived).
  [[nodiscard]] std::string Body() const {
    const auto end = raw.find("\r\n\r\n");
    if (end == std::string::npos) return {};
    const std::string head = raw.substr(0, end), rest = raw.substr(end + 4);
    if (head.find("Transfer-Encoding: chunked") == std::string::npos) return rest;
    std::string out;
    for (std::size_t i = 0;;) {
      const auto eol = rest.find("\r\n", i);
      if (eol == std::string::npos) break;
      const std::size_t n = std::stoul(rest.substr(i, eol - i), nullptr, 16);
      if (n == 0 || eol + 2 + n > rest.size()) break;
      out += rest.substr(eol + 2, n);
      i = eol + 2 + n + 2;
    }
    return out;
  }
};

/// The text of a Responses object's message items.
std::string OutputText(const Value& response) {
  std::string out;
  for (const auto& item : response.find("output")->items())
    if (item.find("type")->str() == "message")
      for (const auto& part : item.find("content")->items()) out += part.find("text")->str();
  return out;
}

/// What a client got: the whole reply (a terminal Responses event, or a whole JSON object), and its text.
struct Delivery {
  bool full = false;
  std::string status, text;
};

Delivery Delivered(const Conn& c, bool stream) {
  Delivery d;
  if (c.Status() != 200) return d;
  const std::string body = c.Body();
  if (!stream) {
    std::string err;
    const auto v = json::Parse(body, &err);
    if (!v) return d;
    d.full = true;
    d.status = v->find("status")->str();
    d.text = OutputText(*v);
    return d;
  }
  bool done = false;
  for (const auto& e : Events(body, &done)) {
    const std::string type = e.find("type")->str();
    if (type == "response.output_text.delta") d.text += e.find("delta")->str();
    if (type == "response.completed" || type == "response.incomplete") {
      d.full = true;
      d.status = e.find("response")->find("status")->str();
    }
  }
  return d;
}

/// One finished request, as the test saw it.
struct Finished {
  std::string marker, body;
  int attempt = 0;  // the how-many-th request with this marker
  bool stream = false;
  Delivery got;
  std::shared_ptr<SteppedGenerator::Run> run;
  int work_at_leave = -1;  // the run's work when its client left (-1: it did not leave)
  bool store_failed = false;  // its store write failed (injected)
};

std::string RequestBody(const std::string& marker, bool stream, std::uint32_t max_tokens, const std::string& prev = "") {
  std::string b = R"({"input": ")" + marker + R"(", "stream": )" + (stream ? "true" : "false") +
                  R"(, "max_output_tokens": )" + std::to_string(max_tokens);
  if (!prev.empty()) b += R"(, "previous_response_id": ")" + prev + "\"";
  return b + "}";
}

bool Await(const std::function<bool()>& f, int ms = 5000) {
  for (int i = 0; i < ms; ++i) {
    if (f()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return f();
}

}  // namespace

// Clients that complete, leave while queued / mid-prefill / mid-decode, stall (stop reading), retry, and continue
// from earlier responses, in any order. Invariants after every step:
//   - a response is "completed" (stored status, observer finish_reason stop or length) only when its client got it
//     in full; otherwise it is cancelled: finish_reason "cancelled", stored as a cancelled object without a turn;
//   - a stored turn holds exactly the text its client received;
//   - no work for a departed client: its run stops at the unit where it left (never queued work, never another
//     chunk or step);
//   - a stalled client delays another one by about the send timeout, not indefinitely;
//   - continuing from a completed response sends its delivered text back to the generator; from a cancelled one, 400.
TEST(OpenAiService, ServingModelUnderDeparturesStallsAndRetries) {
  hegel::test([](hegel::TestCase& tc) {
    // G3: the same machine on the scheduled generator (fair scheduler, engine thread; a fake engine) as well; there
    // three sandboxes also run at once, and a sandbox's second request is refused 429 while its first is in flight.
    const bool scheduled = tc.draw(gs::booleans());
    ServingRig rig(0, scheduled);
    rig.gen.spec = !scheduled && tc.draw(gs::booleans());  // decode in speculative cycles (--mtp-model; B = 1 only)
    rig.gen.spec_seed = tc.draw(gs::integers<std::uint64_t>());
    const int port = rig.server.port();
    std::vector<Finished> done;
    std::map<std::string, int> attempts;
    int next = 0;
    const auto fresh = [&] { return "m" + std::to_string(next++) + "x"; };
    const auto finish = [&](Finished f) {
      f.attempt = attempts[f.marker]++;
      done.push_back(std::move(f));
    };
    // The observer's record and the store's entry for a finished request.
    const auto record_of = [&](const Finished& f) -> std::optional<GenerationRecord> {
      std::optional<GenerationRecord> found;
      Await([&] {
        std::lock_guard<std::mutex> lk(rig.observed.mu);
        int seen = 0;
        for (const auto& r : rig.observed.records)
          if (LastMarker(r.input) == f.marker && seen++ == f.attempt) return found = r, true;
        return false;
      });
      return found;
    };
    const auto check = [&](const Finished& f) {
      ASSERT_TRUE(Await([&] { return f.run->phase.load() == SteppedGenerator::kDone; })) << f.marker << " still runs";
      const auto rec = record_of(f);
      ASSERT_TRUE(rec) << "no record for " << f.marker;
      if (f.store_failed) {
        // Answered with an error, recorded as one, and nothing kept: the id the client may hold is unknown.
        ASSERT_FALSE(f.got.full);
        ASSERT_EQ(rec->finish_reason, "error");
        ASSERT_NE(rec->error.find("not stored"), std::string::npos) << rec->error;
        ASSERT_FALSE(rig.store.Get(rec->request_id)) << "a response answered with a store error is stored";
        return;
      }
      if (f.got.full) {
        ASSERT_TRUE(rec->finish_reason == "stop" || rec->finish_reason == "length") << rec->finish_reason;
        ASSERT_FALSE(f.run->cancelled.load());
      } else {
        ASSERT_EQ(rec->finish_reason, "cancelled") << f.marker << ": a response its client did not get in full";
      }
      if (f.work_at_leave >= 0)
        ASSERT_LE(f.run->work.load(), f.work_at_leave) << f.marker << ": work after its client left";
      if (!f.got.full && !f.stream) {
        // A non-streaming client that left never got the id: nothing is kept. (Every handler has returned once
        // the server holds no connection.)
        ASSERT_TRUE(Await([&] { return rig.server.active_connections() == 0; }));
        ASSERT_FALSE(rig.store.Get(rec->request_id)) << f.marker << ": a departed non-streaming response is stored";
        return;
      }
      std::optional<StoredResponse> stored;  // put just after the observer's record
      Await([&] { return (stored = rig.store.Get(rec->request_id)).has_value(); });
      ASSERT_TRUE(stored) << rec->request_id;
      const std::string status = stored->response.find("status")->str();
      if (f.got.full) {
        ASSERT_EQ(status, f.got.status);
        ASSERT_NE(status, "cancelled");
        ASSERT_EQ(OutputText(stored->response), f.got.text);
        ASSERT_FALSE(stored->conversation.is(Value::Kind::kNull));
        ASSERT_NE(CompactJson(stored->conversation).find(f.got.text), std::string::npos);
      } else {
        ASSERT_EQ(status, "cancelled") << "a response its client did not get in full is stored as " << status;
        ASSERT_TRUE(stored->conversation.is(Value::Kind::kNull)) << "the undelivered turn was kept";
      }
    };
    const auto complete = [&](const std::string& marker, const std::string& body, bool stream, int words) {
      Finished f{marker, body, 0, stream, {}, rig.gen.Plan(marker, -1, words)};
      Conn c(port, body);
      c.ReadToEnd();
      f.got = Delivered(c, stream);
      EXPECT_TRUE(f.got.full) << c.raw.substr(0, 300);
      return f;
    };
    const int steps = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 6}));
    for (int step = 0; step < steps; ++step) {
      const auto rule = tc.draw(gs::sampled_from<std::string>(
          {"complete", "leave-queued", "leave-prefill", "leave-decode", "stall", "retry", "continue",
           "store-fails", "concurrent", "limit429"}));
      const bool stream = tc.draw(gs::booleans());
      const int words = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 8}));
      const auto max_tokens = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 12}));
      if (rule == "complete") {
        const auto m = fresh();
        finish(complete(m, RequestBody(m, stream, max_tokens), stream, words));
      } else if (rule == "leave-queued") {
        // A holder keeps the engine (held in decode) while the victim queues behind it and leaves.
        const auto hm = fresh(), vm = fresh();
        Finished h{hm, RequestBody(hm, stream, 64), 0, stream, {}, rig.gen.Plan(hm, SteppedGenerator::kChunks + 1, words)};
        Conn hc(port, h.body);
        std::thread reader([&] { hc.ReadToEnd(); });
        struct Join {  // also when an assertion returns early
          std::thread& t;
          SteppedGenerator::Run& r;
          ~Join() {
            r.release = true;
            if (t.joinable()) t.join();
          }
        } join{reader, *h.run};
        ASSERT_TRUE(Await([&] { return h.run->held.load(); }));
        Finished v{vm, RequestBody(vm, tc.draw(gs::booleans()), 64), 0, false, {}, rig.gen.Plan(vm, -1, words)};
        v.stream = v.body.find("\"stream\": true") != std::string::npos;
        {
          Conn vc(port, v.body);
          ASSERT_TRUE(Await([&] { return v.run->phase.load() == SteppedGenerator::kQueued; }));
          v.work_at_leave = 0;
        }
        // Scheduled: the holder's unit holds the engine thread, so the victim is fenced at the next unit boundary,
        // after the release (never having run) -- once the scheduler knows its client left (its HTTP thread polls
        // the connection every 20 ms).
        if (scheduled) {
          ASSERT_TRUE(Await([&] {
            int connected = 0;
            for (const auto& r : rig.sched->scheduler().Requests()) connected += !r.departed;
            return connected == 1;  // the holder
          }));
          h.run->release = true;
        }
        ASSERT_TRUE(Await([&] { return v.run->phase.load() == SteppedGenerator::kDone; })) << "the departed request stayed queued";
        h.run->release = true;
        reader.join();
        h.got = Delivered(hc, stream);
        EXPECT_TRUE(h.got.full);
        finish(std::move(h));
        finish(std::move(v));
      } else if (rule == "leave-prefill" || rule == "leave-decode") {
        const auto m = fresh();
        const int at = rule == "leave-prefill"
                           ? tc.draw(gs::integers<int>({.min_value = 1, .max_value = SteppedGenerator::kChunks - 1}))
                           : SteppedGenerator::kChunks + tc.draw(gs::integers<int>({.min_value = 1, .max_value = words}));
        Finished f{m, RequestBody(m, stream, 64), 0, stream, {}, rig.gen.Plan(m, at, words)};
        {
          Conn c(port, f.body);
          ASSERT_TRUE(Await([&] { return f.run->held.load(); }));
          f.work_at_leave = f.run->work_at_hold.load();  // `at` stepping; the cycles done so far speculating
        }
        finish(std::move(f));
      } else if (rule == "stall") {
        // A streaming client that never reads a reply far larger than the socket buffers; another client must get
        // the engine within about the send timeout (200 ms here).
        const auto sm = fresh();
        Finished s{sm, RequestBody(sm, true, 64), 0, true, {}, rig.gen.Plan(sm, -1, 60, 256 * 1024)};
        Conn sc(port, s.body, 4096);
        ASSERT_TRUE(Await([&] { return s.run->phase.load() == SteppedGenerator::kDecode; }));
        const auto t0 = std::chrono::steady_clock::now();
        const auto m = fresh();
        finish(complete(m, RequestBody(m, stream, max_tokens), stream, words));
        EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(3)) << "a stalled client held the engine";
        ASSERT_TRUE(Await([&] { return s.run->phase.load() == SteppedGenerator::kDone; }));
        sc.Close();
        finish(std::move(s));
      } else if (rule == "store-fails") {
        // The store refuses the write: the client gets an error instead of a finished response (non-streaming: a
        // 500; streaming: an error event in place of response.completed), never an id that later 404s.
        const auto m = fresh();
        Finished f{m, RequestBody(m, stream, max_tokens), 0, stream, {}, rig.gen.Plan(m, -1, words)};
        f.store_failed = true;
        rig.store.fail = true;
        Conn c(port, f.body);
        c.ReadToEnd();
        rig.store.fail = false;
        f.got = Delivered(c, stream);
        if (stream) {
          ASSERT_EQ(c.Status(), 200);
          bool ended = false;
          const auto events = Events(c.Body(), &ended);
          ASSERT_FALSE(events.empty());
          ASSERT_EQ(events.back().find("type")->str(), "error") << c.raw.substr(0, 300);
          ASSERT_NE(CompactJson(events.back()).find("not stored"), std::string::npos);
        } else {
          ASSERT_EQ(c.Status(), 500) << c.raw.substr(0, 300);
          ASSERT_NE(c.Body().find("not stored"), std::string::npos) << c.Body();
        }
        finish(std::move(f));
      } else if (rule == "concurrent") {
        // Three sandboxes (distinct principals) at once: each gets its whole answer.
        std::vector<Finished> fs;
        std::vector<std::unique_ptr<Conn>> cs;
        for (int k = 0; k < 3; ++k) {
          const auto m = fresh();
          fs.push_back({m, RequestBody(m, stream, max_tokens), 0, stream, {}, rig.gen.Plan(m, -1, words)});
          cs.push_back(std::make_unique<Conn>(port, fs.back().body, 0, "X-Halo-Principal: sandbox-" + m + "\r\n"));
        }
        std::vector<std::thread> readers;
        for (auto& c : cs) readers.emplace_back([&c] { c->ReadToEnd(); });
        for (auto& t : readers) t.join();
        for (int k = 0; k < 3; ++k) {
          fs[static_cast<std::size_t>(k)].got = Delivered(*cs[static_cast<std::size_t>(k)], stream);
          EXPECT_TRUE(fs[static_cast<std::size_t>(k)].got.full) << cs[static_cast<std::size_t>(k)]->raw.substr(0, 300);
          finish(std::move(fs[static_cast<std::size_t>(k)]));
        }
      } else if (rule == "limit429" && scheduled) {
        // D3: a sandbox's second request while its first is in flight is refused 429 with Retry-After before
        // anything is sent; it never runs, and nothing is recorded or stored for it.
        const auto hm = fresh(), sm = fresh();
        Finished h{hm, RequestBody(hm, stream, 64), 0, stream, {}, rig.gen.Plan(hm, 1, words)};
        Conn hc(port, h.body, 0, "X-Halo-Principal: sandbox-limited\r\n");
        std::thread reader([&] { hc.ReadToEnd(); });
        struct Join {
          std::thread& t;
          SteppedGenerator::Run& r;
          ~Join() {
            r.release = true;
            if (t.joinable()) t.join();
          }
        } join{reader, *h.run};
        ASSERT_TRUE(Await([&] { return h.run->held.load(); }));
        const auto second = rig.gen.Plan(sm, -1, words);
        std::size_t refused_before = 0;
        {
          std::lock_guard<std::mutex> lk(rig.observed.mu);
          refused_before = rig.refused.size();
        }
        Conn sc(port, RequestBody(sm, tc.draw(gs::booleans()), 64), 0, "X-Halo-Principal: sandbox-limited\r\n");
        sc.ReadToEnd();
        ASSERT_EQ(sc.Status(), 429) << sc.raw.substr(0, 300);
        ASSERT_NE(sc.raw.find("Retry-After: "), std::string::npos) << sc.raw.substr(0, 300);
        ASSERT_NE(sc.Body().find("rate_limit_exceeded"), std::string::npos) << sc.Body();
        ASSERT_EQ(second->phase.load(), SteppedGenerator::kPlanned) << "a refused request was admitted";
        {
          std::lock_guard<std::mutex> lk(rig.observed.mu);
          ASSERT_EQ(rig.refused.size(), refused_before + 1);
          for (const auto& r : rig.observed.records) ASSERT_NE(LastMarker(r.input), sm) << "a refused request was recorded";
        }
        h.run->release = true;
        reader.join();
        h.got = Delivered(hc, stream);
        EXPECT_TRUE(h.got.full);
        finish(std::move(h));
      } else if (rule == "retry" && !done.empty()) {
        // The same body again (a client retrying after a lost answer): a new, independent response.
        const Finished prev = done[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = done.size() - 1}))];
        const std::string marker = prev.marker, body = prev.body;
        const bool s = prev.stream;
        if (prev.run->big) continue;
        finish(complete(marker, body, s, words));
      } else if (rule == "continue" && !done.empty()) {
        const Finished prev = done[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = done.size() - 1}))];
        const auto rec = record_of(prev);
        ASSERT_TRUE(rec);
        const auto m = fresh();
        const std::string body = RequestBody(m, stream, max_tokens, rec->request_id);
        if (prev.store_failed) {
          Conn c(port, body);
          c.ReadToEnd();
          ASSERT_EQ(c.Status(), 404) << "continued from a response that was not stored: " << c.raw.substr(0, 300);
        } else if (prev.got.full) {
          const std::size_t before = rig.gen.prompts.size();
          finish(complete(m, body, stream, words));
          std::lock_guard<std::mutex> lk(rig.gen.mu);
          ASSERT_EQ(rig.gen.prompts.size(), before + 1);
          ASSERT_NE(rig.gen.prompts.back().find(prev.got.text), std::string::npos)
              << "the continuation lost the delivered turn";
        } else {
          Conn c(port, body);
          c.ReadToEnd();
          // Streaming: the cancelled object is kept (its client may hold the id), so a clear 400; non-streaming:
          // nothing was kept, 404.
          ASSERT_EQ(c.Status(), prev.stream ? 400 : 404) << "continued from a cancelled response: " << c.raw.substr(0, 300);
        }
      }
      for (const auto& f : done) check(f);
    }
  }, {.test_cases = 25});
}

// G3 D3 through the service: a sandbox's second turn while its first is in flight is answered 429 with
// Retry-After and an OpenAI rate-limit error before anything streams; it never reaches the engine. Another sandbox
// (or the same one once its turn finished) is served.
TEST(OpenAiService, ScheduledSecondTurnOfASandboxIsRefused429) {
  ServingRig rig(0, true);
  const int port = rig.server.port();
  auto held = rig.gen.Plan("m901x", 1, 3);
  Conn first(port, RequestBody("m901x", true, 16), 0, "X-Halo-Principal: sb\r\n");
  std::thread reader([&] { first.ReadToEnd(); });
  ASSERT_TRUE(Await([&] { return held->held.load(); }));
  auto refused = rig.gen.Plan("m902x", -1, 3);
  Conn second(port, RequestBody("m902x", true, 16), 0, "X-Halo-Principal: sb\r\n");
  second.ReadToEnd();
  EXPECT_EQ(second.Status(), 429) << second.raw.substr(0, 300);
  EXPECT_NE(second.raw.find("Retry-After: "), std::string::npos);
  EXPECT_NE(second.Body().find("\"type\":\"rate_limit_exceeded\""), std::string::npos) << second.Body();
  EXPECT_EQ(refused->phase.load(), SteppedGenerator::kPlanned);
  Conn other(port, RequestBody("m903x", false, 16), 0, "X-Halo-Principal: sb-other\r\n");
  held->release = true;
  other.ReadToEnd();
  reader.join();
  EXPECT_EQ(other.Status(), 200);
  EXPECT_TRUE(Delivered(first, true).full);
  Conn again(port, RequestBody("m904x", false, 16), 0, "X-Halo-Principal: sb\r\n");
  again.ReadToEnd();
  EXPECT_EQ(again.Status(), 200) << again.raw.substr(0, 300);
}

// G3 M4 (--decider-mode scheduled): a decider evaluation is a scheduler unit on the engine thread. Six arrive while
// a request's first prefill slice runs: the four with credit (D4 burst 4, no refill here) run at the next unit
// boundary, ahead of that request's remaining slices; the two without credit take the decider class's fair share;
// every evaluation runs once, on the engine thread, and the request's answer is unchanged. /metrics counts them.
TEST(ScheduledDecider, CreditedDecidersRunAtTheNextBoundaryOnTheEngineThread) {
  SteppedGenerator gen;
  auto plan = gen.Plan("m1x", -1, 3);
  ScheduledOptions o;
  o.sched.decider_burst = 4;
  o.sched.decider_refill_per_s = 0;
  std::mutex mu;
  std::vector<Scheduler::Kind> kinds;
  std::vector<std::future<bool>> deciders;
  std::atomic<int> ran{0}, off_engine{0};
  std::atomic<bool> launched{false};
  ScheduledGenerator* self = nullptr;
  const auto main_id = std::this_thread::get_id();
  o.before_unit = [&](const Scheduler::Unit& u) {
    {
      std::lock_guard<std::mutex> lk(mu);
      kinds.push_back(u.kind);
    }
    if (u.kind != Scheduler::Kind::kSlice || launched.exchange(true)) return;
    const auto engine_id = std::this_thread::get_id();
    for (int i = 0; i < 6; ++i)
      deciders.push_back(std::async(std::launch::async, [&, engine_id] {
        std::string e;
        return self->RunDecider(
            [&, engine_id] {
              if (std::this_thread::get_id() != engine_id || std::this_thread::get_id() == main_id) ++off_engine;
              ++ran;
            },
            &e);
      }));
    // Wait (on the engine thread, between units) until all six are admitted.
    for (int spin = 0; spin < 5000; ++spin) {
      int n = 0;
      for (const auto& v : self->scheduler().Requests()) n += v.decider ? 1 : 0;
      if (n == 6) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };
  ScheduledGenerator sg(std::make_unique<FakeBackend>(gen), 1u << 20, {}, o);
  self = &sg;
  const std::string prompt = "a turn m1x";
  std::vector<std::int32_t> ids(prompt.begin(), prompt.end());
  TextGenerator::Params p;
  p.max_tokens = 16;
  std::vector<text::TokenId> got;
  const auto r = sg.GenerateIds(ids, ids.size() - 1, p, [&](text::TokenId t) {
    got.push_back(t);
    return true;
  });
  for (auto& f : deciders) EXPECT_TRUE(f.get());
  EXPECT_TRUE(r.error.empty()) << r.error;
  EXPECT_FALSE(r.cancelled);
  {  // the same request served alone
    SteppedGenerator gen2;
    gen2.Plan("m1x", -1, 3);
    ScheduledGenerator alone(std::make_unique<FakeBackend>(gen2), 1u << 20);
    std::vector<text::TokenId> want;
    const auto r2 = alone.GenerateIds(ids, ids.size() - 1, p, [&](text::TokenId t) {
      want.push_back(t);
      return true;
    });
    EXPECT_TRUE(r2.error.empty());
    EXPECT_FALSE(want.empty());
    EXPECT_EQ(got, want);
  }
  EXPECT_EQ(ran.load(), 6);
  EXPECT_EQ(off_engine.load(), 0);
  std::vector<Scheduler::Kind> k;
  {
    std::lock_guard<std::mutex> lk(mu);
    k = kinds;
  }
  ASSERT_GE(k.size(), 5u);
  EXPECT_EQ(k[0], Scheduler::Kind::kSlice);
  for (int i = 1; i <= 4; ++i) EXPECT_EQ(k[static_cast<std::size_t>(i)], Scheduler::Kind::kDecide) << "unit " << i;
  EXPECT_EQ(std::count(k.begin(), k.end(), Scheduler::Kind::kDecide), 6);
  const auto c = sg.scheduler().counters();
  EXPECT_EQ(c.decides, 6u);
  EXPECT_EQ(c.decider_out_of_credit, 2u);
  std::string m;
  sg.RenderMetrics(m, "model_name=\"t\"");
  EXPECT_NE(m.find("halo:sched_decides_total{model_name=\"t\"} 6\n"), std::string::npos) << m;
  EXPECT_NE(m.find("halo:sched_unit_seconds_count{model_name=\"t\",kind=\"decide\"} 6\n"), std::string::npos);
  EXPECT_NE(m.find("halo:sched_queue_wait_seconds_count{model_name=\"t\"} 1\n"), std::string::npos);
  EXPECT_EQ(m.find("# TYPE halo:sched_unit_seconds histogram"), m.rfind("# TYPE halo:sched_unit_seconds histogram"));
}

// Audit F7: a store write that fails is answered with an error, never a finished response whose id later 404s.
// Non-streaming: a 500 naming the store; streaming: an error event in place of response.completed. Nothing is
// kept, the observer records an error, and continuing from the id is a 404; once the store works again the same
// request succeeds (a retry is safe: every attempt is a new immutable response).
TEST(OpenAiService, AFailedStoreWriteIsAnErrorAndKeepsNothing) {
  ServingRig rig;
  const int port = rig.server.port();
  for (const bool stream : {false, true}) {
    const std::string m = stream ? "m1x" : "m0x";
    (void)rig.gen.Plan(m, -1, 3);
    rig.store.fail = true;
    Conn c(port, RequestBody(m, stream, 8));
    c.ReadToEnd();
    rig.store.fail = false;
    if (stream) {
      ASSERT_EQ(c.Status(), 200);
      bool ended = false;
      const auto events = Events(c.Body(), &ended);
      ASSERT_FALSE(events.empty());
      EXPECT_EQ(events.back().find("type")->str(), "error") << c.Body();
      EXPECT_EQ(c.Body().find("response.completed"), std::string::npos) << "a finished event for an unstored response";
    } else {
      ASSERT_EQ(c.Status(), 500) << c.raw.substr(0, 300);
      EXPECT_NE(c.Body().find("not stored"), std::string::npos) << c.Body();
    }
    const auto rec = AwaitRecord(rig.observed, stream ? 2 : 1);
    ASSERT_TRUE(rec);
    EXPECT_EQ(rec->finish_reason, "error");
    EXPECT_NE(rec->error.find("injected store failure"), std::string::npos) << rec->error;
    EXPECT_FALSE(rig.store.Get(rec->request_id));
    Conn cont(port, RequestBody(stream ? "m3x" : "m2x", false, 8, rec->request_id));
    cont.ReadToEnd();
    EXPECT_EQ(cont.Status(), 404) << cont.raw.substr(0, 300);
  }
  Conn ok(port, RequestBody("m4x", false, 8));
  ok.ReadToEnd();
  EXPECT_EQ(ok.Status(), 200);
  EXPECT_EQ(J(ok.Body()).find("status")->str(), "completed");
}

// Audit F5: a non-streaming client that leaves mid-decode stops the run at the next step, and nothing is stored
// (it never received the id).
TEST(OpenAiService, ADepartedNonStreamingResponseStopsAndIsNotStored) {
  ServingRig rig;
  const auto run = rig.gen.Plan("m0x", SteppedGenerator::kChunks + 2, 6);
  {
    Conn c(rig.server.port(), RequestBody("m0x", false, 64));
    ASSERT_TRUE(Await([&] { return run->held.load(); }));
  }
  ASSERT_TRUE(Await([&] { return run->phase.load() == SteppedGenerator::kDone; }));
  EXPECT_TRUE(run->cancelled.load());
  EXPECT_LE(run->work.load(), SteppedGenerator::kChunks + 2) << "work after the client left";
  const auto rec = AwaitRecord(rig.observed, 1);
  ASSERT_TRUE(rec);
  EXPECT_EQ(rec->finish_reason, "cancelled");
  ASSERT_TRUE(Await([&] { return rig.server.active_connections() == 0; }));
  EXPECT_FALSE(rig.store.Get(rec->request_id)) << "a response nobody received is stored";
}

// Keep-alive and pipelining: the next request's bytes arriving while a non-streaming request runs make the
// connection readable, which is not a departure; the first request completes and then the second is answered.
TEST(OpenAiService, APipelinedNextRequestIsNotADeparture) {
  ServingRig rig;
  const auto first = rig.gen.Plan("m0x", SteppedGenerator::kChunks + 1, 3);
  (void)rig.gen.Plan("m1x", -1, 2);
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  const timeval tv{10, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(rig.server.port()));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  ASSERT_EQ(::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a), 0);
  const auto request = [](const std::string& body, bool close) {
    return "POST /v1/responses HTTP/1.1\r\n" + std::string(close ? "Connection: close\r\n" : "") +
           "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
  };
  const std::string r1 = request(RequestBody("m0x", false, 8), false);
  ::send(fd, r1.data(), r1.size(), MSG_NOSIGNAL);
  ASSERT_TRUE(Await([&] { return first->held.load(); }));
  const std::string r2 = request(RequestBody("m1x", false, 8), true);
  ::send(fd, r2.data(), r2.size(), MSG_NOSIGNAL);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));  // several liveness polls see the readable socket
  first->release = true;
  std::string raw;
  char b[65536];
  for (ssize_t n; (n = ::recv(fd, b, sizeof b, 0)) > 0;) raw.append(b, static_cast<std::size_t>(n));
  ::close(fd);
  EXPECT_FALSE(first->cancelled.load()) << "the pipelined request was taken for a departure";
  std::size_t oks = 0;
  for (std::size_t i = 0; (i = raw.find("HTTP/1.1 200", i)) != std::string::npos; ++i) ++oks;
  EXPECT_EQ(oks, 2u) << raw.substr(0, 400);
  EXPECT_EQ(raw.find("\"cancelled\""), std::string::npos);
}

// Audit D3: with sse_keepalive_ms, a streamed request that waits (here held mid-prefill) writes SSE comments, which
// SSE parsers ignore, so the stream still parses to the same events and completes; without it nothing is written
// before the first event. HALO_SSE_DUMP=<path> saves the stream body for the OpenAI SDK replay check
// (tests/functional/sse_keepalive_sdk_check.py).
TEST(OpenAiService, KeepaliveCommentsWhileAStreamWaitsAndClientsIgnoreThem) {
  for (const int keepalive : {0, 20}) {
    ServingRig rig(keepalive);
    const auto run = rig.gen.Plan("m0x", 1, 3);
    Conn c(rig.server.port(), RequestBody("m0x", true, 8));
    ASSERT_TRUE(Await([&] { return run->held.load(); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    run->release = true;
    c.ReadToEnd();
    const std::string body = c.Body();
    std::size_t comments = 0;
    for (std::size_t i = 0; (i = body.find(": keepalive\n\n", i)) != std::string::npos; ++i) ++comments;
    if (keepalive) EXPECT_GE(comments, 3u) << "about one every 20 ms over 150 ms";
    else EXPECT_EQ(comments, 0u) << "keepalives are opt-in";
    const auto got = Delivered(c, true);
    EXPECT_TRUE(got.full);
    EXPECT_EQ(got.status, "completed");
    EXPECT_FALSE(run->cancelled.load());
    if (const char* dump = std::getenv("HALO_SSE_DUMP"); dump && keepalive) {
      FILE* f = std::fopen(dump, "wb");
      ASSERT_TRUE(f);
      std::fwrite(body.data(), 1, body.size(), f);
      std::fclose(f);
    }
  }
}
