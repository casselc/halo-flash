// The scheduler model's traces replayed through the serving stack: HTTP clients on /v1/completions, the shared
// generation turn (server/turn.hpp), the OpenAI service and the real ScheduledGenerator with its engine thread
// (docs/SCHEDULER.md section 9; tests/server/scheduler_replay_test.cpp replays the same traces against the Scheduler
// class alone).
//
// The engine is lock-stepped to each trace through ScheduledOptions' test hooks: a tick clock (the model's time, also
// every unit's elapsed time), a gate before each Scheduler::Next, and before_unit, which holds the picked unit until
// the trace's unit end; first_token_in_slice is off (the model starts decoding in the next unit). A fake backend
// (TraceBackend) does what the model's unit ends say (prefilled tokens, prefill done), binds and parks as the model
// does, and reports the model's retained prefixes. Each model request is a client:
//   an agent request: a streamed POST /v1/completions with X-Halo-Principal, X-Halo-Chain-Id and Idempotency-Key
//     "<conv>/<turn>" (the model's key), its prompt encoding the conversation and length; a request the server
//     already knows gone (conn false) sends and shuts its socket at once; a departure shuts the client's socket;
//   a decider: ScheduledGenerator::RunDecider (the /v1/systemone route needs a decider model).
// Compared after every step, as the Scheduler-level replay: each admission outcome (from the scheduler's counters),
// a decider's credit, the requests fenced before each pick, the unit picked (kind and requests), every root's and
// principal's virtual time. And at the HTTP surface:
//   a connected admission's response head: 429 with Retry-After (the memory 429's 5 s, a limit 429's 1..cap) and the
//     API's rate-limit error; 200 with Halo-Answer generated / adopted / replayed;
//   a refused request never reached the engine;
//   a client still connected when its generation completed gets the whole stream (every token, then [DONE]);
//   the observer: one record per admitted reader; a reader whose client left is recorded cancelled or detached, one
//     that got its answer stop or length; completion tokens counted once per generation.
// Weaker than the Scheduler-level replay, and why:
//   - deciders do not go through HTTP (no decider model here); their admission and units are compared as above;
//   - the code's first-token-in-slice (production default) is off: the model does not have it, and with it the code's
//     progress differs from the model's at every completing slice (a unit mismatch at once);
//   - departures are what the server learns (the model's depart event); a client that left earlier but is not yet
//     known gone is not modelled at the socket (the trace does not say when its packets stopped);
//   - a reader adopted from a departed one, and the departed one, see only the outcome, not when each piece went out.
// Known refinement of the code over the model, accepted as in the Scheduler-level replay: a request whose whole prompt
// is reused starts with an empty slice; the replay runs it and compares the next pick.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/json.hpp"
#include "server/http.hpp"
#include "server/observer.hpp"
#include "server/openai_service.hpp"
#include "server/scheduled_generator.hpp"
#include "server/scheduler.hpp"

using halo::json::Value;
using namespace halo::server;
namespace text = halo::text;
namespace sampling = halo::sampling;

namespace {

using Clock = std::chrono::steady_clock;

Scheduler::Options ReplayOptions() {  // the model's numbers, as scheduler_replay_test.cpp
  Scheduler::Options o;
  o.per_second = 8;
  o.slice_contended = o.slice_alone = 1;
  o.plimit = 1;
  o.max_batch = 8;
  o.decode_steps = 1;
  o.slice_cost_per_token = 2;
  o.step_cost = 1;
  o.batch_step_extra = 0;
  o.decider_cost = 1;
  o.decider_burst = 1;
  o.decider_refill_per_s = 1;
  o.capacity = 14;
  o.replay_ttl_s = 1000;
  o.keyed_grace_s = 0.5;
  return o;
}

std::vector<std::int32_t> Tokens(int c, int n) {
  std::vector<std::int32_t> v;
  for (int i = 0; i < n; ++i) v.push_back(c * 100 + i);
  return v;
}
std::shared_ptr<const std::vector<std::int32_t>> SharedTokens(int c, int n) {
  return std::make_shared<const std::vector<std::int32_t>>(Tokens(c, n));
}

std::int64_t I(const Value& v, const char* k) {
  const Value* x = v.find(k);
  return x ? static_cast<std::int64_t>(x->number()) : 0;
}
bool B(const Value& v, const char* k) {
  const Value* x = v.find(k);
  return x && x->boolean();
}
std::string S(const Value& v, const char* k) {
  const Value* x = v.find(k);
  return x ? x->str() : "";
}
std::string Cls(const std::string& q) { return q == "D" ? Scheduler::kDeciderClass : q; }
const char* KindName(Scheduler::Kind k) {
  switch (k) {
    case Scheduler::Kind::kSlice: return "slice";
    case Scheduler::Kind::kBatch: return "batch";
    case Scheduler::Kind::kDecide: return "decide";
    default: return "idle";
  }
}
std::string Set(const std::set<std::int64_t>& s) {
  std::string out = "{";
  for (auto x : s) out += (out.size() > 1 ? "," : "") + std::to_string(x);
  return out + "}";
}
bool Await(const std::function<bool()>& f, int ms = 5000) {
  for (const auto until = Clock::now() + std::chrono::milliseconds(ms); Clock::now() < until;) {
    if (f()) return true;
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  return f();
}

/// The prompt text of model conversation `conv` with `len` prompt tokens (TraceBackend::Encode reads it back).
std::string PromptText(int conv, int len) { return "conv " + std::to_string(conv) + " len " + std::to_string(len); }

/// The engine as the model says: the driver tells it, before it releases a unit, each request's progress at the
/// unit's end, the binds of the pick, the parks of the fences, and the retained prefixes.
class TraceBackend : public ScheduledBackend {
 public:
  static constexpr std::int32_t kWord = 1 << 20;
  bool Encode(const std::string& prompt, std::vector<std::int32_t>* ids, std::size_t* stable) const override {
    int conv = 0, len = 0;
    if (std::sscanf(prompt.c_str(), "conv %d len %d", &conv, &len) != 2) return false;
    *ids = Tokens(conv, len + 1);  // + the token the first decode step reads
    *stable = ids->size() - 1;
    return true;
  }
  [[nodiscard]] std::string TokenBytes(text::TokenId) const override { return "w "; }
  [[nodiscard]] bool IsStop(text::TokenId) const override { return false; }
  [[nodiscard]] bool HasTokenizer() const override { return true; }
  [[nodiscard]] double BytesPerToken() const override { return 1; }
  void Admitted(Id id, std::span<const std::int32_t>) override {
    std::lock_guard<std::mutex> lk(mu);
    admitted.insert(id);
  }
  bool Begin(Id id, std::span<const std::int32_t>, std::size_t, std::uint32_t, bool, bool, SpecDecodeStats*,
             std::size_t* reused, std::uint64_t* bound, std::string*) override {
    std::lock_guard<std::mutex> lk(mu);
    const auto mid = mid_of.at(id);
    const auto b = binds.count(mid) ? binds.at(mid) : 0;
    *reused = static_cast<std::size_t>(b);
    *bound = b ? static_cast<std::uint64_t>(conv_of.at(mid)) : 0;
    begun.insert(id);
    return true;
  }
  bool PrefillSlice(Id id, std::uint32_t, std::size_t* ran, bool* done, std::string*) override {
    std::lock_guard<std::mutex> lk(mu);
    const auto mid = mid_of.at(id);
    const auto it = prog.find(mid);
    *ran = it == prog.end() ? 0 : static_cast<std::size_t>(it->second.first);
    *done = it != prog.end() && it->second.second;
    if (it == prog.end()) *done = empty_slice.count(mid) > 0;  // the code's empty slice
    return true;
  }
  bool Step(Id, std::int32_t, std::uint32_t, const sampling::SamplingConfig&, sampling::SamplerState*,
            DecodeCycle* c, std::string*) override {
    c->tokens.assign(1, kWord);
    return true;
  }
  bool StepBatch(std::vector<BatchItem>& items, std::string*) override {
    for (auto& it : items) it.out = kWord;
    return true;
  }
  void Ran(Id, std::int32_t) override {}
  void Finish(Id id) override { Drop(id); }
  void Abandon(Id id) override { Drop(id); }
  std::optional<Scheduler::Retained> Park(Id id) override {
    std::lock_guard<std::mutex> lk(mu);
    begun.erase(id);
    const auto mid = mid_of.at(id);
    if (auto it = parks.find(mid); it != parks.end())
      return Scheduler::Retained{static_cast<Scheduler::Key>(conv_of.at(mid)), static_cast<std::uint64_t>(it->second),
                                 SharedTokens(conv_of.at(mid), it->second), true};
    return std::nullopt;
  }
  void Fenced(Id id) override {
    std::lock_guard<std::mutex> lk(mu);
    if (auto it = mid_of.find(id); it != mid_of.end()) fenced.insert(it->second);
  }
  void Evict(Scheduler::Key) override {}
  [[nodiscard]] bool HasLease(Id id) const override {
    std::lock_guard<std::mutex> lk(mu);
    return begun.count(id) > 0;
  }
  [[nodiscard]] std::uint64_t Generation() const override { return generation.load(); }
  [[nodiscard]] std::vector<Scheduler::Retained> Retained() const override {
    std::lock_guard<std::mutex> lk(mu);
    return held;
  }
  void SetPins(const std::set<std::uint64_t>&) override {}

  /// The model's retained prefixes (after an arrival or a unit end).
  void SetHeld(const Value& h) {
    std::vector<Scheduler::Retained> rs;
    for (const auto& [c, v] : h.members()) {
      const int len = static_cast<int>(v.number());
      if (len > 0)
        rs.push_back({static_cast<Scheduler::Key>(std::stoi(c)), static_cast<std::uint64_t>(len), SharedTokens(std::stoi(c), len),
                      true});
    }
    std::lock_guard<std::mutex> lk(mu);
    held = std::move(rs);
    ++generation;
  }

  mutable std::mutex mu;
  std::map<Id, std::int64_t> mid_of;           // code id -> model id
  std::map<std::int64_t, int> conv_of;         // model id -> conversation
  std::map<std::int64_t, int> binds;           // model id -> reused prompt at its first unit
  std::map<std::int64_t, int> parks;           // model id -> parked prefix (the fences before the next pick)
  std::map<std::int64_t, std::pair<int, bool>> prog;  // model id -> (prefilled, prefill done) at the unit's end
  std::set<std::int64_t> empty_slice;           // model ids whose next slice is the code's empty one
  std::set<Id> admitted, begun;
  std::set<std::int64_t> fenced;                // model ids the code fenced since the driver last took them
  std::vector<Scheduler::Retained> held;
  std::atomic<std::uint64_t> generation{1};

 private:
  void Drop(Id id) {
    std::lock_guard<std::mutex> lk(mu);
    begun.erase(id);
  }
};

/// Holds the engine thread before each Next and each unit, so the driver decides when it moves.
struct Gate {
  std::mutex mu;
  std::condition_variable cv;
  int next_permits = 0, boundary_permits = 0;
  bool at_next = false, free = false;
  std::optional<Scheduler::Unit> unit;  // picked, held until released
  bool release = false;
  std::uint64_t nexts = 0, boundaries = 0;

  bool BeforeNext() {
    std::unique_lock<std::mutex> lk(mu);
    at_next = true;
    cv.notify_all();
    cv.wait(lk, [&] { return free || next_permits > 0 || boundary_permits > 0; });
    at_next = false;
    if (!free && boundary_permits > 0) {
      --boundary_permits;
      ++boundaries;
      return false;  // a boundary first
    }
    if (!free) --next_permits;
    ++nexts;
    return true;
  }
  void BeforeUnit(const Scheduler::Unit& u) {
    std::unique_lock<std::mutex> lk(mu);
    if (free) return;
    unit = u;
    cv.notify_all();
    cv.wait(lk, [&] { return free || release; });
    release = false;
    unit.reset();
  }
  /// A boundary only (the model's fence step): the engine fences the requests known gone and comes back.
  bool Boundary() {
    std::unique_lock<std::mutex> lk(mu);
    if (!cv.wait_for(lk, std::chrono::seconds(10), [&] { return at_next; })) return false;
    const auto before = boundaries;
    ++boundary_permits;
    cv.notify_all();
    return cv.wait_for(lk, std::chrono::seconds(10), [&] { return at_next && boundaries > before; });
  }
  /// Lets one Next through; returns the unit it picked, or nullopt when the engine came back to Next idle.
  std::optional<Scheduler::Unit> Pick() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait_for(lk, std::chrono::seconds(10), [&] { return at_next; });
    const auto before = nexts;
    ++next_permits;
    cv.notify_all();
    cv.wait_for(lk, std::chrono::seconds(10), [&] { return unit.has_value() || (at_next && nexts > before); });
    return unit;
  }
  /// Releases the held unit and waits until the engine is back before Next (the unit is done and published).
  bool Release() {
    std::unique_lock<std::mutex> lk(mu);
    const auto before = nexts;
    release = true;
    cv.notify_all();
    return cv.wait_for(lk, std::chrono::seconds(10), [&] { return at_next && nexts == before && !unit; });
  }
  void Free() {
    std::lock_guard<std::mutex> lk(mu);
    free = true;
    cv.notify_all();
  }
};

struct Recording : GenerationObserver {
  void OnGeneration(const GenerationRecord& r) noexcept override {
    std::lock_guard<std::mutex> lk(mu);
    records.push_back(r);
  }
  std::mutex mu;
  std::vector<GenerationRecord> records;
};

/// One HTTP client: sends its request at once and reads in its own thread.
struct Client {
  int fd = -1;
  std::thread reader;
  std::mutex mu;
  std::string raw;
  std::atomic<bool> eof{false}, left{false};
  Client(int port, const std::string& request, bool gone_now) {
    fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<std::uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) throw std::runtime_error("connect");
    ::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
    if (gone_now) Leave();
    reader = std::thread([this] {
      char b[4096];
      for (ssize_t n; (n = ::recv(fd, b, sizeof b, 0)) > 0;) {
        std::lock_guard<std::mutex> lk(mu);
        raw.append(b, static_cast<std::size_t>(n));
      }
      eof = true;
    });
  }
  ~Client() {
    ::shutdown(fd, SHUT_RDWR);
    if (reader.joinable()) reader.join();
    ::close(fd);
  }
  void Leave() {
    left = true;
    ::shutdown(fd, SHUT_RDWR);
  }
  std::string Raw() {
    std::lock_guard<std::mutex> lk(mu);
    return raw;
  }
  bool HeadIn() { return Raw().find("\r\n\r\n") != std::string::npos; }
  int Status() {
    const auto r = Raw();
    return r.size() >= 12 ? std::stoi(r.substr(9, 3)) : 0;
  }
  std::string Header(const std::string& name) {
    const auto r = Raw();
    const auto head = r.substr(0, r.find("\r\n\r\n"));
    const auto at = head.find("\r\n" + name + ": ");
    if (at == std::string::npos) return "";
    const auto from = at + 4 + name.size();
    return head.substr(from, head.find("\r\n", from) - from);
  }
  /// The streamed text and whether it ended with [DONE].
  std::pair<std::string, bool> Stream() {
    const auto r = Raw();
    std::string text;
    bool done = false;
    for (std::size_t i = r.find("data: "); i != std::string::npos; i = r.find("data: ", i + 6)) {
      const auto end = r.find("\n", i);
      const std::string data = r.substr(i + 6, end - i - 6);
      if (data == "[DONE]") {
        done = true;
        continue;
      }
      std::string err;
      const auto v = halo::json::Parse(data, &err);
      if (!v) continue;
      if (const Value* c = v->find("choices"); c && !c->items().empty())
        if (const Value* t = c->items()[0].find("text")) text += t->str();
    }
    return {text, done};
  }
};

/// One trace through the stack.
class Replay {
 public:
  Replay() : server({.send_timeout_ms = 2000}) {
    ScheduledOptions so;
    so.sched = ReplayOptions();
    so.fixed_bytes = 0;
    so.now_ms = [this] { return now.load(); };
    so.before_next = [this] { return gate.BeforeNext(); };
    so.before_unit = [this](const Scheduler::Unit& u) { gate.BeforeUnit(u); };
    so.first_token_in_slice = false;
    auto b = std::make_unique<TraceBackend>();
    be = b.get();
    gen = std::make_unique<ScheduledGenerator>(std::move(b), 1u << 20, halo::engine::MtpDepthOptions{}, so);
    ServiceOptions o;
    o.observer = &observed;
    o.now_s = [] { return std::int64_t{1700000000}; };
    o.on_refused = [this](const std::string& why) {
      std::lock_guard<std::mutex> lk(observed.mu);
      refused.push_back(why);
    };
    service = std::make_unique<OpenAiService>(*gen, o);
    service->Register(server);
    std::string err;
    if (!server.Start(&err)) throw std::runtime_error(err);
  }
  ~Replay() {
    gate.Free();
    clients.clear();
    for (auto& t : deciders)
      if (t.joinable()) t.join();
    server.Stop();
    gen.reset();
  }

  /// Applies one event; returns a mismatch ("" when the code did what the model did).
  std::string Step(const Value& e) {
    const std::string kind = S(e, "e");
    now = I(e, "t");
    if (kind == "arrive") return Arrive(e);
    if (kind == "depart") return Depart(e);
    if (kind == "fence") {
      for (const auto& x : e.find("ids")->items()) fenced_model_.insert(static_cast<std::int64_t>(x.number()));
      {
        std::lock_guard<std::mutex> lk(be->mu);
        for (const auto& [k, v] : e.find("parks")->members()) be->parks[std::stoll(k)] = static_cast<int>(v.number());
      }
      // The model fences at a boundary as a step of its own; the engine runs one at the same time.
      return gate.Boundary() ? "" : "fence: the engine did not run a boundary";
    }
    if (kind == "pick") return Pick(e);
    if (kind == "end") return End(e);
    return "unknown event " + kind;
  }

  /// After the trace: the HTTP surface's outcomes (whole answers, records, tokens counted once).
  std::string Finish() {
    std::ostringstream why;
    why << CompareFences();
    // Whole answers for every client still there when its generation completed.
    for (const auto& [mid, c] : reader_of_) {
      const auto g = gen_of_.at(mid);
      if (!completed_.count(g) || c->left) continue;
      if (!Await([&] { return c->Stream().second; }))
        why << "request " << mid << ": its generation completed but the stream did not end; ";
      else if (const auto text = c->Stream().first; text.size() != 2 * static_cast<std::size_t>(decoded_[g]))
        why << "request " << mid << ": streamed " << text.size() / 2 << " tokens, the generation decoded "
            << decoded_[g] << "; ";
      whole_++;
    }
    // Every admitted reader is recorded once its client leaves: every client still waiting goes while the engine is
    // still held (nothing runs past the trace), the server learns it, then the engine is let go.
    for (auto& [mid, c] : reader_of_)
      if (!(completed_.count(gen_of_.at(mid)) && !c->left)) c->Leave();
    for (const auto& [mid, id] : cid_)
      if (!Await([&] {
            const auto v = gen->scheduler().Request(id);
            return !v || v->departed || v->decider || v->phase == Scheduler::Phase::kFinished ||
                   v->phase == Scheduler::Phase::kCancelled;
          }))
        why << "teardown: the server did not learn that request " << mid << " left; ";
    gate.Free();
    const std::size_t want = reader_of_.size();
    if (!Await([&] {
          std::lock_guard<std::mutex> lk(observed.mu);
          return observed.records.size() >= want;
        }, 10000))
      why << "observer: " << observed.records.size() << " records, " << want << " admitted readers; ";
    std::lock_guard<std::mutex> lk(observed.mu);
    std::uint64_t counted = 0, want_counted = 0;
    std::map<std::string, int> finishes;
    for (const auto& r : observed.records) {
      counted += r.completion_tokens;
      ++finishes[r.finish_reason];
    }
    for (const auto& g : delivered_) want_counted += static_cast<std::uint64_t>(decoded_[g]);
    if (counted != want_counted) {
      why << "completion tokens counted " << counted << ", generations delivered whole decoded " << want_counted << "; ";
      for (const auto& r : observed.records)
        why << "\n    record " << r.request_id << " finish " << r.finish_reason << " tokens " << r.completion_tokens
            << " input '" << r.input << "'";
      for (const auto& [m, c] : reader_of_)
        why << "\n    reader " << m << " gen " << gen_of_.at(m) << " left " << c->left << " decoded "
            << (gen_of_.at(m) >= 0 ? decoded_[gen_of_.at(m)] : -1) << " completed " << completed_.count(gen_of_.at(m))
            << " delivered " << delivered_.count(gen_of_.at(m)) << " streamed " << c->Stream().first.size() / 2;
    }
    const int got_whole = finishes["length"] + finishes["stop"];
    if (got_whole != static_cast<int>(whole_))
      why << "records of answered readers " << got_whole << ", whole streams " << whole_ << "; ";
    for (const auto& [f, n] : finishes)
      if (f != "length" && f != "stop" && f != "cancelled" && f != "detached") why << "a record finished " << f << "; ";
    return why.str();
  }

  std::uint64_t compared = 0, empty_slices = 0, heads = 0, whole_ = 0;
  std::map<std::string, int> outcomes;

 private:
  std::string Arrive(const Value& e) {
    const std::int64_t mid = I(e, "id");
    const int conv = static_cast<int>(I(e, "conv")), prompt = static_cast<int>(I(e, "prompt"));
    const std::string out = S(e, "out");
    const auto c0 = gen->scheduler().counters();
    std::set<Scheduler::Id> before;
    for (const auto& v : gen->scheduler().Requests()) before.insert(v.id);
    {
      std::lock_guard<std::mutex> lk(be->mu);
      be->conv_of[mid] = conv;
    }
    std::shared_ptr<Client> client;
    if (B(e, "dec")) {
      deciders.emplace_back([this] {
        std::string err;
        (void)gen->RunDecider([] {}, &err);
      });
    } else {
      const std::string body = "{\"prompt\": \"" + PromptText(conv, prompt) + "\", \"max_tokens\": " +
                               std::to_string(I(e, "max")) + ", \"stream\": true}";
      const std::string req = "POST /v1/completions HTTP/1.1\r\nConnection: close\r\nX-Halo-Principal: " + S(e, "p") +
                              "\r\nX-Halo-Chain-Id: " + S(e, "root") + "\r\nIdempotency-Key: " + std::to_string(conv) +
                              "/" + std::to_string(I(e, "turn")) + "\r\nContent-Length: " + std::to_string(body.size()) +
                              "\r\n\r\n" + body;
      client = std::make_shared<Client>(server.port(), req, !B(e, "conn"));
      clients.push_back(client);
    }
    // The code's outcome, from the scheduler's counters.
    std::string code;
    const bool decided = Await([&] {
      const auto c = gen->scheduler().counters();
      code = c.adopted > c0.adopted       ? "adopted"
             : c.replayed > c0.replayed   ? "replayed"
             : c.duplicates > c0.duplicates ? "duplicate"
             : c.memory429 > c0.memory429 ? "memory"
             : c.limit429 > c0.limit429   ? "limit"
             : c.admitted > c0.admitted   ? "queued"
                                          : "";
      return !code.empty();
    });
    std::ostringstream why;
    if (!decided) return "admission: no outcome within 5 s";
    ++outcomes[code];
    if (code != out) why << "admission: code " << code << ", model " << out << "; ";
    if (code == "queued") {
      Scheduler::Id id = 0;
      for (const auto& v : gen->scheduler().Requests())
        if (!before.count(v.id)) id = v.id;
      if (!id) return why.str() + "queued, but no new request";
      {
        std::lock_guard<std::mutex> lk(be->mu);
        be->mid_of[id] = mid;
      }
      cid_[mid] = id;
      prompt_[mid] = prompt;
      const auto v = gen->scheduler().Request(id);
      if (B(e, "dec") && (!v || v->prio != B(e, "prio")))
        why << "decider credit: code prio " << (v && v->prio) << ", model " << B(e, "prio") << "; ";
      if (!B(e, "dec") && v && v->departed == B(e, "conn"))
        why << "arrival: the server " << (v->departed ? "knew the client gone" : "thought it there") << "; ";
    }
    // The HTTP surface of a connected agent request.
    if (client && B(e, "conn")) {
      if (!Await([&] { return client->HeadIn(); })) return why.str() + "no response head";
      ++heads;
      const int st = client->Status();
      const std::string answer = client->Header("Halo-Answer"), ra = client->Header("Retry-After");
      if (code == "memory" || code == "limit") {
        if (st != 429) why << "a " << code << " refusal answered " << st << "; ";
        const int secs = ra.empty() ? 0 : std::atoi(ra.c_str());
        if (code == "memory" && secs != static_cast<int>(gen->scheduler().options().retry_after_memory_s))
          why << "memory 429 Retry-After '" << ra << "'; ";
        if (code == "limit" && (secs < 1 || secs > gen->scheduler().options().retry_after_cap_s))
          why << "limit 429 Retry-After '" << ra << "'; ";
        if (client->Raw().find(code == "memory" ? "insufficient_memory" : "too_many_requests") == std::string::npos)
          why << "429 body without its code; ";
        if (!answer.empty()) why << "a 429 with Halo-Answer " << answer << "; ";
      } else {
        const std::string want = code == "queued" ? "generated" : code == "duplicate" ? "" : code;
        if (st != 200) why << "admitted, answered " << st << "; ";
        if (answer != want) why << "Halo-Answer '" << answer << "', want '" << want << "'; ";
      }
    }
    if (client && code != "memory" && code != "limit") {
      reader_of_[mid] = client;
      // The generation it reads: its own, the adopted one's, or none (a dropped duplicate).
      gen_of_[mid] = code == "queued" ? mid : code == "duplicate" ? -1 : I(e, "target");
      // Adopted: the model's earlier client had left (unknown to the server: the trace has no departure for it); the
      // code hands its reader off (detached). Its socket goes now.
      if (code == "adopted")
        if (auto it = last_reader_.find(gen_of_[mid]); it != last_reader_.end()) reader_of_.at(it->second)->Leave();
      if (code == "queued" || code == "adopted") last_reader_[gen_of_[mid]] = mid;
      if (code == "queued") ++agent_queued_;
    }
    be->SetHeld(*e.find("held"));
    ++compared;
    return why.str();
  }

  std::string Depart(const Value& e) {
    const std::int64_t mid = I(e, "id");
    const auto it = cid_.find(mid);
    // The model's request is the generation; its connection is its last reader's (an adopter's, once adopted).
    if (auto r = last_reader_.find(mid); r != last_reader_.end()) reader_of_.at(r->second)->Leave();
    if (it == cid_.end()) return "";
    const auto id = it->second;
    if (!Await([&] {
          const auto v = gen->scheduler().Request(id);
          return !v || v->departed;
        }))
      return "depart: the server did not learn that request " + std::to_string(mid) + " left";
    return "";
  }

  std::string CompareFences() {
    {
      std::lock_guard<std::mutex> lk(be->mu);
      fenced_code_.insert(be->fenced.begin(), be->fenced.end());
      be->fenced.clear();
    }
    std::string why;
    if (fenced_code_ != fenced_model_)
      why = "fenced: code " + Set(fenced_code_) + ", model " + Set(fenced_model_) + "; ";
    fenced_code_.clear();
    fenced_model_.clear();
    return why;
  }

  std::string Pick(const Value& e) {
    std::ostringstream why;
    const Value* binds = e.find("binds");
    {
      std::lock_guard<std::mutex> lk(be->mu);
      be->prog.clear();
      be->empty_slice.clear();
      if (binds)
        for (const auto& [k, v] : binds->members()) be->binds[std::stoll(k)] = static_cast<int>(v.number());
    }
    auto u = gate.Pick();
    for (;;) {
      if (!u) return why.str() + "unit: code idle, model " + S(e, "kind");
      // The code's refinement: a request whose whole prompt is reused starts with an empty slice.
      if (u->kind == Scheduler::Kind::kSlice && binds && S(e, "kind") != "slice") {
        const auto mid = Mid(u->ids[0]);
        const Value* b = binds->find(std::to_string(mid));
        if (b && static_cast<int>(b->number()) >= prompt_.at(mid)) {
          {
            std::lock_guard<std::mutex> lk(be->mu);
            be->empty_slice.insert(mid);
          }
          if (!gate.Release()) return why.str() + "the empty slice did not end";
          {
            std::lock_guard<std::mutex> lk(be->mu);
            be->empty_slice.clear();
          }
          ++empty_slices;
          u = gate.Pick();
          continue;
        }
      }
      break;
    }
    why << CompareFences();
    {
      std::lock_guard<std::mutex> lk(be->mu);
      be->parks.clear();
    }
    std::set<std::int64_t> ids, want_ids;
    for (auto id : u->ids) ids.insert(Mid(id));
    for (const auto& x : e.find("ids") ? e.find("ids")->items() : Value::Array{})
      want_ids.insert(static_cast<std::int64_t>(x.number()));
    if (KindName(u->kind) != S(e, "kind") || ids != want_ids)
      why << "unit: code " << KindName(u->kind) << " " << Set(ids) << ", model " << S(e, "kind") << " " << Set(want_ids)
          << "; ";
    const auto& s = gen->scheduler();
    if (const Value* vr = e.find("vroot"))
      for (const auto& [q, v] : vr->members())
        if (s.vtime_root(Cls(q)) != static_cast<std::int64_t>(v.number()))
          why << "vtime root " << q << ": code " << s.vtime_root(Cls(q)) << ", model " << v.number() << "; ";
    if (const Value* vp = e.find("vprin"))
      for (const auto& [q, v] : vp->members())
        if (s.vtime_principal(Cls(q)) != static_cast<std::int64_t>(v.number()))
          why << "vtime principal " << q << ": code " << s.vtime_principal(Cls(q)) << ", model " << v.number() << "; ";
    held_unit_ = true;
    ++compared;
    return why.str();
  }

  std::string End(const Value& e) {
    if (!held_unit_) return "a unit end without a held unit";
    {
      std::lock_guard<std::mutex> lk(be->mu);
      be->prog.clear();
      for (const auto& p : e.find("prog")->items())
        be->prog[I(p, "id")] = {static_cast<int>(I(p, "pre")), B(p, "pdone")};
    }
    for (const auto& p : e.find("prog")->items()) {
      decoded_[I(p, "id")] += static_cast<int>(I(p, "dec"));
      if (B(p, "fin") && B(p, "comp") && S(e, "kind") != "decide") completed_.insert(I(p, "id"));
    }
    be->SetHeld(*e.find("held"));
    held_unit_ = false;
    if (!gate.Release()) return "the unit did not end";
    // Generations that completed with a reader still connected: delivered whole (counted once).
    for (const auto& p : e.find("prog")->items()) {
      const auto g = I(p, "id");
      if (!completed_.count(g) || S(e, "kind") == "decide") continue;
      for (const auto& [m, c] : reader_of_)
        if (gen_of_[m] == g && !c->left) {
          delivered_.insert(g);
          break;
        }
    }
    return "";
  }

  std::int64_t Mid(Scheduler::Id id) {
    std::lock_guard<std::mutex> lk(be->mu);
    const auto it = be->mid_of.find(id);
    return it == be->mid_of.end() ? -1 : it->second;
  }

 public:
  std::atomic<std::int64_t> now{0};
  Gate gate;
  TraceBackend* be = nullptr;
  std::unique_ptr<ScheduledGenerator> gen;
  Recording observed;
  std::vector<std::string> refused;
  HttpServer server;
  std::unique_ptr<OpenAiService> service;
  std::vector<std::shared_ptr<Client>> clients;
  std::vector<std::thread> deciders;

  /// Refused requests never reached the engine.
  std::string RefusedNeverRan() {
    std::lock_guard<std::mutex> lk(be->mu);
    return be->admitted.size() == agent_queued_ ? ""
                                                : "the engine saw " + std::to_string(be->admitted.size()) +
                                                      " agent admissions, the scheduler queued " +
                                                      std::to_string(agent_queued_) + "; ";
  }

 private:
  std::map<std::int64_t, Scheduler::Id> cid_;
  std::map<std::int64_t, int> prompt_;
  std::map<std::int64_t, std::shared_ptr<Client>> reader_of_;  // model request -> its client (admitted readers)
  std::map<std::int64_t, std::int64_t> gen_of_;                // model request -> the generation it reads
  std::map<std::int64_t, std::int64_t> last_reader_;
  std::map<std::int64_t, int> decoded_;
  std::set<std::int64_t> completed_, delivered_;
  std::size_t agent_queued_ = 0;
  std::set<std::int64_t> fenced_code_, fenced_model_;
  bool held_unit_ = false;
};

std::vector<std::filesystem::path> Traces() {
  std::vector<std::filesystem::path> out;
  const std::filesystem::path dir = std::getenv("HALO_SCHED_TRACE_DIR") ? std::getenv("HALO_SCHED_TRACE_DIR")
                                                                         : HALO_SERVER_TEST_DATA_DIR "/sched_traces";
  if (std::filesystem::exists(dir))
    for (const auto& f : std::filesystem::directory_iterator(dir))
      if (f.path().extension() == ".jsonl") out.push_back(f.path());
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

TEST(ServingReplay, ModelTracesThroughHttpAndTheScheduledGenerator) {
  const auto traces = Traces();
  ASSERT_FALSE(traces.empty()) << "no traces: spec/scheduler/replay.sh writes them";
  std::uint64_t events = 0, compared = 0, empty = 0, heads = 0, whole = 0;
  std::map<std::string, int> outcomes;
  for (const auto& path : traces) {
    std::ifstream in(path);
    Replay r;
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
      ++n;
      std::string err;
      const auto e = halo::json::Parse(line, &err);
      ASSERT_TRUE(e) << path << ":" << n << ": " << err;
      const std::string why = r.Step(*e);
      ASSERT_TRUE(why.empty()) << path.filename().string() << ":" << n << " " << line << "\n  " << why;
    }
    const std::string why = r.Finish() + r.RefusedNeverRan();
    ASSERT_TRUE(why.empty()) << path.filename().string() << " at the end\n  " << why;
    events += static_cast<std::uint64_t>(n);
    compared += r.compared;
    empty += r.empty_slices;
    heads += r.heads;
    whole += r.whole_;
    for (const auto& [k, v] : r.outcomes) outcomes[k] += v;
  }
  std::string mix;
  for (const auto& [k, v] : outcomes) mix += " " + k + " " + std::to_string(v);
  std::printf("served %zu traces, %llu events, %llu compared steps (%llu empty slices), %llu response heads checked, "
              "%llu whole streams; admissions:%s\n",
              traces.size(), static_cast<unsigned long long>(events), static_cast<unsigned long long>(compared),
              static_cast<unsigned long long>(empty), static_cast<unsigned long long>(heads),
              static_cast<unsigned long long>(whole), mix.c_str());
}
