// halo_scope_eval: halo-scope datasets, evaluators and experiments from the
// shell, over a halo_serve --scope-dir server's /v1/scope API (the server
// owns the databases; see src/scope/eval_routes.hpp).
//
//   halo_scope_eval [--server http://127.0.0.1:8080] [--api-key KEY] COMMAND ...
//     prompt create NAME TEMPLATE|@FILE [CONFIG_JSON]
//     dataset create NAME [DESCRIPTION]
//     dataset import NAME FILE.jsonl
//     dataset freeze NAME
//     dataset show NAME
//     evaluator create NAME KIND CONFIG_JSON|@FILE
//     run NAME DATASET TARGET_JSON|@FILE EVALUATORS [--version N] [--concurrency N] [--max-attempts N]
//         creates (or resumes) the run and follows its progress to the end,
//         one line per change; exits 0 when complete, 3 when items failed
//     show RUN            the run, progress and per-score summary
//     items RUN           every item with output, trace id and scores
//     compare RUN_A RUN_B SCORE [--items]
//         paired per-item deltas (b - a): n, mean +- SE, wins/losses/ties

#include <cctype>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/json.hpp"
#include "scope/http_client.hpp"

using namespace halo;
using json::Value;
using Kind = json::Value::Kind;

namespace {

std::string g_server = "http://127.0.0.1:8080";
scope::HttpHeaders g_headers;

std::string ReadArg(const std::string& a) {
  if (a.empty() || a[0] != '@') return a;
  std::ifstream f(a.substr(1), std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", a.c_str() + 1);
    std::exit(2);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string Escape(const std::string& s) {
  std::string out;
  for (unsigned char c : s) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') out += static_cast<char>(c);
    else {
      char b[4];
      std::snprintf(b, sizeof b, "%%%02X", c);
      out += b;
    }
  }
  return out;
}

Value Call(const std::string& method, const std::string& path, const std::string& body = "", int* status = nullptr) {
  const auto r = scope::HttpRequestJson(method, g_server, path, body, g_headers, 600000);
  if (!r.transport_ok) {
    std::fprintf(stderr, "%s %s: %s\n", method.c_str(), path.c_str(), r.error.c_str());
    std::exit(1);
  }
  std::string err;
  auto v = json::Parse(r.body, &err);
  if (status) *status = r.status;
  if (r.status >= 300 && !status) {
    std::fprintf(stderr, "%s %s: HTTP %d %s\n", method.c_str(), path.c_str(), r.status, r.body.c_str());
    std::exit(1);
  }
  return v ? *v : Value();
}

void Print(const Value& v) { std::printf("%s\n", json::DumpPython(v, 2).c_str()); }

std::string Fmt(const Value* v, const char* fmt = "%.4f") {
  if (!v || v->is(Kind::kNull)) return "-";
  char b[64];
  std::snprintf(b, sizeof b, fmt, v->number());
  return b;
}

void PrintSummary(const Value& run) {
  if (const Value* scores = run.find("scores"))
    for (const auto& s : scores->items())
      std::printf("  %-24s %-11s n=%-4s mean=%s se=%s errors=%s\n", s.find("name")->str().c_str(),
                  s.find("data_type")->str().c_str(), s.find("n")->int_digits().c_str(), Fmt(s.find("mean")).c_str(),
                  Fmt(s.find("se")).c_str(), s.find("errors")->int_digits().c_str());
}

int Usage() {
  std::fprintf(stderr, "see the header of tools/scope_eval.cpp for usage\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--server" && i + 1 < argc) g_server = argv[++i];
    else if (a == "--api-key" && i + 1 < argc) g_headers.emplace_back("Authorization", std::string("Bearer ") + argv[++i]);
    else args.push_back(a);
  }
  if (args.empty()) return Usage();
  const std::string cmd = args[0];
  auto obj = [](std::initializer_list<std::pair<const char*, Value>> kv) {
    Value o = Value::MakeObject();
    for (const auto& [k, v] : kv) o.set(k, v);
    return json::DumpPython(o);
  };
  std::string err;
  if (cmd == "prompt" && args.size() >= 4 && args[1] == "create") {
    Value body = Value::MakeObject();
    body.set("name", Value::String(args[2]));
    body.set("template", Value::String(ReadArg(args[3])));
    if (args.size() > 4) {
      auto c = json::Parse(ReadArg(args[4]), &err);
      if (!c) return std::fprintf(stderr, "config: %s\n", err.c_str()), 2;
      body.set("config", *c);
    }
    Print(Call("POST", "/v1/scope/prompts", json::DumpPython(body)));
    return 0;
  }
  if (cmd == "dataset" && args.size() >= 3) {
    const std::string& name = args[2];
    if (args[1] == "create")
      Print(Call("POST", "/v1/scope/datasets",
                 obj({{"name", Value::String(name)}, {"description", Value::String(args.size() > 3 ? args[3] : "")}})));
    else if (args[1] == "import" && args.size() == 4)
      Print(Call("POST", "/v1/scope/datasets/" + Escape(name) + "/import", ReadArg("@" + args[3])));
    else if (args[1] == "freeze") Print(Call("POST", "/v1/scope/datasets/" + Escape(name) + "/freeze", "{}"));
    else if (args[1] == "show") Print(Call("GET", "/v1/scope/datasets/" + Escape(name)));
    else return Usage();
    return 0;
  }
  if (cmd == "evaluator" && args.size() == 5 && args[1] == "create") {
    auto c = json::Parse(ReadArg(args[4]), &err);
    if (!c) return std::fprintf(stderr, "config: %s\n", err.c_str()), 2;
    Value body = Value::MakeObject();
    body.set("name", Value::String(args[2]));
    body.set("kind", Value::String(args[3]));
    body.set("config", *c);
    Print(Call("POST", "/v1/scope/evaluators", json::DumpPython(body)));
    return 0;
  }
  if (cmd == "run" && args.size() >= 5) {
    Value body = Value::MakeObject();
    body.set("name", Value::String(args[1]));
    body.set("dataset", Value::String(args[2]));
    auto target = json::Parse(ReadArg(args[3]), &err);
    if (!target) return std::fprintf(stderr, "target: %s\n", err.c_str()), 2;
    body.set("target", *target);
    Value evs = Value::MakeArray();
    std::stringstream ss(args[4]);
    for (std::string e; std::getline(ss, e, ',');)
      if (!e.empty()) evs.push(Value::String(e));
    body.set("evaluators", std::move(evs));
    for (std::size_t i = 5; i + 1 < args.size(); i += 2) {
      const std::int64_t n = std::atoll(args[i + 1].c_str());
      if (args[i] == "--version") body.set("dataset_version", Value::Int(n));
      else if (args[i] == "--concurrency") body.set("concurrency", Value::Int(n));
      else if (args[i] == "--max-attempts") body.set("max_attempts", Value::Int(n));
      else return Usage();
    }
    Value run = Call("POST", "/v1/scope/runs", json::DumpPython(body));
    const std::string id = run.find("id")->str();
    std::printf("run %s (%s) %s on %s v%s, binding %s\n", id.c_str(), args[1].c_str(),
                run.find("resumed") && run.find("resumed")->boolean() ? "resumed" : "created", args[2].c_str(),
                run.find("dataset_version")->int_digits().c_str(), run.find("binding")->str().c_str());
    std::string last;
    while (true) {
      Value r = Call("GET", "/v1/scope/runs/" + id);
      const Value& p = *r.find("progress");
      const std::string line = "done " + p.find("done")->int_digits() + "/" + p.find("total")->int_digits() +
                               " running " + p.find("running")->int_digits() + " failed " +
                               std::to_string(static_cast<long long>(p.find("error")->number() + p.find("exhausted")->number())) +
                               " status " + r.find("status")->str();
      if (line != last) {
        const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char ts[32];
        std::strftime(ts, sizeof ts, "%H:%M:%S", std::localtime(&t));
        std::printf("%s %s\n", ts, line.c_str());
        std::fflush(stdout);
        last = line;
      }
      if (!r.find("active")->boolean() && r.find("status")->str() != "running" && r.find("status")->str() != "pending") {
        PrintSummary(r);
        return r.find("status")->str() == "completed" ? 0 : 3;
      }
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
  }
  if (cmd == "show" && args.size() == 2) {
    Value r = Call("GET", "/v1/scope/runs/" + Escape(args[1]));
    Print(r);
    return 0;
  }
  if (cmd == "items" && args.size() == 2) {
    Print(Call("GET", "/v1/scope/runs/" + Escape(args[1]) + "/items"));
    return 0;
  }
  if (cmd == "compare" && args.size() >= 4) {
    const bool items = args.size() > 4 && args[4] == "--items";
    Value c = Call("GET", "/v1/scope/compare?a=" + Escape(args[1]) + "&b=" + Escape(args[2]) + "&score=" +
                              Escape(args[3]) + (items ? "&items=1" : ""));
    std::printf("score %s: b (%s) - a (%s), paired over n=%s items (unpaired: a %s, b %s)\n", args[3].c_str(),
                c.find("b_name")->str().c_str(), c.find("a_name")->str().c_str(), c.find("n")->int_digits().c_str(),
                c.find("only_a")->int_digits().c_str(), c.find("only_b")->int_digits().c_str());
    std::printf("  mean a %s, mean b %s, delta %s +- %s (SE), wins %s losses %s ties %s\n",
                Fmt(c.find("mean_a")).c_str(), Fmt(c.find("mean_b")).c_str(), Fmt(c.find("mean_delta")).c_str(),
                Fmt(c.find("se_delta")).c_str(), c.find("wins")->int_digits().c_str(),
                c.find("losses")->int_digits().c_str(), c.find("ties")->int_digits().c_str());
    if (items)
      for (const auto& d : c.find("items")->items())
        std::printf("  %-12s a %s b %s delta %s\n", d.find("item_id")->str().c_str(), Fmt(d.find("a"), "%.3f").c_str(),
                    Fmt(d.find("b"), "%.3f").c_str(), Fmt(d.find("delta"), "%+.3f").c_str());
    return 0;
  }
  return Usage();
}
