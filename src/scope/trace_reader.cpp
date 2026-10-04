#include "scope/trace_reader.hpp"

#include <charconv>
#include <sstream>

#include "server/openai_response.hpp"

namespace halo::scope {

namespace {

using json::Value;

std::string PercentDecode(std::string_view s) {
  std::string out;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '+') {
      out += ' ';
    } else if (s[i] == '%' && i + 2 < s.size()) {
      int v = 0;
      const auto [p, ec] = std::from_chars(s.data() + i + 1, s.data() + i + 3, v, 16);
      if (ec == std::errc() && p == s.data() + i + 3) {
        out += static_cast<char>(v);
        i += 2;
      } else {
        out += s[i];
      }
    } else {
      out += s[i];
    }
  }
  return out;
}

std::string QueryParam(const std::string& query, std::string_view key) {
  std::size_t i = 0;
  while (i <= query.size()) {
    const auto j = std::min(query.find('&', i), query.size());
    const std::string_view kv(query.data() + i, j - i);
    const auto eq = kv.find('=');
    if (PercentDecode(kv.substr(0, eq)) == key) return eq == std::string_view::npos ? "" : PercentDecode(kv.substr(eq + 1));
    i = j + 1;
  }
  return {};
}

server::HttpResponse Error(int status, const std::string& message) {
  Value err = Value::MakeObject();
  err.set("message", Value::String(message));
  Value body = Value::MakeObject();
  body.set("error", std::move(err));
  return server::HttpResponse::Json(status, server::CompactJson(body));
}

}  // namespace

std::optional<Value> TraceReader::Rows(const std::string& sql, std::string* error) {
  Conn::Result r;
  {
    std::lock_guard<std::mutex> lk(mu_);
    r = conn_->Query(sql, "JSONEachRow");
  }
  if (!r.ok) {
    *error = r.error;
    return std::nullopt;
  }
  Value rows = Value::MakeArray();
  std::istringstream in(r.data);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto v = json::Parse(line, error);
    if (!v) return std::nullopt;
    rows.push(std::move(*v));
  }
  return rows;
}

std::optional<Value> TraceReader::ListTraces(const std::string& session_id, std::uint32_t limit, std::string* error) {
  std::string sql =
      "SELECT trace_id, toUnixTimestamp64Nano(min(start_time)) AS start_ns, "
      "toUnixTimestamp64Nano(max(end_time)) AS end_ns, count() AS observations, "
      "sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens, "
      "anyIf(session_id, session_id != '') AS session_id "
      "FROM halo_scope.llm_observations WHERE project = " + SqlQuote(project_);
  if (!session_id.empty()) sql += " AND session_id = " + SqlQuote(session_id);
  // The output names reuse column names; WHERE must see the columns.
  sql += " GROUP BY trace_id ORDER BY start_ns DESC, trace_id LIMIT " + std::to_string(limit) +
         " SETTINGS prefer_column_name_to_alias = 1";
  return Rows(sql, error);
}

std::optional<Value> TraceReader::Trace(const std::string& trace_id, std::string* error) {
  const std::string p = SqlQuote(project_), t = SqlQuote(trace_id);
  auto observations = Rows(
      "SELECT span_id, parent_span_id, type, name, toUnixTimestamp64Nano(start_time) AS start_ns, "
      "toUnixTimestamp64Nano(end_time) AS end_ns, toUnixTimestamp64Nano(completion_start_time) AS first_text_ns, "
      "status_code, model, session_id, input_tokens, output_tokens, cached_tokens, tool_name, tool_call_ids, "
      "finish_reason, tpot_p50_us, tpot_p90_us, tags, arrayMap(x -> lower(hex(x)), input_refs) AS input_refs, "
      "arrayMap(x -> lower(hex(x)), output_refs) AS output_refs "
      "FROM halo_scope.llm_observations WHERE project = " + p + " AND trace_id = " + t +
          " ORDER BY start_time, span_id",
      error);
  if (!observations) return std::nullopt;
  auto scores = Rows(
      "SELECT score_id, observation_id, name, source, data_type, value, string_value, comment, "
      "toUnixTimestamp64Nano(event_time) AS event_ns FROM halo_scope.llm_scores WHERE project = " + p +
          " AND trace_id = " + t + " AND kind = 'assert' AND score_id NOT IN "
          "(SELECT supersedes FROM halo_scope.llm_scores WHERE project = " + p + " AND supersedes != '') "
          "ORDER BY event_time, score_id",
      error);
  if (!scores) return std::nullopt;
  auto context = Rows(
      "SELECT argMaxIf(session_id, start_time, session_id != '') AS session_id, "
      "argMaxIf(user_id, start_time, user_id != '') AS user_id, argMaxIf(name, start_time, name != '') AS name, "
      "argMaxIf(release, start_time, release != '') AS release, arraySort(groupUniqArrayArray(tags)) AS tags "
      "FROM halo_scope.llm_traces WHERE project = " + p + " AND trace_id = " + t,
      error);
  if (!context) return std::nullopt;
  auto metadata = Rows(
      "SELECT key, argMax(value, start_time) AS value FROM halo_scope.llm_traces "
      "ARRAY JOIN mapKeys(metadata) AS key, mapValues(metadata) AS value WHERE project = " + p +
          " AND trace_id = " + t + " GROUP BY key ORDER BY key",
      error);
  if (!metadata) return std::nullopt;
  Value out = Value::MakeObject();
  out.set("trace_id", Value::String(trace_id));
  out.set("context", context->items().empty() ? Value::MakeObject() : context->items()[0]);
  Value meta = Value::MakeObject();
  for (const auto& row : metadata->items()) meta.set(row.find("key")->str(), *row.find("value"));
  out.set("metadata", std::move(meta));
  out.set("observations", std::move(*observations));
  out.set("scores", std::move(*scores));
  return out;
}

void TraceReader::Register(server::HttpServer& srv) {
  srv.Route("GET", "/v1/scope/traces", [this](const server::HttpRequest& r) {
    std::uint32_t limit = 50;
    if (const std::string l = QueryParam(r.query, "limit"); !l.empty()) {
      const auto [p, ec] = std::from_chars(l.data(), l.data() + l.size(), limit);
      if (ec != std::errc() || p != l.data() + l.size() || limit == 0 || limit > 1000)
        return Error(400, "limit must be 1..1000");
    }
    std::string error;
    auto rows = ListTraces(QueryParam(r.query, "session_id"), limit, &error);
    if (!rows) return Error(500, error);
    Value body = Value::MakeObject();
    body.set("data", std::move(*rows));
    return server::HttpResponse::Json(200, server::CompactJson(body));
  });
  srv.RoutePrefix("GET", "/v1/scope/traces/", [this](const server::HttpRequest& r) {
    const std::string id = PercentDecode(r.path.substr(std::string_view("/v1/scope/traces/").size()));
    if (id.empty() || id.find('/') != std::string::npos) return Error(404, "no such trace route");
    std::string error;
    auto trace = Trace(id, &error);
    if (!trace) return Error(500, error);
    if (trace->find("observations")->items().empty() && trace->find("scores")->items().empty())
      return Error(404, "no trace " + id);
    return server::HttpResponse::Json(200, server::CompactJson(*trace));
  });
}

}  // namespace halo::scope
