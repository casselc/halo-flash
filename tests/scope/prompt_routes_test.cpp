// Prompt management over HTTP: versions are appended and immutable, latest
// follows the newest, other labels move only when set, lookups by label or
// version, and bad requests are refused.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <string>

#include "core/json.hpp"
#include "scope/metadata.hpp"
#include "scope/prompt_routes.hpp"
#include "scope_env.hpp"
#include "server/http.hpp"

using namespace halo;
using namespace halo::scope;

namespace {

std::pair<int, json::Value> Call(int port, const std::string& method, const std::string& target,
                                 const std::string& body = "") {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) throw std::runtime_error("connect");
  const std::string req = method + " " + target + " HTTP/1.1\r\nConnection: close\r\nContent-Length: " +
                          std::to_string(body.size()) + "\r\n\r\n" + body;
  ::send(fd, req.data(), req.size(), MSG_NOSIGNAL);
  std::string raw;
  char b[4096];
  for (ssize_t n; (n = ::recv(fd, b, sizeof b, 0)) > 0;) raw.append(b, static_cast<std::size_t>(n));
  ::close(fd);
  std::string err;
  auto v = json::Parse(raw.substr(raw.find("\r\n\r\n") + 4), &err);
  return {std::stoi(raw.substr(9, 3)), v ? *v : json::Value()};
}

}  // namespace

TEST(ScopePrompts, VersionsLabelsAndLookups) {
  std::string err;
  auto store = MetadataStore::Open(halo::scope::testing::TempDir("halo-scope-prompts") + "/meta.db", &err);
  ASSERT_TRUE(store) << err;
  server::HttpServer srv({});
  RegisterPromptRoutes(srv, *store);
  ASSERT_TRUE(srv.Start(&err)) << err;
  const int port = srv.port();
  auto [s1, v1] = Call(port, "POST", "/v1/scope/prompts", R"({"name": "triage", "template": "Classify: {{text}}", "config": {"temperature": 0.2}})");
  ASSERT_EQ(s1, 200);
  EXPECT_EQ(v1.find("version")->number(), 1);
  EXPECT_EQ(Call(port, "POST", "/v1/scope/prompts/triage/labels", R"({"label": "production", "version": 1})").first, 200);
  auto [s2, v2] = Call(port, "POST", "/v1/scope/prompts", R"({"name": "triage", "template": "Classify carefully: {{text}}"})");
  ASSERT_EQ(s2, 200);
  EXPECT_EQ(v2.find("version")->number(), 2);
  // latest follows; production stays until moved.
  auto [g1, latest] = Call(port, "GET", "/v1/scope/prompts/triage");
  ASSERT_EQ(g1, 200);
  EXPECT_EQ(latest.find("version")->number(), 2);
  EXPECT_EQ(latest.find("template")->str(), "Classify carefully: {{text}}");
  auto [g2, prod] = Call(port, "GET", "/v1/scope/prompts/triage?label=production");
  ASSERT_EQ(g2, 200);
  EXPECT_EQ(prod.find("version")->number(), 1);
  EXPECT_EQ(prod.find("config")->find("temperature")->number(), 0.2);
  EXPECT_EQ(prod.find("labels")->items()[0].str(), "production");
  EXPECT_EQ(Call(port, "POST", "/v1/scope/prompts/triage/labels", R"({"label": "production", "version": 2})").first, 200);
  EXPECT_EQ(Call(port, "GET", "/v1/scope/prompts/triage?label=production").second.find("version")->number(), 2);
  EXPECT_EQ(Call(port, "GET", "/v1/scope/prompts/triage?version=1").second.find("template")->str(), "Classify: {{text}}");
  // Refusals.
  EXPECT_EQ(Call(port, "GET", "/v1/scope/prompts/nope").first, 404);
  EXPECT_EQ(Call(port, "GET", "/v1/scope/prompts/triage?version=0").first, 400);
  EXPECT_EQ(Call(port, "GET", "/v1/scope/prompts/triage?version=9").first, 404);
  EXPECT_EQ(Call(port, "POST", "/v1/scope/prompts/triage/labels", R"({"label": "x", "version": 9})").first, 400);
  EXPECT_EQ(Call(port, "POST", "/v1/scope/prompts", R"({"name": "a/b", "template": "x"})").first, 400);
  EXPECT_EQ(Call(port, "POST", "/v1/scope/prompts", R"({"name": "t", "template": 3})").first, 400);
  srv.Stop();
}
