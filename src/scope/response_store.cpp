#include "scope/response_store.hpp"

#include <cstdio>

#include "server/openai_response.hpp"

namespace halo::scope {

bool SqliteResponseStore::Put(const std::string& id, const server::StoredResponse& r, std::string* error) {
  std::string err;
  if (store_.PutResponse(id, server::CompactJson(r.conversation), server::CompactJson(r.response), keep_, &err))
    return true;
  std::fprintf(stderr, "halo-scope: storing response %s failed: %s\n", id.c_str(), err.c_str());
  if (error) *error = "storing the response failed: " + err;
  return false;
}

std::optional<server::StoredResponse> SqliteResponseStore::Get(const std::string& id) {
  const auto row = store_.GetResponse(id);
  if (!row) return std::nullopt;
  std::string err;
  auto conversation = json::Parse(row->first, &err);
  auto response = json::Parse(row->second, &err);
  if (!conversation || !response) return std::nullopt;  // unreadable rows count as missing
  return server::StoredResponse{std::move(*conversation), std::move(*response)};
}

}  // namespace halo::scope
