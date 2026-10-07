#ifndef HALO_SCOPE_RESPONSE_STORE_HPP_
#define HALO_SCOPE_RESPONSE_STORE_HPP_

// The server's stored /v1/responses in halo-scope's SQLite metadata store,
// so previous_response_id and GET /v1/responses/{id} survive restarts.

#include "scope/metadata.hpp"
#include "server/response_store.hpp"

namespace halo::scope {

class SqliteResponseStore : public server::ResponseStore {
 public:
  /// Keeps the newest `keep` responses (0: all). `store` must outlive this.
  SqliteResponseStore(MetadataStore& store, std::size_t keep) : store_(store), keep_(keep) {}
  bool Put(const std::string& id, const server::StoredResponse& r, std::string* error = nullptr) override;
  std::optional<server::StoredResponse> Get(const std::string& id) override;

 private:
  MetadataStore& store_;
  const std::size_t keep_;
};

}  // namespace halo::scope

#endif  // HALO_SCOPE_RESPONSE_STORE_HPP_
