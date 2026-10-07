#ifndef HALO_SERVER_RESPONSE_STORE_HPP_
#define HALO_SERVER_RESPONSE_STORE_HPP_

// Where /v1/responses keeps stored responses (store: true) for
// previous_response_id and GET /v1/responses/{id}: the conversation through
// the response's own turn (template messages) and the response object.
// In memory by default; halo-scope's SQLite metadata store persists them
// across restarts (scope::SqliteResponseStore).

#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>

#include "core/json.hpp"

namespace halo::server {

struct StoredResponse {
  json::Value conversation;  // template messages, through this response's turn
  json::Value response;      // the response object as returned
};

class ResponseStore {
 public:
  virtual ~ResponseStore() = default;
  /// Thread-safe. A put of an existing id replaces it. Returns false, with
  /// the reason in *error when given, only when nothing was written (an
  /// implementation settles its own unknowns first, e.g. a failed commit is
  /// rolled back): the service then answers the request with an error.
  virtual bool Put(const std::string& id, const StoredResponse& r, std::string* error = nullptr) = 0;
  virtual std::optional<StoredResponse> Get(const std::string& id) = 0;
};

/// Keeps the newest `capacity` responses.
class InMemoryResponseStore : public ResponseStore {
 public:
  explicit InMemoryResponseStore(std::size_t capacity = 1000) : capacity_(capacity) {}
  bool Put(const std::string& id, const StoredResponse& r, std::string* = nullptr) override {
    std::lock_guard<std::mutex> lk(mu_);
    if (map_.insert_or_assign(id, r).second) order_.push_back(id);
    while (order_.size() > capacity_) {
      map_.erase(order_.front());
      order_.pop_front();
    }
    return true;
  }
  std::optional<StoredResponse> Get(const std::string& id) override {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = map_.find(id);
    if (it == map_.end()) return std::nullopt;
    return it->second;
  }

 private:
  const std::size_t capacity_;
  std::mutex mu_;
  std::map<std::string, StoredResponse> map_;
  std::deque<std::string> order_;
};

}  // namespace halo::server

#endif  // HALO_SERVER_RESPONSE_STORE_HPP_
