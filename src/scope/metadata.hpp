#ifndef HALO_SCOPE_METADATA_HPP_
#define HALO_SCOPE_METADATA_HPP_

// Mutable halo-scope metadata in SQLite (WAL mode), where Langfuse uses
// Postgres. This spike holds prompts, their immutable versions and movable
// labels (`production`, `latest`, ...), and stored /v1/responses: ordered
// checksummed migrations, each applied in one transaction.

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "scope/migrate.hpp"

struct sqlite3;

namespace halo::scope {

struct PromptVersion {
  std::string name;
  std::int64_t version = 0;
  std::string template_text;
  std::string config_json;
};

/// Thread-safe: one connection, each method under a mutex.
class MetadataStore {
 public:
  /// Opens (creating if missing) the database at `path`, sets WAL mode and
  /// foreign keys, and migrates. Returns nullptr with `*err` on failure.
  static std::unique_ptr<MetadataStore> Open(const std::string& path, std::string* err);
  static std::unique_ptr<MetadataStore> Open(const std::string& path, const std::vector<Migration>& migrations,
                                             std::string* err);
  ~MetadataStore();

  static const std::vector<Migration>& Migrations();

  /// Appends a new immutable version of prompt `name` (creating the prompt)
  /// and points the `latest` label at it. Returns the version, or 0.
  std::int64_t CreatePromptVersion(std::string_view name, std::string_view template_text,
                                   std::string_view config_json, std::string* err);
  /// Points `label` of prompt `name` at an existing `version`.
  bool SetLabel(std::string_view name, std::string_view label, std::int64_t version, std::string* err);
  std::optional<PromptVersion> GetByLabel(std::string_view name, std::string_view label);
  std::optional<PromptVersion> GetByVersion(std::string_view name, std::int64_t version);
  /// The labels pointing at (name, version), sorted.
  std::vector<std::string> LabelsOf(std::string_view name, std::int64_t version);
  /// Stores a /v1/responses response (JSON texts), replacing one with the
  /// same id, and keeps only the newest `keep` (0: all). Thread-safe.
  bool PutResponse(std::string_view id, std::string_view conversation, std::string_view response, std::size_t keep,
                   std::string* err);
  /// (conversation, response) JSON texts. Thread-safe.
  std::optional<std::pair<std::string, std::string>> GetResponse(std::string_view id);
  /// PRAGMA journal_mode (expected "wal").
  std::string JournalMode();

 private:
  MetadataStore() = default;
  bool Exec(const char* sql, std::string* err);
  sqlite3* db_ = nullptr;
  std::mutex mu_;  // every public method may be called from any thread
};

}  // namespace halo::scope

#endif  // HALO_SCOPE_METADATA_HPP_
