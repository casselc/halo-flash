#include "scope/migrate.hpp"

#include <cstdio>
#include <string_view>

namespace halo::scope {

std::string MigrationChecksum(const Migration& m) {
  std::uint64_t h = 14695981039346656037ull;
  auto mix = [&h](std::string_view s) {
    for (unsigned char c : s) {
      h ^= c;
      h *= 1099511628211ull;
    }
    h ^= 0xff;  // separator: ("ab","c") and ("a","bc") differ
    h *= 1099511628211ull;
  };
  mix(m.name);
  for (const char* s : m.statements) mix(s);
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
  return buf;
}

int PlanMigrations(const std::vector<Migration>& known, const std::vector<AppliedMigration>& applied,
                   std::string* err) {
  if (applied.size() > known.size()) {
    *err = "database has " + std::to_string(applied.size()) + " migrations; this build knows " +
           std::to_string(known.size()) + " (database is newer than the binary)";
    return -1;
  }
  for (std::size_t i = 0; i < applied.size(); ++i) {
    const auto& a = applied[i];
    const auto& k = known[i];
    if (a.version != k.version) {
      *err = "migration order mismatch at position " + std::to_string(i) + ": database has version " +
             std::to_string(a.version) + ", build expects " + std::to_string(k.version);
      return -1;
    }
    const auto sum = MigrationChecksum(k);
    if (a.checksum != sum || a.name != k.name) {
      *err = "migration " + std::to_string(k.version) + " (" + k.name + ") checksum mismatch: database " +
             a.checksum + ", build " + sum + " (an applied migration was edited)";
      return -1;
    }
  }
  return static_cast<int>(applied.size());
}

}  // namespace halo::scope
