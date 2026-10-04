// halo-scope metadata store (SQLite, WAL): migrations, prompt versions and
// labels, against a model.

#include <hegel/gtest.h>
#include <stdlib.h>
#include <unistd.h>

#include <cstdlib>
#include <map>
#include <string>

#include "scope/metadata.hpp"
#include "scope_env.hpp"

namespace gs = hegel::generators;
using namespace halo::scope;

namespace {

std::string TempDb() {
  return halo::scope::testing::TempDir("halo-scope-meta") + "/meta.db";
}

}  // namespace

TEST(ScopeMetadata, OpensInWalModeAndMigratesOnce) {
  const auto path = TempDb();
  std::string err;
  {
    auto s = MetadataStore::Open(path, &err);
    ASSERT_TRUE(s) << err;
    EXPECT_EQ(s->JournalMode(), "wal");
  }
  auto again = MetadataStore::Open(path, &err);
  ASSERT_TRUE(again) << err;

  auto edited = MetadataStore::Migrations();
  edited[0].statements[0] = "CREATE TABLE prompts (id INTEGER PRIMARY KEY)";
  EXPECT_FALSE(MetadataStore::Open(path, edited, &err));
  EXPECT_NE(err.find("checksum mismatch"), std::string::npos) << err;

  // A failing migration leaves nothing behind: it runs in one transaction.
  auto extended = MetadataStore::Migrations();
  extended.push_back({2, "broken", {"CREATE TABLE t2 (a)", "THIS IS NOT SQL"}});
  EXPECT_FALSE(MetadataStore::Open(path, extended, &err));
  extended.back().statements = {"CREATE TABLE t2 (a)"};
  auto fixed = MetadataStore::Open(path, extended, &err);
  EXPECT_TRUE(fixed) << err;  // would fail with "table t2 already exists" had the first attempt leaked
}

TEST(ScopeMetadata, PromptVersionsAndLabelsMatchModel) {
  hegel::test(
      [](hegel::TestCase& tc) {
        std::string err;
        auto s = MetadataStore::Open(TempDb(), &err);
        if (!s) throw std::runtime_error(err);
        // model: name -> versions' templates; (name, label) -> version
        std::map<std::string, std::vector<std::string>> versions;
        std::map<std::pair<std::string, std::string>, std::int64_t> labels;
        const auto steps = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 30}));
        for (int i = 0; i < steps; ++i) {
          const auto name = tc.draw(gs::sampled_from<std::string>({"sys", "judge", "é'\"", ""}));
          if (tc.draw(gs::booleans())) {
            const auto tmpl = tc.draw(gs::text({.max_size = 30}));
            const auto v = s->CreatePromptVersion(name, tmpl, "", &err);
            auto& vs = versions[name];
            vs.push_back(tmpl);
            if (v != static_cast<std::int64_t>(vs.size())) throw std::runtime_error("version " + std::to_string(v) + ": " + err);
            labels[{name, "latest"}] = v;
          } else {
            const auto label = tc.draw(gs::sampled_from<std::string>({"production", "staging", "latest", ""}));
            const auto v = tc.draw(gs::integers<std::int64_t>({.min_value = 0, .max_value = 4}));
            const bool ok = s->SetLabel(name, label, v, &err);
            const bool expect = !label.empty() && v >= 1 && versions.count(name) &&
                                v <= static_cast<std::int64_t>(versions[name].size());
            if (ok != expect) throw std::runtime_error("SetLabel(" + label + ", " + std::to_string(v) + ") = " +
                                                       std::to_string(ok) + ": " + err);
            if (ok) labels[{name, label}] = v;
          }
        }
        for (const auto& [k, v] : labels) {
          const auto got = s->GetByLabel(k.first, k.second);
          if (!got || got->version != v || got->template_text != versions[k.first][v - 1])
            throw std::runtime_error("label " + k.second + " of " + k.first + " resolves wrong");
        }
        for (const auto& [name, vs] : versions)
          for (std::size_t i = 0; i < vs.size(); ++i) {
            const auto got = s->GetByVersion(name, static_cast<std::int64_t>(i + 1));
            if (!got || got->template_text != vs[i]) throw std::runtime_error("version content changed");
          }
      },
      {.test_cases = 50});
}
