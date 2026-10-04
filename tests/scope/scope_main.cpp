// gtest main for the halo-scope tests: closes the shared chDB connections
// before the process exits (see scope_env.hpp). Linked ahead of
// GTest::gtest_main, whose main() is then not pulled from the archive.

#include <gtest/gtest.h>

#include "scope_env.hpp"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  halo::scope::testing::Shutdown();
  return rc;
}
