// halo_serve's KV-kind resolution (server/kv_choice.hpp): an explicit --kv-f16 wins, otherwise fp16-only KV
// strictly above --kv-f16-above (default 150000), and --kv-f16-above 0 turns the automatic choice off.

#include <gtest/gtest.h>

#include "server/kv_choice.hpp"

using namespace halo::server;

namespace {
KvChoice Resolve(int explicit_kv, std::uint32_t ctx, std::uint32_t above = KvChoiceOptions{}.f16_above) {
  KvChoiceOptions o;
  o.explicit_kv_f16 = explicit_kv;
  o.max_context = ctx;
  o.f16_above = above;
  return ResolveKvKind(o);
}
}  // namespace

TEST(KvChoice, DefaultThresholdIs150000) { EXPECT_EQ(KvChoiceOptions{}.f16_above, 150000u); }

TEST(KvChoice, ExplicitFlagWinsOverTheThreshold) {
  EXPECT_FALSE(Resolve(0, 262144).f16);  // forced default kinds above the threshold
  EXPECT_TRUE(Resolve(1, 32768).f16);    // forced fp16 below it
  EXPECT_TRUE(Resolve(1, 262144, 0).f16);
  EXPECT_FALSE(Resolve(0, 262144, 0).f16);
  EXPECT_EQ(Resolve(0, 262144).reason, "explicit --kv-f16 0");
  EXPECT_EQ(Resolve(1, 32768).reason, "explicit --kv-f16 1");
}

TEST(KvChoice, AutoIsStrictlyAboveTheThreshold) {
  EXPECT_FALSE(Resolve(-1, 32768).f16);
  EXPECT_FALSE(Resolve(-1, 131072).f16);
  EXPECT_FALSE(Resolve(-1, 150000).f16);  // at the threshold: default kinds
  EXPECT_TRUE(Resolve(-1, 150001).f16);
  EXPECT_TRUE(Resolve(-1, 262144).f16);
  EXPECT_EQ(Resolve(-1, 262144).reason, "auto: --max-context 262144 > --kv-f16-above 150000");
  EXPECT_EQ(Resolve(-1, 150000).reason, "auto: --max-context 150000 <= --kv-f16-above 150000");
  // A custom threshold moves the boundary.
  EXPECT_FALSE(Resolve(-1, 65536, 65536).f16);
  EXPECT_TRUE(Resolve(-1, 65537, 65536).f16);
}

TEST(KvChoice, ZeroDisablesTheAutomaticChoice) {
  EXPECT_FALSE(Resolve(-1, 262144, 0).f16);
  EXPECT_FALSE(Resolve(-1, 1, 0).f16);
  EXPECT_EQ(Resolve(-1, 262144, 0).reason, "auto disabled (--kv-f16-above 0)");
}
