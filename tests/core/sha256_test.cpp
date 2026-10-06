// SHA-256: FIPS 180-4 test vectors, the hardware (SHA-NI) and portable
// paths agree on generated inputs, and streaming in arbitrary pieces equals
// hashing at once.

#include <gtest/gtest.h>
#include <hegel/gtest.h>

#include <string>

#include "core/sha256.hpp"

using halo::core::Sha256;
namespace gs = hegel::generators;

TEST(Sha256, KnownVectors) {
  EXPECT_EQ(halo::core::Sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(halo::core::Sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(halo::core::Sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  std::string million(1000000, 'a');
  EXPECT_EQ(halo::core::Sha256Hex(million), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, PortableAndHardwarePathsAgreeAndStreamingIsSplitInvariant) {
  hegel::test([](hegel::TestCase& tc) {
    // Lengths drawn explicitly (generated binaries stay short) so multi-block
    // inputs and every padding boundary (55, 56, 63, 64 mod 64) are covered.
    const auto len = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 700}));
    auto seed = tc.draw(gs::integers<std::uint32_t>());
    std::string s(len, '\0');
    for (auto& c : s) c = static_cast<char>((seed = seed * 1664525u + 1013904223u) >> 24);
    Sha256::ForcePortable(true);
    const auto portable = halo::core::Sha256Hex(s);
    Sha256::ForcePortable(false);
    const auto native = halo::core::Sha256Hex(s);
    ASSERT_EQ(portable, native);
    Sha256 h;
    std::size_t at = 0;
    while (at < s.size()) {
      const auto n = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 130}));
      const std::size_t take = std::min(n, s.size() - at);
      h.Update(s.data() + at, take);
      at += take;
    }
    ASSERT_EQ(halo::core::HexDigest(h.Finish()), native);
  }, {.test_cases = 400});
}
