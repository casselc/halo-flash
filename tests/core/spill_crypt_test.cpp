// Spill encryption primitives (core/spill_crypt.hpp; docs/SESSIONS.md section 10): AES-256-GCM refuses every
// changed bit, the piece nonces are salt || be32(index) and never repeat, the KEK derivation separates sessions,
// epochs and masters (RFC 5869 vector for the HKDF), and the DEK wrap authenticates the whole header and never
// leaves the DEK in the clear.

#include <gtest/gtest.h>
#include <hegel/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "core/spill_crypt.hpp"

namespace gs = hegel::generators;
namespace sc = halo::core::crypt;

namespace {
std::vector<std::uint8_t> Hex(const std::string& h) {
  std::vector<std::uint8_t> v;
  for (std::size_t i = 0; i + 1 < h.size(); i += 2) v.push_back(static_cast<std::uint8_t>(std::stoi(h.substr(i, 2), nullptr, 16)));
  return v;
}
sc::SecretBytes RandomKey() {
  sc::SecretBytes k(32);
  EXPECT_TRUE(k.ok() && sc::RandomBytes({k.data(), 32}));
  return k;
}
}  // namespace

TEST(SpillCrypt, HkdfMatchesRfc5869AndSecretBytesAreLocked) {
  // RFC 5869 A.1.
  std::string ikm_hex;
  for (int i = 0; i < 22; ++i) ikm_hex += "0b";
  const auto ikm = Hex(ikm_hex);
  const auto salt = Hex("000102030405060708090a0b0c"), info = Hex("f0f1f2f3f4f5f6f7f8f9");
  std::vector<std::uint8_t> okm(42);
  ASSERT_TRUE(sc::HkdfSha256(ikm, salt, info, okm));
  EXPECT_EQ(okm, Hex("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"));
  sc::SecretBytes s(32);
  ASSERT_TRUE(s.ok());
  EXPECT_TRUE(s.locked()) << "mlock refused (RLIMIT_MEMLOCK?)";
  EXPECT_TRUE(std::all_of(s.data(), s.data() + 32, [](std::uint8_t b) { return b == 0; }));
  sc::SecretBytes moved = std::move(s);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(moved.ok());
}

// Seal then open is the identity; any single changed bit in the key, nonce, AAD, ciphertext or tag is refused.
TEST(SpillCrypt, GcmRefusesEveryChangedBit) {
  hegel::test([](hegel::TestCase& tc) {
    const auto n = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 5000}));
    const auto aad_n = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 600}));
    std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
    const auto byte = [&] { return static_cast<std::uint8_t>((seed = seed * 1664525u + 1013904223u) >> 24); };
    std::vector<std::uint8_t> key(32), pt(n), aad(aad_n);
    std::array<std::uint8_t, 12> nonce{};
    for (auto* v : {&key, &pt, &aad})
      for (auto& b : *v) b = byte();
    for (auto& b : nonce) b = byte();
    std::vector<std::uint8_t> ct = pt;
    std::array<std::uint8_t, 16> tag{};
    sc::Gcm g;
    ASSERT_TRUE(g.Seal(key, nonce, aad, ct.data(), ct.size(), tag));
    if (n >= 16) ASSERT_NE(ct, pt);
    auto back = ct;
    ASSERT_TRUE(g.Open(key, nonce, aad, back.data(), back.size(), tag));
    ASSERT_EQ(back, pt);
    const int what = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4}));
    auto k2 = key;
    auto n2 = nonce;
    auto a2 = aad;
    auto c2 = ct;
    auto t2 = tag;
    const auto bit = static_cast<std::uint8_t>(1u << tc.draw(gs::integers<int>({.min_value = 0, .max_value = 7})));
    const auto at = [&](std::size_t size) { return tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = size - 1})); };
    if (what == 0) k2[at(32)] ^= bit;
    if (what == 1) n2[at(12)] ^= bit;
    if (what == 2 && !a2.empty()) a2[at(a2.size())] ^= bit;
    if (what == 3 && !c2.empty()) c2[at(c2.size())] ^= bit;
    if (what == 4) t2[at(16)] ^= bit;
    const bool changed = k2 != key || n2 != nonce || a2 != aad || c2 != ct || t2 != tag;
    ASSERT_EQ(g.Open(k2, n2, a2, c2.data(), c2.size(), t2), !changed) << "what " << what;
  }, {.test_cases = 300});
}

// The nonce of piece i is salt || be32(i) (the format: docs/SESSIONS.md section 10); over random salts and
// indices (pieces and the meta's 0xffffffff), no two (file, part) pairs share a nonce.
TEST(SpillCrypt, PieceNoncesAreSaltIndexAndNeverRepeat) {
  hegel::test([](hegel::TestCase& tc) {
    const int files = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 6}));
    std::set<std::array<std::uint8_t, 12>> seen;
    std::set<std::array<std::uint8_t, 8>> salts;
    std::size_t parts = 0;
    for (int f = 0; f < files; ++f) {
      std::array<std::uint8_t, 8> salt{};
      ASSERT_TRUE(sc::RandomBytes(salt));
      if (!salts.insert(salt).second) continue;  // 2^-64
      const auto pieces = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 3000}));
      std::vector<std::uint32_t> idx(pieces);
      for (std::uint32_t i = 0; i < pieces; ++i) idx[i] = i;
      idx.push_back(sc::kMetaIndex);
      for (const auto i : idx) {
        const auto n = sc::PieceNonce(salt, i);
        ASSERT_EQ(std::memcmp(n.data(), salt.data(), 8), 0);
        ASSERT_EQ(n[8], static_cast<std::uint8_t>(i >> 24));
        ASSERT_EQ(n[9], static_cast<std::uint8_t>(i >> 16));
        ASSERT_EQ(n[10], static_cast<std::uint8_t>(i >> 8));
        ASSERT_EQ(n[11], static_cast<std::uint8_t>(i));
        ASSERT_TRUE(seen.insert(n).second) << "nonce repeated: file " << f << " index " << i;
        ++parts;
      }
    }
    ASSERT_EQ(seen.size(), parts);
  }, {.test_cases = 60});
}

// KEK = HKDF(master, "halo-spill-v2" || session || epoch): deterministic; another session, epoch or master gives
// another key and another id; ids are bounded and printable.
TEST(SpillCrypt, KekSeparatesSessionsEpochsAndMasters) {
  const auto m1 = RandomKey(), m2 = RandomKey();
  std::string e;
  sc::Kek a, b, c, d, f, g;
  ASSERT_TRUE(sc::DeriveKek(m1.span(), "sess-1", 1, "tpm:x", &a, &e)) << e;
  ASSERT_TRUE(sc::DeriveKek(m1.span(), "sess-1", 1, "tpm:x", &b, &e)) << e;
  ASSERT_TRUE(sc::DeriveKek(m1.span(), "sess-1", 2, "tpm:x", &c, &e)) << e;
  ASSERT_TRUE(sc::DeriveKek(m1.span(), "sess-2", 1, "tpm:x", &d, &e)) << e;
  ASSERT_TRUE(sc::DeriveKek(m2.span(), "sess-1", 1, "tpm:x", &f, &e)) << e;
  ASSERT_TRUE(sc::DeriveKek(m1.span(), "sess-", 11, "tpm:x", &g, &e)) << e;  // no concatenation ambiguity
  EXPECT_TRUE(sc::Equal(a.key.span(), b.key.span()));
  EXPECT_EQ(a.id, "tpm:x:e1:sess-1");
  for (const auto* k : {&c, &d, &f, &g}) EXPECT_FALSE(sc::Equal(a.key.span(), k->key.span()));
  EXPECT_NE(a.id, c.id);
  EXPECT_NE(a.id, d.id);
  std::string n1, n2;
  ASSERT_TRUE(sc::KeyName(m1.span(), &n1));
  ASSERT_TRUE(sc::KeyName(m2.span(), &n2));
  EXPECT_NE(n1, n2);
  EXPECT_EQ(n1.size(), 16u);
  EXPECT_FALSE(sc::DeriveKek(m1.span(), std::string(300, 'a'), 1, "tpm:x", &g, &e));
  EXPECT_FALSE(sc::DeriveKek(m1.span(), "has space", 1, "tpm:x", &g, &e));
  EXPECT_FALSE(sc::DeriveKek(std::span<const std::uint8_t>(m1.data(), 16), "s", 1, "tpm:x", &g, &e));
}

// The wrap authenticates every byte before the wrapped key (the header's AAD) and itself; the meta tag after it is
// the meta's business. The DEK, the KEK and the master never appear in the header.
TEST(SpillCrypt, DekWrapAuthenticatesTheHeaderAndHidesTheKey) {
  const auto master = RandomKey();
  std::string e;
  sc::Kek kek, other;
  ASSERT_TRUE(sc::DeriveKek(master.span(), "s", 7, "file:k", &kek, &e)) << e;
  ASSERT_TRUE(sc::DeriveKek(master.span(), "s", 8, "file:k", &other, &e)) << e;
  hegel::test([&](hegel::TestCase& tc) {
    std::array<std::uint8_t, sc::kHeaderBytes> h{};
    for (std::size_t i = 0; i < sc::kWrapNonceOff; ++i)
      h[i] = static_cast<std::uint8_t>(tc.draw(gs::integers<int>({.min_value = 0, .max_value = 255})) * (i % 7 == 0));
    const auto dek = RandomKey();
    ASSERT_TRUE(sc::WrapDek(kek, h, dek.span()));
    const auto contains = [&](std::span<const std::uint8_t> needle) {
      return std::search(h.begin(), h.end(), needle.begin(), needle.end()) != h.end();
    };
    ASSERT_FALSE(contains(dek.span())) << "the DEK is in the header in the clear";
    ASSERT_FALSE(contains(kek.key.span()));
    ASSERT_FALSE(contains(master.span()));
    ASSERT_EQ(sc::HeaderAad(h).size(), sc::kAadBytes);
    sc::SecretBytes back;
    ASSERT_TRUE(sc::UnwrapDek(kek, h, &back));
    ASSERT_TRUE(sc::Equal(back.span(), dek.span()));
    ASSERT_FALSE(sc::UnwrapDek(other, h, &back)) << "another epoch's KEK unwrapped the DEK";
    const auto at = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = sc::kHeaderBytes - 1}));
    auto h2 = h;
    h2[at] ^= static_cast<std::uint8_t>(1u << tc.draw(gs::integers<int>({.min_value = 0, .max_value = 7})));
    ASSERT_EQ(sc::UnwrapDek(kek, h2, &back), at >= sc::kMetaTagOff) << "byte " << at;
  }, {.test_cases = 300});
}

// Secure memory fails closed: when mlock / madvise fail (here: injected) the allocation is empty and every user
// refuses; nothing is handed out unlocked. Live() counts what is really still mapped (keys are really freed).
TEST(SpillCrypt, SecretBytesFailClosed) {
  const long live = sc::SecretBytes::Live();
  {
    sc::SecretBytes a(32);
    ASSERT_TRUE(a.ok() && a.locked());
    EXPECT_EQ(sc::SecretBytes::Live(), live + 1);
    sc::SecretBytes::FailNextForTest(1);
    sc::SecretBytes b(32);
    EXPECT_FALSE(b.ok()) << "an unlockable allocation was handed out";
    EXPECT_EQ(sc::SecretBytes::Live(), live + 1);
    sc::SecretBytes c(32);
    EXPECT_TRUE(c.ok()) << "the injected failure leaked into the next allocation";
    // A key derivation into memory that cannot be locked refuses.
    sc::SecretBytes::FailNextForTest(1);
    sc::Kek k;
    std::string e;
    EXPECT_FALSE(sc::DeriveKek(a.span(), "s", 1, "file:x", &k, &e));
  }
  EXPECT_EQ(sc::SecretBytes::Live(), live) << "SecretBytes not unmapped on destruction";
}

// A failed open never leaves unauthenticated plaintext behind: the buffer is zeroed (a bad tag on authentic
// ciphertext would otherwise leave the real plaintext there). A non-in-place seal leaves the source intact and a
// failed seal leaves no half-encrypted output.
TEST(SpillCrypt, FailedOpenWipesAndSealCanWriteElsewhere) {
  std::vector<std::uint8_t> key(32, 7), pt(1000), ct(1000), aad(20, 3);
  for (std::size_t i = 0; i < pt.size(); ++i) pt[i] = static_cast<std::uint8_t>(i * 13 + 1);
  std::array<std::uint8_t, 12> nonce{};
  std::array<std::uint8_t, 16> tag{};
  sc::Gcm g;
  ASSERT_TRUE(g.Seal(key, nonce, aad, pt.data(), ct.data(), ct.size(), tag));
  EXPECT_NE(ct, pt);
  EXPECT_EQ(pt[5], static_cast<std::uint8_t>(5 * 13 + 1)) << "the source was changed";
  auto buf = ct;
  ASSERT_TRUE(g.Open(key, nonce, aad, buf.data(), buf.size(), tag));
  EXPECT_EQ(buf, pt);
  buf = ct;
  tag[0] ^= 1;
  EXPECT_FALSE(g.Open(key, nonce, aad, buf.data(), buf.size(), tag));
  EXPECT_TRUE(std::all_of(buf.begin(), buf.end(), [](std::uint8_t b) { return b == 0; }))
      << "plaintext of a failed open is still in the buffer";
  // The context is reset after each call: the same object keeps working under other keys.
  std::vector<std::uint8_t> key2(32, 9);
  std::array<std::uint8_t, 16> tag2{};
  buf = pt;
  ASSERT_TRUE(g.Seal(key2, nonce, aad, buf.data(), buf.size(), tag2));
  ASSERT_TRUE(g.Open(key2, nonce, aad, buf.data(), buf.size(), tag2));
  EXPECT_EQ(buf, pt);
  // A seal with a bad key length fails and writes nothing usable.
  std::vector<std::uint8_t> out(pt.size(), 0xaa);
  EXPECT_FALSE(g.Seal(std::span<const std::uint8_t>(key.data(), 16), nonce, aad, pt.data(), out.data(), out.size(), tag2));
}

// Sha256 reports success and fills exactly 32 bytes (the FIPS 180-2 "abc" vector).
TEST(SpillCrypt, Sha256Vector) {
  std::array<std::uint8_t, 32> d{};
  const std::string abc = "abc";
  ASSERT_TRUE(sc::Sha256({reinterpret_cast<const std::uint8_t*>(abc.data()), abc.size()}, d));
  EXPECT_EQ(std::vector<std::uint8_t>(d.begin(), d.end()),
            Hex("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
}
