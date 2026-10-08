// The yard key route (server/yard_keys.hpp; docs/SESSIONS.md section 10, "The key route"): RS256 verification
// against a test JWKS (the token shapes yard's internal/sts/jwt.go signs), every refusal (expired, wrong audience,
// issuer, session, alg, signature, kid, lifetime, use or key binding, replayed jti, stale epoch, open before or at
// the seal), duplicates as no-ops, and the epoch rule: an open at E + 1 opens the file sealed at E, an open after a
// further bump does not (crypto-shred), revocation forgets the keys.

#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <string>

#include "server/yard_keys.hpp"
#include "yard_test_util.hpp"

namespace sc = halo::core::crypt;
using halo::server::SpillKeys;
using halo::server::YardSpillKeys;
using namespace yardtest;

namespace {
struct Fixture {
  Signer signer{"kid-a"}, other{"kid-b"};
  std::string dir, jwks;
  std::unique_ptr<YardSpillKeys> y;
  explicit Fixture(SpillKeys* fallback = nullptr) {
    dir = (std::filesystem::temp_directory_path() / ("halo-yard-" + std::to_string(::getpid()))).string();
    std::filesystem::create_directories(dir);
    jwks = dir + "/jwks.json";
    WriteFile(jwks, JwksJson({&signer}));  // `other` signs, but is not published
    YardSpillKeys::Options o;
    o.jwks_path = jwks;
    o.issuer = "https://id.example.org/yard";
    o.fallback = fallback;
    y = std::make_unique<YardSpillKeys>(o);
    std::string e;
    EXPECT_TRUE(y->LoadJwks(&e)) << e;
  }
  ~Fixture() { std::filesystem::remove_all(dir); }
};
}  // namespace

TEST(YardKeys, Base64UrlRoundTripsAndRefusesJunk) {
  for (std::size_t n = 0; n < 70; ++n) {
    std::vector<std::uint8_t> b(n), back;
    for (std::size_t i = 0; i < n; ++i) b[i] = static_cast<std::uint8_t>(i * 37 + n);
    ASSERT_TRUE(halo::server::Base64UrlDecode(halo::server::Base64UrlEncode(b), &back));
    ASSERT_EQ(back, b);
  }
  std::vector<std::uint8_t> out;
  EXPECT_FALSE(halo::server::Base64UrlDecode("ab$d", &out));
  EXPECT_FALSE(halo::server::Base64UrlDecode("AB", &out)) << "non-canonical trailing bits";
}

TEST(YardKeys, TokenRefusals) {
  Fixture f;
  const std::string key = Key32(1);
  const auto with = [&](auto edit, const std::string& alg = "RS256", const Signer* s = nullptr) {
    Claims c;
    c.key_sha256 = KeyHash(key);
    edit(c);
    return f.y->Register("sess-1", Body(Token(s ? *s : f.signer, c, alg), "seal", key));
  };
  const std::int64_t now = std::time(nullptr);
  struct Case {
    const char* what;
    YardSpillKeys::Answer a;
    int status;
    const char* reason;  // in the error message: refused for this reason, not another
  };
  const Case cases[] = {
      {"expired", with([&](Claims& c) { c.iat = now - 400; c.exp = now - 100; }), 401, "expired"},
      {"wrong audience", with([](Claims& c) { c.aud = "scope"; }), 401, "audience"},
      {"wrong issuer", with([](Claims& c) { c.iss = "https://evil.example/yard"; }), 401, "issuer"},
      {"another session's token", with([](Claims& c) { c.sess = "sess-2"; }), 401, "another session"},
      {"lifetime over 300 s", with([&](Claims& c) { c.iat = now; c.exp = now + 3600; }), 401, "lifetime"},
      {"issued in the future", with([&](Claims& c) { c.iat = now + 600; c.exp = now + 700; }), 401, "not yet valid"},
      {"no halo_spill claim", with([](Claims& c) { c.with_halo_spill = false; }), 401, "does not authorise"},
      {"use not authorised", with([](Claims& c) { c.use = "open"; }), 401, "does not authorise"},
      {"bound to another key", with([&](Claims& c) { c.key_sha256 = KeyHash(Key32(2)); }), 401, "another key"},
      {"no epoch", with([](Claims& c) { c.epoch = 0; }), 401, "yard_epoch"},
      {"alg none", with([](Claims&) {}, "none"), 401, "RS256"},
      {"alg HS256", with([](Claims&) {}, "HS256"), 401, "RS256"},
      {"signed by an unpublished key", with([](Claims&) {}, "RS256", &f.other), 401, "kid"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(c.a.status, c.status) << c.what << ": " << c.a.body;
    EXPECT_NE(c.a.body.find(c.reason), std::string::npos) << c.what << ": " << c.a.body;
    EXPECT_EQ(c.a.body.find(KeyB64(key)), std::string::npos) << "the key is echoed";
  }
  // A forged signature: a valid token with its last signature byte changed.
  Claims c;
  c.key_sha256 = KeyHash(key);
  std::string t = Token(f.signer, c);
  t.back() = t.back() == 'A' ? 'B' : 'A';
  EXPECT_EQ(f.y->Register("sess-1", Body(t, "seal", key)).status, 401);
  // Malformed requests.
  EXPECT_EQ(f.y->Register("sess 1", Body(Token(f.signer, c), "seal", key)).status, 400);
  EXPECT_EQ(f.y->Register("sess-1", "{\"token\":\"x\",\"use\":\"seal\",\"key\":\"short\"}").status, 400);
  EXPECT_EQ(f.y->Register("sess-1", "not json").status, 400);
  EXPECT_EQ(f.y->Register("sess-1", "{\"token\":\"x\",\"use\":\"steal\"}").status, 400);
  // Nothing was registered by any of these.
  sc::Kek k;
  std::string why;
  EXPECT_FALSE(f.y->Key("sess-1", SpillKeys::Use::kSeal, &k, &why));
}

// Seal at E, open at E + 1 opens exactly that file; replay, stale, early open and a skipped epoch are refused or
// cannot open; keys are one-shot; revocation forgets.
TEST(YardKeys, EpochRulesReplayDuplicatesAndShredding) {
  Fixture f;
  const std::string k1 = Key32(1), k2 = Key32(2);
  // Seal at epoch 3.
  Claims c;
  c.epoch = 3;
  c.key_sha256 = KeyHash(k1);
  const std::string t = Token(f.signer, c);
  auto a = f.y->Register("sess-1", Body(t, "seal", k1));
  ASSERT_EQ(a.status, 200) << a.body;
  EXPECT_NE(a.body.find("\"duplicate\":false"), std::string::npos);
  EXPECT_EQ(f.y->Register("sess-1", Body(t, "seal", k1)).status, 409) << "replayed jti accepted";
  a = Register(*f.y, f.signer, "sess-1", "seal", 3, k1);  // a retry with a fresh token: a no-op
  ASSERT_EQ(a.status, 200);
  EXPECT_NE(a.body.find("\"duplicate\":true"), std::string::npos);
  EXPECT_EQ(Register(*f.y, f.signer, "sess-1", "seal", 2, k1).status, 409) << "stale epoch accepted";
  sc::Kek seal;
  std::string why;
  ASSERT_TRUE(f.y->Key("sess-1", SpillKeys::Use::kSeal, &seal, &why)) << why;
  EXPECT_EQ(seal.id.rfind("yard:", 0), 0u);
  EXPECT_NE(seal.id.find(":e3:sess-1"), std::string::npos) << seal.id;
  EXPECT_NE(f.y->ListJson().find("\"sealed_epoch\":3"), std::string::npos) << f.y->ListJson();
  EXPECT_EQ(f.y->ListJson().find(KeyB64(k1)), std::string::npos);
  f.y->Used("sess-1", SpillKeys::Use::kSeal);  // the spill is done: one-shot
  sc::Kek k;
  EXPECT_FALSE(f.y->Key("sess-1", SpillKeys::Use::kSeal, &k, &why)) << "a used seal key is still held";
  // The file waits for yard: no open key yet.
  EXPECT_FALSE(f.y->KeyForFile("sess-1", seal.id, &k, &why));
  // An open at the seal's own epoch (a token minted before the suspend committed) is refused.
  EXPECT_EQ(Register(*f.y, f.signer, "sess-1", "open", 3, k1).status, 409);
  // The open at E + 1 = 4 opens exactly the file sealed at 3.
  a = Register(*f.y, f.signer, "sess-1", "open", 4, k1);
  ASSERT_EQ(a.status, 200) << a.body;
  sc::Kek open;
  ASSERT_TRUE(f.y->KeyForFile("sess-1", seal.id, &open, &why)) << why;
  EXPECT_EQ(open.id, seal.id);
  EXPECT_TRUE(sc::Equal(open.key.span(), seal.key.span()));
  // Another key under a good token does not open it (another key name, another KEK).
  a = Register(*f.y, f.signer, "sess-1", "open", 4, k2);
  ASSERT_EQ(a.status, 200) << a.body;
  ASSERT_TRUE(f.y->KeyForFile("sess-1", seal.id, &k, &why));
  EXPECT_NE(k.id, seal.id);
  // A skipped epoch: an open at 5 derives the file epoch 4, not 3: the file sealed at 3 is shredded.
  a = Register(*f.y, f.signer, "sess-1", "open", 5, k1);
  ASSERT_EQ(a.status, 200) << a.body;
  ASSERT_TRUE(f.y->KeyForFile("sess-1", seal.id, &k, &why));
  EXPECT_NE(k.id, seal.id);
  EXPECT_FALSE(sc::Equal(k.key.span(), seal.key.span()));
  // A late open at 4 is now stale.
  EXPECT_EQ(Register(*f.y, f.signer, "sess-1", "open", 4, k1).status, 409);
  // Revoke at 6: keys forgotten, the session reported for its entries to be erased; then older tokens are stale.
  a = Register(*f.y, f.signer, "sess-1", "revoke", 6, "");
  ASSERT_EQ(a.status, 200) << a.body;
  EXPECT_EQ(f.y->TakeRevoked(), std::vector<std::string>{"sess-1"});
  EXPECT_FALSE(f.y->KeyForFile("sess-1", seal.id, &k, &why));
  EXPECT_EQ(Register(*f.y, f.signer, "sess-1", "open", 5, k1).status, 409);
  // Sessions are independent.
  EXPECT_EQ(Register(*f.y, f.signer, "sess-2", "seal", 1, k2).status, 200);
}

// A session without a registered key: refused, or sealed by the fallback (phase A) provider; a file sealed by the
// fallback opens with it; a yard-sealed file never falls back.
TEST(YardKeys, FallbackToTheMasterProvider) {
  sc::SecretBytes m(32);
  ASSERT_TRUE(sc::RandomBytes({m.data(), 32}));
  halo::server::MasterSpillKeys master("file", std::move(m), 1);
  Fixture none, fb(&master);
  sc::Kek k, k2;
  std::string why;
  EXPECT_FALSE(none.y->Key("s1", SpillKeys::Use::kSeal, &k, &why));
  ASSERT_TRUE(fb.y->Key("s1", SpillKeys::Use::kSeal, &k, &why)) << why;
  EXPECT_EQ(k.id.rfind("file:", 0), 0u);
  ASSERT_TRUE(fb.y->KeyForFile("s1", k.id, &k2, &why)) << why;
  EXPECT_EQ(k2.id, k.id);
  EXPECT_FALSE(fb.y->KeyForFile("s1", "yard:0011223344556677:e3:s1", &k2, &why)) << "a yard file fell back";
}
