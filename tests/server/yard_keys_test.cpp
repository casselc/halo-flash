// The yard key route (server/yard_keys.hpp; docs/SESSIONS.md section 10, "The key route"): RS256 verification
// against a test JWKS (the token shapes yard's internal/sts/jwt.go signs) and every refusal; the JWKS's own checks
// (key size, exponent, canonical modulus, duplicate kids, use / key_ops) and its refresh (a removed kid stops
// verifying, an empty or broken set denies); fail-closed configuration; the epoch rules (an open at E + 1 opens the
// file sealed at E; stale, early, replayed and used registrations are refused); revoke fencing (no seal or open at
// or below the fence, not even with a fresh or a delayed token; idempotent); one-shot keys that a fresh token never
// re-arms; real expiry (keys zeroed by the sweeper; authorization ends at min(token exp, registration + TTL));
// persisted epochs across a restart; generation-bound leases (a finished lease never consumes a newer
// registration; an in-flight restore at a revoke); and custody (a yard-bound entry never falls back to the master).

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "server/yard_keys.hpp"
#include "yard_test_util.hpp"

namespace sc = halo::core::crypt;
using halo::server::SpillKeys;
using halo::server::SpillLease;
using halo::server::YardSpillKeys;
using namespace yardtest;

namespace {
std::string FreshDir() {
  static std::atomic<int> n{0};
  auto p = std::filesystem::temp_directory_path() /
           ("halo-yard-" + std::to_string(::getpid()) + "-" + std::to_string(n++));
  std::filesystem::remove_all(p);
  std::filesystem::create_directories(p);
  std::filesystem::permissions(p, std::filesystem::perms::owner_all);
  return p.string();
}

struct Fixture {
  Signer signer{"kid-a"}, other{"kid-b"};
  std::string dir, jwks;
  YardSpillKeys::Options o;
  std::unique_ptr<YardSpillKeys> y;
  explicit Fixture(SpillKeys* fallback = nullptr, bool persist = false, std::int64_t ttl = 600) {
    dir = FreshDir();
    jwks = dir + "/jwks.json";
    WriteFile(jwks, JwksJson({&signer}));  // `other` signs, but is not published
    o.jwks_path = jwks;
    o.issuer = "https://id.example.org/yard";
    o.fallback = fallback;
    o.key_ttl_s = ttl;
    o.sweeper = false;
    if (persist) o.state_path = dir + "/yard-state";
    Restart();
  }
  void Restart() {
    y.reset();
    std::string e;
    y = YardSpillKeys::Create(o, &e);
    EXPECT_TRUE(y) << e;
  }
  ~Fixture() {
    y.reset();
    std::filesystem::remove_all(dir);
  }
  YardSpillKeys::Answer Reg(const std::string& use, std::int64_t epoch, const std::string& key, std::int64_t entry = 1,
                            const std::string& sess = "sess-1", std::int64_t through = 0) {
    return Register(*y, signer, sess, use, epoch, key, entry, through);
  }
  bool Seal(std::uint64_t entry, const std::string& label, bool bound, SpillLease* l, std::string* why = nullptr) {
    std::string w;
    return y->SealLease(entry, label, bound, l, why ? why : &w);
  }
  bool Open(const std::string& label, const std::string& id, SpillLease* l, std::string* why = nullptr) {
    std::string w;
    return y->OpenLease(label, id, l, why ? why : &w);
  }
};

bool Has(const YardSpillKeys::Answer& a, const std::string& s) { return a.body.find(s) != std::string::npos; }
}  // namespace

TEST(YardKeys, Base64UrlIsStrict) {
  for (std::size_t n = 0; n < 70; ++n) {
    std::vector<std::uint8_t> b(n), back;
    for (std::size_t i = 0; i < n; ++i) b[i] = static_cast<std::uint8_t>(i * 37 + n);
    ASSERT_TRUE(halo::server::Base64UrlDecode(halo::server::Base64UrlEncode(b), &back));
    ASSERT_EQ(back, b);
  }
  std::vector<std::uint8_t> out;
  EXPECT_FALSE(halo::server::Base64UrlDecode("ab$d", &out));
  EXPECT_FALSE(halo::server::Base64UrlDecode("AB", &out)) << "non-canonical trailing bits";
  EXPECT_FALSE(halo::server::Base64UrlDecode("ab+/", &out)) << "standard base64 alphabet accepted";
  EXPECT_FALSE(halo::server::Base64UrlDecode("YQ==", &out)) << "padding accepted";
  EXPECT_FALSE(halo::server::Base64UrlDecode("A", &out));
  std::array<std::uint8_t, 32> k{};
  const std::string good = halo::server::Base64UrlEncode(std::vector<std::uint8_t>(32, 9));
  EXPECT_TRUE(halo::server::Base64UrlDecodeExact(good, k));
  EXPECT_FALSE(halo::server::Base64UrlDecodeExact(good + "AA", k)) << "length is checked before decoding";
  EXPECT_FALSE(halo::server::Base64UrlDecodeExact(good.substr(1), k));
  EXPECT_TRUE(std::all_of(k.begin(), k.end(), [](std::uint8_t b) { return b == 0; })) << "a refused decode left bytes";
}

// Every refusal: one 401 body for all reasons (no oracle), the reason only in the log (Answer::reason).
TEST(YardKeys, TokenRefusals) {
  Fixture f;
  const std::string key = Key32(1);
  const auto with = [&](auto edit, const std::string& alg = "RS256", const Signer* s = nullptr) {
    Claims c;
    c.key_sha256 = KeyHash(key);
    edit(c);
    std::string body = Body(Token(s ? *s : f.signer, c, alg), c.use == "revoke" ? "revoke" : "seal",
                            c.use == "revoke" ? "" : key);
    return f.y->Register("sess-1", body);
  };
  const std::int64_t now = std::time(nullptr);
  struct Case {
    const char* what;
    YardSpillKeys::Answer a;
    const char* reason;
  };
  const Case cases[] = {
      {"expired", with([&](Claims& c) { c.iat = now - 400; c.exp = now - 100; }), "expired"},
      {"at exp (no post-exp leeway)", with([&](Claims& c) { c.iat = now - 300; c.exp = now; }), "expired"},
      {"wrong audience", with([](Claims& c) { c.aud = "scope"; }), "audience"},
      {"audience array without ours", with([](Claims& c) { c.aud_json = "[\"scope\",\"zfs\"]"; }), "audience"},
      {"wrong issuer", with([](Claims& c) { c.iss = "https://evil.example/yard"; }), "issuer"},
      {"another session's token", with([](Claims& c) { c.sess = "sess-2"; }), "another session"},
      {"lifetime over 300 s", with([&](Claims& c) { c.iat = now; c.exp = now + 3600; }), "lifetime"},
      {"exp before iat", with([&](Claims& c) { c.iat = now; c.exp = now - 1; }), "lifetime"},
      {"exp equal to iat", with([&](Claims& c) { c.iat = now; c.exp = now; }), "lifetime"},
      {"issued in the future", with([&](Claims& c) { c.iat = now + 600; c.exp = now + 700; }), "not yet valid"},
      {"nbf in the future", with([&](Claims& c) { c.nbf = std::to_string(now + 200); }), "not yet valid"},
      {"nbf not an integer", with([](Claims& c) { c.nbf = "\"soon\""; }), "claims"},
      {"nbf fractional", with([&](Claims& c) { c.nbf = std::to_string(now) + ".5"; }), "claims"},
      {"no halo_spill claim", with([](Claims& c) { c.with_halo_spill = false; }), "use not authorised"},
      {"use not authorised", with([](Claims& c) { c.use = "open"; }), "use not authorised"},
      {"bound to another key", with([&](Claims& c) { c.key_sha256 = KeyHash(Key32(2)); }), "another key"},
      {"no epoch", with([](Claims& c) { c.epoch = 0; }), "yard_epoch"},
      {"seal without an entry", with([](Claims& c) { c.entry = 0; }), "entry"},
      {"revoke through a later epoch", with([](Claims& c) { c.use = "revoke"; c.through = 5; c.epoch = 3; }), "through"},
      {"duplicate claim (aud twice)", with([](Claims& c) { c.aud = "scope"; c.extra = ",\"aud\":\"halo_serve\""; }),
       "malformed"},
      {"alg none", with([](Claims&) {}, "none"), "alg"},
      {"alg HS256", with([](Claims&) {}, "HS256"), "alg"},
      {"alg PS256", with([](Claims&) {}, "PS256"), "alg"},
      {"signed by an unpublished key", with([](Claims&) {}, "RS256", &f.other), "unknown kid"},
  };
  const std::string refused = "{\"ok\":false,\"error\":\"token refused\"}";
  for (const auto& c : cases) {
    EXPECT_EQ(c.a.status, 401) << c.what << ": " << c.a.body;
    EXPECT_EQ(c.a.body, refused) << c.what << ": the 401 body is not normalized";
    EXPECT_NE(c.a.reason.find(c.reason), std::string::npos) << c.what << ": refused for " << c.a.reason;
  }
  // The audience as an array containing ours is accepted.
  {
    Claims c;
    c.key_sha256 = KeyHash(key);
    c.aud_json = "[\"scope\",\"halo_serve\"]";
    std::string body = Body(Token(f.signer, c), "seal", key);
    EXPECT_EQ(f.y->Register("sess-1", body).status, 200);
  }
  // A forged signature: a valid token with its last signature byte changed.
  Claims c;
  c.key_sha256 = KeyHash(key);
  std::string t = Token(f.signer, c);
  // A character well inside the signature (every bit significant: still canonical base64url).
  char& ch = t[t.size() - 20];
  ch = ch == 'A' ? 'B' : 'A';
  std::string body = Body(t, "seal", key);
  const auto forged = f.y->Register("sess-1", body);
  EXPECT_EQ(forged.status, 401);
  EXPECT_EQ(forged.reason, "signature");
  // The body (key and token) is overwritten whatever the outcome.
  EXPECT_TRUE(std::all_of(body.begin(), body.end(), [](char ch) { return ch == 0; })) << "the request body survived";
  // Malformed requests.
  EXPECT_EQ(RegisterRaw(*f.y, "sess 1", Body(Token(f.signer, c), "seal", key)).status, 400);
  EXPECT_EQ(RegisterRaw(*f.y, "sess-1", "{\"token\":\"x\",\"use\":\"seal\",\"key\":\"short\"}").status, 400);
  EXPECT_EQ(RegisterRaw(*f.y, "sess-1", "not json").status, 400);
  EXPECT_EQ(RegisterRaw(*f.y, "sess-1", "{\"token\":\"x\",\"use\":\"steal\"}").status, 400);
  EXPECT_EQ(RegisterRaw(*f.y, "sess-1", "{\"token\":\"a.b+c.d\",\"use\":\"seal\",\"key\":\"" + KeyB64(key) + "\"}").status,
            401) << "a non-base64url JWT segment was decoded";
}

// The verifier never runs without an issuer and an audience; the provider refuses to start misconfigured.
TEST(YardKeys, ConfigurationFailsClosed) {
  halo::server::Jwks none;
  halo::server::JwtError je{};
  EXPECT_FALSE(halo::server::VerifyJwt("a.b.c", none, {.issuer = "i", .audience = ""}, &je));
  EXPECT_EQ(je, halo::server::JwtError::kConfig);
  EXPECT_FALSE(halo::server::VerifyJwt("a.b.c", none, {.issuer = "", .audience = "a"}, &je));
  EXPECT_EQ(je, halo::server::JwtError::kConfig);
  Fixture f;
  const auto refuses = [&](auto edit) {
    YardSpillKeys::Options o = f.o;
    edit(o);
    std::string e;
    return YardSpillKeys::Create(o, &e) == nullptr && !e.empty();
  };
  EXPECT_TRUE(refuses([](auto& o) { o.audience = ""; })) << "empty audience accepted";
  EXPECT_TRUE(refuses([](auto& o) { o.issuer = ""; }));
  EXPECT_TRUE(refuses([](auto& o) { o.jwks_path = ""; }));
  EXPECT_TRUE(refuses([](auto& o) { o.key_ttl_s = 0; }));
  EXPECT_TRUE(refuses([](auto& o) { o.key_ttl_s = -5; }));
  EXPECT_TRUE(refuses([](auto& o) { o.key_ttl_s = 3601; }));
  EXPECT_TRUE(refuses([](auto& o) { o.jwks_refresh_s = 0; }));
  EXPECT_TRUE(refuses([&](auto& o) { o.jwks_path = f.dir + "/missing.json"; }));
  WriteFile(f.dir + "/empty.json", "{\"keys\":[]}");
  EXPECT_TRUE(refuses([&](auto& o) { o.jwks_path = f.dir + "/empty.json"; })) << "started with nothing to verify";
}

// Only sound RS256 signing keys are loaded: >= 2048-bit modulus (counted in bits, not encoded bytes), e = 65537,
// canonical encoding, use / key_ops for signatures; a duplicate kid invalidates the whole document.
TEST(YardKeys, JwksKeyChecks) {
  Signer good("good"), small("small", 1024, 65537), e3("e3", 2048, 3), padded("padded"), enc("enc"), ops("ops"),
      ops_ok("ops-ok");
  padded.pad_n = true;
  enc.extra = ",\"use\":\"enc\"";
  ops.extra = ",\"key_ops\":[\"sign\"]";
  ops_ok.extra = ",\"key_ops\":[\"verify\"]";
  halo::server::Jwks j;
  std::string e;
  ASSERT_TRUE(j.Parse(JwksJson({&good, &small, &e3, &padded, &enc, &ops, &ops_ok}), &e)) << e;
  EXPECT_TRUE(j.Find("good"));
  EXPECT_TRUE(j.Find("ops-ok"));
  EXPECT_FALSE(j.Find("small")) << "a 1024-bit key was loaded";
  EXPECT_FALSE(j.Find("e3")) << "e = 3 was loaded";
  EXPECT_FALSE(j.Find("padded")) << "a zero-padded modulus was loaded";
  EXPECT_FALSE(j.Find("enc")) << "an encryption key was loaded";
  EXPECT_FALSE(j.Find("ops")) << "a key without verify in key_ops was loaded";
  Signer dup("good");
  EXPECT_FALSE(j.Parse(JwksJson({&good, &dup}), &e)) << "duplicate kid accepted";
  EXPECT_NE(e.find("duplicate"), std::string::npos);
  EXPECT_TRUE(j.Find("good")) << "a refused document replaced the loaded set";
}

// The JWKS file is re-read when it changes: a removed (known) kid stops verifying at once, an empty or broken
// document denies everything, a restored one works again.
TEST(YardKeys, JwksRefreshRemovesKeys) {
  Fixture f;
  const std::string key = Key32(3);
  ASSERT_EQ(f.Reg("seal", 1, key).status, 200);
  const auto bump = [&](const std::string& doc) {  // a changed file (size or mtime) is re-read on the next call
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    WriteFile(f.jwks, doc);
  };
  bump(JwksJson({&f.other}));  // kid-a removed, kid-b published
  auto a = f.Reg("seal", 2, key, 2);
  EXPECT_EQ(a.status, 401) << "a removed signing key still verifies";
  EXPECT_EQ(a.reason, "unknown kid");
  EXPECT_EQ(Register(*f.y, f.other, "sess-1", "seal", 2, key, 2).status, 200);
  bump("{\"keys\":");
  EXPECT_EQ(Register(*f.y, f.other, "sess-1", "seal", 3, key, 3).status, 401) << "a broken set kept the old keys";
  bump(JwksJson({&f.other}));
  EXPECT_EQ(Register(*f.y, f.other, "sess-1", "seal", 3, key, 3).status, 200);
  bump("{\"keys\":[]}");
  EXPECT_EQ(Register(*f.y, f.other, "sess-1", "seal", 4, key, 4).status, 401) << "an empty set verified";
  bump(JwksJson({&f.signer}));
  EXPECT_EQ(f.Reg("seal", 4, key, 4).status, 200);
}

// Seal at E, open at E + 1 opens exactly that file; replay, stale, early open and a skipped epoch are refused or
// cannot open; keys are one-shot.
TEST(YardKeys, EpochRulesReplayDuplicatesAndShredding) {
  Fixture f;
  const std::string k1 = Key32(1), k2 = Key32(2);
  Claims c;
  c.epoch = 3;
  c.entry = 7;
  c.key_sha256 = KeyHash(k1);
  const std::string t = Token(f.signer, c);
  std::string body = Body(t, "seal", k1);
  auto a = f.y->Register("sess-1", body);
  ASSERT_EQ(a.status, 200) << a.body;
  EXPECT_TRUE(Has(a, "\"duplicate\":false"));
  body = Body(t, "seal", k1);
  EXPECT_EQ(f.y->Register("sess-1", body).status, 409) << "replayed jti accepted";
  a = f.Reg("seal", 3, k1, 7);  // a retry with a fresh token: a no-op
  ASSERT_EQ(a.status, 200);
  EXPECT_TRUE(Has(a, "\"duplicate\":true"));
  EXPECT_EQ(f.Reg("seal", 2, k1, 7).status, 409) << "stale epoch accepted";
  SpillLease seal, l;
  EXPECT_FALSE(f.Seal(8, "sess-1", false, &l)) << "a seal registration for entry 7 sealed entry 8";
  ASSERT_TRUE(f.Seal(7, "s123-7", false, &seal));
  EXPECT_EQ(seal.kek.id.rfind("yard:", 0), 0u);
  EXPECT_NE(seal.kek.id.find(":e3:sess-1"), std::string::npos) << seal.kek.id;
  EXPECT_TRUE(seal.yard);
  EXPECT_EQ(seal.session, "sess-1");
  EXPECT_FALSE(f.Seal(7, "sess-1", true, &l)) << "a reserved key leased twice";
  EXPECT_NE(f.y->ListJson().find("\"sealed_epoch\":3"), std::string::npos) << f.y->ListJson();
  EXPECT_EQ(f.y->ListJson().find(KeyB64(k1)), std::string::npos);
  f.y->Finish(seal, SpillKeys::Use::kSeal, true);  // the spill is done: one-shot
  EXPECT_FALSE(f.Seal(7, "sess-1", true, &l)) << "a used seal key is still held";
  // The file waits for yard: no open key yet.
  EXPECT_FALSE(f.Open("sess-1", seal.kek.id, &l));
  EXPECT_TRUE(f.y->MayOpen("sess-1", seal.kek.id));
  // An open at the seal's own epoch (a token minted before the suspend committed) is refused.
  EXPECT_EQ(f.Reg("open", 3, k1).status, 409);
  // Another key under a good open token does not open it (another key name, another KEK id).
  ASSERT_EQ(f.Reg("open", 4, k2).status, 200);
  EXPECT_FALSE(f.Open("sess-1", seal.kek.id, &l));
  // The open at E + 1 = 4 with the right key opens exactly the file sealed at 3.
  a = f.Reg("open", 4, k1);
  ASSERT_EQ(a.status, 200) << a.body;
  SpillLease open;
  ASSERT_TRUE(f.Open("sess-1", seal.kek.id, &open));
  EXPECT_EQ(open.kek.id, seal.kek.id);
  EXPECT_TRUE(sc::Equal(open.kek.key.span(), seal.kek.key.span()));
  f.y->Finish(open, SpillKeys::Use::kOpen, true);
  EXPECT_FALSE(f.y->MayOpen("sess-1", seal.kek.id)) << "a restored file may open again";
  // A skipped epoch: an open at 5 is not right after the seal (3).
  EXPECT_EQ(f.Reg("open", 5, k1).status, 409);
  // A late open at 4 again: used.
  EXPECT_EQ(f.Reg("open", 4, k2).status, 409);
  // Sessions are independent.
  EXPECT_EQ(f.Reg("seal", 1, k2, 9, "sess-2").status, 200);
  // An entry is sealed for one session: another session's seal for entry 9 is refused while one is live.
  EXPECT_EQ(f.Reg("seal", 6, k2, 9).status, 409);
}

// Revoke fences the session: no seal or open at a file epoch at or below the fence, with a fresh token or a
// delayed one minted before the revoke; repeating it is a no-op; a strictly later epoch starts a new lifecycle.
TEST(YardKeys, RevokeFencesTheSession) {
  Fixture f;
  const std::string k = Key32(5), k2 = Key32(6);
  ASSERT_EQ(f.Reg("seal", 4, k).status, 200);
  SpillLease seal;
  ASSERT_TRUE(f.Seal(1, "x", false, &seal));
  f.y->Finish(seal, SpillKeys::Use::kSeal, true);
  // A delayed open token: minted (for E + 1 = 5) before the revoke, delivered after it.
  Claims late;
  late.use = "open";
  late.epoch = 5;
  late.key_sha256 = KeyHash(k);
  std::string late_body = Body(Token(f.signer, late), "open", k);
  // Revoke at the session's epoch 4 (fence = 4).
  auto a = f.Reg("revoke", 4, "");
  ASSERT_EQ(a.status, 200) << a.body;
  EXPECT_EQ(a.revoked_session, "sess-1");
  EXPECT_EQ(a.revoked_through, 4u);
  EXPECT_TRUE(Has(a, "\"duplicate\":false"));
  EXPECT_FALSE(f.y->MayOpen("sess-1", seal.kek.id)) << "a revoked file may still open";
  EXPECT_EQ(f.Reg("seal", 4, k2).status, 409) << "a fresh seal at the revoked epoch";
  EXPECT_EQ(f.Reg("seal", 4, k).status, 409) << "a re-seal of the same key at the revoked epoch";
  EXPECT_EQ(f.Reg("open", 5, k).status, 409) << "a fresh open of the revoked file (E + 1)";
  EXPECT_EQ(f.y->Register("sess-1", late_body).status, 409) << "a delayed pre-revoke open token";
  SpillLease l;
  EXPECT_FALSE(f.Open("sess-1", seal.kek.id, &l));
  // Idempotent repeat (and a revoke never counts as stale).
  a = f.Reg("revoke", 4, "");
  ASSERT_EQ(a.status, 200);
  EXPECT_TRUE(Has(a, "\"duplicate\":true"));
  a = f.Reg("revoke", 9, "", 1, "sess-1", 3);  // through 3 < fence 4: still a no-op, the fence never goes back
  ASSERT_EQ(a.status, 200);
  EXPECT_EQ(a.revoked_through, 4u);
  // A later lifecycle (seal at 9 > fence) works.
  ASSERT_EQ(f.Reg("seal", 9, k2).status, 200);
  ASSERT_TRUE(f.Seal(1, "sess-1", true, &l));
  // GLM M1's repro: seal at E, revoke at E, a fresh open at E + 1 with the same K: refused.
  Fixture g;
  ASSERT_EQ(g.Reg("seal", 2, k).status, 200);
  ASSERT_TRUE(g.Seal(1, "x", false, &seal));
  g.y->Finish(seal, SpillKeys::Use::kSeal, true);
  ASSERT_EQ(g.Reg("revoke", 2, "").status, 200);
  EXPECT_EQ(g.Reg("open", 3, k).status, 409);
  EXPECT_FALSE(g.Open("sess-1", seal.kek.id, &l));
}

// One-shot is real: after a key was consumed, a fresh token for the same operation reports "consumed" and never
// re-arms it; another key at that epoch is refused.
TEST(YardKeys, ConsumedKeysAreNeverRearmed) {
  Fixture f;
  const std::string k = Key32(7), k2 = Key32(8);
  ASSERT_EQ(f.Reg("seal", 2, k).status, 200);
  SpillLease s, l;
  ASSERT_TRUE(f.Seal(1, "x", false, &s));
  f.y->Finish(s, SpillKeys::Use::kSeal, true);
  auto a = f.Reg("seal", 2, k);
  ASSERT_EQ(a.status, 200);
  EXPECT_TRUE(Has(a, "\"state\":\"consumed\"")) << a.body;
  EXPECT_FALSE(f.Seal(1, "sess-1", true, &l)) << "a consumed seal key was re-armed";
  EXPECT_EQ(f.Reg("seal", 2, k2).status, 409) << "a second key at a used seal epoch";
  ASSERT_EQ(f.Reg("open", 3, k).status, 200);
  ASSERT_TRUE(f.Open("sess-1", s.kek.id, &l));
  f.y->Finish(l, SpillKeys::Use::kOpen, true);
  a = f.Reg("open", 3, k);
  ASSERT_EQ(a.status, 200);
  EXPECT_TRUE(Has(a, "\"state\":\"consumed\""));
  EXPECT_FALSE(f.Open("sess-1", s.kek.id, &l)) << "a consumed open key was re-armed";
  // A failed spill does not consume: the same key seals the retry.
  ASSERT_EQ(f.Reg("seal", 5, k2, 2).status, 200);
  ASSERT_TRUE(f.Seal(2, "x", false, &s));
  f.y->Finish(s, SpillKeys::Use::kSeal, false);
  ASSERT_TRUE(f.Seal(2, "x", false, &l)) << "a failed spill consumed its key";
}

// Expiry zeroes keys (not just hides them), by the sweeper alone; authorization ends at the token's exp when that
// comes before registration + TTL.
TEST(YardKeys, ExpiryZeroesKeys) {
  Fixture f(nullptr, false, 1);  // TTL 1 s
  const long live = sc::SecretBytes::Live();
  ASSERT_EQ(f.Reg("seal", 1, Key32(9)).status, 200);
  EXPECT_EQ(sc::SecretBytes::Live(), live + 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  EXPECT_EQ(f.y->Sweep(), 1u);
  EXPECT_EQ(sc::SecretBytes::Live(), live) << "an expired key is still mapped";
  SpillLease l;
  EXPECT_FALSE(f.Seal(1, "x", false, &l));
  // TTL 600 s but a token that expires in 2 s: the key goes with the token.
  Fixture g;
  const std::int64_t now = std::time(nullptr);
  Claims c;
  c.epoch = 1;
  c.iat = now - 298;
  c.exp = now + 2;
  c.key_sha256 = KeyHash(Key32(9));
  std::string body = Body(Token(g.signer, c), "seal", Key32(9));
  ASSERT_EQ(g.y->Register("sess-1", body).status, 200);
  std::this_thread::sleep_for(std::chrono::milliseconds(2300));
  EXPECT_FALSE(g.Seal(1, "x", false, &l)) << "a key outlived its token's exp";
  // The sweeper thread zeroes an idle key with nobody calling in.
  Fixture h(nullptr, false, 1);
  h.o.sweeper = true;
  h.Restart();
  const long live2 = sc::SecretBytes::Live();
  ASSERT_EQ(h.Reg("seal", 1, Key32(10)).status, 200);
  EXPECT_EQ(sc::SecretBytes::Live(), live2 + 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(2500));
  EXPECT_EQ(sc::SecretBytes::Live(), live2) << "the sweeper did not zero an expired key";
}

// The per-session epochs are persisted before a registration is acknowledged and before a key is used: after a
// restart (keys and jtis gone) a replayed seal token of a spilled epoch is refused, the open after the seal still
// works, and a revoke fence holds. A malformed or loose state file refuses the start.
TEST(YardKeys, EpochsSurviveARestart) {
  Fixture f(nullptr, true);
  const std::string k = Key32(11);
  Claims c;
  c.epoch = 3;
  c.key_sha256 = KeyHash(k);
  const std::string seal_token = Token(f.signer, c);
  std::string body = Body(seal_token, "seal", k);
  ASSERT_EQ(f.y->Register("sess-1", body).status, 200);
  SpillLease s, l;
  ASSERT_TRUE(f.Seal(1, "x", false, &s));
  f.y->Finish(s, SpillKeys::Use::kSeal, true);
  ASSERT_EQ(f.Reg("seal", 1, k, 1, "sess-2").status, 200);
  ASSERT_EQ(f.Reg("revoke", 1, "", 1, "sess-2").status, 200);
  struct stat st {};
  ASSERT_EQ(::stat(f.o.state_path.c_str(), &st), 0);
  EXPECT_EQ(st.st_mode & 0777, 0600u);
  f.Restart();
  body = Body(seal_token, "seal", k);
  EXPECT_EQ(f.y->Register("sess-1", body).status, 409) << "a replayed seal token re-armed a spilled epoch";
  EXPECT_EQ(f.Reg("seal", 3, Key32(12)).status, 409);
  ASSERT_EQ(f.Reg("open", 4, k).status, 200) << "the open after the seal is refused after a restart";
  ASSERT_TRUE(f.Open("sess-1", s.kek.id, &l));
  EXPECT_EQ(f.Reg("seal", 1, k, 1, "sess-2").status, 409) << "a revoke fence was lost in the restart";
  // A crash between OpenLease and Finish: the used fence is already on disk.
  f.Restart();
  EXPECT_EQ(f.Reg("open", 4, k).status, 409) << "an open token registered again after its restore began";
  // Refused starts.
  f.y.reset();
  WriteFile(f.o.state_path, "sess-1 3 - x 1 1\n");
  ::chmod(f.o.state_path.c_str(), 0600);
  std::string e;
  EXPECT_FALSE(YardSpillKeys::Create(f.o, &e)) << "a malformed state file was accepted";
  WriteFile(f.o.state_path, "sess-1 3 - 3 3 0\n");
  ::chmod(f.o.state_path.c_str(), 0644);
  EXPECT_FALSE(YardSpillKeys::Create(f.o, &e)) << "a group-readable state file was accepted";
  ::chmod(f.o.state_path.c_str(), 0600);
  EXPECT_TRUE(YardSpillKeys::Create(f.o, &e)) << e;
}

// Leases are bound to the registration they came from: a lease finished after its slot was revoked and replaced
// never consumes the newer key. A restore in flight at a revoke finishes with the key it holds; the revoke's
// cleanup (DropRevoked through the fence) then erases what it produced.
TEST(YardKeys, LeasesAreGenerationBound) {
  Fixture f;
  const std::string k = Key32(13), k2 = Key32(14);
  ASSERT_EQ(f.Reg("seal", 2, k).status, 200);
  SpillLease a, b;
  ASSERT_TRUE(f.Seal(1, "x", false, &a));
  EXPECT_EQ(f.Reg("seal", 2, k2).status, 409) << "a reserved key was replaced";
  ASSERT_EQ(f.Reg("revoke", 2, "").status, 200);  // zeroes the reserved slot
  ASSERT_EQ(f.Reg("seal", 3, k2).status, 200);
  f.y->Finish(a, SpillKeys::Use::kSeal, true);  // the old lease completes late
  ASSERT_TRUE(f.Seal(1, "x", false, &b)) << "a stale lease consumed the newer registration";
  EXPECT_NE(b.gen, a.gen);
  f.y->Finish(b, SpillKeys::Use::kSeal, true);
  // In-flight restore at a revoke.
  ASSERT_EQ(f.Reg("open", 4, k2).status, 200);
  SpillLease o;
  ASSERT_TRUE(f.Open("sess-1", b.kek.id, &o));
  const auto r = f.Reg("revoke", 4, "", 1, "sess-1", 3);
  ASSERT_EQ(r.status, 200);
  EXPECT_EQ(r.revoked_through, 3u);
  EXPECT_TRUE(o.kek.key.ok()) << "the lease's key copy must survive until the restore finishes";
  f.y->Finish(o, SpillKeys::Use::kOpen, true);  // no slot of its generation left: nothing happens
  EXPECT_FALSE(f.y->MayOpen("sess-1", b.kek.id));
}

// Custody: an entry once sealed under a yard key never falls back to the master; an entry never yard-bound may
// (with --spill-yard-fallback master); a yard-sealed file never opens through the fallback.
TEST(YardKeys, NoCustodyDowngrade) {
  sc::SecretBytes m(32);
  ASSERT_TRUE(sc::RandomBytes({m.data(), 32}));
  std::string e;
  auto master = halo::server::MasterSpillKeys::Create("file", std::move(m), "", &e);
  ASSERT_TRUE(master) << e;
  Fixture none, fb(master.get());
  SpillLease l, l2;
  std::string why;
  EXPECT_FALSE(none.Seal(1, "s1", false, &l));
  ASSERT_TRUE(fb.Seal(1, "s1", false, &l, &why)) << why;
  EXPECT_EQ(l.kek.id.rfind("file:", 0), 0u);
  EXPECT_FALSE(l.yard);
  ASSERT_TRUE(fb.Open("s1", l.kek.id, &l2, &why)) << why;
  EXPECT_EQ(l2.kek.id, l.kek.id);
  EXPECT_FALSE(fb.Open("s1", "yard:0011223344556677:e3:s1", &l2)) << "a yard file fell back";
  EXPECT_FALSE(fb.Seal(1, "sess-1", true, &l, &why)) << "a yard-bound entry was sealed under the master";
  EXPECT_NE(why.find("yard-bound"), std::string::npos) << why;
  EXPECT_FALSE(master->SealLease(1, "sess-1", true, &l, &why)) << "the master sealed a yard-bound entry";
  // A yard-bound entry of sess-1 is not sealed by sess-2's registration.
  ASSERT_EQ(fb.Reg("seal", 1, Key32(1), 5, "sess-2").status, 200);
  EXPECT_FALSE(fb.Seal(5, "sess-1", true, &l, &why));
}
