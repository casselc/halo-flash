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
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "server/state_file.hpp"
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
    o.incarnation = kInc;
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
  bool Seal(std::int64_t entry, const std::string& label, bool bound, SpillLease* l, std::string* why = nullptr) {
    std::string w;
    return y->SealLease(Entry(entry), label, bound, l, why ? why : &w);
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
      {"seal without an entry", with([](Claims& c) { c.entry = ""; }), "entry"},
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
  // The verifier itself has no post-exp leeway (the registration's own re-check under the lock is a second layer).
  {
    Claims c;
    c.iat = now - 300;
    c.exp = now;
    halo::server::JwtError je{};
    halo::server::Jwks jwks;
    std::string pe;
    ASSERT_TRUE(jwks.Parse(JwksJson({&f.signer}), &pe)) << pe;
    const std::string tok = Token(f.signer, c);
    EXPECT_TRUE(halo::server::VerifyJwt(tok, jwks, {.issuer = c.iss, .audience = c.aud, .now = now - 1}, &je));
    EXPECT_FALSE(halo::server::VerifyJwt(tok, jwks, {.issuer = c.iss, .audience = c.aud, .now = now}, &je));
    EXPECT_EQ(je, halo::server::JwtError::kExpired);
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
  EXPECT_TRUE(refuses([](auto& o) { o.incarnation = "0f0f"; })) << "a 16-bit incarnation";
  EXPECT_TRUE(refuses([](auto& o) { o.incarnation = std::string(32, 'G'); }));
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
  c.entry = Entry(7);
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
  f.y->Finish(seal, SpillKeys::Use::kSeal, SpillKeys::Outcome::kDone);  // the spill is done: one-shot
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
  f.y->Finish(open, SpillKeys::Use::kOpen, SpillKeys::Outcome::kDone);
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
  f.y->Finish(seal, SpillKeys::Use::kSeal, SpillKeys::Outcome::kDone);
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
  g.y->Finish(seal, SpillKeys::Use::kSeal, SpillKeys::Outcome::kDone);
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
  f.y->Finish(s, SpillKeys::Use::kSeal, SpillKeys::Outcome::kDone);
  auto a = f.Reg("seal", 2, k);
  ASSERT_EQ(a.status, 200);
  EXPECT_TRUE(Has(a, "\"state\":\"consumed\"")) << a.body;
  EXPECT_FALSE(f.Seal(1, "sess-1", true, &l)) << "a consumed seal key was re-armed";
  EXPECT_EQ(f.Reg("seal", 2, k2).status, 409) << "a second key at a used seal epoch";
  ASSERT_EQ(f.Reg("open", 3, k).status, 200);
  ASSERT_TRUE(f.Open("sess-1", s.kek.id, &l));
  f.y->Finish(l, SpillKeys::Use::kOpen, SpillKeys::Outcome::kDone);
  a = f.Reg("open", 3, k);
  ASSERT_EQ(a.status, 200);
  EXPECT_TRUE(Has(a, "\"state\":\"consumed\""));
  EXPECT_FALSE(f.Open("sess-1", s.kek.id, &l)) << "a consumed open key was re-armed";
  // A failed spill does not consume: the same key seals the retry.
  ASSERT_EQ(f.Reg("seal", 5, k2, 2).status, 200);
  ASSERT_TRUE(f.Seal(2, "x", false, &s));
  f.y->Finish(s, SpillKeys::Use::kSeal, SpillKeys::Outcome::kNotDone);
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
  f.y->Finish(s, SpillKeys::Use::kSeal, SpillKeys::Outcome::kDone);
  ASSERT_EQ(f.Reg("seal", 1, k, 1, "sess-2").status, 200);
  ASSERT_EQ(f.Reg("revoke", 1, "", 1, "sess-2", 1).status, 200);  // through 1: nothing was spilled there
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
  f.y->Finish(a, SpillKeys::Use::kSeal, SpillKeys::Outcome::kDone);  // the old lease completes late
  ASSERT_TRUE(f.Seal(1, "x", false, &b)) << "a stale lease consumed the newer registration";
  EXPECT_NE(b.gen, a.gen);
  f.y->Finish(b, SpillKeys::Use::kSeal, SpillKeys::Outcome::kDone);
  // In-flight restore at a revoke.
  ASSERT_EQ(f.Reg("open", 4, k2).status, 200);
  SpillLease o;
  ASSERT_TRUE(f.Open("sess-1", b.kek.id, &o));
  const auto r = f.Reg("revoke", 4, "", 1, "sess-1", 3);
  ASSERT_EQ(r.status, 200);
  EXPECT_EQ(r.revoked_through, 3u);
  EXPECT_TRUE(o.kek.key.ok()) << "the lease's key copy must survive until the restore finishes";
  f.y->Finish(o, SpillKeys::Use::kOpen, SpillKeys::Outcome::kDone);  // no slot of its generation left: nothing happens
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
  EXPECT_FALSE(master->SealLease("x.1", "sess-1", true, &l, &why)) << "the master sealed a yard-bound entry";
  // A yard-bound entry of sess-1 is not sealed by sess-2's registration.
  ASSERT_EQ(fb.Reg("seal", 1, Key32(1), 5, "sess-2").status, 200);
  EXPECT_FALSE(fb.Seal(5, "sess-1", true, &l, &why));
}

// ---- round 3 ----

// A seal authorization names an entry of one server incarnation: after a restart (a new incarnation) the same
// numeric entry key may name another conversation, so a token minted before the restart is refused.
TEST(YardKeys, SealEntriesAreIncarnationBound) {
  Fixture f(nullptr, true);
  const std::string k = Key32(21);
  Claims c;
  c.epoch = 2;
  c.entry = Entry(1);
  c.key_sha256 = KeyHash(k);
  const std::string token = Token(f.signer, c);
  std::string body = Body(token, "seal", k);
  ASSERT_EQ(f.y->Register("sess-1", body).status, 200);
  f.o.incarnation = "0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e";  // the next process
  f.Restart();
  body = Body(token, "seal", k);  // the same, unused authorization, delivered again after the restart
  const auto a = f.y->Register("sess-1", body);
  EXPECT_EQ(a.status, 409) << "a pre-restart seal authorization was accepted: " << a.body;
  EXPECT_TRUE(Has(a, "incarnation")) << a.body;
  SpillLease l;
  std::string why;
  EXPECT_FALSE(f.y->SealLease(Entry(1, "0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e"), "x", false, &l, &why)) << "entry 1 of the new process got the old key";
  EXPECT_EQ(f.Reg("seal", 2, k, 1).status, 409) << "an entry of the old incarnation (fresh token) was accepted";
  Claims nc;  // a non-canonical key ("01"): never an id the cache generates
  nc.epoch = 2;
  nc.entry = Entry(1, "0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e").substr(0, 33) + "01";
  nc.key_sha256 = KeyHash(k);
  body = Body(Token(f.signer, nc), "seal", k);
  EXPECT_EQ(f.y->Register("sess-1", body).status, 409) << "a non-canonical entry id was accepted";
  ASSERT_EQ(Register(*f.y, f.signer, "sess-1", "seal", 2, k, 0).status, 401);  // no entry at all
  Claims ok;
  ok.epoch = 2;
  ok.entry = Entry(1, "0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e");
  ok.key_sha256 = KeyHash(k);
  body = Body(Token(f.signer, ok), "seal", k);
  EXPECT_EQ(f.y->Register("sess-1", body).status, 200);
  EXPECT_TRUE(f.y->SealLease(Entry(1, "0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e"), "x", false, &l, &why)) << why;
}

// One owner per state file: a second provider (another halo_serve on the same spill directory) is refused.
TEST(YardKeys, StateHasOneOwner) {
  Fixture f(nullptr, true);
  std::string e;
  EXPECT_FALSE(YardSpillKeys::Create(f.o, &e)) << "two providers own one state file";
  EXPECT_NE(e.find("held"), std::string::npos) << e;
  sc::SecretBytes m1(32), m2(32);
  ASSERT_TRUE(sc::RandomBytes({m1.data(), 32}));
  std::memcpy(m2.data(), m1.data(), 32);
  auto a = halo::server::MasterSpillKeys::Create("file", std::move(m1), f.dir + "/m.epoch", &e);
  ASSERT_TRUE(a) << e;
  EXPECT_FALSE(halo::server::MasterSpillKeys::Create("file", std::move(m2), f.dir + "/m.epoch", &e));
  EXPECT_NE(e.find("held"), std::string::npos) << e;
}

// Typed persistence. Unpublished: nothing changes (no session, no slot, old slots intact). Uncertain (published, the
// directory fsync failed): the provider stops (keys zeroed, everything refused) and a restart sees the disk's state.
TEST(YardKeys, PersistenceOutcomesAreTyped) {
  Fixture f(nullptr, true);
  const std::string k = Key32(22), k2 = Key32(23);
  ASSERT_EQ(f.Reg("seal", 2, k).status, 200);
  halo::server::g_state_write_fault = 1;
  auto a = f.Reg("seal", 3, k2);
  EXPECT_EQ(a.status, 500) << a.body;
  SpillLease l;
  ASSERT_TRUE(f.Seal(1, "x", false, &l)) << "an unpublished registration destroyed the live slot";
  EXPECT_EQ(l.epoch, 2u);
  f.y->Finish(l, SpillKeys::Use::kSeal, SpillKeys::Outcome::kNotDone);
  halo::server::g_state_write_fault = 1;
  EXPECT_EQ(f.Reg("seal", 1, k, 9, "sess-new").status, 500);
  EXPECT_EQ(f.y->ListJson().find("sess-new"), std::string::npos) << "an unpublished session exists";
  // A lease whose fence cannot be published is refused, and the slot stays usable.
  halo::server::g_state_write_fault = 1;
  EXPECT_FALSE(f.Seal(1, "x", false, &l));
  EXPECT_TRUE(f.Seal(1, "x", false, &l));
  f.y->Finish(l, SpillKeys::Use::kSeal, SpillKeys::Outcome::kNotDone);
  // Uncertain: stop.
  halo::server::g_state_write_fault = 2;
  a = f.Reg("seal", 3, k2);
  EXPECT_EQ(a.status, 503) << a.body;
  EXPECT_TRUE(f.y->stopped());
  EXPECT_FALSE(f.Seal(1, "x", false, &l)) << "a stopped provider leased a key";
  EXPECT_EQ(f.Reg("seal", 4, k2).status, 503);
  f.Restart();  // the disk shows epoch 3 (published): the stricter state
  EXPECT_FALSE(f.y->stopped());
  EXPECT_NE(f.y->ListJson().find("\"high_epoch\":3"), std::string::npos) << f.y->ListJson();
  EXPECT_EQ(f.Reg("seal", 2, k).status, 409) << "the restart lost the published epoch";
  // A revoke that cannot be published still zeroed the keys (fail-closed) and answers "not done".
  ASSERT_EQ(f.Reg("seal", 3, k2).status, 200);
  halo::server::g_state_write_fault = 1;
  EXPECT_EQ(f.Reg("revoke", 3, "").status, 500);
  EXPECT_FALSE(f.Seal(1, "x", false, &l)) << "a failed revoke left the key usable";
  // Master: unpublished keeps the epoch; uncertain takes the new one (what the disk shows).
  sc::SecretBytes m(32);
  ASSERT_TRUE(sc::RandomBytes({m.data(), 32}));
  std::string e;
  auto mk = halo::server::MasterSpillKeys::Create("file", std::move(m), f.dir + "/m.epoch", &e);
  ASSERT_TRUE(mk) << e;
  halo::server::g_state_write_fault = 1;
  EXPECT_FALSE(mk->FenceEpoch(&e));
  EXPECT_EQ(mk->epoch(), 1u);
  halo::server::g_state_write_fault = 2;
  EXPECT_FALSE(mk->FenceEpoch(&e));
  EXPECT_EQ(mk->epoch(), 2u) << "memory disagrees with the published epoch";
  EXPECT_NE(e.find("uncertain"), std::string::npos) << e;
  // Nothing is sealed at an unconfirmed epoch: the next seal re-writes it and proceeds only once that is durable.
  SpillLease ml;
  halo::server::g_state_write_fault = 1;
  EXPECT_FALSE(mk->SealLease("x.1", "a", false, &ml, &e)) << "a file was sealed at an epoch that may revert";
  EXPECT_TRUE(mk->SealLease("x.1", "a", false, &ml, &e)) << e;
  EXPECT_NE(ml.kek.id.find(":e2:"), std::string::npos) << ml.kek.id;
}

// A failed seal whose rollback cannot be published keeps the used fence and disables the key: neither a direct
// lease nor a fresh-token duplicate registration uses it again. A rollback after the slot was revoked meanwhile
// still puts the used fence back (GLM NN1).
TEST(YardKeys, FailedSealRollback) {
  Fixture f(nullptr, true);
  const std::string k = Key32(24);
  ASSERT_EQ(f.Reg("seal", 2, k).status, 200);
  SpillLease l;
  ASSERT_TRUE(f.Seal(1, "x", false, &l));
  halo::server::g_state_write_fault = 1;
  f.y->Finish(l, SpillKeys::Use::kSeal, SpillKeys::Outcome::kNotDone);
  EXPECT_NE(f.y->ListJson().find("\"seal\":null"), std::string::npos) << "the key was not disabled: " << f.y->ListJson();
  EXPECT_FALSE(f.Seal(1, "x", false, &l)) << "a key whose rollback failed was leased again";
  const auto a = f.Reg("seal", 2, k);
  EXPECT_EQ(a.status, 409) << "a fresh-token duplicate re-armed it: " << a.body;
  // Published but not durable: the key counts as used.
  ASSERT_EQ(f.Reg("seal", 3, Key32(25), 2).status, 200);
  ASSERT_TRUE(f.Seal(2, "x", false, &l));
  f.y->Finish(l, SpillKeys::Use::kSeal, SpillKeys::Outcome::kUncertain);
  EXPECT_FALSE(f.Seal(2, "x", false, &l));
  EXPECT_TRUE(Has(f.Reg("seal", 3, Key32(25), 2), "\"state\":\"consumed\""));
  // A failed spill whose slot survived: the used fence goes back and the key seals the retry.
  ASSERT_EQ(f.Reg("seal", 5, Key32(26), 3).status, 200);
  ASSERT_TRUE(f.Seal(3, "x", false, &l));
  ASSERT_EQ(f.Reg("revoke", 4, "", 1, "sess-1", 4).status, 200);  // a revoke at 4: the registration at 5 survives
  ASSERT_TRUE(f.y->ListJson().find("\"spilled_epoch\":5") != std::string::npos) << f.y->ListJson();
  f.y->Finish(l, SpillKeys::Use::kSeal, SpillKeys::Outcome::kNotDone);
  EXPECT_NE(f.y->ListJson().find("\"spilled_epoch\":3"), std::string::npos) << f.y->ListJson();
  SpillLease l2;
  EXPECT_TRUE(f.Seal(3, "x", false, &l2)) << "the failed spill's key was not left for a retry";
  f.y->Finish(l2, SpillKeys::Use::kSeal, SpillKeys::Outcome::kDone);
  // NN1: the slot revoked during the lease, then the spill failed: the used fence still goes back (the revoke's own
  // fence holds the epoch).
  ASSERT_EQ(f.Reg("seal", 7, Key32(27), 4).status, 200);
  ASSERT_TRUE(f.Seal(4, "x", false, &l));
  ASSERT_EQ(f.Reg("revoke", 7, "", 1, "sess-1", 7).status, 200);  // zeroes the reserved slot
  f.y->Finish(l, SpillKeys::Use::kSeal, SpillKeys::Outcome::kNotDone);
  EXPECT_NE(f.y->ListJson().find("\"spilled_epoch\":5"), std::string::npos) << f.y->ListJson();
  EXPECT_NE(f.y->ListJson().find("\"revoked_through\":7"), std::string::npos) << f.y->ListJson();
}

// A registration never destroys a slot a lease holds, the other use's included; both orders.
TEST(YardKeys, OppositeReservedSlotsSurvive) {
  Fixture f;
  const std::string k = Key32(27), k2 = Key32(28);
  ASSERT_EQ(f.Reg("seal", 2, k).status, 200);
  SpillLease s, o;
  ASSERT_TRUE(f.Seal(1, "x", false, &s));
  EXPECT_EQ(f.Reg("open", 3, k).status, 409) << "an open registration destroyed a reserved seal slot";
  f.y->Finish(s, SpillKeys::Use::kSeal, SpillKeys::Outcome::kDone);
  ASSERT_EQ(f.Reg("open", 3, k).status, 200);
  ASSERT_TRUE(f.Open("sess-1", s.kek.id, &o));
  EXPECT_EQ(f.Reg("seal", 4, k2).status, 409) << "a seal registration destroyed a reserved open slot";
  f.y->Finish(o, SpillKeys::Use::kOpen, SpillKeys::Outcome::kDone);
  EXPECT_EQ(f.Reg("seal", 4, k2).status, 200);
}

// The revoke default fences what was spilled, not the token epoch: an abort before any spill at E forgets the keys
// but leaves E sealable (GLM NL3); after a spill at E the default fences E.
TEST(YardKeys, RevokeDefaultFencesWhatWasSpilled) {
  Fixture f;
  const std::string k = Key32(29);
  ASSERT_EQ(f.Reg("seal", 3, k).status, 200);
  auto a = f.Reg("revoke", 3, "");
  ASSERT_EQ(a.status, 200);
  EXPECT_TRUE(Has(a, "\"through\":null")) << a.body;
  SpillLease l;
  EXPECT_FALSE(f.Seal(1, "x", false, &l)) << "a revoke kept the key";
  ASSERT_EQ(f.Reg("seal", 3, Key32(30)).status, 200) << "an abort before any spill poisoned the epoch";
  ASSERT_TRUE(f.Seal(1, "x", false, &l));
  f.y->Finish(l, SpillKeys::Use::kSeal, SpillKeys::Outcome::kDone);
  a = f.Reg("revoke", 3, "");
  ASSERT_EQ(a.status, 200);
  EXPECT_EQ(a.revoked_through, 3u);
  EXPECT_EQ(f.Reg("open", 4, Key32(30)).status, 409);
}

// State files are loaded only whole, safe and consistent; a state that was initialized and then vanished refuses.
TEST(YardKeys, StateLoadingRefusesUnsafeOrPartialState) {
  Fixture f(nullptr, true);
  ASSERT_EQ(f.Reg("seal", 2, Key32(31)).status, 200);
  f.y.reset();
  std::string e;
  const auto refuses = [&](const std::string& text, mode_t mode = 0600) {
    std::filesystem::remove(f.o.state_path);
    WriteFile(f.o.state_path, text);
    ::chmod(f.o.state_path.c_str(), mode);
    auto y = YardSpillKeys::Create(f.o, &e);
    return !y;
  };
  EXPECT_FALSE(refuses("sess-1 2 - 2 0 0\n")) << e;
  EXPECT_TRUE(refuses("sess-1 2 - 2 0 0\nsess-1 3 - 3 0 0\n")) << "duplicate session records";
  EXPECT_TRUE(refuses("sess-1 2 - 2 0 0\njunk")) << "a valid prefix plus junk";
  EXPECT_TRUE(refuses("sess-1 2 - 2 0 0")) << "an unterminated last line";
  EXPECT_TRUE(refuses("sess-1  2 - 2 0 0\n")) << "a double space";
  EXPECT_TRUE(refuses("sess-1 2 5 2 0 0\n")) << "a fence above the highest epoch";
  EXPECT_TRUE(refuses("sess-1 2 - 2 0 0\n", 0640)) << "group-readable";
  std::filesystem::remove(f.o.state_path);
  WriteFile(f.dir + "/other", "sess-1 2 - 2 0 0\n");
  ::chmod((f.dir + "/other").c_str(), 0600);
  std::filesystem::create_symlink(f.dir + "/other", f.o.state_path);
  EXPECT_FALSE(YardSpillKeys::Create(f.o, &e)) << "a symlinked state file";
  std::filesystem::remove(f.o.state_path);
  EXPECT_FALSE(YardSpillKeys::Create(f.o, &e)) << "an initialized state vanished and was silently reset";
  EXPECT_NE(e.find("recovery required"), std::string::npos) << e;
  std::filesystem::remove(f.o.state_path + ".init");
  EXPECT_TRUE(YardSpillKeys::Create(f.o, &e)) << "a deliberate reset (marker removed) must start: " << e;
  // The master's epoch file: exactly "<digits>\n".
  sc::SecretBytes m(32);
  ASSERT_TRUE(sc::RandomBytes({m.data(), 32}));
  const std::string ef = f.dir + "/m.epoch";
  WriteFile(ef, "2\njunk");
  ::chmod(ef.c_str(), 0600);
  WriteFile(ef + ".init", "initialized\n");
  EXPECT_FALSE(halo::server::MasterSpillKeys::Create("file", std::move(m), ef, &e)) << "epoch prefix plus junk";
}

// An unknown kid forces a JWKS re-read at most once per gap (a flood cannot keep the provider loading keys), and the
// loading never holds the provider lock (leases and the sweeper proceed).
TEST(YardKeys, UnknownKidFloodIsRateLimited) {
  Fixture f;
  f.o.jwks_forced_gap_ms = 60000;
  f.Restart();
  const auto before = f.y->jwks_loads();
  ASSERT_EQ(f.Reg("seal", 1, Key32(32)).status, 200);
  for (int i = 0; i < 50; ++i) EXPECT_EQ(Register(*f.y, f.other, "sess-1", "seal", 2, Key32(33), 2).status, 401);
  EXPECT_LE(f.y->jwks_loads() - before, 1u) << "unknown kids forced unbounded JWKS reloads";
  SpillLease l;
  EXPECT_TRUE(f.Seal(1, "x", false, &l));
  EXPECT_EQ(f.y->Sweep(), 0u);
}

// A retry with a fresh token never extends a key past its first registration + TTL.
TEST(YardKeys, RetriesNeverPassTheFirstTtl) {
  Fixture f(nullptr, false, 2);
  const std::string k = Key32(34);
  ASSERT_EQ(f.Reg("seal", 1, k).status, 200);
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  ASSERT_TRUE(Has(f.Reg("seal", 1, k), "\"duplicate\":true"));
  std::this_thread::sleep_for(std::chrono::milliseconds(700));
  SpillLease l;
  EXPECT_FALSE(f.Seal(1, "x", false, &l)) << "a retry extended the key past registration + TTL";
}

// Duplicates are decided by the whole key: two keys whose (display) names collide are different registrations.
TEST(YardKeys, KeyNamesNeverDecideEquality) {
  Fixture f;
  sc::KeyNameCollideForTest(true);
  ASSERT_EQ(f.Reg("seal", 1, Key32(35)).status, 200);
  const auto a = f.Reg("seal", 1, Key32(36));
  sc::KeyNameCollideForTest(false);
  ASSERT_EQ(a.status, 200);
  EXPECT_TRUE(Has(a, "\"duplicate\":false")) << "a different key with the same name was taken for a duplicate";
  SpillLease l;
  ASSERT_TRUE(f.Seal(1, "x", false, &l));
  sc::Kek want;
  std::string e;
  const std::string k2 = Key32(36);
  ASSERT_TRUE(sc::DeriveKek({reinterpret_cast<const std::uint8_t*>(k2.data()), 32}, "sess-1", 1, "yard:x", &want, &e));
  EXPECT_TRUE(sc::Equal(l.kek.key.span(), want.key.span())) << "the lease holds the first key";
}

// ---- round 4 ----

// Every revoke effect is bounded by its token's epoch T: a delayed revoke (no `through`) never fences a checkpoint
// spilled after T, and never zeroes a registration made after T; it still fences what it covers.
TEST(YardKeys, RevokeIsBoundedByItsTokenEpoch) {
  Fixture f;
  const std::string k1 = Key32(41), k2 = Key32(42);
  ASSERT_EQ(f.Reg("seal", 1, k1, 1).status, 200);
  SpillLease l;
  ASSERT_TRUE(f.Seal(1, "x", false, &l));
  f.y->Finish(l, SpillKeys::Use::kSeal, SpillKeys::Outcome::kDone);
  Claims late;  // minted at T = 1, delivered after the session moved on
  late.use = "revoke";
  late.epoch = 1;
  std::string late_body = Body(Token(f.signer, late), "revoke", "");
  ASSERT_EQ(f.Reg("seal", 2, k2, 2).status, 200);
  SpillLease l2;
  ASSERT_TRUE(f.Seal(2, "x", false, &l2));
  f.y->Finish(l2, SpillKeys::Use::kSeal, SpillKeys::Outcome::kDone);  // a checkpoint spilled at T + 1
  auto a = f.y->Register("sess-1", late_body);
  ASSERT_EQ(a.status, 200) << a.body;
  EXPECT_EQ(a.revoked_through, 1u) << "a revoke at T = 1 fenced past its epoch: " << a.body;
  EXPECT_TRUE(f.y->MayOpen("sess-1", l2.kek.id)) << "the newer checkpoint was fenced by an older revoke";
  EXPECT_FALSE(f.y->MayOpen("sess-1", l.kek.id)) << "the revoke's own checkpoint was not fenced";
  EXPECT_EQ(f.Reg("open", 3, k2).status, 200) << "the newer checkpoint can no longer be opened";
  // A registration after T with nothing spilled: the delayed revoke leaves it alone.
  Fixture g;
  Claims late2;
  late2.use = "revoke";
  late2.epoch = 1;
  std::string body2 = Body(Token(g.signer, late2), "revoke", "");
  ASSERT_EQ(g.Reg("seal", 2, k2, 1).status, 200);
  a = g.y->Register("sess-1", body2);
  ASSERT_EQ(a.status, 200) << a.body;
  EXPECT_TRUE(Has(a, "\"through\":null")) << a.body;
  SpillLease l3;
  EXPECT_TRUE(g.Seal(1, "x", false, &l3)) << "an older revoke zeroed a newer registration";
  // A registration at T is a pending operation the token may cancel.
  Fixture h;
  ASSERT_EQ(h.Reg("seal", 1, k1, 1).status, 200);
  ASSERT_EQ(h.Reg("revoke", 1, "").status, 200);
  EXPECT_FALSE(h.Seal(1, "x", false, &l3));
}

// The token is checked again at the transition boundary: one that expired while the state lock was held (another
// registration's fsync) is refused, and a key never outlives the token's absolute exp.
TEST(YardKeys, ExpiryIsCheckedUnderTheLock) {
  Fixture f;
  const std::int64_t now = std::time(nullptr);
  Claims c;
  c.epoch = 1;
  c.iat = now - 298;
  c.exp = now + 2;
  c.key_sha256 = KeyHash(Key32(43));
  std::string body = Body(Token(f.signer, c), "seal", Key32(43));
  YardSpillKeys::Answer a;
  {
    auto hold = f.y->LockForTest();
    std::thread t([&] { a = f.y->Register("sess-1", body); });
    std::this_thread::sleep_for(std::chrono::milliseconds(3200));  // past exp while the request waits
    hold.unlock();
    t.join();
  }
  EXPECT_EQ(a.status, 401) << "a token that expired while waiting took effect: " << a.body;
  EXPECT_NE(a.reason.find("expired"), std::string::npos) << a.reason;
  // Waiting, but not past exp: accepted, and the key still ends at exp (not exp + the wait).
  const std::int64_t n2 = std::time(nullptr);
  Claims c2;
  c2.epoch = 2;
  c2.iat = n2 - 297;
  c2.exp = n2 + 3;
  c2.key_sha256 = KeyHash(Key32(44));
  body = Body(Token(f.signer, c2), "seal", Key32(44));
  {
    auto hold = f.y->LockForTest();
    std::thread t([&] { a = f.y->Register("sess-1", body); });
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    hold.unlock();
    t.join();
  }
  ASSERT_EQ(a.status, 200) << a.body;
  while (std::time(nullptr) < c2.exp) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  SpillLease l;
  EXPECT_FALSE(f.Seal(1, "x", false, &l)) << "the key outlived its token's exp by the lock wait";
}

// A state from before the init markers gets its marker on the first load (under the lock), so a later loss is
// detected; an interrupted first init (state written, marker not) completes the same way.
TEST(YardKeys, MarkerlessStateIsUpgraded) {
  Fixture f(nullptr, true);
  ASSERT_EQ(f.Reg("seal", 2, Key32(45)).status, 200);
  f.y.reset();
  ASSERT_TRUE(std::filesystem::remove(f.o.state_path + ".init"));  // as a round-2 state file
  std::string e;
  f.Restart();
  ASSERT_TRUE(f.y);
  EXPECT_TRUE(std::filesystem::exists(f.o.state_path + ".init")) << "the upgrade did not write the marker";
  f.y.reset();
  std::filesystem::remove(f.o.state_path);
  EXPECT_FALSE(YardSpillKeys::Create(f.o, &e)) << "an upgraded state vanished and was silently reset";
  EXPECT_NE(e.find("recovery required"), std::string::npos) << e;
  // The master's epoch file the same way.
  sc::SecretBytes m(32);
  ASSERT_TRUE(sc::RandomBytes({m.data(), 32}));
  const std::string ef = f.dir + "/m.epoch";
  WriteFile(ef, "3\n");
  {
    sc::SecretBytes m1(32);
    std::memcpy(m1.data(), m.data(), 32);
    auto mk = halo::server::MasterSpillKeys::Create("file", std::move(m1), ef, &e);
    ASSERT_TRUE(mk) << e;
    EXPECT_EQ(mk->epoch(), 3u);
  }
  EXPECT_TRUE(std::filesystem::exists(ef + ".init"));
  std::filesystem::remove(ef);
  EXPECT_FALSE(halo::server::MasterSpillKeys::Create("file", std::move(m), ef, &e));
}

// The state directory is pinned when the provider starts: renaming it and putting another directory at its path
// afterwards cannot split a write from its fsync; every publish goes to (and is made durable in) the pinned one.
TEST(YardKeys, StateDirectoryIsPinned) {
  Fixture f;
  const std::string st = f.dir + "/st", moved = f.dir + "/st-moved";
  std::filesystem::create_directories(st);
  std::filesystem::permissions(st, std::filesystem::perms::owner_all);
  f.o.state_path = st + "/yard-state";
  f.Restart();
  ASSERT_TRUE(f.y);
  std::filesystem::rename(st, moved);
  std::filesystem::create_directories(st);
  std::filesystem::permissions(st, std::filesystem::perms::owner_all);
  ASSERT_EQ(f.Reg("seal", 4, Key32(46)).status, 200);
  EXPECT_FALSE(std::filesystem::exists(st + "/yard-state")) << "a publish followed the path, not the pinned directory";
  std::ifstream in(moved + "/yard-state");
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(text.find("sess-1 4 "), std::string::npos) << text;
  std::string e;
  EXPECT_FALSE(halo::server::StateDir::Open(f.dir + "/nope", &e));
  ::chmod(moved.c_str(), 0770);
  EXPECT_FALSE(halo::server::StateDir::Open(moved, &e)) << "a group-writable state directory";
}

// Master and yard persistence run concurrently (different provider locks): no shared mutable state between them
// (run under ThreadSanitizer in build/cpu-tsan).
TEST(YardKeys, ConcurrentMasterAndYardPersistence) {
  Fixture f(nullptr, true);
  sc::SecretBytes m(32);
  ASSERT_TRUE(sc::RandomBytes({m.data(), 32}));
  std::string e;
  auto mk = halo::server::MasterSpillKeys::Create("file", std::move(m), f.dir + "/m.epoch", &e);
  ASSERT_TRUE(mk) << e;
  std::thread fencer([&] {
    for (int i = 0; i < 40; ++i) {
      std::string w;
      EXPECT_TRUE(mk->FenceEpoch(&w)) << w;
    }
  });
  for (int i = 1; i <= 40; ++i) EXPECT_EQ(f.Reg("seal", i, Key32(static_cast<std::uint8_t>(i)), i).status, 200);
  fencer.join();
  EXPECT_EQ(mk->epoch(), 41u);
}

// The JWKS file is loaded like the state files: no symlink, a regular file not writable by group or others, owned by
// us or root; otherwise nothing verifies.
TEST(YardKeys, JwksFileIsChecked) {
  Fixture f;
  ASSERT_EQ(f.Reg("seal", 1, Key32(47)).status, 200);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  ::chmod(f.jwks.c_str(), 0666);  // only the ctime changes: the refresh notices it
  EXPECT_EQ(f.Reg("seal", 2, Key32(48), 2).status, 401) << "a world-writable JWKS verified a token";
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  ::chmod(f.jwks.c_str(), 0600);
  ASSERT_EQ(f.Reg("seal", 2, Key32(48), 2).status, 200);
  std::filesystem::rename(f.jwks, f.dir + "/real.json");
  std::filesystem::create_symlink(f.dir + "/real.json", f.jwks);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(f.Reg("seal", 3, Key32(49), 3).status, 401) << "a symlinked JWKS verified a token";
}
