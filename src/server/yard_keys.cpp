#include "server/yard_keys.hpp"

#include <openssl/bn.h>
#include <openssl/core_names.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/param_build.h>
#include <openssl/rsa.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <utility>

namespace halo::server {

namespace crypt = core::crypt;

// ---- base64url ----
std::string Base64UrlEncode(std::span<const std::uint8_t> b) {
  static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  std::string o;
  std::size_t i = 0;
  for (; i + 2 < b.size(); i += 3) {
    const std::uint32_t v = (std::uint32_t{b[i]} << 16) | (std::uint32_t{b[i + 1]} << 8) | b[i + 2];
    o += t[v >> 18];
    o += t[(v >> 12) & 63];
    o += t[(v >> 6) & 63];
    o += t[v & 63];
  }
  if (i + 1 == b.size()) {
    const std::uint32_t v = std::uint32_t{b[i]} << 16;
    o += t[v >> 18];
    o += t[(v >> 12) & 63];
  } else if (i + 2 == b.size()) {
    const std::uint32_t v = (std::uint32_t{b[i]} << 16) | (std::uint32_t{b[i + 1]} << 8);
    o += t[v >> 18];
    o += t[(v >> 12) & 63];
    o += t[(v >> 6) & 63];
  }
  return o;
}

namespace {

int B64Val(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '-') return 62;
  if (c == '_') return 63;
  return -1;
}

// Strict decode into a sink; false on any non-alphabet char, a length of 1 mod 4, or non-zero trailing bits.
template <class Put>
bool DecodeStrict(std::string_view s, Put&& put) {
  if (s.size() % 4 == 1) return false;
  std::uint32_t acc = 0;
  int bits = 0;
  for (char c : s) {
    const int v = B64Val(c);
    if (v < 0) return false;
    acc = (acc << 6) | static_cast<std::uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      if (!put(static_cast<std::uint8_t>(acc >> bits))) return false;
      acc &= (1u << bits) - 1;
    }
  }
  return acc == 0;
}

}  // namespace

bool Base64UrlDecode(std::string_view s, std::vector<std::uint8_t>* out) {
  out->clear();
  out->reserve(s.size() * 3 / 4 + 1);
  return DecodeStrict(s, [&](std::uint8_t b) { return out->push_back(b), true; });
}

bool Base64UrlDecodeExact(std::string_view s, std::span<std::uint8_t> out) {
  if (s.size() != (out.size() * 4 + 2) / 3) {  // checked before a byte is decoded
    OPENSSL_cleanse(out.data(), out.size());
    return false;
  }
  std::size_t at = 0;
  const bool ok = DecodeStrict(s, [&](std::uint8_t b) {
    if (at >= out.size()) return false;
    out[at++] = b;
    return true;
  });
  if (!ok || at != out.size()) {
    OPENSSL_cleanse(out.data(), out.size());
    return false;
  }
  return true;
}

// ---- JWKS ----
Jwks::Jwks() = default;
Jwks::~Jwks() { Clear(); }
void Jwks::Clear() {
  for (auto& [k, p] : keys_) EVP_PKEY_free(static_cast<EVP_PKEY*>(p));
  keys_.clear();
}

namespace {

EVP_PKEY* RsaPublic(const std::vector<std::uint8_t>& n, const std::vector<std::uint8_t>& e) {
  BIGNUM* bn = BN_bin2bn(n.data(), static_cast<int>(n.size()), nullptr);
  BIGNUM* be = BN_bin2bn(e.data(), static_cast<int>(e.size()), nullptr);
  OSSL_PARAM_BLD* bld = OSSL_PARAM_BLD_new();
  OSSL_PARAM* params = nullptr;
  EVP_PKEY* pkey = nullptr;
  EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_from_name(nullptr, "RSA", nullptr);
  const int bits = bn ? BN_num_bits(bn) : 0;
  if (bn && be && bld && ctx && bits >= 2048 && bits <= 16384 && BN_is_word(be, 65537) &&
      OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_N, bn) == 1 &&
      OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_E, be) == 1 && (params = OSSL_PARAM_BLD_to_param(bld)) &&
      EVP_PKEY_fromdata_init(ctx) == 1)
    (void)EVP_PKEY_fromdata(ctx, &pkey, EVP_PKEY_PUBLIC_KEY, params);
  OSSL_PARAM_free(params);
  OSSL_PARAM_BLD_free(bld);
  EVP_PKEY_CTX_free(ctx);
  BN_free(bn);
  BN_free(be);
  return pkey;
}

const std::string* Str(const json::Value* v) {
  return v && v->is(json::Value::Kind::kString) ? &v->str() : nullptr;
}
std::string StrOr(const json::Value* v) { return Str(v) ? *Str(v) : std::string(); }

// An integer claim: absent -> nullopt with *bad false; present but not a bounded integer -> *bad true.
std::optional<std::int64_t> Int(const json::Value* v, bool* bad) {
  if (!v) return std::nullopt;
  const std::string& d = v->int_digits();
  if (!v->is(json::Value::Kind::kInt) || d.empty() || d.size() > 18 || d[0] == '-') {
    *bad = true;
    return std::nullopt;
  }
  return std::stoll(d);
}

std::int64_t Now() { return static_cast<std::int64_t>(std::time(nullptr)); }

std::string Esc(std::string_view s) {  // JSON string contents (printable ASCII only)
  std::string o;
  for (char c : s)
    if (c >= 0x20 && c < 0x7f && c != '"' && c != '\\') o += c;
  return o;
}

void Cleanse(const std::string& s) {  // overwrite a string's bytes in place (key or token copies)
  if (!s.empty()) OPENSSL_cleanse(const_cast<char*>(s.data()), s.size());
}

}  // namespace

bool Jwks::Parse(std::string_view text, std::string* error) {
  auto doc = json::Parse(text, error, {.max_depth = 8, .max_bytes = 1u << 20, .reject_duplicate_keys = true});
  if (!doc) return false;
  const json::Value* keys = doc->find("keys");
  if (!keys || !keys->is(json::Value::Kind::kArray)) return *error = "jwks: no keys array", false;
  std::map<std::string, void*, std::less<>> next;
  std::vector<std::string> kids;
  bool dup = false;
  for (const auto& k : keys->items()) {
    const std::string kid = StrOr(k.find("kid"));
    if (!kid.empty()) {
      dup = dup || std::find(kids.begin(), kids.end(), kid) != kids.end();
      kids.push_back(kid);
    }
    if (StrOr(k.find("kty")) != "RSA") continue;
    if (const auto* a = k.find("alg"); a && StrOr(a) != "RS256") continue;
    if (const auto* u = k.find("use"); u && StrOr(u) != "sig") continue;
    if (const auto* ops = k.find("key_ops")) {
      bool verify = false;
      if (ops->is(json::Value::Kind::kArray))
        for (const auto& o : ops->items()) verify = verify || StrOr(&o) == "verify";
      if (!verify) continue;
    }
    std::vector<std::uint8_t> n, e;
    if (kid.empty() || !Base64UrlDecode(StrOr(k.find("n")), &n) || !Base64UrlDecode(StrOr(k.find("e")), &e)) continue;
    if (n.empty() || n[0] == 0 || e.empty() || e[0] == 0) continue;  // canonical: no leading zero octets
    EVP_PKEY* p = RsaPublic(n, e);
    if (p) next[kid] = p;
  }
  if (dup) {
    for (auto& [k, p] : next) EVP_PKEY_free(static_cast<EVP_PKEY*>(p));
    return *error = "jwks: duplicate kid", false;
  }
  Clear();
  keys_ = std::move(next);
  return true;
}

bool Jwks::Load(const std::string& path, std::string* error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return *error = "jwks: cannot read " + path, false;
  std::stringstream ss;
  ss << f.rdbuf();
  return Parse(ss.str(), error);
}

const void* Jwks::Find(std::string_view kid) const {
  const auto it = keys_.find(kid);
  return it == keys_.end() ? nullptr : it->second;
}

const char* JwtErrorName(JwtError e) {
  switch (e) {
    case JwtError::kOk: return "ok";
    case JwtError::kMalformed: return "malformed";
    case JwtError::kAlg: return "alg";
    case JwtError::kUnknownKid: return "unknown kid";
    case JwtError::kSignature: return "signature";
    case JwtError::kClaims: return "claims";
    case JwtError::kExpired: return "expired";
    case JwtError::kNotYetValid: return "not yet valid";
    case JwtError::kLifetime: return "lifetime";
    case JwtError::kIssuer: return "issuer";
    case JwtError::kAudience: return "audience";
    case JwtError::kConfig: return "verifier misconfigured";
  }
  return "?";
}

// ---- JWT ----
std::optional<json::Value> VerifyJwt(std::string_view token, const Jwks& keys, const JwtCheck& c, JwtError* err) {
  const auto fail = [&](JwtError e) -> std::optional<json::Value> {
    *err = e;
    return std::nullopt;
  };
  if (c.issuer.empty() || c.audience.empty()) return fail(JwtError::kConfig);  // never verify without both
  if (token.size() > 16384) return fail(JwtError::kMalformed);
  const auto d1 = token.find('.');
  const auto d2 = d1 == std::string_view::npos ? d1 : token.find('.', d1 + 1);
  if (d2 == std::string_view::npos || token.find('.', d2 + 1) != std::string_view::npos) return fail(JwtError::kMalformed);
  std::vector<std::uint8_t> hb, pb, sig;
  if (!Base64UrlDecode(token.substr(0, d1), &hb) || !Base64UrlDecode(token.substr(d1 + 1, d2 - d1 - 1), &pb) ||
      !Base64UrlDecode(token.substr(d2 + 1), &sig))
    return fail(JwtError::kMalformed);
  std::string e;
  const auto hdr = json::Parse(std::string_view(reinterpret_cast<const char*>(hb.data()), hb.size()), &e,
                               {.max_depth = 4, .max_bytes = 4096, .reject_duplicate_keys = true});
  if (!hdr || !hdr->is(json::Value::Kind::kObject)) return fail(JwtError::kMalformed);
  if (StrOr(hdr->find("alg")) != "RS256") return fail(JwtError::kAlg);
  if (hdr->find("crit")) return fail(JwtError::kAlg);
  const auto* pkey = static_cast<const EVP_PKEY*>(keys.Find(StrOr(hdr->find("kid"))));
  if (!pkey) return fail(JwtError::kUnknownKid);
  {
    EVP_MD_CTX* md = EVP_MD_CTX_new();
    EVP_PKEY_CTX* pctx = nullptr;
    const std::string_view signing = token.substr(0, d2);
    const bool ok = md &&
                    EVP_DigestVerifyInit(md, &pctx, EVP_sha256(), nullptr, const_cast<EVP_PKEY*>(pkey)) == 1 &&
                    EVP_PKEY_CTX_set_rsa_padding(pctx, RSA_PKCS1_PADDING) == 1 &&
                    EVP_DigestVerify(md, sig.data(), sig.size(), reinterpret_cast<const unsigned char*>(signing.data()),
                                     signing.size()) == 1;
    EVP_MD_CTX_free(md);
    if (!ok) return fail(JwtError::kSignature);
  }
  auto claims = json::Parse(std::string_view(reinterpret_cast<const char*>(pb.data()), pb.size()), &e,
                            {.max_depth = 6, .max_bytes = 8192, .reject_duplicate_keys = true});
  if (!claims || !claims->is(json::Value::Kind::kObject)) return fail(JwtError::kMalformed);
  bool bad = false;
  const auto iat = Int(claims->find("iat"), &bad), exp = Int(claims->find("exp"), &bad),
             nbf = Int(claims->find("nbf"), &bad);
  if (bad || !iat || !exp) return fail(JwtError::kClaims);
  const std::int64_t now = c.now ? c.now : Now();
  if (*exp <= *iat || *exp - *iat > c.max_ttl_s) return fail(JwtError::kLifetime);
  if (now >= *exp) return fail(JwtError::kExpired);  // no post-exp leeway: a key registration is a capability
  if ((nbf && now + c.skew_s < *nbf) || *iat > now + c.skew_s) return fail(JwtError::kNotYetValid);
  if (StrOr(claims->find("iss")) != c.issuer) return fail(JwtError::kIssuer);
  const json::Value* aud = claims->find("aud");
  bool aud_ok = StrOr(aud) == c.audience;
  if (aud && aud->is(json::Value::Kind::kArray))
    for (const auto& a : aud->items()) aud_ok = aud_ok || StrOr(&a) == c.audience;
  if (!aud_ok) return fail(JwtError::kAudience);
  *err = JwtError::kOk;
  return claims;
}

// ---- the provider ----
YardSpillKeys::YardSpillKeys(Options o) : o_(std::move(o)) {}

std::unique_ptr<YardSpillKeys> YardSpillKeys::Create(Options o, std::string* error) {
  if (o.jwks_path.empty()) return *error = "yard keys: --yard-jwks is required", nullptr;
  if (o.issuer.empty()) return *error = "yard keys: --yard-issuer is required", nullptr;
  if (o.audience.empty()) return *error = "yard keys: --yard-audience must not be empty", nullptr;
  if (o.key_ttl_s < 1 || o.key_ttl_s > 3600) return *error = "yard keys: --yard-key-ttl must be 1..3600 s", nullptr;
  if (o.jwks_refresh_s < 1 || o.jwks_refresh_s > 3600) return *error = "yard keys: JWKS refresh must be 1..3600 s", nullptr;
  std::unique_ptr<YardSpillKeys> y(new YardSpillKeys(std::move(o)));
  if (!y->LoadState(error)) return nullptr;
  y->RefreshJwks(true);
  if (y->jwks_.size() == 0) return *error = "yard keys: the JWKS " + y->o_.jwks_path + " has no usable RS256 key", nullptr;
  if (y->o_.sweeper)
    y->sweeper_ = std::thread([p = y.get()] {
      std::unique_lock<std::mutex> lk(p->mu_);
      while (!p->stop_) {
        p->cv_.wait_for(lk, std::chrono::seconds(1));
        (void)p->SweepLocked(Clock::now());
      }
    });
  return y;
}

YardSpillKeys::~YardSpillKeys() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (sweeper_.joinable()) sweeper_.join();
}

bool YardSpillKeys::ValidSession(std::string_view s) {
  if (s.empty() || s.size() > 128) return false;
  for (char c : s)
    if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
      return false;
  return true;
}

// State file: one line per session: "<session> <high> <fence|-> <last_seal|-> <last_spill> <last_restore>".
bool YardSpillKeys::LoadState(std::string* error) {
  if (o_.state_path.empty()) return true;
  struct stat st {};
  if (::lstat(o_.state_path.c_str(), &st) != 0) {
    if (errno == ENOENT) return true;
    return *error = "yard keys: cannot stat " + o_.state_path, false;
  }
  if (!S_ISREG(st.st_mode)) return *error = "yard keys: " + o_.state_path + " is not a regular file", false;
  std::ifstream f(o_.state_path);
  if (!f) return *error = "yard keys: cannot read " + o_.state_path, false;
  if ((st.st_mode & 077) != 0)
    return *error = "yard keys: " + o_.state_path + " is readable by group/others (chmod 600)", false;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ls(line);
    std::string sess, fence, seal, high, spill, restore, extra;
    if (!(ls >> sess >> high >> fence >> seal >> spill >> restore) || (ls >> extra) || !ValidSession(sess))
      return *error = "yard keys: " + o_.state_path + " is malformed (refusing to start)", false;
    Session s;
    std::uint64_t v = 0;
    const auto num = [&](const std::string& t, std::uint64_t* out) { return ParseU64(t, std::uint64_t{1} << 62, out); };
    bool ok = num(high, &s.high) && num(spill, &s.last_spill) && num(restore, &s.last_restore);
    if (fence != "-") ok = ok && num(fence, &v), s.fence = v;
    if (seal != "-") ok = ok && num(seal, &v), s.last_seal = v;
    if (!ok) return *error = "yard keys: " + o_.state_path + " is malformed (refusing to start)", false;
    sessions_[sess] = std::move(s);
  }
  return true;
}

bool YardSpillKeys::SaveState(std::string* error) {
  if (o_.state_path.empty()) return true;
  std::string out;
  for (const auto& [name, s] : sessions_)
    out += name + " " + std::to_string(s.high) + " " + (s.fence ? std::to_string(*s.fence) : "-") + " " +
           (s.last_seal ? std::to_string(*s.last_seal) : "-") + " " + std::to_string(s.last_spill) + " " +
           std::to_string(s.last_restore) + "\n";
  return WriteFileDurable(o_.state_path, out, error);
}

void YardSpillKeys::Record(Session& s, Use use, std::uint64_t epoch, const std::array<std::uint8_t, 32>& h,
                           std::uint64_t entry, OpState st) {
  for (auto& r : s.ops)
    if (r.use == use && r.epoch == epoch && crypt::Equal(r.key_hash, h)) {
      r.state = st;
      r.entry = entry;
      return;
    }
  s.ops.push_back({use, epoch, h, entry, st});
  if (s.ops.size() > 32) s.ops.erase(s.ops.begin());
}

const YardSpillKeys::OpRecord* YardSpillKeys::FindOp(const Session& s, Use use, std::uint64_t epoch,
                                                     const std::array<std::uint8_t, 32>& h) const {
  for (const auto& r : s.ops)
    if (r.use == use && r.epoch == epoch && crypt::Equal(r.key_hash, h)) return &r;
  return nullptr;
}

void YardSpillKeys::ZeroSlot(Session& s, std::optional<Slot>& slot, OpState why) {
  if (!slot) return;
  Record(s, &slot == &s.seal ? Use::kSeal : Use::kOpen, slot->epoch, slot->key_hash, slot->entry, why);
  slot->key.Clear();  // zeroed and unmapped now
  slot.reset();
}

std::size_t YardSpillKeys::SweepLocked(Clock::time_point now) {
  std::size_t n = 0;
  for (auto& [name, s] : sessions_)
    for (auto* slot : {&s.seal, &s.open})
      if (*slot && !(*slot)->reserved && now >= (*slot)->deadline) {
        ZeroSlot(s, *slot, OpState::kExpired);
        ++n;
      }
  return n;
}

std::size_t YardSpillKeys::Sweep() {
  std::lock_guard<std::mutex> lk(mu_);
  return SweepLocked(Clock::now());
}

void YardSpillKeys::RefreshJwks(bool force) {
  // Under mu_ or before the provider is shared. A changed file (inode, size, mtime) or the refresh interval re-reads
  // it; an unreadable or invalid file empties the key set (fail closed: nothing verifies until it is fixed).
  struct stat st {};
  const bool have = ::stat(o_.jwks_path.c_str(), &st) == 0;
  const std::uint64_t sig[3] = {have ? st.st_ino : 0, have ? static_cast<std::uint64_t>(st.st_size) : 0,
                                have ? static_cast<std::uint64_t>(st.st_mtim.tv_sec) * 1000000000ull +
                                           static_cast<std::uint64_t>(st.st_mtim.tv_nsec)
                                     : 0};
  const auto now = Clock::now();
  if (!force && std::equal(sig, sig + 3, jwks_sig_) && now - jwks_loaded_ < std::chrono::seconds(o_.jwks_refresh_s))
    return;
  std::string e;
  if (!have || !jwks_.Load(o_.jwks_path, &e)) {
    jwks_.Clear();
    std::fprintf(stderr, "yard keys: JWKS %s unusable (%s): no token verifies until it is fixed\n", o_.jwks_path.c_str(),
                 have ? e.c_str() : "missing");
  }
  std::copy(sig, sig + 3, jwks_sig_);
  jwks_loaded_ = now;
}

YardSpillKeys::Answer YardSpillKeys::Register(std::string_view session, std::string& body, std::int64_t now) {
  struct Wipe {  // the body holds the key (base64url) and the token: overwritten whatever the outcome
    std::string& b;
    ~Wipe() { Cleanse(b); }
  } wipe{body};
  const auto answer = [](int status, const std::string& msg) {
    return Answer{status, status == 200 ? msg : "{\"ok\":false,\"error\":\"" + Esc(msg) + "\"}", "", 0, ""};
  };
  if (!ValidSession(session)) return answer(400, "session must match [A-Za-z0-9._-]{1,128}");
  std::string e;
  auto req = json::Parse(body, &e, {.max_depth = 4, .max_bytes = 32768, .reject_duplicate_keys = true});
  struct WipeJson {  // the parsed copies of the key and the token
    std::optional<json::Value>& v;
    ~WipeJson() {
      if (!v) return;
      for (const auto& [k, m] : v->members())
        if (m.is(json::Value::Kind::kString)) Cleanse(m.str());
    }
  } wipe_json{req};
  if (!req || !req->is(json::Value::Kind::kObject)) return answer(400, "body must be a JSON object");
  const std::string* token = Str(req->find("token"));
  const std::string use = StrOr(req->find("use"));
  if (!token) return answer(400, "token missing");
  if (use != "seal" && use != "open" && use != "revoke") return answer(400, "use must be seal, open or revoke");
  crypt::SecretBytes key;
  std::array<std::uint8_t, 32> key_hash{};
  if (use != "revoke") {
    const std::string* kb = Str(req->find("key"));
    key = crypt::SecretBytes(crypt::kKeyBytes);
    if (!key.ok()) return answer(500, "cannot allocate locked key memory");
    if (!kb || !Base64UrlDecodeExact(*kb, {key.data(), crypt::kKeyBytes}))
      return answer(400, "key must be 32 bytes, base64url without padding");
    if (!crypt::Sha256(key.span(), key_hash)) return answer(500, "SHA-256 failed");
  }
  std::lock_guard<std::mutex> lk(mu_);
  const auto tnow = Clock::now();
  (void)SweepLocked(tnow);
  RefreshJwks(false);
  // The token. Every refusal looks the same to the client; the reason goes to the log (never the token).
  JwtCheck check{.issuer = o_.issuer, .audience = o_.audience, .now = now};
  JwtError je = JwtError::kOk;
  auto claims = VerifyJwt(*token, jwks_, check, &je);
  if (!claims && je == JwtError::kUnknownKid) {  // yard may have just rotated its key: re-read once
    RefreshJwks(true);
    claims = VerifyJwt(*token, jwks_, check, &je);
  }
  const auto refuse = [&](const char* why) {
    std::fprintf(stderr, "yard keys: %.*s: token refused: %s\n", static_cast<int>(session.size()), session.data(), why);
    Answer a = answer(401, "token refused");  // one body for every reason: no oracle for the client
    a.reason = why;
    return a;
  };
  if (!claims) return refuse(JwtErrorName(je));
  if (StrOr(claims->find("yard_sess")) != session) return refuse("another session");
  bool bad = false;
  const auto epoch = Int(claims->find("yard_epoch"), &bad);
  if (bad || !epoch || *epoch < 1) return refuse("yard_epoch");
  const std::string jti = StrOr(claims->find("jti"));
  if (jti.empty() || jti.size() > 128) return refuse("jti");
  const json::Value* hs = claims->find("halo_spill");
  if (!hs || !hs->is(json::Value::Kind::kObject) || StrOr(hs->find("use")) != use) return refuse("use not authorised");
  if (use != "revoke") {
    std::array<std::uint8_t, 32> want{};
    if (!Base64UrlDecodeExact(StrOr(hs->find("key_sha256")), want) || !crypt::Equal(want, key_hash))
      return refuse("bound to another key");
  }
  std::uint64_t entry = 0;
  if (use == "seal") {
    const auto en = Int(hs->find("entry"), &bad);
    if (bad || !en || *en < 1) return refuse("seal without an entry");
    entry = static_cast<std::uint64_t>(*en);
    const auto* body_entry = req->find("entry");
    if (body_entry && Int(body_entry, &bad) != en) return answer(400, "entry differs from the token's");
  }
  const auto E = static_cast<std::uint64_t>(*epoch);
  std::uint64_t through = E;
  if (use == "revoke") {
    const auto th = Int(hs->find("through"), &bad);
    if (bad) return refuse("through");
    if (th) through = static_cast<std::uint64_t>(*th);
    if (through > E) return refuse("revoke through a later epoch than the token's");
  }
  // Single use (in memory: after a restart the persisted epochs refuse what was used).
  const std::int64_t t = now ? now : Now();
  for (auto it = seen_jti_.begin(); it != seen_jti_.end();) it = it->second < t ? seen_jti_.erase(it) : std::next(it);
  if (seen_jti_.contains(jti)) return answer(409, "token replayed (jti already used)");
  const std::int64_t exp = Int(claims->find("exp"), &bad).value_or(t);
  Session& s = sessions_[std::string(session)];
  const auto persist_and = [&](Answer a) {
    std::string pe;
    if (!SaveState(&pe)) {
      std::fprintf(stderr, "yard keys: cannot persist state: %s\n", pe.c_str());
      return answer(500, "cannot persist the registration (not done)");
    }
    seen_jti_[jti] = exp;
    return a;
  };
  const auto ok_body = [&](bool dup, const char* state, std::uint64_t file_epoch, const std::string& name) {
    const std::string kid = name.empty() ? "" : "yard:" + name + ":e" + std::to_string(file_epoch) + ":" + std::string(session);
    return answer(200, "{\"ok\":true,\"session\":\"" + Esc(session) + "\",\"use\":\"" + use + "\",\"epoch\":" +
                           std::to_string(E) + ",\"key_id\":\"" + Esc(kid) + "\",\"state\":\"" + state +
                           "\",\"duplicate\":" + (dup ? "true" : "false") + "}");
  };
  if (use == "revoke") {
    // Never stale: a fence only gets stricter. Keys at file epochs <= the fence are zeroed now.
    const std::uint64_t old_high = s.high;
    const auto old_fence = s.fence, old_seal = s.last_seal;
    const bool dup = s.fence && *s.fence >= through;
    s.fence = std::max(s.fence.value_or(0), through);
    s.high = std::max(s.high, E);
    if (s.seal && s.seal->epoch <= *s.fence) ZeroSlot(s, s.seal, OpState::kRevoked);
    if (s.open && s.open->epoch - 1 <= *s.fence) ZeroSlot(s, s.open, OpState::kRevoked);
    if (s.last_seal && *s.last_seal <= *s.fence) s.last_seal.reset();
    for (auto& r : s.ops)
      if ((r.use == Use::kSeal ? r.epoch : r.epoch - 1) <= *s.fence && r.state == OpState::kRegistered)
        r.state = OpState::kRevoked;
    std::string pe;
    if (!SaveState(&pe)) {  // not done: put the fence back (the zeroed keys stay zeroed: stricter is safe)
      s.high = old_high;
      s.fence = old_fence;
      s.last_seal = old_seal;
      return answer(500, "cannot persist the revocation (not done)");
    }
    seen_jti_[jti] = exp;
    Answer a = answer(200, "{\"ok\":true,\"session\":\"" + Esc(session) + "\",\"use\":\"revoke\",\"epoch\":" +
                               std::to_string(E) + ",\"through\":" + std::to_string(*s.fence) +
                               ",\"duplicate\":" + (dup ? "true" : "false") + "}");
    a.revoked_session = std::string(session);
    a.revoked_through = *s.fence;
    return a;
  }
  std::string name;
  if (!crypt::KeyName(key.span(), &name)) return answer(500, "cannot name the key");
  const auto deadline = tnow + std::min(std::chrono::seconds(o_.key_ttl_s), std::chrono::seconds(std::max<std::int64_t>(exp - t, 0)));
  const Use u = use == "seal" ? Use::kSeal : Use::kOpen;
  const std::uint64_t file_epoch = u == Use::kSeal ? E : E - 1;
  if (s.fence && file_epoch <= *s.fence) return answer(409, "revoked: epoch " + std::to_string(file_epoch) + " is fenced");
  if (E < s.high) return answer(409, "stale epoch " + std::to_string(E) + " (session at " + std::to_string(s.high) + ")");
  // One-shot: the same registration again reports what happened to it (a consumed key is never re-armed); a used
  // epoch never takes a key again.
  if (const OpRecord* r = FindOp(s, u, E, key_hash); r && r->state == OpState::kConsumed && (u == Use::kOpen || r->entry == entry)) {
    seen_jti_[jti] = exp;
    return ok_body(true, "consumed", file_epoch, name);
  }
  std::optional<Slot>& slot = u == Use::kSeal ? s.seal : s.open;
  // The live slot of this very operation (constant-time comparison of the whole key, not its name).
  if (slot && slot->epoch == E && crypt::Equal(slot->key.span(), key.span()) && slot->entry == entry) {
    slot->deadline = std::max(slot->deadline, deadline);  // a retry with a newer token: same key, same op
    return persist_and(ok_body(true, slot->reserved ? "in use" : "registered", file_epoch, name));
  }
  const bool used = u == Use::kSeal ? E <= s.last_spill : E <= s.last_restore;
  if (used) return answer(409, std::string(use) + " at epoch " + std::to_string(E) + " was already used");
  if (u == Use::kOpen && (!s.last_seal || E != *s.last_seal + 1))
    return answer(409, "open at epoch " + std::to_string(E) + " is not right after a seal here");
  if (slot && slot->reserved) return answer(409, "a spill or restore holds the current key; retry");
  if (u == Use::kSeal) {
    // An entry is sealed for one yard session only.
    for (auto& [other, os] : sessions_)
      if (other != session && os.seal && os.seal->entry == entry)
        return answer(409, "entry " + std::to_string(entry) + " has a seal registration of another session");
  }
  ZeroSlot(s, slot, OpState::kExpired);  // replaced before use (a new key for the same op)
  slot = Slot{E, std::move(key), key_hash, name, entry, deadline, ++gen_, false};
  Record(s, u, E, key_hash, entry, OpState::kRegistered);
  const auto prev_high = s.high;
  const auto prev_seal = s.last_seal;
  s.high = E;
  if (u == Use::kSeal) {
    s.last_seal = E;
    ZeroSlot(s, s.open, OpState::kExpired);  // a new checkpoint: an older open key has nothing left to open
  } else {
    ZeroSlot(s, s.seal, OpState::kExpired);
  }
  Answer a = persist_and(ok_body(false, "registered", file_epoch, name));
  if (a.status != 200) {  // not acknowledged: not installed
    ZeroSlot(s, slot, OpState::kExpired);
    s.high = prev_high;
    s.last_seal = prev_seal;
  }
  return a;
}

std::string YardSpillKeys::ListJson() {
  std::lock_guard<std::mutex> lk(mu_);
  (void)SweepLocked(Clock::now());
  std::string o = "{\"sessions\":[";
  bool first = true;
  for (const auto& [name, s] : sessions_) {
    const auto slot = [&](const std::optional<Slot>& x) {
      return x ? "{\"epoch\":" + std::to_string(x->epoch) + ",\"key_name\":\"" + x->name + "\",\"entry\":" +
                     std::to_string(x->entry) + ",\"in_use\":" + (x->reserved ? "true" : "false") + "}"
               : std::string("null");
    };
    o += std::string(first ? "" : ",") + "{\"session\":\"" + Esc(name) + "\",\"high_epoch\":" + std::to_string(s.high) +
         ",\"revoked_through\":" + (s.fence ? std::to_string(*s.fence) : "null") +
         ",\"sealed_epoch\":" + (s.last_seal ? std::to_string(*s.last_seal) : "null") +
         ",\"spilled_epoch\":" + std::to_string(s.last_spill) + ",\"restored_epoch\":" + std::to_string(s.last_restore) +
         ",\"seal\":" + slot(s.seal) + ",\"open\":" + slot(s.open) + "}";
    first = false;
  }
  return o + "]}";
}

std::string YardSpillKeys::Describe() const {
  std::lock_guard<std::mutex> lk(mu_);
  return "yard (" + std::to_string(jwks_.size()) + " JWKS keys, aud " + o_.audience + ", fallback " +
         (o_.fallback ? o_.fallback->Describe() : std::string("refuse")) + ")";
}

bool YardSpillKeys::SealLease(std::uint64_t entry, std::string_view label, bool yard_bound, SpillLease* out,
                              std::string* why) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    (void)SweepLocked(Clock::now());
    for (auto& [name, s] : sessions_) {
      if (!s.seal || s.seal->entry != entry) continue;
      Slot& sl = *s.seal;
      if (sl.reserved) return *why = "the seal key is in use", false;
      if (yard_bound && label != name)
        return *why = "entry is bound to yard session " + std::string(label) + ", the registration is for " + name, false;
      if (!crypt::DeriveKek(sl.key.span(), name, sl.epoch, "yard:" + sl.name, &out->kek, why)) return false;
      // The used-epoch fence is persisted before the key is used (a crash after the spill cannot let the seal
      // token of this epoch register again); a failed spill puts it back (Finish).
      out->prev_fence = s.last_spill;
      s.last_spill = std::max(s.last_spill, sl.epoch);
      if (!SaveState(why)) {
        s.last_spill = out->prev_fence;
        out->kek = {};
        return false;
      }
      out->session = name;
      out->epoch = sl.epoch;
      out->gen = sl.gen;
      out->yard = true;
      sl.reserved = true;
      return true;
    }
  }
  // No registration for this entry. A yard-bound entry needs one (no custody downgrade); others may fall back.
  if (yard_bound) return *why = "yard-bound entry: a fresh seal registration is needed", false;
  if (o_.fallback) return o_.fallback->SealLease(entry, label, false, out, why);
  return *why = "no yard seal registration for entry " + std::to_string(entry), false;
}

bool YardSpillKeys::OpenLease(std::string_view label, std::string_view file_key_id, SpillLease* out, std::string* why) {
  if (file_key_id.rfind("yard:", 0) != 0) {
    if (o_.fallback) return o_.fallback->OpenLease(label, file_key_id, out, why);
    return *why = "file not sealed by yard and no fallback provider", false;
  }
  std::lock_guard<std::mutex> lk(mu_);
  (void)SweepLocked(Clock::now());
  const auto it = sessions_.find(label);
  if (it == sessions_.end() || !it->second.open) return *why = "no yard open key registered for " + std::string(label), false;
  Slot& sl = *it->second.open;
  if (sl.reserved) return *why = "the open key is in use", false;
  if (!crypt::DeriveKek(sl.key.span(), label, sl.epoch - 1, "yard:" + sl.name, &out->kek, why)) return false;
  if (out->kek.id != file_key_id) {
    out->kek = {};
    return *why = "the registered open key is not this file's (key or epoch)", false;
  }
  // Persisted before the key is used: the open token of this epoch never registers again, even after a crash.
  Session& s = it->second;
  out->prev_fence = s.last_restore;
  s.last_restore = std::max(s.last_restore, sl.epoch);
  if (!SaveState(why)) {
    s.last_restore = out->prev_fence;
    out->kek = {};
    return false;
  }
  out->session = std::string(label);
  out->epoch = sl.epoch - 1;
  out->gen = sl.gen;
  out->yard = true;
  sl.reserved = true;
  return true;
}

void YardSpillKeys::Finish(const SpillLease& lease, Use use, bool ok) {
  if (!lease.yard) {
    if (o_.fallback) o_.fallback->Finish(lease, use, ok);
    return;
  }
  std::lock_guard<std::mutex> lk(mu_);
  const auto it = sessions_.find(lease.session);
  if (it == sessions_.end()) return;
  Session& s = it->second;
  std::optional<Slot>& slot = use == Use::kSeal ? s.seal : s.open;
  if (!slot || slot->gen != lease.gen) return;  // replaced or revoked meanwhile: nothing of ours to finish
  if (!ok && use == Use::kSeal) {
    // Nothing was written (SpillSession leaves the path as it was): the key may seal a retry; the used fence the
    // lease raised goes back (if persisting that fails, the stricter fence stays: safe, the retry is refused).
    slot->reserved = false;
    if (s.last_spill == slot->epoch && lease.prev_fence < s.last_spill) {
      const auto raised = s.last_spill;
      s.last_spill = lease.prev_fence;
      std::string e;
      if (!SaveState(&e)) s.last_spill = raised;
    }
    return;
  }
  // A restore consumes its key whatever the outcome: on failure the cache erases the entry and its file, so the key
  // has nothing left to open. The used fence was persisted by OpenLease.
  ZeroSlot(s, slot, OpState::kConsumed);  // one-shot: zeroed now, its record says consumed
}

bool YardSpillKeys::MayOpen(std::string_view label, std::string_view file_key_id) {
  if (file_key_id.rfind("yard:", 0) != 0) return o_.fallback ? o_.fallback->MayOpen(label, file_key_id) : false;
  const auto e = file_key_id.find(":e", 5);
  std::uint64_t fe = 0;
  if (e == std::string_view::npos) return false;
  const auto end = file_key_id.find(':', e + 2);
  if (end == std::string_view::npos || !ParseU64(file_key_id.substr(e + 2, end - e - 2), std::uint64_t{1} << 62, &fe))
    return false;
  std::lock_guard<std::mutex> lk(mu_);
  const auto it = sessions_.find(label);
  if (it == sessions_.end()) return true;
  const Session& s = it->second;
  return !(s.fence && fe <= *s.fence) && fe >= s.last_restore;  // fenced, or already restored: never again
}

}  // namespace halo::server
