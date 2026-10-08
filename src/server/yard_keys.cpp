#include "server/yard_keys.hpp"

#include <openssl/bn.h>
#include <openssl/core_names.h>
#include <openssl/evp.h>
#include <openssl/param_build.h>

#include <array>
#include <cstring>
#include <ctime>
#include <utility>
#include <fstream>
#include <sstream>

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

bool Base64UrlDecode(std::string_view s, std::vector<std::uint8_t>* out) {
  out->clear();
  while (!s.empty() && s.back() == '=') s.remove_suffix(1);
  std::uint32_t acc = 0;
  int bits = 0;
  for (char c : s) {
    int v;
    if (c >= 'A' && c <= 'Z') v = c - 'A';
    else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
    else if (c >= '0' && c <= '9') v = c - '0' + 52;
    else if (c == '-' || c == '+') v = 62;
    else if (c == '_' || c == '/') v = 63;
    else return false;
    acc = (acc << 6) | static_cast<std::uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out->push_back(static_cast<std::uint8_t>(acc >> bits));
      acc &= (1u << bits) - 1;
    }
  }
  return bits < 6 && acc == 0;  // no dangling non-zero bits (a canonical encoding)
}

// ---- JWKS ----
Jwks::Jwks() = default;
Jwks::~Jwks() {
  for (auto& [k, p] : keys_) EVP_PKEY_free(static_cast<EVP_PKEY*>(p));
}

namespace {

EVP_PKEY* RsaPublic(const std::vector<std::uint8_t>& n, const std::vector<std::uint8_t>& e) {
  BIGNUM* bn = BN_bin2bn(n.data(), static_cast<int>(n.size()), nullptr);
  BIGNUM* be = BN_bin2bn(e.data(), static_cast<int>(e.size()), nullptr);
  OSSL_PARAM_BLD* bld = OSSL_PARAM_BLD_new();
  OSSL_PARAM* params = nullptr;
  EVP_PKEY* pkey = nullptr;
  EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_from_name(nullptr, "RSA", nullptr);
  if (bn && be && bld && ctx && OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_N, bn) == 1 &&
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

std::string Str(const json::Value* v) { return v && v->is(json::Value::Kind::kString) ? v->str() : std::string(); }

std::optional<std::int64_t> Int(const json::Value* v) {
  if (!v || !v->is(json::Value::Kind::kInt)) return std::nullopt;
  const std::string& d = v->int_digits();
  if (d.empty() || d.size() > 18) return std::nullopt;
  return std::stoll(d);
}

std::int64_t Now() { return static_cast<std::int64_t>(std::time(nullptr)); }

std::string Esc(std::string_view s) {  // JSON string contents for messages (printable ASCII only)
  std::string o;
  for (char c : s)
    if (c >= 0x20 && c < 0x7f && c != '"' && c != '\\') o += c;
  return o;
}

}  // namespace

bool Jwks::Parse(std::string_view text, std::string* error) {
  auto doc = json::Parse(text, error);
  if (!doc) return false;
  const json::Value* keys = doc->find("keys");
  if (!keys || !keys->is(json::Value::Kind::kArray)) return *error = "jwks: no keys array", false;
  std::map<std::string, void*, std::less<>> next;
  for (const auto& k : keys->items()) {
    if (Str(k.find("kty")) != "RSA") continue;  // the issuer signs RS256 only
    const std::string alg = Str(k.find("alg"));
    if (!alg.empty() && alg != "RS256") continue;
    const std::string kid = Str(k.find("kid"));
    std::vector<std::uint8_t> n, e;
    if (kid.empty() || !Base64UrlDecode(Str(k.find("n")), &n) || !Base64UrlDecode(Str(k.find("e")), &e) || e.empty())
      continue;
    if (n.size() * 8 < 2048) continue;  // RSA below 2048 bits is refused
    EVP_PKEY* p = RsaPublic(n, e);
    if (!p) continue;
    if (auto it = next.find(kid); it != next.end()) EVP_PKEY_free(static_cast<EVP_PKEY*>(it->second));
    next[kid] = p;
  }
  if (next.empty()) return *error = "jwks: no usable RS256 keys", false;
  for (auto& [k, p] : keys_) EVP_PKEY_free(static_cast<EVP_PKEY*>(p));
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

// ---- JWT ----
std::optional<json::Value> VerifyJwt(std::string_view token, const Jwks& keys, const JwtCheck& c, std::string* error) {
  const auto fail = [&](std::string why) -> std::optional<json::Value> {
    *error = "jwt: " + why;
    return std::nullopt;
  };
  if (token.size() > 16384) return fail("too long");
  const auto d1 = token.find('.');
  const auto d2 = d1 == std::string_view::npos ? d1 : token.find('.', d1 + 1);
  if (d2 == std::string_view::npos || token.find('.', d2 + 1) != std::string_view::npos) return fail("not 3 parts");
  std::vector<std::uint8_t> hb, pb, sig;
  if (!Base64UrlDecode(token.substr(0, d1), &hb) || !Base64UrlDecode(token.substr(d1 + 1, d2 - d1 - 1), &pb) ||
      !Base64UrlDecode(token.substr(d2 + 1), &sig))
    return fail("bad base64url");
  std::string e;
  const auto hdr = json::Parse(std::string_view(reinterpret_cast<const char*>(hb.data()), hb.size()), &e);
  if (!hdr || !hdr->is(json::Value::Kind::kObject)) return fail("header: " + e);
  if (Str(hdr->find("alg")) != "RS256") return fail("alg must be RS256");
  if (hdr->find("crit")) return fail("crit header not supported");
  const std::string kid = Str(hdr->find("kid"));
  const auto* pkey = static_cast<const EVP_PKEY*>(keys.Find(kid));
  if (!pkey) return fail("kid not in the JWKS");
  {
    EVP_MD_CTX* md = EVP_MD_CTX_new();
    const std::string_view signing = token.substr(0, d2);
    const bool ok = md &&
                    EVP_DigestVerifyInit(md, nullptr, EVP_sha256(), nullptr, const_cast<EVP_PKEY*>(pkey)) == 1 &&
                    EVP_DigestVerify(md, sig.data(), sig.size(), reinterpret_cast<const unsigned char*>(signing.data()),
                                     signing.size()) == 1;
    EVP_MD_CTX_free(md);
    if (!ok) return fail("signature");
  }
  auto claims = json::Parse(std::string_view(reinterpret_cast<const char*>(pb.data()), pb.size()), &e);
  if (!claims || !claims->is(json::Value::Kind::kObject)) return fail("payload: " + e);
  const auto iat = Int(claims->find("iat")), exp = Int(claims->find("exp")), nbf = Int(claims->find("nbf"));
  if (!iat || !exp) return fail("iat and exp required");
  const std::int64_t now = c.now ? c.now : Now();
  if (now >= *exp + c.leeway_s) return fail("expired");
  if (nbf && now + c.leeway_s < *nbf) return fail("not yet valid");
  if (*iat > now + c.leeway_s) return fail("issued in the future");
  if (*exp - *iat > c.max_ttl_s) return fail("lifetime exceeds " + std::to_string(c.max_ttl_s) + " s");
  if (!c.issuer.empty() && Str(claims->find("iss")) != c.issuer) return fail("issuer");
  if (!c.audience.empty()) {
    const json::Value* aud = claims->find("aud");
    bool ok = Str(aud) == c.audience;
    if (aud && aud->is(json::Value::Kind::kArray))
      for (const auto& a : aud->items()) ok = ok || Str(&a) == c.audience;
    if (!ok) return fail("audience");
  }
  return claims;
}

// ---- the provider ----
YardSpillKeys::YardSpillKeys(Options o) : o_(std::move(o)) {}

bool YardSpillKeys::LoadJwks(std::string* error) {
  std::lock_guard<std::mutex> lk(mu_);
  jwks_loaded_ = std::chrono::steady_clock::now();
  return jwks_.Load(o_.jwks_path, error);
}

bool YardSpillKeys::ValidSession(std::string_view s) {
  if (s.empty() || s.size() > 128) return false;
  for (char c : s)
    if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
      return false;
  return true;
}

bool YardSpillKeys::Expired(const Slot& s) const {
  return std::chrono::steady_clock::now() - s.at > std::chrono::seconds(o_.key_ttl_s);
}

YardSpillKeys::Answer YardSpillKeys::Register(std::string_view session, std::string_view body, std::int64_t now) {
  const auto answer = [](int status, const std::string& msg) {
    return Answer{status, status == 200 ? msg : "{\"ok\":false,\"error\":\"" + Esc(msg) + "\"}"};
  };
  if (!ValidSession(session)) return answer(400, "session must match [A-Za-z0-9._-]{1,128}");
  std::string e;
  const auto req = json::Parse(body, &e, {.max_depth = 4, .max_bytes = 32768});
  if (!req || !req->is(json::Value::Kind::kObject)) return answer(400, "body must be a JSON object");
  const std::string token = Str(req->find("token")), use = Str(req->find("use")), key_b64 = Str(req->find("key"));
  if (use != "seal" && use != "open" && use != "revoke") return answer(400, "use must be seal, open or revoke");
  crypt::SecretBytes key;
  std::array<std::uint8_t, 32> key_hash{};
  if (use != "revoke") {
    std::vector<std::uint8_t> raw;
    if (!Base64UrlDecode(key_b64, &raw) || raw.size() != crypt::kKeyBytes) {
      crypt::Wipe(raw.data(), raw.size());
      return answer(400, "key must be 32 bytes, base64url");
    }
    key = crypt::SecretBytes(crypt::kKeyBytes);
    if (!key.ok()) return answer(500, "cannot allocate key memory");
    std::memcpy(key.data(), raw.data(), raw.size());
    crypt::Wipe(raw.data(), raw.size());
    key_hash = crypt::Sha256(key.span());
  }
  std::lock_guard<std::mutex> lk(mu_);
  // The token. An unknown kid re-reads the JWKS file (yard rotated its key), at most once a minute.
  JwtCheck check{.issuer = o_.issuer, .audience = o_.audience, .now = now};
  auto claims = VerifyJwt(token, jwks_, check, &e);
  if (!claims && e == "jwt: kid not in the JWKS" &&
      std::chrono::steady_clock::now() - jwks_loaded_ > std::chrono::seconds(60)) {
    std::string e2;
    jwks_loaded_ = std::chrono::steady_clock::now();
    if (jwks_.Load(o_.jwks_path, &e2)) claims = VerifyJwt(token, jwks_, check, &e);
  }
  if (!claims) return answer(401, e);
  if (Str(claims->find("yard_sess")) != session) return answer(401, "token is for another session");
  const auto epoch = Int(claims->find("yard_epoch"));
  if (!epoch || *epoch < 1) return answer(401, "token has no yard_epoch");
  const std::string jti = Str(claims->find("jti"));
  if (jti.empty() || jti.size() > 128) return answer(401, "token has no jti");
  const json::Value* hs = claims->find("halo_spill");
  if (!hs || Str(hs->find("use")) != use) return answer(401, "token does not authorise use " + use);
  if (use != "revoke") {
    std::vector<std::uint8_t> want;
    if (!Base64UrlDecode(Str(hs->find("key_sha256")), &want) || want.size() != 32 ||
        !crypt::Equal(want, key_hash))
      return answer(401, "token is bound to another key");
  }
  // Single use: a replayed token is refused (kept until it expires anyway).
  const std::int64_t t = now ? now : Now();
  for (auto it = seen_jti_.begin(); it != seen_jti_.end();) it = it->second + 60 < t ? seen_jti_.erase(it) : std::next(it);
  if (seen_jti_.contains(jti)) return answer(409, "token replayed (jti already used)");
  const std::int64_t exp = Int(claims->find("exp")).value_or(t);
  const auto E = static_cast<std::uint64_t>(*epoch);
  Session& s = sessions_[std::string(session)];
  if (E < s.high)
    return answer(409, "stale epoch " + std::to_string(E) + " (session at " + std::to_string(s.high) + ")");
  const std::string name = use == "revoke" ? "" : crypt::KeyName(key.span());
  const auto ok = [&](bool duplicate, std::uint64_t file_epoch) {
    seen_jti_[jti] = exp;
    std::string kid = use == "revoke" ? "" : "yard:" + name + ":e" + std::to_string(file_epoch) + ":" + std::string(session);
    return answer(200, "{\"ok\":true,\"session\":\"" + Esc(session) + "\",\"use\":\"" + use + "\",\"epoch\":" +
                           std::to_string(E) + ",\"key_id\":\"" + Esc(kid) + "\",\"duplicate\":" +
                           (duplicate ? "true" : "false") + "}");
  };
  if (use == "revoke") {
    s.seal.reset();
    s.open.reset();
    s.high = E;
    revoked_.emplace_back(session);
    return ok(false, 0);
  }
  if (use == "seal") {
    const bool dup = s.seal && s.seal->epoch == E && s.seal->name == name && !Expired(*s.seal);
    if (!dup) {
      s.seal = Slot{E, std::move(key), name, std::chrono::steady_clock::now()};
      s.open.reset();  // a new checkpoint: an older open key has nothing left to open
    }
    s.high = E;
    s.sealed = E;
    return ok(dup, E);
  }
  // open: only after a seal, at a later epoch; it opens the file of epoch E - 1.
  if (!s.sealed) return answer(409, "no sealed state of this session here");
  if (E <= *s.sealed)
    return answer(409, "open at epoch " + std::to_string(E) + " is not after the seal at " + std::to_string(*s.sealed));
  const bool dup = s.open && s.open->epoch == E && s.open->name == name && !Expired(*s.open);
  if (!dup) {
    s.open = Slot{E, std::move(key), name, std::chrono::steady_clock::now()};
    s.seal.reset();
  }
  s.high = E;
  return ok(dup, E - 1);
}

std::string YardSpillKeys::ListJson() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::string o = "{\"sessions\":[";
  bool first = true;
  for (const auto& [name, s] : sessions_) {
    const auto slot = [&](const std::optional<Slot>& x) {
      return x && !Expired(*x) ? "{\"epoch\":" + std::to_string(x->epoch) + ",\"key_name\":\"" + x->name + "\"}"
                               : std::string("null");
    };
    o += std::string(first ? "" : ",") + "{\"session\":\"" + Esc(name) + "\",\"high_epoch\":" + std::to_string(s.high) +
         ",\"sealed_epoch\":" + (s.sealed ? std::to_string(*s.sealed) : "null") + ",\"seal\":" + slot(s.seal) +
         ",\"open\":" + slot(s.open) + "}";
    first = false;
  }
  return o + "]}";
}

std::vector<std::string> YardSpillKeys::TakeRevoked() {
  std::lock_guard<std::mutex> lk(mu_);
  return std::exchange(revoked_, {});
}

std::string YardSpillKeys::Describe() const {
  std::lock_guard<std::mutex> lk(mu_);
  return "yard (" + std::to_string(jwks_.size()) + " JWKS keys, aud " + o_.audience + ", fallback " +
         (o_.fallback ? o_.fallback->Describe() : std::string("refuse")) + ")";
}

bool YardSpillKeys::Key(std::string_view session, Use use, crypt::Kek* out, std::string* why) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = sessions_.find(session);
    if (it != sessions_.end()) {
      const auto& slot = use == Use::kSeal ? it->second.seal : it->second.open;
      if (slot && !Expired(*slot)) {
        const std::uint64_t file_epoch = use == Use::kSeal ? slot->epoch : slot->epoch - 1;
        return crypt::DeriveKek(slot->key.span(), session, file_epoch, "yard:" + slot->name, out, why);
      }
    }
  }
  if (o_.fallback) return o_.fallback->Key(session, use, out, why);
  if (why) *why = "no yard key registered for session " + std::string(session) + " (" +
                  (use == Use::kSeal ? "seal" : "open") + ")";
  return false;
}

bool YardSpillKeys::KeyForFile(std::string_view session, std::string_view file_key_id, crypt::Kek* out,
                               std::string* why) {
  // A file sealed under a yard key opens only with the open key yard-ctl registers; one sealed by the fallback
  // provider (a session without a yard key at spill time) opens with the fallback.
  if (file_key_id.rfind("yard:", 0) == 0) {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = sessions_.find(session);
    if (it != sessions_.end() && it->second.open && !Expired(*it->second.open)) {
      const auto& slot = *it->second.open;
      return crypt::DeriveKek(slot.key.span(), session, slot.epoch - 1, "yard:" + slot.name, out, why);
    }
    if (why) *why = "no yard open key registered for session " + std::string(session);
    return false;
  }
  if (o_.fallback) return o_.fallback->KeyForFile(session, file_key_id, out, why);
  if (why) *why = "file not sealed by yard and no fallback provider";
  return false;
}

void YardSpillKeys::Used(std::string_view session, Use use) {
  std::lock_guard<std::mutex> lk(mu_);
  const auto it = sessions_.find(session);
  if (it == sessions_.end()) {
    if (o_.fallback) o_.fallback->Used(session, use);
    return;
  }
  // One-shot: the key that sealed (or opened) the file is forgotten (zeroed) now.
  (use == Use::kSeal ? it->second.seal : it->second.open).reset();
}

}  // namespace halo::server
