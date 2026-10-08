// Test helpers for the yard key route: an RS256 signer with its JWKS, and yard-shaped tokens
// (internal/sts/jwt.go's format: header {"alg","typ","kid"}, claims iss, aud, sub, iat, nbf, exp, jti, yard_sess,
// yard_epoch, plus the halo_spill claim).
#ifndef HALO_TESTS_SERVER_YARD_TEST_UTIL_HPP_
#define HALO_TESTS_SERVER_YARD_TEST_UTIL_HPP_

#include <openssl/bn.h>
#include <openssl/core_names.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>

#include <array>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

#include "core/spill_crypt.hpp"
#include "server/yard_keys.hpp"

namespace yardtest {

inline std::string B64(const std::string& s) {
  return halo::server::Base64UrlEncode({reinterpret_cast<const std::uint8_t*>(s.data()), s.size()});
}

struct Signer {
  EVP_PKEY* key = nullptr;
  std::string kid;
  explicit Signer(std::string k) : key(EVP_RSA_gen(2048)), kid(std::move(k)) {}
  ~Signer() { EVP_PKEY_free(key); }
  Signer(const Signer&) = delete;
  Signer& operator=(const Signer&) = delete;
  std::string Jwk() const {
    BIGNUM *n = nullptr, *e = nullptr;
    EVP_PKEY_get_bn_param(key, OSSL_PKEY_PARAM_RSA_N, &n);
    EVP_PKEY_get_bn_param(key, OSSL_PKEY_PARAM_RSA_E, &e);
    std::vector<std::uint8_t> nb(static_cast<std::size_t>(BN_num_bytes(n))), eb(static_cast<std::size_t>(BN_num_bytes(e)));
    BN_bn2bin(n, nb.data());
    BN_bn2bin(e, eb.data());
    BN_free(n);
    BN_free(e);
    return "{\"kty\":\"RSA\",\"kid\":\"" + kid + "\",\"use\":\"sig\",\"alg\":\"RS256\",\"n\":\"" +
           halo::server::Base64UrlEncode(nb) + "\",\"e\":\"" + halo::server::Base64UrlEncode(eb) + "\"}";
  }
  std::string Sign(const std::string& header_json, const std::string& claims_json) const {
    const std::string signing = B64(header_json) + "." + B64(claims_json);
    EVP_MD_CTX* md = EVP_MD_CTX_new();
    std::size_t n = 0;
    EVP_DigestSignInit(md, nullptr, EVP_sha256(), nullptr, key);
    EVP_DigestSign(md, nullptr, &n, reinterpret_cast<const unsigned char*>(signing.data()), signing.size());
    std::vector<std::uint8_t> sig(n);
    EVP_DigestSign(md, sig.data(), &n, reinterpret_cast<const unsigned char*>(signing.data()), signing.size());
    EVP_MD_CTX_free(md);
    sig.resize(n);
    return signing + "." + halo::server::Base64UrlEncode(sig);
  }
};

inline std::string JwksJson(std::initializer_list<const Signer*> s) {
  std::string o = "{\"keys\":[";
  bool first = true;
  for (const auto* k : s) o += (first ? "" : ",") + k->Jwk(), first = false;
  return o + "]}";
}

struct Claims {
  std::string iss = "https://id.example.org/yard", aud = "halo_serve", sess = "sess-1", use = "seal", jti;
  std::int64_t epoch = 1, iat = 0, exp = 0;  // 0: now, now + 300
  std::string key_sha256;                    // base64url SHA-256 of the key ("" = none)
  bool with_halo_spill = true;
};

inline std::string Key32(std::uint8_t seed) {
  std::string k(32, '\0');
  for (int i = 0; i < 32; ++i) k[static_cast<std::size_t>(i)] = static_cast<char>(seed * 31 + i * 7);
  return k;
}
inline std::string KeyB64(const std::string& k) { return B64(k); }
inline std::string KeyHash(const std::string& k) {
  const auto h = halo::core::crypt::Sha256({reinterpret_cast<const std::uint8_t*>(k.data()), k.size()});
  return halo::server::Base64UrlEncode(h);
}

inline std::string Token(const Signer& s, Claims c, const std::string& alg = "RS256") {
  static int counter = 0;
  const std::int64_t now = static_cast<std::int64_t>(std::time(nullptr));
  if (!c.iat) c.iat = now;
  if (!c.exp) c.exp = c.iat + 300;
  if (c.jti.empty()) c.jti = "jti-" + std::to_string(++counter) + "-" + std::to_string(now);
  std::string body = "{\"iss\":\"" + c.iss + "\",\"aud\":\"" + c.aud + "\",\"sub\":\"chuck\",\"iat\":" +
                     std::to_string(c.iat) + ",\"nbf\":" + std::to_string(c.iat) + ",\"exp\":" + std::to_string(c.exp) +
                     ",\"jti\":\"" + c.jti + "\",\"yard_kind\":\"internal\",\"yard_sess\":\"" + c.sess +
                     "\",\"yard_epoch\":" + std::to_string(c.epoch);
  if (c.with_halo_spill)
    body += ",\"halo_spill\":{\"use\":\"" + c.use + "\",\"key_sha256\":\"" + c.key_sha256 + "\"}";
  body += "}";
  return s.Sign("{\"alg\":\"" + alg + "\",\"typ\":\"JWT\",\"kid\":\"" + s.kid + "\"}", body);
}

// A registration body; `key` raw bytes ("" for revoke).
inline std::string Body(const std::string& token, const std::string& use, const std::string& key) {
  std::string b = "{\"token\":\"" + token + "\",\"use\":\"" + use + "\"";
  if (!key.empty()) b += ",\"key\":\"" + KeyB64(key) + "\"";
  return b + "}";
}

// One-call registration with a fresh, correct token.
inline halo::server::YardSpillKeys::Answer Register(halo::server::YardSpillKeys& y, const Signer& s,
                                                    const std::string& sess, const std::string& use,
                                                    std::int64_t epoch, const std::string& key) {
  Claims c;
  c.sess = sess;
  c.use = use;
  c.epoch = epoch;
  c.key_sha256 = key.empty() ? "" : KeyHash(key);
  return y.Register(sess, Body(Token(s, c), use, key));
}

inline void WriteFile(const std::string& path, const std::string& s) { std::ofstream(path) << s; }

}  // namespace yardtest

#endif  // HALO_TESTS_SERVER_YARD_TEST_UTIL_HPP_
