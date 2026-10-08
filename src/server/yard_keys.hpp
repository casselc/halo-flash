#ifndef HALO_SERVER_YARD_KEYS_HPP_
#define HALO_SERVER_YARD_KEYS_HPP_

// Spill keys provided by yard-ctl, per session (phase B; docs/SESSIONS.md section 10, "The key route").
//
// yard-ctl registers a 256-bit key for one of its sessions over POST /admin/keys/{session}. The request carries a
// short-lived JWT from yard-ctl's STS (RS256, kid = RFC 7638 thumbprint, verified against yard's JWKS file):
//   aud = halo_serve (configurable), iss = yard's issuer, exp - iat <= 300 s, a jti (single use here),
//   yard_sess = the session (must equal the route's), yard_epoch = the session's credential epoch,
//   halo_spill = {"use": "seal" | "open" | "revoke", "key_sha256": base64url(SHA-256(key))}  (binds the body's key).
// The key is in the JSON body, never in a header, never logged. halo_serve keeps it in mlocked memory only.
//
// Epoch rules (yard's lifecycle bumps the epoch at SuspendCommit: the checkpoint is taken at E, the resume runs
// at E + 1):
//   - seal at E (>= every epoch seen for the session): the next spill of the session is sealed under
//     KEK = HKDF(key, "halo-spill-v2" || session || E); the key is forgotten once that spill is done;
//   - open at E (> the last seal epoch): opens a file sealed at exactly E - 1, so only a token minted after the
//     suspend committed can authorise the resume, and a second epoch bump (a revocation) leaves the file unreadable;
//     the key is forgotten once that restore is done;
//   - revoke at E (>= every epoch seen): forgets the session's keys; its spilled entries are crypto-shredded;
//   - a registration below the session's highest epoch seen is refused (stale), a replayed jti is refused, the
//     same registration again (same use, epoch and key) is a no-op success (a retry after a lost answer).
// Unused keys expire after `key_ttl_s`.

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/json.hpp"
#include "core/spill_crypt.hpp"
#include "server/spill_keys.hpp"

namespace halo::server {

/// base64url without padding (RFC 4648 section 5); decoding also takes standard base64 and padding.
[[nodiscard]] std::string Base64UrlEncode(std::span<const std::uint8_t> bytes);
[[nodiscard]] bool Base64UrlDecode(std::string_view text, std::vector<std::uint8_t>* out);

/// RS256 public keys by kid, from a JWKS document ({"keys":[{"kty":"RSA","kid","n","e"}...]}).
class Jwks {
 public:
  Jwks();
  ~Jwks();
  Jwks(const Jwks&) = delete;
  Jwks& operator=(const Jwks&) = delete;
  [[nodiscard]] bool Parse(std::string_view json, std::string* error);
  [[nodiscard]] bool Load(const std::string& path, std::string* error);
  [[nodiscard]] const void* Find(std::string_view kid) const;  // EVP_PKEY*, or null
  [[nodiscard]] std::size_t size() const { return keys_.size(); }

 private:
  std::map<std::string, void*, std::less<>> keys_;  // kid -> EVP_PKEY*
};

struct JwtCheck {
  std::string issuer;    ///< required iss (empty: not checked)
  std::string audience;  ///< required aud (a string, or an array containing it)
  std::int64_t max_ttl_s = 300;
  std::int64_t leeway_s = 30;
  std::int64_t now = 0;  ///< seconds since the epoch (0: the clock)
};

/// Verifies a compact RS256 JWT (only RS256: "none", HS* and others are refused) and returns its claims.
[[nodiscard]] std::optional<json::Value> VerifyJwt(std::string_view token, const Jwks& keys, const JwtCheck& check,
                                                   std::string* error);

/// The phase-B provider and the registry behind POST /admin/keys/{session}.
class YardSpillKeys final : public SpillKeys {
 public:
  struct Options {
    std::string jwks_path;          ///< yard's JWKS (jwks.Publisher writes <dir>/.well-known/jwks.json)
    std::string issuer;             ///< yard's iss
    std::string audience = "halo_serve";
    std::int64_t key_ttl_s = 600;   ///< an unused registered key is forgotten after this
    SpillKeys* fallback = nullptr;  ///< sessions without a registered key: this provider (A), or refused (null)
  };
  struct Answer {
    int status = 500;  ///< 200 done, 400 malformed, 401 token refused, 409 refused by the epoch/replay rules
    std::string body;  ///< JSON; never key bytes or the token
  };
  explicit YardSpillKeys(Options o);
  [[nodiscard]] bool LoadJwks(std::string* error);
  /// POST /admin/keys/{session} with the JSON body {"token","use","key"} (`key` absent for revoke).
  [[nodiscard]] Answer Register(std::string_view session, std::string_view body, std::int64_t now = 0);
  /// GET /admin/keys: per session, the highest epoch seen, the last seal epoch, and the registered slots (use,
  /// epoch, key name = HMAC name of the key, not the key).
  [[nodiscard]] std::string ListJson() const;
  /// Sessions revoked since the last call (the admin route erases their spilled entries).
  [[nodiscard]] std::vector<std::string> TakeRevoked();

  [[nodiscard]] std::string Describe() const override;
  [[nodiscard]] bool Key(std::string_view session, Use use, core::crypt::Kek* out, std::string* why) override;
  [[nodiscard]] bool KeyForFile(std::string_view session, std::string_view file_key_id, core::crypt::Kek* out,
                                std::string* why) override;
  void Used(std::string_view session, Use use) override;

  static bool ValidSession(std::string_view s);  ///< yard's rule: [A-Za-z0-9._-]{1,128}

 private:
  struct Slot {
    std::uint64_t epoch = 0;
    core::crypt::SecretBytes key;
    std::string name;  // KeyName(key)
    std::chrono::steady_clock::time_point at;
  };
  struct Session {
    std::uint64_t high = 0;                 // highest epoch of any accepted registration
    std::optional<std::uint64_t> sealed;    // epoch of the last seal registration
    std::optional<Slot> seal, open;
  };
  bool Expired(const Slot& s) const;
  Options o_;
  mutable std::mutex mu_;
  Jwks jwks_;
  std::chrono::steady_clock::time_point jwks_loaded_{};
  std::map<std::string, Session, std::less<>> sessions_;
  std::map<std::string, std::int64_t, std::less<>> seen_jti_;  // jti -> exp (pruned once expired)
  std::vector<std::string> revoked_;
};

}  // namespace halo::server

#endif  // HALO_SERVER_YARD_KEYS_HPP_
