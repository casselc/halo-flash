#ifndef HALO_SERVER_YARD_KEYS_HPP_
#define HALO_SERVER_YARD_KEYS_HPP_

// Spill keys provided by yard-ctl, per checkpoint (phase B; docs/SESSIONS.md section 10, "The key route").
//
// yard-ctl registers a fresh random 256-bit key K per checkpoint over POST /admin/keys/{session} (on the admin Unix
// socket by default). The request carries a short-lived JWT from yard-ctl's STS (RS256, kid = RFC 7638 thumbprint,
// verified against yard's JWKS file):
//   aud (mandatory, default halo_serve), iss (mandatory), 0 < exp - iat <= 300 s, a jti (single use), yard_sess =
//   the route's session, yard_epoch = the session's credential epoch, and
//   halo_spill = {"use": "seal" | "open" | "revoke", "key_sha256": base64url(SHA-256(K)) (seal, open),
//                 "entry": the cache entry id "<incarnation>.<key>" (seal), "through": the fenced epoch (revoke,
//                 optional; default: the last spilled epoch, none if nothing was spilled)}.
// The key is in the JSON body, never in a header, never logged; halo_serve holds it in mlocked memory only.
//
// Per-session state (the halo_serve side of yard SR1's spec/keys model), persisted before any registration is
// acknowledged: highest epoch seen, revoke fence, last seal epoch, last spilled and last restored epochs.
//   - seal at E: E > fence, E >= highest, E > last spill; the next spill of cache entry `entry` is sealed under
//     KEK = HKDF(K, "halo-spill-v2" || be32(len) || session || be64(E)); consumed by that spill;
//   - open at E: E - 1 > fence, E >= highest, E == last seal + 1, E > last restore; opens exactly the file of
//     E - 1; consumed by that restore;
//   - revoke at token epoch T through R (R <= T; default the last spilled epoch, none if nothing was spilled):
//     fence = max(fence, R); every key at a file epoch <= fence (all keys when there is no R) is zeroed, the last
//     seal is forgotten if fenced, spilled entries at those epochs are erased; never stale, idempotent;
//   - a retry (fresh token, same use, epoch, key and entry) is a no-op 200; after the key was consumed it returns
//     the recorded outcome ("consumed") and never re-arms the key;
//   - a key's authorization ends at min(token exp, registration + key TTL); expired keys are zeroed by a sweeper.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "core/json.hpp"
#include "core/spill_crypt.hpp"
#include "server/spill_keys.hpp"

namespace halo::server {

/// base64url without padding (RFC 4648 section 5).
[[nodiscard]] std::string Base64UrlEncode(std::span<const std::uint8_t> bytes);
/// Strict base64url: only [A-Za-z0-9_-], no padding, canonical (zero trailing bits).
[[nodiscard]] bool Base64UrlDecode(std::string_view text, std::vector<std::uint8_t>* out);
/// Strict base64url of exactly out.size() bytes, decoded straight into `out` (no intermediate copies).
[[nodiscard]] bool Base64UrlDecodeExact(std::string_view text, std::span<std::uint8_t> out);

/// RS256 public keys by kid, from a JWKS document ({"keys":[{"kty":"RSA","kid","n","e"}...]}). A key is usable
/// only if: kty RSA, alg absent or RS256, use absent or "sig", key_ops absent or containing "verify", n a canonical
/// base64url integer of 2048..16384 bits, e = 65537. A duplicate kid makes the whole document invalid. An empty
/// (or all-unusable) key set is valid and verifies nothing.
class Jwks {
 public:
  Jwks();
  ~Jwks();
  Jwks(const Jwks&) = delete;
  Jwks& operator=(const Jwks&) = delete;
  [[nodiscard]] bool Parse(std::string_view json, std::string* error);
  [[nodiscard]] bool Load(const std::string& path, std::string* error);
  void Clear();
  [[nodiscard]] const void* Find(std::string_view kid) const;  // EVP_PKEY*, or null
  [[nodiscard]] std::size_t size() const { return keys_.size(); }

 private:
  std::map<std::string, void*, std::less<>> keys_;  // kid -> EVP_PKEY*
};

enum class JwtError : std::uint8_t {
  kOk, kMalformed, kAlg, kUnknownKid, kSignature, kClaims, kExpired, kNotYetValid, kLifetime, kIssuer, kAudience,
  kConfig
};
[[nodiscard]] const char* JwtErrorName(JwtError e);

struct JwtCheck {
  std::string issuer;    ///< required iss (empty: refused as kConfig)
  std::string audience;  ///< required aud (a string, or an array containing it; empty: refused as kConfig)
  std::int64_t max_ttl_s = 300;
  std::int64_t skew_s = 30;  ///< iat / nbf may be this far in the future (clock skew); exp has no leeway
  std::int64_t now = 0;      ///< seconds since the epoch (0: the clock)
};

/// Verifies a compact RS256 JWT (PKCS#1 v1.5, SHA-256; only RS256) and returns its claims.
[[nodiscard]] std::optional<json::Value> VerifyJwt(std::string_view token, const Jwks& keys, const JwtCheck& check,
                                                   JwtError* error);

/// The phase-B provider and the registry behind POST /admin/keys/{session}.
class YardSpillKeys final : public SpillKeys {
 public:
  struct Options {
    std::string jwks_path;          ///< yard's JWKS (jwks.Publisher writes <dir>/.well-known/jwks.json)
    std::string issuer;             ///< yard's iss (mandatory)
    std::string audience = "halo_serve";  ///< mandatory
    std::int64_t key_ttl_s = 600;   ///< a registered key is zeroed at most this long after its first registration
    std::int64_t jwks_refresh_s = 60;  ///< the JWKS file is re-read at least this often (and whenever it changes)
    std::int64_t jwks_forced_gap_ms = 5000;  ///< an unknown kid forces a re-read at most this often
    std::string state_path;         ///< the persisted per-session epochs (0600, owned via "<path>.lock"); empty: memory
    SpillKeys* fallback = nullptr;  ///< entries never yard-bound: this provider (A), or refused (null)
    bool sweeper = true;            ///< a thread that zeroes expired keys (tests may sweep by hand)
    std::string incarnation;        ///< this server incarnation's id: 32 hex digits (empty: 128 random bits)
    std::size_t max_sessions = 4096;  ///< persisted sessions (the state file is rewritten whole: bounded)
    std::size_t max_ops_per_session = 1024;  ///< operation tombstones; a full table refuses new operations (409)
    std::function<void()> clock_gap_for_test;  ///< test seam: runs between the steady and the wall clock samples
  };
  struct Answer {
    int status = 500;  ///< 200 done, 400 malformed, 401 token refused, 409 refused by the rules, 500 not done, 503 stopped
    std::string body;  ///< JSON; never key bytes or the token
    std::string revoked_session;  ///< a revoke was applied: erase this session's entries at epochs <= revoked_through
    std::uint64_t revoked_through = 0;
    std::string reason;  ///< 401: why the token was refused (logs and tests; never sent)
  };
  /// Validates the options, takes the state lock and loads the persisted state; null with `error` on any
  /// misconfiguration, a held lock, or unsafe / malformed / missing-after-init state.
  static std::unique_ptr<YardSpillKeys> Create(Options o, std::string* error);
  ~YardSpillKeys() override;

  /// POST /admin/keys/{session}. `body` is wiped (its bytes overwritten) before return. `now`: seconds since the
  /// epoch for the token checks (0: the clock; tests).
  [[nodiscard]] Answer Register(std::string_view session, std::string& body, std::int64_t now = 0);
  /// GET /admin/keys: the incarnation, and per session the epochs and the live keys (epoch, key name, entry).
  [[nodiscard]] std::string ListJson();
  /// Zeroes every expired key now (the sweeper calls it once a second). Returns how many.
  std::size_t Sweep();
  /// Re-reads the JWKS if the file changed or the refresh interval passed; `force` (an unknown kid) at most once per
  /// jwks_forced_gap_ms. Loads and validates outside the provider lock, then swaps (never blocks leases or sweeps).
  void RefreshJwks(bool force = false);
  [[nodiscard]] std::uint64_t jwks_loads() const { return jwks_loads_.load(); }
  /// The id this incarnation's cache entries carry ("<incarnation>.<key>"); a seal for another incarnation's entry
  /// (a token minted before a restart) is refused.
  [[nodiscard]] const std::string& incarnation() const { return o_.incarnation; }
  /// Persistence became uncertain (a state file published but not durable): the provider stopped (every key zeroed,
  /// every registration and lease refused) until halo_serve restarts and reloads what the disk holds.
  [[nodiscard]] bool stopped() const;
  /// Test seam: holds the provider's state lock (a registration then waits at its transition boundary).
  [[nodiscard]] std::unique_lock<std::mutex> LockForTest() { return std::unique_lock<std::mutex>(mu_); }

  [[nodiscard]] std::string Describe() const override;
  [[nodiscard]] bool SealLease(std::string_view entry, std::string_view label, bool yard_bound, SpillLease* out,
                               std::string* why) override;
  [[nodiscard]] bool OpenLease(std::string_view label, std::string_view file_key_id, SpillLease* out,
                               std::string* why) override;
  void Finish(const SpillLease& lease, Use use, Outcome outcome) override;
  [[nodiscard]] bool MayOpen(std::string_view label, std::string_view file_key_id) override;

  static bool ValidSession(std::string_view s);  ///< yard's rule: [A-Za-z0-9._-]{1,128}

 private:
  explicit YardSpillKeys(Options o);
  using Clock = std::chrono::steady_clock;
  struct Slot {
    std::uint64_t epoch = 0;  // registration epoch (seal: the file epoch; open: file epoch + 1)
    core::crypt::SecretBytes key;
    std::array<std::uint8_t, 32> key_hash{};
    std::string name;   // KeyName(key): display and ids only
    std::string entry;  // seal: the cache entry id it is for
    std::string op;     // the signed operation id (halo_spill.op; "" when the token carries none)
    Clock::time_point deadline, cap;  // authorization end; cap = the operation's first registration + TTL
    std::uint64_t gen = 0;
    bool reserved = false;  // a spill / restore holds a lease on it
  };
  enum class OpState : std::uint8_t { kRegistered, kConsumed, kExpired, kRevoked };
  /// Non-secret: what happened to one operation. Identity: the signed op id when the token carries one, and always
  /// (use, epoch, key hash) too (K is fresh per checkpoint); a record matches a request on either.
  struct OpRecord {
    Use use;
    std::uint64_t epoch;
    std::array<std::uint8_t, 32> key_hash;
    std::string entry;
    std::string op;
    OpState state;
    Clock::time_point cap;         // first registration + TTL: a re-registration after expiry never passes it
    Clock::time_point keep_until;  // kRevoked: kept at least this long (no pre-cancel token can still arrive)
  };
  struct Epochs {  // the persisted part
    std::uint64_t high = 0;
    std::optional<std::uint64_t> fence, last_seal;
    std::uint64_t last_spill = 0, last_restore = 0;
  };
  struct Session {
    Epochs e;
    std::optional<Slot> seal, open;
    std::vector<OpRecord> ops;  // tombstones, each until its keep_until (bounded: max_ops_per_session)
  };
  bool LoadState(std::string* error);
  /// Publishes the state with `cand` as `name`'s epochs (under mu_). kDurable: memory takes `cand`. kUnpublished:
  /// nothing changes. kUncertain: memory takes `cand` (what the disk shows) and the provider stops.
  Persist Commit(std::string_view name, const Epochs& cand, std::string* why);
  void Stop(const std::string& why);  // under mu_
  void ZeroSlot(Session& s, std::optional<Slot>& slot, OpState why);
  OpRecord* Record(Session& s, Use use, std::uint64_t epoch, const std::array<std::uint8_t, 32>& h,
                   const std::string& entry, const std::string& op, OpState st);
  OpRecord* FindOp(Session& s, Use use, std::uint64_t epoch, const std::array<std::uint8_t, 32>& h,
                   const std::string& op);
  std::size_t SweepLocked(Clock::time_point now);
  static bool SameOperation(const std::string& a_op, std::uint64_t a_epoch, const std::array<std::uint8_t, 32>& a_h,
                            const std::string& b_op, std::uint64_t b_epoch, const std::array<std::uint8_t, 32>& b_h);
  void PruneOps(Session& s, Clock::time_point now);  // tombstones past their keep_until
  /// The steady anchor, then the wall clock (ms; `fixed_now` seconds if non-zero).
  Clock::time_point SampleClocks(std::int64_t fixed_now, std::int64_t* wall_ms) const;
  std::shared_ptr<const Jwks> Keys() const;

  Options o_;
  mutable std::mutex mu_;  // sessions, jtis, slots; never held across JWKS loading
  std::map<std::string, Session, std::less<>> sessions_;
  std::map<std::string, std::int64_t, std::less<>> seen_jti_;  // jti -> exp
  std::uint64_t gen_ = 0;
  bool stopped_ = false;
  std::unique_ptr<StateDir> dir_;
  std::string state_name_;
  std::unique_ptr<StateLock> lock_;
  // JWKS: an immutable snapshot swapped under jwks_mu_; refresh bookkeeping under refresh_mu_.
  mutable std::mutex jwks_mu_;
  std::shared_ptr<const Jwks> jwks_;
  std::mutex refresh_mu_;
  Clock::time_point jwks_loaded_{}, jwks_forced_{};
  std::uint64_t jwks_sig_[4] = {0, 0, 0, 0};  // ino, size, mtime ns, ctime ns of the loaded file
  std::atomic<std::uint64_t> jwks_loads_{0};
  std::thread sweeper_;
  std::condition_variable cv_;
  bool stop_ = false;
};

}  // namespace halo::server

#endif  // HALO_SERVER_YARD_KEYS_HPP_
