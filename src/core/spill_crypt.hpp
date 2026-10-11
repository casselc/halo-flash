#ifndef HALO_CORE_SPILL_CRYPT_HPP_
#define HALO_CORE_SPILL_CRYPT_HPP_

// Envelope encryption for session spill files (format v2, docs/SESSIONS.md section 10), on OpenSSL libcrypto
// (EVP AES-256-GCM: the VAES / AVX-512 path on Zen 5, ~14 GB/s a core).
//
//   master (provider: TPM-sealed credential, file, or a yard-ctl registered session key)
//     -> KEK = HKDF-SHA256(ikm = master, salt = none, info = "halo-spill-v2" || be32(len) || session || be64(epoch))
//     -> wraps a fresh random 256-bit DEK per file (AES-256-GCM, random 96-bit nonce, AAD = the file's header)
//   DEK -> each 16 MB payload piece and the meta: AES-256-GCM, nonce = file salt (8 random bytes) || be32(piece
//          index) (the meta uses index 0xffffffff), AAD = the header. The tags replace v1's CRC32Cs.
//
// Key material lives in SecretBytes (mlocked, kept out of core dumps, zeroed when freed); nothing here logs keys.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace halo::core::crypt {

constexpr std::size_t kKeyBytes = 32, kNonceBytes = 12, kTagBytes = 16, kSaltBytes = 8;
constexpr std::uint32_t kMetaIndex = 0xffffffffu;  ///< the nonce index of the meta (pieces use 0 .. n-1)

/// Bytes holding key material: their own pages (mmap), mlocked, excluded from core dumps (MADV_DONTDUMP) and from
/// forked children (MADV_WIPEONFORK), zeroed (OPENSSL_cleanse) before they are unmapped. Fails closed: if any of
/// mmap, mlock or madvise fails, the object is empty (`ok()` false) and every caller refuses (a service needs
/// RLIMIT_MEMLOCK of a few pages per live key: docs/SESSIONS.md section 10). Movable, not copyable.
class SecretBytes {
 public:
  SecretBytes() = default;
  explicit SecretBytes(std::size_t n);
  ~SecretBytes();
  SecretBytes(SecretBytes&& o) noexcept;
  SecretBytes& operator=(SecretBytes&& o) noexcept;
  SecretBytes(const SecretBytes&) = delete;
  SecretBytes& operator=(const SecretBytes&) = delete;
  [[nodiscard]] std::uint8_t* data() noexcept { return p_; }
  [[nodiscard]] const std::uint8_t* data() const noexcept { return p_; }
  [[nodiscard]] std::size_t size() const noexcept { return n_; }
  [[nodiscard]] bool ok() const noexcept { return p_ != nullptr; }
  [[nodiscard]] bool locked() const noexcept { return locked_; }
  [[nodiscard]] std::span<const std::uint8_t> span() const noexcept { return {p_, n_}; }
  /// Zeroes and frees now.
  void Clear() noexcept;
  /// Live (allocated, not yet cleared) SecretBytes in this process: tests check keys are really freed.
  [[nodiscard]] static long Live() noexcept;
  /// Test hook: the next `n` allocations fail as if mlock failed.
  static void FailNextForTest(int n) noexcept;

 private:
  std::uint8_t* p_ = nullptr;
  std::size_t n_ = 0, mapped_ = 0;
  bool locked_ = false;
};

/// Constant-time comparison.
[[nodiscard]] bool Equal(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b);
/// Zeroes memory the compiler may not elide.
void Wipe(void* p, std::size_t n) noexcept;
/// Cryptographically secure random bytes (OpenSSL's DRBG).
[[nodiscard]] bool RandomBytes(std::span<std::uint8_t> out);
[[nodiscard]] bool HkdfSha256(std::span<const std::uint8_t> ikm, std::span<const std::uint8_t> salt,
                              std::span<const std::uint8_t> info, std::span<std::uint8_t> out);
[[nodiscard]] bool HmacSha256(std::span<const std::uint8_t> key, std::span<const std::uint8_t> data,
                              std::span<std::uint8_t, 32> out);
[[nodiscard]] bool Sha256(std::span<const std::uint8_t> data, std::span<std::uint8_t, 32> out);

/// The nonce of piece `index` of a file with `salt`: salt || be32(index). Unique within a file because indices are,
/// and across files because every file has its own random DEK (the salt is a second, independent guard).
[[nodiscard]] std::array<std::uint8_t, kNonceBytes> PieceNonce(std::span<const std::uint8_t, kSaltBytes> salt,
                                                               std::uint32_t index);

/// AES-256-GCM on one EVP context (one per thread). In place: `buf` holds the plaintext / ciphertext.
class Gcm {
 public:
  Gcm();
  ~Gcm();
  Gcm(const Gcm&) = delete;
  Gcm& operator=(const Gcm&) = delete;
  [[nodiscard]] bool ok() const noexcept { return ctx_ != nullptr; }
  /// Test seam: true when the context holds no cipher (and so no key schedule): every call resets it.
  [[nodiscard]] bool IdleForTest() const noexcept;
  /// Encrypts `n` bytes of `src` into `dst` (they may be the same buffer).
  [[nodiscard]] bool Seal(std::span<const std::uint8_t> key, std::span<const std::uint8_t, kNonceBytes> nonce,
                          std::span<const std::uint8_t> aad, const std::uint8_t* src, std::uint8_t* dst, std::size_t n,
                          std::span<std::uint8_t, kTagBytes> tag);
  [[nodiscard]] bool Seal(std::span<const std::uint8_t> key, std::span<const std::uint8_t, kNonceBytes> nonce,
                          std::span<const std::uint8_t> aad, std::uint8_t* buf, std::size_t n,
                          std::span<std::uint8_t, kTagBytes> tag) {
    return Seal(key, nonce, aad, buf, buf, n, tag);
  }
  /// False when the tag does not authenticate (wrong key, nonce, AAD, ciphertext or tag) or any EVP step fails:
  /// `buf` is then zeroed (never unauthenticated plaintext). The context is reset after every call (no key
  /// schedule outlives it).
  [[nodiscard]] bool Open(std::span<const std::uint8_t> key, std::span<const std::uint8_t, kNonceBytes> nonce,
                          std::span<const std::uint8_t> aad, std::uint8_t* buf, std::size_t n,
                          std::span<const std::uint8_t, kTagBytes> tag);

 private:
  void* ctx_ = nullptr;  // EVP_CIPHER_CTX
};

/// A key-encryption key and the id stored in the files it seals (the file names its key; a restore under another
/// id is refused before any decryption is tried).
struct Kek {
  SecretBytes key;
  std::string id;  ///< printable, at most kKeyIdBytes
};
constexpr std::size_t kKeyIdBytes = 128;

/// KEK = HKDF-SHA256(ikm, salt = none, info = "halo-spill-v2" || be32(len(session)) || session || be64(epoch)).
/// `id_prefix` names the provider and the ikm (e.g. "tpm:<key name>"); the id is "<prefix>:e<epoch>:<session>".
[[nodiscard]] bool DeriveKek(std::span<const std::uint8_t> ikm, std::string_view session, std::uint64_t epoch,
                             std::string_view id_prefix, Kek* out, std::string* error);
/// A short public name of a key, for ids and display only (never an equality test between keys): hex of the first
/// 8 bytes of HMAC-SHA256(key, "halo-spill-master-id"). False if the HMAC fails.
[[nodiscard]] bool KeyName(std::span<const std::uint8_t> key, std::string* out);
/// Test seam: every key gets the same name (a forced 64-bit name collision: names must never decide key equality).
void KeyNameCollideForTest(bool on) noexcept;

// ---- the v2 header's envelope (the engine owns the rest of the header) ----
constexpr std::size_t kHeaderBytes = 512;  ///< v2 header, fixed size
constexpr std::size_t kAadBytes = 448;     ///< [0, 448): every identity and envelope field: the AAD of everything
constexpr std::size_t kWrapNonceOff = 232, kCheckpointOff = 244, kWrappedDekOff = 448, kWrapTagOff = 480,
                      kMetaTagOff = 496;
constexpr std::size_t kCheckpointBytes = 16;  ///< random per file, in the AAD: which checkpoint this is

/// The AAD of a v2 file (the DEK wrap, the meta, every piece): the header's first kAadBytes.
[[nodiscard]] std::span<const std::uint8_t> HeaderAad(std::span<const std::uint8_t, kHeaderBytes> header);
/// Draws the wrap nonce into the header, then wraps `dek` under `kek` with the header's AAD bytes as AAD. Call after
/// every other AAD field is final. The DEK is encrypted from its SecretBytes straight into the header (never copied
/// there in the clear); on failure the wrapped-key and tag bytes are zeroed. The wrap nonce is random (96 bits)
/// under a KEK that wraps one DEK per file it seals: q wraps under one KEK collide with probability about
/// q^2 / 2^97, negligible for the at most a few thousand files a session key or master epoch seals.
[[nodiscard]] bool WrapDek(const Kek& kek, std::span<std::uint8_t, kHeaderBytes> header,
                           std::span<const std::uint8_t> dek);
/// The DEK, if `kek` and the header's AAD bytes authenticate the wrap.
[[nodiscard]] bool UnwrapDek(const Kek& kek, std::span<const std::uint8_t, kHeaderBytes> header,
                             SecretBytes* dek);

}  // namespace halo::core::crypt

#endif  // HALO_CORE_SPILL_CRYPT_HPP_
