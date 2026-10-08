#include "core/spill_crypt.hpp"

#include <openssl/core_names.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/kdf.h>
#include <openssl/params.h>
#include <openssl/rand.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <utility>
#include <vector>

namespace halo::core::crypt {

// ---- SecretBytes ----
namespace {
std::atomic<long> g_live{0};
std::atomic<int> g_fail_next{0};
}  // namespace

long SecretBytes::Live() noexcept { return g_live.load(); }
void SecretBytes::FailNextForTest(int n) noexcept { g_fail_next.store(n); }

SecretBytes::SecretBytes(std::size_t n) {
  if (n == 0) return;
  const std::size_t page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
  const std::size_t m = (n + page - 1) / page * page;
  void* p = ::mmap(nullptr, m, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return;
  // Fail closed: a key that could be swapped, dumped or inherited is not handed out.
  int injected = g_fail_next.load();
  while (injected > 0 && !g_fail_next.compare_exchange_weak(injected, injected - 1)) {
  }
  const bool ok = injected <= 0 && ::madvise(p, m, MADV_DONTDUMP) == 0 && ::madvise(p, m, MADV_WIPEONFORK) == 0 &&
                  ::mlock(p, m) == 0;
  if (!ok) {
    (void)::munlock(p, m);
    (void)::munmap(p, m);
    return;
  }
  locked_ = true;
  p_ = static_cast<std::uint8_t*>(p);
  n_ = n;
  mapped_ = m;
  ++g_live;
}

void SecretBytes::Clear() noexcept {
  if (!p_) return;
  OPENSSL_cleanse(p_, mapped_);
  (void)::munlock(p_, mapped_);
  (void)::munmap(p_, mapped_);
  p_ = nullptr;
  n_ = mapped_ = 0;
  locked_ = false;
  --g_live;
}

SecretBytes::~SecretBytes() { Clear(); }

SecretBytes::SecretBytes(SecretBytes&& o) noexcept
    : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)), mapped_(std::exchange(o.mapped_, 0)),
      locked_(std::exchange(o.locked_, false)) {}

SecretBytes& SecretBytes::operator=(SecretBytes&& o) noexcept {
  if (this != &o) {
    Clear();
    p_ = std::exchange(o.p_, nullptr);
    n_ = std::exchange(o.n_, 0);
    mapped_ = std::exchange(o.mapped_, 0);
    locked_ = std::exchange(o.locked_, false);
  }
  return *this;
}

// ---- primitives ----
bool Equal(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
  return a.size() == b.size() && CRYPTO_memcmp(a.data(), b.data(), a.size()) == 0;
}

void Wipe(void* p, std::size_t n) noexcept { OPENSSL_cleanse(p, n); }

bool RandomBytes(std::span<std::uint8_t> out) {
  return out.empty() || RAND_priv_bytes(out.data(), static_cast<int>(out.size())) == 1;
}

bool HkdfSha256(std::span<const std::uint8_t> ikm, std::span<const std::uint8_t> salt,
                std::span<const std::uint8_t> info, std::span<std::uint8_t> out) {
  EVP_KDF* kdf = EVP_KDF_fetch(nullptr, "HKDF", nullptr);
  if (!kdf) return false;
  EVP_KDF_CTX* ctx = EVP_KDF_CTX_new(kdf);
  EVP_KDF_free(kdf);
  if (!ctx) return false;
  char digest[] = "SHA256";
  OSSL_PARAM params[5];
  int i = 0;
  params[i++] = OSSL_PARAM_construct_utf8_string(OSSL_KDF_PARAM_DIGEST, digest, 0);
  params[i++] = OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_KEY, const_cast<std::uint8_t*>(ikm.data()), ikm.size());
  if (!salt.empty())
    params[i++] = OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_SALT, const_cast<std::uint8_t*>(salt.data()), salt.size());
  params[i++] = OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_INFO, const_cast<std::uint8_t*>(info.data()), info.size());
  params[i] = OSSL_PARAM_construct_end();
  const bool ok = EVP_KDF_derive(ctx, out.data(), out.size(), params) == 1;
  EVP_KDF_CTX_free(ctx);
  return ok;
}

bool HmacSha256(std::span<const std::uint8_t> key, std::span<const std::uint8_t> data, std::span<std::uint8_t, 32> out) {
  EVP_MAC* mac = EVP_MAC_fetch(nullptr, "HMAC", nullptr);
  if (!mac) return false;
  EVP_MAC_CTX* ctx = EVP_MAC_CTX_new(mac);
  EVP_MAC_free(mac);
  if (!ctx) return false;
  char digest[] = "SHA256";
  OSSL_PARAM params[] = {OSSL_PARAM_construct_utf8_string(OSSL_MAC_PARAM_DIGEST, digest, 0), OSSL_PARAM_construct_end()};
  std::size_t len = 0;
  const bool ok = EVP_MAC_init(ctx, key.data(), key.size(), params) == 1 &&
                  EVP_MAC_update(ctx, data.data(), data.size()) == 1 && EVP_MAC_final(ctx, out.data(), &len, 32) == 1 &&
                  len == 32;
  EVP_MAC_CTX_free(ctx);
  return ok;
}

bool Sha256(std::span<const std::uint8_t> data, std::span<std::uint8_t, 32> out) {
  unsigned n = 0;
  if (EVP_Digest(data.data(), data.size(), out.data(), &n, EVP_sha256(), nullptr) == 1 && n == 32) return true;
  OPENSSL_cleanse(out.data(), 32);
  return false;
}

std::array<std::uint8_t, kNonceBytes> PieceNonce(std::span<const std::uint8_t, kSaltBytes> salt, std::uint32_t index) {
  std::array<std::uint8_t, kNonceBytes> n{};
  std::memcpy(n.data(), salt.data(), kSaltBytes);
  const std::uint32_t be = __builtin_bswap32(index);  // big-endian index
  std::memcpy(n.data() + kSaltBytes, &be, 4);
  return n;
}

std::span<const std::uint8_t> HeaderAad(std::span<const std::uint8_t, kHeaderBytes> header) {
  return header.first(kAadBytes);
}

// ---- AES-256-GCM ----
Gcm::Gcm() : ctx_(EVP_CIPHER_CTX_new()) {}
Gcm::~Gcm() { EVP_CIPHER_CTX_free(static_cast<EVP_CIPHER_CTX*>(ctx_)); }

namespace {
// EVP takes int lengths: feed in pieces below 2^30.
constexpr std::size_t kChunk = std::size_t{1} << 30;
}  // namespace

bool Gcm::Seal(std::span<const std::uint8_t> key, std::span<const std::uint8_t, kNonceBytes> nonce,
               std::span<const std::uint8_t> aad, const std::uint8_t* src, std::uint8_t* dst, std::size_t n,
               std::span<std::uint8_t, kTagBytes> tag) {
  auto* c = static_cast<EVP_CIPHER_CTX*>(ctx_);
  if (!c || key.size() != kKeyBytes) return false;
  int len = 0;
  bool ok = EVP_EncryptInit_ex(c, EVP_aes_256_gcm(), nullptr, key.data(), nonce.data()) == 1 &&
            (aad.empty() || EVP_EncryptUpdate(c, nullptr, &len, aad.data(), static_cast<int>(aad.size())) == 1);
  for (std::size_t at = 0; ok && at < n; at += kChunk) {
    const int m = static_cast<int>(std::min(kChunk, n - at));
    ok = EVP_EncryptUpdate(c, dst + at, &len, src + at, m) == 1 && len == m;
  }
  ok = ok && EVP_EncryptFinal_ex(c, nullptr, &len) == 1 &&
       EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_GCM_GET_TAG, static_cast<int>(kTagBytes), tag.data()) == 1;
  (void)EVP_CIPHER_CTX_reset(c);  // no key schedule outlives the call
  if (!ok) {
    if (n) OPENSSL_cleanse(dst, n);  // never half-encrypted output
    OPENSSL_cleanse(tag.data(), kTagBytes);
  }
  return ok;
}

bool Gcm::Open(std::span<const std::uint8_t> key, std::span<const std::uint8_t, kNonceBytes> nonce,
               std::span<const std::uint8_t> aad, std::uint8_t* buf, std::size_t n,
               std::span<const std::uint8_t, kTagBytes> tag) {
  auto* c = static_cast<EVP_CIPHER_CTX*>(ctx_);
  if (!c || key.size() != kKeyBytes) {
    if (n) OPENSSL_cleanse(buf, n);
    return false;
  }
  int len = 0;
  bool ok = EVP_DecryptInit_ex(c, EVP_aes_256_gcm(), nullptr, key.data(), nonce.data()) == 1 &&
            (aad.empty() || EVP_DecryptUpdate(c, nullptr, &len, aad.data(), static_cast<int>(aad.size())) == 1);
  for (std::size_t at = 0; ok && at < n; at += kChunk) {
    const int m = static_cast<int>(std::min(kChunk, n - at));
    ok = EVP_DecryptUpdate(c, buf + at, &len, buf + at, m) == 1 && len == m;
  }
  ok = ok &&
       EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_GCM_SET_TAG, static_cast<int>(kTagBytes), const_cast<std::uint8_t*>(tag.data())) == 1;
  ok = ok && EVP_DecryptFinal_ex(c, nullptr, &len) == 1;
  (void)EVP_CIPHER_CTX_reset(c);
  if (!ok && n) OPENSSL_cleanse(buf, n);  // unauthenticated plaintext never survives a failed open
  return ok;
}

// ---- KEK ----
bool DeriveKek(std::span<const std::uint8_t> ikm, std::string_view session, std::uint64_t epoch,
               std::string_view id_prefix, Kek* out, std::string* error) {
  const auto fail = [&](const char* why) {
    if (error) *error = why;
    return false;
  };
  if (ikm.size() < kKeyBytes) return fail("spill key: key material shorter than 32 bytes");
  if (session.size() > 256) return fail("spill key: session label longer than 256 bytes");
  std::vector<std::uint8_t> info;
  const std::string_view label = "halo-spill-v2";
  info.insert(info.end(), label.begin(), label.end());
  const auto len = static_cast<std::uint32_t>(session.size());
  for (int s = 24; s >= 0; s -= 8) info.push_back(static_cast<std::uint8_t>(len >> s));
  info.insert(info.end(), session.begin(), session.end());
  for (int s = 56; s >= 0; s -= 8) info.push_back(static_cast<std::uint8_t>(epoch >> s));
  SecretBytes key(kKeyBytes);
  if (!key.ok()) return fail("spill key: cannot allocate key memory");
  if (!HkdfSha256(ikm, {}, info, {key.data(), kKeyBytes})) return fail("spill key: HKDF failed");
  std::string id = std::string(id_prefix) + ":e" + std::to_string(epoch) + ":" + std::string(session);
  if (id.size() > kKeyIdBytes) return fail("spill key: key id longer than 128 bytes");
  for (char ch : id)
    if (ch < 0x21 || ch > 0x7e) return fail("spill key: key id must be printable ASCII without spaces");
  out->key = std::move(key);
  out->id = std::move(id);
  return true;
}

bool KeyName(std::span<const std::uint8_t> key, std::string* out) {
  std::array<std::uint8_t, 32> h{};
  const std::string_view label = "halo-spill-master-id";
  if (!HmacSha256(key, {reinterpret_cast<const std::uint8_t*>(label.data()), label.size()}, h)) return false;
  static const char* hex = "0123456789abcdef";
  std::string s;
  for (int i = 0; i < 8; ++i) {
    s += hex[h[i] >> 4];
    s += hex[h[i] & 15];
  }
  *out = std::move(s);
  return true;
}

// ---- DEK wrap ----
bool WrapDek(const Kek& kek, std::span<std::uint8_t, kHeaderBytes> header, std::span<const std::uint8_t> dek) {
  if (!kek.key.ok() || dek.size() != kKeyBytes) return false;
  std::span<std::uint8_t, kNonceBytes> nonce(header.data() + kWrapNonceOff, kNonceBytes);
  if (!RandomBytes(nonce)) return false;
  std::uint8_t* wrapped = header.data() + kWrappedDekOff;
  Gcm g;
  // Not in place: the DEK goes from its SecretBytes through the cipher; only ciphertext reaches the header.
  if (g.Seal(kek.key.span(), nonce, HeaderAad(header), dek.data(), wrapped, kKeyBytes,
             std::span<std::uint8_t, kTagBytes>(header.data() + kWrapTagOff, kTagBytes)))
    return true;
  OPENSSL_cleanse(wrapped, kKeyBytes + kTagBytes);
  return false;
}

bool UnwrapDek(const Kek& kek, std::span<const std::uint8_t, kHeaderBytes> header, SecretBytes* dek) {
  if (!kek.key.ok()) return false;
  SecretBytes d(kKeyBytes);
  if (!d.ok()) return false;
  std::memcpy(d.data(), header.data() + kWrappedDekOff, kKeyBytes);
  Gcm g;
  if (!g.Open(kek.key.span(), std::span<const std::uint8_t, kNonceBytes>(header.data() + kWrapNonceOff, kNonceBytes),
              HeaderAad(header), d.data(), kKeyBytes,
              std::span<const std::uint8_t, kTagBytes>(header.data() + kWrapTagOff, kTagBytes)))
    return false;
  *dek = std::move(d);
  return true;
}

}  // namespace halo::core::crypt
