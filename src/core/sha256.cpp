#include "core/sha256.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__x86_64__)
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace halo::core {

namespace {

constexpr std::uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline std::uint32_t Rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void BlocksPortable(std::uint32_t* h, const std::uint8_t* p, std::size_t n) {
  for (; n > 0; --n, p += 64) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (std::uint32_t{p[4 * i]} << 24) | (std::uint32_t{p[4 * i + 1]} << 16) |
             (std::uint32_t{p[4 * i + 2]} << 8) | std::uint32_t{p[4 * i + 3]};
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t S1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t t1 = hh + S1 + ch + kK[i] + w[i];
      const std::uint32_t S0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = S0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }
}

#if defined(__x86_64__)
__attribute__((target("sha,sse4.1,ssse3"))) void BlocksShaNi(std::uint32_t* state, const std::uint8_t* data,
                                                              std::size_t n) {
  const __m128i mask = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
  __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[0]));
  __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[4]));
  tmp = _mm_shuffle_epi32(tmp, 0xB1);        // CDAB
  s1 = _mm_shuffle_epi32(s1, 0x1B);          // EFGH
  __m128i s0 = _mm_alignr_epi8(tmp, s1, 8);  // ABEF
  s1 = _mm_blend_epi16(s1, tmp, 0xF0);       // CDGH
  for (; n > 0; --n, data += 64) {
    const __m128i abef = s0, cdgh = s1;
    __m128i w[4];
    for (int g = 0; g < 16; ++g) {
      __m128i m;
      if (g < 4) {
        m = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16 * g)), mask);
      } else {
        // W[g] = msg2(msg1(W[g-4], W[g-3]) + alignr(W[g-1], W[g-2], 4), W[g-1])
        const __m128i x = _mm_add_epi32(_mm_sha256msg1_epu32(w[g & 3], w[(g + 1) & 3]),
                                        _mm_alignr_epi8(w[(g + 3) & 3], w[(g + 2) & 3], 4));
        m = _mm_sha256msg2_epu32(x, w[(g + 3) & 3]);
      }
      w[g & 3] = m;
      __m128i k = _mm_add_epi32(m, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kK[4 * g])));
      s1 = _mm_sha256rnds2_epu32(s1, s0, k);
      k = _mm_shuffle_epi32(k, 0x0E);
      s0 = _mm_sha256rnds2_epu32(s0, s1, k);
    }
    s0 = _mm_add_epi32(s0, abef);
    s1 = _mm_add_epi32(s1, cdgh);
  }
  tmp = _mm_shuffle_epi32(s0, 0x1B);    // FEBA
  s1 = _mm_shuffle_epi32(s1, 0xB1);     // DCHG
  s0 = _mm_blend_epi16(tmp, s1, 0xF0);  // DCBA
  s1 = _mm_alignr_epi8(s1, tmp, 8);     // HGFE
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[0]), s0);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[4]), s1);
}

bool CpuHasShaNi() {
  unsigned a = 0, b = 0, c = 0, d = 0;
  if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
  const bool sha = (b >> 29) & 1;
  if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
  const bool sse41 = (c >> 19) & 1, ssse3 = (c >> 9) & 1;
  return sha && sse41 && ssse3;
}
#endif

std::atomic<bool> g_force_portable{false};

bool UseShaNi() {
#if defined(__x86_64__)
  static const bool has = CpuHasShaNi();
  return has && !g_force_portable.load(std::memory_order_relaxed);
#else
  return false;
#endif
}

}  // namespace

void Sha256::ForcePortable(bool on) noexcept { g_force_portable.store(on, std::memory_order_relaxed); }
bool Sha256::HardwareAccelerated() noexcept { return UseShaNi(); }

void Sha256::Reset() {
  static constexpr std::uint32_t kInit[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::memcpy(h_, kInit, sizeof h_);
  buf_len_ = 0;
  total_ = 0;
}

void Sha256::Blocks(const std::uint8_t* p, std::size_t nblocks) {
#if defined(__x86_64__)
  if (UseShaNi()) {
    BlocksShaNi(h_, p, nblocks);
    return;
  }
#endif
  BlocksPortable(h_, p, nblocks);
}

void Sha256::Update(const void* data, std::size_t size) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  total_ += size;
  if (buf_len_ > 0) {
    const std::size_t take = std::min<std::size_t>(64 - buf_len_, size);
    std::memcpy(buf_ + buf_len_, p, take);
    buf_len_ += take;
    p += take;
    size -= take;
    if (buf_len_ < 64) return;
    Blocks(buf_, 1);
    buf_len_ = 0;
  }
  if (size >= 64) {
    Blocks(p, size / 64);
    p += size / 64 * 64;
    size %= 64;
  }
  std::memcpy(buf_, p, size);
  buf_len_ = size;
}

Sha256::Digest Sha256::Finish() {
  const std::uint64_t bits = total_ * 8;
  std::uint8_t pad[72] = {0x80};
  const std::size_t padlen = (buf_len_ < 56 ? 56 : 120) - buf_len_;
  std::uint8_t len[8];
  for (int i = 0; i < 8; ++i) len[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
  Update(pad, padlen);
  Update(len, 8);
  Digest d;
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j) d[4 * i + j] = static_cast<std::uint8_t>(h_[i] >> (24 - 8 * j));
  Reset();
  return d;
}

std::string HexDigest(const Sha256::Digest& d) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string s;
  s.reserve(64);
  for (auto b : d) {
    s.push_back(kHex[b >> 4]);
    s.push_back(kHex[b & 15]);
  }
  return s;
}

std::string Sha256Hex(std::string_view data) {
  Sha256 h;
  h.Update(data);
  return HexDigest(h.Finish());
}

std::optional<std::string> Sha256File(const std::string& path, std::string* error) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    if (error) *error = "cannot open " + path;
    return std::nullopt;
  }
  Sha256 h;
  std::vector<std::uint8_t> buf(8u << 20);
  for (;;) {
    const std::size_t n = std::fread(buf.data(), 1, buf.size(), f);
    if (n > 0) h.Update(buf.data(), n);
    if (n < buf.size()) break;
  }
  const bool bad = std::ferror(f) != 0;
  std::fclose(f);
  if (bad) {
    if (error) *error = "read error on " + path;
    return std::nullopt;
  }
  return HexDigest(h.Finish());
}

}  // namespace halo::core
