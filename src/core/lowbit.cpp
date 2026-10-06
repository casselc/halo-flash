#include "core/lowbit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "core/quant.hpp"

namespace halo::quant::lowbit {
namespace {

std::uint16_t ToHalf(float v) noexcept {
  const auto h = static_cast<_Float16>(v);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}
float Half(float v) noexcept { return static_cast<float>(static_cast<_Float16>(v)); }

// Index of the codebook entry nearest to v (codebook sorted ascending).
int Nearest(const float* book, int n, float v) noexcept {
  int lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    (v < book[mid] ? hi : lo) = mid;
  }
  return v - book[lo] <= book[hi] - v ? lo : hi;
}

// Least-squares 4-level uniform code (a + s l, l = 0..3, a <= 0) of n values:
// llama.cpp make_qkx2-style search over initial scales, each refit by least
// squares on its assignment.
void Fit2(const float* x, std::size_t n, float& s_out, float& a_out) {
  float mn = x[0], mx = x[0];
  for (std::size_t i = 1; i < n; ++i) {
    mn = std::min(mn, x[i]);
    mx = std::max(mx, x[i]);
  }
  mn = std::min(mn, 0.0f);
  if (mx <= mn) {
    s_out = 0.0f;
    a_out = mn;
    return;
  }
  const auto sse = [&](float s, float a) {
    double e = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const float l = s > 0 ? std::clamp(std::nearbyint((x[i] - a) / s), 0.0f, 3.0f) : 0.0f;
      const double d = x[i] - (a + s * l);
      e += d * d;
    }
    return e;
  };
  float best_s = (mx - mn) / 3.0f, best_a = mn;
  double best = sse(best_s, best_a);
  for (int step = -5; step <= 10; ++step) {
    const float iscale = (3.0f + 0.1f * static_cast<float>(step)) / (mx - mn);
    double sl = 0, sll = 0, sx = 0, sxl = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const float l = std::clamp(std::nearbyint(iscale * (x[i] - mn)), 0.0f, 3.0f);
      sl += l;
      sll += l * l;
      sx += x[i];
      sxl += x[i] * l;
    }
    const double det = static_cast<double>(n) * sll - sl * sl;
    if (det <= 0) continue;
    double s = (n * sxl - sl * sx) / det, a = (sx - s * sl) / n;
    if (a > 0) {
      a = 0;
      s = sxl / sll;
    }
    if (!(s > 0)) continue;
    const double e = sse(static_cast<float>(s), static_cast<float>(a));
    if (e < best) {
      best = e;
      best_s = static_cast<float>(s);
      best_a = static_cast<float>(a);
    }
  }
  s_out = best_s;
  a_out = best_a;
}

// Level index of x for decoded step and offset.
std::uint32_t Level(float x, float step, float offset) noexcept {
  if (!(step > 0)) return 0;
  return static_cast<std::uint32_t>(std::clamp(std::nearbyint((x - offset) / step), 0.0f, 3.0f));
}

// The 6-bit scale/min quantization shared by every container: per group of
// 32, (sc, m) with s ~= sc * D and -a ~= m * Dm, D and Dm the super-block
// maxima over 63.
struct Groups {
  std::vector<float> s, a;
  float smax = 0, amax = 0;
};
Groups FitGroups(const float* x, std::size_t n) {
  Groups g;
  g.s.resize(n / 32);
  g.a.resize(n / 32);
  for (std::size_t j = 0; j < n / 32; ++j) {
    Fit2(x + 32 * j, 32, g.s[j], g.a[j]);
    g.smax = std::max(g.smax, g.s[j]);
    g.amax = std::max(g.amax, -g.a[j]);
  }
  return g;
}
std::uint8_t Six(float v, float unit) noexcept {
  return unit > 0 ? static_cast<std::uint8_t>(std::clamp(std::nearbyint(v / unit), 0.0f, 63.0f)) : 0;
}

// Q4_K (f = 5, 144-byte blocks) or Q5_K (f = 10, 176-byte blocks).
void EncodeK(const float* x, std::size_t cols, bool q5, std::uint8_t* out) {
  const float f = q5 ? 10.0f : 5.0f;
  const std::size_t bytes = q5 ? 176 : 144;
  for (std::size_t b = 0; b < cols / 256; ++b) {
    const float* xb = x + 256 * b;
    std::uint8_t* o = out + bytes * b;
    std::memset(o, 0, bytes);
    const Groups g = FitGroups(xb, 256);
    const float d = Half(g.smax / 63.0f / f), dmin = Half(g.amax / 63.0f);
    std::array<std::uint8_t, 8> sc{}, m{};
    for (int j = 0; j < 8; ++j) {
      sc[j] = Six(g.s[j], d * f);
      m[j] = Six(-g.a[j], dmin);
    }
    const std::uint16_t dh = ToHalf(d), mh = ToHalf(dmin);
    std::memcpy(o, &dh, 2);
    std::memcpy(o + 2, &mh, 2);
    std::uint8_t* scales = o + 4;
    for (int j = 0; j < 4; ++j) {
      scales[j] = static_cast<std::uint8_t>(sc[j] | ((sc[j + 4] >> 4) << 6));
      scales[j + 4] = static_cast<std::uint8_t>(m[j] | ((m[j + 4] >> 4) << 6));
      scales[j + 8] = static_cast<std::uint8_t>((sc[j + 4] & 0xF) | ((m[j + 4] & 0xF) << 4));
    }
    std::uint8_t* qh = q5 ? o + 16 : nullptr;
    std::uint8_t* qs = o + (q5 ? 48 : 16);
    for (std::size_t i = 0; i < 256; ++i) {
      const std::size_t j = i / 32, lane = i % 32;
      // Decoded as the dequantizers compute it: d * sc * q - dmin * m.
      const float step = d * static_cast<float>(sc[j]) * f, off = -dmin * static_cast<float>(m[j]);
      const auto q = static_cast<std::uint32_t>(f) * Level(xb[i], step, off);
      qs[(j / 2) * 32 + lane] |= static_cast<std::uint8_t>((q & 0xF) << (j % 2 ? 4 : 0));
      if (q5 && (q >> 4)) qh[lane] |= static_cast<std::uint8_t>(1u << j);
    }
  }
}

void EncodeQ5_1(const float* x, std::size_t cols, std::uint8_t* out) {
  const Groups g = FitGroups(x, cols);
  const float unit = g.smax / 63.0f, munit = g.amax / 63.0f;
  for (std::size_t j = 0; j < cols / 32; ++j) {
    const float step = static_cast<float>(Six(g.s[j], unit)) * unit;
    const float a = -static_cast<float>(Six(-g.a[j], munit)) * munit;
    const float d = Half(step / 10.0f), m = Half(a);
    std::uint8_t* o = out + 24 * j;
    std::memset(o, 0, 24);
    const std::uint16_t dh = ToHalf(d), mh = ToHalf(m);
    std::memcpy(o, &dh, 2);
    std::memcpy(o + 2, &mh, 2);
    std::uint32_t qh = 0;
    for (std::size_t i = 0; i < 32; ++i) {
      const std::uint32_t q = 10 * Level(x[32 * j + i], d * 10.0f, m);
      o[8 + i % 16] |= static_cast<std::uint8_t>((q & 0xF) << (i < 16 ? 0 : 4));
      qh |= (q >> 4) << i;
    }
    std::memcpy(o + 4, &qh, 4);
  }
}

void EncodeQ8_0Code(const float* x, std::size_t cols, std::uint8_t* out) {
  const Groups g = FitGroups(x, cols);
  const float unit = g.smax / 63.0f, munit = g.amax / 63.0f;
  for (std::size_t j = 0; j < cols / 32; ++j) {
    const float step = static_cast<float>(Six(g.s[j], unit)) * unit;
    const float a = -static_cast<float>(Six(-g.a[j], munit)) * munit;
    // Levels d * (o + jj l): the step is jj quanta, the offset o quanta.
    int jj = 0;
    float d = 0;
    if (step > 0) {
      jj = a < 0 ? static_cast<int>(std::min(40.0f, std::floor(127.0f * step / -a))) : 40;
      jj = std::max(jj, 1);
      d = Half(step / static_cast<float>(jj));
    } else {
      d = Half(-a / 127.0f);
    }
    const int o = d > 0 ? static_cast<int>(std::clamp(std::nearbyint(a / d), -127.0f, static_cast<float>(127 - 3 * jj))) : 0;
    std::uint8_t* blk = out + 34 * j;
    const std::uint16_t dh = ToHalf(d);
    std::memcpy(blk, &dh, 2);
    for (std::size_t i = 0; i < 32; ++i) {
      const std::uint32_t l = Level(x[32 * j + i], d * static_cast<float>(jj), d * static_cast<float>(o));
      blk[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(o + jj * static_cast<int>(l)));
    }
  }
}

}  // namespace

void Fwht(float* x, std::size_t n) {
  for (std::size_t h = 1; h < n; h <<= 1)
    for (std::size_t i = 0; i < n; i += 2 * h)
      for (std::size_t j = i; j < i + h; ++j) {
        const float a = x[j], b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
  const float norm = 1.0f / std::sqrt(static_cast<float>(n));
  for (std::size_t i = 0; i < n; ++i) x[i] *= norm;
}

std::size_t HadamardBlock(std::size_t cols) noexcept {
  for (std::size_t b = 128; b >= 2; b /= 2)
    if (cols % b == 0) return b;
  return 0;
}

void RotationSigns(std::uint64_t seed, std::span<float> signs) {
  std::uint64_t s = seed ^ 0x9E3779B97F4A7C15ull;
  for (auto& v : signs) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    v = (s >> 32) & 1 ? -1.0f : 1.0f;
  }
}

void RotatedQ4Row(const float* w, std::size_t cols, const float* signs, float* out) {
  const std::size_t b = HadamardBlock(cols);
  static const std::array<float, 16> book = [] {
    std::array<float, 16> v{};
    for (int i = 0; i < 16; ++i) v[i] = static_cast<float>(kValuesIq4Nl[i]);
    return v;
  }();
  std::vector<float> r(cols);
  for (std::size_t i = 0; i < cols; ++i) r[i] = w[i] * signs[i];
  for (std::size_t g = 0; g < cols; g += b) Fwht(r.data() + g, b);
  for (std::size_t g = 0; g < cols; g += b) {
    const float* x = r.data() + g;
    float amax = 0;
    for (std::size_t i = 0; i < b; ++i) amax = std::max(amax, std::fabs(x[i]));
    float* y = out + g;
    if (amax == 0) {
      std::fill(y, y + b, 0.0f);
      continue;
    }
    const auto err = [&](float s) {
      double e = 0;
      for (std::size_t i = 0; i < b; ++i) {
        const double d = x[i] - s * book[Nearest(book.data(), 16, x[i] / s)];
        e += d * d;
      }
      return e;
    };
    // Grid over the clipping of the extreme codes, each refit by least
    // squares on its assignment.
    float best_s = amax / 127.0f;
    double best = err(best_s);
    for (int t = 0; t <= 24; ++t) {
      float s = amax / (113.0f + 2.0f * static_cast<float>(t));
      double sxc = 0, scc = 0;
      for (std::size_t i = 0; i < b; ++i) {
        const float c = book[Nearest(book.data(), 16, x[i] / s)];
        sxc += x[i] * c;
        scc += c * c;
      }
      if (scc > 0) s = static_cast<float>(sxc / scc);
      if (!(s > 0)) continue;
      const double e = err(s);
      if (e < best) {
        best = e;
        best_s = s;
      }
    }
    const float s = Half(best_s);
    for (std::size_t i = 0; i < b; ++i) y[i] = s * book[Nearest(book.data(), 16, x[i] / s)];
  }
  for (std::size_t g = 0; g < cols; g += b) Fwht(out + g, b);
  for (std::size_t i = 0; i < cols; ++i) out[i] *= signs[i];
}

void EncodeQ8_0(const float* x, std::size_t cols, std::uint8_t* out) {
  for (std::size_t j = 0; j < cols / 32; ++j) {
    float amax = 0;
    for (std::size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[32 * j + i]));
    const float d = Half(amax / 127.0f);
    std::uint8_t* blk = out + 34 * j;
    const std::uint16_t dh = ToHalf(d);
    std::memcpy(blk, &dh, 2);
    for (std::size_t i = 0; i < 32; ++i) {
      const float q = d > 0 ? std::clamp(std::nearbyint(x[32 * j + i] / d), -127.0f, 127.0f) : 0.0f;
      blk[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
    }
  }
}

bool Encode2Bit(core::GgmlType type, const float* x, std::size_t cols, std::uint8_t* out) {
  switch (type) {
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
      if (cols % 256 != 0) return false;
      EncodeK(x, cols, type == core::GgmlType::kQ5_K, out);
      return true;
    case core::GgmlType::kQ5_1:
      if (cols % 32 != 0) return false;
      EncodeQ5_1(x, cols, out);
      return true;
    case core::GgmlType::kQ8_0:
      if (cols % 32 != 0) return false;
      EncodeQ8_0Code(x, cols, out);
      return true;
    default:
      return false;
  }
}

double Code2BitBpw(core::GgmlType type, std::size_t cols) noexcept {
  const double super = type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ? 256.0
                                                                                       : static_cast<double>(cols);
  return 2.0 + 12.0 / 32.0 + 32.0 / super;
}

}  // namespace halo::quant::lowbit
