#ifndef HALO_SCOPE_ROWBINARY_HPP_
#define HALO_SCOPE_ROWBINARY_HPP_

// ClickHouse RowBinary primitives: little-endian fixed-width numbers, LEB128
// lengths, length-prefixed strings, and arrays/maps as a length then items.

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <string_view>

namespace halo::scope::rb {

inline void Varint(std::string& out, std::uint64_t v) {
  while (v >= 0x80) {
    out.push_back(static_cast<char>((v & 0x7f) | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<char>(v));
}

template <typename T>
inline void Fixed(std::string& out, T v) {
  static_assert(std::is_trivially_copyable_v<T>);
  char b[sizeof(T)];
  std::memcpy(b, &v, sizeof(T));  // x86-64 and aarch64 are little-endian
  out.append(b, sizeof(T));
}

inline void Str(std::string& out, std::string_view s) {
  Varint(out, s.size());
  out.append(s.data(), s.size());
}

using Hash16 = std::array<std::uint8_t, 16>;

inline void FixedStr16(std::string& out, const Hash16& h) {
  out.append(reinterpret_cast<const char*>(h.data()), h.size());
}

/// Bounds-checked reader over a RowBinary buffer.
class Reader {
 public:
  explicit Reader(std::string_view b) : b_(b) {}
  bool done() const { return p_ >= b_.size(); }
  std::size_t pos() const { return p_; }

  std::uint64_t Varint() {
    std::uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      const auto c = static_cast<std::uint8_t>(Take(1)[0]);
      v |= static_cast<std::uint64_t>(c & 0x7f) << shift;
      if (!(c & 0x80)) return v;
    }
    throw std::runtime_error("rowbinary: varint too long");
  }
  template <typename T>
  T Fixed() {
    T v;
    std::memcpy(&v, Take(sizeof(T)).data(), sizeof(T));
    return v;
  }
  std::string Str() {
    const auto n = Varint();
    return std::string(Take(n));
  }
  Hash16 FixedStr16() {
    Hash16 h;
    std::memcpy(h.data(), Take(16).data(), 16);
    return h;
  }

 private:
  std::string_view Take(std::uint64_t n) {
    if (n > b_.size() - p_) throw std::runtime_error("rowbinary: truncated");
    auto s = b_.substr(p_, n);
    p_ += n;
    return s;
  }
  std::string_view b_;
  std::size_t p_ = 0;
};

}  // namespace halo::scope::rb

#endif  // HALO_SCOPE_ROWBINARY_HPP_
