#ifndef HALO_TEXT_UNICODE_HPP_
#define HALO_TEXT_UNICODE_HPP_

#include <cstddef>
#include <string>
#include <string_view>

namespace halo::text {

/// One decoded UTF-8 unit. `valid` is false for any byte that does not start a
/// well-formed sequence (overlong forms, surrogates, values above U+10FFFF,
/// truncated or bad continuation bytes); such a unit is exactly one byte long
/// and `value` holds the byte.
struct Utf8Unit {
  char32_t value = 0;
  std::size_t length = 1;
  bool valid = true;
};

[[nodiscard]] Utf8Unit DecodeUtf8(std::string_view text, std::size_t offset) noexcept;
void AppendUtf8(std::string& out, char32_t cp);

// General-category classes and White_Space, Unicode version kUnicodeVersion
// (see unicode_tables.inc). Invalid units (bytes) are in none of them.
[[nodiscard]] bool IsLetter(char32_t cp) noexcept;   // \p{L}
[[nodiscard]] bool IsMark(char32_t cp) noexcept;     // \p{M}
[[nodiscard]] bool IsNumber(char32_t cp) noexcept;   // \p{N}
[[nodiscard]] bool IsWhiteSpace(char32_t cp) noexcept;  // \s (White_Space)
[[nodiscard]] int CombiningClass(char32_t cp) noexcept;
[[nodiscard]] const char* UnicodeVersion() noexcept;

/// Unicode Normalization Form C. Ill-formed bytes are passed through
/// unchanged and act as normalization boundaries, so the function is total
/// over arbitrary bytes and the identity on ASCII.
[[nodiscard]] std::string Nfc(std::string_view text);

}  // namespace halo::text

#endif  // HALO_TEXT_UNICODE_HPP_
