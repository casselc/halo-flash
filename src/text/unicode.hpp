#ifndef HALO_TEXT_UNICODE_HPP_
#define HALO_TEXT_UNICODE_HPP_

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

/// True when `cp` starts a segment whose NFC is independent of everything
/// before it: a starter (ccc 0) that is never the second half of a canonical
/// or Hangul composition. NFC(a + b) == NFC(a) + NFC(b) when b begins so.
[[nodiscard]] bool IsStableStarter(char32_t cp) noexcept;

/// Unicode Normalization Form C. Ill-formed bytes are passed through
/// unchanged and act as normalization boundaries, so the function is total
/// over arbitrary bytes and the identity on ASCII.
[[nodiscard]] std::string Nfc(std::string_view text);

/// NFC plus, for every output byte, the [begin, end) byte range of the input
/// code point it derives from (composition keeps the starter's range, as HF
/// tokenizers aligns offsets). Ill-formed bytes map to themselves.
[[nodiscard]] std::string NfcAligned(std::string_view text,
                                     std::vector<std::pair<std::size_t, std::size_t>>* origin);

}  // namespace halo::text

#endif  // HALO_TEXT_UNICODE_HPP_
