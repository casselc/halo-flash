#include "text/unicode.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace halo::text {

namespace {

struct CodepointRange {
  char32_t first;
  char32_t last;
};
struct CombiningClassEntry {
  char32_t cp;
  std::uint8_t ccc;
};
struct Decomposition {
  char32_t cp;
  std::uint32_t offset;
  std::uint32_t length;
};
struct Composition {
  char32_t first;
  char32_t second;
  char32_t composite;
};

#include "text/unicode_tables.inc"

template <std::size_t N>
bool InRanges(const CodepointRange (&ranges)[N], char32_t cp) noexcept {
  const auto it = std::upper_bound(std::begin(ranges), std::end(ranges), cp,
                                   [](char32_t v, const CodepointRange& r) { return v < r.first; });
  return it != std::begin(ranges) && cp <= std::prev(it)->last;
}

// Hangul syllable arithmetic (Unicode 3.12).
constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28;
constexpr char32_t kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

void DecomposeInto(char32_t cp, std::vector<char32_t>& out) {
  if (cp >= kSBase && cp < kSBase + kSCount) {
    const char32_t s = cp - kSBase;
    out.push_back(kLBase + s / kNCount);
    out.push_back(kVBase + (s % kNCount) / kTCount);
    if (s % kTCount) out.push_back(kTBase + s % kTCount);
    return;
  }
  const auto it = std::lower_bound(std::begin(kDecompositions), std::end(kDecompositions), cp,
                                   [](const Decomposition& d, char32_t v) { return d.cp < v; });
  if (it == std::end(kDecompositions) || it->cp != cp) {
    out.push_back(cp);
    return;
  }
  for (std::uint32_t i = 0; i < it->length; ++i) DecomposeInto(kDecompositionData[it->offset + i], out);
}

char32_t Compose(char32_t a, char32_t b) noexcept {
  if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount) {
    return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
  }
  if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase &&
      b < kTBase + kTCount) {
    return a + (b - kTBase);
  }
  const auto it = std::lower_bound(std::begin(kCompositions), std::end(kCompositions),
                                   std::pair{a, b}, [](const Composition& c, std::pair<char32_t, char32_t> v) {
                                     return c.first != v.first ? c.first < v.first : c.second < v.second;
                                   });
  return it != std::end(kCompositions) && it->first == a && it->second == b ? it->composite : 0;
}

// Normalizes one run of well-formed codepoints in place.
void NfcRun(std::vector<char32_t>& cps) {
  std::vector<char32_t> d;
  d.reserve(cps.size() + 8);
  for (char32_t cp : cps) DecomposeInto(cp, d);
  // Canonical ordering: stable sort each maximal run of non-starters.
  for (std::size_t i = 0; i < d.size();) {
    if (CombiningClass(d[i]) == 0) { ++i; continue; }
    std::size_t j = i;
    while (j < d.size() && CombiningClass(d[j]) != 0) ++j;
    std::stable_sort(d.begin() + i, d.begin() + j,
                     [](char32_t a, char32_t b) { return CombiningClass(a) < CombiningClass(b); });
    i = j;
  }
  // Canonical composition.
  cps.clear();
  std::size_t starter = SIZE_MAX;  // index in cps of the last starter
  int last_ccc = -1;               // ccc of the last character appended after the starter
  for (char32_t cp : d) {
    const int ccc = CombiningClass(cp);
    if (starter != SIZE_MAX) {
      const bool blocked = last_ccc != -1 && (last_ccc == 0 || last_ccc >= ccc);
      if (!blocked) {
        if (const char32_t c = Compose(cps[starter], cp)) {
          cps[starter] = c;
          continue;
        }
      }
    }
    if (ccc == 0) {
      starter = cps.size();
      last_ccc = -1;
    } else {
      last_ccc = ccc;
    }
    cps.push_back(cp);
  }
}

}  // namespace

Utf8Unit DecodeUtf8(std::string_view text, std::size_t offset) noexcept {
  const auto b0 = static_cast<std::uint8_t>(text[offset]);
  if (b0 < 0x80) return {b0, 1, true};
  std::size_t len;
  char32_t cp, min;
  if (b0 >= 0xC2 && b0 <= 0xDF) { len = 2; cp = b0 & 0x1F; min = 0x80; }
  else if ((b0 & 0xF0) == 0xE0) { len = 3; cp = b0 & 0x0F; min = 0x800; }
  else if (b0 >= 0xF0 && b0 <= 0xF4) { len = 4; cp = b0 & 0x07; min = 0x10000; }
  else return {b0, 1, false};
  if (offset + len > text.size()) return {b0, 1, false};
  for (std::size_t i = 1; i < len; ++i) {
    const auto b = static_cast<std::uint8_t>(text[offset + i]);
    if ((b & 0xC0) != 0x80) return {b0, 1, false};
    cp = (cp << 6) | (b & 0x3F);
  }
  if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return {b0, 1, false};
  return {cp, len, true};
}

void AppendUtf8(std::string& out, char32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

bool IsLetter(char32_t cp) noexcept { return InRanges(kLetterRanges, cp); }
bool IsMark(char32_t cp) noexcept { return InRanges(kMarkRanges, cp); }
bool IsNumber(char32_t cp) noexcept { return InRanges(kNumberRanges, cp); }

bool IsWhiteSpace(char32_t cp) noexcept {
  // The White_Space property (PropList.txt); stable across Unicode versions.
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
         cp == 0x205F || cp == 0x3000;
}

int CombiningClass(char32_t cp) noexcept {
  const auto it = std::lower_bound(std::begin(kCombiningClasses), std::end(kCombiningClasses), cp,
                                   [](const CombiningClassEntry& e, char32_t v) { return e.cp < v; });
  return it != std::end(kCombiningClasses) && it->cp == cp ? it->ccc : 0;
}

const char* UnicodeVersion() noexcept { return kUnicodeVersion; }

std::string Nfc(std::string_view text) {
  if (std::all_of(text.begin(), text.end(), [](char c) { return static_cast<unsigned char>(c) < 0x80; })) {
    return std::string(text);
  }
  std::string out;
  out.reserve(text.size());
  std::vector<char32_t> run;
  auto flush = [&] {
    if (run.empty()) return;
    NfcRun(run);
    for (char32_t cp : run) AppendUtf8(out, cp);
    run.clear();
  };
  for (std::size_t i = 0; i < text.size();) {
    const auto u = DecodeUtf8(text, i);
    if (u.valid) {
      run.push_back(u.value);
    } else {
      flush();
      out.push_back(text[i]);
    }
    i += u.length;
  }
  flush();
  return out;
}

}  // namespace halo::text
