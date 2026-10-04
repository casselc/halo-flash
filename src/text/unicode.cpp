#include "text/unicode.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
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

// One normalized code point and its alignment change, as the
// unicode-normalization-alignments crate (HF tokenizers' NFC) emits them:
//   0  replaces the current original character,
//   +1 is inserted (aligned to the previous original character),
//   -N replaces the current character and absorbs the N that follow.
struct Changed {
  char32_t cp;
  int change;
};

// Port of the crate's Decompositions + Recompositions iterators (canonical),
// kept literal so its ordering of changes is reproduced exactly.
std::vector<Changed> NfcWithChanges(const std::vector<char32_t>& input) {
  // Decomposition: each input character's first decomposed code point
  // replaces it (0), the rest are insertions (+1); before every starter the
  // pending tail is stably sorted by combining class.
  struct Item { std::uint8_t ccc; char32_t cp; int change; };
  std::vector<Item> buf;
  std::size_t ready = 0;
  const auto push = [&](char32_t c, bool first) {
    const auto cls = static_cast<std::uint8_t>(CombiningClass(c));
    if (cls == 0) {
      std::stable_sort(buf.begin() + static_cast<std::ptrdiff_t>(ready), buf.end(),
                       [](const Item& a, const Item& b) { return a.ccc < b.ccc; });
      ready = buf.size();
    }
    buf.push_back({cls, c, first ? 0 : 1});
  };
  std::vector<char32_t> tmp;
  for (char32_t c : input) {
    tmp.clear();
    DecomposeInto(c, tmp);
    for (std::size_t k = 0; k < tmp.size(); ++k) push(tmp[k], k == 0);
  }
  std::stable_sort(buf.begin() + static_cast<std::ptrdiff_t>(ready), buf.end(),
                   [](const Item& a, const Item& b) { return a.ccc < b.ccc; });

  // Recomposition (canonical), with the composee absorbing changes.
  std::vector<Changed> out;
  std::vector<Changed> pending;  // the crate's buffer of blocked non-starters
  std::optional<Changed> composee;
  int last_ccc = -1;  // None
  for (const auto& it : buf) {
    const Changed ch{it.cp, it.change};
    const int ch_class = it.ccc;
    if (!composee) {
      if (ch_class != 0) {
        out.push_back(ch);
        continue;
      }
      composee = ch;
      continue;
    }
    if (last_ccc < 0) {
      if (const char32_t r = Compose(composee->cp, ch.cp)) {
        composee = Changed{r, composee->change + ch.change - 1};
        continue;
      }
      if (ch_class == 0) {
        out.push_back(*composee);
        composee = ch;
        continue;
      }
      pending.push_back(ch);
      last_ccc = ch_class;
    } else {
      if (last_ccc >= ch_class) {
        if (ch_class == 0) {
          out.push_back(*composee);
          out.insert(out.end(), pending.begin(), pending.end());
          pending.clear();
          composee = ch;
          last_ccc = -1;
          continue;
        }
        pending.push_back(ch);
        last_ccc = ch_class;
        continue;
      }
      if (const char32_t r = Compose(composee->cp, ch.cp)) {
        composee = Changed{r, composee->change + ch.change - 1};
        continue;
      }
      pending.push_back(ch);
      last_ccc = ch_class;
    }
  }
  if (composee) out.push_back(*composee);
  out.insert(out.end(), pending.begin(), pending.end());
  return out;
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

bool IsWhiteSpace(char32_t cp) noexcept { return InRanges(kSpaceRanges, cp); }

int CombiningClass(char32_t cp) noexcept {
  const auto it = std::lower_bound(std::begin(kCombiningClasses), std::end(kCombiningClasses), cp,
                                   [](const CombiningClassEntry& e, char32_t v) { return e.cp < v; });
  return it != std::end(kCombiningClasses) && it->cp == cp ? it->ccc : 0;
}

const char* UnicodeVersion() noexcept { return kUnicodeVersion; }

bool IsStableStarter(char32_t cp) noexcept {
  if (CombiningClass(cp) != 0) return false;
  if (cp >= kVBase && cp < kVBase + kVCount) return false;          // Hangul V joins L
  if (cp > kTBase && cp < kTBase + kTCount) return false;            // Hangul T joins LV
  static const std::vector<char32_t> seconds = [] {
    std::vector<char32_t> v;
    for (const auto& c : kCompositions) v.push_back(c.second);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
  }();
  return !std::binary_search(seconds.begin(), seconds.end(), cp);
}

std::string Nfc(std::string_view text) { return NfcAligned(text, nullptr); }

std::string NfcAligned(std::string_view text, std::vector<std::pair<std::size_t, std::size_t>>* origin) {
  const bool ascii = std::all_of(text.begin(), text.end(), [](char c) { return static_cast<unsigned char>(c) < 0x80; });
  if (ascii) {
    if (origin) {
      origin->clear();
      for (std::size_t i = 0; i < text.size(); ++i) origin->emplace_back(i, i + 1);
    }
    return std::string(text);
  }
  std::string out;
  out.reserve(text.size());
  if (origin) origin->clear();
  // Runs of well-formed code points are normalized; ill-formed bytes (which
  // HF never sees) pass through as boundaries.
  std::vector<char32_t> run;
  std::vector<std::pair<std::size_t, std::size_t>> run_bytes;  // byte range per run code point
  auto flush = [&] {
    if (run.empty()) return;
    // HF tokenizers' transform: walk the original characters positionally.
    std::size_t cur = 0;
    for (const auto& [cp, change] : NfcWithChanges(run)) {
      std::pair<std::size_t, std::size_t> align;
      if (change > 0) {
        align = cur > 0 ? run_bytes[cur - 1]
                : run_bytes[0].first > 0 && origin && !origin->empty() ? origin->back()
                                                                       : std::pair<std::size_t, std::size_t>{0, 0};
      } else {
        align = run_bytes[std::min(cur, run_bytes.size() - 1)];
        cur += 1 + static_cast<std::size_t>(-change);
      }
      const std::size_t before = out.size();
      AppendUtf8(out, cp);
      if (origin)
        for (std::size_t k = before; k < out.size(); ++k) origin->push_back(align);
    }
    run.clear();
    run_bytes.clear();
  };
  for (std::size_t i = 0; i < text.size();) {
    const auto u = DecodeUtf8(text, i);
    if (u.valid) {
      run.push_back(u.value);
      run_bytes.emplace_back(i, i + u.length);
    } else {
      flush();
      out.push_back(text[i]);
      if (origin) origin->emplace_back(i, i + 1);
    }
    i += u.length;
  }
  flush();
  return out;
}

}  // namespace halo::text
