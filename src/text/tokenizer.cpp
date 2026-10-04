// BPE merge loop adapted from gufo (MIT), src/models/qwen/tokenizer.cpp; the
// pre-tokenizer is written from the regex's backtracking semantics and is
// checked against Python `regex` and HF `tokenizers` (tests/text/).
#include "text/tokenizer.hpp"

#include <algorithm>
#include <limits>
#include <queue>

#include "core/gguf.hpp"
#include "text/unicode.hpp"

namespace halo::text {

namespace {

// Character classes of one decoded unit. Ill-formed bytes are "other".
struct Ch {
  char32_t cp;
  std::size_t len;
  bool valid;
  bool L() const { return valid && IsLetter(cp); }
  bool M() const { return valid && IsMark(cp); }
  bool N() const { return valid && IsNumber(cp); }
  bool S() const { return valid && IsWhiteSpace(cp); }
  bool NL() const { return valid && (cp == '\r' || cp == '\n'); }
  bool LM() const { return L() || M(); }
  bool Other() const { return !S() && !L() && !M() && !N(); }
  bool Other2() const { return !S() && !L() && !N(); }
};

Ch At(std::string_view t, std::size_t i) {
  const auto u = DecodeUtf8(t, i);
  return {u.value, u.length, u.valid};
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d). Case-insensitive matching uses simple case
// folding; among these letters only 's' has a non-ASCII fold partner (U+017F).
std::size_t ContractionEnd(std::string_view t, std::size_t i) {
  if (t[i] != '\'') return i;
  static constexpr std::string_view kSuffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
  for (auto suffix : kSuffixes) {
    std::size_t j = i + 1;
    bool ok = true;
    for (char want : suffix) {
      if (j >= t.size()) { ok = false; break; }
      const Ch c = At(t, j);
      const bool match = c.valid && (c.cp == static_cast<char32_t>(want) ||
                                     c.cp == static_cast<char32_t>(want - 'a' + 'A') ||
                                     (want == 's' && c.cp == 0x017F));
      if (!match) { ok = false; break; }
      j += c.len;
    }
    if (ok) return j;
  }
  return i;
}

std::size_t RunOf(std::string_view t, std::size_t i, bool (Ch::*pred)() const) {
  while (i < t.size()) {
    const Ch c = At(t, i);
    if (!(c.*pred)()) break;
    i += c.len;
  }
  return i;
}

std::size_t PieceEnd(std::string_view t, std::size_t i, bool marks_are_letters) {
  // Qwen3.5/3.8: letter runs are [\p{L}\p{M}]+ and "other" excludes \p{M}.
  // Qwen2-style: letter runs are \p{L}+ and marks count as "other".
  const auto word = marks_are_letters ? &Ch::LM : &Ch::L;
  const auto other = marks_are_letters ? &Ch::Other : &Ch::Other2;
  // 1. Contractions.
  if (const auto e = ContractionEnd(t, i); e != i) return e;
  const Ch c = At(t, i);
  const std::size_t after = i + c.len;
  // 2. [^\r\n\p{L}\p{N}]?<word>+
  if ((c.*word)()) return RunOf(t, after, word);
  if (!c.NL() && !c.L() && !c.N() && after < t.size() && (At(t, after).*word)()) {
    return RunOf(t, after, word);
  }
  // 3. \p{N}
  if (c.N()) return after;
  // 4. ' ?<other>+[\r\n]*'
  std::size_t p = i;
  if (c.valid && c.cp == ' ' && after < t.size() && (At(t, after).*other)()) p = after;
  if ((At(t, p).*other)()) {
    const std::size_t e = RunOf(t, p, other);
    return RunOf(t, e, &Ch::NL);
  }
  // c is whitespace here.
  // 5. \s*[\r\n]+ : ends after the last newline in the whitespace run.
  std::size_t j = i, last_nl_end = i, last_start = i;
  while (j < t.size()) {
    const Ch w = At(t, j);
    if (!w.S()) break;
    last_start = j;
    j += w.len;
    if (w.NL()) last_nl_end = j;
  }
  if (last_nl_end != i) return last_nl_end;
  // 6. \s+(?!\S) : whole run at end of text, else leave its last char for the
  //    next piece (which then begins with that whitespace).
  // 7. \s+ : a single whitespace char before non-whitespace.
  if (j == t.size()) return j;
  return last_start > i ? last_start : j;
}

// GPT-2 byte <-> printable codepoint mapping used by ByteLevel.
const std::array<char32_t, 256>& ByteToCodepoint() {
  static const auto table = [] {
    std::array<char32_t, 256> t{};
    char32_t n = 0;
    for (int b = 0; b < 256; ++b) {
      const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
      t[b] = printable ? static_cast<char32_t>(b) : 256 + n++;
    }
    return t;
  }();
  return table;
}

// Inverts ByteLevel on a GGUF token string. Returns false if the token holds a
// codepoint outside the byte alphabet.
bool UnByteLevel(std::string_view token, std::string& out) {
  static const auto inverse = [] {
    std::unordered_map<char32_t, unsigned char> m;
    const auto& f = ByteToCodepoint();
    for (int b = 0; b < 256; ++b) m[f[b]] = static_cast<unsigned char>(b);
    return m;
  }();
  out.clear();
  for (std::size_t i = 0; i < token.size();) {
    const auto u = DecodeUtf8(token, i);
    const auto it = u.valid ? inverse.find(u.value) : inverse.end();
    if (it == inverse.end()) return false;
    out.push_back(static_cast<char>(it->second));
    i += u.length;
  }
  return true;
}

}  // namespace

std::vector<std::string_view> PreTokenizeQwen35(std::string_view text) {
  return PreTokenize(text, PreTokenizer::kQwen35);
}

std::vector<std::string_view> PreTokenize(std::string_view text, PreTokenizer kind) {
  std::vector<std::string_view> pieces;
  for (std::size_t i = 0; i < text.size();) {
    const std::size_t e = PieceEnd(text, i, kind == PreTokenizer::kQwen35);
    pieces.push_back(text.substr(i, e - i));
    i = e;
  }
  return pieces;
}

std::unique_ptr<Tokenizer> Tokenizer::FromGguf(const core::GgufReader& r, std::string* error) {
  const auto fail = [&](const char* m) -> std::unique_ptr<Tokenizer> {
    if (error) *error = m;
    return nullptr;
  };
  if (r.GetMetadataString("tokenizer.ggml.model") != std::optional<std::string_view>{"gpt2"}) {
    return fail("tokenizer.ggml.model is not gpt2 (byte-level BPE)");
  }
  const auto pre = r.GetMetadataString("tokenizer.ggml.pre");
  if (pre != std::optional<std::string_view>{"qwen35"} && pre != std::optional<std::string_view>{"qwen2"}) {
    return fail("tokenizer.ggml.pre must be qwen35 or qwen2");
  }
  const auto tv = r.GetMetadataStringArray("tokenizer.ggml.tokens");
  const auto mv = r.GetMetadataStringArray("tokenizer.ggml.merges");
  const auto* types_meta = r.FindMetadata("tokenizer.ggml.token_type");
  const auto* type_values = types_meta ? std::get_if<std::vector<std::int64_t>>(&types_meta->value) : nullptr;
  if (tv.empty() || mv.empty() || !type_values || type_values->size() != tv.size()) {
    return fail("tokenizer.ggml.{tokens,merges,token_type} missing or inconsistent");
  }
  std::vector<std::string> tokens(tv.begin(), tv.end()), merges(mv.begin(), mv.end());
  std::vector<TokenType> types;
  types.reserve(type_values->size());
  for (auto v : *type_values) types.push_back(static_cast<TokenType>(v));
  auto tok = FromVocabulary(tokens, merges, types, error);
  if (!tok) return nullptr;
  tok->pre_ = *pre == "qwen2" ? PreTokenizer::kQwen2 : PreTokenizer::kQwen35;
  if (auto v = r.GetMetadataUint32("tokenizer.ggml.eos_token_id"); v && *v < tok->VocabSize()) tok->eos_ = *v;
  if (auto v = r.GetMetadataUint32("tokenizer.ggml.padding_token_id"); v && *v < tok->VocabSize()) tok->pad_ = *v;
  return tok;
}

std::unique_ptr<Tokenizer> Tokenizer::FromVocabulary(std::span<const std::string> tokens,
                                                     std::span<const std::string> merges,
                                                     std::span<const TokenType> types,
                                                     std::string* error) {
  const auto fail = [&](std::string m) -> std::unique_ptr<Tokenizer> {
    if (error) *error = std::move(m);
    return nullptr;
  };
  if (tokens.empty() || types.size() != tokens.size() || tokens.size() >= kNoToken) {
    return fail("empty vocabulary or token_type length mismatch");
  }
  auto t = std::unique_ptr<Tokenizer>(new Tokenizer());
  t->bytes_.resize(tokens.size());
  t->special_.assign(tokens.size(), false);
  t->by_text_.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const auto id = static_cast<TokenId>(i);
    const auto type = types[i];
    t->by_text_.emplace(tokens[i], id);
    if (type == TokenType::kControl || type == TokenType::kUserDefined) {
      t->special_[i] = true;
      t->bytes_[i] = tokens[i];
      t->specials_.emplace_back(tokens[i], id);
    } else if (type == TokenType::kUnused) {
      t->bytes_[i].clear();  // padding rows: never produced, decode to nothing
    } else if (!UnByteLevel(tokens[i], t->bytes_[i])) {
      return fail("token " + std::to_string(i) + " is not byte-level encoded");
    }
  }
  // Longest first, so a special that prefixes another never wins a tie.
  std::sort(t->specials_.begin(), t->specials_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  const auto& b2c = ByteToCodepoint();
  for (int b = 0; b < 256; ++b) {
    std::string s;
    AppendUtf8(s, b2c[b]);
    const auto it = t->by_text_.find(s);
    if (it == t->by_text_.end() || t->special_[it->second]) {
      return fail("vocabulary lacks byte token for byte " + std::to_string(b));
    }
    t->byte_token_[b] = it->second;
  }
  t->merges_.reserve(merges.size());
  for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
    const auto& m = merges[rank];
    const auto sp = m.find(' ');
    if (sp == std::string::npos || m.find(' ', sp + 1) != std::string::npos) {
      return fail("malformed merge at rank " + std::to_string(rank));
    }
    const auto a = t->by_text_.find(m.substr(0, sp));
    const auto b = t->by_text_.find(m.substr(sp + 1));
    const auto ab = t->by_text_.find(m.substr(0, sp) + m.substr(sp + 1));
    if (a == t->by_text_.end() || b == t->by_text_.end() || ab == t->by_text_.end()) {
      return fail("merge references unknown token at rank " + std::to_string(rank));
    }
    t->merges_.try_emplace({a->second, b->second}, Merge{rank, ab->second});
  }
  return t;
}

void Tokenizer::MergePiece(std::string_view piece, std::vector<TokenId>& out,
                           std::vector<std::pair<std::size_t, std::size_t>>* ranges) const {
  const std::size_t count = piece.size();
  std::vector<TokenId> sym(count);
  for (std::size_t i = 0; i < count; ++i) sym[i] = byte_token_[static_cast<unsigned char>(piece[i])];
  if (count == 1) {
    out.push_back(sym[0]);
    if (ranges) ranges->emplace_back(0, 1);
    return;
  }
  // Lowest-rank adjacent pair first, leftmost on ties; linked list + heap with
  // lazy invalidation keeps long pieces (runs of letters or spaces) linear-ish.
  constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> prev(count), next(count);
  for (std::size_t i = 0; i < count; ++i) {
    prev[i] = i == 0 ? kNone : i - 1;
    next[i] = i + 1 < count ? i + 1 : kNone;
  }
  struct Cand {
    std::uint32_t rank;
    std::size_t index;
    TokenId left, right;
  };
  const auto later = [](const Cand& a, const Cand& b) {
    return a.rank != b.rank ? a.rank > b.rank : a.index > b.index;
  };
  std::priority_queue<Cand, std::vector<Cand>, decltype(later)> heap(later);
  const auto offer = [&](std::size_t i, std::size_t j) {
    if (const auto m = merges_.find({sym[i], sym[j]}); m != merges_.end()) {
      heap.push({m->second.rank, i, sym[i], sym[j]});
    }
  };
  for (std::size_t i = 0; i + 1 < count; ++i) offer(i, i + 1);
  while (!heap.empty()) {
    const Cand c = heap.top();
    heap.pop();
    const std::size_t j = next[c.index];
    if (sym[c.index] != c.left || j == kNone || sym[j] != c.right) continue;  // stale
    sym[c.index] = merges_.find({c.left, c.right})->second.result;
    sym[j] = kNoToken;
    next[c.index] = next[j];
    if (next[j] != kNone) prev[next[j]] = c.index;
    if (prev[c.index] != kNone) offer(prev[c.index], c.index);
    if (next[c.index] != kNone) offer(c.index, next[c.index]);
  }
  for (std::size_t i = 0; i != kNone; i = next[i]) {
    out.push_back(sym[i]);
    // A merged symbol at i covers bytes [i, next symbol start).
    if (ranges) ranges->emplace_back(i, next[i] == kNone ? count : next[i]);
  }
}

void Tokenizer::EncodeOrdinaryWithOffsets(std::string_view text, std::size_t base, Encoded& out) const {
  std::vector<std::pair<std::size_t, std::size_t>> origin;  // per normalized byte
  const std::string norm = NfcAligned(text, &origin);
  for (auto& [a, b] : origin) {
    a += base;
    b += base;
  }
  std::size_t piece_start = 0;
  for (const auto piece : PreTokenize(norm, pre_)) {
    piece_start = static_cast<std::size_t>(piece.data() - norm.data());
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    MergePiece(piece, out.ids, &ranges);
    for (const auto& [a, b] : ranges) {
      out.offsets.emplace_back(origin[piece_start + a].first, origin[piece_start + b - 1].second);
    }
  }
}

Tokenizer::Encoded Tokenizer::EncodeWithOffsets(std::string_view text, bool parse_special) const {
  Encoded out;  // offsets in bytes until the final conversion
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t best = std::string_view::npos, best_s = 0;
    if (parse_special) {
      for (std::size_t s = 0; s < specials_.size(); ++s) {
        const auto f = text.find(specials_[s].first, pos);
        if (f < best) {
          best = f;
          best_s = s;
        }
      }
    }
    const std::size_t end = best == std::string_view::npos ? text.size() : best;
    if (end > pos) EncodeOrdinaryWithOffsets(text.substr(pos, end - pos), pos, out);
    if (best == std::string_view::npos) break;
    out.ids.push_back(specials_[best_s].second);
    out.offsets.emplace_back(best, best + specials_[best_s].first.size());
    pos = best + specials_[best_s].first.size();
  }
  // Byte offsets -> code point offsets: [char containing lo, char containing hi-1].
  std::vector<std::size_t> char_of(text.size() + 1, 0);
  std::size_t chars = 0;
  for (std::size_t i = 0; i < text.size();) {
    const auto u = DecodeUtf8(text, i);
    for (std::size_t k = 0; k < u.length; ++k) char_of[i + k] = chars;
    i += u.length;
    ++chars;
  }
  char_of[text.size()] = chars;
  for (auto& [lo, hi] : out.offsets) {
    const std::size_t a = char_of[lo];
    const std::size_t b = hi > lo ? char_of[hi - 1] + 1 : a;
    lo = a;
    hi = b;
  }
  return out;
}

void Tokenizer::EncodeOrdinary(std::string_view text, std::vector<TokenId>& out) const {
  const std::string nfc = Nfc(text);
  for (const auto piece : PreTokenize(nfc, pre_)) MergePiece(piece, out);
}

std::vector<TokenId> Tokenizer::Encode(std::string_view text, bool parse_special) const {
  std::vector<TokenId> out;
  out.reserve(text.size() / 3 + 4);
  if (!parse_special || specials_.empty()) {
    EncodeOrdinary(text, out);
    return out;
  }
  // Leftmost occurrence of any special, longest on a tie. Each special keeps
  // its next occurrence so the text is scanned once per special overall.
  std::vector<std::size_t> next_at(specials_.size());
  for (std::size_t s = 0; s < specials_.size(); ++s) next_at[s] = text.find(specials_[s].first);
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t best = std::string_view::npos, best_s = 0;
    for (std::size_t s = 0; s < specials_.size(); ++s) {
      if (next_at[s] != std::string_view::npos && next_at[s] < pos) {
        next_at[s] = text.find(specials_[s].first, pos);
      }
      if (next_at[s] < best) {  // specials_ is longest-first: strict < keeps the longest
        best = next_at[s];
        best_s = s;
      }
    }
    if (best == std::string_view::npos) {
      EncodeOrdinary(text.substr(pos), out);
      break;
    }
    if (best > pos) EncodeOrdinary(text.substr(pos, best - pos), out);
    out.push_back(specials_[best_s].second);
    pos = best + specials_[best_s].first.size();
  }
  return out;
}

std::string Tokenizer::Decode(std::span<const TokenId> tokens) const {
  std::string out;
  for (const TokenId id : tokens) out.append(TokenBytes(id));
  return out;
}

std::string_view Tokenizer::TokenBytes(TokenId id) const noexcept {
  return id < bytes_.size() ? std::string_view(bytes_[id]) : std::string_view();
}

std::optional<TokenId> Tokenizer::Find(std::string_view token) const {
  const auto it = by_text_.find(std::string(token));
  return it == by_text_.end() ? std::nullopt : std::optional<TokenId>(it->second);
}

}  // namespace halo::text
