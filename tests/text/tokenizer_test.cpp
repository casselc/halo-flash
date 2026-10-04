// Tokenizer tests.
//
// Differential (fixtures from tools/gen_tokenizer_fixtures.py, references
// independent of this code):
//   NFC vs Python unicodedata; pre-tokenizer vs Python `regex` on the exact
//   tokenizer.json pattern; Encode vs HF `tokenizers` (needs the real
//   vocabulary: HALO_FLASH_NEXT_GGUF=<shard 00001 of Qwen3.8-Flash-Next>).
// Properties (Hegel, run under the sanitizer presets):
//   pre-tokenizer pieces are non-empty and concatenate to the input; NFC is
//   idempotent, total on bytes and the identity on ASCII; Decode(Encode(x)) ==
//   Nfc(x) for arbitrary bytes with embedded special tokens, on a synthetic
//   byte-level vocabulary and (when available) the real one.

#include <hegel/gtest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/gguf_writer.hpp"
#include "text/tokenizer.hpp"
#include "text/unicode.hpp"

namespace gs = hegel::generators;
using namespace halo::text;

namespace {

std::string Unhex(std::string_view h) {
  std::string out;
  out.reserve(h.size() / 2);
  for (std::size_t i = 0; i + 1 < h.size(); i += 2) {
    out.push_back(static_cast<char>(std::stoi(std::string(h.substr(i, 2)), nullptr, 16)));
  }
  return out;
}

std::string Hex(std::string_view s) {
  static const char* d = "0123456789abcdef";
  std::string out;
  for (unsigned char c : s) {
    out.push_back(d[c >> 4]);
    out.push_back(d[c & 15]);
  }
  return out;
}

std::vector<std::pair<std::string, std::string>> LoadTsv(const char* name) {
  std::ifstream f(std::string(HALO_TEST_DATA_DIR) + "/" + name);
  EXPECT_TRUE(f.good()) << name;
  std::vector<std::pair<std::string, std::string>> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.starts_with('#')) continue;
    const auto tab = line.find('\t');
    rows.emplace_back(line.substr(0, tab), tab == std::string::npos ? "" : line.substr(tab + 1));
  }
  return rows;
}

std::vector<std::string> Split(const std::string& s, char sep) {
  std::vector<std::string> out;
  if (s.empty()) return out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) out.push_back(item);
  return out;
}

// Byte-level vocabulary: the 256 GPT-2 byte tokens, a few merges, and two
// specials. Small enough to build per test case.
std::unique_ptr<Tokenizer> SyntheticTokenizer() {
  std::vector<std::string> tokens;
  std::vector<TokenType> types;
  char32_t n = 0;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    std::string s;
    AppendUtf8(s, printable ? static_cast<char32_t>(b) : 256 + n++);
    tokens.push_back(s);
    types.push_back(TokenType::kNormal);
  }
  const std::vector<std::string> merges = {"Ġ t", "h e", "Ġt he", "i n", "Ġ a", "e r", "a n"};
  for (const auto& m : merges) {
    const auto sp = m.find(' ');
    tokens.push_back(m.substr(0, sp) + m.substr(sp + 1));
    types.push_back(TokenType::kNormal);
  }
  for (const char* sp : {"<|im_start|>", "<|im_end|>", "<|im"}) {
    tokens.push_back(sp);
    types.push_back(TokenType::kControl);
  }
  std::string error;
  auto t = Tokenizer::FromVocabulary(tokens, merges, types, &error);
  EXPECT_TRUE(t) << error;
  return t;
}

std::unique_ptr<Tokenizer> RealTokenizer() {
  const char* path = std::getenv("HALO_FLASH_NEXT_GGUF");
  if (!path) return nullptr;
  std::string error;
  auto reader = halo::core::GgufReader::OpenShard(path, &error);
  EXPECT_TRUE(reader) << error;
  if (!reader) return nullptr;
  auto t = Tokenizer::FromGguf(*reader, &error);
  EXPECT_TRUE(t) << error;
  return t;
}

// Arbitrary bytes, biased towards text that exercises every class.
std::string DrawText(hegel::TestCase& tc) {
  if (tc.draw(gs::booleans())) {
    auto b = tc.draw(gs::binary({.max_size = 64}));
    return {b.begin(), b.end()};
  }
  const auto parts = tc.draw(gs::vectors(
      gs::sampled_from<std::string>({"a", "Z", "'s", "'LL", " ", "  ", "\n", "\r\n", "\t", "1", "١",
                                     "é", "e\xcc\x81", "\xe1\x84\x80\xe1\x85\xa1", "中", "🦙", ".",
                                     "!?", "<|im_start|>", "<|im_end|>", "<|im", "\xff", "\xc3",
                                     "\xa0", " the", "in", "\xc2\xa0", "\xe2\x80\xa8"}),
      {.max_size = 24}));
  std::string s;
  for (const auto& p : parts) s += p;
  return s;
}

}  // namespace

TEST(Unicode, NfcMatchesHfTokenizers) {
  const auto rows = LoadTsv("nfc.tsv");
  ASSERT_GT(rows.size(), 10000U);
  int failures = 0;
  for (const auto& [in, want] : rows) {
    const auto got = Hex(Nfc(Unhex(in)));
    if (got != want && ++failures <= 10) ADD_FAILURE() << "input " << in << " want " << want << " got " << got;
  }
  EXPECT_EQ(failures, 0) << "of " << rows.size() << " (Unicode " << UnicodeVersion() << ")";
}

TEST(Tokenizer, PreTokenizerMatchesRegex) {
  const auto rows = LoadTsv("pretok.tsv");
  ASSERT_GT(rows.size(), 3000U);
  int failures = 0;
  for (const auto& [in, want_pieces] : rows) {
    const std::string text = Unhex(in);
    std::vector<std::string> got;
    for (auto p : PreTokenizeQwen35(text)) got.push_back(Hex(p));
    if (got != Split(want_pieces, ',') && ++failures <= 10) {
      std::string joined;
      for (auto& g : got) joined += g + ",";
      ADD_FAILURE() << "input " << in << "\n want " << want_pieces << "\n got  " << joined;
    }
  }
  EXPECT_EQ(failures, 0) << "of " << rows.size();
}

TEST(Tokenizer, EncodeMatchesHfTokenizers) {
  auto tok = RealTokenizer();
  if (!tok) GTEST_SKIP() << "set HALO_FLASH_NEXT_GGUF to the Flash-Next shard 00001 GGUF";
  const auto rows = LoadTsv("encode.tsv");
  int failures = 0;
  for (const auto& [in, want] : rows) {
    const auto ids = tok->Encode(Unhex(in));
    std::string got;
    for (std::size_t i = 0; i < ids.size(); ++i) got += (i ? " " : "") + std::to_string(ids[i]);
    if (got != want && ++failures <= 10) ADD_FAILURE() << "input " << in << "\n want " << want << "\n got  " << got;
  }
  EXPECT_EQ(failures, 0) << "of " << rows.size();
}

TEST(Unicode, NfcProperties) {
  hegel::test([](hegel::TestCase& tc) {
    const auto s = DrawText(tc);
    const auto n = Nfc(s);
    ASSERT_EQ(Nfc(n), n);  // idempotent
    bool ascii = true;
    for (unsigned char c : s) ascii &= c < 0x80;
    if (ascii) ASSERT_EQ(n, s);
  }, {.test_cases = 500});
}

TEST(Tokenizer, PiecesAreNonEmptyAndConcatenate) {
  hegel::test([](hegel::TestCase& tc) {
    const auto s = DrawText(tc);
    std::string joined;
    for (auto p : PreTokenizeQwen35(s)) {
      ASSERT_FALSE(p.empty());
      joined += p;
    }
    ASSERT_EQ(joined, s);
  }, {.test_cases = 1000});
}

TEST(Tokenizer, RoundTripOnSyntheticVocabulary) {
  auto tok = SyntheticTokenizer();
  ASSERT_TRUE(tok);
  hegel::test([&](hegel::TestCase& tc) {
    const auto s = DrawText(tc);
    const auto ids = tok->Encode(s);
    // Specials are matched in raw text and are not normalized; everything
    // between them is NFC-normalized.
    std::string expect;
    for (std::size_t pos = 0; pos < s.size();) {
      std::size_t best = std::string::npos, len = 0;
      for (std::string_view sp : {"<|im_start|>", "<|im_end|>", "<|im"}) {
        const auto f = s.find(sp, pos);
        if (f < best || (f == best && sp.size() > len)) { best = f; len = sp.size(); }
      }
      expect += Nfc(s.substr(pos, best == std::string::npos ? std::string::npos : best - pos));
      if (best == std::string::npos) break;
      expect += s.substr(best, len);
      pos = best + len;
    }
    ASSERT_EQ(tok->Decode(ids), expect);
    for (auto id : ids) ASSERT_LT(id, tok->VocabSize());
    // With specials disabled the whole text is ordinary.
    ASSERT_EQ(tok->Decode(tok->Encode(s, false)), Nfc(s));
  }, {.test_cases = 1000});
}

TEST(Tokenizer, RoundTripOnRealVocabulary) {
  auto tok = RealTokenizer();
  if (!tok) GTEST_SKIP() << "set HALO_FLASH_NEXT_GGUF to the Flash-Next shard 00001 GGUF";
  ASSERT_EQ(tok->VocabSize(), 248320U);
  ASSERT_EQ(tok->eos(), 248046U);
  hegel::test([&](hegel::TestCase& tc) {
    const auto s = DrawText(tc);
    const auto ids = tok->Encode(s, /*parse_special=*/false);
    ASSERT_EQ(tok->Decode(ids), Nfc(s));
    for (auto id : ids) ASSERT_FALSE(tok->IsSpecial(id));
  }, {.test_cases = 300});
}

TEST(Tokenizer, RejectsBrokenVocabularies) {
  std::string error;
  std::vector<std::string> tokens = {"a"};
  std::vector<TokenType> types = {TokenType::kNormal};
  EXPECT_FALSE(Tokenizer::FromVocabulary(tokens, {}, types, &error));  // lacks byte tokens
  EXPECT_FALSE(error.empty());
  std::vector<TokenType> short_types;
  EXPECT_FALSE(Tokenizer::FromVocabulary(tokens, {}, short_types, &error));
}

// --- Qwen2-style pre-tokenizer and offsets (strands-decider tokenizer) -------

namespace {

std::unique_ptr<Tokenizer> DeciderTokenizer() {
  const char* path = std::getenv("HALO_DECIDER_TOKENIZER_GGUF");
  if (!path) return nullptr;
  std::string error;
  auto reader = halo::core::GgufReader::OpenFile(path, &error);
  EXPECT_TRUE(reader) << error;
  if (!reader) return nullptr;
  auto t = Tokenizer::FromGguf(*reader, &error);
  EXPECT_TRUE(t) << error;
  return t;
}

std::vector<std::string> SplitTab(const std::string& line) {
  std::vector<std::string> cols;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= line.size(); ++i)
    if (i == line.size() || line[i] == '\t') { cols.push_back(line.substr(start, i - start)); start = i + 1; }
  return cols;
}

}  // namespace

TEST(Tokenizer, Qwen2PreTokenizerMatchesRegex) {
  const auto rows = LoadTsv("pretok_qwen2.tsv");
  ASSERT_GT(rows.size(), 3000U);
  int failures = 0;
  for (const auto& [in, want] : rows) {
    std::vector<std::string> got;
    const std::string text = Unhex(in);  // pieces are views into it
    for (auto p : PreTokenize(text, PreTokenizer::kQwen2)) got.push_back(Hex(p));
    if (got != Split(want, ',') && ++failures <= 10) ADD_FAILURE() << "input " << in;
  }
  EXPECT_EQ(failures, 0) << "of " << rows.size();
}

TEST(Tokenizer, DeciderEncodeAndOffsetsMatchHf) {
  auto tok = DeciderTokenizer();
  if (!tok) GTEST_SKIP() << "set HALO_DECIDER_TOKENIZER_GGUF (tools/tokenizer_json_to_gguf.py output)";
  ASSERT_EQ(tok->pre_tokenizer(), PreTokenizer::kQwen2);
  std::ifstream f(std::string(HALO_TEST_DATA_DIR) + "/encode_decider.tsv");
  std::string line;
  int failures = 0, rows = 0, offset_failures = 0;
  while (std::getline(f, line)) {
    if (line.starts_with('#')) continue;
    ++rows;
    const auto cols = SplitTab(line);
    const auto enc = tok->EncodeWithOffsets(Unhex(cols[0]));
    std::string ids, offs;
    for (std::size_t i = 0; i < enc.ids.size(); ++i) {
      ids += (i ? " " : "") + std::to_string(enc.ids[i]);
      offs += (i ? " " : "") + std::to_string(enc.offsets[i].first) + ":" + std::to_string(enc.offsets[i].second);
    }
    if (ids != cols[1] && ++failures <= 10) ADD_FAILURE() << "ids for " << cols[0] << "\n want " << cols[1] << "\n got  " << ids;
    if (ids == cols[1] && offs != cols[2] && ++offset_failures <= 10)
      ADD_FAILURE() << "offsets for " << cols[0] << "\n want " << cols[2] << "\n got  " << offs;
  }
  EXPECT_EQ(failures, 0) << "of " << rows;
  EXPECT_EQ(offset_failures, 0) << "of " << rows;
}

TEST(Tokenizer, OffsetsAgreeWithEncodeAndAreOrdered) {
  auto tok = SyntheticTokenizer();
  ASSERT_TRUE(tok);
  hegel::test([&](hegel::TestCase& tc) {
    const auto s = DrawText(tc);
    const auto enc = tok->EncodeWithOffsets(s);
    ASSERT_EQ(enc.ids, tok->Encode(s));
    ASSERT_EQ(enc.offsets.size(), enc.ids.size());
    std::size_t chars = 0;
    for (std::size_t i = 0; i < s.size(); i += DecodeUtf8(s, i).length) ++chars;
    std::size_t prev_lo = 0;
    for (const auto& [lo, hi] : enc.offsets) {
      ASSERT_LE(lo, hi);
      ASSERT_LE(hi, chars);
      ASSERT_GE(lo, prev_lo);
      prev_lo = lo;
    }
  }, {.test_cases = 1000});
}

TEST(Unicode, SegmentwiseNfcEqualsWholeNfc) {
  hegel::test([](hegel::TestCase& tc) {
    const auto s = DrawText(tc);
    std::string joined;
    std::size_t seg = 0;
    for (std::size_t i = 0; i < s.size();) {
      const auto u = DecodeUtf8(s, i);
      if (i > seg && (!u.valid || IsStableStarter(u.value))) {
        joined += Nfc(s.substr(seg, i - seg));
        seg = i;
      }
      i += u.length;
    }
    joined += Nfc(s.substr(seg));
    ASSERT_EQ(joined, Nfc(s));
  }, {.test_cases = 2000});
}

// --- Boundaries Mull found unchecked -----------------------------------------

// The UTF-8 decoder at every length and range boundary (RFC 3629 table 3.7).
TEST(Unicode, DecodeUtf8Boundaries) {
  struct Case {
    const char* bytes;
    char32_t value;
    std::size_t length;
    bool valid;
  };
  const Case cases[] = {
      {"\x7f", 0x7F, 1, true},          {"\xc2\x80", 0x80, 2, true},
      {"\xdf\xbf", 0x7FF, 2, true},     {"\xe0\xa0\x80", 0x800, 3, true},
      {"\xed\x9f\xbf", 0xD7FF, 3, true}, {"\xee\x80\x80", 0xE000, 3, true},
      {"\xef\xbf\xbf", 0xFFFF, 3, true}, {"\xf0\x90\x80\x80", 0x10000, 4, true},
      {"\xf4\x8f\xbf\xbf", 0x10FFFF, 4, true},
      // Invalid: one byte consumed, its value reported.
      {"\x80", 0x80, 1, false},         {"\xc1\xbf", 0xC1, 1, false},
      {"\xe0\x9f\xbf", 0xE0, 1, false},  {"\xed\xa0\x80", 0xED, 1, false},
      {"\xed\xbf\xbf", 0xED, 1, false},  {"\xf0\x8f\xbf\xbf", 0xF0, 1, false},
      {"\xf4\x90\x80\x80", 0xF4, 1, false}, {"\xf5\x80\x80\x80", 0xF5, 1, false},
      {"\xe2\x82", 0xE2, 1, false},      {"\xc3\x28", 0xC3, 1, false},
  };
  for (const auto& c : cases) {
    const auto u = DecodeUtf8(c.bytes, 0);
    EXPECT_EQ(u.valid, c.valid) << Hex(c.bytes);
    EXPECT_EQ(u.length, c.length) << Hex(c.bytes);
    EXPECT_EQ(static_cast<std::uint32_t>(u.value), static_cast<std::uint32_t>(c.value)) << Hex(c.bytes);
  }
}

// Scalar values encode to their shortest form and decode back: all of them
// up to 0x900 and around each boundary, every 97th elsewhere (the whole range
// is slow in the -O0 mutation build).
TEST(Unicode, AppendUtf8RoundTripsScalars) {
  std::vector<char32_t> cps;
  for (char32_t cp = 0; cp <= 0x10FFFF; cp += cp < 0x900 ? 1 : 97) cps.push_back(cp);
  for (char32_t edge : {0xD7FFu, 0xE000u, 0xFFFFu, 0x10000u, 0x10FFFFu})
    for (char32_t d = 0; d < 3; ++d) cps.push_back(edge + d <= 0x10FFFF ? edge + d : edge - d);
  for (char32_t cp : cps) {
    if (cp >= 0xD800 && cp <= 0xDFFF) continue;
    std::string s;
    AppendUtf8(s, cp);
    const std::size_t want = cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
    ASSERT_EQ(s.size(), want) << std::hex << static_cast<std::uint32_t>(cp);
    const auto u = DecodeUtf8(s, 0);
    ASSERT_TRUE(u.valid && u.length == want && u.value == cp) << std::hex << static_cast<std::uint32_t>(cp);
  }
}

// Hangul is composed algorithmically; these are the range edges, with the
// expected results from Python unicodedata (Unicode 16).
TEST(Unicode, HangulCompositionBoundaries) {
  const std::vector<std::pair<std::vector<char32_t>, std::vector<char32_t>>> cases = {
      {{0x1100, 0x1161}, {0xAC00}},         {{0x1112, 0x1175}, {0xD788}},
      {{0x1112, 0x1176}, {0x1112, 0x1176}}, {{0x1113, 0x1161}, {0x1113, 0x1161}},
      {{0x10FF, 0x1161}, {0x10FF, 0x1161}}, {{0x1100, 0x1160}, {0x1100, 0x1160}},
      {{0xAC00, 0x11A7}, {0xAC00, 0x11A7}}, {{0xAC00, 0x11A8}, {0xAC01}},
      {{0xAC00, 0x11C2}, {0xAC1B}},         {{0xAC00, 0x11C3}, {0xAC00, 0x11C3}},
      {{0xAC01, 0x11A8}, {0xAC01, 0x11A8}}, {{0xD7A3}, {0xD7A3}},
      {{0xD7A4, 0x11A8}, {0xD7A4, 0x11A8}}, {{0xABFF, 0x11A8}, {0xABFF, 0x11A8}},
      {{0xD788, 0x11C2}, {0xD7A3}},         {{0x1100, 0x1161, 0x11A8}, {0xAC01}},
  };
  for (const auto& [in, out] : cases) {
    std::string a, b;
    for (auto c : in) AppendUtf8(a, c);
    for (auto c : out) AppendUtf8(b, c);
    EXPECT_EQ(Hex(Nfc(a)), Hex(b)) << Hex(a);
  }
  // A Hangul vowel or trailing consonant can join the syllable before it.
  EXPECT_TRUE(IsStableStarter(0x1100));
  EXPECT_FALSE(IsStableStarter(0x1161));
  EXPECT_FALSE(IsStableStarter(0x1175));
  EXPECT_TRUE(IsStableStarter(0x1176));
  EXPECT_TRUE(IsStableStarter(0x11A7));
  EXPECT_FALSE(IsStableStarter(0x11A8));
  EXPECT_FALSE(IsStableStarter(0x11C2));
  EXPECT_TRUE(IsStableStarter(0x11C3));
  EXPECT_FALSE(IsStableStarter(0x0301));  // a combining mark
}

namespace {

// The synthetic vocabulary as a GGUF, as tools/tokenizer_json_to_gguf.py writes it.
std::vector<std::uint8_t> SyntheticGguf(const char* model, const char* pre, std::int64_t eos,
                                        bool with_merges = true, bool types_as_int = true) {
  std::vector<std::string> tokens;
  std::vector<std::int32_t> types;
  char32_t n = 0;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    std::string s;
    AppendUtf8(s, printable ? static_cast<char32_t>(b) : 256 + n++);
    tokens.push_back(s);
    types.push_back(1);
  }
  tokens.push_back("he");
  types.push_back(1);
  tokens.push_back("<|endoftext|>");
  types.push_back(3);
  halo::core::GgufWriter w;
  if (model) w.Set("tokenizer.ggml.model", std::string(model));
  if (pre) w.Set("tokenizer.ggml.pre", std::string(pre));
  w.Set("tokenizer.ggml.tokens", tokens);
  if (with_merges) w.Set("tokenizer.ggml.merges", std::vector<std::string>{"h e"});
  if (types_as_int) {
    w.Set("tokenizer.ggml.token_type", types);
  } else {
    w.Set("tokenizer.ggml.token_type", std::vector<std::uint32_t>(types.begin(), types.end()));
  }
  if (eos >= 0) {
    w.Set("tokenizer.ggml.eos_token_id", static_cast<std::uint32_t>(eos));
    w.Set("tokenizer.ggml.padding_token_id", static_cast<std::uint32_t>(eos));
  }
  std::string error;
  auto image = w.Serialize(&error);
  EXPECT_FALSE(image.empty()) << error;
  return image;
}

std::unique_ptr<Tokenizer> FromImage(const std::vector<std::uint8_t>& image, std::string* error) {
  auto reader = halo::core::GgufReader::OpenMemory(image.data(), image.size(), error);
  EXPECT_TRUE(reader) << *error;
  return reader ? Tokenizer::FromGguf(*reader, error) : nullptr;
}

}  // namespace

TEST(Tokenizer, FromGgufValidatesMetadata) {
  std::string error;
  const std::uint32_t vocab = 258;
  for (const char* pre : {"qwen35", "qwen2"}) {
    const auto image = SyntheticGguf("gpt2", pre, vocab - 1);
    auto tok = FromImage(image, &error);
    ASSERT_TRUE(tok) << error;
    EXPECT_EQ(tok->VocabSize(), vocab);
    EXPECT_EQ(tok->pre_tokenizer(), std::string_view(pre) == "qwen2" ? PreTokenizer::kQwen2 : PreTokenizer::kQwen35);
    EXPECT_EQ(tok->eos(), vocab - 1);
    EXPECT_EQ(tok->pad(), vocab - 1);
    EXPECT_TRUE(tok->IsSpecial(vocab - 1));
    EXPECT_FALSE(tok->IsSpecial(vocab - 2));
    EXPECT_FALSE(tok->IsSpecial(vocab));
    EXPECT_EQ(tok->Find("he"), std::optional<TokenId>(256));
    EXPECT_EQ(tok->Find("hex"), std::nullopt);
    EXPECT_EQ(tok->TokenBytes(256), "he");
    EXPECT_EQ(tok->TokenBytes(vocab), "");
    EXPECT_EQ(tok->Encode("he"), std::vector<TokenId>{256});
    EXPECT_EQ(tok->Encode("<|endoftext|>he"), (std::vector<TokenId>{vocab - 1, 256}));
  }
  // An id outside the vocabulary is ignored, not trusted.
  auto tok = FromImage(SyntheticGguf("gpt2", "qwen35", vocab), &error);
  ASSERT_TRUE(tok) << error;
  EXPECT_EQ(tok->eos(), kNoToken);
  EXPECT_EQ(tok->pad(), kNoToken);
  tok = FromImage(SyntheticGguf("gpt2", "qwen35", -1), &error);
  ASSERT_TRUE(tok) << error;
  EXPECT_EQ(tok->eos(), kNoToken);

  struct Bad {
    const char* what;
    std::vector<std::uint8_t> image;
  };
  const Bad bad[] = {
      {"model llama", SyntheticGguf("llama", "qwen35", 0)},
      {"no model", SyntheticGguf(nullptr, "qwen35", 0)},
      {"pre llama3", SyntheticGguf("gpt2", "llama3", 0)},
      {"no pre", SyntheticGguf("gpt2", nullptr, 0)},
      {"no merges", SyntheticGguf("gpt2", "qwen35", 0, false)},
      {"unsigned token types", SyntheticGguf("gpt2", "qwen35", 0, true, false)},
  };
  for (const auto& b : bad) {
    error.clear();
    EXPECT_FALSE(FromImage(b.image, &error)) << b.what;
    EXPECT_FALSE(error.empty()) << b.what;
  }
}

TEST(Tokenizer, FromVocabularyRejectsEachDefect) {
  std::vector<std::string> tokens;
  std::vector<TokenType> types;
  char32_t n = 0;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    std::string s;
    AppendUtf8(s, printable ? static_cast<char32_t>(b) : 256 + n++);
    tokens.push_back(s);
    types.push_back(TokenType::kNormal);
  }
  tokens.push_back("he");
  types.push_back(TokenType::kNormal);
  tokens.push_back("<pad>");
  types.push_back(TokenType::kUnused);
  std::string error;
  auto ok = Tokenizer::FromVocabulary(tokens, std::vector<std::string>{"h e"}, types, &error);
  ASSERT_TRUE(ok) << error;
  EXPECT_EQ(ok->TokenBytes(257), "");  // unused rows decode to nothing
  EXPECT_EQ(ok->Encode("hehe"), (std::vector<TokenId>{256, 256}));

  auto rejects = [&](const char* what, std::vector<std::string> t, std::vector<TokenType> ty,
                     std::vector<std::string> merges) {
    error.clear();
    EXPECT_FALSE(Tokenizer::FromVocabulary(t, merges, ty, &error)) << what;
    EXPECT_FALSE(error.empty()) << what;
  };
  auto t2 = tokens;
  t2.push_back("a b");  // a raw space is not byte-level text
  auto ty2 = types;
  ty2.push_back(TokenType::kNormal);
  rejects("raw space", t2, ty2, {"h e"});
  auto ty3 = types;
  ty3['!' - '!'] = TokenType::kControl;  // "!" is the byte token for 0x21
  rejects("special byte token", tokens, ty3, {"h e"});
  rejects("merge without a space", tokens, types, {"he"});
  rejects("merge with two spaces", tokens, types, {"h e x"});
  rejects("merge of unknown tokens", tokens, types, {"h q"});
  rejects("merge to unknown token", tokens, types, {"e h"});
}
