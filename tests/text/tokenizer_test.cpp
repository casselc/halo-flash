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

TEST(Unicode, NfcMatchesPythonUnicodedata) {
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
