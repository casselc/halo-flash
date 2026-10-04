// libFuzzer target: Nfc, the pre-tokenizer and Encode/Decode on arbitrary
// bytes, with a synthetic byte-level vocabulary. Checks the round-trip
// invariant (specials disabled) as well as memory safety.
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "text/tokenizer.hpp"
#include "text/unicode.hpp"

using namespace halo::text;

static const Tokenizer& Tok() {
  static const auto tok = [] {
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
    const std::vector<std::string> merges = {"Ġ t", "h e", "Ġt he", "i n", "e r"};
    for (const auto& m : merges) {
      tokens.push_back(m.substr(0, m.find(' ')) + m.substr(m.find(' ') + 1));
      types.push_back(TokenType::kNormal);
    }
    tokens.push_back("<|im_end|>");
    types.push_back(TokenType::kControl);
    std::string error;
    auto t = Tokenizer::FromVocabulary(tokens, merges, types, &error);
    if (!t) std::abort();
    return t;
  }();
  return *tok;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view s(reinterpret_cast<const char*>(data), size);
  const std::string n = Nfc(s);
  if (Nfc(n) != n) std::abort();
  std::size_t total = 0;
  for (auto p : PreTokenizeQwen35(s)) {
    if (p.empty()) std::abort();
    total += p.size();
  }
  if (total != s.size()) std::abort();
  if (Tok().Decode(Tok().Encode(s, false)) != n) std::abort();
  (void)Tok().Encode(s, true);
  return 0;
}
