#ifndef HALO_TEXT_TOKENIZER_HPP_
#define HALO_TEXT_TOKENIZER_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace halo::core {
class GgufReader;
}

namespace halo::text {

using TokenId = std::uint32_t;
inline constexpr TokenId kNoToken = 0xFFFFFFFFu;

/// GGUF `tokenizer.ggml.token_type` values (llama.cpp's llama_token_type).
enum class TokenType : std::int32_t {
  kNormal = 1,
  kUnknown = 2,
  kControl = 3,
  kUserDefined = 4,
  kUnused = 5,
  kByte = 6,
};

/// Splits NFC text into pre-tokenizer pieces exactly as the Qwen3.5/3.8
/// regex does:
///   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
///   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// Ill-formed UTF-8 bytes are single "other" characters. Pieces are views into
/// `text`, non-empty, and concatenate to it.
[[nodiscard]] std::vector<std::string_view> PreTokenizeQwen35(std::string_view text);

/// The pre-tokenizer regexes this tokenizer implements (GGUF
/// tokenizer.ggml.pre). kQwen2 is the older pattern
///   ...|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|...
/// in which combining marks are punctuation rather than part of words; the
/// strands-decider checkpoint's tokenizer.json uses it.
enum class PreTokenizer { kQwen35, kQwen2 };
[[nodiscard]] std::vector<std::string_view> PreTokenize(std::string_view text, PreTokenizer kind);

/// Byte-level BPE tokenizer for Qwen3.8-Flash-Next (HF `tokenizers` model
/// with NFC normalizer, the regex above, and ByteLevel pre-tokenizer/decoder).
///
/// Encode matches HF `tokenizer.encode(text, add_special_tokens=False)`:
/// added tokens (GGUF types control and user-defined) are split out of the raw
/// text first, every other span is NFC-normalized, pre-tokenized and merged.
/// Arbitrary bytes are accepted; ill-formed UTF-8 is carried through byte
/// tokens so Decode(Encode(x)) == Nfc(x) for every byte string x.
class Tokenizer {
 public:
  [[nodiscard]] static std::unique_ptr<Tokenizer> FromGguf(const core::GgufReader& reader,
                                                           std::string* error);
  [[nodiscard]] static std::unique_ptr<Tokenizer> FromVocabulary(
      std::span<const std::string> tokens, std::span<const std::string> merges,
      std::span<const TokenType> types, std::string* error);

  [[nodiscard]] std::vector<TokenId> Encode(std::string_view text,
                                            bool parse_special = true) const;
  /// Encode, plus each token's [start, end) in the ORIGINAL text counted in
  /// code points, as HF `tokenizers` reports offsets for a Python str. Spans
  /// that NFC changes map every token inside them to the whole original
  /// segment.
  struct Encoded {
    std::vector<TokenId> ids;
    std::vector<std::pair<std::size_t, std::size_t>> offsets;
  };
  [[nodiscard]] Encoded EncodeWithOffsets(std::string_view text, bool parse_special = true) const;

  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens) const;
  /// Raw bytes of one token (special tokens decode to their text).
  [[nodiscard]] std::string_view TokenBytes(TokenId id) const noexcept;

  [[nodiscard]] std::size_t VocabSize() const noexcept { return bytes_.size(); }
  [[nodiscard]] std::optional<TokenId> Find(std::string_view token) const;
  [[nodiscard]] bool IsSpecial(TokenId id) const noexcept {
    return id < special_.size() && special_[id];
  }
  [[nodiscard]] TokenId eos() const noexcept { return eos_; }
  [[nodiscard]] PreTokenizer pre_tokenizer() const noexcept { return pre_; }
  [[nodiscard]] TokenId pad() const noexcept { return pad_; }

 private:
  Tokenizer() = default;
  void EncodeOrdinary(std::string_view text, std::vector<TokenId>& out) const;
  void MergePiece(std::string_view piece, std::vector<TokenId>& out,
                  std::vector<std::pair<std::size_t, std::size_t>>* ranges = nullptr) const;
  void EncodeOrdinaryWithOffsets(std::string_view text, std::size_t base, Encoded& out) const;

  struct PairHash {
    std::size_t operator()(std::pair<TokenId, TokenId> p) const noexcept {
      return (static_cast<std::size_t>(p.first) << 32) ^ p.second;
    }
  };
  struct Merge {
    std::uint32_t rank;
    TokenId result;
  };

  std::vector<std::string> bytes_;  // decoded bytes per token id
  std::vector<bool> special_;
  std::unordered_map<std::string, TokenId> by_text_;  // GGUF token text -> id
  std::unordered_map<std::pair<TokenId, TokenId>, Merge, PairHash> merges_;
  std::vector<std::pair<std::string, TokenId>> specials_;  // matched in raw text
  std::array<TokenId, 256> byte_token_{};
  PreTokenizer pre_ = PreTokenizer::kQwen35;
  TokenId eos_ = kNoToken;
  TokenId pad_ = kNoToken;
};

}  // namespace halo::text

#endif  // HALO_TEXT_TOKENIZER_HPP_
