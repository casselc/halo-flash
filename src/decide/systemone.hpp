#ifndef HALO_DECIDE_SYSTEMONE_HPP_
#define HALO_DECIDE_SYSTEMONE_HPP_

// The System One decision API (POST /v1/systemone), wire-compatible with
// strands-decider's schema (which follows TypeSafe Jev's documented shape).
// Three question kinds share one mechanism: the question renders N numbered
// options, a readout gives one logit per option, and a softmax over them is
// read back as P(true) (noul), a choice, or an expected level (score).
//
// Rendering and readout reproduce strands_decider.prompting / infer /
// schema byte for byte; tests/decide checks them against fixtures produced by
// that package.

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/json.hpp"

namespace halo::decide {

enum class Kind { kNoul, kChoice, kScore };
const char* KindName(Kind k) noexcept;

inline constexpr std::size_t kMaxChoiceOptions = 255;
inline constexpr std::size_t kMinScoreLevels = 2;
inline constexpr std::size_t kMaxScoreLevels = 10;

struct Question {
  std::string name;
  Kind kind = Kind::kNoul;
  json::Value instructions;  ///< Content: string, object or array
  /// noul: optional {"true"/"false" -> Content|null}; choice: option name ->
  /// Content|null (2..255); score: ascending level descriptions (2..10).
  std::vector<std::pair<std::string, std::optional<json::Value>>> criteria;
};

struct Request {
  json::Value state;  ///< Content
  std::vector<Question> questions;
  std::string model = "strands-decider-latest";
  std::vector<std::string> images;  ///< base64; text-only engines refuse them
};

/// Validates like the pydantic SystemOneRequest model: unknown fields are
/// ignored, a question without "type" is read as the first kind that
/// validates (noul, choice, score), Content is a string, object or array.
[[nodiscard]] std::optional<Request> ParseRequest(const json::Value& body, std::string* error);

struct RenderedQuestion {
  std::string text;
  std::vector<std::string> slot_labels;
  std::vector<std::string> slot_descriptions;
  Kind kind = Kind::kNoul;
  /// [start, end) of each option line in `text`, in Unicode code points (as
  /// Python indexes strings); the pointer head reads each line's last token.
  std::vector<std::pair<std::size_t, std::size_t>> option_spans;
};

/// Python `render_content`: strings are str.strip()ed, objects and arrays are
/// json.dumps(indent=2, ensure_ascii=False); null renders as "".
[[nodiscard]] std::string RenderContent(const json::Value* content);
[[nodiscard]] std::string RenderState(const json::Value& state);
/// `order` permutes the rendered options (training augmentation).
[[nodiscard]] RenderedQuestion RenderQuestion(const Question& q,
                                              const std::vector<std::size_t>* order = nullptr);

/// Softmax over one question's option logits after dividing by the kind's
/// temperature, computed in FP32 as the reference (torch log_softmax) does.
[[nodiscard]] std::vector<float> SlotProbabilities(std::span<const float> logits, float temperature);

struct Answer {
  Kind kind = Kind::kNoul;
  double noul = 0;
  std::string choice;
  double score = 0;
  std::vector<std::pair<std::string, std::string>> legend;
  std::vector<std::pair<std::string, double>> probabilities;
  double confidence = 0;
};

[[nodiscard]] Answer ReadAnswer(const RenderedQuestion& rq, std::span<const float> probs,
                                double ordinal_smoothing);
[[nodiscard]] json::Value AnswerToJson(const Answer& a);

// Exposed for tests.
[[nodiscard]] double PyRound(double x, int ndigits);
[[nodiscard]] double DeriveConfidence(std::span<const double> p);
[[nodiscard]] double DeriveScoreConfidence(std::span<const double> p, double ordinal_smoothing);
[[nodiscard]] bool PyIsSpace(char32_t cp) noexcept;
[[nodiscard]] std::size_t CodepointLength(std::string_view utf8);

}  // namespace halo::decide

#endif  // HALO_DECIDE_SYSTEMONE_HPP_
