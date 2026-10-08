#ifndef HALO_CORE_JSON_HPP_
#define HALO_CORE_JSON_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace halo::json {

/// A JSON value with Python `json` semantics where they matter for
/// byte-exact prompt rendering:
///  - objects keep insertion order; a repeated key replaces the earlier value
///    in place (Python dict semantics);
///  - integers and floats are distinct and keep full precision: integers are
///    stored as their decimal digits (arbitrary size), floats as doubles.
class Value {
 public:
  enum class Kind : std::uint8_t { kNull, kBool, kInt, kFloat, kString, kArray, kObject };
  using Array = std::vector<Value>;
  using Member = std::pair<std::string, Value>;
  using Object = std::vector<Member>;

  Value() = default;
  static Value Null() { return {}; }
  static Value Bool(bool b);
  static Value Int(std::string digits);  // optional '-', then digits, no leading zeros
  static Value Int(std::int64_t v);
  static Value Float(double d);
  static Value String(std::string s);
  static Value MakeArray(Array items = {});
  static Value MakeObject(Object members = {});

  Kind kind() const noexcept { return kind_; }
  bool is(Kind k) const noexcept { return kind_ == k; }
  bool boolean() const noexcept { return bool_; }
  const std::string& int_digits() const noexcept { return text_; }
  double number() const;  // int or float as double
  const std::string& str() const noexcept { return text_; }
  const Array& items() const noexcept { return array_; }
  const Object& members() const noexcept { return object_; }
  Array& items() noexcept { return array_; }

  const Value* find(std::string_view key) const noexcept;
  /// Python dict assignment: replaces in place or appends.
  void set(std::string key, Value v);
  void push(Value v) { array_.push_back(std::move(v)); }

  friend bool operator==(const Value& a, const Value& b);

 private:
  Kind kind_ = Kind::kNull;
  bool bool_ = false;
  double float_ = 0;
  std::string text_;  // string contents or integer digits
  Array array_;
  Object object_;
};

struct ParseLimits {
  std::size_t max_depth = 128;
  std::size_t max_bytes = 64u << 20;
  bool reject_duplicate_keys = false;  ///< an object with a repeated key is an error (security tokens, JWKS)
};

/// Strict RFC 8259 parser plus Python's NaN/Infinity/-Infinity literals.
/// Strings must be valid UTF-8; escapes may form surrogate pairs, but lone
/// surrogates are rejected (they cannot be tokenized as UTF-8).
[[nodiscard]] std::optional<Value> Parse(std::string_view text, std::string* error,
                                         const ParseLimits& limits = {});

/// Python `json.dumps(v, indent=indent, ensure_ascii=False)` byte for byte
/// (indent < 0 means the compact default separators ", " and ": ").
[[nodiscard]] std::string DumpPython(const Value& v, int indent = -1);

/// Python `repr(float)` (the shortest round-trip form), as json.dumps emits it.
[[nodiscard]] std::string PythonFloatRepr(double d);

}  // namespace halo::json

#endif  // HALO_CORE_JSON_HPP_
