#include "core/json.hpp"

#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>

namespace halo::json {

Value Value::Bool(bool b) {
  Value v;
  v.kind_ = Kind::kBool;
  v.bool_ = b;
  return v;
}
Value Value::Int(std::string digits) {
  Value v;
  v.kind_ = Kind::kInt;
  v.text_ = digits == "-0" ? "0" : std::move(digits);
  return v;
}
Value Value::Int(std::int64_t i) { return Int(std::to_string(i)); }
Value Value::Float(double d) {
  Value v;
  v.kind_ = Kind::kFloat;
  v.float_ = d;
  return v;
}
Value Value::String(std::string s) {
  Value v;
  v.kind_ = Kind::kString;
  v.text_ = std::move(s);
  return v;
}
Value Value::MakeArray(Array items) {
  Value v;
  v.kind_ = Kind::kArray;
  v.array_ = std::move(items);
  return v;
}
Value Value::MakeObject(Object members) {
  Value v;
  v.kind_ = Kind::kObject;
  for (auto& [k, val] : members) v.set(std::move(k), std::move(val));
  return v;
}

double Value::number() const {
  if (kind_ == Kind::kFloat) return float_;
  if (kind_ == Kind::kInt) return std::strtod(text_.c_str(), nullptr);
  return std::numeric_limits<double>::quiet_NaN();
}

const Value* Value::find(std::string_view key) const noexcept {
  for (const auto& [k, v] : object_)
    if (k == key) return &v;
  return nullptr;
}

void Value::set(std::string key, Value v) {
  for (auto& [k, existing] : object_) {
    if (k == key) {
      existing = std::move(v);
      return;
    }
  }
  object_.emplace_back(std::move(key), std::move(v));
}

bool operator==(const Value& a, const Value& b) {
  if (a.kind_ != b.kind_) return false;
  switch (a.kind_) {
    case Value::Kind::kNull: return true;
    case Value::Kind::kBool: return a.bool_ == b.bool_;
    case Value::Kind::kFloat:
      return (std::isnan(a.float_) && std::isnan(b.float_)) ||
             std::memcmp(&a.float_, &b.float_, sizeof(double)) == 0;
    case Value::Kind::kInt:
    case Value::Kind::kString: return a.text_ == b.text_;
    case Value::Kind::kArray: return a.array_ == b.array_;
    case Value::Kind::kObject: return a.object_ == b.object_;
  }
  return false;
}

namespace {

class Parser {
 public:
  Parser(std::string_view s, const ParseLimits& l) : s_(s), limits_(l) {}

  std::optional<Value> Run(std::string* error) {
    if (s_.size() > limits_.max_bytes) return Fail(error, "input too large");
    SkipWs();
    auto v = ParseValue(0);
    if (!v) return Fail(error, err_);
    SkipWs();
    if (i_ != s_.size()) return Fail(error, "trailing characters");
    return v;
  }

 private:
  std::optional<Value> Fail(std::string* error, std::string m) {
    if (error) *error = m + " at offset " + std::to_string(i_);
    return std::nullopt;
  }
  std::optional<Value> Err(const char* m) {
    if (err_.empty()) err_ = m;
    return std::nullopt;
  }
  void SkipWs() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
  }
  bool Literal(std::string_view lit) {
    if (s_.substr(i_, lit.size()) != lit) return false;
    i_ += lit.size();
    return true;
  }

  std::optional<Value> ParseValue(std::size_t depth) {
    if (depth > limits_.max_depth) return Err("nesting too deep");
    if (i_ >= s_.size()) return Err("unexpected end");
    const char c = s_[i_];
    if (c == '{') return ParseObject(depth);
    if (c == '[') return ParseArray(depth);
    if (c == '"') {
      std::string str;
      if (!ParseString(str)) return std::nullopt;
      return Value::String(std::move(str));
    }
    if (Literal("true")) return Value::Bool(true);
    if (Literal("false")) return Value::Bool(false);
    if (Literal("null")) return Value::Null();
    if (Literal("NaN")) return Value::Float(std::numeric_limits<double>::quiet_NaN());
    if (Literal("Infinity")) return Value::Float(std::numeric_limits<double>::infinity());
    if (Literal("-Infinity")) return Value::Float(-std::numeric_limits<double>::infinity());
    if (c == '-' || (c >= '0' && c <= '9')) return ParseNumber();
    return Err("unexpected character");
  }

  std::optional<Value> ParseNumber() {
    const std::size_t start = i_;
    if (s_[i_] == '-') ++i_;
    if (i_ >= s_.size()) return Err("bad number");
    if (s_[i_] == '0') {
      ++i_;
    } else if (s_[i_] >= '1' && s_[i_] <= '9') {
      while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
    } else {
      return Err("bad number");
    }
    bool is_float = false;
    if (i_ < s_.size() && s_[i_] == '.') {
      is_float = true;
      ++i_;
      const std::size_t d = i_;
      while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
      if (i_ == d) return Err("bad fraction");
    }
    if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      is_float = true;
      ++i_;
      if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
      const std::size_t d = i_;
      while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
      if (i_ == d) return Err("bad exponent");
    }
    const std::string_view lex = s_.substr(start, i_ - start);
    if (!is_float) return Value::Int(std::string(lex));
    double d = 0;
    const auto r = std::from_chars(lex.data(), lex.data() + lex.size(), d);
    // Python float() overflows to inf rather than failing.
    if (r.ec == std::errc::result_out_of_range) {
      d = std::strtod(std::string(lex).c_str(), nullptr);
    } else if (r.ec != std::errc()) {
      return Err("bad number");
    }
    return Value::Float(d);
  }

  bool Hex4(std::uint32_t& out) {
    if (i_ + 4 > s_.size()) return false;
    out = 0;
    for (int k = 0; k < 4; ++k) {
      const char h = s_[i_++];
      out <<= 4;
      if (h >= '0' && h <= '9') out |= h - '0';
      else if (h >= 'a' && h <= 'f') out |= h - 'a' + 10;
      else if (h >= 'A' && h <= 'F') out |= h - 'A' + 10;
      else return false;
    }
    return true;
  }

  static void Append(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) out.push_back(static_cast<char>(cp));
    else if (cp < 0x800) { out.push_back(static_cast<char>(0xC0 | (cp >> 6))); out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) { out.push_back(static_cast<char>(0xE0 | (cp >> 12))); out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F))); out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
    else { out.push_back(static_cast<char>(0xF0 | (cp >> 18))); out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F))); out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F))); out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
  }

  bool ParseString(std::string& out) {
    ++i_;  // opening quote
    while (true) {
      if (i_ >= s_.size()) { Err("unterminated string"); return false; }
      const auto c = static_cast<unsigned char>(s_[i_]);
      if (c == '"') { ++i_; return true; }
      if (c < 0x20) { Err("control character in string"); return false; }
      if (c == '\\') {
        ++i_;
        if (i_ >= s_.size()) { Err("bad escape"); return false; }
        const char e = s_[i_++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            std::uint32_t cp;
            if (!Hex4(cp)) { Err("bad \\u escape"); return false; }
            if (cp >= 0xD800 && cp <= 0xDBFF) {
              std::uint32_t lo;
              if (s_.substr(i_, 2) != "\\u") { Err("lone surrogate"); return false; }
              i_ += 2;
              if (!Hex4(lo) || lo < 0xDC00 || lo > 0xDFFF) { Err("lone surrogate"); return false; }
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
              Err("lone surrogate");
              return false;
            }
            Append(out, cp);
            break;
          }
          default: Err("bad escape"); return false;
        }
        continue;
      }
      // Raw UTF-8: validate the sequence.
      std::size_t len;
      std::uint32_t cp, min;
      if (c < 0x80) { len = 1; cp = c; min = 0; }
      else if (c >= 0xC2 && c <= 0xDF) { len = 2; cp = c & 0x1F; min = 0x80; }
      else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; min = 0x800; }
      else if (c >= 0xF0 && c <= 0xF4) { len = 4; cp = c & 0x07; min = 0x10000; }
      else { Err("invalid UTF-8"); return false; }
      if (i_ + len > s_.size()) { Err("invalid UTF-8"); return false; }
      for (std::size_t k = 1; k < len; ++k) {
        const auto b = static_cast<unsigned char>(s_[i_ + k]);
        if ((b & 0xC0) != 0x80) { Err("invalid UTF-8"); return false; }
        cp = (cp << 6) | (b & 0x3F);
      }
      if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) { Err("invalid UTF-8"); return false; }
      out.append(s_.substr(i_, len));
      i_ += len;
    }
  }

  std::optional<Value> ParseArray(std::size_t depth) {
    ++i_;
    Value v = Value::MakeArray();
    SkipWs();
    if (i_ < s_.size() && s_[i_] == ']') { ++i_; return v; }
    while (true) {
      SkipWs();
      auto item = ParseValue(depth + 1);
      if (!item) return std::nullopt;
      v.push(std::move(*item));
      SkipWs();
      if (i_ < s_.size() && s_[i_] == ',') { ++i_; continue; }
      if (i_ < s_.size() && s_[i_] == ']') { ++i_; return v; }
      return Err("expected , or ]");
    }
  }

  std::optional<Value> ParseObject(std::size_t depth) {
    ++i_;
    Value v = Value::MakeObject();
    SkipWs();
    if (i_ < s_.size() && s_[i_] == '}') { ++i_; return v; }
    while (true) {
      SkipWs();
      if (i_ >= s_.size() || s_[i_] != '"') return Err("expected key");
      std::string key;
      if (!ParseString(key)) return std::nullopt;
      SkipWs();
      if (i_ >= s_.size() || s_[i_] != ':') return Err("expected :");
      ++i_;
      SkipWs();
      auto item = ParseValue(depth + 1);
      if (!item) return std::nullopt;
      if (limits_.reject_duplicate_keys && v.find(key)) return Err("duplicate object key");
      v.set(std::move(key), std::move(*item));
      SkipWs();
      if (i_ < s_.size() && s_[i_] == ',') { ++i_; continue; }
      if (i_ < s_.size() && s_[i_] == '}') { ++i_; return v; }
      return Err("expected , or }");
    }
  }

  std::string_view s_;
  ParseLimits limits_;
  std::size_t i_ = 0;
  std::string err_;
};

void DumpString(std::string& out, const std::string& s) {
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

void Dump(std::string& out, const Value& v, int indent, int level) {
  const auto newline = [&](int lvl) {
    out.push_back('\n');
    out.append(static_cast<std::size_t>(indent * lvl), ' ');
  };
  switch (v.kind()) {
    case Value::Kind::kNull: out += "null"; break;
    case Value::Kind::kBool: out += v.boolean() ? "true" : "false"; break;
    case Value::Kind::kInt: out += v.int_digits(); break;
    case Value::Kind::kFloat: out += PythonFloatRepr(v.number()); break;
    case Value::Kind::kString: DumpString(out, v.str()); break;
    case Value::Kind::kArray: {
      if (v.items().empty()) { out += "[]"; break; }
      out.push_back('[');
      bool first = true;
      for (const auto& item : v.items()) {
        if (!first) out += indent >= 0 ? "," : ", ";
        first = false;
        if (indent >= 0) newline(level + 1);
        Dump(out, item, indent, level + 1);
      }
      if (indent >= 0) newline(level);
      out.push_back(']');
      break;
    }
    case Value::Kind::kObject: {
      if (v.members().empty()) { out += "{}"; break; }
      out.push_back('{');
      bool first = true;
      for (const auto& [k, item] : v.members()) {
        if (!first) out += indent >= 0 ? "," : ", ";
        first = false;
        if (indent >= 0) newline(level + 1);
        DumpString(out, k);
        out += ": ";
        Dump(out, item, indent, level + 1);
      }
      if (indent >= 0) newline(level);
      out.push_back('}');
      break;
    }
  }
}

}  // namespace

std::optional<Value> Parse(std::string_view text, std::string* error, const ParseLimits& limits) {
  return Parser(text, limits).Run(error);
}

std::string DumpPython(const Value& v, int indent) {
  std::string out;
  Dump(out, v, indent, 0);
  return out;
}

std::string PythonFloatRepr(double d) {
  if (std::isnan(d)) return "NaN";
  if (std::isinf(d)) return d > 0 ? "Infinity" : "-Infinity";
  if (d == 0) return std::signbit(d) ? "-0.0" : "0.0";
  // Shortest round-trip digits via scientific to_chars, then Python's layout:
  // fixed notation when -4 <= exponent < 16, else d.ddde±XX.
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::scientific);
  std::string sci(buf, r.ptr);
  const bool neg = sci[0] == '-';
  if (neg) sci.erase(0, 1);
  const auto e = sci.find('e');
  const int exp = std::stoi(sci.substr(e + 1));
  std::string digits;
  for (char c : sci.substr(0, e))
    if (c != '.') digits.push_back(c);
  std::string out = neg ? "-" : "";
  if (exp >= -4 && exp < 16) {
    if (exp >= 0) {
      if (static_cast<int>(digits.size()) <= exp + 1) {
        out += digits + std::string(static_cast<std::size_t>(exp + 1 - static_cast<int>(digits.size())), '0') + ".0";
      } else {
        out += digits.substr(0, static_cast<std::size_t>(exp + 1)) + "." + digits.substr(static_cast<std::size_t>(exp + 1));
      }
    } else {
      out += "0." + std::string(static_cast<std::size_t>(-exp - 1), '0') + digits;
    }
  } else {
    out += digits.substr(0, 1);
    if (digits.size() > 1) out += "." + digits.substr(1);
    char eb[16];
    std::snprintf(eb, sizeof(eb), "e%c%02d", exp < 0 ? '-' : '+', exp < 0 ? -exp : exp);
    out += eb;
  }
  return out;
}

}  // namespace halo::json
