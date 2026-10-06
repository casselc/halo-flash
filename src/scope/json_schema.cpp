#include "scope/json_schema.hpp"

#include <cmath>
#include <regex>
#include <set>
#include <string_view>

#include "scope/evals.hpp"

namespace halo::scope {

namespace {

using json::Value;
using Kind = json::Value::Kind;

const std::set<std::string, std::less<>> kAnnotations = {"title", "description", "$schema", "$id",
                                                          "default", "examples", "format", "$comment"};
const std::set<std::string, std::less<>> kTypes = {"object", "array", "string", "number", "integer", "boolean", "null"};

bool IsNumber(const Value& v) { return v.is(Kind::kInt) || v.is(Kind::kFloat); }
bool IsNonNegInt(const Value& v) { return v.is(Kind::kInt) && v.number() >= 0; }

std::size_t CodePoints(const std::string& s) {
  std::size_t n = 0;
  for (unsigned char c : s) n += (c & 0xC0) != 0x80;
  return n;
}

bool IsInteger(const Value& v) {
  if (v.is(Kind::kInt)) return true;
  if (!v.is(Kind::kFloat)) return false;
  const double d = v.number();
  return std::isfinite(d) && std::floor(d) == d;
}

bool TypeMatches(const std::string& t, const Value& v) {
  if (t == "object") return v.is(Kind::kObject);
  if (t == "array") return v.is(Kind::kArray);
  if (t == "string") return v.is(Kind::kString);
  if (t == "number") return IsNumber(v);
  if (t == "integer") return IsInteger(v);
  if (t == "boolean") return v.is(Kind::kBool);
  return v.is(Kind::kNull);
}

// JSON equality with numbers compared by value (1 == 1.0), as JSON Schema does.
bool JsonEqual(const Value& a, const Value& b) {
  if (IsNumber(a) && IsNumber(b)) return a.number() == b.number();
  if (a.kind() != b.kind()) return false;
  switch (a.kind()) {
    case Kind::kObject: {
      if (a.members().size() != b.members().size()) return false;
      for (const auto& [k, v] : a.members()) {
        const Value* o = b.find(k);
        if (!o || !JsonEqual(v, *o)) return false;
      }
      return true;
    }
    case Kind::kArray: {
      if (a.items().size() != b.items().size()) return false;
      for (std::size_t i = 0; i < a.items().size(); ++i)
        if (!JsonEqual(a.items()[i], b.items()[i])) return false;
      return true;
    }
    default:
      return a == b;
  }
}

bool Check(const Value& s, const std::string& path, std::string* err) {
  if (s.is(Kind::kBool)) return true;
  if (!s.is(Kind::kObject)) return *err = path + ": a schema must be an object or a boolean", false;
  for (const auto& [k, v] : s.members()) {
    const std::string at = path + "/" + k;
    if (kAnnotations.count(k)) continue;
    if (k == "type") {
      auto ok = [](const Value& t) { return t.is(Kind::kString) && kTypes.count(t.str()); };
      if (v.is(Kind::kArray)) {
        if (v.items().empty()) return *err = at + ": empty type list", false;
        for (const auto& t : v.items())
          if (!ok(t)) return *err = at + ": unknown type", false;
      } else if (!ok(v)) {
        return *err = at + ": unknown type", false;
      }
    } else if (k == "properties") {
      if (!v.is(Kind::kObject)) return *err = at + ": must be an object", false;
      for (const auto& [pk, pv] : v.members())
        if (!Check(pv, at + "/" + pk, err)) return false;
    } else if (k == "required") {
      if (!v.is(Kind::kArray)) return *err = at + ": must be an array of strings", false;
      for (const auto& r : v.items())
        if (!r.is(Kind::kString)) return *err = at + ": must be an array of strings", false;
    } else if (k == "additionalProperties" || k == "items" || k == "not") {
      if (!Check(v, at, err)) return false;
    } else if (k == "allOf" || k == "anyOf" || k == "oneOf") {
      if (!v.is(Kind::kArray) || v.items().empty()) return *err = at + ": must be a non-empty array", false;
      for (std::size_t i = 0; i < v.items().size(); ++i)
        if (!Check(v.items()[i], at + "/" + std::to_string(i), err)) return false;
    } else if (k == "minItems" || k == "maxItems" || k == "minLength" || k == "maxLength") {
      if (!IsNonNegInt(v)) return *err = at + ": must be a non-negative integer", false;
    } else if (k == "minimum" || k == "maximum" || k == "exclusiveMinimum" || k == "exclusiveMaximum") {
      if (!IsNumber(v)) return *err = at + ": must be a number", false;
    } else if (k == "enum") {
      if (!v.is(Kind::kArray)) return *err = at + ": must be an array", false;
    } else if (k == "const") {
      // any value
    } else if (k == "pattern") {
      if (!v.is(Kind::kString)) return *err = at + ": must be a string", false;
      try {
        std::regex re(v.str(), std::regex::ECMAScript);
      } catch (const std::regex_error& e) {
        return *err = at + ": invalid regex: " + e.what(), false;
      }
    } else {
      return *err = at + ": unsupported keyword (not checked by this evaluator)", false;
    }
  }
  return true;
}

bool Validate(const Value& s, const Value& d, const std::string& path, std::string* why) {
  if (s.is(Kind::kBool)) return s.boolean() || (*why = path + ": schema false", false);
  auto fail = [&](const std::string& msg) { return *why = (path.empty() ? "/" : path) + ": " + msg, false; };
  if (const Value* t = s.find("type")) {
    bool ok = false;
    if (t->is(Kind::kArray)) {
      for (const auto& x : t->items()) ok = ok || TypeMatches(x.str(), d);
    } else {
      ok = TypeMatches(t->str(), d);
    }
    if (!ok) return fail("type is not " + CanonicalJson(*t));
  }
  if (const Value* e = s.find("enum")) {
    bool ok = false;
    for (const auto& x : e->items()) ok = ok || JsonEqual(x, d);
    if (!ok) return fail("not in enum");
  }
  if (const Value* c = s.find("const"); c && !JsonEqual(*c, d)) return fail("not the const value");
  if (IsNumber(d)) {
    const double x = d.number();
    if (const Value* m = s.find("minimum"); m && x < m->number()) return fail("below minimum");
    if (const Value* m = s.find("maximum"); m && x > m->number()) return fail("above maximum");
    if (const Value* m = s.find("exclusiveMinimum"); m && x <= m->number()) return fail("not above exclusiveMinimum");
    if (const Value* m = s.find("exclusiveMaximum"); m && x >= m->number()) return fail("not below exclusiveMaximum");
  }
  if (d.is(Kind::kString)) {
    const auto n = CodePoints(d.str());
    if (const Value* m = s.find("minLength"); m && n < static_cast<std::size_t>(m->number())) return fail("too short");
    if (const Value* m = s.find("maxLength"); m && n > static_cast<std::size_t>(m->number())) return fail("too long");
    if (const Value* p = s.find("pattern")) {
      std::regex re(p->str(), std::regex::ECMAScript);
      if (!std::regex_search(d.str(), re)) return fail("does not match pattern");
    }
  }
  if (d.is(Kind::kArray)) {
    const auto n = d.items().size();
    if (const Value* m = s.find("minItems"); m && n < static_cast<std::size_t>(m->number())) return fail("too few items");
    if (const Value* m = s.find("maxItems"); m && n > static_cast<std::size_t>(m->number())) return fail("too many items");
    if (const Value* it = s.find("items"))
      for (std::size_t i = 0; i < n; ++i)
        if (!Validate(*it, d.items()[i], path + "/" + std::to_string(i), why)) return false;
  }
  if (d.is(Kind::kObject)) {
    if (const Value* r = s.find("required"))
      for (const auto& k : r->items())
        if (!d.find(k.str())) return fail("missing required property " + k.str());
    const Value* props = s.find("properties");
    const Value* extra = s.find("additionalProperties");
    for (const auto& [k, v] : d.members()) {
      const Value* ps = props ? props->find(k) : nullptr;
      if (ps) {
        if (!Validate(*ps, v, path + "/" + k, why)) return false;
      } else if (extra && !Validate(*extra, v, path + "/" + k, why)) {
        return false;
      }
    }
  }
  if (const Value* all = s.find("allOf"))
    for (const auto& x : all->items())
      if (!Validate(x, d, path, why)) return false;
  if (const Value* any = s.find("anyOf")) {
    bool ok = false;
    std::string ignored;
    for (const auto& x : any->items()) ok = ok || Validate(x, d, path, &ignored);
    if (!ok) return fail("matches no anyOf branch");
  }
  if (const Value* one = s.find("oneOf")) {
    int matches = 0;
    std::string ignored;
    for (const auto& x : one->items()) matches += Validate(x, d, path, &ignored);
    if (matches != 1) return fail("matches " + std::to_string(matches) + " oneOf branches");
  }
  if (const Value* n = s.find("not")) {
    std::string ignored;
    if (Validate(*n, d, path, &ignored)) return fail("matches the not schema");
  }
  return true;
}

}  // namespace

bool CheckJsonSchema(const json::Value& schema, std::string* err) { return Check(schema, "", err); }

bool ValidateJsonSchema(const json::Value& schema, const json::Value& doc, std::string* why) {
  return Validate(schema, doc, "", why);
}

}  // namespace halo::scope
