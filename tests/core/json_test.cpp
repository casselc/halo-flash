// halo::json: differential fixtures against CPython's json module plus Hegel
// properties (round trip, float repr round trip, hostile input).

#include <hegel/gtest.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "core/json.hpp"

namespace gs = hegel::generators;
using halo::json::Value;

namespace {

std::string Unhex(std::string_view h) {
  std::string out;
  for (std::size_t i = 0; i + 1 < h.size(); i += 2)
    out.push_back(static_cast<char>(std::stoi(std::string(h.substr(i, 2)), nullptr, 16)));
  return out;
}

std::vector<std::vector<std::string>> LoadTsv(const char* name) {
  std::ifstream f(std::string(HALO_CORE_TEST_DATA_DIR) + "/" + name);
  EXPECT_TRUE(f.good()) << name;
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.starts_with('#')) continue;
    std::vector<std::string> cols;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i) {
      if (i == line.size() || line[i] == '\t') {
        cols.push_back(line.substr(start, i - start));
        start = i + 1;
      }
    }
    rows.push_back(cols);
  }
  return rows;
}

Value DrawValue(hegel::TestCase& tc, int depth) {
  const int kind = tc.draw(gs::integers<int>({.min_value = 0, .max_value = depth > 3 ? 5 : 7}));
  switch (kind) {
    case 0: return Value::Null();
    case 1: return Value::Bool(tc.draw(gs::booleans()));
    case 2: return Value::Int(tc.draw(gs::integers<std::int64_t>()));
    case 3: {
      std::uint64_t bits = tc.draw(gs::integers<std::uint64_t>());
      double d;
      std::memcpy(&d, &bits, 8);
      return Value::Float(d);
    }
    case 4: return Value::Float(tc.draw(gs::floats<double>()));
    case 5: return Value::String(tc.draw(gs::text({.max_size = 12})));
    case 6: {
      Value a = Value::MakeArray();
      const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4}));
      for (int i = 0; i < n; ++i) a.push(DrawValue(tc, depth + 1));
      return a;
    }
    default: {
      Value o = Value::MakeObject();
      const auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4}));
      for (int i = 0; i < n; ++i)
        o.set(tc.draw(gs::sampled_from<std::string>({"a", "b", "é", "", "k\"q"})), DrawValue(tc, depth + 1));
      return o;
    }
  }
}

}  // namespace

TEST(Json, MatchesCPythonLoadsAndDumps) {
  const auto rows = LoadTsv("json.tsv");
  ASSERT_GT(rows.size(), 1000U);
  int failures = 0;
  for (const auto& r : rows) {
    const auto input = Unhex(r[0]);
    std::string error;
    const auto v = halo::json::Parse(input, &error);
    const bool want_ok = r[1] == "ok";
    if (v.has_value() != want_ok) {
      if (++failures <= 10) ADD_FAILURE() << "accept mismatch on " << r[0] << " (" << error << ")";
      continue;
    }
    if (!v) continue;
    const auto indent2 = halo::json::DumpPython(*v, 2);
    const auto compact = halo::json::DumpPython(*v);
    if (indent2 != Unhex(r[2]) || compact != Unhex(r[3])) {
      if (++failures <= 10)
        ADD_FAILURE() << "dumps mismatch on " << r[0] << "\n got " << compact << "\n want " << Unhex(r[3]);
    }
  }
  EXPECT_EQ(failures, 0) << "of " << rows.size();
}

TEST(Json, FloatReprMatchesCPython) {
  const auto rows = LoadTsv("float.tsv");
  int failures = 0;
  for (const auto& r : rows) {
    const std::uint64_t bits = std::stoull(r[0], nullptr, 16);
    double d;
    std::memcpy(&d, &bits, 8);
    const auto got = halo::json::PythonFloatRepr(d);
    if (got != r[1] && ++failures <= 10) ADD_FAILURE() << r[0] << ": got " << got << " want " << r[1];
  }
  EXPECT_EQ(failures, 0) << "of " << rows.size();
}

TEST(Json, DumpParseRoundTrip) {
  hegel::test([](hegel::TestCase& tc) {
    const Value v = DrawValue(tc, 0);
    for (int indent : {-1, 0, 2, 4}) {
      const auto text = halo::json::DumpPython(v, indent);
      std::string error;
      const auto back = halo::json::Parse(text, &error);
      ASSERT_TRUE(back) << error << "\n" << text;
      ASSERT_EQ(halo::json::DumpPython(*back, indent), text);
      ASSERT_TRUE(*back == v) << text;
    }
  }, {.test_cases = 500});
}

TEST(Json, FloatReprRoundTripsBits) {
  hegel::test([](hegel::TestCase& tc) {
    const auto bits = tc.draw(gs::integers<std::uint64_t>());
    double d;
    std::memcpy(&d, &bits, 8);
    const auto text = halo::json::PythonFloatRepr(d);
    const auto back = halo::json::Parse(text, nullptr);
    ASSERT_TRUE(back) << text;
    ASSERT_TRUE(back->is(Value::Kind::kFloat)) << text;
    const double e = back->number();
    if (std::isnan(d)) {
      ASSERT_TRUE(std::isnan(e));
    } else {
      ASSERT_EQ(std::memcmp(&d, &e, 8), 0) << text;
    }
  }, {.test_cases = 2000});
}

TEST(Json, HostileInputIsRejectedSafely) {
  hegel::test([](hegel::TestCase& tc) {
    std::string text;
    if (tc.draw(gs::booleans())) {
      auto b = tc.draw(gs::binary({.max_size = 200}));
      text.assign(b.begin(), b.end());
    } else {
      text = halo::json::DumpPython(DrawValue(tc, 0), tc.draw(gs::booleans()) ? 2 : -1);
      const auto n = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 4}));
      for (int i = 0; i < n && !text.empty(); ++i)
        text[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = text.size() - 1}))] =
            static_cast<char>(tc.draw(gs::integers<std::uint8_t>()));
    }
    std::string exact(text);  // exact-size heap buffer for ASan
    std::string error;
    const auto v = halo::json::Parse(exact, &error);
    if (v) (void)halo::json::DumpPython(*v, 2);  // whatever parses must dump
  }, {.test_cases = 2000});
}

TEST(Json, DepthLimitIsEnforced) {
  const std::string deep = std::string(200, '[') + std::string(200, ']');
  std::string error;
  EXPECT_FALSE(halo::json::Parse(deep, &error));
  EXPECT_NE(error.find("deep"), std::string::npos);
  EXPECT_TRUE(halo::json::Parse(std::string(100, '[') + std::string(100, ']'), &error));
}

// The limits are exact: max_depth nested containers below the top value, and
// max_bytes of input.
TEST(Json, LimitsAreExact) {
  std::string error;
  const halo::json::ParseLimits depth2{.max_depth = 2};
  EXPECT_TRUE(halo::json::Parse("[[[]]]", &error, depth2)) << error;
  EXPECT_TRUE(halo::json::Parse("{\"a\":{\"b\":[]}}", &error, depth2)) << error;
  EXPECT_FALSE(halo::json::Parse("[[[[]]]]", &error, depth2));
  EXPECT_NE(error.find("deep"), std::string::npos) << error;
  EXPECT_FALSE(halo::json::Parse("{\"a\":{\"b\":[[]]}}", &error, depth2));
  const halo::json::ParseLimits bytes3{.max_bytes = 3};
  EXPECT_TRUE(halo::json::Parse("[1]", &error, bytes3)) << error;
  EXPECT_FALSE(halo::json::Parse("[1] ", &error, bytes3));
  EXPECT_NE(error.find("too large"), std::string::npos) << error;
}

// CPython's json.dumps(indent=0): newlines, no indentation.
TEST(Json, IndentZeroMatchesCPython) {
  std::string error;
  auto v = halo::json::Parse(R"([1,{"a":2,"b":[]},[],"x"])", &error);
  ASSERT_TRUE(v) << error;
  EXPECT_EQ(halo::json::DumpPython(*v, 0), "[\n1,\n{\n\"a\": 2,\n\"b\": []\n},\n[],\n\"x\"\n]");
  v = halo::json::Parse(R"({"k":[1.5,null,true]})", &error);
  ASSERT_TRUE(v) << error;
  EXPECT_EQ(halo::json::DumpPython(*v, 0), "{\n\"k\": [\n1.5,\nnull,\ntrue\n]\n}");
}

TEST(Json, ValueAccessors) {
  EXPECT_EQ(Value::Int("123").number(), 123.0);
  EXPECT_EQ(Value::Int(-7).number(), -7.0);
  EXPECT_EQ(Value::Float(2.5).number(), 2.5);
  EXPECT_TRUE(std::isnan(Value::String("1").number()));
  EXPECT_TRUE(std::isnan(Value::Null().number()));
  const Value o = Value::MakeObject({{"a", Value::Int(1)}, {"b", Value::Bool(true)}, {"a", Value::Int(2)}});
  ASSERT_NE(o.find("a"), nullptr);
  EXPECT_TRUE(*o.find("a") == Value::Int(2));  // a later key replaces an earlier one, as in CPython
  ASSERT_NE(o.find("b"), nullptr);
  EXPECT_EQ(o.find("c"), nullptr);
  EXPECT_EQ(halo::json::DumpPython(o), R"({"a": 2, "b": true})");
}
