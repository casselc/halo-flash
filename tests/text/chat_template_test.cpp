// Chat-template renderer tests.
//
// Differential: tests/text/data/chat/chat_{flash_next,qwen35}.json hold
// conversations rendered by HF transformers' apply_chat_template with the
// models' own Jinja templates (tools/gen_chat_fixtures.py). Every case must
// match byte for byte, or fail where the template raises.
// Properties (Hegel): PyStrip agrees with a direct definition of Python's
// str.strip() on generated text.

#include <hegel/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

#include "core/json.hpp"
#include "text/chat_template.hpp"
#include "text/unicode.hpp"

namespace gs = hegel::generators;
using halo::json::Value;
using namespace halo::text;

namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  EXPECT_TRUE(f.good()) << path;
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string DataPath(const char* name) { return std::string(HALO_TEST_DATA_DIR) + "/chat/" + name; }

void CheckFixtures(ChatTemplateKind kind, const char* fixture, const char* jinja) {
  const std::string source = ReadFile(DataPath(jinja));
  ASSERT_EQ(IdentifyChatTemplate(source), kind) << ChatTemplateSha256(source);
  std::string error;
  const auto doc = halo::json::Parse(ReadFile(DataPath(fixture)), &error);
  ASSERT_TRUE(doc) << error;
  ASSERT_EQ(doc->find("template_sha256")->str(), ChatTemplateSha256(source));
  std::size_t rendered = 0, raised = 0;
  for (const auto& c : doc->find("cases")->items()) {
    const std::string name = c.find("name")->str();
    ChatTemplateOptions options;
    ASSERT_TRUE(ChatOptionsFromJson(*c.find("kwargs"), &options, &error)) << name << ": " << error;
    error.clear();
    const auto out = RenderChat(kind, *c.find("messages"), c.find("tools"), options, &error);
    if (const Value* expected = c.find("output")) {
      ASSERT_TRUE(out) << name << ": unexpected error " << error;
      EXPECT_EQ(*out, expected->str()) << name;
      ++rendered;
    } else {
      EXPECT_FALSE(out) << name << ": template raises " << c.find("error")->str() << "\n" << *out;
      EXPECT_FALSE(error.empty()) << name;
      ++raised;
    }
  }
  // The fixtures exercise both outcomes in quantity.
  EXPECT_GT(rendered, 250u);
  EXPECT_GT(raised, 40u);
}

}  // namespace

TEST(ChatTemplate, FlashNextMatchesTransformers) {
  CheckFixtures(ChatTemplateKind::kFlashNext, "chat_flash_next.json", "flash_next.jinja");
}

TEST(ChatTemplate, Qwen35MatchesTransformers) {
  CheckFixtures(ChatTemplateKind::kQwen35, "chat_qwen35.json", "qwen35.jinja");
}

TEST(ChatTemplate, UnknownTemplateIsNotIdentified) {
  std::string source = ReadFile(DataPath("flash_next.jinja"));
  source += "\n";
  EXPECT_FALSE(IdentifyChatTemplate(source));
  EXPECT_EQ(ChatTemplateSha256(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(ChatTemplateSha256(std::string(1000, 'a')),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

TEST(ChatTemplate, GenerationPromptDefaultsDiffer) {
  ChatTemplateOptions o;
  EXPECT_EQ(GenerationPrompt(ChatTemplateKind::kFlashNext, o), "<|im_start|>assistant\n<think>\n");
  EXPECT_EQ(GenerationPrompt(ChatTemplateKind::kQwen35, o), "<|im_start|>assistant\n<think>\n\n</think>\n\n");
  o.enable_thinking = false;
  EXPECT_EQ(GenerationPrompt(ChatTemplateKind::kFlashNext, o), "<|im_start|>assistant\n<think>\n\n</think>\n\n");
  o.enable_thinking = true;
  EXPECT_EQ(GenerationPrompt(ChatTemplateKind::kQwen35, o), "<|im_start|>assistant\n<think>\n");
}

TEST(ChatTemplate, OptionsRejectWrongTypes) {
  std::string error;
  ChatTemplateOptions o;
  auto kw = Value::MakeObject();
  kw.set("enable_thinking", Value::String("yes"));
  EXPECT_FALSE(ChatOptionsFromJson(kw, &o, &error));
  kw = Value::MakeObject();
  kw.set("reasoning_effort", Value::Bool(true));
  EXPECT_FALSE(ChatOptionsFromJson(kw, &o, &error));
  kw = Value::MakeObject();
  kw.set("unrelated", Value::Int(1));
  kw.set("reasoning_effort", Value::String("low"));
  ASSERT_TRUE(ChatOptionsFromJson(kw, &o, &error));
  EXPECT_EQ(o.reasoning_effort, "low");
}

// Python's str.isspace() code points (bidi WS/B/S or category Zs).
bool PythonIsSpace(char32_t c) {
  static const char32_t kSpaces[] = {0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
                                     0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004,
                                     0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029,
                                     0x202f, 0x205f, 0x3000};
  for (char32_t s : kSpaces)
    if (s == c) return true;
  return false;
}

TEST(ChatTemplate, PyStripMatchesDefinition) {
  hegel::test([](hegel::TestCase& tc) {
    const auto cps = tc.draw(gs::vectors(
        gs::one_of({gs::sampled_from<std::uint32_t>({0x09, 0x0a, 0x0d, 0x1c, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
                                                0x200a, 0x200b, 0x2028, 0x202f, 0x205f, 0x3000, 0xfeff,
                                                0x180e, 'a', 0xe9, 0x1f999}),
                    gs::integers<std::uint32_t>({.min_value = 0, .max_value = 0x10ffff})}),
        {.max_size = 12}));
    std::vector<char32_t> chars;
    for (std::uint32_t c : cps)
      if (c < 0xd800 || c > 0xdfff) chars.push_back(c);
    std::size_t lo = 0, hi = chars.size();
    while (lo < hi && PythonIsSpace(chars[lo])) ++lo;
    while (hi > lo && PythonIsSpace(chars[hi - 1])) --hi;
    std::string s, want;
    for (std::size_t i = 0; i < chars.size(); ++i) {
      AppendUtf8(s, chars[i]);
      if (i >= lo && i < hi) AppendUtf8(want, chars[i]);
    }
    // Ill-formed bytes are never whitespace and stop the strip.
    if (tc.draw(gs::booleans())) {
      s = "\x85" + s + "\xe3\x80";
      want = s;
      want = std::string(PyStrip(want));
      if (!(want.starts_with("\x85") && want.ends_with("\xe3\x80"))) throw std::runtime_error("stripped a bad byte");
      return;
    }
    if (std::string(PyStrip(s)) != want) throw std::runtime_error("PyStrip mismatch");
  });
}
