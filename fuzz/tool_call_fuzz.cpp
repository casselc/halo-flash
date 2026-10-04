// libFuzzer target: the streaming tool-call parser on arbitrary bytes.
// The first byte picks the options and a split pattern; the rest is the
// model output. Checks memory safety plus the parser's contract: every byte
// lands in exactly one event, no call opener in content, and the chunked
// parse equals the one-shot parse (text events of one kind merged).
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "text/tool_call_parser.hpp"

using namespace halo::text;

namespace {

std::vector<StreamEvent> Merge(const std::vector<StreamEvent>& events) {
  std::vector<StreamEvent> out;
  for (const auto& e : events) {
    const bool text = e.kind == StreamEventKind::kContent || e.kind == StreamEventKind::kReasoning ||
                      e.kind == StreamEventKind::kFraming;
    if (text && !out.empty() && out.back().kind == e.kind) out.back().text += e.text;
    else out.push_back(e);
  }
  return out;
}

void Require(bool ok) {
  if (!ok) std::abort();
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size == 0) return 0;
  const std::uint8_t mode = data[0];
  const std::string_view text(reinterpret_cast<const char*>(data + 1), size - 1);
  ToolCallParserOptions o;
  o.start = static_cast<StreamStart>(mode % 3);
  o.tool_calls_in_reasoning = (mode & 4) != 0;
  if (mode & 8) o.max_tool_call_bytes = 64;

  const auto whole = ParseAssistantText(text, o);
  std::string all;
  for (const auto& e : whole) {
    Require(!e.text.empty());
    if (e.kind == StreamEventKind::kContent) Require(e.text.find("<tool_call>") == std::string::npos);
    all += e.text;
  }
  Require(all == text);

  // Chunk sizes 1..16 from the high bits of the mode byte, cycling.
  ToolCallStreamParser p(o);
  std::vector<StreamEvent> events;
  std::size_t at = 0, step = 1 + (mode >> 4);
  while (at < text.size()) {
    const std::size_t n = std::min(step, text.size() - at);
    p.Push(text.substr(at, n), &events);
    at += n;
    step = step % 16 + 1;
  }
  p.Finish(&events);
  Require(Merge(events) == Merge(whole));
  return 0;
}
