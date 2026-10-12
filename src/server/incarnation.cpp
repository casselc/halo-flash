#include "server/incarnation.hpp"

#include <sys/random.h>

#include <array>
#include <cerrno>
#include <stdexcept>

namespace halo::server {

namespace {

std::string RandomId() {
  std::array<unsigned char, 16> bytes{};
  std::size_t got = 0;
  while (got < bytes.size()) {
    const ssize_t n = ::getrandom(bytes.data() + got, bytes.size() - got, 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("incarnation: getrandom failed");
    }
    got += static_cast<std::size_t>(n);
  }
  static constexpr char kHex[] = "0123456789abcdef";
  std::string id;
  id.reserve(2 * bytes.size());
  for (const unsigned char b : bytes) {
    id += kHex[b >> 4];
    id += kHex[b & 15];
  }
  return id;
}

}  // namespace

Incarnation::Incarnation() : id_(RandomId()), start_(Clock::now()) {}

std::int64_t Incarnation::NowMs() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_).count();
}

void Incarnation::Stamp(std::vector<std::pair<std::string, std::string>>& headers) const {
  headers.emplace_back("Halo-Incarnation", id_);
  headers.emplace_back("Halo-Time", std::to_string(NowMs()));
}

}  // namespace halo::server
