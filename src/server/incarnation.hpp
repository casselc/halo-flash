#ifndef HALO_SERVER_INCARNATION_HPP_
#define HALO_SERVER_INCARNATION_HPP_

// The executor's incarnation and clock (docs/OPERATIONS.md section 2; cordis RFC 0007).
//
// An incarnation is 128 random bits chosen when the process starts: halo persists nothing an operation depends on,
// so a restart is a new incarnation that holds and remembers nothing, and every answer names the one that gave it.
// Its clock is the monotonic time since the incarnation started, in milliseconds (`Halo-Time`): deadlines are
// written in it and only halo compares them, so no two hosts' clocks are ever compared, and it never runs backward.

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace halo::server {

class Incarnation {
 public:
  using Clock = std::chrono::steady_clock;

  /// A new incarnation: 128 bits from getrandom(2), its clock starting now. Throws if no randomness is available.
  Incarnation();
  /// Test seam: a given id and clock origin.
  Incarnation(std::string id, Clock::time_point start) : id_(std::move(id)), start_(start) {}

  /// 32 lowercase hex digits.
  [[nodiscard]] const std::string& id() const noexcept { return id_; }
  /// Milliseconds on this incarnation's clock.
  [[nodiscard]] std::int64_t NowMs() const;
  /// Adds Halo-Incarnation and Halo-Time to an answer's headers (HttpServerOptions::stamp).
  void Stamp(std::vector<std::pair<std::string, std::string>>& headers) const;

 private:
  std::string id_;
  Clock::time_point start_;
};

}  // namespace halo::server

#endif  // HALO_SERVER_INCARNATION_HPP_
