#ifndef HALO_SCOPE_WAL_HPP_
#define HALO_SCOPE_WAL_HPP_

// The writer's write-ahead log: each batch is appended as a CRC-framed
// record before it is inserted, and the log is reset once the insert has
// an outcome. After a crash, the frames still in the log are re-inserted
// with the same insert_deduplication_token, which is derived from the
// frame's position (WAL id + sequence number), so a batch that did reach
// chDB before the crash is not written twice.
//
// File: header [8 magic "HSCPWAL1"][u64 wal_id][u64 next_seq], then frames
// [u32 0x46524D31][u64 seq][u8 table][u32 rows][u32 len][u32 crc32][len
// bytes]; the CRC covers seq..len and the body. Replay stops at the first
// short or corrupt frame (a torn tail: that batch was never inserted).

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace halo::scope {

struct WalFrame {
  std::uint64_t seq = 0;
  std::uint8_t table = 0;
  std::uint32_t rows = 0;
  std::string body;
};

class Wal {
 public:
  /// Opens or creates the log at `path`. Frames already in it are returned
  /// by Pending() for replay. Returns false with `*err` on failure.
  bool Open(const std::string& path, bool sync, std::string* err);
  ~Wal();

  const std::vector<WalFrame>& Pending() const { return pending_; }
  /// Appends one frame and returns its sequence number (0 on error).
  std::uint64_t Append(std::uint8_t table, std::uint32_t rows, std::string_view body, std::string* err);
  /// Drops every frame (all have an outcome); sequence numbers continue.
  bool Reset(std::string* err);

  /// The insert_deduplication_token for frame `seq` of this log.
  std::string Token(std::uint64_t seq) const;
  std::uint64_t id() const { return id_; }

  /// Reads a log without opening it for writing (tests, tools).
  static bool Read(const std::string& path, std::uint64_t* id, std::vector<WalFrame>* frames, std::string* err);

 private:
  int fd_ = -1;
  bool sync_ = false;
  std::uint64_t id_ = 0;
  std::uint64_t next_seq_ = 1;
  std::uint64_t end_ = 0;  // file offset of the next frame
  std::vector<WalFrame> pending_;
};

std::uint32_t Crc32(std::string_view data, std::uint32_t crc = 0);

}  // namespace halo::scope

#endif  // HALO_SCOPE_WAL_HPP_
