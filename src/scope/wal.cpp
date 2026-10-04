#include "scope/wal.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <random>

namespace halo::scope {

namespace {

constexpr char kMagic[8] = {'H', 'S', 'C', 'P', 'W', 'A', 'L', '1'};
constexpr std::uint32_t kFrameMagic = 0x46524D31;
constexpr std::size_t kHeader = 24;
constexpr std::size_t kFrameHeader = 4 + 8 + 1 + 4 + 4 + 4;

template <typename T>
void Put(std::string& out, T v) {
  char b[sizeof(T)];
  std::memcpy(b, &v, sizeof(T));
  out.append(b, sizeof(T));
}

template <typename T>
T Get(const char* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

bool ReadAll(int fd, std::string* out) {
  out->clear();
  char buf[1 << 16];
  for (off_t off = 0;;) {
    const ssize_t n = pread(fd, buf, sizeof buf, off);
    if (n < 0) return false;
    if (n == 0) return true;
    out->append(buf, static_cast<std::size_t>(n));
    off += n;
  }
}

bool WriteAll(int fd, const std::string& s, off_t off) {
  for (std::size_t done = 0; done < s.size();) {
    const ssize_t n = pwrite(fd, s.data() + done, s.size() - done, off + static_cast<off_t>(done));
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    done += static_cast<std::size_t>(n);
  }
  return true;
}

// Parses a whole log image. Returns the offset just past the last good frame.
std::size_t Parse(const std::string& img, std::uint64_t* id, std::uint64_t* next_seq, std::vector<WalFrame>* frames) {
  *id = Get<std::uint64_t>(img.data() + 8);
  *next_seq = Get<std::uint64_t>(img.data() + 16);
  std::size_t p = kHeader;
  while (img.size() - p >= kFrameHeader) {
    const char* h = img.data() + p;
    if (Get<std::uint32_t>(h) != kFrameMagic) break;
    const auto len = Get<std::uint32_t>(h + 17);
    if (img.size() - p - kFrameHeader < len) break;
    const std::string_view covered(h + 4, 8 + 1 + 4 + 4);
    const std::string_view body(h + kFrameHeader, len);
    if (Crc32(body, Crc32(covered)) != Get<std::uint32_t>(h + 21)) break;
    WalFrame f;
    f.seq = Get<std::uint64_t>(h + 4);
    f.table = static_cast<std::uint8_t>(h[12]);
    f.rows = Get<std::uint32_t>(h + 13);
    f.body.assign(body);
    if (f.seq >= *next_seq) *next_seq = f.seq + 1;
    frames->push_back(std::move(f));
    p += kFrameHeader + len;
  }
  return p;
}

}  // namespace

std::uint32_t Crc32(std::string_view data, std::uint32_t crc) {
  static const auto table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      t[i] = c;
    }
    return t;
  }();
  crc = ~crc;
  for (unsigned char ch : data) crc = table[(crc ^ ch) & 0xff] ^ (crc >> 8);
  return ~crc;
}

bool Wal::Read(const std::string& path, std::uint64_t* id, std::vector<WalFrame>* frames, std::string* err) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    *err = path + ": " + std::strerror(errno);
    return false;
  }
  std::string img;
  const bool ok = ReadAll(fd, &img);
  ::close(fd);
  if (!ok || img.size() < kHeader || std::memcmp(img.data(), kMagic, 8) != 0) {
    *err = path + ": not a halo-scope WAL";
    return false;
  }
  std::uint64_t next;
  Parse(img, id, &next, frames);
  return true;
}

bool Wal::Open(const std::string& path, bool sync, std::string* err) {
  sync_ = sync;
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd_ < 0) {
    *err = path + ": " + std::strerror(errno);
    return false;
  }
  std::string img;
  if (!ReadAll(fd_, &img)) {
    *err = path + ": read failed";
    return false;
  }
  if (img.size() >= kHeader && std::memcmp(img.data(), kMagic, 8) == 0) {
    end_ = Parse(img, &id_, &next_seq_, &pending_);
    // A torn tail is cut off so new frames follow the last good one.
    if (end_ < img.size() && ftruncate(fd_, static_cast<off_t>(end_)) != 0) {
      *err = path + ": truncate failed";
      return false;
    }
    return true;
  }
  if (!img.empty()) {
    *err = path + ": not a halo-scope WAL";
    return false;
  }
  // A new log gets a random id, so its tokens never collide with another
  // log's (a deleted WAL restarting at seq 1 must not be deduplicated away).
  std::random_device rd;
  id_ = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
  next_seq_ = 1;
  return Reset(err);
}

Wal::~Wal() {
  if (fd_ >= 0) ::close(fd_);
}

std::uint64_t Wal::Append(std::uint8_t table, std::uint32_t rows, std::string_view body, std::string* err) {
  const std::uint64_t seq = next_seq_;
  std::string covered;
  Put<std::uint64_t>(covered, seq);
  covered.push_back(static_cast<char>(table));
  Put<std::uint32_t>(covered, rows);
  Put<std::uint32_t>(covered, static_cast<std::uint32_t>(body.size()));
  std::string frame;
  frame.reserve(kFrameHeader + body.size());
  Put<std::uint32_t>(frame, kFrameMagic);
  frame += covered;
  Put<std::uint32_t>(frame, Crc32(body, Crc32(covered)));
  frame.append(body);
  if (!WriteAll(fd_, frame, static_cast<off_t>(end_)) || (sync_ && fdatasync(fd_) != 0)) {
    *err = std::string("wal append: ") + std::strerror(errno);
    return 0;
  }
  end_ += frame.size();
  ++next_seq_;
  return seq;
}

bool Wal::Reset(std::string* err) {
  std::string hdr(kMagic, 8);
  Put<std::uint64_t>(hdr, id_);
  Put<std::uint64_t>(hdr, next_seq_);
  // Header first (it carries next_seq), then cut the frames.
  if (!WriteAll(fd_, hdr, 0) || ftruncate(fd_, static_cast<off_t>(kHeader)) != 0 ||
      (sync_ && fdatasync(fd_) != 0)) {
    *err = std::string("wal reset: ") + std::strerror(errno);
    return false;
  }
  end_ = kHeader;
  pending_.clear();
  return true;
}

std::string Wal::Token(std::uint64_t seq) const {
  char buf[48];
  std::snprintf(buf, sizeof buf, "wal-%016llx-%llu", static_cast<unsigned long long>(id_),
                static_cast<unsigned long long>(seq));
  return buf;
}

}  // namespace halo::scope
