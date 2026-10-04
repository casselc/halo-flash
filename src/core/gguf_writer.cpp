#include "core/gguf_writer.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <set>
#include <type_traits>

#include "core/quant.hpp"

namespace halo::core {

namespace {

class Out {
 public:
  template <typename T>
  void Pod(T v) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    buf_.insert(buf_.end(), p, p + sizeof(T));
  }
  void Str(const std::string& s) {
    Pod<std::uint64_t>(s.size());
    buf_.insert(buf_.end(), s.begin(), s.end());
  }
  void Bytes(std::span<const std::uint8_t> b) {
    buf_.insert(buf_.end(), b.begin(), b.end());
  }
  void PadTo(std::size_t alignment) {
    buf_.resize((buf_.size() + alignment - 1) / alignment * alignment, 0);
  }
  std::vector<std::uint8_t>& buf() { return buf_; }

 private:
  std::vector<std::uint8_t> buf_;
};

template <typename T>
constexpr GgufValueType ScalarType() {
  if constexpr (std::is_same_v<T, std::uint32_t>) return GgufValueType::kUint32;
  if constexpr (std::is_same_v<T, std::int32_t>) return GgufValueType::kInt32;
  if constexpr (std::is_same_v<T, std::uint64_t>) return GgufValueType::kUint64;
  if constexpr (std::is_same_v<T, std::int64_t>) return GgufValueType::kInt64;
  if constexpr (std::is_same_v<T, float>) return GgufValueType::kFloat32;
  if constexpr (std::is_same_v<T, double>) return GgufValueType::kFloat64;
}

void WriteValue(Out& out, const GgufWriter::Value& value) {
  std::visit(
      [&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) {
          out.Pod<std::uint32_t>(static_cast<std::uint32_t>(GgufValueType::kBool));
          out.Pod<std::uint8_t>(v ? 1 : 0);
        } else if constexpr (std::is_same_v<T, std::string>) {
          out.Pod<std::uint32_t>(static_cast<std::uint32_t>(GgufValueType::kString));
          out.Str(v);
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          out.Pod<std::uint32_t>(static_cast<std::uint32_t>(GgufValueType::kArray));
          out.Pod<std::uint32_t>(static_cast<std::uint32_t>(GgufValueType::kString));
          out.Pod<std::uint64_t>(v.size());
          for (const auto& s : v) out.Str(s);
        } else if constexpr (std::is_arithmetic_v<T>) {
          out.Pod<std::uint32_t>(static_cast<std::uint32_t>(ScalarType<T>()));
          out.Pod<T>(v);
        } else {
          using E = typename T::value_type;
          out.Pod<std::uint32_t>(static_cast<std::uint32_t>(GgufValueType::kArray));
          out.Pod<std::uint32_t>(static_cast<std::uint32_t>(ScalarType<E>()));
          out.Pod<std::uint64_t>(v.size());
          for (const auto& e : v) out.Pod<E>(e);
        }
      },
      value);
}

}  // namespace

void GgufWriter::Set(std::string key, Value value) {
  metadata_.emplace_back(std::move(key), std::move(value));
}

void GgufWriter::AddTensor(std::string name, std::vector<std::uint64_t> dims,
                           GgmlType type, std::span<const std::uint8_t> bytes) {
  tensors_.push_back(
      {std::move(name), std::move(dims), type, {bytes.begin(), bytes.end()}});
}

std::vector<std::uint8_t> GgufWriter::Serialize(std::string* error) const {
  const auto fail = [&](std::string msg) {
    if (error) *error = std::move(msg);
    return std::vector<std::uint8_t>{};
  };
  if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0) {
    return fail("alignment must be a power of two");
  }
  std::set<std::string_view> keys;
  for (const auto& [k, _] : metadata_) {
    if (k.empty() || !keys.insert(k).second) return fail("bad or duplicate key: " + k);
    if (k == "general.alignment") return fail("general.alignment is set by the writer");
  }
  std::set<std::string_view> names;
  for (const auto& t : tensors_) {
    if (t.name.empty() || !names.insert(t.name).second) {
      return fail("bad or duplicate tensor: " + t.name);
    }
    if (t.dims.empty() || t.dims.size() > 8) return fail("bad rank: " + t.name);
    std::uint64_t elements = 1;
    for (auto d : t.dims) elements *= d;
    if (elements == 0 || quant::EncodedSizeBytes(t.type, elements) != t.bytes.size()) {
      return fail("size mismatch: " + t.name);
    }
  }

  Out out;
  out.Bytes(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>("GGUF"), 4));
  out.Pod<std::uint32_t>(3);
  out.Pod<std::uint64_t>(tensors_.size());
  const bool emit_alignment = alignment_ != 32;
  out.Pod<std::uint64_t>(metadata_.size() + (emit_alignment ? 1 : 0));
  if (emit_alignment) {
    out.Str("general.alignment");
    WriteValue(out, alignment_);
  }
  for (const auto& [k, v] : metadata_) {
    out.Str(k);
    WriteValue(out, v);
  }
  std::uint64_t offset = 0;
  for (const auto& t : tensors_) {
    out.Str(t.name);
    out.Pod<std::uint32_t>(static_cast<std::uint32_t>(t.dims.size()));
    for (auto d : t.dims) out.Pod<std::uint64_t>(d);
    out.Pod<std::uint32_t>(static_cast<std::uint32_t>(t.type));
    out.Pod<std::uint64_t>(offset);
    offset += (t.bytes.size() + alignment_ - 1) / alignment_ * alignment_;
  }
  out.PadTo(alignment_);
  for (const auto& t : tensors_) {
    out.Bytes(t.bytes);
    out.PadTo(alignment_);
  }
  return std::move(out.buf());
}

bool GgufWriter::WriteFile(const std::filesystem::path& path,
                           std::string* error) const {
  const auto image = Serialize(error);
  if (image.empty()) return false;
  const auto tmp = path.string() + ".tmp";
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) {
    if (error) *error = "cannot create " + tmp;
    return false;
  }
  std::size_t done = 0;
  while (done < image.size()) {
    const auto n = ::write(fd, image.data() + done, image.size() - done);
    if (n <= 0) {
      ::close(fd);
      ::unlink(tmp.c_str());
      if (error) *error = "write failed: " + tmp;
      return false;
    }
    done += static_cast<std::size_t>(n);
  }
  if (::fsync(fd) != 0 || ::close(fd) != 0 ||
      ::rename(tmp.c_str(), path.c_str()) != 0) {
    ::unlink(tmp.c_str());
    if (error) *error = "commit failed: " + path.string();
    return false;
  }
  return true;
}

}  // namespace halo::core
