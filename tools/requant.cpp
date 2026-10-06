// halo_requant: variant GGUFs for the CODESIGN section 5 sweep (simulated
// quantization the unchanged engine loads). Every tensor keeps its type and
// size, so the output shards are reflink copies of the input with the chosen
// tensors' bytes rewritten in place; headers, metadata and every other
// tensor are the original bytes.
//
//   halo_requant list --model <shard-00001.gguf>
//   halo_requant A --model <shard-00001.gguf> --out <dir> [--threads N]
//       Dense projections (every 2-D Q8_0 tensor but token_embd, routed
//       experts excluded) and the LM head: 4.125-bit randomized-Hadamard
//       code (lowbit::RotatedQ4Row), stored back as Q8_0.
//   halo_requant D --model <shard-00001.gguf> --out <dir> [--layers 12-23]
//       Routed experts of the layers: the exact 2-bit code (lowbit::Encode2Bit)
//       in their own containers.
//
// Memory: tensors stream through in chunks of rows (pread / pwrite, ~64 MB
// buffers); page cache of both files is dropped as each tensor finishes.
// Resumable: the output is built in <dir>.partial with progress.tsv, one line
// per finished (fdatasync'ed) tensor under a header binding the arm, its
// parameters and the source shards (size, mtime). A rerun with the same
// header skips finished tensors and redoes a partial one; a different header
// is refused. On completion manifest.tsv is written and <dir>.partial renamed
// to <dir>.

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/gguf.hpp"
#include "core/lowbit.hpp"
#include "core/quant.hpp"
#include "model/cpu_ops.hpp"
#include "model/weights.hpp"

using namespace halo;
namespace fs = std::filesystem;
namespace lowbit = halo::quant::lowbit;

namespace {

std::uint64_t Fnv(const std::uint8_t* p, std::size_t n, std::uint64_t h = 1469598103934665603ull) {
  for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

std::size_t RowBytes(core::GgmlType type, std::size_t cols) {
  return type == core::GgmlType::kQ5_1 ? cols / 32 * 24 : quant::QuantizedRowBytes(type, cols);
}

void Decode(core::GgmlType type, const std::uint8_t* row, std::size_t cols, float* out) {
  model::TensorRef t;
  t.data = row;
  t.type = type;
  t.cols = cols;
  model::cpu::DequantizeRow(t, 0, 0, out);
}

std::vector<fs::path> ShardPaths(const fs::path& first, std::size_t count) {
  if (count == 1) return {first};
  std::smatch m;
  const std::string name = first.filename().string();
  static const std::regex re{R"(^(.*)-([0-9]+)-of-([0-9]+)\.gguf$)"};
  if (!std::regex_match(name, m, re)) return {};
  std::vector<fs::path> out;
  for (std::size_t i = 0; i < count; ++i) {
    std::ostringstream s;
    s << m[1] << '-' << std::setw(static_cast<int>(m[2].length())) << std::setfill('0') << (i + 1) << "-of-" << m[3]
      << ".gguf";
    out.push_back(first.parent_path() / s.str());
  }
  return out;
}

bool CloneFile(const fs::path& src, const fs::path& dst, std::string* error) {
  const int in = open(src.c_str(), O_RDONLY);
  const int out = open(dst.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  bool ok = in >= 0 && out >= 0 && ioctl(out, FICLONE, in) == 0;
  if (!ok && in >= 0 && out >= 0) {  // no reflink: an in-kernel copy
    struct stat st{};
    fstat(in, &st);
    off_t left = st.st_size;
    ok = true;
    while (ok && left > 0) {
      const ssize_t n = copy_file_range(in, nullptr, out, nullptr, static_cast<std::size_t>(left), 0);
      ok = n > 0;
      left -= n;
    }
  }
  if (!ok) *error = "cannot copy " + src.string() + " to " + dst.string() + ": " + std::strerror(errno);
  if (in >= 0) close(in);
  if (out >= 0) {
    ok = ok && fsync(out) == 0;
    close(out);
  }
  return ok;
}

struct Job {
  const core::GgufTensorInfo* info;
  std::uint32_t shard;
  std::uint64_t offset;  // payload offset in the shard
  std::size_t cols, rows, row_bytes;
};

struct Stats {
  double ref2 = 0, code2 = 0, cont2 = 0, tot2 = 0;  // sum w^2; code err; container err; total err
  Stats& operator+=(const Stats& o) {
    ref2 += o.ref2;
    code2 += o.code2;
    cont2 += o.cont2;
    tot2 += o.tot2;
    return *this;
  }
};

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc < 2) {
    std::fprintf(stderr, "usage: halo_requant list|A|D --model <shard-00001.gguf> [--out dir] [--layers a-b] [--threads N]\n");
    return 2;
  }
  const std::string arm = argv[1];
  std::string model, out_dir, layers = "12-23";
  unsigned threads = 12;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--model") model = argv[i + 1];
    else if (a == "--out") out_dir = argv[i + 1];
    else if (a == "--layers") layers = argv[i + 1];
    else if (a == "--threads") threads = static_cast<unsigned>(std::max(1, std::atoi(argv[i + 1])));
    else return std::fprintf(stderr, "unknown option %s\n", a.c_str()), 2;
  }
  if (model.empty() || (arm != "list" && arm != "A" && arm != "D") || (arm != "list" && out_dir.empty()))
    return std::fprintf(stderr, "need list|A|D, --model, and --out for A and D\n"), 2;
  std::string error;
  auto reader = core::GgufReader::OpenFile(model, &error);
  if (!reader) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
  const auto regions = reader->GetMappedRegions();
  const auto shards = ShardPaths(model, regions.size());
  if (shards.size() != regions.size()) return std::fprintf(stderr, "cannot name the shards of %s\n", model.c_str()), 1;

  int lo = 12, hi = 23;
  if (std::sscanf(layers.c_str(), "%d-%d", &lo, &hi) != 2) return std::fprintf(stderr, "--layers a-b\n"), 2;
  std::vector<Job> jobs;
  for (const auto& t : reader->GetTensors()) {
    const auto addr = reinterpret_cast<std::uintptr_t>(t.data);
    Job j{&t, 0, 0, t.dimensions.front(), 1, 0};
    for (std::size_t d = 1; d < t.dimensions.size(); ++d) j.rows *= t.dimensions[d];
    j.row_bytes = RowBytes(t.type, j.cols);
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (addr >= base && addr < base + regions[i].size) {
        j.shard = i;
        j.offset = addr - base;
      }
    }
    const std::string name(t.name);
    if (arm == "list") {
      std::printf("%-40s %-6s shard %u  offset %12llu  cols %6zu rows %9zu\n", name.c_str(),
                  std::string(core::ToString(t.type)).c_str(), j.shard, static_cast<unsigned long long>(j.offset),
                  j.cols, j.rows);
      continue;
    }
    bool take = false;
    if (arm == "A") {
      take = t.type == core::GgmlType::kQ8_0 && t.dimensions.size() == 2 && name != "token_embd.weight" &&
             name.find("_exps") == std::string::npos && j.cols % 32 == 0 && lowbit::HadamardBlock(j.cols) >= 32;
    } else {
      // Exactly blk.<il>.ffn_{gate,up,down}_exps.weight (not the shared
      // expert, not the router).
      for (int il = lo; il <= hi && !take; ++il)
        for (const char* kind : {"gate", "up", "down"})
          take = take || name == "blk." + std::to_string(il) + ".ffn_" + kind + "_exps.weight";
      if (take && j.cols % (t.type == core::GgmlType::kQ4_K || t.type == core::GgmlType::kQ5_K ? 256 : 32) != 0)
        return std::fprintf(stderr, "%s: cols %zu do not fit %s\n", name.c_str(), j.cols,
                            std::string(core::ToString(t.type)).c_str()), 1;
    }
    if (take && j.row_bytes == 0)
      return std::fprintf(stderr, "%s: no row size for %s\n", name.c_str(), std::string(core::ToString(t.type)).c_str()), 1;
    if (take) jobs.push_back(j);
  }
  if (arm == "list") return 0;
  if (jobs.empty()) return std::fprintf(stderr, "no tensors selected\n"), 1;

  // The header binds the run: arm, parameters, sources.
  std::ostringstream hdr;
  hdr << "# halo_requant arm " << arm;
  if (arm == "A") hdr << " code iq4nl-g=hadamard-block signs=fnv(name) container Q8_0";
  else hdr << " layers " << lo << "-" << hi << " code 2bit-6bit-scales";
  for (const auto& s : shards) {
    struct stat st{};
    if (stat(s.c_str(), &st) != 0) return std::fprintf(stderr, "cannot stat %s\n", s.c_str()), 1;
    hdr << " | " << s.filename().string() << " " << st.st_size << " " << st.st_mtim.tv_sec;
  }
  const std::string header = hdr.str();
  const fs::path final_dir = out_dir, partial = out_dir + ".partial";
  if (fs::exists(final_dir)) return std::fprintf(stderr, "%s exists\n", final_dir.c_str()), 1;
  fs::create_directories(partial);
  const fs::path progress = partial / "progress.tsv";
  std::set<std::string> done;
  std::vector<std::string> done_lines;
  {
    std::ifstream f(progress);
    std::string line;
    if (std::getline(f, line)) {
      if (line != header) return std::fprintf(stderr, "%s belongs to a different run:\n  %s\n", partial.c_str(), line.c_str()), 1;
      while (std::getline(f, line))
        if (!line.empty()) {
          done.insert(line.substr(0, line.find('\t')));
          done_lines.push_back(line);
        }
    }
  }
  if (!fs::exists(progress)) {
    for (std::size_t i = 0; i < shards.size(); ++i) {
      std::printf("cloning %s\n", shards[i].filename().c_str());
      if (!CloneFile(shards[i], partial / shards[i].filename(), &error)) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    }
    std::ofstream f(progress);
    f << header << "\n";
    if (!f.flush()) return std::fprintf(stderr, "cannot write %s\n", progress.c_str()), 1;
  } else {
    std::printf("resuming: %zu of %zu tensors done\n", done.size(), jobs.size());
  }
  std::vector<int> in_fd(shards.size()), out_fd(shards.size());
  for (std::size_t i = 0; i < shards.size(); ++i) {
    in_fd[i] = open(shards[i].c_str(), O_RDONLY);
    out_fd[i] = open((partial / shards[i].filename()).c_str(), O_WRONLY);
    if (in_fd[i] < 0 || out_fd[i] < 0) return std::fprintf(stderr, "cannot open shard %zu\n", i), 1;
  }
  const auto t0 = std::chrono::steady_clock::now();
  Stats grand;
  std::size_t index = 0;
  for (const auto& j : jobs) {
    ++index;
    const std::string name(j.info->name);
    if (done.contains(name)) continue;
    const auto type = j.info->type;
    std::vector<float> signs;
    if (arm == "A") {
      signs.resize(j.cols);
      lowbit::RotationSigns(Fnv(reinterpret_cast<const std::uint8_t*>(name.data()), name.size()), signs);
    }
    const std::size_t chunk_rows = std::max<std::size_t>(1, (64u << 20) / j.row_bytes);
    std::vector<std::uint8_t> in(chunk_rows * j.row_bytes), out(chunk_rows * j.row_bytes);
    Stats ts;
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t r0 = 0; r0 < j.rows; r0 += chunk_rows) {
      const std::size_t n = std::min(chunk_rows, j.rows - r0), bytes = n * j.row_bytes;
      const off_t at = static_cast<off_t>(j.offset + r0 * j.row_bytes);
      if (pread(in_fd[j.shard], in.data(), bytes, at) != static_cast<ssize_t>(bytes))
        return std::fprintf(stderr, "%s: short read\n", name.c_str()), 1;
      std::vector<Stats> part(threads);
      std::atomic<std::size_t> next{0};
      std::atomic<bool> bad{false};
      std::vector<std::thread> pool;
      for (unsigned w = 0; w < threads; ++w)
        pool.emplace_back([&, w] {
          std::vector<float> x(j.cols), q(j.cols), y(j.cols);
          Stats& s = part[w];
          for (std::size_t r; (r = next.fetch_add(1)) < n;) {
            const std::uint8_t* src = in.data() + r * j.row_bytes;
            std::uint8_t* dst = out.data() + r * j.row_bytes;
            Decode(type, src, j.cols, x.data());
            if (arm == "A") {
              lowbit::RotatedQ4Row(x.data(), j.cols, signs.data(), q.data());
              lowbit::EncodeQ8_0(q.data(), j.cols, dst);
            } else if (!lowbit::Encode2Bit(type, x.data(), j.cols, dst)) {
              bad = true;
              return;
            }
            Decode(type, dst, j.cols, y.data());
            for (std::size_t c = 0; c < j.cols; ++c) {
              const double w0 = x[c], code = arm == "A" ? q[c] : y[c], stored = y[c];
              s.ref2 += w0 * w0;
              s.code2 += (code - w0) * (code - w0);
              s.cont2 += (stored - code) * (stored - code);
              s.tot2 += (stored - w0) * (stored - w0);
            }
          }
        });
      for (auto& t : pool) t.join();
      if (bad) return std::fprintf(stderr, "%s: cannot encode %s\n", name.c_str(), std::string(core::ToString(type)).c_str()), 1;
      for (const auto& s : part) ts += s;
      hash = Fnv(out.data(), bytes, hash);
      if (pwrite(out_fd[j.shard], out.data(), bytes, at) != static_cast<ssize_t>(bytes))
        return std::fprintf(stderr, "%s: short write\n", name.c_str()), 1;
    }
    const off_t first = static_cast<off_t>(j.offset), len = static_cast<off_t>(j.rows * j.row_bytes);
    if (fdatasync(out_fd[j.shard]) != 0) return std::fprintf(stderr, "fdatasync: %s\n", std::strerror(errno)), 1;
    posix_fadvise(in_fd[j.shard], first, len, POSIX_FADV_DONTNEED);
    posix_fadvise(out_fd[j.shard], first, len, POSIX_FADV_DONTNEED);
    char line[512];
    std::snprintf(line, sizeof line, "%s\t%s\t%zu\t%zu\t%.6e\t%.6e\t%.6e\t%.6e\t%016llx", name.c_str(),
                  std::string(core::ToString(type)).c_str(), j.cols, j.rows, ts.ref2, ts.code2, ts.cont2, ts.tot2,
                  static_cast<unsigned long long>(hash));
    {
      std::ofstream f(progress, std::ios::app);
      f << line << "\n";
      f.flush();
      if (!f) return std::fprintf(stderr, "cannot append %s\n", progress.c_str()), 1;
      const int fd = open(progress.c_str(), O_WRONLY);
      if (fd >= 0) {
        fsync(fd);
        close(fd);
      }
    }
    done_lines.push_back(line);
    grand += ts;
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[%zu/%zu] %-36s %-5s rel err code %.4f  container +%.4f  total %.4f   (%.0f s)\n", index, jobs.size(),
                name.c_str(), std::string(core::ToString(type)).c_str(), std::sqrt(ts.code2 / ts.ref2),
                std::sqrt(ts.cont2 / ts.ref2), std::sqrt(ts.tot2 / ts.ref2), secs);
  }
  for (std::size_t i = 0; i < shards.size(); ++i) {
    if (fsync(out_fd[i]) != 0) return std::fprintf(stderr, "fsync shard %zu: %s\n", i, std::strerror(errno)), 1;
    close(out_fd[i]);
    close(in_fd[i]);
  }
  // Totals from the progress lines (covers resumed runs too).
  Stats all;
  std::map<std::string, Stats> by_class;
  for (const auto& l : done_lines) {
    std::istringstream s(l);
    std::string name, type;
    std::size_t cols, rows;
    Stats t;
    s >> name >> type >> cols >> rows >> t.ref2 >> t.code2 >> t.cont2 >> t.tot2;
    all += t;
    std::string cls = name;
    if (cls.starts_with("blk.")) cls = "blk.*" + cls.substr(cls.find('.', 4));
    by_class[cls] += t;
  }
  {
    std::ofstream f(partial / "manifest.tsv");
    f << header << "\n# name\ttype\tcols\trows\tsum_w2\tcode_err2\tcontainer_err2\ttotal_err2\tfnv64(bytes)\n";
    for (const auto& l : done_lines) f << l << "\n";
    f << "# class\trel_err_code\trel_err_container\trel_err_total\n";
    for (const auto& [cls, t] : by_class)
      f << "# " << cls << "\t" << std::sqrt(t.code2 / t.ref2) << "\t" << std::sqrt(t.cont2 / t.ref2) << "\t"
        << std::sqrt(t.tot2 / t.ref2) << "\n";
    f << "# all\t" << std::sqrt(all.code2 / all.ref2) << "\t" << std::sqrt(all.cont2 / all.ref2) << "\t"
      << std::sqrt(all.tot2 / all.ref2) << "\tcontainer/code error energy " << all.cont2 / all.code2 << "\n";
    if (!f.flush()) return std::fprintf(stderr, "cannot write manifest\n"), 1;
  }
  std::printf("all %zu tensors: rel err code %.4f, container +%.4f, total %.4f; container adds %.3f%% of the code's error energy\n",
              done_lines.size(), std::sqrt(all.code2 / all.ref2), std::sqrt(all.cont2 / all.ref2),
              std::sqrt(all.tot2 / all.ref2), 100.0 * all.cont2 / all.code2);
  for (const auto& [cls, t] : by_class)
    std::printf("  %-36s code %.4f  container +%.4f  total %.4f\n", cls.c_str(), std::sqrt(t.code2 / t.ref2),
                std::sqrt(t.cont2 / t.ref2), std::sqrt(t.tot2 / t.ref2));
  fs::rename(partial, final_dir);
  std::printf("wrote %s\n", final_dir.c_str());
}
