// SR4: a session's state spilled to a file and restored from it (docs/SESSIONS.md section 10).
//
// File, little-endian. Version 2 (encrypted, the default):
//   [0, 512)          header (SpillHeaderV2): identity (model fingerprint, numerics options, KV kind, index window,
//                     position, layer count), sizes, the file salt, the key id, the wrapped data key, the meta tag.
//                     Bytes [0, 448) are the AAD of the DEK wrap, the meta and every payload piece.
//   [512, 4096 * k)   meta, AES-256-GCM sealed (nonce salt || 0xffffffff): per-layer QSA block counts, n-gram
//                     window, the task table (file offset, bytes, section, GCM tag of each payload piece), the
//                     caller's blob; padded to 4096.
//   payload           the sections of SpillLayout in order, each starting on a 4096-byte boundary, cut into pieces
//                     of at most 16 MB, each AES-256-GCM sealed in place (nonce salt || be32(piece index)).
//   The data key (DEK) is fresh and random per file and stored only wrapped (AES-256-GCM) under the caller's KEK
//   (core/spill_crypt.hpp: KEK = HKDF(master or session key, session, epoch)).
// Version 1 (plaintext, written only when the caller chose plaintext; read only with accept_v1): a 96-byte header
// (SpillHeader) with its own CRC32C and the meta's, meta in the clear, CRC32C per payload piece.
//
// Restore checks the header, the key id and (v2) the DEK unwrap and the meta tag, then the identity (model
// fingerprint, numerics options, KV kind, index window, layer count), the position against max_context and the whole
// task table against the layout this engine computes for that position before touching the session; each piece's
// tag (or checksum) is checked as it is read, before its bytes go to the device.
//
// Write protocol: <path>.tmp.<pid>.<n> (O_EXCL, 0600) -> payload pieces (parallel, O_DIRECT) -> header+meta ->
// fsync -> rename onto <path> -> fsync of the directory. A crash leaves the old file or none at <path> plus maybe a
// temp file, which no reader opens.

#include <dirent.h>
#include <fcntl.h>
#include <nmmintrin.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <thread>

#include "core/spill_crypt.hpp"
#include "engine/flash_next_gpu.hpp"

namespace halo::engine {

namespace {

namespace crypt = core::crypt;

constexpr char kMagic1[8] = {'H', 'A', 'L', 'O', 'S', 'P', 'L', '1'};
constexpr char kMagic2[8] = {'H', 'A', 'L', 'O', 'S', 'P', 'L', '2'};
constexpr std::size_t kAlign = 4096;

struct SpillHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_bytes;  // header + meta, padded to kAlign: where the payload starts
  std::uint64_t model_fp;
  std::uint64_t options_fp;
  std::uint32_t kv_kind;  // 0: default kinds (fp32 K/V), 1: fp16-only K/V
  std::uint32_t index_window;
  std::uint32_t position;
  std::uint32_t num_layers;
  std::uint32_t n_tasks;
  std::uint32_t mtp;  // 1: the draft block's sections follow the trunk's
  std::uint32_t mtp_pos;
  std::uint32_t mtp_blocks;
  std::uint32_t ngram_n;
  std::uint32_t blob_bytes;
  std::uint64_t file_bytes;
  std::uint32_t meta_bytes;
  std::uint32_t meta_crc;
  std::uint32_t piece_bytes;  // the I/O and checksum unit the writer cut sections into
  std::uint32_t header_crc;  // over the header with this field 0
};
static_assert(sizeof(SpillHeader) == 96);  // no padding: offsets are the format (docs/SESSIONS.md section 10)

// Version 2. The identity fields keep v1's offsets (0..83).
struct SpillHeaderV2 {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint64_t model_fp;
  std::uint64_t options_fp;
  std::uint32_t kv_kind;
  std::uint32_t index_window;
  std::uint32_t position;
  std::uint32_t num_layers;
  std::uint32_t n_tasks;
  std::uint32_t mtp;
  std::uint32_t mtp_pos;
  std::uint32_t mtp_blocks;
  std::uint32_t ngram_n;
  std::uint32_t blob_bytes;
  std::uint64_t file_bytes;
  std::uint32_t meta_bytes;
  std::uint32_t piece_bytes;
  std::uint8_t salt[crypt::kSaltBytes];  // random per file: the nonce prefix
  std::uint32_t key_id_bytes;
  std::uint32_t reserved0;
  char key_id[crypt::kKeyIdBytes];  // the KEK's id (printable): provider, master name, epoch, session
  std::uint8_t wrap_nonce[crypt::kNonceBytes];
  std::uint8_t checkpoint[crypt::kCheckpointBytes];  // random per file: which checkpoint (the caller keeps it)
  std::uint8_t reserved[188];  // zero (authenticated)
  // ---- end of the AAD (crypt::kAadBytes) ----
  std::uint8_t wrapped_dek[crypt::kKeyBytes];
  std::uint8_t wrap_tag[crypt::kTagBytes];
  std::uint8_t meta_tag[crypt::kTagBytes];
};
static_assert(sizeof(SpillHeaderV2) == crypt::kHeaderBytes);
static_assert(offsetof(SpillHeaderV2, position) == 40 && offsetof(SpillHeaderV2, model_fp) == 16);
static_assert(offsetof(SpillHeaderV2, wrap_nonce) == crypt::kWrapNonceOff);
static_assert(offsetof(SpillHeaderV2, checkpoint) == crypt::kCheckpointOff);
static_assert(offsetof(SpillHeaderV2, wrapped_dek) == crypt::kWrappedDekOff && crypt::kWrappedDekOff == crypt::kAadBytes);
static_assert(offsetof(SpillHeaderV2, wrap_tag) == crypt::kWrapTagOff);
static_assert(offsetof(SpillHeaderV2, meta_tag) == crypt::kMetaTagOff);

// The identity and size fields both versions carry.
struct Fields {
  std::uint32_t version = 2;
  std::uint32_t header_bytes = 0;
  std::uint64_t model_fp = 0, options_fp = 0;
  std::uint32_t kv_kind = 0, index_window = 0, position = 0, num_layers = 0, n_tasks = 0, mtp = 0, mtp_pos = 0,
                mtp_blocks = 0, ngram_n = 0, blob_bytes = 0;
  std::uint64_t file_bytes = 0;
  std::uint32_t meta_bytes = 0, piece_bytes = 0;
};

struct Task {
  std::uint64_t file_off;
  std::uint64_t bytes;
  std::uint32_t crc;      // v1
  std::uint32_t section;
  std::uint8_t tag[crypt::kTagBytes];  // v2
};
constexpr std::size_t kTaskBytesV1 = 24, kTaskBytesV2 = 40;

__attribute__((target("sse4.2"))) std::uint32_t Crc32c(const void* data, std::size_t n, std::uint32_t crc = 0) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  std::uint64_t c = static_cast<std::uint32_t>(~crc);
  while (n >= 8) {
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    c = _mm_crc32_u64(c, v);
    p += 8;
    n -= 8;
  }
  auto c32 = static_cast<std::uint32_t>(c);
  while (n--) c32 = _mm_crc32_u8(c32, *p++);
  return ~c32;
}

std::size_t AlignUp(std::size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

bool Fail(std::string* error, std::string what) {
  if (error) *error = std::move(what);
  return false;
}

bool PwriteAll(int fd, const void* p, std::size_t n, std::uint64_t off) {
  std::size_t done = 0;
  while (done < n) {
    const ssize_t w = ::pwrite(fd, static_cast<const char*>(p) + done, n - done, static_cast<off_t>(off + done));
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) return false;
    done += static_cast<std::size_t>(w);
  }
  return true;
}

bool PreadAll(int fd, void* p, std::size_t n, std::uint64_t off) {
  std::size_t done = 0;
  while (done < n) {
    const ssize_t r = ::pread(fd, static_cast<char*>(p) + done, n - done, static_cast<off_t>(off + done));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    done += static_cast<std::size_t>(r);
  }
  return true;
}

// The task table of a layout: sections in order, each from an aligned offset, cut into kPiece pieces.
std::vector<Task> Tasks(std::span<const std::size_t> section_bytes, std::uint64_t payload_start, std::size_t kPiece) {
  std::vector<Task> t;
  std::uint64_t off = payload_start;
  for (std::size_t s = 0; s < section_bytes.size(); ++s) {
    for (std::size_t done = 0; done < section_bytes[s]; done += kPiece)
      t.push_back({off + done, std::min(kPiece, section_bytes[s] - done), 0, static_cast<std::uint32_t>(s), {}});
    off += AlignUp(section_bytes[s]);
  }
  return t;
}

struct Meta {
  Fields f;
  std::vector<std::uint32_t> blocks;
  std::vector<std::int32_t> ngram;
  std::vector<Task> tasks;
  std::vector<std::uint8_t> blob;
  // v2
  std::uint8_t salt[crypt::kSaltBytes] = {};
  std::uint8_t checkpoint[crypt::kCheckpointBytes] = {};
  std::string key_id;
  alignas(8) std::uint8_t raw[crypt::kHeaderBytes] = {};  // the header as read (its first kAadBytes: the AAD)
};

std::size_t HeadBytes(std::uint32_t version) { return version == 1 ? sizeof(SpillHeader) : crypt::kHeaderBytes; }

std::size_t MetaBytes(const Meta& m) {
  return m.blocks.size() * 4 + m.ngram.size() * 4 +
         m.tasks.size() * (m.f.version == 1 ? kTaskBytesV1 : kTaskBytesV2) + m.blob.size();
}

void PutMeta(const Meta& m, std::uint8_t* out) {
  std::size_t at = 0;
  const auto put = [&](const void* p, std::size_t n) {
    if (n) std::memcpy(out + at, p, n);
    at += n;
  };
  put(m.blocks.data(), m.blocks.size() * 4);
  put(m.ngram.data(), m.ngram.size() * 4);
  for (const auto& t : m.tasks) {
    put(&t.file_off, 8);
    put(&t.bytes, 8);
    if (m.f.version == 1) {
      put(&t.crc, 4);
      put(&t.section, 4);
    } else {
      put(&t.section, 4);
      const std::uint32_t zero = 0;
      put(&zero, 4);
      put(t.tag, crypt::kTagBytes);
    }
  }
  put(m.blob.data(), m.blob.size());
}

bool GetMeta(const std::uint8_t* in, std::size_t n, Meta* m) {
  std::size_t at = 0;
  const auto get = [&](void* p, std::size_t k) {
    if (at + k > n) return false;
    if (k) std::memcpy(p, in + at, k);
    at += k;
    return true;
  };
  m->blocks.resize(m->f.num_layers);
  m->ngram.resize(m->f.ngram_n);
  m->tasks.assign(m->f.n_tasks, Task{});
  m->blob.resize(m->f.blob_bytes);
  if (!get(m->blocks.data(), m->blocks.size() * 4) || !get(m->ngram.data(), m->ngram.size() * 4)) return false;
  for (auto& t : m->tasks) {
    if (!get(&t.file_off, 8) || !get(&t.bytes, 8)) return false;
    if (m->f.version == 1) {
      if (!get(&t.crc, 4) || !get(&t.section, 4)) return false;
    } else {
      std::uint32_t zero = 0;
      if (!get(&t.section, 4) || !get(&zero, 4) || !get(t.tag, crypt::kTagBytes)) return false;
      if (zero != 0) return false;
    }
  }
  return get(m->blob.data(), m->blob.size()) && at == n;
}

// v1: header + meta, the header CRC and meta CRC filled in.
std::vector<std::uint8_t> EncodeV1(Meta& m) {
  const std::size_t meta = MetaBytes(m);
  std::vector<std::uint8_t> out(AlignUp(sizeof(SpillHeader) + meta), 0);
  PutMeta(m, out.data() + sizeof(SpillHeader));
  m.f.header_bytes = static_cast<std::uint32_t>(out.size());
  m.f.meta_bytes = static_cast<std::uint32_t>(meta);
  SpillHeader h{};
  std::memcpy(h.magic, kMagic1, 8);
  const Fields& f = m.f;
  h.version = 1;
  h.header_bytes = f.header_bytes;
  h.model_fp = f.model_fp;
  h.options_fp = f.options_fp;
  h.kv_kind = f.kv_kind;
  h.index_window = f.index_window;
  h.position = f.position;
  h.num_layers = f.num_layers;
  h.n_tasks = f.n_tasks;
  h.mtp = f.mtp;
  h.mtp_pos = f.mtp_pos;
  h.mtp_blocks = f.mtp_blocks;
  h.ngram_n = f.ngram_n;
  h.blob_bytes = f.blob_bytes;
  h.file_bytes = f.file_bytes;
  h.meta_bytes = f.meta_bytes;
  h.piece_bytes = f.piece_bytes;
  h.meta_crc = Crc32c(out.data() + sizeof(SpillHeader), meta);
  h.header_crc = 0;
  h.header_crc = Crc32c(&h, sizeof(h));
  std::memcpy(out.data(), &h, sizeof(h));
  return out;
}

// v2: the header's AAD fields (everything but the wrapped key and the tags), from m.
void FillV2(const Meta& m, SpillHeaderV2* h) {
  std::memset(h, 0, sizeof(*h));
  std::memcpy(h->magic, kMagic2, 8);
  const Fields& f = m.f;
  h->version = 2;
  h->header_bytes = f.header_bytes;
  h->model_fp = f.model_fp;
  h->options_fp = f.options_fp;
  h->kv_kind = f.kv_kind;
  h->index_window = f.index_window;
  h->position = f.position;
  h->num_layers = f.num_layers;
  h->n_tasks = f.n_tasks;
  h->mtp = f.mtp;
  h->mtp_pos = f.mtp_pos;
  h->mtp_blocks = f.mtp_blocks;
  h->ngram_n = f.ngram_n;
  h->blob_bytes = f.blob_bytes;
  h->file_bytes = f.file_bytes;
  h->meta_bytes = f.meta_bytes;
  h->piece_bytes = f.piece_bytes;
  std::memcpy(h->salt, m.salt, sizeof(h->salt));
  std::memcpy(h->checkpoint, m.checkpoint, sizeof(h->checkpoint));
  h->key_id_bytes = static_cast<std::uint32_t>(m.key_id.size());
  std::memcpy(h->key_id, m.key_id.data(), m.key_id.size());
}

Fields FieldsOfV1(const SpillHeader& h) {
  return {1, h.header_bytes, h.model_fp, h.options_fp, h.kv_kind, h.index_window, h.position, h.num_layers,
          h.n_tasks, h.mtp, h.mtp_pos, h.mtp_blocks, h.ngram_n, h.blob_bytes, h.file_bytes, h.meta_bytes,
          h.piece_bytes};
}
Fields FieldsOfV2(const SpillHeaderV2& h) {
  return {2, h.header_bytes, h.model_fp, h.options_fp, h.kv_kind, h.index_window, h.position, h.num_layers,
          h.n_tasks, h.mtp, h.mtp_pos, h.mtp_blocks, h.ngram_n, h.blob_bytes, h.file_bytes, h.meta_bytes,
          h.piece_bytes};
}

// Sizes both versions must satisfy before any byte past the header is read.
bool CheckSizes(const Fields& f, std::uint64_t file_size, std::string* error) {
  const std::size_t head = HeadBytes(f.version);
  if (f.header_bytes % kAlign != 0 || f.header_bytes < head + std::uint64_t{f.meta_bytes} ||
      f.header_bytes > (64u << 20) || f.header_bytes > file_size)
    return Fail(error, "spill file header size invalid or truncated");
  if (f.file_bytes != file_size)
    return Fail(error, "spill file size " + std::to_string(file_size) + " != recorded " + std::to_string(f.file_bytes) +
                           " (truncated or extended)");
  const std::size_t want_meta = std::size_t{f.num_layers} * 4 + std::size_t{f.ngram_n} * 4 +
                                std::size_t{f.n_tasks} * (f.version == 1 ? kTaskBytesV1 : kTaskBytesV2) + f.blob_bytes;
  if (want_meta != f.meta_bytes) return Fail(error, "spill file meta size mismatch");
  return true;
}

// Reads and checks the header and meta: format and checksums (v1), key id, DEK unwrap and meta tag (v2). Identity
// is the caller's. On success (v2) `dek` holds the file's data key.
bool Decode(int fd, std::uint64_t file_size, const FlashNextGpu::SpillProtection& prot, Meta* m,
            crypt::SecretBytes* dek, std::string* error) {
  if (file_size < sizeof(SpillHeader)) return Fail(error, "spill file truncated (no header)");
  char magic[8];
  std::uint32_t version = 0;
  if (!PreadAll(fd, magic, 8, 0) || !PreadAll(fd, &version, 4, 8)) return Fail(error, "spill file: cannot read the header");
  if (std::memcmp(magic, kMagic1, 8) == 0) {
    if (prot.expect_version == 2)
      return Fail(error, "spill file is plaintext v1 but this checkpoint was written encrypted (v2): refused");
    if (!prot.accept_v1)
      return Fail(error, "spill file is plaintext format v1: refused (--spill-accept-v1 0)");
    SpillHeader h;
    if (!PreadAll(fd, &h, sizeof(h), 0)) return Fail(error, "spill file: cannot read the header");
    if (h.version != 1) return Fail(error, "spill file version " + std::to_string(h.version) + " unsupported");
    const std::uint32_t want = h.header_crc;
    h.header_crc = 0;
    if (Crc32c(&h, sizeof(h)) != want) return Fail(error, "spill file header checksum mismatch");
    m->f = FieldsOfV1(h);
    if (!CheckSizes(m->f, file_size, error)) return false;
    std::vector<std::uint8_t> meta(m->f.meta_bytes);
    if (!PreadAll(fd, meta.data(), meta.size(), sizeof(SpillHeader))) return Fail(error, "spill file: cannot read meta");
    if (Crc32c(meta.data(), meta.size()) != h.meta_crc) return Fail(error, "spill file meta checksum mismatch");
    if (!GetMeta(meta.data(), meta.size(), m)) return Fail(error, "spill file meta malformed");
    return true;
  }
  if (std::memcmp(magic, kMagic2, 8) != 0) return Fail(error, "not a spill file (magic)");
  if (version != 2) return Fail(error, "spill file version " + std::to_string(version) + " unsupported");
  if (file_size < crypt::kHeaderBytes) return Fail(error, "spill file truncated (no header)");
  std::uint8_t* raw = m->raw;
  if (!PreadAll(fd, raw, crypt::kHeaderBytes, 0)) return Fail(error, "spill file: cannot read the header");
  SpillHeaderV2 h;
  std::memcpy(&h, raw, sizeof(h));
  m->f = FieldsOfV2(h);
  if (h.key_id_bytes > crypt::kKeyIdBytes) return Fail(error, "spill file key id invalid");
  m->key_id.assign(h.key_id, h.key_id_bytes);
  std::memcpy(m->salt, h.salt, sizeof(m->salt));
  std::memcpy(m->checkpoint, h.checkpoint, sizeof(m->checkpoint));
  if (!CheckSizes(m->f, file_size, error)) return false;
  // The key: the file names its KEK; another id is refused before any decryption is tried (wrong session, epoch,
  // provider or master). The unwrap then authenticates the whole header (its AAD) under that KEK.
  if (!prot.kek) return Fail(error, "spill file is encrypted (key id " + m->key_id + ") and no spill key is configured");
  if (prot.kek->id != m->key_id)
    return Fail(error, "spill file is sealed under key id " + m->key_id + ", not " + prot.kek->id);
  if (!crypt::UnwrapDek(*prot.kek, std::span<const std::uint8_t, crypt::kHeaderBytes>(raw, crypt::kHeaderBytes), dek))
    return Fail(error, "spill file: the data key does not unwrap (wrong key, or the header was changed)");
  // Authentic from here. Which checkpoint: an older authentic file of the same session and key is refused.
  if (prot.expect_checkpoint &&
      !crypt::Equal(*prot.expect_checkpoint, std::span<const std::uint8_t>(m->checkpoint, crypt::kCheckpointBytes)))
    return Fail(error, "spill file is another checkpoint than the one recorded for this entry");
  std::vector<std::uint8_t> meta(m->f.meta_bytes);
  if (!PreadAll(fd, meta.data(), meta.size(), crypt::kHeaderBytes)) return Fail(error, "spill file: cannot read meta");
  crypt::Gcm gcm;
  if (!gcm.Open(dek->span(), crypt::PieceNonce(m->salt, crypt::kMetaIndex),
                crypt::HeaderAad(std::span<const std::uint8_t, crypt::kHeaderBytes>(raw, crypt::kHeaderBytes)), meta.data(), meta.size(),
                std::span<const std::uint8_t, crypt::kTagBytes>(h.meta_tag, crypt::kTagBytes))) {
    crypt::Wipe(meta.data(), meta.size());
    return Fail(error, "spill file meta authentication failed");
  }
  const bool ok = GetMeta(meta.data(), meta.size(), m);
  crypt::Wipe(meta.data(), meta.size());
  return ok || Fail(error, "spill file meta malformed");
}

std::atomic<std::uint64_t> g_temp_counter{0};

bool SyncDir(const std::string& path) {
  const std::string dir = std::filesystem::path(path).parent_path().string();
  const int fd = ::open(dir.empty() ? "." : dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return false;
  const bool ok = ::fsync(fd) == 0;
  ::close(fd);
  return ok;
}

// Runs `work(task index, staging)` on `threads` threads, each with one pinned staging buffer of kPiece bytes.
template <class F>
bool RunPool(rt::Device& dev, std::size_t n, unsigned threads, std::size_t kPiece, F&& work, std::string* error) {
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::mutex error_mu;
  const auto worker = [&] {
    void* staging = dev.AllocHost(kPiece, nullptr);
    if (!staging) {
      std::lock_guard<std::mutex> lk(error_mu);
      if (!failed.exchange(true) && error) *error = "spill: pinned staging allocation failed";
      return;
    }
    std::string e;
    for (std::size_t t = next++; !failed && t < n; t = next++)
      if (!work(t, static_cast<std::uint8_t*>(staging), &e)) {
        std::lock_guard<std::mutex> lk(error_mu);
        if (!failed.exchange(true) && error) *error = e;
        break;
      }
    dev.FreeHost(staging);
  };
  threads = std::max(1u, std::min<unsigned>(threads, static_cast<unsigned>(std::max<std::size_t>(n, 1))));
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < threads; ++i) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  return !failed;
}

}  // namespace

std::uint64_t FlashNextGpu::OptionsFingerprint() const {
  // Every option that changes a state's numbers (or its meaning), strictly: a state computed under other options
  // is not what this engine's own prefill would hold. Graph use and timing do not change results.
  std::vector<std::int64_t> v = {
      repack_q8_, block_gemv_, fused_moe_, decode_attn_batched_, decode_opt_, small_row_kernels_, int8_act_moe_,
      fused_hc_, fused_shexp_, fused_linear_, prefill_chunk_, prefill_long_chunk_, prefill_gemm_slice_,
      prefill_dense_prefix_, batched_prefill_, prefill_exact_projections_, prefill_int8_gemm_, prefill_f16_gemm_,
      prefill_grouped_moe_, prefill_parallel_bucket_, prefill_skinny_f32_, prefill_fused_combine_,
      prefill_fused_inject_, prefill_mix_rebuild_, prefill_blocked_xh_, prefill_banded_gemm_, prefill_reg_gemm_,
      prefill_fused_moe_combine_, prefill_tok_combine_, prefill_moe_in_combine_, prefill_moe_vec_, prefill_q_fused_,
      prefill_kq_fused_, prefill_gated_wave_, prefill_gate_in_place_, prefill_s640_, prefill_f16_out_,
      prefill_moe_blocked_, prefill_gdn_chunked_, prefill_gdn_chunk_fp32_, prefill_gdn_c16_bf16_,
      prefill_gdn_c16_u3_, prefill_kv16_, prefill_scores_wmma_, prefill_scores_b_, prefill_qsel_attention_,
      prefill_select256_, prefill_qsel_h2_, prefill_router_bf16x3_, prefill_router_f16_, prefill_router_tile_,
      prefill_mixed32_skip_, prefill_small_f16_, prefill_mix_h16_, prefill_shexp_w8a8_, prefill_down_w8a8_,
      prefill_gdn_io16_, prefill_gdn_f16_, prefill_res16_, prefill_gdn_scan_h_, prefill_batched_attention_,
      prefill_wmma_attention_, prefill_mq_attention_, prefill_fa_attention_, prefill_split_gdn_,
      prefill_test_perturb_, prefill_test_ulps_, kv16_only_, index_window_, static_cast<std::int64_t>(span_hsaco_.size())};
  std::uint64_t h = 0xcbf29ce484222325ull;
  for (auto x : v)
    for (int i = 0; i < 8; ++i) h = (h ^ static_cast<std::uint8_t>(static_cast<std::uint64_t>(x) >> (8 * i))) * 0x100000001b3ull;
  std::uint64_t lora = 0;  // order-independent: the adapters' scales
  for (const auto& [k, d] : lora_) {
    std::uint32_t bits;
    std::memcpy(&bits, &d.scale, 4);
    (void)k;
    lora += (std::uint64_t{bits} + 1) * 0x9e3779b97f4a7c15ull;
  }
  return h ^ lora;
}

std::vector<FlashNextGpu::SpillSection> FlashNextGpu::SpillLayout(std::uint32_t position,
                                                                  std::span<const std::uint32_t> blocks, bool mtp,
                                                                  std::uint32_t mtp_pos,
                                                                  std::uint32_t mtp_blocks) const {
  // Snapshot's rows (ForEachState) for `position` and `blocks`, then the draft block's.
  const auto& c = w_.config;
  std::vector<SpillSection> out;
  const auto add = [&](const rt::Buffer& b, std::size_t off_floats, std::size_t floats) {
    out.push_back({rt::Buffer{b.device ? static_cast<char*>(b.device) + off_floats * 4 : nullptr, floats * 4, nullptr},
                   floats * 4});
  };
  const std::size_t pos = position, kv = c.AttentionKvDim(), idim = c.indexer_head_dim;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (c.IsLinearLayer(l)) {
      const auto& s = linear_[l];
      add(s.conv, 0, s.conv.bytes / 4);
      add(s.state, 0, s.state.bytes / 4);
    } else {
      const auto& s = attn_[l];
      if (kv16_only_) {
        add(s.k16, 0, (pos * kv + 1) / 2);
        add(s.v16, 0, (pos * kv + 1) / 2);
      } else {
        add(s.k, 0, pos * kv);
        add(s.v, 0, pos * kv);
      }
      const std::size_t live = IndexLiveRow(position);
      add(s.index_k, live * idim, (pos - live) * idim);
      add(s.block_k, 0, std::size_t{blocks[l]} * idim);
    }
  }
  add(ple_history_, 0, ple_history_.bytes / 4);
  if (mtp) {
    add(mtp_hpend_, 0, c.HcDim());
    add(mtp_attn_.k, 0, std::size_t{mtp_pos} * kv);
    add(mtp_attn_.v, 0, std::size_t{mtp_pos} * kv);
    add(mtp_attn_.index_k, 0, std::size_t{mtp_pos} * idim);
    add(mtp_attn_.block_k, 0, std::size_t{mtp_blocks} * idim);
  }
  return out;
}

void FlashNextGpu::SpillCleanTemps(const std::string& dir) {
  std::error_code ec;
  for (const auto& e : std::filesystem::directory_iterator(dir, ec))
    if (e.path().filename().string().find(".tmp.") != std::string::npos) std::filesystem::remove(e.path(), ec);
}


bool FlashNextGpu::SpillSession(const std::string& path, const SpillProtection& prot,
                                std::span<const std::uint8_t> blob, SpillInfo* info, std::string* error) {
  if (poisoned_) return PoisonedFail(error);
  const auto t0 = std::chrono::steady_clock::now();
  if (info) info->published = false;
  const auto& c = w_.config;
  const std::size_t kPiece = std::max(kAlign, spill_piece_bytes_ / kAlign * kAlign);
  const bool v2 = prot.kek != nullptr;
  if (v2 && (!prot.kek->key.ok() || prot.kek->key.size() != crypt::kKeyBytes || prot.kek->id.empty() ||
             prot.kek->id.size() > crypt::kKeyIdBytes))
    return Fail(error, "spill: the spill key is not usable");
  if (!dev_.Synchronize(error)) return false;  // the state is final
  Meta m;
  m.f.version = v2 ? 2 : 1;
  m.f.model_fp = model_fp_;
  m.f.options_fp = OptionsFingerprint();
  m.f.kv_kind = kv16_only_ ? 1 : 0;
  m.f.index_window = index_window_ ? 1 : 0;
  m.f.position = position_;
  m.f.num_layers = c.num_layers;
  m.f.piece_bytes = static_cast<std::uint32_t>(kPiece);
  // The draft block's history goes along only when it follows this session unbroken (else it is not this
  // session's: restoring it would be wrong, and dropping it costs only drafting).
  const bool mtp = mtp_ && MtpPending();
  m.f.mtp = mtp ? 1 : 0;
  m.f.mtp_pos = mtp ? mtp_pos_ : 0;
  m.f.mtp_blocks = mtp ? mtp_attn_.blocks : 0;
  for (const auto& s : attn_) m.blocks.push_back(s.blocks);
  m.ngram.assign(ngram_history_.prev.begin(), ngram_history_.prev.end());
  // The format caps header + meta at 64 MB (the reader refuses more): every size is checked against it before it is
  // narrowed to a 32-bit field.
  if (m.ngram.size() > (16u << 20)) return Fail(error, "spill: n-gram history larger than the format allows");
  m.f.ngram_n = static_cast<std::uint32_t>(m.ngram.size());
  if (blob.size() > (32u << 20)) return Fail(error, "spill: blob larger than 32 MB");
  m.blob.assign(blob.begin(), blob.end());
  m.f.blob_bytes = static_cast<std::uint32_t>(m.blob.size());
  const auto sections = SpillLayout(position_, m.blocks, mtp, m.f.mtp_pos, m.f.mtp_blocks);
  std::vector<std::size_t> sizes;
  for (const auto& s : sections) sizes.push_back(s.bytes);
  // Header size depends on the task count, which depends on the payload start only through offsets.
  m.tasks = Tasks(sizes, 0, kPiece);
  if (m.tasks.size() >= crypt::kMetaIndex) return Fail(error, "spill: too many pieces");
  m.f.n_tasks = static_cast<std::uint32_t>(m.tasks.size());
  if (MetaBytes(m) + HeadBytes(m.f.version) > (64u << 20)) return Fail(error, "spill: meta larger than the 64 MB cap");
  m.f.meta_bytes = static_cast<std::uint32_t>(MetaBytes(m));
  const std::size_t header_bytes = AlignUp(HeadBytes(m.f.version) + m.f.meta_bytes);
  m.f.header_bytes = static_cast<std::uint32_t>(header_bytes);
  m.tasks = Tasks(sizes, header_bytes, kPiece);
  std::uint64_t end = header_bytes;
  for (auto s : sizes) end += AlignUp(s);
  m.f.file_bytes = end;
  switch (spill_test_identity_) {
    case 1: m.f.model_fp ^= 1; break;
    case 2: m.f.options_fp += 1; break;
    case 3: m.f.kv_kind ^= 1; break;
    case 4: m.f.position += 1; break;
    default: break;
  }

  // v2: a fresh data key and salt; the header's AAD fields are final now, so the DEK is wrapped first and the same
  // AAD then covers the meta and every piece.
  crypt::SecretBytes dek;
  alignas(8) std::uint8_t head2[crypt::kHeaderBytes] = {};
  const std::span<std::uint8_t, crypt::kHeaderBytes> head2s(head2, crypt::kHeaderBytes);
  if (v2) {
    dek = crypt::SecretBytes(crypt::kKeyBytes);
    if (!dek.ok() || !crypt::RandomBytes({dek.data(), crypt::kKeyBytes}) || !crypt::RandomBytes(m.salt) ||
        !crypt::RandomBytes(m.checkpoint))
      return Fail(error, "spill: cannot draw a data key (secure memory or RNG)");
    m.key_id = prot.kek->id;
    SpillHeaderV2 h;
    FillV2(m, &h);
    std::memcpy(head2, &h, sizeof(h));
    if (!crypt::WrapDek(*prot.kek, head2s, dek.span())) return Fail(error, "spill: wrapping the data key failed");
  }
  const auto aad = crypt::HeaderAad(head2s);

  const std::string tmp = path + ".tmp." + std::to_string(::getpid()) + "." + std::to_string(g_temp_counter++);
  // O_DIRECT (no page cache: 6+ GB files would evict the n-gram rows); buffered where refused (tmpfs).
  int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_DIRECT, 0600);
  if (fd < 0 && errno == EINVAL) fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) return Fail(error, "spill: cannot create " + tmp + ": " + std::strerror(errno));
  const auto abandon = [&](std::string why) {
    ::close(fd);
    ::unlink(tmp.c_str());
    return Fail(error, std::move(why));
  };
  if (::ftruncate(fd, static_cast<off_t>(end)) != 0) return abandon("spill: ftruncate failed");
  std::mutex dev_mu;
  const std::size_t stop_at = spill_fault_ == 1 ? m.tasks.size() / 2 : m.tasks.size();
  const bool ok = RunPool(
      dev_, stop_at, spill_threads_, kPiece,
      [&](std::size_t t, std::uint8_t* buf, std::string* e) {
        Task& k = m.tasks[t];
        const auto& sec = sections[k.section];
        // Offset of this piece inside its section: pieces of a section are consecutive kPiece cuts.
        std::size_t first = t;
        while (first > 0 && m.tasks[first - 1].section == k.section) --first;
        const std::size_t off = (t - first) * kPiece;
        {
          std::lock_guard<std::mutex> lk(dev_mu);
          if (!dev_.Download(sec.view, off, buf, k.bytes, e)) return false;
        }
        if (v2) {
          thread_local crypt::Gcm gcm;  // one EVP context per I/O thread
          if (!gcm.Seal(dek.span(), crypt::PieceNonce(m.salt, static_cast<std::uint32_t>(t)), aad, buf, k.bytes,
                        std::span<std::uint8_t, crypt::kTagBytes>(k.tag, crypt::kTagBytes)))
            return Fail(e, "spill: encryption failed");
        } else {
          k.crc = Crc32c(buf, k.bytes);
        }
        const std::size_t padded = AlignUp(k.bytes);
        if (padded > k.bytes) std::memset(buf + k.bytes, 0, padded - k.bytes);
        if (!PwriteAll(fd, buf, padded, k.file_off)) return Fail(e, std::string("spill: write failed: ") + std::strerror(errno));
        return true;
      },
      error);
  if (!ok) {
    if (v2) crypt::Wipe(head2, sizeof(head2));
    return abandon(error ? *error : "spill failed");
  }
  if (spill_fault_ == 1) {  // simulated crash mid-write: the temp file stays, nothing renamed
    ::close(fd);
    return Fail(error, "injected fault: crash mid-write");
  }
  {
    std::vector<std::uint8_t> head;
    if (v2) {
      head.assign(header_bytes, 0);
      PutMeta(m, head.data() + crypt::kHeaderBytes);
      crypt::Gcm gcm;
      if (!gcm.Seal(dek.span(), crypt::PieceNonce(m.salt, crypt::kMetaIndex), aad, head.data() + crypt::kHeaderBytes,
                    m.f.meta_bytes, head2s.subspan<crypt::kMetaTagOff, crypt::kTagBytes>()))
        return abandon("spill: meta encryption failed");
      std::memcpy(head.data(), head2, crypt::kHeaderBytes);
    } else {
      head = EncodeV1(m);
    }
    void* hb = nullptr;
    if (::posix_memalign(&hb, kAlign, head.size()) != 0) return abandon("spill: header buffer");
    std::memcpy(hb, head.data(), head.size());
    const bool w = PwriteAll(fd, hb, head.size(), 0);
    std::free(hb);
    if (!w) return abandon(std::string("spill: header write failed: ") + std::strerror(errno));
  }
  if (::fsync(fd) != 0) return abandon(std::string("spill: fsync failed: ") + std::strerror(errno));
  ::close(fd);
  if (spill_fault_ == 2) return Fail(error, "injected fault: crash before rename");
  if (::rename(tmp.c_str(), path.c_str()) != 0) {
    const std::string why = std::string("spill: rename failed: ") + std::strerror(errno);
    ::unlink(tmp.c_str());
    return Fail(error, why);
  }
  if (info) info->published = true;
  if (spill_fault_ == 3 || !SyncDir(path))
    return Fail(error, "spill: fsync of the directory failed (published; the file may not be durable)");
  if (info) {
    info->file_bytes = end;
    info->position = position_;
    info->mtp = mtp;
    info->version = m.f.version;
    info->key_id = m.key_id;
    std::memcpy(info->checkpoint.data(), m.checkpoint, crypt::kCheckpointBytes);
    info->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  return true;
}

namespace {
struct Fd {
  int fd = -1;
  ~Fd() {
    if (fd >= 0) ::close(fd);
  }
};
}  // namespace

bool FlashNextGpu::SpillPeek(const std::string& path, const SpillProtection& prot, std::vector<std::uint8_t>* blob,
                             SpillInfo* info, std::string* error) const {
  Fd f{::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
  if (f.fd < 0) return Fail(error, "spill file: cannot open " + path + ": " + std::strerror(errno));
  struct stat st {};
  if (::fstat(f.fd, &st) != 0) return Fail(error, "spill file: stat failed");
  Meta m;
  crypt::SecretBytes dek;
  if (!Decode(f.fd, static_cast<std::uint64_t>(st.st_size), prot, &m, &dek, error)) return false;
  if (blob) *blob = m.blob;
  if (info) {
    info->file_bytes = m.f.file_bytes;
    info->position = m.f.position;
    info->mtp = m.f.mtp != 0;
    info->version = m.f.version;
    info->key_id = m.key_id;
    std::memcpy(info->checkpoint.data(), m.checkpoint, crypt::kCheckpointBytes);
  }
  return true;
}

bool FlashNextGpu::RestoreSession(const std::string& path, const SpillProtection& prot,
                                  std::vector<std::uint8_t>* blob, SpillInfo* info, std::string* error, bool* reset) {
  if (poisoned_) return PoisonedFail(error);
  const auto t0 = std::chrono::steady_clock::now();
  if (reset) *reset = false;
  const auto& c = w_.config;
  Fd f{::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
  if (f.fd < 0) return Fail(error, "spill file: cannot open " + path + ": " + std::strerror(errno));
  struct stat st {};
  if (::fstat(f.fd, &st) != 0) return Fail(error, "spill file: stat failed");
  Meta m;
  crypt::SecretBytes dek;
  if (!Decode(f.fd, static_cast<std::uint64_t>(st.st_size), prot, &m, &dek, error)) return false;
  const Fields& h = m.f;
  const bool v2 = h.version == 2;
  const std::size_t kPiece = h.piece_bytes;
  if (kPiece == 0 || kPiece % kAlign != 0 || kPiece > (256u << 20)) return Fail(error, "spill file piece size invalid");
  // Identity: refuse anything this engine would not have computed itself.
  if (h.model_fp != model_fp_) return Fail(error, "spill file is of another model (fingerprint mismatch)");
  if (h.options_fp != OptionsFingerprint()) return Fail(error, "spill file was written under other numerics options");
  if (h.kv_kind != (kv16_only_ ? 1u : 0u)) return Fail(error, "spill file has another KV kind (--kv-f16)");
  if (h.index_window != (index_window_ ? 1u : 0u)) return Fail(error, "spill file has another indexer-key window");
  if (h.num_layers != c.num_layers || m.ngram.size() != ngram_history_.prev.size())
    return Fail(error, "spill file layer count / n-gram window mismatch");
  if (h.position > max_context_) return Fail(error, "spill file position exceeds this engine's max context");
  for (std::uint32_t l = 0; l < c.num_layers; ++l)
    if (m.blocks[l] > h.position / std::max(1u, c.compress_ratio))
      return Fail(error, "spill file block counts invalid");
  const bool file_mtp = h.mtp != 0;
  if (file_mtp && (h.mtp_pos + 1 != h.position || h.mtp_blocks > (h.mtp_pos + 1) / std::max(1u, c.compress_ratio)))
    return Fail(error, "spill file draft-block state invalid");
  // The draft block's part is restored when this engine has the block and tracks; otherwise skipped.
  const bool take_mtp = file_mtp && mtp_ && mtp_track_ && mtp_hpend_;
  // The layout this engine computes for the file's position must be the file's, piece for piece.
  {
    const auto sections = SpillLayout(h.position, m.blocks, file_mtp, h.mtp_pos, h.mtp_blocks);
    std::vector<std::size_t> sizes;
    for (const auto& s : sections) sizes.push_back(s.bytes);
    const auto want = Tasks(sizes, h.header_bytes, kPiece);
    std::uint64_t end = h.header_bytes;
    for (auto s : sizes) end += AlignUp(s);
    if (want.size() != m.tasks.size() || end != h.file_bytes) return Fail(error, "spill file layout mismatch");
    for (std::size_t i = 0; i < want.size(); ++i)
      if (want[i].file_off != m.tasks[i].file_off || want[i].bytes != m.tasks[i].bytes ||
          want[i].section != m.tasks[i].section)
        return Fail(error, "spill file layout mismatch");
  }
  // From here the session is written: any failure resets it to empty.
  const auto fail_reset = [&](std::string why) {
    std::string e2;
    // Every worker has joined (RunPool); the device finishes what they queued before the state is reset, so no
    // write of this restore lands after it returns. If that cannot be established (the synchronize fails), queued
    // writes may still land in storage a reset would reuse: the engine poisons itself and refuses all further use.
    if (restore_sync_fault_ || !dev_.Synchronize(&e2)) {
      poisoned_ = true;
      std::fprintf(stderr, "engine: POISONED: device synchronize failed while cleaning up a failed restore (%s)\n",
                   e2.c_str());
      if (reset) *reset = false;
      return Fail(error, std::move(why) + "; device synchronize failed: engine poisoned");
    }
    (void)Reset(&e2);
    if (reset) *reset = true;
    return Fail(error, std::move(why));
  };
  if (!ReleaseKvPages(error) || !EnsureKv(h.position, h.position, error)) return fail_reset(error ? *error : "kv");
  position_ = h.position;
  for (std::size_t l = 0; l < attn_.size(); ++l) {
    attn_[l].blocks = m.blocks[l];
    attn_[l].valid16 = kv16_only_ ? position_ : 0;  // fp32 K/V: the fp16 copy is re-derived on use
  }
  std::copy(m.ngram.begin(), m.ngram.end(), ngram_history_.prev.begin());
  const auto sections = SpillLayout(h.position, m.blocks, file_mtp, h.mtp_pos, h.mtp_blocks);
  std::size_t trunk_sections = sections.size() - (file_mtp ? 5 : 0);
  std::vector<std::size_t> todo;
  for (std::size_t t = 0; t < m.tasks.size(); ++t)
    if (m.tasks[t].section < trunk_sections || take_mtp) todo.push_back(t);
  Fd d{::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT)};
  const int rfd = d.fd >= 0 ? d.fd : f.fd;
  // The AAD of every piece: the header as Decode read it (the unwrap authenticated exactly these bytes).
  const auto aad = crypt::HeaderAad(std::span<const std::uint8_t, crypt::kHeaderBytes>(m.raw, crypt::kHeaderBytes));
  std::mutex dev_mu;
  const bool ok = RunPool(
      dev_, todo.size(), spill_threads_, kPiece,
      [&](std::size_t i, std::uint8_t* buf, std::string* e) {
        const std::size_t t = todo[i];
        const Task& k = m.tasks[t];
        const std::size_t padded = AlignUp(k.bytes);
        if (!PreadAll(rfd, buf, padded, k.file_off)) return Fail(e, "spill file: read failed (truncated?)");
        if (v2) {
          thread_local crypt::Gcm gcm;
          if (!gcm.Open(dek.span(), crypt::PieceNonce(m.salt, static_cast<std::uint32_t>(t)), aad, buf, k.bytes,
                        std::span<const std::uint8_t, crypt::kTagBytes>(k.tag, crypt::kTagBytes)))
            return Fail(e, "spill file payload authentication failed at offset " + std::to_string(k.file_off));
        } else if (Crc32c(buf, k.bytes) != k.crc) {
          return Fail(e, "spill file payload checksum mismatch at offset " + std::to_string(k.file_off));
        }
        std::size_t first = t;
        while (first > 0 && m.tasks[first - 1].section == k.section) --first;
        std::lock_guard<std::mutex> lk(dev_mu);
        if (!dev_.Upload(sections[k.section].view, (t - first) * kPiece, buf, k.bytes, e)) return false;
        spill_uploads_.fetch_add(1, std::memory_order_relaxed);
        return true;
      },
      error);
  if (!ok) return fail_reset(error ? *error : "restore failed");
  if (!dev_.Synchronize(error)) return fail_reset(error ? *error : "sync");
  if (take_mtp) {
    mtp_owner_ = bound_;
    mtp_pos_ = h.mtp_pos;
    mtp_attn_.blocks = h.mtp_blocks;
  } else if (mtp_track_ && bound_ == mtp_owner_) {
    mtp_pos_ = position_ == 0 ? 0 : position_ + 1;  // no history for this state: no drafting (the trunk's results do not change)
  }
  if (blob) *blob = std::move(m.blob);
  if (info) {
    info->file_bytes = h.file_bytes;
    info->position = h.position;
    info->mtp = take_mtp;
    info->version = h.version;
    info->key_id = m.key_id;
    std::memcpy(info->checkpoint.data(), m.checkpoint, crypt::kCheckpointBytes);
    info->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  return true;
}

}  // namespace halo::engine
