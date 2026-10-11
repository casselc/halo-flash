#include "server/spill_keys.hpp"

#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace halo::server {

namespace crypt = core::crypt;

std::unique_ptr<MasterSpillKeys> MasterSpillKeys::Create(std::string provider, crypt::SecretBytes master,
                                                         const std::string& epoch_file, std::string* error) {
  if (!master.ok() || master.size() < crypt::kKeyBytes) return *error = "spill key: no master", nullptr;
  std::unique_ptr<MasterSpillKeys> k(new MasterSpillKeys());
  std::string name;
  if (!crypt::KeyName(master.span(), &name)) return *error = "spill key: cannot name the master (HMAC failed)", nullptr;
  k->provider_ = std::move(provider);
  k->prefix_ = k->provider_ + ":" + name;
  k->master_ = std::move(master);
  if (!epoch_file.empty()) {
    // The directory is pinned once; one owner: a second halo_serve on this file is refused (it would overwrite
    // fences with a stale epoch).
    const auto [dir, name] = SplitPath(epoch_file);
    k->dir_ = StateDir::Open(dir, error);
    if (!k->dir_) return nullptr;
    k->epoch_name_ = name;
    k->lock_ = k->dir_->Lock(name, error);
    if (!k->lock_) return nullptr;
    // The epoch never goes back: a restart continues from the persisted one (an older epoch's files stay fenced).
    std::string text, e2;
    const StateRead r = k->dir_->Read(name, 64, &text, &e2);
    if (r == StateRead::kError) return *error = "spill key: epoch file: " + e2 + " (refusing to start)", nullptr;
    if (r == StateRead::kFresh) {
      if (!k->dir_->Init(name, "1\n", error)) return nullptr;
    } else {
      // Exactly "<digits>\n": no prefix-plus-junk, no sign, no zero.
      std::uint64_t e = 0;
      if (text.size() < 2 || text.back() != '\n' ||
          !ParseU64(std::string_view(text).substr(0, text.size() - 1), std::uint64_t{1} << 62, &e) || e == 0)
        return *error = "spill key: epoch file " + epoch_file + " is malformed (refusing to start)", nullptr;
      k->epoch_.store(e);
      // A state from before the init markers: it gets its marker now, under the lock, before the provider is used.
      if (!k->dir_->EnsureMarker(name, error)) return nullptr;
    }
  }
  return k;
}

std::string MasterSpillKeys::Describe() const {
  return prefix_ + " epoch " + std::to_string(epoch_.load());
}

std::string MasterSpillKeys::IdFor(std::string_view label) const {
  return prefix_ + ":e" + std::to_string(epoch_.load()) + ":" + std::string(label);
}

bool MasterSpillKeys::Derive(std::string_view label, SpillLease* out, std::string* why) {
  const std::uint64_t e = epoch_.load();
  if (!crypt::DeriveKek(master_.span(), label, e, prefix_, &out->kek, why)) return false;
  out->session = std::string(label);
  out->epoch = e;
  out->gen = 0;
  out->yard = false;
  return true;
}

bool MasterSpillKeys::SealLease(std::string_view, std::string_view label, bool yard_bound, SpillLease* out,
                                std::string* why) {
  if (yard_bound) return *why = "a yard-bound entry is never sealed under the master key", false;
  {
    // After an uncertain fence the epoch in memory may not survive a crash: no new file is sealed at it until a
    // durable write confirms it (otherwise a crash could revert the epoch and orphan those files).
    std::lock_guard<std::mutex> lk(mu_);
    if (unconfirmed_) {
      std::string e;
      if (dir_->Write(epoch_name_, std::to_string(epoch_.load()) + "\n", &e) != Persist::kDurable)
        return *why = "the master epoch is not confirmed durable yet (" + e + "): spill refused", false;
      unconfirmed_ = false;
    }
  }
  return Derive(label, out, why);
}

bool MasterSpillKeys::OpenLease(std::string_view label, std::string_view file_key_id, SpillLease* out,
                                std::string* why) {
  // The current epoch only: a file of an earlier epoch names another key id, and the engine refuses it.
  if (file_key_id != IdFor(label)) return *why = "spill file is of another master or epoch (fenced)", false;
  return Derive(label, out, why);
}

bool MasterSpillKeys::MayOpen(std::string_view label, std::string_view file_key_id) {
  return file_key_id == IdFor(label);
}

bool MasterSpillKeys::FenceEpoch(std::string* why) {
  std::lock_guard<std::mutex> lk(mu_);
  const std::uint64_t next = epoch_.load() + 1;
  // Persisted before it takes effect: a crash cannot bring an older epoch back.
  if (dir_) {
    const Persist p = dir_->Write(epoch_name_, std::to_string(next) + "\n", why);
    if (p == Persist::kUnpublished) return false;
    if (p == Persist::kUncertain) {
      // The disk shows the new epoch (maybe not after a crash): take it, the stricter fence, but seal nothing at it
      // until a durable write confirms it (SealLease). Reported as a failure.
      epoch_.store(next);
      unconfirmed_ = true;
      *why = "epoch " + std::to_string(next) + " taken, but its durability is uncertain: " + *why;
      return false;
    }
  }
  epoch_.store(next);
  unconfirmed_ = false;  // a durable epoch confirms itself
  return true;
}

namespace {

int HexVal(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// `n` bytes in `buf` (secret) -> the 32-byte key: raw, or 64 hex digits plus an optional trailing newline.
bool ParseKey(const std::uint8_t* buf, std::size_t n, crypt::SecretBytes* out, std::string* error) {
  crypt::SecretBytes k(crypt::kKeyBytes);
  if (!k.ok()) return *error = "spill key: cannot allocate key memory", false;
  if (n == crypt::kKeyBytes) {
    std::memcpy(k.data(), buf, n);
  } else {
    std::size_t m = n;
    while (m > 0 && (buf[m - 1] == '\n' || buf[m - 1] == '\r')) --m;
    if (m != 2 * crypt::kKeyBytes) return *error = "spill key: want 32 raw bytes or 64 hex digits", false;
    for (std::size_t i = 0; i < crypt::kKeyBytes; ++i) {
      const int hi = HexVal(static_cast<char>(buf[2 * i])), lo = HexVal(static_cast<char>(buf[2 * i + 1]));
      if (hi < 0 || lo < 0) return *error = "spill key: want 32 raw bytes or 64 hex digits", false;
      k.data()[i] = static_cast<std::uint8_t>(hi * 16 + lo);
    }
  }
  *out = std::move(k);
  return true;
}

// Reads all of fd (at most `cap` bytes) into secret memory.
bool ReadSecret(int fd, std::size_t cap, crypt::SecretBytes* buf, std::size_t* n) {
  *buf = crypt::SecretBytes(cap + 1);
  if (!buf->ok()) return false;
  *n = 0;
  for (;;) {
    const ssize_t r = ::read(fd, buf->data() + *n, cap + 1 - *n);
    if (r < 0 && errno == EINTR) continue;
    if (r < 0) return false;
    if (r == 0) return true;
    *n += static_cast<std::size_t>(r);
    if (*n > cap) return false;
  }
}

}  // namespace

bool LoadMasterFile(const std::string& path, crypt::SecretBytes* out, std::string* error) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) return *error = "spill key file " + path + ": " + std::strerror(errno), false;
  struct stat st {};
  if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    ::close(fd);
    return *error = "spill key file " + path + ": not a regular file", false;
  }
  if ((st.st_mode & 077) != 0) {
    ::close(fd);
    return *error = "spill key file " + path + ": readable or writable by group/others (chmod 600)", false;
  }
  crypt::SecretBytes buf;
  std::size_t n = 0;
  const bool ok = ReadSecret(fd, 256, &buf, &n);
  ::close(fd);
  if (!ok) return *error = "spill key file " + path + ": read failed or too long", false;
  return ParseKey(buf.data(), n, out, error);
}

std::string DefaultTpmCredentialPath() {
  const char* x = std::getenv("XDG_CONFIG_HOME");
  std::string base = x && *x ? x : "";
  if (base.empty()) {
    const char* h = std::getenv("HOME");
    base = std::string(h ? h : ".") + "/.config";
  }
  return base + "/halo-flash/spill-master.cred";
}

bool LoadMasterTpm(const std::string& cred_path, const std::string& name, crypt::SecretBytes* out, std::string* error,
                   const std::string& systemd_creds) {
  const std::string bin = systemd_creds.empty() ? kSystemdCreds : systemd_creds;
  if (bin.empty() || bin[0] != '/') return *error = "spill key: systemd-creds must be an absolute path", false;
  // As a systemd service (LoadCredentialEncrypted=<name>:<cred>): systemd decrypted it into a private ramfs.
  if (const char* dir = std::getenv("CREDENTIALS_DIRECTORY"); dir && *dir) {
    const std::string p = std::string(dir) + "/" + name;
    struct stat st {};
    if (::stat(p.c_str(), &st) == 0) return LoadMasterFile(p, out, error);
  }
  // Otherwise ask systemd's credential service (io.systemd.Credentials, reachable unprivileged) to decrypt the
  // user-scoped, TPM2-bound credential. Its stdout goes through a pipe straight into locked memory.
  int pipefd[2];
  if (::pipe2(pipefd, O_CLOEXEC) != 0) return *error = "spill key: pipe failed", false;
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_adddup2(&fa, pipefd[1], 1);
  const std::string name_arg = "--name=" + name;
  std::vector<char*> argv = {const_cast<char*>(bin.c_str()), const_cast<char*>("decrypt"),
                             const_cast<char*>("--user"), const_cast<char*>(name_arg.c_str()),
                             const_cast<char*>(cred_path.c_str()), const_cast<char*>("-"), nullptr};
  pid_t pid = 0;
  // A scrubbed environment: nothing from ours (PATH, LD_*, SYSTEMD_*) steers the binary that produces the master.
  char env_path[] = "PATH=/usr/bin:/bin";
  char env_lang[] = "LANG=C";
  char* envp[] = {env_path, env_lang, nullptr};
  const int rc = ::posix_spawn(&pid, bin.c_str(), &fa, nullptr, argv.data(), envp);
  posix_spawn_file_actions_destroy(&fa);
  ::close(pipefd[1]);
  if (rc != 0) {
    ::close(pipefd[0]);
    return *error = std::string("spill key: cannot run systemd-creds: ") + std::strerror(rc), false;
  }
  crypt::SecretBytes buf;
  std::size_t n = 0;
  const bool read_ok = ReadSecret(pipefd[0], 256, &buf, &n);
  ::close(pipefd[0]);
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    return *error = "spill key: systemd-creds decrypt " + cred_path + " failed (exit " +
                    std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1) + "; see its stderr)",
           false;
  if (!read_ok) return *error = "spill key: credential too long", false;
  return ParseKey(buf.data(), n, out, error);
}

bool MakeSpillKeys(const std::string& spec, const std::string& state_dir, std::unique_ptr<MasterSpillKeys>* out,
                   std::string* error) {
  out->reset();
  if (spec == "none") return true;
  crypt::SecretBytes master;
  std::string provider;
  if (spec == "tpm" || spec.rfind("tpm:", 0) == 0) {
    const std::string cred = spec.size() > 4 ? spec.substr(4) : DefaultTpmCredentialPath();
    if (!LoadMasterTpm(cred, kSpillCredentialName, &master, error)) return false;
    provider = "tpm";
  } else if (spec.rfind("file:", 0) == 0) {
    if (!LoadMasterFile(spec.substr(5), &master, error)) return false;
    provider = "file";
  } else {
    return *error = "--spill-key: want tpm, tpm:CRED, file:PATH or none (got " + spec + ")", false;
  }
  std::string name;
  if (!crypt::KeyName(master.span(), &name)) return *error = "spill key: cannot name the master", false;
  *out = MasterSpillKeys::Create(provider, std::move(master),
                                 state_dir.empty() ? "" : state_dir + "/master-" + name + ".epoch", error);
  return *out != nullptr;
}

}  // namespace halo::server
