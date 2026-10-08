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

extern char** environ;

namespace halo::server {

namespace crypt = core::crypt;

MasterSpillKeys::MasterSpillKeys(std::string provider, crypt::SecretBytes master, std::uint64_t epoch)
    : provider_(std::move(provider)), master_(std::move(master)), epoch_(epoch) {
  prefix_ = provider_ + ":" + crypt::KeyName(master_.span());
}

std::string MasterSpillKeys::Describe() const {
  return prefix_ + " epoch " + std::to_string(epoch_.load()) + (master_.locked() ? "" : " (not mlocked)");
}

bool MasterSpillKeys::Key(std::string_view session, Use, crypt::Kek* out, std::string* why) {
  // Seal and open use the current epoch: a file of an earlier epoch names another key id and is refused.
  return crypt::DeriveKek(master_.span(), session, epoch_.load(), prefix_, out, why);
}

bool MasterSpillKeys::Rotate(std::string*) {
  ++epoch_;
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

bool LoadMasterTpm(const std::string& cred_path, const std::string& name, crypt::SecretBytes* out, std::string* error) {
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
  std::vector<char*> argv = {const_cast<char*>("systemd-creds"), const_cast<char*>("decrypt"),
                             const_cast<char*>("--user"), const_cast<char*>(name_arg.c_str()),
                             const_cast<char*>(cred_path.c_str()), const_cast<char*>("-"), nullptr};
  pid_t pid = 0;
  const int rc = ::posix_spawnp(&pid, "systemd-creds", &fa, nullptr, argv.data(), environ);
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

bool MakeSpillKeys(const std::string& spec, std::uint64_t epoch, std::unique_ptr<SpillKeys>* out, std::string* error) {
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
  *out = std::make_unique<MasterSpillKeys>(provider, std::move(master), epoch);
  return true;
}

}  // namespace halo::server
