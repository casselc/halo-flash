#ifndef HALO_SERVER_SPILL_KEYS_HPP_
#define HALO_SERVER_SPILL_KEYS_HPP_

// Key providers for encrypted session spill files (format v2, docs/SESSIONS.md section 10).
//
// A provider hands the session cache the KEK that seals (or opens) the spill file of one session:
//   - "tpm":  a 256-bit master key from a TPM2-sealed systemd credential, read once at startup (from
//             $CREDENTIALS_DIRECTORY when run as a service with LoadCredentialEncrypted=, else by running
//             `systemd-creds decrypt --user`, which works unprivileged for a user-scoped credential);
//   - "file": the master key from a 0600 file (tests, machines without a TPM);
//   - "none": no provider (spills are refused unless the operator chose plaintext);
//   - "yard": per-session keys that yard-ctl registers over POST /admin/keys/{session} (yard_keys.hpp).
// Master providers derive KEK = HKDF-SHA256(master, "halo-spill-v2" || session || epoch); bumping the epoch (Rotate)
// makes every file sealed before it unreadable (crypto-shredding). Masters live in mlocked SecretBytes, zeroed on
// exit; nothing here logs key bytes.

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include "core/spill_crypt.hpp"

namespace halo::server {

class SpillKeys {
 public:
  enum class Use : std::uint8_t { kSeal, kOpen };
  virtual ~SpillKeys() = default;
  /// Provider and key name, for logs and GET /admin/sessions (never key bytes).
  [[nodiscard]] virtual std::string Describe() const = 0;
  /// The KEK for sealing a new file of `session`, or for opening one. False: no key; the caller refuses (a seal
  /// never falls back to plaintext; an open leaves the file for a later attempt or re-prefills).
  [[nodiscard]] virtual bool Key(std::string_view session, Use use, core::crypt::Kek* out, std::string* why) = 0;
  /// The KEK to open an existing file of `session` whose header names `file_key_id` (a provider holding several
  /// kinds of key picks by the id's prefix). Default: Key(session, kOpen).
  [[nodiscard]] virtual bool KeyForFile(std::string_view session, std::string_view file_key_id, core::crypt::Kek* out,
                                        std::string* why) {
    (void)file_key_id;
    return Key(session, Use::kOpen, out, why);
  }
  /// The file was sealed / opened (done). One-shot providers (yard) forget the session's key here.
  virtual void Used(std::string_view session, Use use) {
    (void)session;
    (void)use;
  }
  /// Moves to a new key epoch: files sealed before are unreadable from now on. False when not supported.
  [[nodiscard]] virtual bool Rotate(std::string* why) {
    if (why) *why = "this spill key provider does not rotate";
    return false;
  }
};

/// "tpm" / "file": one master key, an epoch (starts at `epoch`, Rotate adds one).
class MasterSpillKeys final : public SpillKeys {
 public:
  MasterSpillKeys(std::string provider, core::crypt::SecretBytes master, std::uint64_t epoch);
  [[nodiscard]] std::string Describe() const override;
  [[nodiscard]] bool Key(std::string_view session, Use use, core::crypt::Kek* out, std::string* why) override;
  [[nodiscard]] bool Rotate(std::string* why) override;
  [[nodiscard]] std::uint64_t epoch() const { return epoch_.load(); }
  [[nodiscard]] bool locked() const { return master_.locked(); }
  /// The prefix of every key id: "<provider>:<master name>".
  [[nodiscard]] const std::string& prefix() const { return prefix_; }

 private:
  std::string provider_, prefix_;
  core::crypt::SecretBytes master_;
  std::atomic<std::uint64_t> epoch_;
};

/// Reads a 32-byte master key: exactly 32 raw bytes, or 64 hex digits (and an optional newline). The file must not
/// be readable or writable by group or others.
[[nodiscard]] bool LoadMasterFile(const std::string& path, core::crypt::SecretBytes* out, std::string* error);
/// The master key from the TPM-sealed credential `name`: $CREDENTIALS_DIRECTORY/<name> when set and present, else
/// `systemd-creds decrypt --user --name=<name> <cred_path> -` (stdout read straight into SecretBytes).
[[nodiscard]] bool LoadMasterTpm(const std::string& cred_path, const std::string& name, core::crypt::SecretBytes* out,
                                 std::string* error);
/// The default credential path: $XDG_CONFIG_HOME (or $HOME/.config) + "/halo-flash/spill-master.cred".
[[nodiscard]] std::string DefaultTpmCredentialPath();
constexpr const char* kSpillCredentialName = "halo-spill-master";

/// `--spill-key` value: "tpm" | "tpm:CRED" | "file:PATH" | "none". "none" yields null (and true).
[[nodiscard]] bool MakeSpillKeys(const std::string& spec, std::uint64_t epoch, std::unique_ptr<SpillKeys>* out,
                                 std::string* error);

}  // namespace halo::server

#endif  // HALO_SERVER_SPILL_KEYS_HPP_
