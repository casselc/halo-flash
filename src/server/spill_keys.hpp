#ifndef HALO_SERVER_SPILL_KEYS_HPP_
#define HALO_SERVER_SPILL_KEYS_HPP_

// Key providers for encrypted session spill files (format v2, docs/SESSIONS.md section 10).
//
// A provider hands the session cache a lease on the KEK that seals (or opens) one spill file:
//   - "tpm":  a 256-bit master key from a TPM2-sealed systemd credential, read once at startup (from
//             $CREDENTIALS_DIRECTORY when run as a service with LoadCredentialEncrypted=, else by running
//             /usr/bin/systemd-creds decrypt --user with a scrubbed environment; unprivileged for a user-scoped
//             credential);
//   - "file": the master key from a 0600 file (tests, machines without a TPM);
//   - "none": no provider (spills are refused unless the operator chose plaintext);
//   - "yard": per-checkpoint keys that yard-ctl registers (yard_keys.hpp).
// Master providers derive KEK = HKDF-SHA256(master, "halo-spill-v2" || be32(len) || session || be64(epoch)). Moving
// the master to a new epoch (FenceEpoch) is an access fence, NOT crypto-shredding: the master still derives every
// old KEK. The epoch is persisted (a 0600 sidecar) and never goes back across restarts. Keys live in mlocked
// SecretBytes (fail closed), zeroed on exit; nothing here logs key bytes.

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include "core/spill_crypt.hpp"
#include "server/state_file.hpp"

namespace halo::server {

/// A KEK handed out for one spill or restore. `gen` names the exact registration it came from: Finish consumes
/// that registration and no newer one.
struct SpillLease {
  core::crypt::Kek kek;
  std::string session;       ///< the label the file is sealed for (yard: the session the registration names)
  std::uint64_t epoch = 0;   ///< the file epoch in the KEK
  std::uint64_t gen = 0;     ///< provider-specific registration generation (0: not one-shot)
  bool yard = false;         ///< a yard key (custody: the entry is yard-bound from now on)
  std::uint64_t prev_fence = 0;  ///< yard: the used-epoch fence before this lease raised it (Finish restores it)
};

class SpillKeys {
 public:
  enum class Use : std::uint8_t { kSeal, kOpen };
  /// How a spill or restore under a lease ended. kNotDone: nothing was published (the spill file at its path is what
  /// it was). kUncertain: the file was published but its durability is not known (the directory fsync failed).
  enum class Outcome : std::uint8_t { kDone, kNotDone, kUncertain };
  virtual ~SpillKeys() = default;
  /// Provider and key name, for logs and GET /admin/sessions (never key bytes).
  [[nodiscard]] virtual std::string Describe() const = 0;
  /// A lease to seal a new file of cache entry `entry` (its incarnation-safe id, "<incarnation>.<key>"; label
  /// `label`). `yard_bound`: the entry is or was sealed under a yard key; such an entry is never sealed under a master
  /// key again (no custody downgrade). False: no key; the caller refuses (never a plaintext fallback).
  [[nodiscard]] virtual bool SealLease(std::string_view entry, std::string_view label, bool yard_bound, SpillLease* out,
                                       std::string* why) = 0;
  /// A lease to open the file of `label` whose header names `file_key_id`. False: no key now (the entry waits).
  [[nodiscard]] virtual bool OpenLease(std::string_view label, std::string_view file_key_id, SpillLease* out,
                                       std::string* why) = 0;
  /// The spill / restore under `lease` ended with `outcome`: one-shot providers consume exactly that registration;
  /// only a seal that published nothing (kNotDone) may leave its registration for a retry.
  virtual void Finish(const SpillLease& lease, Use use, Outcome outcome) {
    (void)lease;
    (void)use;
    (void)outcome;
  }
  /// Could `file_key_id` still open? False for a file the provider can never open again (an older master epoch, a
  /// fenced yard epoch); true when it may (including "not registered yet").
  [[nodiscard]] virtual bool MayOpen(std::string_view label, std::string_view file_key_id) = 0;
  /// Master providers: moves to the next epoch (persisted first). Files of older epochs no longer open through
  /// this process; this is an access fence, not erasure (the master still derives them). False when unsupported.
  [[nodiscard]] virtual bool FenceEpoch(std::string* why) {
    if (why) *why = "this spill key provider has no epoch to fence";
    return false;
  }
};

/// "tpm" / "file": one master key and a persisted epoch.
class MasterSpillKeys final : public SpillKeys {
 public:
  /// `epoch_file`: where the epoch is kept (0600; created at 1 on the first start, with an init marker; a malformed,
  /// unsafe or missing-after-init file refuses the start; held exclusively through "<epoch_file>.lock" for the
  /// provider's lifetime). Empty: in memory only (tests).
  static std::unique_ptr<MasterSpillKeys> Create(std::string provider, core::crypt::SecretBytes master,
                                                 const std::string& epoch_file, std::string* error);
  [[nodiscard]] std::string Describe() const override;
  [[nodiscard]] bool SealLease(std::string_view entry, std::string_view label, bool yard_bound, SpillLease* out,
                               std::string* why) override;
  [[nodiscard]] bool OpenLease(std::string_view label, std::string_view file_key_id, SpillLease* out,
                               std::string* why) override;
  [[nodiscard]] bool MayOpen(std::string_view label, std::string_view file_key_id) override;
  /// Persisted first. Unpublished: nothing changes (false). Uncertain (published, directory fsync failed): the new
  /// epoch is taken anyway, so memory matches what the disk shows and the stricter fence holds (false, with why).
  [[nodiscard]] bool FenceEpoch(std::string* why) override;
  [[nodiscard]] std::uint64_t epoch() const { return epoch_.load(); }
  /// The prefix of every key id: "<provider>:<master name>".
  [[nodiscard]] const std::string& prefix() const { return prefix_; }
  /// The id this provider gives a file of `label` at the current epoch.
  [[nodiscard]] std::string IdFor(std::string_view label) const;

 private:
  MasterSpillKeys() = default;
  bool Derive(std::string_view label, SpillLease* out, std::string* why);
  std::string provider_, prefix_, epoch_file_;
  std::unique_ptr<StateLock> lock_;
  core::crypt::SecretBytes master_;
  std::atomic<std::uint64_t> epoch_{1};
  std::mutex mu_;
};

/// Reads a 32-byte master key: exactly 32 raw bytes, or 64 hex digits (and an optional newline). The file must not
/// be readable or writable by group or others, and must not be a symlink.
[[nodiscard]] bool LoadMasterFile(const std::string& path, core::crypt::SecretBytes* out, std::string* error);
/// The master key from the TPM-sealed credential `name`: $CREDENTIALS_DIRECTORY/<name> when set and present, else
/// `<systemd_creds> decrypt --user --name=<name> <cred_path> -` run with a scrubbed environment (stdout read
/// straight into SecretBytes). `systemd_creds`: an absolute path (empty: kSystemdCreds).
[[nodiscard]] bool LoadMasterTpm(const std::string& cred_path, const std::string& name, core::crypt::SecretBytes* out,
                                 std::string* error, const std::string& systemd_creds = "");
/// The default credential path: $XDG_CONFIG_HOME (or $HOME/.config) + "/halo-flash/spill-master.cred".
[[nodiscard]] std::string DefaultTpmCredentialPath();
constexpr const char* kSpillCredentialName = "halo-spill-master";
constexpr const char* kSystemdCreds = "/usr/bin/systemd-creds";

/// `--spill-key` value: "tpm" | "tpm:CRED" | "file:PATH" | "none". "none" yields null (and true). `state_dir`: where
/// the master's epoch sidecar lives (master-<key name>.epoch); empty: in memory (tests).
[[nodiscard]] bool MakeSpillKeys(const std::string& spec, const std::string& state_dir,
                                 std::unique_ptr<MasterSpillKeys>* out, std::string* error);


}  // namespace halo::server

#endif  // HALO_SERVER_SPILL_KEYS_HPP_
