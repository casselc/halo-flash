// Spill key providers (server/spill_keys.hpp): the master file's checks, the TPM credential path through
// $CREDENTIALS_DIRECTORY and through `systemd-creds decrypt` (a stand-in script on PATH; the real, unprivileged
// user-scoped credential with HALO_TEST_SYSTEMD_CREDS=1), epochs and rotation, and the --spill-key parser.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "core/spill_crypt.hpp"
#include "server/spill_keys.hpp"

namespace sc = halo::core::crypt;
using halo::server::MasterSpillKeys;
using halo::server::SpillKeys;

namespace {
std::string TempDir(const std::string& tag) {
  auto p = std::filesystem::temp_directory_path() / ("halo-spill-keys-" + std::to_string(::getpid()) + "-" + tag);
  std::filesystem::remove_all(p);
  std::filesystem::create_directories(p);
  std::filesystem::permissions(p, std::filesystem::perms::owner_all);
  return p.string();
}
void Write(const std::string& path, const std::string& bytes, mode_t mode) {
  std::ofstream(path, std::ios::binary | std::ios::trunc) << bytes;
  ::chmod(path.c_str(), mode);
}
const std::string kHex = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
// The same 32 bytes, raw.
std::string Raw() {
  std::string r;
  for (int i = 0; i < 32; ++i) r += static_cast<char>(i);
  return r;
}
struct Env {
  std::string name, saved;
  bool had;
  Env(std::string n, const std::string& v) : name(std::move(n)) {
    const char* o = std::getenv(name.c_str());
    had = o != nullptr;
    if (had) saved = o;
    ::setenv(name.c_str(), v.c_str(), 1);
  }
  ~Env() {
    if (had) ::setenv(name.c_str(), saved.c_str(), 1);
    else ::unsetenv(name.c_str());
  }
};
}  // namespace

TEST(SpillKeys, MasterFileFormatsAndPermissions) {
  const auto dir = TempDir("file");
  std::string e;
  sc::SecretBytes a, b;
  Write(dir + "/hex", kHex + "\n", 0600);
  Write(dir + "/raw", Raw(), 0600);
  ASSERT_TRUE(halo::server::LoadMasterFile(dir + "/hex", &a, &e)) << e;
  ASSERT_TRUE(halo::server::LoadMasterFile(dir + "/raw", &b, &e)) << e;
  EXPECT_TRUE(sc::Equal(a.span(), b.span()));
  EXPECT_TRUE(a.locked());
  Write(dir + "/open", kHex, 0644);
  EXPECT_FALSE(halo::server::LoadMasterFile(dir + "/open", &a, &e));
  EXPECT_NE(e.find("chmod 600"), std::string::npos) << e;
  Write(dir + "/short", kHex.substr(0, 62), 0600);
  EXPECT_FALSE(halo::server::LoadMasterFile(dir + "/short", &a, &e));
  Write(dir + "/nothex", std::string(64, 'g'), 0600);
  EXPECT_FALSE(halo::server::LoadMasterFile(dir + "/nothex", &a, &e));
  std::filesystem::create_symlink(dir + "/hex", dir + "/link");
  EXPECT_FALSE(halo::server::LoadMasterFile(dir + "/link", &a, &e)) << "followed a symlink";
  EXPECT_FALSE(halo::server::LoadMasterFile(dir + "/missing", &a, &e));
  std::filesystem::remove_all(dir);
}

// Epochs: the same label at the same epoch gets the same KEK (a file can be opened); FenceEpoch moves every label
// to new keys and ids (files sealed before no longer open here: an access fence; the master still derives them).
// The epoch is persisted in a 0600 sidecar and never goes back across a restart; a malformed sidecar refuses the
// start. A yard-bound entry is never sealed under the master.
TEST(SpillKeys, MasterProviderEpochsAndFence) {
  const auto dir = TempDir("epoch");
  const std::string ef = dir + "/master.epoch";
  const auto master = [] {
    sc::SecretBytes m(32);
    for (int i = 0; i < 32; ++i) m.data()[i] = static_cast<std::uint8_t>(i * 3);
    return m;
  };
  std::string e;
  auto k = MasterSpillKeys::Create("file", master(), ef, &e);
  ASSERT_TRUE(k) << e;
  EXPECT_EQ(k->epoch(), 1u);
  struct stat st {};
  ASSERT_EQ(::stat(ef.c_str(), &st), 0);
  EXPECT_EQ(st.st_mode & 0777, 0600u);
  halo::server::SpillLease s1, s1b, s2, s1r;
  ASSERT_TRUE(k->SealLease("e.1", "a", false, &s1, &e)) << e;
  ASSERT_TRUE(k->OpenLease("a", s1.kek.id, &s1b, &e)) << e;
  ASSERT_TRUE(k->SealLease("e.2", "b", false, &s2, &e)) << e;
  EXPECT_TRUE(sc::Equal(s1.kek.key.span(), s1b.kek.key.span()));
  EXPECT_FALSE(sc::Equal(s1.kek.key.span(), s2.kek.key.span()));
  EXPECT_EQ(s1.kek.id, k->prefix() + ":e1:a");
  EXPECT_EQ(k->prefix().rfind("file:", 0), 0u);
  EXPECT_FALSE(k->SealLease("e.1", "sess-1", true, &s1r, &e)) << "a yard-bound entry sealed under the master";
  ASSERT_TRUE(k->FenceEpoch(&e)) << e;
  ASSERT_TRUE(k->FenceEpoch(&e)) << e;
  EXPECT_EQ(k->epoch(), 3u);
  EXPECT_FALSE(k->MayOpen("a", s1.kek.id));
  EXPECT_FALSE(k->OpenLease("a", s1.kek.id, &s1r, &e)) << "a fenced epoch's file opened";
  ASSERT_TRUE(k->SealLease("e.1", "a", false, &s1r, &e)) << e;
  EXPECT_FALSE(sc::Equal(s1.kek.key.span(), s1r.kek.key.span()));
  EXPECT_NE(s1.kek.id, s1r.kek.id);
  EXPECT_EQ(k->Describe().find(kHex), std::string::npos);
  // A restart continues at the persisted epoch (never back to 1).
  k.reset();
  k = MasterSpillKeys::Create("file", master(), ef, &e);
  ASSERT_TRUE(k) << e;
  EXPECT_EQ(k->epoch(), 3u) << "the epoch went back across a restart";
  EXPECT_FALSE(k->MayOpen("a", s1.kek.id));
  k.reset();
  Write(ef, "zero\n", 0600);
  EXPECT_FALSE(MasterSpillKeys::Create("file", master(), ef, &e)) << "a malformed epoch file was accepted";
  Write(ef, "0\n", 0600);
  EXPECT_FALSE(MasterSpillKeys::Create("file", master(), ef, &e));
  // MakeSpillKeys keeps the sidecar per master (by key name) in the state directory.
  Write(dir + "/k", kHex, 0600);
  std::unique_ptr<MasterSpillKeys> m2;
  ASSERT_TRUE(halo::server::MakeSpillKeys("file:" + dir + "/k", dir, &m2, &e)) << e;
  ASSERT_TRUE(m2->FenceEpoch(&e));
  m2.reset();
  ASSERT_TRUE(halo::server::MakeSpillKeys("file:" + dir + "/k", dir, &m2, &e)) << e;
  EXPECT_EQ(m2->epoch(), 2u);
  std::filesystem::remove_all(dir);
}

// The TPM path: $CREDENTIALS_DIRECTORY first (a systemd service), else `systemd-creds decrypt --user` by absolute
// path with a scrubbed environment (here a stand-in that checks its arguments and its environment); a relative
// path is refused; a failing decrypt is an error, never a zero key.
TEST(SpillKeys, TpmCredentialFromServiceDirectoryOrSystemdCreds) {
  const auto dir = TempDir("tpm");
  std::string e;
  sc::SecretBytes a, b;
  {
    Write(dir + "/" + halo::server::kSpillCredentialName, kHex, 0400);
    Env cd("CREDENTIALS_DIRECTORY", dir);
    ASSERT_TRUE(halo::server::LoadMasterTpm(dir + "/unused.cred", halo::server::kSpillCredentialName, &a, &e)) << e;
  }
  const std::string bin = dir + "/bin";
  std::filesystem::create_directories(bin);
  Write(bin + "/systemd-creds",
        "#!/bin/sh\n"
        "[ \"$1 $2 $3 $4\" = \"decrypt --user --name=halo-spill-master " + dir + "/m.cred\" ] || exit 3\n"
        "[ \"$5\" = - ] || exit 4\n"
        "[ -f \"$4\" ] || exit 5\n"
        "[ \"$PATH\" = /usr/bin:/bin ] || exit 6\n"
        "[ -z \"$HALO_LEAK\" ] || exit 7\n"
        "printf '" + kHex + "\\n'\n",
        0700);
  Env path("PATH", bin + ":" + std::string(std::getenv("PATH") ? std::getenv("PATH") : ""));
  Env leak("HALO_LEAK", "1");
  Env cd("CREDENTIALS_DIRECTORY", "");
  Write(dir + "/m.cred", "ciphertext", 0600);
  const std::string creds = bin + "/systemd-creds";
  ASSERT_TRUE(halo::server::LoadMasterTpm(dir + "/m.cred", halo::server::kSpillCredentialName, &b, &e, creds)) << e;
  EXPECT_TRUE(sc::Equal(a.span(), b.span()));
  EXPECT_FALSE(halo::server::LoadMasterTpm(dir + "/other.cred", halo::server::kSpillCredentialName, &b, &e, creds));
  EXPECT_NE(e.find("failed"), std::string::npos) << e;
  EXPECT_FALSE(halo::server::LoadMasterTpm(dir + "/m.cred", halo::server::kSpillCredentialName, &b, &e, "systemd-creds"))
      << "a PATH-searched systemd-creds was run";
  std::unique_ptr<MasterSpillKeys> keys;
  ASSERT_TRUE(halo::server::MakeSpillKeys("none", "", &keys, &e));
  EXPECT_FALSE(keys);
  EXPECT_FALSE(halo::server::MakeSpillKeys("tmp", "", &keys, &e));
  std::filesystem::remove_all(dir);
}

// Opt-in (HALO_TEST_SYSTEMD_CREDS=1): the real thing on this host. A user-scoped credential, sealed with
// `systemd-creds encrypt --user` (host+TPM2 key, no root), decrypts through the provider to the same 32 bytes.
TEST(SpillKeys, RealUserScopedTpmCredential) {
  if (!std::getenv("HALO_TEST_SYSTEMD_CREDS")) GTEST_SKIP() << "set HALO_TEST_SYSTEMD_CREDS=1";
  const auto dir = TempDir("real");
  Write(dir + "/plain", kHex + "\n", 0600);
  const std::string cmd = "systemd-creds encrypt --user --with-key=host+tpm2 --name=halo-spill-master " + dir + "/plain " +
                          dir + "/m.cred";
  ASSERT_EQ(std::system(cmd.c_str()), 0) << cmd;
  std::filesystem::remove(dir + "/plain");
  Env cd("CREDENTIALS_DIRECTORY", "");
  sc::SecretBytes a, b;
  std::string e;
  ASSERT_TRUE(halo::server::LoadMasterTpm(dir + "/m.cred", halo::server::kSpillCredentialName, &a, &e)) << e;
  Write(dir + "/hex", kHex, 0600);
  ASSERT_TRUE(halo::server::LoadMasterFile(dir + "/hex", &b, &e)) << e;
  EXPECT_TRUE(sc::Equal(a.span(), b.span()));
  std::filesystem::remove_all(dir);
}
