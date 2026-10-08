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

// Epochs: the same session at the same epoch gets the same KEK (a file can be opened); Rotate changes every
// session's key and id (files sealed before are unreadable: crypto-shredding); sessions never share a key.
TEST(SpillKeys, MasterProviderEpochsAndRotation) {
  sc::SecretBytes m(32);
  ASSERT_TRUE(sc::RandomBytes({m.data(), 32}));
  MasterSpillKeys k("file", std::move(m), 5);
  std::string e;
  sc::Kek s1, s1b, s2, s1r;
  ASSERT_TRUE(k.Key("a", SpillKeys::Use::kSeal, &s1, &e)) << e;
  ASSERT_TRUE(k.Key("a", SpillKeys::Use::kOpen, &s1b, &e)) << e;
  ASSERT_TRUE(k.Key("b", SpillKeys::Use::kSeal, &s2, &e)) << e;
  EXPECT_TRUE(sc::Equal(s1.key.span(), s1b.key.span()));
  EXPECT_EQ(s1.id, s1b.id);
  EXPECT_FALSE(sc::Equal(s1.key.span(), s2.key.span()));
  EXPECT_EQ(s1.id, k.prefix() + ":e5:a");
  EXPECT_EQ(k.prefix().rfind("file:", 0), 0u);
  ASSERT_TRUE(k.Rotate(&e));
  EXPECT_EQ(k.epoch(), 6u);
  ASSERT_TRUE(k.Key("a", SpillKeys::Use::kOpen, &s1r, &e)) << e;
  EXPECT_FALSE(sc::Equal(s1.key.span(), s1r.key.span()));
  EXPECT_NE(s1.id, s1r.id);
  EXPECT_EQ(k.Describe().find(kHex), std::string::npos);
}

// The TPM path: $CREDENTIALS_DIRECTORY first (a systemd service), else `systemd-creds decrypt --user` (here a
// stand-in on PATH that checks its arguments); a failing decrypt is an error, never a zero key.
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
        "printf '" + kHex + "\\n'\n",
        0700);
  Env path("PATH", bin + ":" + std::string(std::getenv("PATH") ? std::getenv("PATH") : ""));
  Env cd("CREDENTIALS_DIRECTORY", "");
  Write(dir + "/m.cred", "ciphertext", 0600);
  ASSERT_TRUE(halo::server::LoadMasterTpm(dir + "/m.cred", halo::server::kSpillCredentialName, &b, &e)) << e;
  EXPECT_TRUE(sc::Equal(a.span(), b.span()));
  EXPECT_FALSE(halo::server::LoadMasterTpm(dir + "/other.cred", halo::server::kSpillCredentialName, &b, &e));
  EXPECT_NE(e.find("failed"), std::string::npos) << e;
  std::unique_ptr<SpillKeys> keys;
  ASSERT_TRUE(halo::server::MakeSpillKeys("tpm:" + dir + "/m.cred", 1, &keys, &e)) << e;
  EXPECT_EQ(keys->Describe().rfind("tpm:", 0), 0u);
  ASSERT_TRUE(halo::server::MakeSpillKeys("none", 1, &keys, &e));
  EXPECT_FALSE(keys);
  EXPECT_FALSE(halo::server::MakeSpillKeys("tmp", 1, &keys, &e));
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
