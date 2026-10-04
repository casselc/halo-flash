"""Tests of tools/mutants.py (no build needed): python3 -m unittest tools/mutants_test.py"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutants  # noqa: E402

SUITE = '[suite.s]\nbuild_dir = "b"\n'
MUTANT = "[[mutant]]\nname = '{name}'\nfile = 'f'\nfind = '{find}'\nreplace = 'y'\nsuite = '{suite}'\n" \
         "binary = 't'\nfilter = '*'\nwhy = 'w'\n"


def write(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    f.write(text)
    f.close()
    return f.name


class Manifest(unittest.TestCase):
    def test_checked_in_manifest_loads(self):
        suites, ms = mutants.load(os.path.join(mutants.ROOT, "tests", "mutants.toml"))
        self.assertTrue(ms)
        self.assertTrue(all(m.suite in suites for m in ms))

    def test_every_anchor_matches_exactly_once(self):
        # A refactor that moves mutated code makes its mutant INVALID; catch
        # that here, without a build or a GPU.
        _, ms = mutants.load(os.path.join(mutants.ROOT, "tests", "mutants.toml"))
        for m in ms:
            with open(os.path.join(mutants.ROOT, m.file)) as f:
                self.assertEqual(f.read().count(m.find), 1, m.name)

    def test_rejects_duplicates_unknown_suites_and_noops(self):
        for bad in (SUITE + MUTANT.format(name="a", find="x", suite="s") * 2,
                    SUITE + MUTANT.format(name="a", find="x", suite="nope"),
                    SUITE + MUTANT.format(name="a", find="y", suite="s")):
            with self.assertRaises(ValueError):
                mutants.load(write(bad))


class Apply(unittest.TestCase):
    def mutant(self, find):
        return mutants.Mutant("m", "f", find, "Z", "s", "t", "*", "w", 1)

    def test_exactly_once(self):
        self.assertEqual(mutants.apply("a b c", self.mutant("b")), "a Z c")
        for text in ("a c", "b b"):  # absent, ambiguous
            with self.assertRaises(ValueError):
                mutants.apply(text, self.mutant("b"))

    def test_skipped_suites_are_detected(self):
        self.assertTrue(mutants.skipped("[  SKIPPED ] X.Y (0 ms)"))
        self.assertFalse(mutants.skipped("[  PASSED  ] 3 tests."))


if __name__ == "__main__":
    unittest.main()
