#!/usr/bin/env python3
"""Tests for tools/mull_survivors.py: python3 -m unittest tools/mull_survivors_test.py"""
import contextlib
import io
import os
import sqlite3
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True  # no __pycache__ in tools/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mull_survivors as ms  # noqa: E402

SRC = r'''#include "x.hpp"
#define BRACE {
namespace halo::core {
namespace {
const int kTable[] = {1, 2, 3};
int Helper(int a) {
  auto f = [&](int b) { return a + b; };  // { in a comment
  return f(1) - 2;
}
}  // namespace

struct Box {
  int v;
  int Get() const { return v * 2; }
  bool operator==(const Box& o) const { return v == o.v; }
};

Parser::Parser(int a) : a_(a), b_{a + 1} {
  const char* s = "}{";
  char c = '{';
  (void)s; (void)c;
}

std::string_view Parser::Name(int i) const {
  /* } */
  if (i < 0) {
    return "neg";
  }
  return R"x(})x";
}

bool operator<(const Box& a, const Box& b) { return a.v < b.v; }
}  // namespace halo::core
'''


def line_of(text, needle):
    return text[:text.index(needle)].count('\n') + 1


class FunctionNames(unittest.TestCase):
    def test_names_and_scopes(self):
        cases = {
            'return a + b': 'Helper',          # a lambda belongs to its function
            'return f(1) - 2': 'Helper',
            'return v * 2': 'Box::Get',        # in-class method keeps its class
            'return v == o.v': 'Box::operator==',
            'const char* s': 'Parser::Parser',  # brace in an init list is not the body
            'return "neg"': 'Parser::Name',
            'return R"x': 'Parser::Name',      # braces in raw strings and comments ignored
            'return a.v < b.v': 'operator<',
            'kTable[]': '<global>',            # a namespace-scope initializer
        }
        for needle, want in cases.items():
            with self.subTest(needle=needle):
                self.assertEqual(ms.enclosing(SRC, line_of(SRC, needle)), want)

    def test_blank_keeps_lines(self):
        self.assertEqual(ms._blank(SRC).count('\n'), SRC.count('\n'))
        self.assertEqual(len(ms._blank(SRC)), len(SRC))

    def test_out_of_range_line(self):
        self.assertEqual(ms.enclosing(SRC, 0), '<global>')
        self.assertEqual(ms.enclosing(SRC, 10_000), '<global>')


def make_report(path, rows):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE mutant (mutant_id TEXT, mutator TEXT, filename TEXT, directory TEXT, '
               'line_number INT, column_number INT, end_line_number INT, end_column_number INT, '
               'execution_status INT, exit_status INT, duration INT, stdout TEXT, stderr TEXT, '
               'mutation_replacement TEXT)')
    for mutator, fn, line, col, st in rows:
        mid = f'{mutator}:{fn}:{line}:{col}:{line}:{col + 1}'
        db.execute('INSERT INTO mutant VALUES (?,?,?,?,?,?,?,?,?,0,0,"","","")',
                   (mid, mutator, fn, '', line, col, line, col + 1, st))
    db.commit()
    db.close()


class Reports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, 'src'))
        self.src = os.path.join(self.root, 'src', 'x.cpp')
        with open(self.src, 'w') as fh:
            fh.write(SRC)
        self.h = line_of(SRC, 'return f(1) - 2')
        self.n = line_of(SRC, 'return "neg"')

    def tearDown(self):
        self.tmp.cleanup()

    def report(self, name, rows):
        p = os.path.join(self.root, name)
        make_report(p, rows)
        return p

    def test_killed_by_any_binary_and_numbering(self):
        a = self.report('a.sqlite', [
            ('cxx_sub_to_add', self.src, self.h, 14, ms.PASSED),
            ('cxx_lt_to_le', self.src, self.n, 9, ms.PASSED),
            ('cxx_lt_to_le', self.src, self.n, 3, ms.PASSED),
            ('cxx_eq_to_ne', self.src, self.n, 5, ms.PASSED),
        ])
        b = self.report('b.sqlite', [
            ('cxx_sub_to_add', self.src, self.h, 14, ms.FAILED),   # killed elsewhere
            ('cxx_eq_to_ne', self.src, self.n, 5, ms.TIMEDOUT),    # a hang counts as killed
            ('cxx_lt_to_le', self.src, self.n, 9, ms.PASSED),
        ])
        keys = ms.survivor_keys([a, b], self.root)
        self.assertEqual(keys, ['src/x.cpp Parser::Name cxx_lt_to_le',
                                'src/x.cpp Parser::Name cxx_lt_to_le #2'])

    def test_keys_ignore_line_moves(self):
        a = self.report('a.sqlite', [('cxx_sub_to_add', self.src, self.h, 14, ms.PASSED)])
        before = ms.survivor_keys([a], self.root)
        with open(self.src, 'w') as fh:
            fh.write('// a new comment line\n\n' + SRC)
        b = self.report('b.sqlite', [('cxx_sub_to_add', self.src, self.h + 2, 14, ms.PASSED)])
        self.assertEqual(ms.survivor_keys([b], self.root), before)

    def test_summary_counts(self):
        a = self.report('a.sqlite', [
            ('cxx_sub_to_add', self.src, self.h, 14, ms.FAILED),
            ('cxx_lt_to_le', self.src, self.n, 9, ms.PASSED),
            ('cxx_eq_to_ne', self.src, self.n, 5, ms.TIMEDOUT),
            ('cxx_eq_to_ne', self.src, self.n, 7, ms.CRASHED),
        ])
        row = [l for l in ms.summary([a], self.root).splitlines() if l.startswith('src/x.cpp')][0]
        self.assertEqual(row.split(), ['src/x.cpp', '2', '1', '1', '75.0%'])


class Check(unittest.TestCase):
    def run_check(self, baseline, survivors):
        with tempfile.TemporaryDirectory() as d:
            b, s = os.path.join(d, 'baseline.txt'), os.path.join(d, 'surv.txt')
            with open(b, 'w') as fh:
                fh.write(baseline)
            with open(s, 'w') as fh:
                fh.write(survivors)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = ms.check(b, [s])
            return rc, out.getvalue()

    def test_accepted(self):
        rc, _ = self.run_check('# comment\n\nsrc/x.cpp F cxx_lt_to_le | equivalent\n',
                               'src/x.cpp F cxx_lt_to_le\n')
        self.assertEqual(rc, 0)

    def test_new_survivor_fails(self):
        rc, out = self.run_check('src/x.cpp F cxx_lt_to_le | equivalent\n',
                                 'src/x.cpp F cxx_lt_to_le\nsrc/x.cpp F cxx_lt_to_le #2\n')
        self.assertEqual(rc, 1)
        self.assertIn('`src/x.cpp F cxx_lt_to_le #2`', out)

    def test_gone_entry_is_reported_not_fatal(self):
        rc, out = self.run_check('src/x.cpp F cxx_lt_to_le | equivalent\n', '')
        self.assertEqual(rc, 0)
        self.assertIn('no longer surviving', out)

    def test_reason_required(self):
        rc, out = self.run_check('src/x.cpp F cxx_lt_to_le\n', 'src/x.cpp F cxx_lt_to_le\n')
        self.assertEqual(rc, 1)
        self.assertIn('no reason', out)


if __name__ == '__main__':
    unittest.main()
