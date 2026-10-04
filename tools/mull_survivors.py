#!/usr/bin/env python3
"""Surviving Mull mutants against a baseline of accepted survivors.

    tools/mull_survivors.py summary REPORT.sqlite...            # killed/survived/timeout per file
    tools/mull_survivors.py keys REPORT.sqlite... > survivors.txt
    tools/mull_survivors.py where REPORT.sqlite...               # keys with file:line:col, to triage
    tools/mull_survivors.py check tests/mull-baseline.txt survivors.txt [--summary FILE]

REPORT.sqlite is what `mull-runner --reporters SQLite --report-name NAME` writes. Pass the
report of every test binary: a mutant counts as killed when any binary kills it (fails,
crashes or times out), and survives when every binary that ran it passed.

Each survivor is written as a key that does not move when lines do:

    <repo-relative file> <function> <mutator> [#n]    (root: MULL_SURVIVORS_ROOT, else the repo)
    e.g.  src/core/json.cpp Parser::number cxx_lt_to_le #2

<function> is the innermost function definition enclosing the mutant, found by a small
brace scanner over the source (class scopes kept, namespaces dropped; a lambda counts as
part of the function it is written in). #n numbers equal keys in source order, from #2.

`check` fails (exit 1) when a survivor is not in the baseline (a regression: a test lost
its teeth, or new code has none) or a baseline line has no reason, and lists baseline
entries that no longer survive (remove them). Every baseline line is `key | reason`;
blank lines and lines starting with `#` are ignored.
"""
import collections
import os
import re
import sqlite3
import sys

# Mull's ExecutionStatus (mull-state/src/execution_status.rs).
FAILED, PASSED, TIMEDOUT, CRASHED, ABNORMAL, DRYRUN, FAILFAST, NOTCOVERED = 1, 2, 3, 4, 5, 6, 7, 8
KILLED = {FAILED, CRASHED, ABNORMAL, FAILFAST}

SCOPE_KW = re.compile(r'\b(namespace|class|struct|union|enum)\b(?:\s+class|\s+struct)?\s*'
                      r'(?:\[\[[^\]]*\]\]\s*)?([A-Za-z_]\w*)?')
NAME = re.compile(r'((?:[A-Za-z_]\w*\s*(?:<[^<>()]*>)?\s*::\s*)*'
                  r'(?:operator\s*(?:\(\)|[^\s(]+)|~?[A-Za-z_]\w*))\s*$')
# a constructor's member-initializer list ending in a braced member init: `X::X(int a) : b_`
INIT_LIST = re.compile(r'\)\s*(?:noexcept\s*)?:(?!:)[^;{}]*[\w>]\s*$')
CONTROL = {'if', 'for', 'while', 'switch', 'catch', 'return', 'sizeof', 'decltype', 'alignof'}


def _blank(src):
    """Source with comments, string/char literals and preprocessor lines blanked out.
    Newlines are kept, so offsets map to the same lines."""
    out = list(src)
    i, n = 0, len(src)
    line_start = True
    while i < n:
        c = src[i]
        if line_start and c == '#':
            j = i
            while j < n and src[j] != '\n':
                if src[j] == '\\' and j + 1 < n and src[j + 1] == '\n':
                    j += 2
                    continue
                j += 1
            for k in range(i, j):
                if out[k] != '\n':
                    out[k] = ' '
            i = j
            continue
        if c == '\n':
            line_start = True
            i += 1
            continue
        if not c.isspace():
            line_start = False
        if src.startswith('//', i):
            j = src.find('\n', i)
            j = n if j < 0 else j
            out[i:j] = ' ' * (j - i)
            i = j
        elif src.startswith('/*', i):
            j = src.find('*/', i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != '\n':
                    out[k] = ' '
            i = j
        elif c == 'R' and src.startswith('R"', i):
            m = re.match(r'R"([^(\s]*)\(', src[i:])
            end = src.find(')' + m.group(1) + '"', i) if m else -1
            j = n if end < 0 else end + len(m.group(1)) + 2
            for k in range(i, j):
                if out[k] != '\n':
                    out[k] = ' '
            out[i] = '0'
            i = j
        elif c in '"\'' and not (c == "'" and i > 0 and src[i - 1].isalnum()):  # 1'000 digit separator
            j = i + 1
            while j < n and src[j] != c and src[j] != '\n':
                j += 2 if src[j] == '\\' else 1
            j = min(j + 1, n)
            out[i:j] = ' ' * (j - i)
            out[i] = '0'
            i = j
        else:
            i += 1
    return ''.join(out)


def _func_name(head):
    """Name of the function whose body follows `head`, or None if head is not a function."""
    # cut a constructor's member-initializer list: `Foo::Foo(int a) : x_(a), y_(b)`
    depth = 0
    for i, c in enumerate(head):
        if c in '(<[':
            depth += 1
        elif c in ')>]':
            depth -= 1
        elif c == ':' and depth == 0 and head[i - 1:i] != ':' and head[i + 1:i + 2] != ':':
            if ')' in head[:i]:
                head = head[:i]
                break
    if '(' not in head:
        return None
    if re.search(r'\]\s*(\([^()]*\))?\s*(mutable\s*)?(->[^{]*)?$', head.strip()) and '[' in head:
        return None  # a lambda at namespace scope
    # the name is what stands before the first top-level '(' (skipping `operator()`)
    m = re.search(r'\boperator\s*(\(\s*\)|[^\s(\w]+|\s\w[\w\s:*&<>]*?)\s*\(', head)
    if m:
        cut = m.end() - 1
    else:
        depth, cut = 0, -1
        for i, c in enumerate(head):
            if c == '<':
                depth += 1
            elif c == '>' and depth:
                depth -= 1
            elif c == '(' and depth == 0:
                cut = i
                break
        if cut < 0:
            return None
    m = NAME.search(head[:cut])
    if not m:
        return None
    before = head[:m.start(1)]
    if re.search(r'(?<![=!<>])=(?!=)', before) and 'operator' not in m.group(1):
        return None  # `auto x = f(...) {` is an initializer, not a definition
    name = re.sub(r'\s+', '', m.group(1))
    name = re.sub(r'<[^<>]*>', '', name)
    if name.split('::')[-1] in CONTROL:
        return None
    return name


def functions(src):
    """[(start offset, end offset, qualified name)] of the function bodies in src."""
    text = _blank(src)
    out = []
    stack = []  # ('ns'|'class'|'func'|'block', name, start)
    head_start = 0
    for i, c in enumerate(text):
        if c in ';':
            head_start = i + 1
        elif c == '{':
            head = text[head_start:i]
            in_func = any(k == 'func' for k, _, _ in stack)
            if not in_func and INIT_LIST.search(head):
                stack.append(('init', None, i))  # `: a_(a), b_{a}` -- the head goes on
                continue
            if in_func:
                stack.append(('block', None, i))
            else:
                m = SCOPE_KW.search(head)
                fname = _func_name(head)
                if m and fname is None and '(' not in head.split(m.group(0))[-1]:
                    kind = 'ns' if m.group(1) == 'namespace' else 'class'
                    if m.group(1) == 'enum':
                        kind = 'block'
                    stack.append((kind, m.group(2), i))
                elif fname is not None:
                    stack.append(('func', fname, i))
                elif re.search(r'extern\s*0?\s*$', head.strip()) or head.strip() == 'extern':
                    stack.append(('ns', None, i))
                else:
                    stack.append(('block', None, i))
            head_start = i + 1
        elif c == '}':
            if stack:
                kind, name, start = stack.pop()
                if kind == 'init':
                    continue
                if kind == 'func':
                    cls = [n for k, n, _ in stack if k == 'class' and n]
                    q = name if '::' in name or not cls else '::'.join(cls + [name])
                    out.append((start, i, q))
            head_start = i + 1
    return out


def enclosing(src, line, funcs=None):
    """Innermost function containing 1-based `line`, or '<global>'. `funcs` is
    functions(src), when the caller has it already."""
    starts = [0]
    for m in re.finditer('\n', src):
        starts.append(m.end())
    if line < 1 or line > len(starts):
        return '<global>'
    lo = starts[line - 1]
    hi = starts[line] - 1 if line < len(starts) else len(src)
    best = None
    for s, e, name in funcs if funcs is not None else functions(src):
        if s <= hi and e >= lo and (best is None or s >= best[0]):
            best = (s, e, name)
    return best[2] if best else '<global>'


def load(reports):
    """{mutant id: (file, line, col, mutator, status)} merged over reports: killed wins."""
    rank = {PASSED: 0, NOTCOVERED: 0, TIMEDOUT: 1}
    merged = {}
    for r in reports:
        db = sqlite3.connect(f'file:{r}?mode=ro', uri=True)
        for mid, mutator, fn, line, col, st in db.execute(
                'select mutant_id, mutator, filename, line_number, column_number, '
                'execution_status from mutant'):
            if st == DRYRUN:
                continue
            old = merged.get(mid)
            if old is None or rank.get(st, 2) > rank.get(old[4], 2):
                merged[mid] = (fn, int(line), int(col), mutator, st)
        db.close()
    return merged


def rel(path, root):
    p = os.path.normpath(os.path.relpath(path, root)) if os.path.isabs(path) else path
    return p.replace(os.sep, '/')


def survivors(reports, root):
    """[(key, 'file:line:col')] of the surviving mutants, sorted by key."""
    merged = load(reports)
    cache = {}
    pairs = []
    for fn, line, col, mutator, st in merged.values():
        if st not in (PASSED, NOTCOVERED):
            continue
        if fn not in cache:
            try:
                with open(fn, encoding='utf-8', errors='replace') as fh:
                    cache[fn] = fh.read()
            except OSError:
                cache[fn] = ''
            src = cache[fn]
            cache[fn] = (src, functions(src) if src else [])
        func = enclosing(cache[fn][0], line, cache[fn][1]) if cache[fn][0] else '<unknown>'
        f = rel(fn, root)
        pairs.append((f'{f} {func} {mutator}', (f, line, col, mutator)))
    seen = collections.Counter()
    out = []
    for base, (f, line, col, _) in sorted(pairs, key=lambda p: p[1]):
        seen[base] += 1
        out.append((base if seen[base] == 1 else f'{base} #{seen[base]}', f'{f}:{line}:{col}'))
    return sorted(out)


def survivor_keys(reports, root):
    return [k for k, _ in survivors(reports, root)]


def summary(reports, root):
    merged = load(reports)
    per = collections.defaultdict(collections.Counter)
    for fn, _, _, _, st in merged.values():
        f = rel(fn, root)
        per[f]['killed' if st in KILLED else 'timeout' if st == TIMEDOUT else 'survived'] += 1
    rows = [('file', 'killed', 'survived', 'timeout', 'score')]
    tot = collections.Counter()
    for f in sorted(per):
        c = per[f]
        tot.update(c)
        n = sum(c.values())
        rows.append((f, c['killed'], c['survived'], c['timeout'],
                     f"{100 * (c['killed'] + c['timeout']) / n:.1f}%"))
    n = sum(tot.values()) or 1
    rows.append(('TOTAL', tot['killed'], tot['survived'], tot['timeout'],
                 f"{100 * (tot['killed'] + tot['timeout']) / n:.1f}%"))
    w = max(len(str(r[0])) for r in rows)
    return '\n'.join(f'{r[0]:<{w}} {r[1]:>7} {r[2]:>8} {r[3]:>7} {r[4]:>7}' for r in rows)


def check(baseline, files, summary_path=None):
    base, bad = {}, []
    with open(baseline, encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    for n, l in enumerate(lines, 1):
        l = l.rstrip('\n')
        if not l.strip() or l.lstrip().startswith('#'):
            continue
        k, sep, reason = l.partition(' | ')
        if not sep or not reason.strip():
            bad.append(f'{baseline}:{n}: no reason: {l}')
        base[k.strip()] = reason.strip()
    surv = []
    for f in files:
        with open(f, encoding='utf-8') as fh:
            surv += [l.strip() for l in fh if l.strip()]
    new = [s for s in surv if s not in base]
    survset = set(surv)
    gone = [k for k in base if k not in survset]
    out = [f'## Mutation testing: {len(surv)} surviving mutants, {len(surv) - len(new)} accepted '
           f'in {baseline}, {len(new)} new', '']
    if new:
        out += ['### New survivors (not in the baseline): a test to kill each, '
                'or a baseline line with its reason', '']
        out += [f'- `{s}`' for s in new] + ['']
    if gone:
        out += ['### Baseline entries no longer surviving (remove them)', '']
        out += [f'- `{k}`' for k in gone] + ['']
    if bad:
        out += ['### Baseline lines without a reason', ''] + [f'- {b}' for b in bad] + ['']
    text = '\n'.join(out) + '\n'
    print(text)
    if summary_path:
        with open(summary_path, 'a', encoding='utf-8') as fh:
            fh.write(text)
    return 1 if new or bad else 0


def main(a):
    # keys name files relative to the repository (this script's parent's parent)
    root = os.environ.get('MULL_SURVIVORS_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if a[:1] == ['keys'] and len(a) >= 2:
        print('\n'.join(survivor_keys(a[1:], root)))
        return 0
    if a[:1] == ['where'] and len(a) >= 2:
        print('\n'.join(f'{k}\t{w}' for k, w in survivors(a[1:], root)))
        return 0
    if a[:1] == ['summary'] and len(a) >= 2:
        print(summary(a[1:], root))
        return 0
    if a[:1] == ['check'] and len(a) >= 3:
        s = None
        if '--summary' in a:
            i = a.index('--summary')
            s = a[i + 1]
            a = a[:i] + a[i + 2:]
        return check(a[1], a[2:], s)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
