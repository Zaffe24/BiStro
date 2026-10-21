#!/usr/bin/env python3
"""
Equivalence tests for the vectorised sbs96 code in bistro.normcountlib.

Compares the new implementation (this repo's src/) with the ORIGINAL code
(per-line / per-base Python loops) on synthetic and real inputs. Plain script,
no pytest: exits non-zero on the first failure.

    python tests/test_normcount.py           # quick tests (tests/data + synthetic), ~1-2 min
    python tests/test_normcount.py --real    # + real-data comparisons and timings
                                             #   (run through srun, e.g. -c 8 --mem 16G)
    python tests/test_normcount.py --only-e2e        # only the end-to-end check below
    python tests/test_normcount.py --e2e --e2e-fullref

    --e2e: `python -m bistro sbs96` (this repo) on the ORIGINAL pipeline runs'
    shared.context / shared.muts of A01 B01 E01 F01 in duplex and collapsed mode
    ($BISTRO_VALID_DIR/runs/orig_{mode}), with the original runs' exact options;
    each TSV must equal the original run's byte for byte (command line compared
    after swapping program and --out paths). Needs -c 16 (the runs used -t 16).
    --e2e-fullref: the same for A01 duplex with the full --ref FASTA scan, against
    the ORIGINAL code's output made by scratch/normcount/e2e/run_orig_fullref.sh.

Environment:
    BISTRO_ORIG_SRC   directory containing the ORIGINAL bistro/ package
                      (default: the refactor-validation snapshot)
    BISTRO_TEST_TMP   parent directory for temporary files (default: system tmp)
    BISTRO_REAL_DIR   directory with the --real inputs (see real_tests())
    BISTRO_VALID_DIR  refactor-validation dir (runs/, scratch/) for --e2e*
    BISTRO_E2E_OUT    where --e2e* write their outputs (default under scratch/normcount)

The original sbs96 needs nothing beyond python_env; generating the tests/data
shared.* inputs runs the ORIGINAL preprocess + somatic (mosdepth, bedtools).
"""

import gzip
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.join(os.path.dirname(HERE), "src")
DATA = os.path.join(HERE, "data")
ORIG_SRC = os.environ.get(
    "BISTRO_ORIG_SRC",
    "/home/csic/gpo/pza/lustre/bistro_refactor_validation/original_src")
REAL_DIR = os.environ.get(
    "BISTRO_REAL_DIR",
    "/home/csic/gpo/pza/lustre/bistro_refactor_validation/scratch/normcount/real")

sys.path.insert(0, REPO_SRC)
from bistro import normcountlib as new  # noqa: E402


def load_original():
    # Import the original package under another name, side by side with the new one.
    pkg = os.path.join(ORIG_SRC, "bistro")
    spec = importlib.util.spec_from_file_location(
        "orig_bistro", os.path.join(pkg, "__init__.py"), submodule_search_locations=[pkg])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orig_bistro"] = mod
    spec.loader.exec_module(mod)
    import orig_bistro.normcountlib as old
    return old


old = load_original()
TRI_LST, SBS96_LST, SBS96_TO_TRI = old.build_context_lists()
TMP = tempfile.mkdtemp(prefix="test_normcount_", dir=os.environ.get("BISTRO_TEST_TMP"))
N_CHECKS = 0


def check(cond, msg):
    global N_CHECKS
    N_CHECKS += 1
    if not cond:
        raise AssertionError(msg)


def run_py(src_dir, code, cwd=None, expect_ok=True):
    # Run a Python snippet with `src_dir` (dir containing bistro/) on PYTHONPATH,
    # from a cwd without a bistro/ dir. Returns the CompletedProcess.
    env = dict(os.environ, PYTHONPATH=src_dir)
    p = subprocess.run([sys.executable, "-c", code], cwd=cwd or TMP, env=env,
                       capture_output=True, text=True)
    if expect_ok and p.returncode != 0:
        raise AssertionError(f"subprocess failed:\n{code}\n{p.stderr}")
    return p


def orig_read_shared_context(path, nproc):
    # The ORIGINAL read_shared_context in a subprocess (its Pool needs an importable package).
    code = ("import json; from bistro import normcountlib as m; "
            "t = m.build_context_lists()[0]; "
            f"print(json.dumps(m.read_shared_context({path!r}, t, {nproc})))")
    p = run_py(ORIG_SRC, code)
    r, c = json.loads(p.stdout.strip().splitlines()[-1])
    return r, c


def same_dicts(a, b):
    # equal values AND key order AND Python-int types
    return (list(a.items()) == list(b.items())
            and all(type(v) is int for v in b.values()))


def reference_tally(lines):
    # The original per-line logic (body of _count_context_contig) on str lines.
    ref = {t: 0 for t in TRI_LST}
    ccs = {t: 0 for t in TRI_LST}
    for line in lines:
        _, _, _, triplet, depth = line.rstrip("\n").split("\t")
        ctri = old.canonical_tri(triplet)
        if ctri is None:
            continue
        ref[ctri] += 1
        ccs[ctri] += int(depth)
    return ref, ccs


def arrays_to_dicts(r, c):
    return ({t: int(r[i]) for i, t in enumerate(TRI_LST)},
            {t: int(c[i]) for i, t in enumerate(TRI_LST)})


def bgzip_tabix(path_plain):
    # bgzip + tabix -p bed (deletes the plain file); returns the .gz path
    gz = pysam.tabix_index(path_plain, preset="bed", force=True, keep_original=False)
    return gz


def write_gzip(path, text_bytes):
    with gzip.open(path, "wb") as fh:
        fh.write(text_bytes)
    return path


# ---------------------------------------------------------------------------
# 1. tally_context_chunk vs the original per-line logic
# ---------------------------------------------------------------------------
def test_fold_table():
    alphabet = "ACGTacgtNnRY-*"
    trips = [a + b + c for a in alphabet for b in alphabet for c in alphabet]
    trips += ["", "A", "AC", "ACGT", "CCCCC", "NNNN", "TT"]
    rng = random.Random(1)
    lines = [f"chr1\t{i}\t{i + 1}\t{t}\t{rng.randint(0, 99999)}\n" for i, t in enumerate(trips)]
    r, c = new.tally_context_chunk("".join(lines).encode())
    check(r.dtype == np.int64 and c.dtype == np.int64 and r.shape == (32,) and c.shape == (32,),
          "tally_context_chunk must return two int64[32]")
    check(arrays_to_dicts(r, c) == reference_tally(lines), "fold table differs from canonical_tri")
    # The genome scan upper-cases first: fold of _CODE_ANYCASE codes == canonical_tri(s.upper())
    for t in trips:
        if len(t) != 3:
            continue
        b = new._CODE_ANYCASE[np.frombuffer(t.encode(), np.uint8)]
        code = int(b[0]) * 25 + int(b[1]) * 5 + int(b[2])
        f = int(new._FOLD[code])
        exp = old.canonical_tri(t.upper())
        check((f == 32 and exp is None) or (exp is not None and TRI_LST[f] == exp),
              f"genome fold of {t!r}")
    print("  fold table: OK")


def random_context_lines(n, rng, weird=True):
    valid = "ACGT"
    other = "acgtNnRYSWKMBDHV-"
    lines = []
    pos = 0
    contigs = ["NC_000085.7", "chrUn.1", "c3"]
    for i in range(n):
        u = rng.random()
        if u < 0.70:
            tri = "".join(rng.choice(valid) for _ in range(3))
        elif u < 0.90:
            tri = "".join(rng.choice(valid + other) for _ in range(3))
        else:
            tri = "".join(rng.choice(valid) for _ in range(rng.choice([0, 1, 2, 4, 5])))
        u = rng.random()
        if u < 0.6:
            depth = str(rng.randint(0, 120))
        elif u < 0.9:
            # (sums are int64 in the new code: keep totals < 2**63; 19-digit
            # fields are covered by test_parse_int_fields / test_tally_wide_depth)
            depth = str(rng.choice([0, 9, 10, 99, 100, 999, 1000, 9999, 10000, 65535, 65536,
                                    10 ** 12, 10 ** 13 - 1]))
        elif weird:
            depth = rng.choice([" 5", "+7", "5 ", "1_000", "007", "5\r", "\t".join([]) + "12",
                                "0000000000000000000000042"])
        else:
            depth = str(rng.randint(0, 120))
        if old.canonical_tri(tri) is None and weird and rng.random() < 0.3:
            depth = rng.choice(["", "abc", "-", "1.5"])   # never parsed by the original
        pos += rng.randint(0, 3)
        lines.append(f"{contigs[i * 3 // n]}\t{pos}\t{pos + 1}\t{tri}\t{depth}\n")
    return lines


def test_tally_random():
    rng = random.Random(7)
    lines = random_context_lines(60000, rng)
    exp = reference_tally(lines)
    buf = "".join(lines).encode()
    check(arrays_to_dicts(*new.tally_context_chunk(buf)) == exp, "random tally (one chunk)")
    # arbitrary chunking at line boundaries sums to the same
    tot_r = np.zeros(32, np.int64)
    tot_c = np.zeros(32, np.int64)
    i = 0
    while i < len(lines):
        k = rng.randint(1, 5000)
        r, c = new.tally_context_chunk("".join(lines[i:i + k]).encode())
        tot_r += r
        tot_c += c
        i += k
    check(arrays_to_dicts(tot_r, tot_c) == exp, "random tally (chunked)")
    check(arrays_to_dicts(*new.tally_context_chunk(b"")) == arrays_to_dicts(
        np.zeros(32, np.int64), np.zeros(32, np.int64)), "empty chunk")
    # memoryview input (used after range trimming)
    check(arrays_to_dicts(*new.tally_context_chunk(memoryview(buf))) == exp, "memoryview input")
    print("  random tally (60000 lines, weird depths/triplets): OK")


def test_tally_wide_depth():
    # 18/19/25-digit DEPTH fields (vectorised limit is 18 digits; wider go through int())
    lines = [f"c\t{i}\t{i + 1}\tACG\t{d}\n" for i, d in enumerate(
        ["999999999999999999", "1000000000000000000", "0000000000000000000000042", "3"])]
    lines.append("c\t9\t10\tCCN\t" + "9" * 40 + "\n")      # skipped triplet: never parsed
    check(arrays_to_dicts(*new.tally_context_chunk("".join(lines).encode())) == reference_tally(lines),
          "wide DEPTH fields")
    print("  wide DEPTH fields: OK")


def test_tally_errors():
    good = "c\t1\t2\tACG\t5\n"
    bad_cases = {
        "4 fields": good + "c\t1\t2\tACG\n",
        "6 fields": good + "c\t1\t2\tACG\t5\t9\n",
        "empty line": good + "\n" + good,
        "garbage depth on valid triplet": good + "c\t1\t2\tACG\tabc\n",
        "empty depth on valid triplet": good + "c\t1\t2\tACG\t\n",
        "float depth on valid triplet": good + "c\t1\t2\tCCC\t1.5\n",
    }
    for name, text in bad_cases.items():
        lines = text.splitlines(keepends=True)
        try:
            reference_tally(lines)
            orig_raises = False
        except ValueError:
            orig_raises = True
        check(orig_raises, f"reference should raise for {name}")
        try:
            new.tally_context_chunk(text.encode())
            new_raises = False
        except ValueError:
            new_raises = True
        check(new_raises, f"tally_context_chunk should raise ValueError for {name}")
    try:
        new.tally_context_chunk(good.encode()[:-1])
        check(False, "chunk without trailing newline must raise")
    except ValueError:
        pass
    print("  malformed lines raise ValueError like the original: OK")


def test_parse_int_fields():
    vals = ["0", "7", "10", "65535", "123456789012345678", "999999999999999999",
            "1000000000000000000", " 3", "+4", "1_0", "6\r", "00012"]
    text = "\t".join(vals).encode()
    u8 = np.frombuffer(text, np.uint8)
    fs, fe, p = [], [], 0
    for v in vals:
        fs.append(p)
        fe.append(p + len(v.encode()))
        p += len(v.encode()) + 1
    out = new._parse_int_fields(u8, np.array(fs), np.array(fe))
    check([int(x) for x in out] == [int(v) for v in vals], "_parse_int_fields")
    print("  _parse_int_fields: OK")


# ---------------------------------------------------------------------------
# 2. read_shared_context vs the ORIGINAL on files (tabix, ranges, serial path)
# ---------------------------------------------------------------------------
HEADER = "##INTERVALS FROM X SHARED ACROSS: X Y\n##bistro_command=test\n"


def synthetic_context_file(name, rng, n_per_contig=(30000, 5000, 3), range_bp=1 << 14):
    # Sorted, 1-bp records + records sitting exactly on / around range boundaries
    # (multiples of range_bp), zero-length records (END == START, incl. at 0 and on
    # a boundary) and multi-bp records spanning boundaries.
    lines = []
    contigs = ["NC_000085.7", "chr.2", "tiny"]
    for contig, n in zip(contigs, n_per_contig):
        recs = []
        pos = 0
        for _ in range(n):
            pos += rng.randint(1, 4)
            recs.append((pos, pos + 1))
        if n > 1000:
            for b in range(range_bp, pos, range_bp):
                recs += [(b - 1, b), (b, b + 1), (b, b), (b - 1, b - 1), (b + 1, b + 1),
                         (b - 7, b + 3), (b - 2, b + 40)]
            recs += [(0, 0), (0, 1)]
        recs.sort()
        for s, e in recs:
            tri = "".join(rng.choice("ACGT") for _ in range(3)) if rng.random() < 0.9 \
                else rng.choice(["ANG", "acg", "RCT", "NNN"])
            lines.append(f"{contig}\t{s}\t{e}\t{tri}\t{rng.randint(1, 200)}\n")
    plain = os.path.join(TMP, name)
    with open(plain, "w") as fh:
        fh.write(HEADER)
        fh.writelines(lines)
    return bgzip_tabix(plain), lines


def new_rsc(path, nproc, **patch):
    # new read_shared_context with module attributes temporarily patched (in
    # this process only: pool children re-import the unpatched module)
    patch.setdefault("_fastio_failed", False)
    saved = {k: getattr(new, k) for k in patch}
    try:
        for k, v in patch.items():
            setattr(new, k, v)
        return new.read_shared_context(path, TRI_LST, nproc)
    finally:
        for k, v in saved.items():
            setattr(new, k, v)


def test_read_shared_context_files():
    rng = random.Random(11)
    path, lines = synthetic_context_file("syn1.bed", rng)
    exp_r, exp_c = orig_read_shared_context(path, 2)
    check(list(exp_r.keys()) == TRI_LST, "orig keys")
    check(sum(exp_r.values()) > 30000, "synthetic file too small?")
    # the original whole-contig fetch drops the zero-length record at 0; the
    # pure line tally does not -> make sure that edge is really present
    check(reference_tally(lines)[0] != exp_r, "expected zero-length record at 0 to be skipped by tabix")

    paths = [("pysam fallback", None)]
    if new._fastio_iter_tabix_chunks is not None:
        paths.append(("fastio", new._fastio_iter_tabix_chunks))
    for label, it in paths:
        for nproc, rbp, batch in [(1, None, None), (1, None, 7), (2, 1 << 14, None)]:
            patch = {"_fastio_iter_tabix_chunks": it}
            if batch:
                patch["_PYSAM_BATCH"] = batch
            if rbp:
                patch["CONTEXT_RANGE_BP"] = rbp
            r, c = new_rsc(path, nproc, **patch)
            check(same_dicts(exp_r, r) and same_dicts(exp_c, c),
                  f"read_shared_context [{label}] nproc={nproc} range={rbp} batch={batch}")
        # ranges in-process (the pool children re-import the module; here the
        # patched iterator and tiny batches really apply to the range workers)
        for rbp, batch in [(1 << 14, 3), (5000, 50), (1 << 14, 1 << 18)]:
            saved = (new._fastio_iter_tabix_chunks, new.CONTEXT_RANGE_BP, new._PYSAM_BATCH)
            new._fastio_iter_tabix_chunks, new.CONTEXT_RANGE_BP, new._PYSAM_BATCH = it, rbp, batch
            new._fastio_failed = False
            try:
                tasks = new._context_tasks(path, ["NC_000085.7", "chr.2", "tiny"], nproc=2)
                check(sum(t[2] is not None for t in tasks) > 3, "expected split ranges")
                r = np.zeros(32, np.int64)
                c = np.zeros(32, np.int64)
                for t in tasks:
                    a, b = new._count_context_task(t)
                    r += a
                    c += b
            finally:
                new._fastio_iter_tabix_chunks, new.CONTEXT_RANGE_BP, new._PYSAM_BATCH = saved
            check(not new._fastio_failed, f"fastio must not have failed [{label}]")
            dr, dc = arrays_to_dicts(r, c)
            check(dr == exp_r and dc == exp_c,
                  f"in-process ranges [{label}] range={rbp} batch={batch}")
    # the .tbi span parser sees every contig
    spans = new._tbi_spans(path)
    check(spans is not None and set(spans) == {"NC_000085.7", "chr.2", "tiny"}, "tbi spans")
    maxpos = {}
    for ln in lines:
        f = ln.split("\t")
        maxpos[f[0]] = max(maxpos.get(f[0], 0), int(f[1]))
    check(all(spans[k] > maxpos[k] for k in spans), "tbi span must exceed every START")

    # no index -> serial gzip path, in both implementations
    noidx = os.path.join(TMP, "syn1_noidx.bed.gz")
    shutil.copy(path, noidx)
    exp_r2, exp_c2 = orig_read_shared_context(noidx, 2)
    for nproc in (1, 3):
        r, c = new_rsc(noidx, nproc)
        check(same_dicts(exp_r2, r) and same_dicts(exp_c2, c), f"serial path nproc={nproc}")
    for chunk in (1000, 4096, 1 << 16):        # tiny gzip reads: lines split across reads
        r, c = new_rsc(noidx, 1, _GZIP_CHUNK=chunk)
        check(same_dicts(exp_r2, r) and same_dicts(exp_c2, c), f"serial path chunk={chunk}")
    print(f"  read_shared_context on tabix/no-index files ({', '.join(l for l, _ in paths)}; "
          f"whole contigs, split ranges, tiny batches): OK")


def test_reader_fallback():
    # The fastio reader is an optimisation: whenever it is unusable or fails
    # (before or after yielding data), the task is redone with pysam and the
    # result is unchanged; after a failure the process stops using fastio.
    rng = random.Random(13)
    path, _ = synthetic_context_file("syn_fb.bed", rng, n_per_contig=(20000, 3000, 5))
    exp_r, exp_c = orig_read_shared_context(path, 1)
    calls = []

    def raise_now(*a, **k):
        calls.append(a)
        raise FileNotFoundError("no tabix here")

    def raise_lazily(*a, **k):
        calls.append(a)
        raise RuntimeError("tabix failed (exit 1)")
        yield b""                              # generator: raises on first next()

    def raise_midway(p, contig, start=None, end=None, chunk_bytes=16 << 20):
        calls.append((p, contig))
        n = 0
        for buf in new._iter_tabix_lines_pysam(p, contig, start, end):
            yield buf
            n += 1
            if n == 2:                         # after real data was tallied
                raise RuntimeError("tabix died")

    def as_list(p, contig, start=None, end=None, chunk_bytes=16 << 20):
        calls.append((p, contig))              # a non-generator (no .close()) that works
        return list(new._iter_tabix_lines_pysam(p, contig, start, end))

    for label, fake, sticky in [("raise at call", raise_now, True),
                                ("raise on first chunk", raise_lazily, True),
                                ("raise after 2 chunks", raise_midway, True),
                                ("list result", as_list, False)]:
        for nproc, rbp in [(1, None), (2, 1 << 14)]:
            del calls[:]
            patch = {"_fastio_iter_tabix_chunks": fake, "_PYSAM_BATCH": 50}
            if rbp:
                patch["CONTEXT_RANGE_BP"] = rbp
            # nproc=2 runs the range tasks in-process (the pool would re-import
            # the real fastio), via the same task list read_shared_context builds
            if nproc == 1:
                r, c = new_rsc(path, 1, **patch)
            else:
                saved = {k: getattr(new, k) for k in list(patch) + ["_fastio_failed"]}
                for k, v in patch.items():
                    setattr(new, k, v)
                new._fastio_failed = False
                try:
                    tasks = new._context_tasks(path, ["NC_000085.7", "chr.2", "tiny"], 2)
                    check(len(tasks) > 3, "expected split ranges")
                    acc = [np.zeros(32, np.int64), np.zeros(32, np.int64)]
                    for t in tasks:
                        a, b = new._count_context_task(t)
                        acc[0] += a
                        acc[1] += b
                    failed = new._fastio_failed
                finally:
                    for k, v in saved.items():
                        setattr(new, k, v)
                r, c = arrays_to_dicts(*acc)
                check(failed == sticky, f"sticky flag [{label}]")
                check(len(calls) == (1 if sticky else len(tasks)),
                      f"fastio must not be retried after a failure [{label}]: {len(calls)} calls")
            check(r == exp_r and c == exp_c, f"fallback result [{label}] nproc={nproc}")
    # `tabix` not on PATH -> pysam without trying fastio
    del calls[:]
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = TMP                   # no tabix executable in there
    try:
        r, c = new_rsc(path, 1, _fastio_iter_tabix_chunks=raise_now)
    finally:
        os.environ["PATH"] = old_path
    check(not calls and r == exp_r and c == exp_c, "no tabix on PATH")
    # a malformed file still raises ValueError when fastio fails first
    plain = os.path.join(TMP, "bad_fb.bed")
    with open(plain, "w") as fh:
        fh.write(HEADER + "c1\t1\t2\tACG\t5\nc1\t2\t3\tACG\n")
    bad = bgzip_tabix(plain)
    for fake in (raise_now, new._fastio_iter_tabix_chunks):
        if fake is None:
            continue
        try:
            new_rsc(bad, 1, _fastio_iter_tabix_chunks=fake)
            check(False, "malformed file must raise ValueError")
        except ValueError:
            pass
    print("  tabix reader fallback (fastio raising at call / first chunk / mid-stream, "
          "non-generator, no tabix on PATH): OK")


def test_import_fallback():
    # normcountlib must import and give the original's tallies whatever state
    # fastio.py is in (absent, missing the function, syntax error, raising).
    rng = random.Random(17)
    path, _ = synthetic_context_file("syn_imp.bed", rng, n_per_contig=(5000, 500, 2))
    exp = orig_read_shared_context(path, 1)
    src_pkg = os.path.join(REPO_SRC, "bistro")
    variants = {
        "absent": None,
        "no_function": "X = 1\n",
        "syntax_error": "def iter_tabix_chunks(:\n",
        "import_error": "import _no_such_module_\n",
        "raises": "raise RuntimeError('half-written module')\n",
    }
    for name, text in variants.items():
        pkg_root = os.path.join(TMP, f"pkg_{name}")
        os.makedirs(os.path.join(pkg_root, "bistro"))
        for f in ("__init__.py", "arglib.py", "normcountlib.py"):
            shutil.copy(os.path.join(src_pkg, f), os.path.join(pkg_root, "bistro", f))
        if text is not None:
            with open(os.path.join(pkg_root, "bistro", "fastio.py"), "w") as fh:
                fh.write(text)
        code = ("import json; from bistro import normcountlib as m; "
                "assert m._fastio_iter_tabix_chunks is None; t = m.build_context_lists()[0]; "
                f"print(json.dumps([m.read_shared_context({path!r}, t, n) for n in (1, 2)]))")
        p = run_py(pkg_root, code)
        got = json.loads(p.stdout.strip().splitlines()[-1])
        check(all(g == list(exp) for g in got), f"import fallback [{name}]")
    print(f"  import with fastio.py {'/'.join(variants)}: OK")


def test_serial_text_mode_quirks():
    # The original's no-index loop reads in text mode (universal newlines).
    rng = random.Random(3)
    body = "".join(random_context_lines(3000, rng, weird=False))
    variants = {
        "crlf": HEADER + body.replace("\n", "\r\n"),
        "lone_cr": HEADER + body.replace("\n", "\r"),
        "no_final_newline": HEADER + body.rstrip("\n"),
        "mid_comments": HEADER + body[:5000].rsplit("\n", 1)[0] + "\n#comment\n# x\t1\n"
                        + body[5000:].split("\n", 1)[1],
        "header_only": HEADER,
        "empty": "",
        "cr_at_end": HEADER + body + "\r",
    }
    variants["cr_at_end_raises"] = variants.pop("cr_at_end")   # '\r' after the last '\n' = empty line
    variants["blank_line_raises"] = HEADER + body + "\n"
    for name, text in variants.items():
        p = write_gzip(os.path.join(TMP, f"quirk_{name}.bed.gz"), text.encode())
        if name.endswith("_raises"):
            code = ("from bistro import normcountlib as m; "
                    f"m.read_shared_context({p!r}, m.build_context_lists()[0], 1)")
            po = run_py(ORIG_SRC, code, expect_ok=False)
            check(po.returncode != 0 and "ValueError" in po.stderr, f"original should fail: {name}")
            for chunk in (37, 4096, 16 << 20):
                try:
                    new_rsc(p, 1, _GZIP_CHUNK=chunk)
                    check(False, f"new should raise ValueError: {name} chunk={chunk}")
                except ValueError:
                    pass
            continue
        exp = orig_read_shared_context(p, 1)
        for chunk in (37, 4096, 16 << 20):     # odd sizes split '\r\n' across reads
            r, c = new_rsc(p, 1, _GZIP_CHUNK=chunk)
            check(same_dicts(exp[0], r) and same_dicts(exp[1], c),
                  f"serial quirk {name} chunk={chunk}")
    # CRLF file with a tabix index: DEPTH keeps its '\r' (int() strips it)
    plain = os.path.join(TMP, "crlf_idx.bed")
    with open(plain, "w", newline="") as fh:
        fh.write(HEADER + body.replace("\n", "\r\n"))
    p = bgzip_tabix(plain)
    exp = orig_read_shared_context(p, 1)
    r, c = new_rsc(p, 1)
    check(same_dicts(exp[0], r) and same_dicts(exp[1], c), "tabix CRLF file")
    print("  serial no-index path text-mode quirks (CRLF, lone CR, comments, empty): OK")


def test_errors_on_files():
    for label, indexed in (("tabix", True), ("serial", False)):
        text = HEADER + "c1\t1\t2\tACG\t5\nc1\t2\t3\tACG\n"
        if indexed:
            plain = os.path.join(TMP, "bad_idx.bed")
            with open(plain, "w") as fh:
                fh.write(text)
            p = bgzip_tabix(plain)
        else:
            p = write_gzip(os.path.join(TMP, "bad_noidx.bed.gz"), text.encode())
        code = ("from bistro import normcountlib as m; "
                f"m.read_shared_context({p!r}, m.build_context_lists()[0], 1)")
        po = run_py(ORIG_SRC, code, expect_ok=False)
        check(po.returncode != 0 and "ValueError" in po.stderr, f"original should fail ({label})")
        try:
            new_rsc(p, 1)
            check(False, f"new should raise ValueError ({label})")
        except ValueError:
            pass
    print("  malformed files raise ValueError in both: OK")


# ---------------------------------------------------------------------------
# 3. get_genome_tricounts vs the original
# ---------------------------------------------------------------------------
def synthetic_fasta(rng, n_tiny=0, name="syn.fa"):
    seqs = {
        "len1": "A", "len2": "AC", "len3": "acg", "len4": "TTGA",
        "iupac": "ACGTRYSWKMBDHVNacgtrysn-*" * 41,
        "chr.mixed": "".join(rng.choice("ACGTACGTACGTacgtNn") for _ in range(120001)),
        "nrun": "N" * 5000 + "".join(rng.choice("ACGT") for _ in range(33333)) + "n" * 77,
    }
    for i in range(n_tiny):                    # many small scaffolds (task packing)
        seqs[f"scaf_{i}"] = "".join(rng.choice("ACGTacgtN") for _ in range(rng.randint(1, 400)))
    plain = os.path.join(TMP, name)
    with open(plain, "w") as fh:
        for name, s in seqs.items():
            fh.write(f">{name} desc\n")
            for i in range(0, len(s), 60):
                fh.write(s[i:i + 60] + "\n")
    gz = plain + ".gz"
    subprocess.run(["bgzip", "-f", plain], check=True)
    pysam.faidx(gz)
    return gz


def check_genome_tasks(ref, range_bp, block):
    # every window [0, len-2) of every contig in exactly one piece; tasks <= range_bp
    # windows unless a single piece; one FASTA open per task
    fa = pysam.FastaFile(ref)
    contigs = list(zip(fa.references, fa.lengths))
    fa.close()
    tasks = new._genome_tasks(ref, contigs, range_bp, block)
    seen = {}
    for r, pieces, blk in tasks:
        check(r == ref and blk == block and pieces, "task layout")
        check(sum(e - s for _, s, e in pieces) <= range_bp, "task size")
        for c, s, e in pieces:
            check(0 <= s < e, "piece bounds")
            seen.setdefault(c, []).append((s, e))
    for c, length in contigs:
        iv = sorted(seen.get(c, []))
        n_win = max(length - 2, 0)
        covered = 0
        for s, e in iv:
            check(s == covered, f"gap/overlap in pieces of {c}")
            covered = e
        check(covered == n_win, f"pieces of {c} cover {covered} of {n_win} windows")
    return tasks


def test_genome_tricounts():
    rng = random.Random(5)
    for ref in (os.path.join(DATA, "reference.fa.gz"), synthetic_fasta(rng),
                synthetic_fasta(rng, n_tiny=600, name="syn_tiny.fa")):
        exp = old.get_genome_tricounts(ref, TRI_LST)
        for rbp, blk in [(1 << 25, 1 << 22), (1000, 333), (997, 50), (5, 2)]:
            check_genome_tasks(ref, rbp, blk)
        if "tiny" in ref:
            check(len(check_genome_tasks(ref, 1 << 25, 1 << 22)) == 1,
                  "small contigs must share a task")
        for nproc, rbp, blk in [(1, None, None), (3, None, None), (1, 1000, 333), (3, 997, 50),
                                (2, 1 << 25, 7), (1, 101, 2)]:
            saved = (new.GENOME_RANGE_BP, new._GENOME_BLOCK_BP)
            if rbp:
                new.GENOME_RANGE_BP = rbp
            if blk:
                new._GENOME_BLOCK_BP = blk
            try:
                got = new.get_genome_tricounts(ref, TRI_LST, nproc)
            finally:
                new.GENOME_RANGE_BP, new._GENOME_BLOCK_BP = saved
            check(same_dicts(exp, got),
                  f"genome tricounts {os.path.basename(ref)} nproc={nproc} range={rbp} block={blk}")
        # default nproc keeps the old 2-argument call working
        check(same_dicts(exp, new.get_genome_tricounts(ref, TRI_LST)), "2-arg call")
    print("  get_genome_tricounts (tests/data + synthetic lowercase/N/IUPAC/short contigs): OK")


# ---------------------------------------------------------------------------
# 4. full sbs96 TSV byte identity (main())
# ---------------------------------------------------------------------------
def run_sbs96(src_dir, context, muts, ref, out, nproc, preset=None, no_plot=True):
    # Call normcountlib.main() with a fixed sys.argv so even the command-line
    # header line is identical between implementations.
    argv = ["bistro", "sbs96", "--context", context, "--muts", muts, "--out", out, "-t", str(nproc)]
    argv += ["--preset_genome", preset] if preset else ["--ref", ref]
    if no_plot:
        argv.append("--no_plot")
    code = ("import sys, time; from bistro import normcountlib as m; "
            f"sys.argv = {argv!r}; t = time.perf_counter(); "
            f"m.main({context!r}, {muts!r}, {ref!r}, {out!r}, None, None, {no_plot!r}, "
            f"{nproc}, {preset!r}); print('ELAPSED', time.perf_counter() - t)")
    p = run_py(src_dir, code)
    return float(p.stdout.strip().split("ELAPSED")[-1])


def make_testdata_shared_inputs():
    # Original preprocess + somatic on tests/data, as tests/run_smoke_test.sh does.
    out = os.path.join(TMP, "smoke")
    os.makedirs(out)
    env = dict(os.environ, PYTHONPATH=ORIG_SRC)
    common = ["--ref", os.path.join(DATA, "reference.fa.gz"), "--out_dir", out, "-t", "1",
              "--min_mapq", "0", "--min_bq", "0", "--min_sequence_identity", "0",
              "--min_qlen", "0", "--min_ec", "0", "--min_depth", "1"]
    for s in ("E01", "F01"):
        subprocess.run([sys.executable, "-m", "bistro", "preprocess", "--bam",
                        os.path.join(DATA, f"{s}.bam"), "--sample", s] + common,
                       cwd=TMP, env=env, check=True, capture_output=True)
    subprocess.run([sys.executable, "-m", "bistro", "somatic", "-i",
                    os.path.join(out, "E01.muts.bed.gz"), os.path.join(out, "F01.muts.bed.gz"),
                    "--bams", os.path.join(DATA, "E01.bam"), os.path.join(DATA, "F01.bam"),
                    "--ref", os.path.join(DATA, "reference.fa.gz"), "--thr", "1",
                    "--max_muts_per_duplex", "5", "-t", "1", "--min_mapq", "0",
                    "--min_baseq", "0", "--min_alt_support", "1"],
                   cwd=TMP, env=env, check=True, capture_output=True)
    return out


def compare_sbs96(label, context, muts, ref, nprocs=(1, 2), preset=None):
    # Same --out path and argv for both implementations, so the whole file
    # (incl. the ##bistro_normcounts_command= line) must be byte-identical.
    out = os.path.join(TMP, f"{label}.normcounts.tsv")
    run_sbs96(ORIG_SRC, context, muts, ref, out, 2, preset)
    want = open(out, "rb").read()
    os.remove(out)
    run_sbs96(REPO_SRC, context, muts, ref, out, 2, preset)
    check(open(out, "rb").read() == want, f"sbs96 TSV differs: {label} -t 2")
    for nproc in nprocs:                       # other -t: only the command line may differ
        run_sbs96(REPO_SRC, context, muts, ref, out, nproc, preset)
        got = open(out, "rb").read()
        check(got.split(b"\n", 1)[1] == want.split(b"\n", 1)[1],
              f"sbs96 TSV body differs: {label} -t {nproc}")
    return want


def test_sbs96_tsv():
    if not os.path.isdir(os.path.join(ORIG_SRC, "bistro")):
        print("  SKIP sbs96 TSV identity: original snapshot not found")
        return
    smoke = make_testdata_shared_inputs()
    ref = os.path.join(DATA, "reference.fa.gz")
    for s in ("E01", "F01"):
        ctx = os.path.join(smoke, f"{s}.shared.context.bed.gz")
        muts = os.path.join(smoke, f"{s}.shared.muts.bed.gz")
        want = compare_sbs96(f"smoke_{s}", ctx, muts, ref)
        check(want.count(b"\n") == 98, "96 rows + 2 header lines")
        compare_sbs96(f"smoke_{s}_preset", ctx, muts, ref, nprocs=(1,), preset="GRCm39")
        # the same with no tabix index (serial path)
        noidx = os.path.join(TMP, f"{s}.noidx.shared.context.bed.gz")
        shutil.copy(ctx, noidx)
        compare_sbs96(f"smoke_{s}_noidx", noidx, muts, ref, nprocs=(1,))
    # bigger synthetic context + the E01 muts
    syn, _ = synthetic_context_file("syn_tsv.bed", random.Random(21))
    compare_sbs96("synthetic", syn, os.path.join(smoke, "E01.shared.muts.bed.gz"), ref)
    # plot still renders (not compared: PDF carries a creation date)
    out = os.path.join(TMP, "plot.normcounts.tsv")
    run_sbs96(REPO_SRC, os.path.join(smoke, "E01.shared.context.bed.gz"),
              os.path.join(smoke, "E01.shared.muts.bed.gz"), ref, out, 2, no_plot=False)
    check(os.path.getsize(out[:-4] + ".png") > 0 and os.path.getsize(out[:-4] + ".pdf") > 0,
          "plot files")
    print("  sbs96 TSV byte-identical on tests/data-derived inputs "
          "(--ref, --preset_genome, no index, synthetic context; -t 1/2): OK")


# ---------------------------------------------------------------------------
# 5. real data (--real; run through srun)
# ---------------------------------------------------------------------------
def real_tests():
    """Inputs in BISTRO_REAL_DIR:
    A01.sub.shared.context.bed.gz(.tbi)  tabix -h extract of a real shared context
                                          (e.g. NC_000085.7 + small contigs), bgzip + tabix -p bed
    NC_000085.7.fa(.fai)                  one real chromosome of the reference
    muts: BISTRO_REAL_MUTS (default A01.shared.muts.bed.gz of the 081026 collapsed run)
    """
    ctx = os.path.join(REAL_DIR, "A01.sub.shared.context.bed.gz")
    fa = os.path.join(REAL_DIR, "NC_000085.7.fa")
    muts = os.environ.get("BISTRO_REAL_MUTS",
                          "/home/csic/gpo/pza/lustre/bistro/081026_collapsed_bq70/A01.shared.muts.bed.gz")
    ncpu = len(os.sched_getaffinity(0))
    print(f"  real data in {REAL_DIR} ({ncpu} CPUs)")

    # genome composition of one real chromosome
    t = time.perf_counter()
    exp = old.get_genome_tricounts(fa, TRI_LST)
    t_old = time.perf_counter() - t
    for nproc in sorted({1, ncpu}):
        t = time.perf_counter()
        got = new.get_genome_tricounts(fa, TRI_LST, nproc)
        t_new = time.perf_counter() - t
        check(same_dicts(exp, got), f"real genome tricounts nproc={nproc}")
        print(f"    genome tricounts NC_000085.7 ({sum(exp.values())} windows): original {t_old:.1f}s, "
              f"new nproc={nproc} {t_new:.2f}s ({t_old / t_new:.0f}x)")

    # shared context tally
    t = time.perf_counter()
    exp_r, exp_c = orig_read_shared_context(ctx, ncpu)
    t_old = time.perf_counter() - t
    t = time.perf_counter()
    exp_r1, exp_c1 = orig_read_shared_context(ctx, 1)
    t_old1 = time.perf_counter() - t
    check(exp_r1 == exp_r and exp_c1 == exp_c, "original nproc consistency")
    variants = [("pysam fallback", None)]
    if new._fastio_iter_tabix_chunks is not None:
        variants.append(("fastio", new._fastio_iter_tabix_chunks))
    for label, it in variants:
        for nproc in sorted({1, ncpu}):
            saved = new._fastio_iter_tabix_chunks
            new._fastio_iter_tabix_chunks = it
            try:
                t = time.perf_counter()
                r, c = new.read_shared_context(ctx, TRI_LST, nproc)
                t_new = time.perf_counter() - t
            finally:
                new._fastio_iter_tabix_chunks = saved
            check(same_dicts(exp_r, r) and same_dicts(exp_c, c),
                  f"real read_shared_context [{label}] nproc={nproc}")
            note = "" if nproc == 1 or label == "fastio" or saved is None else \
                " (pool workers use fastio when importable)"
            print(f"    shared context ({sum(exp_r.values())} rows) [{label}{note}] nproc={nproc}: "
                  f"{t_new:.1f}s vs original nproc=1 {t_old1:.1f}s / nproc={ncpu} {t_old:.1f}s")
    noidx = os.path.join(TMP, "real_noidx.shared.context.bed.gz")
    os.symlink(ctx, noidx)
    t = time.perf_counter()
    exp_r2, exp_c2 = orig_read_shared_context(noidx, 1)
    t_old = time.perf_counter() - t
    check(exp_r2 == exp_r and exp_c2 == exp_c, "original serial == tabix path")
    t = time.perf_counter()
    r, c = new.read_shared_context(noidx, TRI_LST, 1)
    t_new = time.perf_counter() - t
    check(same_dicts(exp_r2, r) and same_dicts(exp_c2, c), "real serial path")
    print(f"    no-index serial path: original {t_old:.1f}s, new {t_new:.1f}s ({t_old / t_new:.1f}x)")

    # full sbs96 TSV
    out = os.path.join(TMP, "real.normcounts.tsv")     # same path + argv -> whole file comparable
    t_old = run_sbs96(ORIG_SRC, ctx, muts, fa, out, ncpu)
    want = open(out, "rb").read()
    os.remove(out)
    t_new = run_sbs96(REPO_SRC, ctx, muts, fa, out, ncpu)
    check(open(out, "rb").read() == want, "real sbs96 TSV")
    check(want.count(b"\n") == 98, "real TSV rows")
    print(f"    sbs96 main() -t {ncpu} --ref NC_000085.7.fa: original {t_old:.1f}s, new {t_new:.1f}s "
          f"({t_old / t_new:.1f}x); TSV byte-identical")


# ---------------------------------------------------------------------------
# 6. end-to-end on the ORIGINAL pipeline's runs (--e2e, --e2e-fullref; srun)
# ---------------------------------------------------------------------------
VALID = os.environ.get("BISTRO_VALID_DIR", "/home/csic/gpo/pza/lustre/bistro_refactor_validation")
E2E_OUT = os.environ.get("BISTRO_E2E_OUT", os.path.join(VALID, "scratch/normcount/e2e/new"))
MOUSE_REF = os.environ.get(
    "BISTRO_MOUSE_REF",
    "/mnt/lustre/scratch/nlsas/home/csic/gpo/pza/mouse_genome/GCF_000001635.27_GRCm39_genomic.fa.gz")
GNU_TIME = "/mnt/netapp1/Optcesga_FT2_RHEL7/2020/gentoo/22072020/usr/bin/time"


def parse_gnu_time(text):
    # (wall seconds, max RSS MB) from `time -v` output
    wall = rss = float("nan")
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("Elapsed (wall clock) time"):
            parts = [float(x) for x in ln.rsplit(" ", 1)[1].split(":")]
            wall = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
        elif ln.startswith("Maximum resident set size (kbytes):"):
            rss = int(ln.rsplit(" ", 1)[1]) / 1024
    return wall, rss


def run_cli_sbs96(args, log):
    # `python -m bistro sbs96 <args>` with this repo's src on PYTHONPATH, under GNU time
    cmd = ([GNU_TIME, "-v"] if os.path.exists(GNU_TIME) else []) + \
          [sys.executable, "-m", "bistro", "sbs96"] + args
    env = dict(os.environ, PYTHONPATH=REPO_SRC)
    with open(log, "w") as fh:
        p = subprocess.run(cmd, cwd=TMP, env=env, stdout=fh, stderr=subprocess.STDOUT)
    text = open(log).read()
    check(p.returncode == 0, f"sbs96 CLI failed (see {log}):\n{text[-3000:]}")
    return parse_gnu_time(text)


def compare_with_original_tsv(orig_tsv, new_tsv, label):
    # Body (everything after the ##bistro_normcounts_command= line) byte-identical;
    # the command line equal once the program path and the --out path are swapped.
    want = open(orig_tsv, "rb").read()
    got = open(new_tsv, "rb").read()
    w1, wbody = want.split(b"\n", 1)
    g1, gbody = got.split(b"\n", 1)
    check(gbody == wbody, f"{label}: TSV body differs from the original run's ({orig_tsv} vs {new_tsv})")
    check(wbody.count(b"\n") == 97, f"{label}: header + 96 rows")
    orig_main = os.path.join(ORIG_SRC, "bistro", "__main__.py").encode()
    new_main = os.path.join(REPO_SRC, "bistro", "__main__.py").encode()
    w1 = w1.replace(orig_main, new_main).replace(
        b"--out " + os.path.abspath(orig_tsv).encode() + b" ",
        b"--out " + os.path.abspath(new_tsv).encode() + b" ")
    check(g1 == w1, f"{label}: command line\n  {g1!r}\n  {w1!r}")
    return wbody


def orig_args(orig_tsv):
    # argv the original run used (from its ##bistro_normcounts_command= line), minus argv[0]
    line = open(orig_tsv).readline().rstrip("\n")
    argv = line.split("=", 1)[1].split(" ")
    check(argv[1] == "sbs96", "original command line")
    return argv[2:]


def with_out(args, out):
    args = list(args)
    args[args.index("--out") + 1] = out
    return args


def e2e_tests():
    # sbs96 on the original runs' shared.context / shared.muts of every sample in
    # both modes, with the original runs' exact options (--preset_genome GRCm39,
    # -t 16, plot on); the TSV must match the original run's byte for byte.
    os.makedirs(E2E_OUT, exist_ok=True)
    print(f"  e2e outputs in {E2E_OUT} ({len(os.sched_getaffinity(0))} CPUs)")
    rows = []
    for mode in ("duplex", "collapsed"):
        run = os.path.join(VALID, "runs", f"orig_{mode}")
        for s in ("A01", "B01", "E01", "F01"):
            orig_tsv = os.path.join(run, f"{s}.normcounts.tsv")
            new_tsv = os.path.join(E2E_OUT, f"{mode}.{s}.normcounts.tsv")
            args = with_out(orig_args(orig_tsv), new_tsv)
            check("--preset_genome" in args, "original e2e runs used --preset_genome")
            wall, rss = run_cli_sbs96(args, new_tsv[:-4] + ".log")
            body = compare_with_original_tsv(orig_tsv, new_tsv, f"{mode} {s}")
            check(os.path.getsize(new_tsv[:-4] + ".png") > 0 and
                  os.path.getsize(new_tsv[:-4] + ".pdf") > 0, "plot files")
            o_wall, o_rss = parse_gnu_time(open(os.path.join(run, "logs", f"{s}.sbs96.log")).read())
            n_ref = sum(int(ln.split(b"\t")[15]) for ln in body.split(b"\n")[1:-1:3])
            n_ccs = sum(int(ln.split(b"\t")[16]) for ln in body.split(b"\n")[1:-1:3])
            n_dnm = sum(int(ln.split(b"\t")[3]) for ln in body.split(b"\n")[1:-1])
            rows.append((mode, s, n_ref, n_ccs, n_dnm, o_wall, wall, o_rss, rss))
            print(f"    {mode:9s} {s}: identical; R={n_ref} rows, C={n_ccs}, DNM counts={n_dnm}; "
                  f"wall {o_wall:.1f}s -> {wall:.1f}s ({o_wall / wall:.1f}x), "
                  f"max RSS {o_rss:.0f} -> {rss:.0f} MB")
    tot_o = sum(r[5] for r in rows)
    tot_n = sum(r[6] for r in rows)
    print(f"    total wall {tot_o:.0f}s -> {tot_n:.0f}s ({tot_o / tot_n:.1f}x) over {len(rows)} runs")


def e2e_fullref_tests():
    # --ref FASTA scan (no --preset_genome) of the whole mouse genome for one
    # sample, against the ORIGINAL code's output produced separately (slow:
    # ~45 min single-threaded genome loop; see scratch/normcount/e2e/run_orig_fullref.sh).
    orig_dir = os.path.join(VALID, "scratch/normcount/e2e/orig_fullref")
    orig_tsv = os.path.join(orig_dir, "A01.duplex.fullref.normcounts.tsv")
    if not os.path.exists(orig_tsv):
        print(f"  SKIP full --ref e2e: {orig_tsv} not found")
        return
    os.makedirs(E2E_OUT, exist_ok=True)
    o_wall, o_rss = parse_gnu_time(open(os.path.join(orig_dir, "run.log")).read())
    args = orig_args(orig_tsv)
    check("--preset_genome" not in args and "--ref" in args, "original fullref args")
    for t in (None, "16"):
        new_tsv = os.path.join(E2E_OUT, f"duplex.A01.fullref{'' if t is None else '.t' + t}.normcounts.tsv")
        a = with_out(args, new_tsv)
        if t is not None:
            a[a.index("-t") + 1] = t
        wall, rss = run_cli_sbs96(a, new_tsv[:-4] + ".log")
        if t is None:
            body = compare_with_original_tsv(orig_tsv, new_tsv, "fullref A01")
        else:
            check(open(new_tsv, "rb").read().split(b"\n", 1)[1] == body, "fullref -t 16 body")
        print(f"    full --ref A01 duplex ({' '.join(a[a.index('-t'):a.index('-t') + 2])}): identical; "
              f"wall {o_wall:.0f}s -> {wall:.1f}s ({o_wall / wall:.0f}x), max RSS {o_rss:.0f} -> {rss:.0f} MB")
    # Observation only (not a test of the new code): does the hardcoded GRCm39
    # table equal a scan of this FASTA?
    scanned = {ln.split(b"\t")[1].decode(): int(ln.split(b"\t")[14]) for ln in body.split(b"\n")[1:-1]}
    diff = {t: (new.MOUSE_TRINUC_COUNT[t], v) for t, v in scanned.items() if new.MOUSE_TRINUC_COUNT[t] != v}
    print(f"    MOUSE_TRINUC_COUNT (preset) vs scan of {os.path.basename(MOUSE_REF)}: "
          + ("identical" if not diff else f"{len(diff)} contexts differ, e.g. {list(diff.items())[:3]}"))


def main():
    args = sys.argv[1:]
    real = "--real" in args
    e2e = "--e2e" in args
    fullref = "--e2e-fullref" in args
    only_e2e = "--only-e2e" in args
    print(f"new: {new.__file__}\noriginal: {old.__file__}\ntmp: {TMP}")
    print(f"fastio.iter_tabix_chunks available: {new._fastio_iter_tabix_chunks is not None}")
    try:
        t0 = time.perf_counter()
        if not only_e2e:
            test_fold_table()
            test_parse_int_fields()
            test_tally_random()
            test_tally_wide_depth()
            test_tally_errors()
            test_genome_tricounts()
            test_read_shared_context_files()
            test_reader_fallback()
            test_import_fallback()
            test_serial_text_mode_quirks()
            test_errors_on_files()
            test_sbs96_tsv()
        if real:
            real_tests()
        if e2e or only_e2e:
            e2e_tests()
        if fullref:
            e2e_fullref_tests()
        print(f"ALL PASSED ({N_CHECKS} checks, {time.perf_counter() - t0:.0f}s)")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
