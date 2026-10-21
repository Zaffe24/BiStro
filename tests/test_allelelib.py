#!/usr/bin/env python
"""Equivalence tests: bistro.allelelib.count_alleles vs the ORIGINAL sharelib.count_alleles.

Plain script (no pytest).  Exits non-zero on any mismatch.

    python tests/test_allelelib.py            # quick (~3-4 CPU-min; run through srun):
                                              #   synthetic edge-case BAMs (every rule R1-R9 of
                                              #   bistro/allelelib.py), forkserver pool tasks,
                                              #   every site of the tests/data BAMs
    python tests/test_allelelib.py --real --nproc 6 [--bams A.bam ...] [--runs-dir DIR]
           [--skip-dnm] [--skip-sites] [--report out.tsv]
                                              # real subset BAMs (heavy, srun; ~1 CPU-h):
                                              #  (1) every COUNTS field of the original runs'
                                              #      *.shared.muts.bed.gz recounted in each other
                                              #      BAM must equal the file (and the original fn)
                                              #  (2) >= 5000 sites per BAM (random, deletions,
                                              #      insertion flanks, soft-clip edges, contig ends,
                                              #      all original-run sites, empty/absent contigs)
                                              #      x (mapq, baseq) 0/0, 30/70, 60/93 x 4 speed
                                              #      variants, vs the original; prints timings

The ORIGINAL function is embedded verbatim below (ORIGINAL_SRC) so the script is self-contained;
when the pristine snapshot is reachable its AST is checked to be identical to the embedded copy.
"""

import argparse
import array
import ast
import glob
import gzip
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np
import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from bistro import allelelib  # noqa: E402

ORIGINAL_SHARELIB = os.environ.get(
    "BISTRO_ORIGINAL_SHARELIB",
    "/home/csic/gpo/pza/lustre/bistro_refactor_validation/original_src/bistro/sharelib.py")
VALIDATION = "/home/csic/gpo/pza/lustre/bistro_refactor_validation"

# Verbatim copy of the original sharelib.BASE_INDEX + sharelib.count_alleles.
ORIGINAL_SRC = '''
BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def count_alleles(bam_path, positions_by_contig, min_mapq, min_baseq, max_depth=1_000_000):
    counts = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        references = set(bam.references)
        for contig, positions in positions_by_contig.items():
            if contig not in references:
                continue
            for pos in positions:
                row = [0, 0, 0, 0]
                counts[(contig, pos)] = row
                for col in bam.pileup(contig, pos - 1, pos,
                                      truncate=True,
                                      min_base_quality=0,
                                      min_mapping_quality=0,
                                      max_depth=max_depth):
                    for read in col.pileups:
                        if read.is_del or read.is_refskip:
                            continue
                        aln = read.alignment
                        if aln.is_secondary or aln.is_supplementary or aln.is_duplicate:
                            continue
                        if aln.mapping_quality < min_mapq:
                            continue
                        qpos = read.query_position
                        if qpos is None:
                            continue
                        quals = aln.query_qualities
                        if quals is None or quals[qpos] < min_baseq:
                            continue
                        idx = BASE_INDEX.get(aln.query_sequence[qpos].upper())
                        if idx is not None:  # N and other ambiguity codes are ignored
                            row[idx] += 1
    return counts
'''
_ns = {"pysam": pysam}
exec(compile(ORIGINAL_SRC, "<original sharelib.count_alleles>", "exec"), _ns)
orig_count_alleles_dict = _ns["count_alleles"]


def check_embedded_original():
    """The embedded copy must be AST-identical to the pristine snapshot (if reachable)."""
    if not os.path.exists(ORIGINAL_SHARELIB):
        print(f"[warn] {ORIGINAL_SHARELIB} not found; using embedded original copy unchecked")
        return
    with open(ORIGINAL_SHARELIB) as f:
        tree = ast.parse(f.read())
    want = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "count_alleles":
            want["f"] = node
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "BASE_INDEX"):
            want["b"] = node
    emb = {}
    for node in ast.parse(ORIGINAL_SRC).body:
        emb["f" if isinstance(node, ast.FunctionDef) else "b"] = node
    for k in ("f", "b"):
        assert ast.dump(want[k]) == ast.dump(emb[k]), f"embedded original differs from snapshot ({k})"
    print("[ok] embedded original count_alleles is AST-identical to the pristine snapshot")


def orig_counts(bam_path, contig, positions, min_mapq, min_baseq, max_depth=1_000_000):
    """Original function, reshaped to the new API: (n, 4) int64 in `positions` order."""
    d = orig_count_alleles_dict(bam_path, {contig: list(positions)}, min_mapq, min_baseq, max_depth)
    zeros = [0, 0, 0, 0]
    return np.array([d.get((contig, int(p)), zeros) for p in positions], dtype=np.int64).reshape(-1, 4)


class FallbackCounter:
    """Counts calls of allelelib._pileup_site (the exact per-site fallback)."""

    def __init__(self):
        self.calls = 0
        self._fn = allelelib._pileup_site

    def __call__(self, *a, **k):
        self.calls += 1
        return self._fn(*a, **k)


def install_counter():
    c = FallbackCounter()
    allelelib._pileup_site = c
    return c


VARIANTS = [  # (cluster_gap, _dense_sites): speed knobs only; every variant must give equal rows
    (allelelib.CLUSTER_GAP, allelelib._DENSE_SITES),
    (0, 10 ** 9),            # one fetch per site, scalar CIGAR walk only
    (10 ** 12, 1),           # one fetch for everything, vectorised per-read path only
    (500, 3),
]


def new_counts(bam, contig, positions, mq, bq, max_depth=1_000_000, variant=0):
    gap, dense = VARIANTS[variant]
    return allelelib.count_alleles(bam, contig, positions, mq, bq, max_depth=max_depth,
                                   cluster_gap=gap, _dense_sites=dense)


# --------------------------------------------------------------------------------------------
# Synthetic edge-case BAM
# --------------------------------------------------------------------------------------------
SYN_LEN = 3000
IUPAC = "ACGTACGTACGTACGTNRYKMacgtn="


def _rand_cigar(rng, allow_zero):
    ops = []
    if rng.random() < 0.1:
        ops.append((5, rng.randint(1, 10)))                  # H
    if rng.random() < 0.35:
        ops.append((4, rng.randint(1, 25)))                  # S
    for _ in range(rng.randint(1, 9)):
        r = rng.random()
        if r < 0.30:
            op, ln = 0, rng.randint(1, 60)                   # M
        elif r < 0.55:
            op, ln = 7, rng.randint(1, 60)                   # =
        elif r < 0.65:
            op, ln = 8, rng.randint(1, 4)                    # X
        elif r < 0.77:
            op, ln = 1, rng.randint(1, 12)                   # I
        elif r < 0.89:
            op, ln = 2, rng.randint(1, 12)                   # D
        elif r < 0.94:
            op, ln = 3, rng.randint(1, 80)                   # N
        else:
            op, ln = 6, rng.randint(1, 5)                    # P
        if allow_zero and rng.random() < 0.04:
            ln = 0
        ops.append((op, ln))
    if any(ln == 0 for _, ln in ops):
        # htslib parks one column on each empty op; enough trailing bases keep its qpos in range
        # (otherwise the ORIGINAL raises IndexError, see test_original_crash_replicated).
        ops.append((7, 20))
    if rng.random() < 0.35:
        ops.append((4, rng.randint(1, 25)))
    if rng.random() < 0.1:
        ops.append((5, rng.randint(1, 10)))
    if rng.random() < 0.03:                                  # rlen 0: insertion/clip only
        ops = [(4, 5), (1, 3)]
    ref_ops = [o for o in ops if o[0] in (0, 2, 3, 7, 8)]
    if len(ops) == 1 and ops[0][0] in (2, 3):
        ops = [(0, 10)] + ops + [(0, 10)]                    # single-op D/N: htslib UB, avoid
    if ref_ops and len(ops) > 1 and all(o[0] in (2, 3) for o in ref_ops):
        ops.append((0, 5))
    return ops


def _qlen(ops):
    return sum(ln for op, ln in ops if op in (0, 1, 4, 7, 8))


def make_synthetic_bam(path, seed=7, n_reads=900, quirks=False):
    """Coordinate-sorted indexed BAM with the flag/CIGAR/quality corners of rules R1-R7.

    quirks=True adds the R8 triggers (zero-length CIGAR ops, PROPER_PAIR reads incl. overlapping
    mates); quirks=False contains none, so every site must be served by the fast path."""
    hdr = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chrS", "LN": SYN_LEN}, {"SN": "chrE", "LN": 1000}]})
    rng = random.Random(seed)
    reads = []

    def mk(name, flag, start, ops, mapq=None, seq_mode=None, qual_mode=None):
        a = pysam.AlignedSegment(hdr)
        a.query_name = name
        a.flag = flag
        a.reference_id = 0
        a.reference_start = start
        a.mapping_quality = rng.choice([0, 1, 20, 29, 30, 31, 59, 60, 60, 60, 255]) if mapq is None else mapq
        q = _qlen(ops)
        if seq_mode is None:
            seq_mode = "none" if rng.random() < 0.02 else "seq"
        if seq_mode == "seq" and q > 0:
            a.query_sequence = "".join(rng.choice(IUPAC) for _ in range(q))
            qm = qual_mode or rng.choices(["ok", "none", "ff0", "ffmid"], [0.86, 0.05, 0.04, 0.05])[0]
            if qm == "none":
                a.query_qualities = None
            else:
                qq = array.array("B", (rng.randint(0, 93) for _ in range(q)))
                if qm == "ff0":
                    qq[0] = 255
                elif qm == "ffmid":
                    for _ in range(3):
                        qq[rng.randrange(q)] = 255
                a.query_qualities = qq
        a.cigartuples = ops if ops else None
        a.next_reference_id = -1
        a.next_reference_start = -1
        reads.append(a)
        return a

    for i in range(n_reads):
        flag = rng.choice([0, 16])
        for bit, p in [(0x800, 0.08), (0x100, 0.05), (0x400, 0.05), (0x200, 0.03), (0x4, 0.02)]:
            if rng.random() < p:
                flag |= bit
        r = rng.random()
        if r < 0.06:
            flag |= 0x1                                       # orphan: paired, not proper
        elif r < 0.09 and quirks:
            flag |= 0x2                                       # proper-pair bit without paired
        elif r < 0.11:
            flag |= 0x1 | 0x8                                 # mate unmapped (not proper)
        ops = _rand_cigar(rng, allow_zero=quirks)
        if rng.random() < 0.01:
            ops = []                                          # mapped flag but no CIGAR
        mk(f"r{i}", flag, rng.randint(0, SYN_LEN - 50), ops)

    # Overlapping proper pairs (htslib mate-overlap quality tweak; exact fallback R8b).
    for i in range(40 if quirks else 0):
        p = rng.randint(0, SYN_LEN - 300)
        d = rng.randint(0, 80)
        ln = rng.randint(60, 200)
        a = mk(f"pp{i}", 0x1 | 0x2 | 0x20 | 0x40, p, [(0, ln)], mapq=60, seq_mode="seq", qual_mode="ok")
        b = mk(f"pp{i}", 0x1 | 0x2 | 0x10 | 0x80, p + d, [(0, ln)], mapq=60, seq_mode="seq",
               qual_mode="ok")
        if rng.random() < 0.5:                                # identical bases on the overlap
            b.query_sequence = a.query_sequence[d:] + "".join(rng.choice("ACGT") for _ in range(d))
            b.query_qualities = array.array("B", (rng.randint(0, 93) for _ in range(ln)))
        for x, y in ((a, b), (b, a)):
            x.next_reference_id = 0
            x.next_reference_start = y.reference_start
            x.template_length = d + ln if x is a else -(d + ln)
    # A deep stack at one spot, for max_depth (R8a).
    for i in range(30):
        mk(f"deep{i}", 0, 1500 + (i % 3), [(4, 3), (0, 40)], mapq=60, seq_mode="seq", qual_mode="ok")
    # Placed-unmapped read.
    mk("unm", 0x4, 100, [], mapq=0)

    reads.sort(key=lambda a: a.reference_start)
    with pysam.AlignmentFile(path, "wb", header=hdr) as out:
        for a in reads:
            out.write(a)
    pysam.index(path)
    return path


def naive_counts(bam_path, contig, positions, mq, bq):
    """Textbook CIGAR-walk reference WITHOUT htslib's pileup quirks (R8); informational only."""
    pos = np.asarray(positions, dtype=np.int64)
    out = np.zeros((pos.size, 4), dtype=np.int64)
    index = {int(p) - 1: i for i, p in enumerate(pos)}
    with pysam.AlignmentFile(bam_path) as bam:
        if contig not in bam.references:
            return out
        for r in bam.fetch(contig):
            f = r.flag
            if f & 0xF04 or (f & 1 and not f & 2) or f & 0x800 or r.mapping_quality < mq:
                continue
            quals = r.query_qualities
            if quals is None:
                continue
            seq = r.query_sequence
            rp, qp = r.reference_start, 0
            for op, ln in r.cigartuples or ():   # (pysam get_aligned_pairs mishandles P ops)
                for _ in range(ln):
                    if op in (0, 7, 8):
                        i = index.get(rp)
                        if i is not None and not quals[qp] < bq and seq[qp] in "ACGT":
                            out[i, "ACGT".index(seq[qp])] += 1
                    if op in (0, 2, 3, 7, 8):
                        rp += 1
                    if op in (0, 1, 4, 7, 8):
                        qp += 1
    return out


def compare(tag, exp, got, positions, fails, limit=5):
    bad = np.flatnonzero(np.any(exp != got, axis=1))
    for i in bad[:limit]:
        print(f"  MISMATCH {tag} pos={int(positions[i])} orig={exp[i].tolist()} new={got[i].tolist()}")
    if bad.size:
        fails.append((tag, int(bad.size)))
    return int(bad.size)


def test_synthetic(tmpdir, fails):
    positions = np.arange(1, SYN_LEN + 12)
    combos = [(0, 0), (0, 1), (20, 30), (30, 70), (60, 93), (0, 94), (255, 0), (61, 0)]
    counter = install_counter()
    for quirks in (False, True):
        path = make_synthetic_bam(os.path.join(tmpdir, f"synthetic_{quirks}.bam"), quirks=quirks)
        fb0 = counter.calls
        n_naive_diff = 0
        for mq, bq in combos:
            exp = orig_counts(path, "chrS", positions, mq, bq)
            for v in range(len(VARIANTS)):
                compare(f"syn quirks={quirks} mq={mq} bq={bq} v={v}", exp,
                        new_counts(path, "chrS", positions, mq, bq, variant=v), positions, fails)
            n_naive_diff += int(np.any(exp != naive_counts(path, "chrS", positions, mq, bq), axis=1).sum())
        fb = counter.calls - fb0
        print(f"[synthetic quirks={quirks}] {len(combos)} combos x {positions.size} sites x "
              f"{len(VARIANTS)} variants equal; exact-fallback site calls: {fb}; original != textbook "
              f"CIGAR walk at {n_naive_diff} site-combos")
        if not quirks and (fb or n_naive_diff):
            fails.append((f"clean synthetic: fallback={fb} naive_diff={n_naive_diff} (both must be 0)", 1))
        if quirks and not (fb and n_naive_diff):
            fails.append(("quirks synthetic did not exercise the R8 fallback", 1))
    path = os.path.join(tmpdir, "synthetic_False.bam")
    fb0 = counter.calls
    for md in (5, 12, 0):                                     # max_depth guard (0 -> htslib 8000)
        exp = orig_counts(path, "chrS", positions, 0, 0, max_depth=md)
        for v in range(len(VARIANTS)):
            compare(f"syn max_depth={md} v={v}", exp,
                    new_counts(path, "chrS", positions, 0, 0, max_depth=md, variant=v), positions, fails)
    print(f"[synthetic max_depth 5/12/0] equal; exact-fallback site calls: {counter.calls - fb0}")
    for ctg in ("chrE", "chrNOPE"):                           # no reads / absent from header
        p = np.array([1, 2, 500, 999, 1000, 1001, 5000])
        exp = orig_counts(path, ctg, p, 0, 0)
        got = new_counts(path, ctg, p, 0, 0)
        assert not exp.any() and not got.any()
        compare(f"syn {ctg}", exp, got, p, fails)
    # Invalid input mirrors pysam (start out of range) / the contract (sorted unique).
    for bad in ([0, 5], [5, 3], [4, 4]):
        try:
            allelelib.count_alleles(path, "chrS", bad, 0, 0)
        except ValueError:
            pass
        else:
            fails.append((f"no ValueError for {bad}", 1))
    # ... but, like the original (which skips absent contigs before any pileup), position 0 on a
    # contig missing from the header is not an error: all-zero rows.
    p0 = np.array([0, 5])
    exp = orig_counts(path, "chrNOPE", p0, 0, 0)
    compare("syn chrNOPE pos 0", exp, allelelib.count_alleles(path, "chrNOPE", p0, 0, 0), p0, fails)
    assert allelelib.count_alleles(path, "chrS", [], 0, 0).shape == (0, 4)
    # task helper + open-handle input
    with pysam.AlignmentFile(path) as fh:
        c, p, k = allelelib.count_alleles_task((fh, "chrS", positions[:100], 0, 0))
        assert c == "chrS" and np.array_equal(p, positions[:100])
        compare("syn task", orig_counts(path, "chrS", positions[:100], 0, 0), k, positions[:100], fails)


def test_pool_tasks(fails):
    """count_alleles_task through a real pool (default start method = forkserver on 3.14: every
    task and result is pickled), with tasks from split_positions, reassembled by contig."""
    bam = os.path.join(HERE, "data", "E01.bam")
    with pysam.AlignmentFile(bam) as fh:
        lens = dict(zip(fh.references, fh.lengths))
    tasks = []
    for ctg, ln in lens.items():
        for chunk in allelelib.split_positions(np.arange(1, ln + 3), 997, cluster_gap=50):
            tasks.append((bam, ctg, chunk, 30, 70))
    tasks.append((bam, "chrNOT_IN_HEADER", np.array([1, 2, 3]), 30, 70))
    with mp.Pool(2) as pool:
        res = list(pool.imap_unordered(allelelib.count_alleles_task, tasks))
        res2 = pool.starmap(allelelib.count_alleles_task, tasks[:2])
    by_ctg = defaultdict(list)
    for ctg, pos, cnt in res + res2:
        by_ctg[ctg].append((pos, cnt))
    for ctg, parts in by_ctg.items():
        pos = np.concatenate([p for p, _ in parts])
        cnt = np.concatenate([c for _, c in parts])
        order = np.argsort(pos, kind="stable")
        pos, cnt = pos[order], cnt[order]
        uniq, first = np.unique(pos, return_index=True)          # starmap repeats two tasks
        compare(f"pool {ctg}", orig_counts(bam, ctg, uniq, 30, 70), cnt[first], uniq, fails)
    print(f"[ok] count_alleles_task via {mp.get_start_method()} pool: {len(tasks)} tasks")


def test_original_crash_replicated(tmpdir, fails):
    """CIGAR 21S0M10D: htslib reports qpos == l_qseq at the 0M column and the ORIGINAL raises
    IndexError on quals[qpos].  The new code sends that read's sites to the exact fallback, so it
    raises the same error instead of silently returning a different answer."""
    hdr = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6", "SO": "coordinate"},
                                           "SQ": [{"SN": "chrS", "LN": 500}]})
    path = os.path.join(tmpdir, "crash.bam")
    with pysam.AlignmentFile(path, "wb", header=hdr) as out:
        a = pysam.AlignedSegment(hdr)
        a.query_name, a.flag, a.reference_id, a.reference_start, a.mapping_quality = "x", 0, 0, 100, 60
        a.query_sequence = "A" * 21
        a.query_qualities = array.array("B", [30] * 21)
        a.cigartuples = [(4, 21), (0, 0), (2, 10)]
        out.write(a)
    pysam.index(path)
    errs = []
    for fn in (orig_counts, new_counts):
        try:
            fn(path, "chrS", np.arange(95, 115), 0, 0)
            errs.append(None)
        except Exception as e:  # noqa: BLE001
            errs.append(type(e).__name__)
    if errs != ["IndexError", "IndexError"]:
        fails.append((f"crash replication {errs}", 1))
    else:
        print("[ok] original IndexError on 21S0M10D replicated by the new code")


def test_split_positions(fails):
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(0, 400))
        pos = np.unique(rng.integers(1, 10 ** int(rng.integers(3, 8)), n))
        for ms in (1, 7, 50, 1000):
            ch = allelelib.split_positions(pos, ms)
            cat = np.concatenate(ch) if ch else np.zeros(0, dtype=np.int64)
            if not np.array_equal(cat, pos) or any(c.size == 0 for c in ch):
                fails.append(("split_positions", 1))
                return
    print("[ok] split_positions partitions positions")


def test_tests_data(fails):
    data = os.path.join(HERE, "data")
    t0 = time.time()
    nsites = 0
    for s in ("E01", "F01"):
        bam = os.path.join(data, f"{s}.bam")
        with pysam.AlignmentFile(bam) as fh:
            lens = dict(zip(fh.references, fh.lengths))
        for ctg, ln in lens.items():
            all_pos = np.arange(1, ln + 4)
            for k, (mq, bq) in enumerate([(0, 0), (30, 70), (60, 93), (20, 30)]):
                p = all_pos if k == 0 else all_pos[k::3]     # every site once, thinned for others
                exp = orig_counts(bam, ctg, p, mq, bq)
                for v in (0, 1, 2):
                    compare(f"tests/data {s} {ctg} mq={mq} bq={bq} v={v}", exp,
                            new_counts(bam, ctg, p, mq, bq, variant=v), p, fails)
                nsites += p.size
    print(f"[tests/data] {nsites} site-combos compared in {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------------------------
# Real subset BAMs (heavy; srun)
# --------------------------------------------------------------------------------------------
REAL_CONTIGS = {  # contig -> number of random sites
    "NC_000085.7": 2400, "NT_166280.1": 350, "NC_005089.1": 250,
    "NT_187064.1": 110, "NW_023337853.1": 110,
}
FEATURE_SITES = {"NC_000085.7": 1500, "NT_166280.1": 200, "NC_005089.1": 150,
                 "NT_187064.1": 50, "NW_023337853.1": 50}
COMBOS = [(0, 0), (30, 70), (60, 93)]
FEATURE_QUOTA = {"del": 0.4, "ins": 0.4, "clip": 0.2}   # share of FEATURE_SITES per category


def feature_positions(bam_path, contig, length, rng, n_windows=40, win=3000):
    """1-based sites at CIGAR features of real reads, by category: 'del' (inside and at the edges
    of deletions), 'ins' (reference bases flanking an insertion), 'clip' (soft-clip boundaries)."""
    feats = {"del": set(), "ins": set(), "clip": set()}
    with pysam.AlignmentFile(bam_path) as bam:
        for _ in range(n_windows):
            a = int(rng.integers(0, max(1, length - win)))
            for k, r in enumerate(bam.fetch(contig, a, a + win)):
                if k > 60:
                    break
                cig = r.cigartuples
                if not cig:
                    continue
                rp = r.reference_start
                ops = [op for op, _ in cig if op != 5]                # ignore hard clips
                if ops and ops[0] == 4:
                    feats["clip"].update((rp - 1, rp, rp + 1))        # left soft-clip boundary
                for op, ln in cig:
                    if op == 2:
                        feats["del"].update((rp - 1, rp, rp + ln // 2, rp + ln - 1, rp + ln))
                    elif op == 1:
                        feats["ins"].update((rp - 1, rp))             # bases flanking the insertion
                    if op in (0, 2, 3, 7, 8):
                        rp += ln
                if ops and ops[-1] == 4:
                    feats["clip"].update((rp - 2, rp - 1, rp))        # right soft-clip boundary
    return {k: np.array(sorted(p + 1 for p in v if 0 <= p < length + 5), dtype=np.int64)
            for k, v in feats.items()}


def _parse_somatic_cmd(run_dir):
    """(min_mapq, min_baseq, {sample: bam}) from the 'Command being timed' line of the run log."""
    log = os.path.join(run_dir, "logs", "somatic.log")
    mq, bq, bams = 30, 70, []
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                if "Command being timed" in line and " somatic " in line:
                    tok = line.split('"')[1].split()
                    for i, t in enumerate(tok):
                        if t == "--min_mapq":
                            mq = int(tok[i + 1])
                        elif t == "--min_baseq":
                            bq = int(tok[i + 1])
                        elif t == "--bams":
                            j = i + 1
                            while j < len(tok) and not tok[j].startswith("-"):
                                bams.append(tok[j])
                                j += 1
    else:
        print(f"[warn] {log} missing: assuming --min_mapq 30 --min_baseq 70")
    return mq, bq, bams


def original_count_rows(runs_dir):
    """Rows of the original runs' *.shared.muts.bed.gz whose COUNTS column was computed.

    COUNTS = one A:C:G:T field per OTHER sample, in TOTAL SAMPLES (-i) order excluding self; it is
    filled for every row that was DNM when flag_fp_denovo ran (some became FPD there).  Returns
    (rows, info): rows = [dict(file, sample, others, bams, contig, end, ann, fields, mq, bq)]."""
    rows, info = [], []
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "orig_*"))):
        files = sorted(glob.glob(os.path.join(run_dir, "*.shared.muts.bed.gz")))
        if not files:
            continue
        mq, bq, bam_list = _parse_somatic_cmd(run_dir)
        for fn in files:
            sample, total, cols = None, None, None
            n_dnm = n_rows = 0
            with gzip.open(fn, "rt") as f:
                for line in f:
                    if line.startswith("##sample="):
                        sample = line.rstrip("\n").split("=", 1)[1]
                    elif "TOTAL SAMPLES:" in line:
                        total = line.split("TOTAL SAMPLES:")[1].split()
                    elif line.startswith("#CONTIG"):
                        cols = line[1:].rstrip("\n").split("\t")
                    elif not line.startswith("#"):
                        rec = dict(zip(cols, line.rstrip("\n").split("\t")))
                        n_dnm += rec["ANNOTATION"] == "DNM"
                        assert rec["ANNOTATION"] != "DNM" or rec["COUNTS"] != ".", (fn, line)
                        if rec["COUNTS"] == ".":
                            continue
                        n_rows += 1
                        if bam_list:
                            assert len(bam_list) == len(total), (bam_list, total)
                            bams = dict(zip(total, bam_list))
                        else:
                            bams = {s: os.path.join(VALIDATION, "data", f"{s}.bam") for s in total}
                        others = [s for s in total if s != sample]
                        fields = rec["COUNTS"].split(",")
                        assert len(fields) == len(others), (fn, line)
                        rows.append(dict(file=fn, sample=sample, others=others, bams=bams,
                                         contig=rec["CONTIG"], end=int(rec["END"]),
                                         ann=rec["ANNOTATION"], fields=fields, mq=mq, bq=bq))
            info.append((fn, sample, n_dnm, n_rows, mq, bq))
    return rows, info


def dnm_sites(rows):
    """{contig: set(1-based END)} of every site piled up by the original runs."""
    out = defaultdict(set)
    for r in rows:
        out[r["contig"]].add(r["end"])
    return out


def build_real_sites(bam_path, seed, dnm):
    """{contig: sorted unique 1-based sites} for one BAM, plus per-category tallies."""
    rng = np.random.default_rng(seed)
    with pysam.AlignmentFile(bam_path) as fh:
        lens = dict(zip(fh.references, fh.lengths))
        mapped = {s.contig: s.mapped for s in fh.get_index_statistics()}
    sites, tally = {}, defaultdict(int)
    for ctg, n_rand in REAL_CONTIGS.items():
        L = lens[ctg]
        parts = [rng.integers(1, L + 1, n_rand)]
        tally["random"] += n_rand
        for cat, arr in feature_positions(bam_path, ctg, L, rng).items():
            k = int(FEATURE_SITES[ctg] * FEATURE_QUOTA[cat])
            if arr.size > k:
                arr = rng.choice(arr, k, replace=False)
            parts.append(arr)
            tally[cat] += arr.size
        ends = np.array([1, 2, 3, 100, L - 100, L - 2, L - 1, L, L + 1, L + 50])
        parts.append(ends)
        tally["contig_ends"] += ends.size
        dn = np.array(sorted(dnm.get(ctg, ())), dtype=np.int64)
        parts.append(dn)
        tally["orig_DNM"] += dn.size
        sites[ctg] = np.unique(np.concatenate(parts))
    empty = [c for c in lens if mapped.get(c, 0) == 0][:1]      # in header, no reads
    for ctg in empty + ["chrNOT_IN_HEADER"]:
        sites[ctg] = np.array([1, 2, 1000, 50_000, 1_000_000])
        tally["no_reads/absent"] += 5
    for ctg, d in dnm.items():                                   # DNMs on other contigs, if any
        if ctg not in sites:
            sites[ctg] = np.array(sorted(d), dtype=np.int64)
            tally["orig_DNM"] += len(d)
    return sites, dict(tally)


_COUNTER = None


def _init_worker():
    global _COUNTER
    _COUNTER = install_counter()


def _timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t


def real_job(job):
    """Compare original vs new on one (bam, contig, chunk) for all combos, timing both.

    The first combo of a job is the first touch of its BAM region ("cold"); which implementation
    pays it alternates between jobs (orig_first), so cold and warm costs of both can be reported."""
    bam, contig, pos, combos, max_depth, variants, orig_first = job
    res = []
    for k, (mq, bq) in enumerate(combos):
        if orig_first:
            exp, t_orig = _timed(orig_counts, bam, contig, pos, mq, bq, max_depth=max_depth)
        fb0 = _COUNTER.calls
        new1, t_new = _timed(new_counts, bam, contig, pos, mq, bq, max_depth=max_depth)
        fb = _COUNTER.calls - fb0
        if not orig_first:
            exp, t_orig = _timed(orig_counts, bam, contig, pos, mq, bq, max_depth=max_depth)
        new2, t_new2 = _timed(new_counts, bam, contig, pos, mq, bq, max_depth=max_depth)
        bad = [(int(pos[i]), exp[i].tolist(), new1[i].tolist())
               for i in np.flatnonzero(np.any(exp != new1, axis=1) | np.any(exp != new2, axis=1))]
        if variants:
            for v in range(1, len(VARIANTS)):
                got = new_counts(bam, contig, pos, mq, bq, max_depth=max_depth, variant=v)
                bad += [(int(pos[i]), exp[i].tolist(), got[i].tolist(), f"v{v}")
                        for i in np.flatnonzero(np.any(exp != got, axis=1))]
        cold = ("orig" if orig_first else "new") if k == 0 else None
        res.append(dict(bam=os.path.basename(bam), contig=contig, mq=mq, bq=bq, n=int(pos.size),
                        max_depth=max_depth, t_orig=t_orig, t_new=t_new, t_new2=t_new2, cold=cold,
                        fallback=fb, bad=bad, n_counted=int(exp.sum())))
    return res


def run_real(args, dnm):
    bams = args.bams or sorted(glob.glob(os.path.join(VALIDATION, "data", "*.bam")))
    jobs, per_bam = [], {}
    for bi, bam in enumerate(bams):
        sites, tally = build_real_sites(bam, 1000 + bi, dnm)
        per_bam[os.path.basename(bam)] = sum(p.size for p in sites.values())
        print(f"[real] {os.path.basename(bam)}: {per_bam[os.path.basename(bam)]} unique sites; "
              f"by contig { {c: int(p.size) for c, p in sites.items()} }; by origin {tally}")
        for ctg, pos in sites.items():
            chunk = 8 if ctg in ("NT_187064.1", "NW_023337853.1", "NC_005089.1") else 150
            for ci, c in enumerate(allelelib.split_positions(pos, chunk)):
                jobs.append((bam, ctg, c, COMBOS, 1_000_000, True, ci % 2 == 0))
        # small max_depth on real reads: the R8a guard must catch every affected site
        for ctg in ("NC_000085.7", "NT_187064.1"):
            sub = sites[ctg][:: max(1, sites[ctg].size // 24)][:24]
            jobs.append((bam, ctg, sub, [(0, 0)], 40, False, False))
    print(f"[real] jobs: {len(jobs)}")
    jobs.sort(key=lambda j: -(j[2].size * (60 if j[1] in ("NT_187064.1", "NW_023337853.1") else 1)))
    t0 = time.time()
    with mp.Pool(args.nproc, initializer=_init_worker) as pool:
        results = [r for rs in pool.imap_unordered(real_job, jobs) for r in rs]
    print(f"[real] wall {time.time() - t0:.0f}s with {args.nproc} processes")

    agg = defaultdict(lambda: defaultdict(float))
    all_bad = []
    for r in results:
        a = agg[(r["bam"], r["contig"], r["mq"], r["bq"], r["max_depth"])]
        a["n"] += r["n"]
        a["bad"] += len(r["bad"])
        a["fb"] += r["fallback"]
        a["counted"] += r["n_counted"]
        a["t_orig"] += r["t_orig"]
        a["t_new2"] += r["t_new2"]
        all_bad += [((r["bam"], r["contig"], r["mq"], r["bq"], r["max_depth"]), b) for b in r["bad"]]
    print(f"{'bam':8} {'contig':16} {'mq':>3} {'bq':>3} {'maxdp':>7} {'sites':>6} {'mism':>5} "
          f"{'fallbk':>6} {'reads/site':>10} {'orig ms/site':>12} {'new(warm)':>10}")
    for key in sorted(agg):
        a = agg[key]
        n = max(a["n"], 1)
        print(f"{key[0]:8} {key[1][:16]:16} {key[2]:>3} {key[3]:>3} {key[4]:>7} {int(a['n']):>6} "
              f"{int(a['bad']):>5} {int(a['fb']):>6} {a['counted'] / n:>10.1f} "
              f"{a['t_orig'] / n * 1e3:>12.2f} {a['t_new2'] / n * 1e3:>10.3f}")

    # Per-contig speed summary (all BAMs, default max_depth): cold = first touch of the region.
    spd = defaultdict(lambda: defaultdict(float))
    for r in results:
        if r["max_depth"] != 1_000_000:
            continue
        s = spd[r["contig"]]
        if r["cold"] == "orig":
            s["oc_t"] += r["t_orig"]
            s["oc_n"] += r["n"]
        else:
            s["ow_t"] += r["t_orig"]
            s["ow_n"] += r["n"]
        if r["cold"] == "new":
            s["nc_t"] += r["t_new"]
            s["nc_n"] += r["n"]
        s["nw_t"] += r["t_new2"]
        s["nw_n"] += r["n"]
    print("[real] per-site speed (ms/site, single process, while the pool's other processes "
          "compete for I/O):")
    print(f"  {'contig':16} {'orig cold':>10} {'new cold':>10} {'orig warm':>10} {'new warm':>10} "
          f"{'x cold':>7} {'x warm':>7}")
    tot = defaultdict(float)
    for ctg in sorted(spd):
        s = spd[ctg]
        for kk, vv in s.items():
            tot[kk] += vv
        _speed_row(ctg, s)
    _speed_row("ALL", tot)
    n_tot = sum(a["n"] for a in agg.values())
    n_bad = sum(a["bad"] for a in agg.values())
    print(f"[real] TOTAL site-combos compared {int(n_tot)} (sites per BAM: {per_bam}), "
          f"mismatches {int(n_bad)}, exact-fallback sites {int(sum(a['fb'] for a in agg.values()))}")
    for key, b in all_bad[:30]:
        print("  MISMATCH", key, b)
    if args.report:
        with open(args.report, "w") as f:
            f.write("bam\tcontig\tmq\tbq\tmax_depth\tsites\tmismatches\tfallback\torig_s\tnew_warm_s\n")
            for key in sorted(agg):
                a = agg[key]
                f.write("\t".join(map(str, (*key, int(a["n"]), int(a["bad"]), int(a["fb"]),
                                            round(a["t_orig"], 4), round(a["t_new2"], 4)))) + "\n")
    return int(n_bad)


def _speed_row(name, s):
    def ms(t, n):
        return t / n * 1e3 if n else float("nan")
    oc, nc = ms(s["oc_t"], s["oc_n"]), ms(s["nc_t"], s["nc_n"])
    ow, nw = ms(s["ow_t"], s["ow_n"]), ms(s["nw_t"], s["nw_n"])
    print(f"  {name[:16]:16} {oc:>10.2f} {nc:>10.2f} {ow:>10.2f} {nw:>10.3f} {oc / nc:>7.1f} "
          f"{ow / nw:>7.1f}")


def dnm_job(job):
    bam, contig, pos, mq, bq, orig_first = job
    if orig_first:
        exp, t_orig = _timed(orig_counts, bam, contig, pos, mq, bq)
    got, t_new = _timed(allelelib.count_alleles, bam, contig, pos, mq, bq)
    if not orig_first:
        exp, t_orig = _timed(orig_counts, bam, contig, pos, mq, bq)
    return bam, contig, pos, mq, bq, got, exp, t_new, t_orig, orig_first


def run_dnm(args, rows, info):
    """Recount every site of the original runs' COUNTS column in each OTHER sample's BAM."""
    for fn, sample, n_dnm, n_rows, mq, bq in info:
        print(f"[dnm] {os.path.relpath(fn, args.runs_dir)}: sample {sample}, DNM rows {n_dnm}, "
              f"rows with COUNTS {n_rows} (min_mapq {mq}, min_baseq {bq})")
    need = defaultdict(set)                   # (bam, contig, mq, bq) -> 1-based sites
    for r in rows:
        for o in r["others"]:
            need[(r["bams"][o], r["contig"], r["mq"], r["bq"])].add(r["end"])
    jobs = [(bam, ctg, np.array(sorted(s), dtype=np.int64), mq, bq, i % 2 == 0)
            for i, ((bam, ctg, mq, bq), s) in enumerate(sorted(need.items()))]
    t0 = time.time()
    with mp.Pool(args.nproc) as pool:
        res = list(pool.imap_unordered(dnm_job, jobs))
    got, exp = {}, {}
    t = defaultdict(float)
    for bam, ctg, pos, mq, bq, g, e, t_new, t_orig, orig_first in res:
        for i, p in enumerate(pos.tolist()):
            got[(bam, ctg, p, mq, bq)] = ":".join(map(str, g[i].tolist()))
            exp[(bam, ctg, p, mq, bq)] = ":".join(map(str, e[i].tolist()))
        t["new_cold" if not orig_first else "new_warm"] += t_new
        t["orig_cold" if orig_first else "orig_warm"] += t_orig
        t["n_new_cold" if not orig_first else "n_new_warm"] += pos.size
        t["n_orig_cold" if orig_first else "n_orig_warm"] += pos.size
    n_pairs = len(got)
    bad_new = bad_orig = n_fields = 0
    for r in rows:
        for o, field in zip(r["others"], r["fields"]):
            key = (r["bams"][o], r["contig"], r["end"], r["mq"], r["bq"])
            n_fields += 1
            if got[key] != field:
                bad_new += 1
                if bad_new <= 20:
                    print(f"  DNM MISMATCH {os.path.basename(r['file'])} {r['contig']}:{r['end']} "
                          f"{r['ann']} BAM {o}: file {field} new {got[key]} orig-fn {exp[key]}")
            bad_orig += exp[key] != field
    nz = sum(v != "0:0:0:0" for v in got.values())
    print(f"[dnm] {len(rows)} rows, {n_fields} COUNTS fields, {n_pairs} unique (BAM, site) pileups "
          f"({nz} non-zero) in {time.time() - t0:.0f}s wall; new != file: {bad_new}; "
          f"original function != file: {bad_orig}")

    def ms(a, b):
        return t[a] / t[b] * 1e3 if t[b] else float("nan")
    print(f"[dnm] ms/site: orig cold {ms('orig_cold', 'n_orig_cold'):.1f}, new cold "
          f"{ms('new_cold', 'n_new_cold'):.1f}, orig warm {ms('orig_warm', 'n_orig_warm'):.1f}, "
          f"new warm {ms('new_warm', 'n_new_warm'):.1f}")
    return bad_new + bad_orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="real subset BAMs: original-run COUNTS check + >=5000 sites per BAM")
    ap.add_argument("--skip-sites", action="store_true", help="with --real: only the COUNTS check")
    ap.add_argument("--skip-dnm", action="store_true", help="with --real: skip the COUNTS check")
    ap.add_argument("--bams", nargs="*")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--runs-dir", default=os.path.join(VALIDATION, "runs"))
    ap.add_argument("--report")
    args = ap.parse_args()
    check_embedded_original()
    fails = []
    if args.real:
        rows, info = original_count_rows(args.runs_dir)
        print(f"[real] original runs: {len(info)} shared.muts files, {len(rows)} rows with COUNTS")
        if not args.skip_dnm:
            if not rows:
                fails.append(("no original-run COUNTS rows found", 1))
            elif run_dnm(args, rows, info):
                fails.append(("dnm", 1))
        if not args.skip_sites and run_real(args, dnm_sites(rows)):
            fails.append(("real", 1))
    else:
        test_split_positions(fails)
        with tempfile.TemporaryDirectory() as tmp:
            test_original_crash_replicated(tmp, fails)
            test_synthetic(tmp, fails)
        test_pool_tasks(fails)
        test_tests_data(fails)
    if fails:
        print("FAILED:", fails)
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
