#!/usr/bin/env python
"""Plain-script tests for bistro.covlib (no pytest).

    PYTHONPATH=src python tests/test_covlib.py      # from a cwd without a bistro/ dir

Checks the in-process replacement of the mosdepth / bedtools mask path:
  * pure-Python / brute-force references for the interval maths and the LCR
    parser (copies of the ORIGINAL high_coverage_bed / build_mask logic);
  * if `mosdepth` is on PATH: a crafted BAM (tests/data reads + supplementary,
    secondary, duplicate, QC-fail, unmapped-placed, low-MAPQ, CIGAR-less,
    D/N, past-the-end reads, contigs with zero / only-filtered reads, means at
    exact %.2f ties) and tests/data/{E01,F01}.bam against `mosdepth -x -Q 20`
    (multi-contig and `-c` mode): contig listing, means, merged >= cap intervals;
  * if `bedtools` is on PATH: mask vs the original build_mask (zcat | filter |
    sort | bedtools merge) + `bedtools subtract -A` on 1-bp records.
Exit status is non-zero on failure.
"""

import gzip
import os
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pysam

from bistro import covlib, reportlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RNG = np.random.default_rng(12345)
random.seed(12345)


# ---------------------------------------------------------------------------
# references: copies of the ORIGINAL logic
# ---------------------------------------------------------------------------

def orig_merge_rows(rows, caps, contigs):
    """original covlib.high_coverage_bed on in-memory per-base rows."""
    want = set(contigs)
    merged, cur = [], None
    for chrom, start, end, depth in rows:
        cap = caps.get(chrom)
        if cap is None or chrom not in want or float(depth) < cap:
            continue
        s, e = int(start), int(end)
        if cur is not None and cur[0] == chrom and s <= cur[2]:
            cur[2] = max(cur[2], e)
        else:
            if cur is not None:
                merged.append(cur)
            cur = [chrom, s, e]
    if cur is not None:
        merged.append(cur)
    return merged


def orig_lcr_records(text_lines, contigs):
    """original covlib.build_mask's line filter (records it piped to sort)."""
    keep = set(contigs)
    out = []
    for line in text_lines:
        if line.startswith(("#", "track", "browser")):
            continue
        f = line.split("\t")
        if len(f) >= 3 and f[0] in keep:
            out.append((f[0], int(f[1]), int(f[2].rstrip())))
    return out


def text_lines_like_zcat(path):
    """What build_mask iterated: `zcat -f` output through a text-mode pipe."""
    p = subprocess.run(["zcat", "-f", path], stdout=subprocess.PIPE, check=True)
    import io
    return list(io.TextIOWrapper(io.BytesIO(p.stdout)))   # universal newlines, utf-8


def depth_array(starts, ends, length):
    d = np.zeros(length + 1, dtype=np.int64)
    s = np.minimum(starts, length)
    e = np.minimum(ends, length)
    np.add.at(d, s, 1)
    np.add.at(d, e, -1)
    return np.cumsum(d)[:length]


def bedtools_mask_ref(pairs, length):
    """Pure-Python model of `sort -k1,1 -k2,2n | bedtools merge` + `subtract -A`
    (one contig): ties sorted by the END field as a string; zero-length [s,s)
    read as [s-1,s+1); a merged block starts at its first record's start."""
    recs = [(a, b) for s, e in pairs for a, b in zip(np.asarray(s).tolist(), np.asarray(e).tolist()) if a <= b]
    recs.sort(key=lambda r: (r[0], str(r[1])))
    blocks = []
    for a, b in recs:
        if a == b:
            a, b = a - 1, b + 1
        if blocks and a <= blocks[-1][1]:
            blocks[-1][1] = max(blocks[-1][1], b)
        else:
            blocks.append([a, b])
    m = np.zeros(length, dtype=bool)
    for a, b in blocks:
        m[max(a, 0):max(min(b, length), 0)] = True
    return m


def per_base_rows(chrom, depth):
    """mosdepth per-base rows (runs of equal depth) from a depth array."""
    if depth.size == 0:
        return []
    brk = np.flatnonzero(np.diff(depth)) + 1
    st = np.r_[0, brk]
    en = np.r_[brk, depth.size]
    return [(chrom, str(a), str(b), str(depth[a])) for a, b in zip(st, en)]


failures = []


_n0 = [0]


def ok(msg):
    """Per-test summary line; says FAILED if the current test recorded failures."""
    print(("[ok] " if len(failures) == _n0[0] else "[FAILED] ") + msg)


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("FAIL:", msg)


# ---------------------------------------------------------------------------
# 1. interval maths vs brute force
# ---------------------------------------------------------------------------

def test_high_coverage_random():
    n_cases = 0
    for trial in range(400):
        length = int(RNG.integers(1, 3000))
        n = int(RNG.integers(0, 80))
        st = RNG.integers(0, length + 5, n)            # a few start at/after the end
        ln = RNG.integers(1, 400, n)
        en = st + ln                                  # some run past the end
        if trial % 7 == 0 and n:                      # pile-ups at identical coordinates
            st[: n // 2] = st[0]
            en[: n // 2] = en[0]
        depth = depth_array(st, en, length)
        rows = per_base_rows("c", depth)
        mx = int(depth.max()) if depth.size else 0
        caps = [0, -1, 0.5, 1, 1.5, 2, 2.01, 3.99, mx, mx + 0.5, mx + 1]
        caps += [round(float(x), 2) for x in RNG.uniform(0, mx + 2, 4)]
        for cap in caps:
            want = orig_merge_rows(rows, {"c": cap}, ["c"])
            hs, he = covlib.high_coverage_intervals(st, en, cap, length)
            got = [["c", int(a), int(b)] for a, b in zip(hs, he)]
            check(got == want, f"high_coverage_intervals trial={trial} cap={cap}: {got[:3]} != {want[:3]}")
            check(hs.dtype == np.int64 and he.dtype == np.int64, "high_coverage_intervals dtype")
            if cap > 0 and (n == 0 or (st.max() < length and en.max() <= length)):
                # reads inside the contig: the length argument is optional
                hs2, he2 = covlib.high_coverage_intervals(st, en, cap)
                check(np.array_equal(hs2, hs) and np.array_equal(he2, he),
                      f"high_coverage_intervals without length trial={trial} cap={cap}")
            n_cases += 1
        check(covlib.covered_bases(st, en, length) == int(depth.sum()), f"covered_bases trial={trial}")
    n_nolen = 0
    for trial in range(200):                          # reads inside the contig, no length given
        length = int(RNG.integers(50, 3000))
        n = int(RNG.integers(1, 80))
        st = RNG.integers(0, length - 1, n)
        en = np.minimum(st + RNG.integers(1, 400, n), length)
        rows = per_base_rows("c", depth_array(st, en, length))
        for cap in (0.5, 1, 2, 2.5, 3, 5):
            want = orig_merge_rows(rows, {"c": cap}, ["c"])
            hs, he = covlib.high_coverage_intervals(st, en, cap)
            check([["c", int(a), int(b)] for a, b in zip(hs, he)] == want,
                  f"high_coverage_intervals(no length) trial={trial} cap={cap}")
            n_nolen += 1
    try:
        covlib.high_coverage_intervals([1], [5], 0)
        check(False, "cap <= 0 without length must raise")
    except ValueError:
        pass
    n_cases += n_nolen
    ok(f"high_coverage_intervals vs brute-force per-base merge: {n_cases} cases")


def test_mosdepth_mean_format():
    for bases, length, want in [(1, 8, 0.12), (3, 8, 0.38), (5, 8, 0.62), (7, 8, 0.88),
                                (201, 200, 1.0), (2675, 1000, 2.67), (15, 1000, 0.01),
                                (0, 100, 0.0), (38468, 7367, 5.22)]:
        got = covlib.mosdepth_mean(bases, length)
        check(got == want and str(got) == str(want), f"mosdepth_mean({bases},{length}) = {got} != {want}")
    ok("mosdepth_mean formatting")


def test_merge_and_mask():
    n = 0
    for trial in range(300):
        length = int(RNG.integers(1, 400))
        pairs = []
        for _ in range(int(RNG.integers(0, 4))):
            k = int(RNG.integers(0, 30))
            hi = length + 10 if trial % 2 else max(3, length // 10)   # many shared starts
            s = RNG.integers(0, hi, k)
            e = s + RNG.choice([0, 0, 1, 2, 5, 40, 95, 120, 1000], k)   # zero-length; '99' vs '120'-style ties
            pairs.append((s, e))
        brute = bedtools_mask_ref(pairs, length)
        m = covlib.mask_bool(length, *pairs)
        check(m.dtype == bool and m.shape == (length,) and np.array_equal(m, brute), f"mask_bool trial={trial}")
        ms, me = covlib.merge_intervals(*pairs, length=length)
        check(bool(np.all(ms[1:] > me[:-1])) and bool(np.all(ms < me)), f"merge_intervals disjoint trial={trial}")
        pos = RNG.integers(0, length, 50)
        check(np.array_equal(covlib.in_intervals(pos, ms, me), brute[pos]), f"in_intervals trial={trial}")
        n += 1
    ms, me = covlib.merge_intervals()
    check(ms.size == 0 and me.size == 0, "merge_intervals() empty")
    check(covlib.mask_bool(5).tolist() == [False] * 5, "mask_bool empty")
    ok(f"merge_intervals / mask_bool / in_intervals vs brute force: {n} cases")


def test_write_highcov_bed(tmp):
    check(covlib.write_highcov_bed(tmp, "S0", []) is None, "write_highcov_bed none")
    check(covlib.write_highcov_bed(tmp, "S0", [("a", np.zeros(0, np.int64), np.zeros(0, np.int64))]) is None,
          "write_highcov_bed empty arrays")
    check(not os.path.exists(os.path.join(tmp, "S0.highcov.bed")), "no file when empty")
    per = [("chrA", np.array([5, 100]), np.array([10, 2000])), ("chrB", np.array([], np.int64), np.array([], np.int64)),
           ("chrC.1", np.array([0], np.int64), np.array([999], np.int64))]
    p = covlib.write_highcov_bed(tmp, "S1", per)
    want = "chrA\t5\t10\nchrA\t100\t2000\nchrC.1\t0\t999\n"
    check(p == os.path.join(tmp, "S1.highcov.bed") and open(p).read() == want, "write_highcov_bed text")
    ok("write_highcov_bed")


# ---------------------------------------------------------------------------
# 2. LCR parsing vs the original filter (+ bedtools mask)
# ---------------------------------------------------------------------------

def make_lcr_text():
    lines = ["#comment\tline\there", "track name=foo", "browser position chr1:1-10",
             "trackX\t1\t5", "browserY\t2\t6",          # also skipped (prefix match)
             "chr1\t10\t20", "chr10\t5\t9", "chr1\t15\t30\textra\tcols", "chr2\t0\t3 ",
             "chr1\t40\t40", "chr1\t50", "", "chr3\t1\t2", "chr1 \t1\t2", "chr1\t60\t70\t",
             "chr2\t100\t105  \t", "chrUn_x\t1\t5", "c\t7\t8",
             "tra\t1\t2", "browse\t3\t4", "track\t5\t6", "t", "#", "b\t1\t9", "tracks\t2\t3"]
    for _ in range(3000):
        c = random.choice(["chr1", "chr2", "chr10", "chrX", "c", "chr1_alt"])
        s = random.randint(0, 5000)
        lines.append(f"{c}\t{s}\t{s + random.randint(0, 50)}")
    return lines


def test_parse_lcr(tmp):
    contigs = ["chr1", "chr2", "chr10", "c", "notpresent", "tra", "browse", "track", "b", "tracks"]
    lines = make_lcr_text()
    variants = {
        "plain.bed": ("\n".join(lines) + "\n").encode(),
        "noeol.bed": "\n".join(lines).encode(),
        "crlf.bed": ("\r\n".join(lines) + "\r\n").encode(),
        "cr.bed": ("\r".join(lines) + "\r").encode(),
        "mixed.bed": "".join(l + random.choice(["\n", "\r\n", "\r"]) for l in lines).encode(),
    }
    paths = []
    for name, raw in variants.items():
        p = os.path.join(tmp, name)
        with open(p, "wb") as fh:
            fh.write(raw)
        paths.append(p)
        pg = p + ".gz"
        with gzip.open(pg, "wb") as fh:
            fh.write(raw)
        paths.append(pg)
        pb = p + ".bgz"
        with pysam.BGZFile(pb, "wb") as fh:
            fh.write(raw)
        paths.append(pb)
    n = 0
    for p in paths:
        want = orig_lcr_records(text_lines_like_zcat(p), contigs)
        for chunk in (64 << 20, 7, 64, 1000):     # tiny chunks: lines (and \r|\n) split across reads
            got_d = covlib.parse_lcr_bed(p, contigs, chunk_bytes=chunk)
            check(set(got_d) == set(contigs), f"parse_lcr_bed keys {p}")
            for c in contigs:
                w = [(s, e) for cc, s, e in want if cc == c]
                g = list(zip(got_d[c][0].tolist(), got_d[c][1].tolist()))
                check(g == w, f"parse_lcr_bed {os.path.basename(p)} chunk={chunk} contig={c}: {g[:4]} != {w[:4]}")
                check(got_d[c][0].dtype == np.int64 and got_d[c][1].dtype == np.int64, "dtype")
            n += 1
    # missing / empty / None -> empty arrays (build_mask ignored such files)
    empty = os.path.join(tmp, "empty.bed")
    open(empty, "w").close()
    for p in (None, "", os.path.join(tmp, "missing.bed"), empty):
        d = covlib.parse_lcr_bed(p, ["chr1"])
        check(list(d) == ["chr1"] and d["chr1"][0].size == 0, f"parse_lcr_bed({p!r}) not empty")
    # malformed records on analysed contigs -> ValueError (bedtools failed in the original)
    for bad in ("chr1\t30\t20\n", "chr1\t-5\t3\n", "chr1\tx\t3\n", "chr1\t5\t\n"):
        p = os.path.join(tmp, "bad.bed")
        with open(p, "w") as fh:
            fh.write("chr2\t1\t5\n" + bad)
        try:
            covlib.parse_lcr_bed(p, ["chr1", "chr2"])
            check(False, f"parse_lcr_bed accepted {bad!r}")
        except ValueError:
            pass
        d = covlib.parse_lcr_bed(p, ["chr2"])          # not analysed -> ignored, as before
        check(d["chr2"][0].tolist() == [1], "bad record on non-analysed contig must be ignored")
    ok(f"parse_lcr_bed vs original filter: {n} file/chunk combinations")


def orig_build_mask_and_subtract(tmp, lcr_path, highcov_bed, contigs, lengths):
    """Original build_mask (copy) + `bedtools subtract -A` of 1-bp records of
    every position; returns {contig: bool removed[length]}."""
    beds = [b for b in (lcr_path, highcov_bed) if b and os.path.isfile(b) and os.path.getsize(b) > 0]
    removed = {c: np.zeros(lengths[c], dtype=bool) for c in contigs}
    if not beds:
        return removed
    keep = set(contigs)
    mask = os.path.join(tmp, "orig.mask.bed")
    cat = subprocess.Popen(["zcat", "-f", *beds], stdout=subprocess.PIPE, text=True)
    srt = subprocess.Popen(["sort", "-k1,1", "-k2,2n"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for line in cat.stdout:
        if line.startswith(("#", "track", "browser")):
            continue
        f = line.split("\t")
        if len(f) >= 3 and f[0] in keep:
            srt.stdin.write(f"{f[0]}\t{f[1]}\t{f[2].rstrip()}\n")
    cat.stdout.close()
    cat.wait()
    srt.stdin.close()
    with open(mask, "w") as fh:
        subprocess.run(["bedtools", "merge", "-i", "-"], stdin=srt.stdout, stdout=fh, check=True)
    srt.stdout.close()
    srt.wait()
    if os.path.getsize(mask) == 0:
        return removed
    recs = os.path.join(tmp, "recs.bed")
    with open(recs, "w") as fh:
        fh.write("#header\n")
        for c in contigs:
            fh.write("".join(f"{c}\t{p}\t{p + 1}\tx\n" for p in range(lengths[c])))
    out = subprocess.run(["bedtools", "subtract", "-header", "-A", "-a", recs, "-b", mask],
                         stdout=subprocess.PIPE, text=True, check=True).stdout
    for c in contigs:
        removed[c][:] = True
    for line in out.splitlines():
        if line.startswith("#"):
            continue
        f = line.split("\t")
        removed[f[0]][int(f[1])] = False
    return removed


def test_mask_vs_bedtools(tmp):
    if shutil.which("bedtools") is None:
        print("[skip] bedtools not on PATH")
        return
    contigs = ["chr1", "chr2", "chr3"]
    lengths = {"chr1": 3000, "chr2": 500, "chr3": 50}
    for rnd in range(12):
        lines = ["#hdr", "track x"]
        for _ in range(400):
            c = random.choice(["chr1", "chr2", "chr3", "chrZ"])
            hi = lengths.get(c, 100) + 20 if rnd % 2 else lengths.get(c, 100) // 8   # shared starts
            s = random.randint(1, hi)                 # (a zero-length record at 0 broke the original)
            lines.append(f"{c}\t{s}\t{s + random.choice([0, 0, 1, 2, 5, 30, 95, 120])}")
        lcr = os.path.join(tmp, "lcr_mask_test.bed.gz")
        with gzip.open(lcr, "wt") as fh:
            fh.write("\n".join(lines) + "\n")
        hc = {"chr1": (np.array([100, 2990]), np.array([250, 3000])), "chr2": (np.array([0]), np.array([7])),
              "chr3": (np.zeros(0, np.int64), np.zeros(0, np.int64))}
        hc_path = covlib.write_highcov_bed(tmp, "T", [(c, *hc[c]) for c in contigs])
        want = orig_build_mask_and_subtract(tmp, lcr, hc_path, contigs, lengths)
        lcr_d = covlib.parse_lcr_bed(lcr, contigs)
        for c in contigs:
            got = covlib.mask_bool(lengths[c], lcr_d[c], hc[c])
            check(np.array_equal(got, want[c]), f"mask vs bedtools rnd={rnd} {c}: diff at {np.flatnonzero(got != want[c])[:10]}")
            ms, me = covlib.merge_intervals(lcr_d[c], hc[c])
            pos = np.arange(lengths[c])
            check(np.array_equal(covlib.in_intervals(pos, ms, me), want[c]), f"in_intervals vs bedtools rnd={rnd} {c}")
            check(np.array_equal(bedtools_mask_ref([lcr_d[c], hc[c]], lengths[c]), want[c]), f"python model rnd={rnd} {c}")
    # high-cov only
    want = orig_build_mask_and_subtract(tmp, None, hc_path, contigs, lengths)
    for c in contigs:
        check(np.array_equal(covlib.mask_bool(lengths[c], hc[c]), want[c]), f"highcov-only mask {c}")
    ok("mask vs original build_mask + bedtools subtract -A (12 random LCR sets)")


# ---------------------------------------------------------------------------
# 3. mosdepth comparison
# ---------------------------------------------------------------------------

def build_crafted_bam(path):
    src = pysam.AlignmentFile(os.path.join(DATA, "E01.bam"))
    reads = list(src.fetch("test_dnm_locus")) + list(src.fetch("test_grm_locus"))
    sq = [("test_dnm_locus", 7367), ("empty_contig", 5000), ("test_grm_locus", 10470),
          ("filtered_only", 8000), ("unmapped_only", 8000), ("lowmapq_only", 8000),
          ("empty_tail", 3000), ("edge_contig", 1000),
          ("tie_0125", 8), ("tie_0375", 8), ("tie_0625", 8), ("tie_0875", 8),
          ("c_1005", 200), ("c_2675", 1000), ("c_0015", 1000), ("empty_last", 100)]
    tid = {n: i for i, (n, _) in enumerate(sq)}
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": n, "LN": l} for n, l in sq]}
    base = reads[0]
    qlen = len(base.query_sequence)
    recs = []

    def clone(r, name, contig=None, pos=None, flag=None, mapq=None, cigar="keep"):
        a = pysam.AlignedSegment()
        a.query_name = name
        a.query_sequence = r.query_sequence
        a.query_qualities = r.query_qualities
        a.flag = r.flag if flag is None else flag
        a.reference_id = tid[contig or r.reference_name]
        a.reference_start = r.reference_start if pos is None else pos
        a.mapping_quality = r.mapping_quality if mapq is None else mapq
        if cigar == "keep":
            a.cigartuples = r.cigartuples
        elif cigar is not None:
            a.cigarstring = cigar
        a.next_reference_id = -1
        a.next_reference_start = -1
        a.set_tags(r.get_tags())
        return a

    for r in reads:
        recs.append(clone(r, r.query_name))
    k = [0]

    def add(contig, pos, flag=0, mapq=60, cigar="keep", tlen=0):
        k[0] += 1
        recs.append(clone(base, f"crafted_{k[0]}", contig=contig, pos=pos, flag=flag, mapq=mapq, cigar=cigar))
        recs[-1].template_length = tlen

    def m(n):
        return f"{n}M{qlen - n}S"

    for fl in (2048, 2048 | 16, 256, 1024, 512, 4, 1 | 2 | 64, 1 | 8 | 128, 16, 4 | 2048, 256 | 2048, 1 | 2 | 1024):
        add("test_dnm_locus", 1000 + (10 * fl) % 3000, flag=fl)
    for tl in (-100000, -1, 1, 5, 100000, 2 ** 31 - 1):          # insert size: no -l/-u filter by default
        add("test_dnm_locus", 4000 + (tl % 1000), flag=1 | 2 | 64, tlen=tl)
    for mq in (0, 19, 20, 21, 255):
        add("test_dnm_locus", 2000 + mq, mapq=mq)
    add("test_dnm_locus", 3000, cigar=None)                        # mapped, no CIGAR -> [pos, pos+1)
    add("test_dnm_locus", 3100, cigar=f"{qlen}S")                  # no ref-consuming op -> [pos, pos+1)
    add("test_dnm_locus", 3200, cigar=f"100M500N100M200D{qlen - 200}S")
    add("test_dnm_locus", 0, cigar=m(50))
    add("test_dnm_locus", 7367 - 50, cigar=m(50))
    for fl in (256, 1024, 512):
        add("filtered_only", 100 + fl, flag=fl)
    add("filtered_only", 300, mapq=5)
    for p in (100, 500):
        add("unmapped_only", p, flag=4, mapq=0, cigar=None)
    for p in (100, 500):
        add("lowmapq_only", p, mapq=10)
    add("empty_tail", 0, cigar=m(100))
    add("empty_tail", 50, cigar=m(100))
    for i in range(1, 6):
        add("edge_contig", 0, cigar=m(10 * i))
        add("edge_contig", 1000 - 10 * i, cigar=m(10 * i))
    add("edge_contig", 950, cigar=m(100))                          # runs past the contig end
    add("tie_0125", 3, cigar=m(1))
    add("tie_0375", 2, cigar=m(3))
    add("tie_0625", 0, cigar=m(5))
    add("tie_0875", 1, cigar=m(7))
    add("c_1005", 0, cigar=m(200))
    add("c_1005", 10, cigar=m(1))
    add("c_2675", 0, cigar=m(1000))
    add("c_2675", 0, cigar=m(1000))
    add("c_2675", 100, cigar=m(675))
    add("c_0015", 500, cigar=m(15))
    recs.sort(key=lambda a: (a.reference_id, a.reference_start))
    tmp_unsorted = path + ".unsorted.bam"
    with pysam.AlignmentFile(tmp_unsorted, "wb", header=hdr) as out:
        for a in recs:
            out.write(a)
    pysam.sort("-o", path, tmp_unsorted)
    pysam.index(path)
    os.remove(tmp_unsorted)
    return path


def run_mosdepth(bam, prefix, contig=None):
    cmd = ["mosdepth", "-x", "-Q", str(covlib.COV_MAPQ), "-t", "1"]
    if contig is not None:
        cmd += ["-c", contig]
    subprocess.run(cmd + [prefix, bam], check=True, stdout=subprocess.DEVNULL)
    summary = {}
    with open(prefix + ".mosdepth.summary.txt") as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[0] != "total":
                summary[f[0]] = (int(f[1]), int(f[2]), f[3])
    with gzip.open(prefix + ".per-base.bed.gz", "rt") as fh:
        rows = [tuple(l.rstrip("\n").split("\t")) for l in fh]
    return summary, rows


def compare_bam_with_mosdepth(bam, tmp, tag):
    with pysam.AlignmentFile(bam) as fh:
        contigs = list(fh.references)
        lengths = dict(zip(fh.references, fh.lengths))
        accs = {}
        for c in contigs:
            acc = covlib.RawCoverage()
            for read in fh.fetch(c, 0, lengths[c]):       # what the caller's worker loop sees
                acc.add(read)
            accs[c] = acc
    modes = [(None, contigs)] + [(c, [c]) for c in contigs]
    n_int = 0
    for single, wanted in modes:
        prefix = os.path.join(tmp, f"{tag}_{single or 'all'}")
        summary, rows = run_mosdepth(bam, prefix, single)
        # parse_contig_means equivalent
        orig_means = {c: float(summary[c][2]) for c in wanted if c in summary}
        for c in wanted:
            mean, cap, hs, he = covlib.contig_raw_coverage(accs[c], lengths[c], 1)
            check((mean is None) == (c not in summary), f"{tag} [{single}] listing of {c}: mean={mean}")
            if c not in summary:
                check(hs.size == 0 and not any(r[0] == c for r in rows), f"{tag} unlisted {c} has rows")
                continue
            s, e = accs[c].arrays()
            check(covlib.covered_bases(s, e, lengths[c]) == summary[c][1], f"{tag} bases {c}")
            check(mean == orig_means[c] and f"{mean:.2f}" == summary[c][2], f"{tag} mean {c}: {mean} vs {summary[c][2]}")
            crow = [r for r in rows if r[0] == c]
            mx = max(int(r[3]) for r in crow)
            caps = set()
            for md in (-1, 0, 1, 2, 3, 10):
                caps.add(reportlib.get_maximum_coverage_per_contig({c: orig_means[c]}, min_depth=md)[c])
            caps.update([0.5, 1, 1.5, 2, 2.5, 3, mx - 1, mx - 0.01, mx, mx + 0.01, mx + 1])
            for cap in sorted(caps):
                want = orig_merge_rows(crow, {c: cap}, wanted)
                hs, he = covlib.high_coverage_intervals(s, e, cap, lengths[c])
                got = [[c, int(a), int(b)] for a, b in zip(hs, he)]
                check(got == want, f"{tag} [{single}] {c} cap={cap}: {got[:3]} != {want[:3]}")
                n_int += 1
    ok(f"{tag}: mosdepth listing/means/bases for {len(contigs)} contigs x {len(modes)} modes, "
          f"{n_int} cap comparisons")


def test_against_mosdepth(tmp):
    if shutil.which("mosdepth") is None:
        print("[skip] mosdepth not on PATH")
        return
    crafted = build_crafted_bam(os.path.join(tmp, "crafted.bam"))
    compare_bam_with_mosdepth(crafted, tmp, "crafted")
    for s in ("E01", "F01"):
        compare_bam_with_mosdepth(os.path.join(DATA, f"{s}.bam"), tmp, s)
    # RawCoverage bookkeeping: n_seen counts every record, len() only the counted ones
    with pysam.AlignmentFile(crafted) as fh:
        acc = covlib.RawCoverage()
        n = n_cov = 0
        for r in fh.fetch("test_dnm_locus"):
            acc.add(r)
            n += 1
            n_cov += (not r.flag & 1796) and r.mapping_quality >= 20
        check(acc.n_seen == n and len(acc) == n_cov and n_cov < n, "RawCoverage counts")
    order = covlib.bam_header_order(crafted, ["c_2675", "empty_contig", "test_dnm_locus"])
    check(order == ["test_dnm_locus", "empty_contig", "c_2675"], f"bam_header_order {order}")


def main():
    tmp = tempfile.mkdtemp(prefix="test_covlib_")
    tests = [test_high_coverage_random, test_mosdepth_mean_format, test_merge_and_mask,
             lambda: test_write_highcov_bed(tmp), lambda: test_parse_lcr(tmp),
             lambda: test_mask_vs_bedtools(tmp), lambda: test_against_mosdepth(tmp)]
    try:
        for fn in tests:
            _n0[0] = len(failures)
            fn()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if failures:
        print(f"{len(failures)} FAILURE(S)")
        sys.exit(1)
    print("ALL COVLIB TESTS PASSED")


if __name__ == "__main__":
    main()
