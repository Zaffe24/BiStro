#!/usr/bin/env python
"""Plain-script tests for bistro.fastio (no pytest).

    python tests/test_fastio.py [--quick] [--workdir DIR] [--keep]          # synthetic suite
    python tests/test_fastio.py --workdir DIR --real CTX.bed.gz [--real ...]  # + real files
    python tests/test_fastio.py --workdir DIR --bench 50000000 [--bench-level L]

Run through SLURM (e.g. `srun -p short -c 2 --mem 16G -t 60 ...`); --workdir must be on a
filesystem with room for a rebuilt copy of each --real file.

Checks
  * format_context_lines / parse_bed_chunk / parse_uint_field / field_bytes_fixed /
    select_lines against pure-Python references on random data (digit-width boundaries
    incl. 0, 9, 10, 99, 999999999, 2**31-2, values up to 2**40, multiple formatting blocks,
    error cases); parse_uint_field's SWAR path for every field length 1-18 (leading zeros,
    fields at the buffer start) and a non-digit byte at every position;
  * the BGZF compression level only changes compressed bytes;
  * pysam.BGZFile.tell() after writes returns virtual offsets valid for seek();
  * IndexedBgzfWriter + assemble_bgzf + build_tbi on synthetic BED datasets (many contigs, empty
    contigs/fragments, contigs split over consecutive fragments incl. mid-window, sparse windows
    far apart, dense windows, a window spanning many BGZF blocks, records only at position 0,
    single-record contigs, names with '.' and ':', positions up to 2**29-1):
      - decompressed text == header + fragments (gzip, bgzip -d, pysam);
      - the `tabix -p bed` index of the same file obeys every layout rule build_tbi relies on
        (verify_tabix_layout) and == pysam.tabix_index(preset="bed") (the original's indexer);
      - the custom .tbi decodes to the same structure as `tabix -p bed`'s once virtual offsets
        are mapped to uncompressed positions;
      - pysam.TabixFile.fetch and the tabix CLI return identical records with both indexes for
        >= 1000 random regions + window-edge / gap / whole-contig / missing-contig queries;
      - iter_tabix_chunks == pysam fetch, early generator close leaves no child process;
  * outputs are replaced by rename (hard links to an old output keep their content; failures
    leave no partial or temporary file); positions >= 2**29 are refused (like tabix); error
    paths; FragmentIndex pickling.
  * --real: see check_real() (original-pipeline file + index vs a rebuild through the writer
    path; >= 1500 pysam + 1000 tabix CLI region queries + whole contigs).
Exits non-zero on the first failure.
"""

import argparse
import gzip
import os
import pickle
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("BISTRO_SRC", os.path.join(HERE, "..", "src")))

import pysam  # noqa: E402
from bistro import fastio as F  # noqa: E402

W = 1 << F.WINDOW_SHIFT
SENTINEL = "zz_sentinel"
SENTINEL_POS = 12345


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


# --------------------------------------------------------------------------------------------
# 1. text formatting / parsing
# --------------------------------------------------------------------------------------------

def py_context_lines(contig, pos, tri, depth):
    out, offs, o = [], [], 0
    for p, t, d in zip(pos.tolist(), tri, depth.tolist()):
        s = f"{contig}\t{p}\t{p+1}\t{bytes(t).decode('latin-1')}\t{d}\n".encode("latin-1")
        offs.append(o)
        o += len(s)
        out.append(s)
    return b"".join(out), np.array(offs, np.int64)


BOUNDARY = sorted({0, 1, 2**31 - 2, 2**31 - 1, 2**32 - 2, 2**32 - 1, 2**32, 2**40, 999999999}
                  | {10**k + d for k in range(1, 13) for d in (-2, -1, 0, 1)})


def test_format(rng, n):
    log("format_context_lines vs f-string")
    contigs = ["chr1", "NC_000067.7", "a", "HLA:01:02", "x" * 100, "chrUn_NW_0001.1"]
    for trial in range(8):
        contig = contigs[trial % len(contigs)]
        m = n if trial < 2 else int(rng.integers(1, 5000))
        hi = [2**31 - 1, 2**29, 10**4, 2**33][trial % 4]
        pos = rng.integers(0, hi, m).astype(np.int64)
        if trial % 2 == 0:
            pos = np.sort(pos)
        b = np.array(BOUNDARY, np.int64)
        pos[: min(m, b.size)] = b[: min(m, b.size)]
        if trial == 1:
            pos = np.concatenate([pos, b])           # boundaries at the end too
        k = pos.size
        dtype = [np.uint16, np.int64, np.uint32, np.int32][trial % 4]
        dmax = min(np.iinfo(dtype).max, 2**40)
        depth = rng.integers(0, dmax, k, endpoint=True).astype(dtype)
        db = np.array([0, 1, 9, 10, 99, 100, 999, 1000, 9999, 10000, 65535], dtype)
        depth[: min(k, db.size)] = db[: min(k, db.size)]
        depth[-min(k, db.size):] = db[: min(k, db.size)]
        alphabet = np.frombuffer(b"ACGTNacgtnRY", np.uint8)
        tri = alphabet[rng.integers(0, alphabet.size, (k, 3))]
        data, offs = F.format_context_lines(contig, pos, tri, depth)
        ref, roffs = py_context_lines(contig, pos, tri, depth)
        assert data == ref, f"format mismatch (trial {trial})"
        assert offs.dtype == np.int64 and np.array_equal(offs, roffs), "offset mismatch"
    # blocks whose widths are constant in some / all fields (the formatter skips the padding
    # bookkeeping of digit columns present in every row, and the compress step if none is padded)
    alphabet = np.frombuffer(b"ACGT", np.uint8)
    cases = [
        (np.arange(10**7, 10**7 + 5000), np.full(5000, 37)),             # nothing padded
        (np.arange(10**7, 10**7 + 5000), rng.integers(0, 1000, 5000)),    # depth only
        (np.arange(99_000, 101_000), np.full(2000, 5)),                   # pos and pos+1 cross 10**5
        (np.arange(999_000, 1_000_000), np.full(1000, 12)),               # only pos+1 crosses 10**6
        (np.array([0, 9, 10, 99, 999999999, 2**31 - 2]), np.array([0, 9, 10, 99, 100, 65535])),
        (np.zeros(3, np.int64), np.zeros(3, np.int64)),
        (np.full(4, 2**32 - 1), np.full(4, 2**32)),                       # uint64 path, constant
    ]
    for pos, depth in cases:
        for rows in (F._FORMAT_BLOCK_ROWS, 7):                           # incl. many tiny blocks
            saved, F._FORMAT_BLOCK_ROWS = F._FORMAT_BLOCK_ROWS, rows
            try:
                tri = alphabet[rng.integers(0, 4, (pos.size, 3))]
                data, offs = F.format_context_lines("chrW", pos, tri, depth)
            finally:
                F._FORMAT_BLOCK_ROWS = saved
            ref, roffs = py_context_lines("chrW", pos, tri, depth)
            assert data == ref and np.array_equal(offs, roffs), f"format mismatch {pos[:2]}"
    # tiny / empty / errors
    d, o = F.format_context_lines("c", np.array([], np.int64), np.empty((0, 3), np.uint8),
                                  np.array([], np.uint16))
    assert d == b"" and o.size == 0
    d, o = F.format_context_lines("c", np.array([7]), np.array([[65, 67, 71]], np.uint8), np.array([3]))
    assert d == b"c\t7\t8\tACG\t3\n" and o.tolist() == [0]
    expect_raises(ValueError, F.format_context_lines, "c", np.array([-1]),
                  np.array([[65, 67, 71]], np.uint8), np.array([3]))
    expect_raises(ValueError, F.format_context_lines, "c", np.array([1]),
                  np.array([[65, 67, 71]], np.uint8), np.array([-3]))
    expect_raises(ValueError, F.format_context_lines, "c", np.array([1, 2]),
                  np.array([[65, 67, 71]], np.uint8), np.array([3, 4]))


def random_bed_text(rng, n, ncols):
    """Random BED-like lines: col 0 names, numeric cols, a 3-letter col, free text."""
    lines, fields_all = [], []
    names = ["chr1", "NC_000067.7", "HLA:01:02", "c"]
    for i in range(n):
        f = [names[int(rng.integers(0, len(names)))]]
        for c in range(1, ncols):
            kind = c % 4
            if kind in (1, 2):
                v = int(rng.choice(BOUNDARY)) if rng.random() < 0.2 else int(rng.integers(0, 2**31))
                f.append(str(v))
            elif kind == 3:
                f.append("".join(rng.choice(list("ACGTN"), 3)))
            else:
                f.append("x" * int(rng.integers(0, 6)))      # possibly empty free-text field
        fields_all.append(f)
        lines.append("\t".join(f) + "\n")
    return "".join(lines).encode(), fields_all


def test_parse(rng, n):
    log("parse_bed_chunk / parse_uint_field / field_bytes_fixed / select_lines vs Python")
    for ncols in (1, 3, 5, 8):
        buf, fields = random_bed_text(rng, n, ncols)
        ch = F.parse_bed_chunk(buf, ncols)
        assert ch.n == n and ch.u8.dtype == np.uint8 and ch.u8.size == len(buf)
        o = 0
        ls, le, fs, fe = [], [], [], []
        for f in fields:
            line = "\t".join(f).encode()
            ls.append(o)
            le.append(o + len(line))
            p = o
            rs, rr = [], []
            for x in f:
                rs.append(p)
                rr.append(p + len(x.encode()))
                p += len(x.encode()) + 1
            fs.append(rs)
            fe.append(rr)
            o += len(line) + 1
        assert ch.line_start.tolist() == ls and ch.line_end.tolist() == le
        assert ch.fstart.tolist() == fs and ch.fend.tolist() == fe
        assert ch.line_start.dtype == ch.fstart.dtype == np.int64
        for c in range(1, ncols):
            if c % 4 in (1, 2):
                v = F.parse_uint_field(ch, c)
                assert v.dtype == np.int64 and v.tolist() == [int(f[c]) for f in fields]
            elif c % 4 == 3:
                b = F.field_bytes_fixed(ch, c, 3)
                assert b.dtype == np.uint8 and b.shape == (n, 3)
                assert [bytes(r).decode() for r in b] == [f[c] for f in fields]
        lines = buf.splitlines(keepends=True)
        for density in (0.0, 1.0, 0.001, 0.02, 0.5, 0.97, "runs", "one", "first", "last"):
            if density == "runs":
                keep = np.repeat(rng.random(n // 50 + 1) < 0.5, 50)[:n]
            elif density == "one":
                keep = np.zeros(n, bool)
                keep[n // 2] = True
            elif density == "first":
                keep = np.zeros(n, bool)
                keep[0] = True
            elif density == "last":
                keep = np.ones(n, bool)
                keep[-1] = False
            else:
                keep = rng.random(n) < density
            data, offs = F.select_lines(ch, keep)
            kept = [ln for ln, k in zip(lines, keep) if k]
            assert data == b"".join(kept), f"select_lines mismatch ({density})"
            assert offs.dtype == np.int64
            assert offs.tolist() == list(np.cumsum([0] + [len(x) for x in kept[:-1]]))[:len(kept)]
    # boundaries for parse_uint_field
    vals = [0, 9, 10, 99, 100, 999999999, 2**31 - 2, 2**31 - 1, 2**32, 10**17, 10**18 - 1]
    buf = "".join(f"c\t{v}\n" for v in vals).encode()
    assert F.parse_uint_field(F.parse_bed_chunk(buf, 2), 1).tolist() == vals
    # empty chunk
    e = F.parse_bed_chunk(b"", 5)
    assert e.n == 0 and F.parse_uint_field(e, 1).size == 0
    assert F.select_lines(e, np.zeros(0, bool))[0] == b""
    assert F.field_bytes_fixed(e, 3, 3).shape == (0, 3)
    # errors
    expect_raises(ValueError, F.parse_bed_chunk, b"a\tb\tc", 3)                  # no final '\n'
    expect_raises(ValueError, F.parse_bed_chunk, b"a\tb\tc\na\tb\n", 3)          # missing field
    expect_raises(ValueError, F.parse_bed_chunk, b"a\tb\tc\td\na\tb\n", 3)       # same total tabs
    expect_raises(ValueError, F.parse_bed_chunk, b"a\tb\na\tb\tc\td\n", 3)       # same total tabs
    expect_raises(ValueError, F.parse_bed_chunk, b"a\tb\n", 1)
    ch = F.parse_bed_chunk(b"c\t1\t\nc\t2x\tAC\nc\t-3\tACG\n", 3)
    expect_raises(ValueError, F.parse_uint_field, ch, 2)                          # empty / letters
    expect_raises(ValueError, F.parse_uint_field, ch, 1)                          # '2x', '-3'
    expect_raises(ValueError, F.field_bytes_fixed, ch, 2, 3)
    expect_raises(ValueError, F.parse_uint_field, F.parse_bed_chunk(b"c\t1234567890123456789\n", 2), 1)
    expect_raises(ValueError, F.parse_uint_field, F.parse_bed_chunk(b"c\t1 \n", 2), 1)
    expect_raises(ValueError, F.select_lines, ch, np.ones(2, bool))


def test_parse_uint_exhaustive(rng):
    log("parse_uint_field: every length 1-18, leading zeros, buffer-start fields, bad bytes")
    for L in range(1, 19):
        nums = ["".join(rng.choice(list("0123456789"), L)) for _ in range(300)]
        nums += ["9" * L, "0" * L, "0" * (L - 1) + "1"]
        for layout in ("first", "middle", "last"):
            if layout == "first":
                buf = "".join(f"{x}\tA\n" for x in nums).encode()
                col, ncols = 0, 2
            elif layout == "middle":
                buf = "".join(f"c\t{x}\tAC\n" for x in nums).encode()
                col, ncols = 1, 3
            else:
                buf = "".join(f"chrom.1\t{x}\n" for x in nums).encode()
                col, ncols = 1, 2
            v = F.parse_uint_field(F.parse_bed_chunk(buf, ncols), col)
            assert v.tolist() == [int(x) for x in nums], f"parse mismatch L={L} {layout}"
    for buf, want in ((b"5\n", [5]), (b"12\t3\n", [12]), (b"1234567\n", [1234567]),
                      (b"7\n8\n9\n", [7, 8, 9]), (b"123456789\n", [123456789])):
        assert F.parse_uint_field(F.parse_bed_chunk(buf, buf.split(b"\n")[0].count(b"\t") + 1), 0).tolist() == want
    bad = [0x00, 0x01, 0x08, 0x0B, 0x0D, 0x20, 0x2B, 0x2D, 0x2F, 0x3A, 0x41, 0x7F, 0x80, 0x89, 0x8A,
           0xB0, 0xB9, 0xFF]
    for L in range(1, 19):
        for j in range(L):
            for b in bad:
                digits = bytearray(b"1" * L)
                digits[j] = b
                for pre in (b"", b"c\t"):
                    buf = b"c\t12\n" + pre + bytes(digits) + b"\n"
                    ch = F.parse_bed_chunk(buf, 2 if pre else 1) if pre else None
                    if ch is None:
                        buf = bytes(digits) + b"\n"
                        ch = F.parse_bed_chunk(buf, 1)
                        col = 0
                    else:
                        col = 1
                    expect_raises(ValueError, F.parse_uint_field, ch, col)
    # control bytes 0-8 are field content (not separators) for parse_bed_chunk
    ch = F.parse_bed_chunk(b"a\x01b\t\x00\t3\n", 3)
    assert ch.fstart.tolist() == [[0, 4, 6]] and ch.fend.tolist() == [[3, 5, 7]]


# --------------------------------------------------------------------------------------------
# 2. BGZFile.tell() virtual offsets
# --------------------------------------------------------------------------------------------

def test_bgzf_tell(rng, wd):
    log("pysam.BGZFile.tell() after writes == seekable virtual offsets")
    path = os.path.join(wd, "tell.gz")
    lines, voffs = [], []
    with pysam.BGZFile(path, "wb") as f:
        for i in range(30000):
            r = rng.random()
            if r < 0.001:
                ln = b"y" * (0xff00 - (f.tell() & 0xffff) - 1) + b"\n"     # fill block exactly
            elif r < 0.002:
                ln = b"z" * int(rng.integers(70000, 200000)) + b"\n"         # multi-block line
            else:
                ln = b"x" * int(rng.integers(0, 120)) + f"\t{i}\n".encode()
            voffs.append(f.tell())
            f.write(ln)
            lines.append(ln)
        end_voff = f.tell()
    assert all(b > a for a, b in zip(voffs, voffs[1:])), "tell() not increasing"
    assert any((v & 0xffff) == 0 and v > 0 for v in voffs), "no block-boundary case exercised"
    with pysam.BGZFile(path, "rb") as f:
        for i in list(range(0, len(lines), 7)) + [len(lines) - 1]:
            f.seek(voffs[i])
            assert f.readline() + b"\n" == lines[i], f"seek/readline mismatch at line {i}"
        f.seek(end_voff)
        assert f.readline() == b""
    assert gzip.open(path).read() == b"".join(lines)


# --------------------------------------------------------------------------------------------
# 3. index helpers: decode .tbi, BGZF block table, canonical form
# --------------------------------------------------------------------------------------------

def decode_tbi(path):
    raw = gzip.open(path, "rb").read()
    o = 0

    def rd(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, raw, o)
        o += struct.calcsize("<" + fmt)
        return v

    magic, n_ref, fmt, cs, cb, ce, meta, skip, l_nm = rd("4s8i")
    names = raw[o:o + l_nm].split(b"\0")[:-1]
    o += l_nm
    refs = []
    for _ in range(n_ref):
        (n_bin,) = rd("i")
        bins = {}
        for _ in range(n_bin):
            b, nc = rd("Ii")
            assert b not in bins, "duplicate bin"
            bins[b] = [rd("QQ") for _ in range(nc)]
        (n_intv,) = rd("i")
        lin = list(rd(f"{n_intv}Q")) if n_intv else []
        refs.append((bins, lin))
    (n_no_coor,) = rd("Q")
    assert o == len(raw), f"{len(raw) - o} trailing bytes in {path}"
    return dict(magic=magic, n_ref=n_ref, fmt=fmt, cols=(cs, cb, ce), meta=meta, skip=skip,
                names=[x.decode() for x in names], refs=refs, n_no_coor=n_no_coor)


def bgzf_blocks(path):
    """{block address: (uncompressed start, uncompressed size)} for every BGZF block."""
    table = {}
    u = 0
    with open(path, "rb") as f:
        data = f.read()
    a = 0
    while a < len(data):
        assert data[a:a + 4] == b"\x1f\x8b\x08\x04", f"not a BGZF block at {a}"
        bsize = struct.unpack_from("<H", data, a + 16)[0] + 1
        isize = struct.unpack_from("<I", data, a + bsize - 4)[0]
        table[a] = (u, isize)
        u += isize
        a += bsize
    table[a] = (u, 0)                     # end of file
    return table


TBI_LEAF0, TBI_META = 4681, 37450


def verify_tabix_layout(tbi_path):
    """Assert, on an index written by `tabix -p bed` / pysam.tabix_index for 1-bp BED records,
    every layout rule build_tbi relies on (raw virtual offsets, no canonicalisation):
      header: magic, format 0x10000 (TBX_UCSC), cols 1/2/3, meta '#', skip 0, names; trailing
        uint64 n_no_coor present (decode_tbi consumes every byte) and 0;
      per contig: only bottom-level bins 4681 + window (one chunk each) + pseudo-bin 37450 =
        [(first record voff, end voff), (n_records, 0)]; leaf chunks chained (chunk end == next
        present window's chunk start; last end == pseudo-bin end); n_intv == last window + 1;
        linear index == chunk start of the window, or of the NEXT present window for empty ones.
    Returns (decoded index, {contig: n_records}, stats)."""
    tbi = decode_tbi(tbi_path)
    assert tbi["magic"] == b"TBI\x01" and tbi["fmt"] == 0x10000 and tbi["cols"] == (1, 2, 3)
    assert tbi["meta"] == ord("#") and tbi["skip"] == 0 and tbi["n_no_coor"] == 0
    assert len(tbi["names"]) == tbi["n_ref"] == len(set(tbi["names"]))
    nrec = {}
    st = dict(windows=0, gap_windows=0, gap_fill_direction_matters=0)
    for name, (bins, lin) in zip(tbi["names"], tbi["refs"]):
        bins = dict(bins)
        meta = bins.pop(TBI_META)
        assert len(meta) == 2, f"{name}: pseudo-bin has {len(meta)} chunks"
        (u0, v0), (n_mapped, n_unmapped) = meta
        assert n_unmapped == 0
        leaf = sorted(bins)
        assert leaf and all(TBI_LEAF0 <= b < TBI_LEAF0 + (1 << 15) for b in leaf), \
            f"{name}: non-leaf bin present"
        assert all(len(bins[b]) == 1 for b in leaf), f"{name}: leaf bin with != 1 chunk"
        u = np.array([bins[b][0][0] for b in leaf], np.uint64)
        v = np.array([bins[b][0][1] for b in leaf], np.uint64)
        assert u[0] == u0 and v[-1] == v0, f"{name}: pseudo-bin chunk != leaf chunk span"
        assert np.array_equal(v[:-1], u[1:]), f"{name}: leaf chunks not chained"
        wins = np.array(leaf, np.int64) - TBI_LEAF0
        assert len(lin) == wins[-1] + 1, f"{name}: n_intv != last window + 1"
        nxt = np.searchsorted(wins, np.arange(len(lin)), side="left")
        assert np.array_equal(np.array(lin, np.uint64), u[nxt]), f"{name}: linear index rule"
        gap = np.ones(len(lin), bool)
        gap[wins] = False
        gi = np.flatnonzero(gap)
        st["windows"] += len(lin)
        st["gap_windows"] += gi.size
        # gaps after the first present window, where back-fill and forward-fill would differ
        prv = np.searchsorted(wins, gi, side="right") - 1
        ok = prv >= 0
        st["gap_fill_direction_matters"] += int(np.count_nonzero(u[nxt[gi][ok]] != u[prv[ok]]))
        nrec[name] = n_mapped
    return tbi, nrec, st


def canonical(tbi, blocks):
    def up(v):
        a, off = v >> 16, v & 0xffff
        assert a in blocks, f"virtual offset {hex(v)} does not point at a block"
        assert off <= blocks[a][1], f"virtual offset {hex(v)} beyond block end"
        return blocks[a][0] + off

    refs = []
    for bins, lin in tbi["refs"]:
        cb = {}
        for b, chunks in bins.items():
            if b == 37450:
                (u0, v0), (nm, nu) = chunks
                cb[b] = [(up(u0), up(v0)), (nm, nu)]
            else:
                cb[b] = [(up(u), up(v)) for u, v in chunks]
        refs.append((cb, [up(x) for x in lin]))
    d = dict(tbi)
    d["refs"] = refs
    return d


# --------------------------------------------------------------------------------------------
# 4. datasets
# --------------------------------------------------------------------------------------------

def edge_dataset(rng, scale):
    """Fragments [(contig, positions or None)], hand-made edge cases."""
    dense = np.arange(0, 13 * W + 77, dtype=np.int64)[: max(2000, int(13 * W * scale))]
    s1 = len(dense) // 3                                          # mid-window splits
    s2 = s1 + min(5 * W // 2, (len(dense) - s1) // 2)
    frags = [
        ("chr1", dense[:s1]), ("chr1", dense[s1:s2]), ("chr1", None), ("chr1", dense[s2:]),
        ("empty_contig", None),
        ("chr.2", np.array([5, W - 1, W, W + 1, 10 * W + 7, 1000 * W, 1000 * W + W - 1,
                            30000 * W + 1, 2**29 - 2, 2**29 - 1], np.int64)),
        ("HLA:01:02", np.array([3, 4, 5 * W, 5 * W + 3], np.int64)),
        ("single", np.array([0], np.int64)),
        ("zero", np.zeros(5, np.int64)),
        ("hugewin", np.arange(3 * W, 4 * W, dtype=np.int64)),
        ("boundary", np.unique(np.concatenate([np.arange(1, 60) * W - 1, np.arange(1, 60) * W,
                                               np.arange(80, 90) * W]))),
        ("far", np.array([2**28 + 17], np.int64)),
        ("dups", np.repeat(np.array([W - 1, W, 7 * W], np.int64), 400)),
        (SENTINEL, np.array([SENTINEL_POS], np.int64)),
    ]
    return frags


def random_dataset(rng, scale):
    frags = []
    ncontig = int(rng.integers(3, 9))
    for c in range(ncontig):
        name = f"rnd{c}" + str(rng.choice(["", ".1", "_alt.7", ":x"]))
        length = int(rng.choice([W, 50 * W, 2**22, 2**26, 2**28]))
        parts = []
        for _ in range(int(rng.integers(0, 12))):
            st = int(rng.integers(0, length))
            ln = int(rng.integers(1, max(2, int(60000 * scale))))
            parts.append(np.arange(st, min(length, st + ln)))
        parts.append(rng.integers(0, length, int(rng.integers(0, 200))))
        pos = np.unique(np.concatenate(parts)).astype(np.int64)
        if pos.size == 0:
            frags.append((name, None))
            continue
        nsplit = int(rng.integers(0, 4))
        cuts = np.sort(rng.integers(0, pos.size, nsplit))
        for piece in np.split(pos, cuts):
            frags.append((name, piece if piece.size else None))
            if rng.random() < 0.2:
                frags.append((name, None))
    frags.append((SENTINEL, np.array([SENTINEL_POS], np.int64)))
    return frags


def realistic_dataset(rng, nlines):
    """One dense contig like a context file: long callable runs separated by gaps."""
    runs = []
    p = int(rng.integers(0, 5000))
    total = 0
    while total < nlines:
        ln = int(rng.geometric(1 / 3000))
        runs.append(np.arange(p, p + ln))
        total += ln
        p += ln + int(rng.geometric(1 / 400)) + (int(rng.integers(0, 3 * W)) if rng.random() < 0.01 else 0)
    pos = np.concatenate(runs)[:nlines].astype(np.int64)
    mid = pos.size // 2
    return [("NC_000085.7", pos[:mid]), ("NC_000085.7", pos[mid:]),
            (SENTINEL, np.array([SENTINEL_POS], np.int64))]


def write_dataset(rng, wd, name, frags, header):
    """Write fragments with IndexedBgzfWriter, assemble; return (path, expected text, contigs)."""
    paths, text = [], [header]
    contig_order = []
    alphabet = np.frombuffer(b"ACGT", np.uint8)
    for i, (contig, pos) in enumerate(frags):
        p = os.path.join(wd, f"{name}.frag{i}.gz")
        w = F.IndexedBgzfWriter(p, contig=contig if rng.random() < 0.5 else None)
        if pos is not None:
            n = pos.size
            tri = alphabet[rng.integers(0, 4, (n, 3))]
            depth = rng.integers(1, 400, n).astype(np.uint16)
            data, offs = F.format_context_lines(contig, pos, tri, depth)
            text.append(data)
            # random write_records() call boundaries (incl. single records)
            cuts = np.unique(rng.integers(0, n, int(rng.integers(0, 6))))
            bounds = [0] + [int(c) for c in cuts if c > 0] + [n]
            for a, b in zip(bounds, bounds[1:]):
                lo = int(offs[a])
                hi = int(offs[b]) if b < n else len(data)
                w.write_records(data[lo:hi], pos[a:b], offs[a:b] - lo)
            if not contig_order or contig_order[-1] != contig:
                contig_order.append(contig)
        fi = w.close()
        assert (fi is None) == (pos is None)
        if fi is not None:
            assert fi.contig == contig and fi.n_records == pos.size
            assert np.array_equal(fi.windows, np.unique(pos >> F.WINDOW_SHIFT))
            fi2 = pickle.loads(pickle.dumps(fi))
            assert fi2.contig == fi.contig and np.array_equal(fi2.win_voffs, fi.win_voffs)
        paths.append((p, fi))
    final = os.path.join(wd, f"{name}.bed.gz")
    F.assemble_bgzf(final, header, paths, index=True)
    assert not any(os.path.exists(p) for p, _ in paths), "fragments not removed"
    return final, b"".join(text), contig_order


# --------------------------------------------------------------------------------------------
# 5. query comparisons
# --------------------------------------------------------------------------------------------

def make_regions(rng, frags, n_random, max_log2=24):
    pos_by = {}
    for c, p in frags:
        if p is not None:
            pos_by.setdefault(c, []).append(p)
    pos_by = {c: np.concatenate(v) for c, v in pos_by.items()}
    contigs = [c for c in pos_by if c != SENTINEL]      # random regions: CLI-comparable
    regs = []
    for c, p in pos_by.items():
        regs.append((c, None, None))
        wins = np.unique(p >> F.WINDOW_SHIFT)
        pick = wins if wins.size <= 20 else rng.choice(wins, 20, replace=False)
        for k in pick.tolist():
            for kk in (k, k + 1):
                s = kk * W
                regs += [(c, s, s + W), (c, max(0, s - 1), s + 1), (c, s, s + 1),
                         (c, max(0, s - 1), s), (c, s, s + 2 * W), (c, max(0, s - W), s)]
        gaps = np.flatnonzero(np.diff(wins) > 1)
        for g in gaps[:20].tolist():
            a, b = int(wins[g] + 1) * W, int(wins[g + 1]) * W
            regs += [(c, a, b), (c, a + 1, b - 1), (c, a, b + 1), (c, a - 1, a + 1),
                     (c, (a + b) // 2, (a + b) // 2 + 1)]
        regs += [(c, 0, 1), (c, 0, W), (c, int(p[0]), int(p[0]) + 1), (c, int(p[-1]), int(p[-1]) + 1),
                 (c, int(p[-1]) + 1, int(p[-1]) + 1000), (c, 2**29 - 1, 2**29),
                 (c, 0, 2**29), (c, int(p[-1]), 2**31 - 1)]
    regs += [("no_such_contig", None, None), ("no_such_contig", 0, 100)]
    for _ in range(n_random):
        c = contigs[int(rng.integers(0, len(contigs)))]
        p = pos_by[c]
        if rng.random() < 0.6:
            s = int(p[int(rng.integers(0, p.size))]) + int(rng.integers(-3 * W, 3 * W))
        else:
            s = int(rng.integers(0, int(p[-1]) + 4 * W))
        s = max(0, s)
        ln = int(2 ** rng.uniform(0, max_log2))
        regs.append((c, s, s + ln))
    return regs


def fetch(tf, c, s, e):
    try:
        if s is None:
            return list(tf.fetch(c))
        return list(tf.fetch(c, s, e))
    except ValueError as ex:
        return ("ValueError", str(ex).split("'")[0])


def compare_pysam(path, ref_path, regions, expected_lines_by_contig):
    a = pysam.TabixFile(path)
    b = pysam.TabixFile(ref_path)
    nonempty = 0
    for c, s, e in regions:
        ra, rb = fetch(a, c, s, e), fetch(b, c, s, e)
        assert ra == rb, f"pysam fetch differs for {c}:{s}-{e}: {len(ra)} vs {len(rb)}"
        if isinstance(ra, list):
            nonempty += bool(ra)
            # and against a brute-force overlap scan of the expected records
            exp = expected_lines_by_contig.get(c, ([], np.zeros(0, np.int64)))
            lines, pos = exp
            if s is None:
                want = lines
            else:
                lo, hi = np.searchsorted(pos, s), np.searchsorted(pos, e)
                want = lines[lo:hi]
            assert ra == want, f"pysam fetch != expected for {c}:{s}-{e}"
    # whole-file iteration, on the reused handles (htslib skips the initial seek when the
    # first offset is 0, so this also checks first-record offsets) and on fresh handles
    assert list(a.fetch()) == list(b.fetch())
    assert list(pysam.TabixFile(path).fetch()) == list(pysam.TabixFile(ref_path).fetch())
    a.close()
    b.close()
    return nonempty


def tabix_cli_batch(path, regions):
    """Per-region tabix CLI outputs, one subprocess: regions alternate with a sentinel query."""
    args = []
    for c, s, e in regions:
        args += [F._tabix_region(c, s, e), SENTINEL]
    out = subprocess.run(["tabix", path] + args, check=True, capture_output=True).stdout
    groups, cur = [], []
    for ln in out.decode().split("\n")[:-1]:
        if ln.startswith(SENTINEL + "\t"):
            groups.append(cur)
            cur = []
        else:
            cur.append(ln)
    assert not cur and len(groups) == len(regions), "sentinel parsing failed"
    return groups


def compare_cli(path, ref_path, regions, n_single):
    regs = [r for r in regions if r[0] != SENTINEL and F._tabix_region(*r) is not None]
    ga = tabix_cli_batch(path, regs)
    gb = tabix_cli_batch(ref_path, regs)
    assert ga == gb, "tabix CLI output differs between custom and tabix-built index"
    tf = pysam.TabixFile(path)
    for (c, s, e), g in zip(regs, ga):
        r = fetch(tf, c, s, e)
        if isinstance(r, list):
            assert r == g, f"tabix CLI != pysam for {c}:{s}-{e}"
    tf.close()
    # some single-region calls too (independent of the sentinel trick)
    for c, s, e in regs[:n_single]:
        reg = F._tabix_region(c, s, e)
        oa = subprocess.run(["tabix", path, reg], check=True, capture_output=True).stdout
        ob = subprocess.run(["tabix", ref_path, reg], check=True, capture_output=True).stdout
        assert oa == ob, f"single tabix CLI call differs for {reg}"
    return len(regs)


def compare_iter(path, regions, rng, n):
    tf = pysam.TabixFile(path)
    sel = [r for r in regions if r[0] != "no_such_contig"]
    idx = rng.choice(len(sel), min(n, len(sel)), replace=False)
    for i in idx.tolist():
        c, s, e = sel[i]
        want = fetch(tf, c, s, e)
        if not isinstance(want, list):
            continue
        want = "".join(x + "\n" for x in want).encode()
        small = max(int(rng.integers(1, 200)), len(want) // 200)   # many chunks, bounded count
        for cb in (small, 4096, 16 << 20):
            chunks = list(F.iter_tabix_chunks(path, c, s, e, chunk_bytes=cb))
            assert all(ch.endswith(b"\n") and ch for ch in chunks), "chunk with partial line"
            assert b"".join(chunks) == want, f"iter_tabix_chunks != fetch for {c}:{s}-{e}"
    tf.close()


def no_children():
    """True iff this process has no child at all (running or zombie)."""
    try:
        r = os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
        return True
    if r == (0, 0):
        return False                       # a child is still running
    raise AssertionError(f"reaped a stray (zombie) child {r}")


def check_dataset(rng, wd, name, frags, header, n_random, n_single, n_iter, max_log2=24):
    t0 = time.time()
    path, text, contig_order = write_dataset(rng, wd, name, frags, header)
    nrec = sum(p.size for _, p in frags if p is not None)
    # decompressed content
    assert gzip.open(path).read() == text, "gzip content mismatch"
    with pysam.BGZFile(path, "rb") as f:
        assert f.read() == text, "pysam BGZFile content mismatch"
    out = subprocess.run(["bgzip", "-d", "-c", path], check=True, capture_output=True).stdout
    assert out == text, "bgzip -d content mismatch"
    # tabix-built index on an identical copy
    ref = os.path.join(wd, f"{name}.ref.bed.gz")
    shutil.copyfile(path, ref)
    subprocess.run(["tabix", "-f", "-p", "bed", ref], check=True)
    # the original pipeline indexed with pysam.tabix_index(preset="bed"): same index as the CLI
    ref2 = os.path.join(wd, f"{name}.ref2.bed.gz")
    shutil.copyfile(path, ref2)
    pysam.tabix_index(ref2, preset="bed", force=True)
    assert decode_tbi(ref2 + ".tbi") == decode_tbi(ref + ".tbi"), "pysam.tabix_index != tabix -p bed"
    os.remove(ref2)
    os.remove(ref2 + ".tbi")
    _, ref_counts, _ = verify_tabix_layout(ref + ".tbi")        # tabix's own layout rules
    want_counts = {}
    for c, p in frags:
        if p is not None:
            want_counts[c] = want_counts.get(c, 0) + p.size
    assert ref_counts == want_counts
    blocks = bgzf_blocks(path)
    mine, theirs = decode_tbi(path + ".tbi"), decode_tbi(ref + ".tbi")
    assert mine["names"] == theirs["names"] == contig_order, (mine["names"], theirs["names"])
    for k in ("magic", "n_ref", "fmt", "cols", "meta", "skip", "n_no_coor"):
        assert mine[k] == theirs[k], f"tbi header field {k}: {mine[k]} vs {theirs[k]}"
    cm, ct = canonical(mine, blocks), canonical(theirs, blocks)
    for nm, (bm, lm), (bt, lt) in zip(mine["names"], cm["refs"], ct["refs"]):
        assert bm == bt, f"bins differ for {nm}: {sorted(set(bm) ^ set(bt))[:5]}"
        assert lm == lt, f"linear index differs for {nm}"
    assert F.tabix_contigs(path) == contig_order
    # queries
    lines_by = {}
    for ln in text.decode().split("\n"):
        if ln and not ln.startswith("#"):
            c, s = ln.split("\t", 2)[:2]
            lines_by.setdefault(c, ([], []))
            lines_by[c][0].append(ln)
            lines_by[c][1].append(int(s))
    lines_by = {c: (v[0], np.array(v[1], np.int64)) for c, v in lines_by.items()}
    regions = make_regions(rng, frags, n_random, max_log2)
    ne = compare_pysam(path, ref, regions, lines_by)
    ncli = compare_cli(path, ref, regions, n_single)
    compare_iter(path, regions, rng, n_iter)
    assert no_children()
    log(f"  {name}: {nrec} records, {len(contig_order)} contigs, {len(blocks)} BGZF blocks; "
        f"{len(regions)} pysam regions ({ne} non-empty), {ncli} CLI regions: identical "
        f"[{time.time() - t0:.1f}s]")
    for p in (path, path + ".tbi", ref, ref + ".tbi"):
        os.remove(p)


# --------------------------------------------------------------------------------------------
# 6. misc behaviour
# --------------------------------------------------------------------------------------------

def test_misc(rng, wd):
    log("limits, error paths, iter_tabix_chunks lifecycle")
    tri = np.frombuffer(b"ACG" * 4, np.uint8).reshape(4, 3)
    # positions >= 2**29 cannot be stored in a .tbi (tabix refuses too)
    pos = np.array([1, 2**29 - 1, 2**29, 2**29 + 5], np.int64)
    data, offs = F.format_context_lines("big", pos, tri, np.ones(4, np.int64))
    p = os.path.join(wd, "big.frag.gz")
    with F.IndexedBgzfWriter(p) as w:
        w.write_records(data, pos, offs)
    fi = w.close()
    final = os.path.join(wd, "big.bed.gz")
    expect_raises(ValueError, F.assemble_bgzf, final, b"", [(p, fi)], True, False)
    F.assemble_bgzf(final, b"", [(p, fi)], index=False, remove_fragments=True)
    assert not os.path.exists(final + ".tbi") and gzip.open(final).read() == data
    r = subprocess.run(["tabix", "-f", "-p", "bed", final], capture_output=True)
    assert r.returncode != 0, "tabix unexpectedly indexed pos >= 2**29"
    # 2**29 - 1 itself is fine (checked against tabix in the edge dataset)

    # writer argument checks
    p2 = os.path.join(wd, "w.frag.gz")
    w = F.IndexedBgzfWriter(p2)
    d1, o1 = F.format_context_lines("c1", np.array([5, 6]), tri[:2], np.array([1, 1]))
    expect_raises(ValueError, w.write_records, d1, np.array([6, 5]), o1)          # unsorted
    w.write_records(d1, np.array([5, 6]), o1)
    expect_raises(ValueError, w.write_records, d1, np.array([4, 5]), o1)          # decreasing
    d2, o2 = F.format_context_lines("c2", np.array([7]), tri[:1], np.array([1]))
    expect_raises(ValueError, w.write_records, d2, np.array([7]), o2)             # other contig
    w.write_records(bytearray(b"c1\t9\t10\tACG\t1\n"), np.array([9]), np.array([0]))
    fi = w.close()
    assert fi.n_records == 3 and fi.windows.tolist() == [0] and fi.contig == "c1"

    # raw-only writer -> None; unindexed data in an indexed assembly is refused
    p3 = os.path.join(wd, "raw.frag.gz")
    w = F.IndexedBgzfWriter(p3)
    w.write_raw(b"c\t1\t2\tmuts row\n")
    assert w.close() is None
    expect_raises(ValueError, F.assemble_bgzf, os.path.join(wd, "x.bed.gz"), b"", [(p3, None)], True)
    # non-consecutive contig fragments are refused (tabix: "Chromosome blocks not continuous")
    frs = []
    for i, (c, ps) in enumerate([("a", [1]), ("b", [1]), ("a", [5])]):
        pp = os.path.join(wd, f"nc{i}.gz")
        dd, oo = F.format_context_lines(c, np.array(ps), tri[:1], np.array([1]))
        with F.IndexedBgzfWriter(pp) as w:
            w.write_records(dd, np.array(ps), oo)
        frs.append((pp, w.close()))
    expect_raises(ValueError, F.assemble_bgzf, os.path.join(wd, "nc.bed.gz"), b"", frs, True, False)
    # header must be '#' lines when indexing
    expect_raises(ValueError, F.assemble_bgzf, os.path.join(wd, "nc.bed.gz"), b"#ok\nbad\n", frs[:1], True, False)
    expect_raises(ValueError, F.assemble_bgzf, os.path.join(wd, "nc.bed.gz"), b"#no newline", frs[:1], True, False)
    # index=False: plain concatenation, header b"" allowed, raw fragments allowed
    out = os.path.join(wd, "plain.bed.gz")
    F.assemble_bgzf(out, b"", [(p3, None), (p2, fi)] + frs, index=False, remove_fragments=False)
    want = b"c\t1\t2\tmuts row\n" + d1 + b"c1\t9\t10\tACG\t1\n" + b"".join(
        gzip.open(pp).read() for pp, _ in frs)
    assert gzip.open(out).read() == want
    for pp in [p3, p2, out] + [x for x, _ in frs]:
        os.remove(pp)

    # outputs are replaced via rename: an existing (hard-linked) output is never truncated in
    # place, and a failure leaves neither a partial output nor a temporary file behind
    old = os.path.join(wd, "old.bed.gz")
    link = os.path.join(wd, "old_link.bed.gz")
    with pysam.BGZFile(old, "wb") as f:
        f.write(b"#old\n")
    with pysam.BGZFile(old + ".tbi", "wb") as f:
        f.write(b"old index")
    os.link(old, link)
    os.link(old + ".tbi", link + ".tbi")
    pr = os.path.join(wd, "repl.frag.gz")
    dr, orr = F.format_context_lines("r", np.array([3, 70000]), tri[:2], np.array([1, 2]))
    with F.IndexedBgzfWriter(pr) as w:
        w.write_records(dr, np.array([3, 70000]), orr)
    F.assemble_bgzf(old, b"#new\n", [(pr, w.close())], index=True)
    assert gzip.open(old).read() == b"#new\n" + dr
    assert gzip.open(link).read() == b"#old\n" and gzip.open(link + ".tbi").read() == b"old index"
    assert [x for x in os.listdir(wd) if ".part" in x] == []
    assert os.path.getmtime(old + ".tbi") >= os.path.getmtime(old)
    tf = pysam.TabixFile(old)
    assert list(tf.fetch("r", 69999, 70001)) == ["r\t70000\t70001\tACG\t2"]
    tf.close()
    pe2 = os.path.join(wd, "e2.frag.gz")
    F.IndexedBgzfWriter(pe2).close()
    bad = os.path.join(wd, "a_directory")
    os.mkdir(bad)
    try:
        F.assemble_bgzf(old, b"#x\n", [(pe2, None), (bad, None)], index=False)
        raise AssertionError("assembling a directory did not raise")
    except OSError:
        pass
    assert gzip.open(old).read() == b"#new\n" + dr, "failed assembly modified the output"
    assert [x for x in os.listdir(wd) if ".part" in x] == [], "temporary file left behind"
    os.rmdir(bad)
    for x in (old, old + ".tbi", link, link + ".tbi", pe2):
        os.remove(x)

    # data-less fragments are not copied: a header-less file whose first fragments are empty
    # still starts with the first record at virtual offset 0 (as tabix indexes it)
    pz = [os.path.join(wd, f"z{i}.frag.gz") for i in range(4)]
    F.IndexedBgzfWriter(pz[0]).close()
    open(pz[1], "wb").close()                     # even a 0-byte file
    dz, oz = F.format_context_lines("z", np.array([0, 5, 40000]), tri[:3], np.array([1, 2, 3]))
    with F.IndexedBgzfWriter(pz[2]) as w:
        w.write_records(dz, np.array([0, 5, 40000]), oz)
    fz = w.close()
    F.IndexedBgzfWriter(pz[3]).close()
    sz2 = os.path.getsize(pz[2])
    final = os.path.join(wd, "z.bed.gz")
    F.assemble_bgzf(final, b"", [(pz[0], None), (pz[1], None), (pz[2], fz), (pz[3], None)], index=True)
    assert not any(os.path.exists(x) for x in pz)
    assert os.path.getsize(final) == sz2 and gzip.open(final).read() == dz
    assert decode_tbi(final + ".tbi")["refs"][0][0][37450][0][0] == 0
    ref = os.path.join(wd, "z.ref.bed.gz")
    shutil.copyfile(final, ref)
    subprocess.run(["tabix", "-f", "-p", "bed", ref], check=True)
    blocks = bgzf_blocks(final)
    assert canonical(decode_tbi(final + ".tbi"), blocks) == canonical(decode_tbi(ref + ".tbi"), blocks)
    tz, tr = pysam.TabixFile(final), pysam.TabixFile(ref)
    for q in (("z", 4, 6), None, ("z", 39999, 40001), None):        # incl. reused-handle fetch()
        ra = list(tz.fetch(*q)) if q else list(tz.fetch())
        rb = list(tr.fetch(*q)) if q else list(tr.fetch())
        assert ra == rb
    tz.close()
    tr.close()
    for x in (final, final + ".tbi", ref, ref + ".tbi"):
        os.remove(x)
    # only empty fragments, no header: a valid empty BGZF file
    pz0 = os.path.join(wd, "z0.frag.gz")
    F.IndexedBgzfWriter(pz0).close()
    F.assemble_bgzf(final, b"", [(pz0, None)], index=True)
    assert os.path.getsize(final) == 28 and gzip.open(final).read() == b""
    assert decode_tbi(final + ".tbi")["n_ref"] == 0
    os.remove(final)
    os.remove(final + ".tbi")

    # header-only file with index: same as tabix's empty index
    pe = os.path.join(wd, "empty.frag.gz")
    F.IndexedBgzfWriter(pe).close()
    final = os.path.join(wd, "empty.bed.gz")
    F.assemble_bgzf(final, b"#h1\n#h2\n", [(pe, None)], index=True)
    ref = os.path.join(wd, "empty.ref.bed.gz")
    shutil.copyfile(final, ref)
    subprocess.run(["tabix", "-f", "-p", "bed", ref], check=True)
    assert decode_tbi(final + ".tbi") == decode_tbi(ref + ".tbi")
    assert F.tabix_contigs(final) == []
    assert list(F.iter_tabix_chunks(final, "chr1")) == []
    for x in (final, final + ".tbi", ref, ref + ".tbi"):
        os.remove(x)

    # compression level only changes compressed bytes
    pos = np.arange(0, 300000, 7, dtype=np.int64)
    dl, ol = F.format_context_lines("lv", pos, np.tile(tri[:1], (pos.size, 1)), pos % 97)
    outs = {}
    for lvl in (None, 1, 4, 9):
        pf = os.path.join(wd, f"lv{lvl}.frag.gz")
        with F.IndexedBgzfWriter(pf, level=lvl) as w:
            w.write_records(dl, pos, ol)
        fo = os.path.join(wd, f"lv{lvl}.bed.gz")
        F.assemble_bgzf(fo, b"#h\n", [(pf, w.close())], index=True, level=lvl)
        assert gzip.open(fo).read() == b"#h\n" + dl
        tf = pysam.TabixFile(fo)
        outs[lvl] = [list(tf.fetch("lv", a, a + 5000)) for a in range(0, 310000, 4999)]
        tf.close()
        outs[lvl, "size"] = os.path.getsize(fo)
        os.remove(fo)
        os.remove(fo + ".tbi")
    assert outs[None] == outs[1] == outs[4] == outs[9]
    assert outs[1, "size"] != outs[9, "size"]
    expect_raises(ValueError, F.IndexedBgzfWriter, os.path.join(wd, "bad.gz"), None, 12)

    # iter_tabix_chunks lifecycle
    n = 200000
    pos = np.arange(n, dtype=np.int64) * 3
    data, offs = F.format_context_lines("chrL", pos, np.tile(tri[:1], (n, 1)), np.full(n, 12))
    pl = os.path.join(wd, "life.frag.gz")
    with F.IndexedBgzfWriter(pl) as w:
        w.write_records(data, pos, offs)
    final = os.path.join(wd, "life.bed.gz")
    F.assemble_bgzf(final, b"#h\n", [(pl, w.close())], index=True)
    assert no_children()
    g = F.iter_tabix_chunks(final, "chrL", chunk_bytes=1 << 16)
    first = next(g)
    assert first.endswith(b"\n") and data.startswith(first)
    g.close()                                     # early close: tabix killed and reaped
    assert no_children(), "zombie / running tabix after early close"
    g = F.iter_tabix_chunks(final, "chrL", chunk_bytes=1 << 16)
    next(g)
    del g                                         # dropped reference: same cleanup
    assert no_children(), "child left after generator was garbage-collected"
    for _ in F.iter_tabix_chunks(final, "chrL", 10, 20):
        break
    assert no_children()
    assert b"".join(F.iter_tabix_chunks(final, "chrL")) == data
    assert b"".join(F.iter_tabix_chunks(final, "chrL", 3, 7)) == data[offs[1]:offs[3]]
    assert b"".join(F.iter_tabix_chunks(final, "chrL", None, 4)) == data[:offs[2]]
    assert b"".join(F.iter_tabix_chunks(final, "chrL", 5 * 3)) == data[offs[5]:]
    assert list(F.iter_tabix_chunks(final, "chrL", 10, 10)) == []
    assert list(F.iter_tabix_chunks(final, "missing", 0, 10)) == []
    os.remove(final + ".tbi")
    try:
        list(F.iter_tabix_chunks(final, "chrL"))
        raise AssertionError("missing index did not raise")
    except RuntimeError:
        pass
    try:
        list(F.iter_tabix_chunks(os.path.join(wd, "nope.bed.gz"), "chrL"))
        raise AssertionError("missing file did not raise")
    except RuntimeError:
        pass
    assert no_children()
    os.remove(final)
    # region strings
    assert F._tabix_region("chr1", None, None) == "chr1"
    assert F._tabix_region("chr1", 0, 10) == "chr1:1-10"
    assert F._tabix_region("HLA:01", 4, 9) == "{HLA:01}:5-9"
    assert F._tabix_region("HLA:01", None, None) == "{HLA:01}"
    assert F._tabix_region("chr1", 5, 5) is None


# --------------------------------------------------------------------------------------------
# 7. real file (--real): rebuild an original-pipeline context file through the writer path
# --------------------------------------------------------------------------------------------

def md5_gz(path):
    """md5 + size of the decompressed content (zlib, independent of htslib)."""
    import hashlib
    import zlib
    h = hashlib.md5()
    d = zlib.decompressobj(31)
    n = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 24)
            if not buf:
                break
            while buf:
                out = d.decompress(buf)
                h.update(out)
                n += len(out)
                if d.eof:
                    buf = d.unused_data
                    d = zlib.decompressobj(31)
                else:
                    buf = b""
    return h.hexdigest(), n


def iter_compare(it_a, it_b, what):
    """Compare two line iterators without materialising them; returns the line count."""
    import itertools
    n = 0
    for x, y in itertools.zip_longest(it_a, it_b):
        assert x == y, f"{what}: line {n} differs: {x!r} vs {y!r}"
        n += 1
    return n


def check_real(orig, wd, rng, n_random, n_cli):
    """`orig`: a tabix-indexed context BED written by the ORIGINAL pipeline (read-only).
    1. its .tbi obeys every layout rule build_tbi relies on;
    2. every record is streamed (iter_tabix_chunks), parsed (parse_bed_chunk/parse_uint_field/
       field_bytes_fixed) and re-formatted (format_context_lines) byte-identically, then written
       with IndexedBgzfWriter (the largest contig split over 3 consecutive fragments: mid-window,
       empty, window boundary) and assembled behind the original header (assemble_bgzf);
    3. decompressed content == original (zlib md5); custom .tbi == original .tbi once virtual
       offsets are mapped to uncompressed positions;
    4. original file + original index vs rebuilt file + custom index: whole contigs, >= n_random
       random regions + window-edge / gap regions via pysam, n_cli regions via the tabix CLI."""
    t00 = time.time()
    log(f"real file {orig}")
    tbi, nrec, st = verify_tabix_layout(orig + ".tbi")
    log(f"  original .tbi layout rules hold: {len(nrec)} contigs, {sum(nrec.values())} records, "
        f"{st['windows']} linear-index windows of which {st['gap_windows']} empty "
        f"({st['gap_fill_direction_matters']} where back- and forward-fill differ: all back-filled), "
        f"trailing n_no_coor=0 present")
    hdr = subprocess.run(["tabix", "-H", orig], check=True, capture_output=True).stdout
    contigs = F.tabix_contigs(orig)
    assert contigs == tbi["names"]
    big = max(contigs, key=lambda c: nrec[c])
    base = os.path.basename(orig)
    new = os.path.join(wd, "rebuilt." + base)
    frags, sample, wins_by = [], {}, {}
    tt = dict(stream=0.0, parse=0.0, fmt=0.0, write=0.0)
    nsel = 0
    nsplit = 0

    def frag_path():
        return os.path.join(wd, f"rebuilt.frag{len(frags)}.gz")

    for c in contigs:
        w = F.IndexedBgzfWriter(frag_path())
        n_c = 0
        smp, wins = [], []
        t0 = time.perf_counter()
        for k, chunk in enumerate(F.iter_tabix_chunks(orig, c)):
            t1 = time.perf_counter()
            ch = F.parse_bed_chunk(chunk, 5)
            pos = F.parse_uint_field(ch, 1)
            end = F.parse_uint_field(ch, 2)
            tri = F.field_bytes_fixed(ch, 3, 3)
            depth = F.parse_uint_field(ch, 4)
            t2 = time.perf_counter()
            assert np.array_equal(end, pos + 1), "not 1-bp records"
            data, offs = F.format_context_lines(c, pos, tri, depth)
            assert data == chunk, f"re-formatted lines differ from the original ({c})"
            assert np.array_equal(offs, ch.line_start)
            t3 = time.perf_counter()
            win = pos >> F.WINDOW_SHIFT
            cut = None
            if c == big and nsplit == 0 and ch.n > 2:
                mid = np.flatnonzero(win[1:-1] == win[2:]) + 1     # record m continues m-1's window
                cut = int(mid[mid.size // 2]) if mid.size else None
            elif c == big and nsplit == 1 and k >= 3:
                edge = np.flatnonzero(win[1:] != win[:-1]) + 1     # record m opens a new window
                cut = int(edge[0]) if edge.size else None
            if cut is not None:
                o = int(offs[cut])
                w.write_records(data[:o], pos[:cut], offs[:cut])
                frags.append((w.path, w.close()))
                if nsplit == 0:
                    e = F.IndexedBgzfWriter(frag_path())          # an empty fragment in between
                    frags.append((e.path, e.close()))
                w = F.IndexedBgzfWriter(frag_path())
                w.write_records(data[o:], pos[cut:], offs[cut:] - o)
                nsplit += 1
            else:
                w.write_records(data, pos, offs)
            t4 = time.perf_counter()
            if nsel < 12:                           # select_lines vs naive on some real chunks
                keep = rng.random(ch.n) < [0.001, 0.5, 0.999][nsel % 3]
                sel, so = F.select_lines(ch, keep)
                lines = chunk.split(b"\n")[:-1]
                kept = [ln + b"\n" for ln, kk in zip(lines, keep) if kk]
                assert sel == b"".join(kept)
                assert so.tolist() == np.cumsum([0] + [len(x) for x in kept])[:len(kept)].tolist()
                nsel += 1
            smp.append(pos[rng.random(ch.n) < 2e-3])
            wins.append(np.unique(win))
            n_c += ch.n
            tt["stream"] += t1 - t0
            tt["parse"] += t2 - t1
            tt["fmt"] += t3 - t2
            tt["write"] += t4 - t3
            t0 = time.perf_counter()
        assert n_c == nrec[c], f"{c}: streamed {n_c} records, index says {nrec[c]}"
        frags.append((w.path, w.close()))
        sample[c] = np.concatenate(smp) if smp else np.zeros(0, np.int64)
        wins_by[c] = np.unique(np.concatenate(wins))
    assert nsplit == 2, "the largest contig was not split as planned"
    nl = sum(nrec.values())
    log(f"  {nl} lines streamed/parsed/re-formatted byte-identically and rewritten in "
        f"{len(frags)} fragments: tabix stream {tt['stream']:.1f}s, parse 4 fields "
        f"{tt['parse']:.1f}s, format {tt['fmt']:.1f}s, write {tt['write']:.1f}s")
    t0 = time.perf_counter()
    F.assemble_bgzf(new, hdr, frags, index=True)
    log(f"  assemble_bgzf + .tbi {time.perf_counter() - t0:.2f}s")
    assert not any(os.path.exists(p) for p, _ in frags)
    mo, mn = md5_gz(orig), md5_gz(new)
    assert mo == mn, "decompressed content differs"
    log(f"  decompressed content identical: {mn[1]} bytes, md5 {mn[0]}")
    co = canonical(tbi, bgzf_blocks(orig))
    cn = canonical(decode_tbi(new + ".tbi"), bgzf_blocks(new))
    for k in ("magic", "n_ref", "fmt", "cols", "meta", "skip", "n_no_coor", "names"):
        assert co[k] == cn[k], k
    for nm, (bo, lo), (bn, ln) in zip(co["names"], co["refs"], cn["refs"]):
        assert bo == bn, f"bins differ for {nm}"
        assert lo == ln, f"linear index differs for {nm}"
    log(f"  custom .tbi == original .tbi (canonical: {sum(len(b) for b, _ in cn['refs'])} bins, "
        f"{sum(len(x) for _, x in cn['refs'])} linear-index entries)")

    # regions
    regs = []
    for c in contigs:
        smp, wins = sample[c], wins_by[c]
        last = int(wins[-1] + 1) << F.WINDOW_SHIFT
        for k in rng.choice(wins, min(60, wins.size), replace=False).tolist():
            for kk in (k, k + 1):
                s = kk * W
                regs += [(c, s, s + W), (c, max(0, s - 1), s + 1), (c, s, s + 1),
                         (c, max(0, s - 1), s), (c, max(0, s - W), s)]
        gaps = np.flatnonzero(np.diff(wins) > 1)
        for g in rng.choice(gaps, min(30, gaps.size), replace=False).tolist():
            a, b = int(wins[g] + 1) * W, int(wins[g + 1]) * W
            regs += [(c, a, b), (c, a - 1, a + 1), (c, b - 1, b + 1), (c, (a + b) // 2, (a + b) // 2 + 10)]
        regs += [(c, 0, 1), (c, 0, W), (c, last - 1, last + 10**6), (c, last, 2**29)]
    n_edge = len(regs)
    for _ in range(n_random):
        c = contigs[int(rng.integers(0, len(contigs)))]
        smp = sample[c]
        last = int(wins_by[c][-1] + 1) << F.WINDOW_SHIFT
        if smp.size and rng.random() < 0.6:
            s = int(smp[int(rng.integers(0, smp.size))]) + int(rng.integers(-W, W))
        else:
            s = int(rng.integers(0, last + 50000))
        s = max(0, s)
        regs.append((c, s, s + int(2 ** rng.uniform(0, 17))))
    a, b = pysam.TabixFile(orig), pysam.TabixFile(new)
    ne = nrecs = 0
    for c, s, e in regs:
        ra, rb = list(a.fetch(c, s, e)), list(b.fetch(c, s, e))
        assert ra == rb, f"pysam query differs {c}:{s}-{e}"
        ne += bool(ra)
        nrecs += len(ra)
    nwhole = 0
    for c in contigs + ["no_such_contig"]:
        try:
            nwhole += iter_compare(a.fetch(c), b.fetch(c), f"pysam whole contig {c}")
        except ValueError:
            assert c == "no_such_contig"
    a.close()
    b.close()
    log(f"  pysam: {len(regs)} regions ({n_edge} window-edge/gap/end + {n_random} random; "
        f"{ne} non-empty, {nrecs} records) + {len(contigs)} whole contigs ({nwhole} records): identical")
    # tabix CLI (one process per query and index) and iter_tabix_chunks
    cli_regs = [r for r in regs if F._tabix_region(*r) is not None]     # drop empty [s, s)
    pick = rng.choice(len(cli_regs), min(n_cli, len(cli_regs)), replace=False)
    tf = pysam.TabixFile(new)
    ncli = 0
    for i in pick.tolist():
        c, s, e = cli_regs[i]
        r = F._tabix_region(c, s, e)
        oa = subprocess.run(["tabix", orig, r], check=True, capture_output=True).stdout
        ob = subprocess.run(["tabix", new, r], check=True, capture_output=True).stdout
        assert oa == ob, f"tabix CLI differs {r}"
        assert oa == "".join(x + "\n" for x in tf.fetch(c, s, e)).encode(), f"CLI != pysam {r}"
        if i % 10 == 0:
            assert b"".join(F.iter_tabix_chunks(new, c, s, e, chunk_bytes=1 << 12)) == oa
        ncli += 1
    tf.close()
    for c in contigs:
        ha = md5_of_chunks(F.iter_tabix_chunks(orig, c))
        hb = md5_of_chunks(F.iter_tabix_chunks(new, c))
        assert ha == hb, f"tabix CLI whole contig {c} differs"
    log(f"  tabix CLI: {ncli} regions + {len(contigs)} whole contigs (iter_tabix_chunks): identical")
    assert no_children()
    for p in (new, new + ".tbi"):
        os.remove(p)
    log(f"  REAL CHECK PASSED for {base} [{time.time() - t00:.0f}s]")


def md5_of_chunks(chunks):
    import hashlib
    h = hashlib.md5()
    n = 0
    for ch in chunks:
        h.update(ch)
        n += len(ch)
    return h.hexdigest(), n


# --------------------------------------------------------------------------------------------
# 8. benchmark (--bench N): throughput vs naive Python (run via srun -c 2)
# --------------------------------------------------------------------------------------------

def run_bench(n_lines, wd, level=None):
    if level is not None:
        F.BGZF_LEVEL = level
    rng = np.random.default_rng(0)
    contig = "NC_000085.7"
    naive_n = min(n_lines, 2_000_000)

    def runs_mask(length, mean_on, mean_off):
        m = np.zeros(length, bool)
        p = 0
        while p < length:
            on = int(rng.geometric(1 / mean_on))
            m[p:p + on] = True
            p += on + int(rng.geometric(1 / mean_off))
        return m

    t = time.perf_counter()
    L = int(n_lines / 0.8) + 10
    callable_ = runs_mask(L, 4000, 1000)
    pos = np.flatnonzero(callable_)[:n_lines].astype(np.int64)
    n = pos.size
    depth = rng.integers(1, 80, n).astype(np.uint16)
    tri = np.frombuffer(b"ACGT", np.uint8)[rng.integers(0, 4, (n, 3))]
    common = callable_ & runs_mask(L, 20000, 2000)            # a "shared" space for select
    log(f"bench: {n} synthetic context lines (contig span {L}), BGZF level "
        f"{'htslib default' if F.BGZF_LEVEL is None else F.BGZF_LEVEL} "
        f"[setup {time.perf_counter() - t:.1f}s]")

    # A. format + write (what a preprocess worker does per contig)
    frag = os.path.join(wd, "bench.frag.gz")
    final = os.path.join(wd, "bench.bed.gz")
    step = 1 << 20
    t_fmt = t_wr = 0.0
    nbytes = 0
    t0 = time.perf_counter()
    w = F.IndexedBgzfWriter(frag)
    for a in range(0, n, step):
        b = min(n, a + step)
        t1 = time.perf_counter()
        data, offs = F.format_context_lines(contig, pos[a:b], tri[a:b], depth[a:b])
        t2 = time.perf_counter()
        w.write_records(data, pos[a:b], offs)
        t3 = time.perf_counter()
        t_fmt += t2 - t1
        t_wr += t3 - t2
        nbytes += len(data)
    fi = w.close()
    t4 = time.perf_counter()
    F.assemble_bgzf(final, b"#CONTIG\tSTART\tEND\tCTX\tDEPTH\n", [(frag, fi)], index=True)
    t5 = time.perf_counter()
    log(f"A fastio: format_context_lines {n / t_fmt / 1e6:.2f} M lines/s "
        f"({nbytes / t_fmt / 1e6:.0f} MB/s); IndexedBgzfWriter.write_records (BGZF compression + "
        f"window offsets) {n / t_wr / 1e6:.2f} M lines/s; format+write {n / (t4 - t0) / 1e6:.2f} "
        f"M lines/s [{t4 - t0:.1f}s]; assemble_bgzf+.tbi {t5 - t4:.2f}s; "
        f"{nbytes / 1e6:.0f} MB text -> {os.path.getsize(final) / 1e6:.0f} MB")
    # naive: the original write_context_one inner statement, per line
    tri_s = [bytes(x).decode() for x in tri[:naive_n]]
    pl, dl = pos[:naive_n].tolist(), depth[:naive_n].tolist()
    nf = os.path.join(wd, "naive.gz")
    t0 = time.perf_counter()
    with pysam.BGZFile(nf, "wb") as bed:
        for p, triplet, d in zip(pl, tri_s, dl):
            bed.write(f"{contig}\t{p}\t{p+1}\t{triplet}\t{d}\n".encode())
    tn = time.perf_counter() - t0
    t0 = time.perf_counter()
    txt = "".join(f"{contig}\t{p}\t{p+1}\t{triplet}\t{d}\n"
                  for p, triplet, d in zip(pl, tri_s, dl)).encode()
    tn_fmt = time.perf_counter() - t0
    assert txt == F.format_context_lines(contig, pos[:naive_n], tri[:naive_n], depth[:naive_n])[0]
    log(f"A naive ({naive_n} lines): f-string + BGZFile.write per line {naive_n / tn / 1e6:.2f} M "
        f"lines/s; f-string formatting alone {naive_n / tn_fmt / 1e6:.2f} M lines/s")
    os.remove(nf)
    ref = os.path.join(wd, "bench_ref.bed.gz")
    os.link(final, ref)
    t0 = time.perf_counter()
    subprocess.run(["tabix", "-f", "-p", "bed", ref], check=True)
    log(f"A the `tabix -p bed` re-scan that the custom index replaces: {time.perf_counter() - t0:.1f}s")
    os.remove(ref + ".tbi")
    os.remove(ref)

    # B. stream + parse + select (what a somatic pass-2 task does) and sbs96-style fields
    t0 = time.perf_counter()
    nb = sum(len(ch) for ch in F.iter_tabix_chunks(final, contig))
    ts = time.perf_counter() - t0
    log(f"B iter_tabix_chunks alone (tabix CLI): {nb / ts / 1e6:.0f} MB/s, {n / ts / 1e6:.2f} M lines/s")
    t_parse = t_sel = t_tri = 0.0
    kept = 0
    first = None
    t0 = time.perf_counter()
    for ch in F.iter_tabix_chunks(final, contig):
        t1 = time.perf_counter()
        c = F.parse_bed_chunk(ch, 5)
        st = F.parse_uint_field(c, 1)
        t2 = time.perf_counter()
        data, offs = F.select_lines(c, common[st])
        t3 = time.perf_counter()
        F.field_bytes_fixed(c, 3, 3)
        F.parse_uint_field(c, 4)
        t4 = time.perf_counter()
        t_parse += t2 - t1
        t_sel += t3 - t2
        t_tri += t4 - t3
        kept += offs.size
        if first is None:
            first = ch
    tt = time.perf_counter() - t0
    log(f"B fastio: parse_bed_chunk+parse_uint_field(START) {n / t_parse / 1e6:.2f} M lines/s; "
        f"select_lines {n / t_sel / 1e6:.2f} M lines/s (kept {kept / n:.1%}); parse+select "
        f"{n / (t_parse + t_sel) / 1e6:.2f} M lines/s; TRIPLET+DEPTH fields {n / t_tri / 1e6:.2f} "
        f"M lines/s; end-to-end incl. tabix {n / tt / 1e6:.2f} M lines/s [{tt:.1f}s]")
    lines = first.split(b"\n")[:-1][:naive_n]
    t0 = time.perf_counter()
    out = []
    for ln in lines:
        f = ln.split(b"\t")
        if common[int(f[1])]:
            out.append(ln + b"\n")
    res = b"".join(out)
    tn = time.perf_counter() - t0
    c = F.parse_bed_chunk(b"".join(x + b"\n" for x in lines), 5)
    assert F.select_lines(c, common[F.parse_uint_field(c, 1)])[0] == res
    log(f"B naive (split + int + bitmap lookup + append per line, {len(lines)} lines): "
        f"{len(lines) / tn / 1e6:.2f} M lines/s")
    os.remove(final)
    os.remove(final + ".tbi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller synthetic data (login node)")
    ap.add_argument("--workdir", default=None, help="scratch dir (default: a new temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep the default temp dir")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--real", action="append", default=[], metavar="BED_GZ",
                    help="also check against a real tabix-indexed context BED (read-only; "
                         "repeatable; writes a rebuilt copy of it to --workdir)")
    ap.add_argument("--no-synthetic", action="store_true", help="only run the --real checks")
    ap.add_argument("--bench", type=int, default=0, metavar="N_LINES",
                    help="only run the throughput benchmark on N_LINES synthetic lines")
    ap.add_argument("--bench-level", type=int, default=None, help="BGZF level for --bench")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    wd = args.workdir or tempfile.mkdtemp(prefix="test_fastio_")
    os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    scale = 0.1 if args.quick else 1.0
    try:
        if args.bench:
            run_bench(args.bench, wd, args.bench_level)
            log(f"benchmark done in {time.time() - t0:.1f}s")
            return
        if not args.no_synthetic:
            test_format(rng, 20000 if args.quick else 200000)
            test_parse(rng, 3000 if args.quick else 30000)
            test_parse_uint_exhaustive(rng)
            test_bgzf_tell(rng, wd)
            test_misc(rng, wd)
            n_random = 1000 if args.quick else 1500
            log("index equivalence: custom .tbi vs `tabix -p bed`")
            hdr = (b"##INTERVALS FROM A01 SHARED ACROSS: A01 B01\n"
                   b"#CONTIG\tSTART\tEND\tCTX\tDEPTH\n")
            check_dataset(rng, wd, "edge", edge_dataset(rng, scale), hdr, n_random, 60, 40)
            check_dataset(rng, wd, "edge_nohdr", edge_dataset(rng, 0.05), b"", n_random, 20, 20)
            for i in range(2 if args.quick else 6):
                check_dataset(rng, wd, f"random{i}", random_dataset(rng, scale), hdr, n_random, 20, 20)
            check_dataset(rng, wd, "realistic",
                          realistic_dataset(rng, 200000 if args.quick else 2000000),
                          hdr, n_random, 20, 20, max_log2=17)
        for path in args.real:
            check_real(path, wd, rng, n_random=1500, n_cli=1000)
    finally:
        if not args.keep and args.workdir is None:
            shutil.rmtree(wd, ignore_errors=True)
    log(f"ALL fastio TESTS PASSED in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
