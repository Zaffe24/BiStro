"""
Vectorised text / BGZF / tabix primitives shared by preprocess, somatic and sbs96.

* BED text <-> numpy: `parse_bed_chunk` + `parse_uint_field` / `field_bytes_fixed` /
  `select_lines` read tab-separated records without a per-line Python loop;
  `format_context_lines` builds `.context.bed.gz` rows byte-identical to the original
  f-string.
* Indexed BGZF output: `IndexedBgzfWriter` writes a headerless fragment through
  pysam.BGZFile (htslib + libdeflate) and remembers the virtual offset of the first
  record of every 16-kb window; `assemble_bgzf` concatenates fragments behind a header
  (raw byte copy, no recompression) and `build_tbi` writes the matching `.tbi`, so no
  `tabix` re-scan of the finished file is needed.
* Tabix streaming: `iter_tabix_chunks` streams a region through the `tabix` CLI in large
  chunks of complete lines.

TBI layout (hts-specs "Tabix index"; written identically by `tabix -p bed` (htslib 1.21) and by
pysam.tabix_index(preset="bed"), which the original pipeline used; every rule below is asserted
by tests/test_fastio.py:verify_tabix_layout on tabix-built indexes of the synthetic test files,
on the 16 original-pipeline context / shared-context indexes of the validation subset and on
genome-wide production indexes of 2.1e9 records):
  magic "TBI\\1", n_ref, format=0x10000 (TBX_UCSC: 0-based half-open BED), col_seq=1, col_beg=2,
  col_end=3, meta='#', skip=0, l_nm, NUL-terminated names (contigs that have records, in file
  order); per ref: n_bin, bins {uint32 bin, int32 n_chunk, (uint64 beg, uint64 end)*}, n_intv,
  uint64 linear offsets; finally uint64 n_no_coor (htslib always writes it; 0 here).
For 1-bp records every record lies in the leaf bin 4681 + (pos >> 14). htslib pushes records one
at a time and closes a bin's chunk when the next record falls in another bin, so each leaf bin
holds one chunk [voff of its first record, voff of the first record of the next present window]
(the last one ends at the end of the contig's data); compress_binning() never merges them
because no parent bin exists. The pseudo-bin 37450 holds (first record voff, end voff) and
(n_mapped=n_records, n_unmapped=0). The linear index has last_window+1 entries: a present window
holds its first record's voff and every empty window holds the voff of the NEXT present window
(htslib >= 1.10 update_loff back-fills from the last entry; so leading empty windows hold the
contig's first record voff). E.g. all 10,357 empty windows of a genome-wide production index
(6,621 of them where back- and forward-fill differ) are back-filled.
Virtual offsets written here can differ numerically from tabix's (e.g. "end of block k" vs
"start of block k+1"), but they address the same uncompressed positions, so every query
returns the same records.
"""

from __future__ import annotations

import errno
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pysam

WINDOW_SHIFT = 14                       # tabix linear-index window = 16 kb

_TBI_N_LVLS = 5
_TBI_LEAF0 = ((1 << 15) - 1) // 7       # 4681: first bottom-level bin (min_shift 14, 5 levels)
_TBI_META_BIN = ((1 << 18) - 1) // 7 + 1  # 37450: pseudo-bin with per-contig metadata
_TBI_MAX_WINDOW = 1 << (3 * _TBI_N_LVLS)  # windows >= 2**15 (pos >= 2**29) cannot be in a .tbi
_TBX_UCSC = 0x10000
_BGZF_EOF_LEN = 28                      # an empty BGZF file is exactly the 28-byte EOF block

_NL = 0x0A
_TAB = 0x09
_ZERO = 48

_FORMAT_BLOCK_ROWS = 1 << 16            # rows per formatting block (keeps temporaries in cache)

# BGZF compression level used by IndexedBgzfWriter / assemble_bgzf when none is given.
# None = htslib's default (pysam mode "wb", level 6), as written by the original pipeline.
# Only compressed bytes depend on it. Measured with pysam 0.23.3/libdeflate on real BiStro
# text: level 4 compresses context rows 2.3x faster (+0.1% size) and muts rows 1.6x faster
# (+1.8% size); level 1 is ~6x faster but ~22% larger.
BGZF_LEVEL: int | None = None


def _bgzf_wmode(level: int | None) -> str:
    lvl = BGZF_LEVEL if level is None else level
    if lvl is None:
        return "wb"
    if not 0 <= int(lvl) <= 9:
        raise ValueError("BGZF compression level must be 0-9")
    return f"wb{int(lvl)}"


# --------------------------------------------------------------------------------------------
# BED text -> numpy
# --------------------------------------------------------------------------------------------

@dataclass
class BedChunk:
    """Byte layout of a block of complete, tab-separated text lines (see parse_bed_chunk)."""
    u8: np.ndarray           # np.uint8 view of the buffer (keeps the buffer alive)
    line_start: np.ndarray   # int64 [n] byte offset of each line start
    line_end: np.ndarray     # int64 [n] byte offset of each line's '\n'
    fstart: np.ndarray       # int64 [n, ncols] field start offsets
    fend: np.ndarray         # int64 [n, ncols] field end offsets (exclusive)
    n: int


def parse_bed_chunk(buf: bytes, ncols: int) -> BedChunk:
    """Locate lines and fields of `buf` (complete '\\n'-terminated lines, no header lines).

    Every line must have exactly `ncols` tab-separated fields; raises ValueError otherwise."""
    if ncols < 1:
        raise ValueError("ncols must be >= 1")
    u8 = np.frombuffer(buf, dtype=np.uint8)
    if u8.size == 0:
        e1 = np.empty(0, np.int64)
        e2 = np.empty((0, ncols), np.int64)
        return BedChunk(u8, e1, e1.copy(), e2, e2.copy(), 0)
    if u8[-1] != _NL:
        raise ValueError("chunk does not end with a complete line ('\\n')")
    # One pass for both separators ('\t' = 9, '\n' = 10); bytes 0-8 are ordinary field content.
    sep = np.flatnonzero(u8 <= _NL).astype(np.int64, copy=False)
    kind = u8[sep]
    if kind.min() < _TAB:
        ok = kind >= _TAB
        sep, kind = sep[ok], kind[ok]
    n = int(np.count_nonzero(kind == _NL))
    if sep.size != n * ncols:
        raise ValueError(f"expected {ncols} tab-separated fields per line "
                         f"({n * ncols} separators for {n} lines), found {sep.size}")
    # Row i of the (n, ncols) separator matrix ends with a '\n' for every i iff the newlines
    # are exactly its last column, i.e. every line holds exactly ncols-1 tabs.
    if not (kind.reshape(n, ncols)[:, -1] == _NL).all():
        raise ValueError(f"some line does not have exactly {ncols} tab-separated fields")
    fend = sep.reshape(n, ncols)
    fstart = np.empty((n, ncols), np.int64)
    fs = fstart.reshape(-1)
    fs[0] = 0
    np.add(sep[:-1], 1, out=fs[1:])          # a field starts right after the previous separator
    return BedChunk(u8, np.ascontiguousarray(fstart[:, 0]), np.ascontiguousarray(fend[:, -1]),
                    fstart, fend, n)


def _parse_uint_digits(u8, s, e, length, min_len, max_len, col):
    # digit-by-digit reference path (used for buffers shorter than 8 bytes)
    val = np.zeros(s.size, np.int64)
    scale = 1
    for k in range(1, max_len + 1):          # k-th digit counted from the right
        idx = e - k
        short = None
        if k > min_len:
            short = length < k
            idx[short] = s[short]            # any in-field byte; zeroed below
        dig = u8[idx] - np.uint8(_ZERO)      # uint8 wrap-around: non-digits become > 9
        if short is not None:
            dig[short] = 0
        if (dig > 9).any():
            raise ValueError(f"non-digit character in numeric column {col}")
        val += dig.astype(np.int64) * scale
        scale *= 10
    return val


_U64 = np.uint64
_ASCII0_X8 = _U64(0x3030303030303030)
_PLUS76_X8 = _U64(0x7676767676767676)
_TOP_BITS = _U64(0x8080808080808080)
# simdjson parse_eight_digits_unrolled (first char in the lowest byte); uint64 wrap-around intended
_SWAR_STEPS = ((_U64(0x0F0F0F0F0F0F0F0F), _U64(2561), _U64(8)),
               (_U64(0x00FF00FF00FF00FF), _U64(6553601), _U64(16)),
               (_U64(0x0000FFFF0000FFFF), _U64(42949672960001), _U64(32)))
# _KEEP_TOP[k] keeps the top k bytes of a little-endian uint64 (the last k chars loaded)
_KEEP_TOP = np.array([0] + [((1 << 64) - 1) ^ ((1 << (8 * (8 - k))) - 1) for k in range(1, 9)],
                     dtype=np.uint64)


def parse_uint_field(chunk: BedChunk, col: int) -> np.ndarray:
    """int64 values of column `col` (0-based); fields must be 1-18 ASCII decimal digits."""
    n = chunk.n
    if n == 0:
        return np.empty(0, np.int64)
    s = chunk.fstart[:, col]
    e = chunk.fend[:, col]
    length = e - s
    min_len = int(length.min())
    max_len = int(length.max())
    if min_len < 1:
        raise ValueError(f"empty numeric field in column {col}")
    if max_len > 18:
        raise ValueError(f"numeric field longer than 18 digits in column {col}")
    u8 = chunk.u8
    if u8.size < 8:
        return _parse_uint_digits(u8, s, e, length, min_len, max_len, col)
    # SWAR: load the 8 bytes ending at the field end as one little-endian uint64 (unaligned
    # strided view), zero the bytes before the field, and combine 8 digits with 3 multiplies
    # (simdjson's parse_eight_digits_unrolled). Fields > 8 digits take one more group per 8.
    v64 = np.ndarray((u8.size - 7,), dtype="<u8", buffer=u8, offset=0, strides=(1,))
    val = None
    for g in range((max_len + 7) // 8):
        p = e - 8 * (g + 1)                  # first byte of this group's 8-byte window
        lp = np.maximum(p, 0)
        x = v64[lp]
        if int(p.min()) < 0:                 # window starts before the buffer (fields near
            neg = p < 0                      # its start): load from 0 and shift into place
            x[neg] <<= np.minimum(lp[neg] - p[neg], 7).astype(np.uint64) * _U64(8)
        x ^= _ASCII0_X8                      # '0'..'9' -> 0..9, every other byte -> >= 10
        x &= _KEEP_TOP[np.clip(length - 8 * g, 0, 8)]
        # Every byte is <= 9 iff no byte of (x + 0x76..) | x has its top bit set: bytes <= 9
        # stay < 0x80 without carry; bytes 10..0x89 reach >= 0x80; bytes >= 0x80 are caught
        # by `| x` (a carry out of such a byte only matters for an already-invalid row).
        t = x + _PLUS76_X8
        t |= x
        t &= _TOP_BITS
        if t.any():
            raise ValueError(f"non-digit character in numeric column {col}")
        for mask, mul, shift in _SWAR_STEPS:
            x &= mask
            x *= mul
            x >>= shift
        x = x.astype(np.int64)
        val = x if g == 0 else val + x * (10 ** (8 * g))
    return val


def field_bytes_fixed(chunk: BedChunk, col: int, width: int) -> np.ndarray:
    """uint8 [n, width] copy of column `col`; raises ValueError if any field length != width."""
    if chunk.n == 0:
        return np.empty((0, width), np.uint8)
    s = chunk.fstart[:, col]
    if ((chunk.fend[:, col] - s) != width).any():
        raise ValueError(f"column {col} has fields whose length is not {width}")
    return chunk.u8[s[:, None] + np.arange(width, dtype=np.int64)]


# Below this many kept runs per line, kept runs are copied as slices (Python loop over runs);
# above it, a per-byte mask is used. Both give the same bytes.
_SELECT_RUNS_PER_LINE = 0.1


def select_lines(chunk: BedChunk, keep: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Bytes of the kept lines (verbatim, in order) and the int64 offset of each kept line in them."""
    keep = np.asarray(keep, dtype=bool)
    n = chunk.n
    if keep.shape != (n,):
        raise ValueError(f"keep has shape {keep.shape}, expected ({n},)")
    lens = chunk.line_end - chunk.line_start + 1
    kept_lens = lens[keep]
    nk = kept_lens.size
    offs = np.zeros(nk, np.int64)
    if nk == 0:
        return b"", offs
    np.cumsum(kept_lens[:-1], out=offs[1:])
    if nk == n:
        a, b = int(chunk.line_start[0]), int(chunk.line_end[-1]) + 1
        return chunk.u8[a:b].tobytes(), offs
    # maximal runs of kept lines: [line rs, line re)
    dk = np.diff(keep.view(np.int8), prepend=np.int8(0), append=np.int8(0))
    rs = np.flatnonzero(dk == 1)
    re_ = np.flatnonzero(dk == -1)
    b0 = chunk.line_start[rs]
    b1 = chunk.line_end[re_ - 1] + 1
    if rs.size <= max(1, int(_SELECT_RUNS_PER_LINE * n)):
        mv = memoryview(chunk.u8)
        data = b"".join([mv[a:b] for a, b in zip(b0.tolist(), b1.tolist())])
    else:
        # per-byte mask: +1 at each run start, -1 at each run end, prefix sum.
        # (run ends never coincide with run starts: runs are maximal.)
        lo, hi = int(chunk.line_start[0]), int(chunk.line_end[-1]) + 1
        delta = np.zeros(hi - lo + 1, np.int8)
        delta[b0 - lo] = 1
        delta[b1 - lo] = -1
        mask = np.cumsum(delta[:-1], dtype=np.int8).view(np.bool_)
        data = chunk.u8[lo:hi][mask].tobytes()
    return data, offs


# --------------------------------------------------------------------------------------------
# numpy -> context BED text
# --------------------------------------------------------------------------------------------

def _ndigits(v: int) -> int:
    return len(str(v)) if v > 0 else 1


def _as_unsigned(x: np.ndarray, name: str) -> np.ndarray:
    # Unsigned division is ~10x faster than numpy's signed floor division.
    x = np.asarray(x)
    if x.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array, got {x.dtype}")
    if x.size and x.dtype.kind == "i" and int(x.min()) < 0:
        raise ValueError(f"{name} has negative values")
    return x


def _put_uint(M: np.ndarray, mask: np.ndarray, lens: np.ndarray, c0: int, D: int,
              x: np.ndarray, d_min: int) -> bool:
    """Write x right-aligned in columns [c0, c0+D) of M; clear `mask` on leading padding.

    D = digits of x.max(), d_min = digits of x.min(): the j-th digit from the right exists in
    every row for j < d_min (x >= x.min() >= 10**j), so only columns j >= d_min can hold
    padding. Returns True iff some row may be padded."""
    q = x.copy()
    for j in range(D):                       # j-th digit from the right
        col = c0 + D - 1 - j
        q2 = q // 10
        np.add(q - q2 * 10, _ZERO, out=M[:, col], casting="unsafe")
        q = q2
        if j >= d_min:
            has = x >= 10 ** j               # x has more than j digits
            mask[:, col] = has
            lens -= ~has
    return D > d_min


def _format_block(cb: bytes, pos, tri, depth) -> tuple[bytes, np.ndarray]:
    m = pos.shape[0]
    p_min, p_max = int(pos.min()), int(pos.max())
    d_min, d_max = int(depth.min()), int(depth.max())
    wide = p_max + 1 >= (1 << 32) or d_max >= (1 << 32)
    ut = np.uint64 if wide else np.uint32
    p = pos.astype(ut)
    p1 = p + ut(1)
    d = depth.astype(ut)
    Dp, Dp1, Dd = _ndigits(p_max), _ndigits(p_max + 1), _ndigits(d_max)
    lc = len(cb)
    c_p = lc
    c_p1 = c_p + Dp + 1
    c_tri = c_p1 + Dp1 + 1
    c_d = c_tri + 4
    W = c_d + Dd + 1
    # Fixed-width rows with right-aligned numbers; `mask` drops the leading padding, so
    # M[mask] is the concatenation of the exact f-string lines (M itself if nothing is padded).
    M = np.empty((m, W), np.uint8)
    mask = np.ones((m, W), np.bool_)
    lens = np.full(m, W, np.int64)
    M[:, :lc] = np.frombuffer(cb, np.uint8)
    M[:, c_p1 - 1] = _TAB
    M[:, c_tri - 1] = _TAB
    M[:, c_tri:c_tri + 3] = tri
    M[:, c_d - 1] = _TAB
    M[:, W - 1] = _NL
    pad = _put_uint(M, mask, lens, c_p, Dp, p, _ndigits(p_min))
    pad |= _put_uint(M, mask, lens, c_p1, Dp1, p1, _ndigits(p_min + 1))
    pad |= _put_uint(M, mask, lens, c_d, Dd, d, _ndigits(d_min))
    data = M[mask].tobytes() if pad else M.tobytes()
    offs = np.empty(m, np.int64)
    offs[0] = 0
    np.cumsum(lens[:-1], out=offs[1:])
    return data, offs


def format_context_lines(contig: str, pos: np.ndarray, tri: np.ndarray,
                         depth: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Bytes of f"{contig}\\t{p}\\t{p+1}\\t{tri}\\t{d}\\n" per row, and int64 line offsets.

    pos/depth: non-negative integer arrays [n]; tri: uint8 [n, 3] (ASCII bytes, copied verbatim).
    Byte-identical to the Python f-string for every non-negative value."""
    pos = _as_unsigned(pos, "pos")
    depth = _as_unsigned(depth, "depth")
    tri = np.asarray(tri)
    n = pos.shape[0]
    if pos.ndim != 1 or depth.shape != (n,) or tri.shape != (n, 3):
        raise ValueError("pos/depth must be [n] and tri [n, 3]")
    if tri.dtype != np.uint8:
        tri = tri.astype(np.uint8)
    if n == 0:
        return b"", np.empty(0, np.int64)
    cb = contig.encode() + b"\t"
    parts = []
    offs = np.empty(n, np.int64)
    base = 0
    for a in range(0, n, _FORMAT_BLOCK_ROWS):
        b = min(n, a + _FORMAT_BLOCK_ROWS)
        data, o = _format_block(cb, pos[a:b], tri[a:b], depth[a:b])
        offs[a:b] = o + base
        base += len(data)
        parts.append(data)
    return (parts[0] if len(parts) == 1 else b"".join(parts)), offs


# --------------------------------------------------------------------------------------------
# Indexed BGZF fragments
# --------------------------------------------------------------------------------------------

@dataclass
class FragmentIndex:
    """Index summary of one contig's records in one BGZF fragment (small, picklable).

    Virtual offsets are relative to the file they were recorded in (a fragment) until
    shifted() by the fragment's compressed start in the assembled file."""
    contig: str
    n_records: int
    first_voff: int          # virtual offset of the first record
    end_voff: int            # virtual offset just past the last record
    windows: np.ndarray      # int64 16-kb window ids present (ascending, unique)
    win_voffs: np.ndarray    # uint64 virtual offset of the first record of each window

    def shifted(self, byte_offset: int) -> "FragmentIndex":
        """Same index with every virtual offset moved by `byte_offset` compressed bytes."""
        d = int(byte_offset) << 16
        return FragmentIndex(self.contig, self.n_records, self.first_voff + d, self.end_voff + d,
                             self.windows.copy(), self.win_voffs + np.uint64(d))


class IndexedBgzfWriter:
    """Headerless BGZF fragment writer (pysam.BGZFile, i.e. htslib + libdeflate) that records
    tabix window offsets of the 1-bp BED records written with write_records()."""

    def __init__(self, path: str, contig: str | None = None, level: int | None = None):
        # `contig` is optional: by default it is taken from the first record written.
        # `level`: BGZF compression level (None -> module BGZF_LEVEL -> htslib default).
        self.path = path
        self._f = pysam.BGZFile(path, _bgzf_wmode(level))
        self._contig = contig
        self._cprefix = None if contig is None else contig.encode() + b"\t"
        self._n = 0
        self._first_voff = None
        self._end_voff = None
        self._last_win = None
        self._last_pos = None
        self._wins: list[int] = []
        self._voffs: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def write_records(self, data: bytes, pos: np.ndarray, line_offsets: np.ndarray) -> None:
        """Write complete lines of ONE contig, sorted by `pos` (0-based START; 1-bp records),
        nondecreasing across calls. line_offsets[i] = byte offset of record i's line in data."""
        if self._f is None:
            raise ValueError("write_records() on a closed IndexedBgzfWriter")
        if not isinstance(data, bytes):
            data = bytes(data)               # pysam.BGZFile.write only accepts bytes
        pos = np.asarray(pos, dtype=np.int64)
        line_offsets = np.asarray(line_offsets, dtype=np.int64)
        n = pos.shape[0]
        if line_offsets.shape != (n,):
            raise ValueError("pos and line_offsets must have the same length")
        if n == 0:
            if data:
                raise ValueError("data without records")
            return
        if int(line_offsets[0]) != 0 or not data.endswith(b"\n"):
            raise ValueError("data must be complete lines starting with the first record")
        if self._cprefix is None:
            tab = data.find(b"\t")
            if tab <= 0:
                raise ValueError("cannot read the contig name from the first record")
            self._cprefix = data[:tab + 1]
            self._contig = self._cprefix[:-1].decode()
        last = int(line_offsets[-1])
        if not (data.startswith(self._cprefix)
                and data[last:last + len(self._cprefix)] == self._cprefix):
            raise ValueError(f"records of a contig other than {self._contig!r}")
        if n > 1 and (pos[1:] < pos[:-1]).any():
            raise ValueError("positions are not sorted")
        if int(pos[0]) < 0 or (self._last_pos is not None and int(pos[0]) < self._last_pos):
            raise ValueError("positions are negative or decrease across write_records() calls")
        win = pos >> WINDOW_SHIFT
        starts = np.flatnonzero(win[1:] != win[:-1]) + 1      # records opening a new window
        if self._last_win is None or int(win[0]) != self._last_win:
            starts = np.concatenate(([0], starts))
        f = self._f
        prev = 0
        # Split the write at each window's first record and read the virtual offset there.
        for i, cut in zip(starts.tolist(), line_offsets[starts].tolist()):
            if cut > prev:
                f.write(data[prev:cut])
                prev = cut
            self._wins.append(int(win[i]))
            self._voffs.append(f.tell())
        f.write(data[prev:] if prev else data)
        if self._first_voff is None:
            self._first_voff = self._voffs[0]
        self._end_voff = f.tell()
        self._n += n
        self._last_win = int(win[-1])
        self._last_pos = int(pos[-1])

    def write_raw(self, data: bytes) -> None:
        """Write unindexed bytes (e.g. muts rows); no index bookkeeping."""
        if self._f is None:
            raise ValueError("write_raw() on a closed IndexedBgzfWriter")
        if data:
            self._f.write(bytes(data))

    def close(self) -> FragmentIndex | None:
        """Close the fragment; returns its FragmentIndex (None if no indexed record was written)."""
        if self._f is not None:
            self._f.close()
            self._f = None
        if self._n == 0:
            return None
        return FragmentIndex(self._contig, self._n, self._first_voff, self._end_voff,
                             np.asarray(self._wins, dtype=np.int64),
                             np.asarray(self._voffs, dtype=np.uint64))


# --------------------------------------------------------------------------------------------
# .tbi
# --------------------------------------------------------------------------------------------

def _merge_contig_runs(entries: list[FragmentIndex]) -> list[FragmentIndex]:
    """Merge FragmentIndex entries (absolute offsets, file order) of consecutive fragments of the
    same contig; raise if a contig reappears after another one (tabix would refuse too)."""
    merged: list[FragmentIndex] = []
    seen = set()
    for fi in entries:
        if merged and merged[-1].contig == fi.contig:
            a = merged[-1]
            if fi.windows.size and a.windows.size and fi.windows[0] < a.windows[-1]:
                raise ValueError(f"fragments of {fi.contig!r} are not in position order")
            wins, voffs = fi.windows, fi.win_voffs
            if wins.size and a.windows.size and wins[0] == a.windows[-1]:
                # window continues from the previous fragment: its first record is there
                wins, voffs = wins[1:], voffs[1:]
            merged[-1] = FragmentIndex(a.contig, a.n_records + fi.n_records, a.first_voff,
                                       fi.end_voff, np.concatenate((a.windows, wins)),
                                       np.concatenate((a.win_voffs, voffs)))
        else:
            if fi.contig in seen:
                raise ValueError(f"contig {fi.contig!r} appears in non-consecutive fragments")
            seen.add(fi.contig)
            merged.append(fi)
    return merged


_BIN_DT = np.dtype([("bin", "<u4"), ("n_chunk", "<i4"), ("beg", "<u8"), ("end", "<u8")])


def build_tbi(contigs: list[tuple[str, FragmentIndex]]) -> bytes:
    """Uncompressed TBI (as written by `tabix -p bed`) for 1-bp BED records; `contigs` in file
    order, one entry per contig, with absolute virtual offsets. See the module docstring."""
    names = [c for c, _ in contigs]
    if len(set(names)) != len(names):
        raise ValueError("a contig appears twice in build_tbi input")
    nm = b"".join(c.encode() + b"\0" for c in names)
    out = [struct.pack("<4s8i", b"TBI\1", len(contigs), _TBX_UCSC, 1, 2, 3, ord("#"), 0, len(nm)),
           nm]
    for name, fi in contigs:
        if fi.contig != name:
            raise ValueError(f"FragmentIndex of {fi.contig!r} given for contig {name!r}")
        wins = np.asarray(fi.windows, dtype=np.int64)
        voffs = np.asarray(fi.win_voffs, dtype=np.uint64)
        nb = wins.size
        if nb == 0 or voffs.shape != (nb,) or fi.n_records < nb:
            raise ValueError(f"inconsistent FragmentIndex for {name!r}")
        if nb > 1 and (np.diff(wins) <= 0).any():
            raise ValueError(f"windows of {name!r} are not ascending/unique")
        if int(wins[0]) < 0 or int(wins[-1]) >= _TBI_MAX_WINDOW:
            raise ValueError(f"{name!r}: positions >= 2**29 cannot be stored in a .tbi index")
        bins = np.empty(nb, _BIN_DT)
        bins["bin"] = _TBI_LEAF0 + wins
        bins["n_chunk"] = 1
        bins["beg"] = voffs
        bins["end"][:-1] = voffs[1:]
        bins["end"][-1] = fi.end_voff
        meta = struct.pack("<Ii4Q", _TBI_META_BIN, 2, fi.first_voff, fi.end_voff, fi.n_records, 0)
        # linear index (htslib >= 1.10 update_loff): an empty window takes the offset of the
        # NEXT present window (backward fill from the last window, which is always present);
        # leading empty windows thus get the first record's offset.
        n_intv = int(wins[-1]) + 1
        lin = voffs[np.searchsorted(wins, np.arange(n_intv, dtype=np.int64), side="left")]
        out += [struct.pack("<i", nb + 1), bins.tobytes(), meta,
                struct.pack("<i", n_intv), lin.astype("<u8").tobytes()]
    out.append(struct.pack("<Q", 0))                       # n_no_coor
    return b"".join(out)


def write_tbi(tbi_path: str, contigs: list[tuple[str, FragmentIndex]]) -> None:
    """build_tbi() + BGZF-compress it to `tbi_path` (written to a temporary name, then renamed)."""
    data = build_tbi(contigs)
    tmp = _tmp_path(tbi_path)
    try:
        with pysam.BGZFile(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, tbi_path)
    except BaseException:
        _remove_quietly(tmp)
        raise


# --------------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------------

def _tmp_path(path: str) -> str:
    # Outputs are written under a temporary name in the same directory and renamed into place:
    # an existing output (possibly hard-linked elsewhere) is replaced, never truncated in place,
    # and a crash never leaves a partial file under the final name.
    return f"{path}.part{os.getpid()}"


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _copy_into(src_path: str, dst, dst_offset: int, size: int) -> None:
    """Append the `size` bytes of src_path to the open file `dst` at dst_offset (kernel copy)."""
    copied = 0
    with open(src_path, "rb") as src:
        sfd, dfd = src.fileno(), dst.fileno()
        try:
            while copied < size:
                k = os.copy_file_range(sfd, dfd, size - copied, copied, dst_offset + copied)
                if k == 0:
                    break
                copied += k
        except (OSError, AttributeError) as e:
            if isinstance(e, OSError) and e.errno not in (errno.EXDEV, errno.ENOSYS, errno.EINVAL,
                                                          errno.EOPNOTSUPP, errno.ENOTSUP,
                                                          errno.EBADF, errno.EPERM):
                raise
        if copied < size:
            # portable fallback for the remainder
            src.seek(copied)
            dst.seek(dst_offset + copied)
            while copied < size:
                buf = src.read(min(16 << 20, size - copied))
                if not buf:
                    break
                dst.write(buf)
                copied += len(buf)
            dst.flush()
    if copied != size:
        raise IOError(f"short copy of {src_path}: {copied} of {size} bytes")


def assemble_bgzf(final_path: str, header: bytes,
                  fragments: list[tuple[str, FragmentIndex | None]],
                  index: bool, remove_fragments: bool = True, level: int | None = None) -> None:
    """Write `header` as a BGZF member (may be b"") followed by the raw bytes of each fragment,
    in order (decompressed text == header + fragments' text). With index=True, also write
    final_path + ".tbi" from the fragments' FragmentIndex (shifted by each fragment's start).

    Consecutive fragments of one contig are merged in the index; a contig in non-consecutive
    fragments, data without a FragmentIndex, positions >= 2**29 or a header line not starting
    with '#' raise ValueError (before anything is written). The file and the index are each
    written under a temporary name and renamed into place (the index after the data)."""
    if isinstance(header, str):
        header = header.encode()
    sizes = [os.path.getsize(p) for p, _ in fragments]
    if index:
        if header and (not header.endswith(b"\n")
                       or any(not ln.startswith(b"#") for ln in header[:-1].split(b"\n"))):
            raise ValueError("an indexed file's header must be complete lines starting with '#'")
        for (p, fi), sz in zip(fragments, sizes):
            if fi is None and sz > _BGZF_EOF_LEN:
                raise ValueError(f"fragment {p} has data but no FragmentIndex")
            if fi is not None and fi.windows.size and int(fi.windows[-1]) >= _TBI_MAX_WINDOW:
                raise ValueError(f"{fi.contig!r}: positions >= 2**29 cannot be stored in a .tbi index")
        # fail before writing anything if the contig order cannot be indexed
        _merge_contig_runs([fi for _, fi in fragments if fi is not None])
    # Fragments without data (no index entry, <= 28 bytes: at most the BGZF EOF block; any
    # block holding data is longer) add nothing to the decompressed text and are not copied.
    copy = [(p, fi, sz) for (p, fi), sz in zip(fragments, sizes)
            if fi is not None or sz > _BGZF_EOF_LEN]
    tmp = _tmp_path(final_path)
    entries = []
    try:
        if header or not copy:
            with pysam.BGZFile(tmp, _bgzf_wmode(level)) as f:
                f.write(header)
        else:
            # No header: start directly with the first fragment so the first record sits at
            # virtual offset 0, exactly as in a tabix-built index. (htslib's whole-file iterator
            # skips its initial seek when that offset is 0; matching it keeps even that
            # behaviour identical.)
            open(tmp, "wb").close()
        # Each fragment is a complete BGZF stream ending with the 28-byte EOF block; the
        # embedded empty blocks are valid BGZF (gzip/bgzip/htslib read straight through them),
        # and the file ends with the last fragment's (or the header's) EOF block.
        with open(tmp, "r+b") as out:
            off = out.seek(0, os.SEEK_END)
            for p, fi, sz in copy:
                _copy_into(p, out, off, sz)
                if fi is not None:
                    entries.append(fi.shifted(off))
                off += sz
            out.truncate(off)
        if os.path.getsize(tmp) != off:
            raise IOError(f"assembled size mismatch for {final_path}")
        os.replace(tmp, final_path)
    except BaseException:
        _remove_quietly(tmp)
        raise
    if index:
        merged = _merge_contig_runs(entries)
        write_tbi(final_path + ".tbi", [(fi.contig, fi) for fi in merged])
    if remove_fragments:
        for p, _ in fragments:
            os.remove(p)


# --------------------------------------------------------------------------------------------
# tabix streaming
# --------------------------------------------------------------------------------------------

def _tabix_region(contig: str, start: int | None, end: int | None) -> str | None:
    """tabix CLI region for 0-based half-open [start, end); None if the interval is empty."""
    coords = None
    if start is not None or end is not None:
        s = 0 if start is None else int(start)
        if s < 0:
            raise ValueError("start must be >= 0")
        if end is not None and int(end) <= s:
            return None
        coords = f"{s + 1}-{int(end)}" if end is not None else f"{s + 1}-"
    name = contig
    if ":" in contig or contig.startswith("{"):
        # htslib region syntax: brace-quote names that contain ':'
        if "}" in contig:
            raise ValueError(f"contig name {contig!r} cannot be expressed as a tabix region")
        name = "{" + contig + "}"
    return name if coords is None else f"{name}:{coords}"


def _tabix_exe() -> str:
    exe = shutil.which("tabix")
    if exe is None:
        raise FileNotFoundError("`tabix` executable not found in PATH")
    return exe


def iter_tabix_chunks(path: str, contig: str, start: int | None = None, end: int | None = None,
                      chunk_bytes: int = 16 << 20) -> Iterator[bytes]:
    """Stream the records of a bgzipped + tabix-indexed BED that overlap 0-based half-open
    [start, end) (whole contig if both None) via the `tabix` CLI; yields chunks of complete
    lines of about `chunk_bytes`. A contig absent from the index yields nothing (tabix exit 0).
    Raises RuntimeError if tabix fails. Closing the generator early kills and reaps tabix."""
    region = _tabix_region(contig, start, end)
    if region is None:
        return
    chunk_bytes = max(1, int(chunk_bytes))   # read(0) would end the loop with tabix still blocked
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen([_tabix_exe(), path, region], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=err)
        try:
            try:   # bigger pipe => fewer context switches (Linux F_SETPIPE_SZ; best effort)
                import fcntl
                fcntl.fcntl(proc.stdout.fileno(), 1031, 1 << 20)
            except Exception:
                pass
            pending = b""
            while True:
                data = proc.stdout.read(chunk_bytes)
                if not data:
                    break
                k = data.rfind(b"\n")
                if k < 0:
                    pending += data
                    continue
                chunk = pending + data[:k + 1] if pending else data[:k + 1]
                pending = data[k + 1:]
                yield chunk
            rc = proc.wait()
            if rc != 0:
                err.seek(0)
                msg = err.read().decode(errors="replace").strip()
                raise RuntimeError(f"tabix failed (exit {rc}) on {path} {region}: {msg}")
            if pending:
                raise RuntimeError(f"tabix output for {path} {region} ends without a newline")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
            proc.wait()


def tabix_contigs(path: str) -> list[str]:
    """Contig names present in the .tbi of `path` (pysam.TabixFile(path).contigs), in index order."""
    tf = pysam.TabixFile(path)
    try:
        return list(tf.contigs)
    finally:
        tf.close()
