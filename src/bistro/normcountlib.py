#!/usr/bin/env python3
"""
normcounts.py — SBS96 count correction for BiStro

The correction reshapes the raw somatic SBS96 spectrum to remove trinucleotide
*opportunity* biases:

        normcount = count * freq_G(c) / freq_C(c)

where, for each of the 32 pyrimidine-centred trinucleotide contexts c:

    C(c) = read-weighted callable count   (each position weighted by DEPTH)
    G(c) = whole-genome trinucleotide composition of the reference
    freq_X(c) = X(c) / sum_over_32_contexts(X)          (compositional fraction)

The factor freq_G/freq_C rescales each raw count to the spectrum that would have
been observed had the CCS reads sampled every context in proportion to its share
of the reference genome: a context sequenced more deeply than its genome share
is scaled down (factor < 1), one sequenced less deeply is scaled up (factor > 1),
and a context that is genome-proportional at average depth has a factor of ~1 so
its raw count passes through essentially unchanged.

The same correction is also run with freq_G(c) swapped for the fixed GRCh38
composition (freq_G_human), giving the Human_normcounts / Human_normfrac
columns:

        Human_normcount = count * freq_G_human(c) / freq_C(c)

COSMIC signatures are defined on human-genome opportunity, so Human_normfrac is
the column to fit against them; normfrac stays on this sample's own genome.

The position-unique callable count R(c) — each callable position counted once,
regardless of depth — is still reported as the ref_callable_tri_count column for
diagnostics.

An alternative freq_G(c) is also computed and reported alongside the genome one
(the normcounts_new / normfrac_new / freq_G(c)_new columns):

        normcount_new = count * freq_G_new(c) / freq_C(c)

where freq_G_new(c) is the position-unique callable composition — R(c), each
callable position in shared.context.bed.gz counted once regardless of depth,
divided by its sum over the 32 contexts. This rescales the spectrum to the
composition of the *callable* genome rather than the whole reference. No
human-opportunity variant is produced for this method.

The freq_G/freq_C(c) and freq_G(c)_new/freq_C(c) columns report the per-context
correction factor freq_G(c)/freq_C(c) for the genome method and the alternative
method respectively. No context is masked, however large its factor; only a
context with freq_C(c) == 0 (never sampled) gets normcount 0.

Usage:
    BiStro sbs96 \
        --context   SAMPLE.shared.context.bed.gz \
        --muts      SAMPLE.shared.muts.bed.gz \
        --ref       genome.fa \
        --out       SAMPLE.normcounts.tsv
"""

import gzip
import itertools
import os
import shutil
import statistics
import struct
import sys
import multiprocessing as mp

import numpy as np
import pysam  # already a BiStro dependency; used only for random FASTA access

from . import arglib

# Streaming tabix reader shared with sharelib (fastio.iter_tabix_chunks: the
# `tabix` CLI piped in large chunks of complete lines). It is an optimisation
# only: if fastio cannot be imported, `tabix` is not on PATH, or a query through
# it fails, the same records are read through pysam instead
# (_iter_tabix_lines_pysam, the reader the original used), so results never
# depend on which reader ran.
try:
    from .fastio import iter_tabix_chunks as _fastio_iter_tabix_chunks
except Exception:                  # missing / incomplete / broken module
    _fastio_iter_tabix_chunks = None

import matplotlib
matplotlib.use("Agg")            # headless backend: render straight to a file
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# The six substitution classes, in canonical COSMIC order, with the same
# colours himut uses for its SBS96 spectrum so the two plots are comparable.
SUB_ORDER = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
SUB_COLORS = {
    "C>A": "#14BDEB",   # sky blue
    "C>G": "#000000",   # black
    "C>T": "#E32926",   # red
    "T>A": "#A6A6A6",   # grey
    "T>C": "#A1CE63",   # green
    "T>G": "#EDC8C5",   # pale pink
}


# ---------------------------------------------------------------------------
# Sequence constants and canonicalisation helpers
# ---------------------------------------------------------------------------
# SBS96 is expressed on the pyrimidine strand: the central base of every
# context is C or T. When the central base is a purine (A or G) we take the
# reverse-complement so the same physical mutation is reported on the opposite,
# pyrimidine-centred strand. These helpers enforce that convention identically
# for the genome count, the callable count, and the mutation count so that all
# three keep the same 32 keys.
PURINES = frozenset("AG")
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def revcomp(seq):
    # Reverse-complement a short DNA string; returns None if any base is not ACGT
    # (e.g. an 'N'), which lets callers skip ambiguous positions cleanly.
    try:
        return "".join(COMPLEMENT[b] for b in reversed(seq))
    except KeyError:
        return None


def canonical_tri(tri):
    # Fold a raw reference trinucleotide onto its pyrimidine-centred form.
    # Returns the canonical 3-mer, or None if it contains a non-ACGT base.
    if len(tri) != 3 or any(b not in COMPLEMENT for b in tri):
        return None
    if tri[1] in PURINES:            # central base is a purine -> flip strand
        return revcomp(tri)
    return tri                        # central base already pyrimidine (C/T)


def build_context_lists():
    # Enumerate the canonical 32 trinucleotide contexts and the 96 SBS types in
    # a fixed order so every context appears in the output even with zero counts.
    bases = ["A", "C", "G", "T"]
    tri_lst = []
    for left in bases:
        for centre in ("C", "T"):     # pyrimidine centres only
            for right in bases:
                tri_lst.append(left + centre + right)
    # For each context, the central pyrimidine can mutate to the 3 other bases.
    sbs96_lst = []
    sbs96_to_tri = {}
    for tri in tri_lst:
        centre = tri[1]
        for alt in bases:
            if alt == centre:
                continue
            sbs96 = "{}[{}>{}]{}".format(tri[0], centre, alt, tri[2])
            sbs96_lst.append(sbs96)
            sbs96_to_tri[sbs96] = tri
    return tri_lst, sbs96_lst, sbs96_to_tri


def mut_to_sbs96(ref, alt, tri):
    # Turn a single (REF, ALT, reference-triplet) call into its canonical SBS96
    # label. If REF is a purine, complement the substitution and flip the
    # context so the result is pyrimidine-centred. Returns None on any non-ACGT.
    if ref not in COMPLEMENT or alt not in COMPLEMENT or ref == alt:
        return None
    ctri = canonical_tri(tri)
    if ctri is None:
        return None
    if ref in PURINES:
        ref, alt = COMPLEMENT[ref], COMPLEMENT[alt]
    return "{}[{}>{}]{}".format(ctri[0], ref, alt, ctri[2])


# ---------------------------------------------------------------------------
# Vectorised trinucleotide coding (shared by Step 1 and Step 2)
# ---------------------------------------------------------------------------
# A 3-mer is coded in base 5, one digit per base (A,C,G,T -> 0..3, any other
# byte -> 4): code = 25*b0 + 5*b1 + b2 in [0, 124]; code _BAD_LEN (125) marks a
# TRIPLET field that is not exactly 3 bytes long. _FOLD maps every code to the
# index (build_context_lists() order) of its canonical pyrimidine-centred
# context, or to _SKIP where canonical_tri() returns None. _FOLD is generated by
# calling canonical_tri() itself on every 3-mer, so the vectorised tallies fold
# exactly like the original per-line / per-window loops.
_TRI_LST = build_context_lists()[0]
_TRI_INDEX = {tri: i for i, tri in enumerate(_TRI_LST)}
_N_TRI = len(_TRI_LST)             # 32
_SKIP = _N_TRI                     # fold target of non-ACGT / malformed 3-mers
_BAD_LEN = 125


def _build_fold_table():
    fold = np.full(_BAD_LEN + 1, _SKIP, dtype=np.uint8)
    for code in range(_BAD_LEN):
        digits = (code // 25, code // 5 % 5, code % 5)
        if 4 in digits:                # some base outside ACGT
            continue
        fold[code] = _TRI_INDEX[canonical_tri("".join("ACGT"[d] for d in digits))]
    return fold


_FOLD = _build_fold_table()

# byte -> base-5 digit. canonical_tri() accepts uppercase ACGT only, so a context
# TRIPLET with lowercase letters is skipped (_CODE_UPPER). The genome scan
# upper-cases the sequence first (on ASCII text str.upper() only maps a-z), so
# there lowercase acgt count as well (_CODE_ANYCASE).
_CODE_UPPER = np.full(256, 4, dtype=np.uint8)
_CODE_UPPER[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4, dtype=np.uint8)
_CODE_ANYCASE = _CODE_UPPER.copy()
_CODE_ANYCASE[np.frombuffer(b"acgt", dtype=np.uint8)] = np.arange(4, dtype=np.uint8)


def _add_partials(ref_callable, ccs_callable, ref_counts, ccs_counts):
    # Merge int64[32] tallies into the {tri: int} dicts as the original merged
    # its per-contig partial dicts: only contexts seen at least once are touched
    # (seen <=> ref count > 0), values stay Python ints.
    for i in np.flatnonzero(ref_counts):
        tri = _TRI_LST[i]
        ref_callable[tri] += int(ref_counts[i])
        ccs_callable[tri] += int(ccs_counts[i])
    return ref_callable, ccs_callable


# ---------------------------------------------------------------------------
# Step 1 — callable denominators R(c) and C(c) from shared.context.bed.gz
# ---------------------------------------------------------------------------
CONTEXT_RANGE_BP = 1 << 24    # positions per task when a contig is split (multiple of 16 kb)
_OPEN_END = 1 << 62           # query end of a contig's last range: beyond any indexable position
_GZIP_CHUNK = 16 << 20        # bytes per read in the no-index serial scan
_PYSAM_BATCH = 1 << 18        # lines per chunk in the pysam stand-in for fastio


def _split_context_lines(buf):
    # Locate the 4 tabs and the newline of every line in `buf` (complete
    # '\n'-terminated lines). Every line must have exactly 5 tab-separated
    # fields, as the original's 5-way tuple unpacking required (it raised
    # ValueError otherwise; so do we).
    u8 = np.frombuffer(buf, dtype=np.uint8)
    if u8.size and u8[-1] != 10:
        raise ValueError("shared.context chunk does not end with a newline")
    nl = np.flatnonzero(u8 == 10)
    tabs = np.flatnonzero(u8 == 9)
    n = nl.size
    if tabs.size != 4 * n:
        raise ValueError("shared.context line without exactly 5 tab-separated fields")
    tabs = tabs.reshape(n, 4)
    # With 4n tabs in total, every line has exactly 4 iff the k-th group of
    # 4 tabs lies between newline k-1 and newline k.
    if n and ((tabs[:, 3] > nl).any() or (tabs[1:, 0] < nl[:-1]).any()):
        raise ValueError("shared.context line without exactly 5 tab-separated fields")
    return u8, tabs, nl


def _parse_int_fields(u8, fs, fe):
    # int64 value of each field u8[fs[i]:fe[i]]. Plain ASCII-digit fields (all
    # BiStro writes) are parsed vectorised, one right-aligned digit column at a
    # time. Any other field (empty, sign, blanks, '\r', '_', > 18 digits, ...)
    # goes through Python's int() exactly like the original int(depth),
    # including raising ValueError on garbage.
    out = np.zeros(fs.size, dtype=np.int64)
    if fs.size == 0:
        return out
    width = fe - fs
    wmax = int(width.max())
    bad = (width == 0) | (width > 18)
    for k in range(wmax):
        pos = fe - (wmax - k)                  # k-th byte of a right-aligned wmax-wide field
        active = pos >= fs
        digit = u8[np.maximum(pos, 0)] - 48    # uint8: bytes below '0' wrap to > 9
        bad |= active & (digit > 9)
        out = np.where(active, out * 10 + digit, out)
    for i in np.flatnonzero(bad):              # rare: never for BiStro-written files
        out[i] = int(u8[fs[i]:fe[i]].tobytes().decode())
    return out


def tally_context_chunk(buf):
    """Tally complete shared.context lines (CONTIG START END TRIPLET DEPTH).

    buf: bytes-like of complete '\\n'-terminated data lines (no header lines).
    Returns (ref_counts, ccs_counts), each int64[32] in build_context_lists()
    tri_lst order: per canonical context the number of lines R(c) and the sum of
    DEPTH C(c). A line whose TRIPLET is not exactly 3 uppercase A/C/G/T bytes is
    skipped (canonical_tri() -> None) and its DEPTH is not parsed, as in the
    original loop; a line without exactly 5 fields raises ValueError.
    """
    u8, tabs, nl = _split_context_lines(buf)
    if nl.size == 0:
        return np.zeros(_N_TRI, dtype=np.int64), np.zeros(_N_TRI, dtype=np.int64)
    ts = tabs[:, 2] + 1                        # TRIPLET field is u8[ts:te]
    te = tabs[:, 3]
    last = u8.size - 1                         # clip: short fields may end at the buffer end
    code = (_CODE_UPPER[u8[ts]] * 25
            + _CODE_UPPER[u8[np.minimum(ts + 1, last)]] * 5
            + _CODE_UPPER[u8[np.minimum(ts + 2, last)]])
    code[te - ts != 3] = _BAD_LEN
    ctx = _FOLD[code]
    keep = ctx != _SKIP
    ctx = ctx[keep]
    depth = _parse_int_fields(u8, te[keep] + 1, nl[keep])   # DEPTH = after 4th tab .. '\n'
    ref_counts = np.bincount(ctx, minlength=_N_TRI).astype(np.int64)
    ccs_counts = np.zeros(_N_TRI, dtype=np.int64)
    np.add.at(ccs_counts, ctx, depth)          # exact int64 sum (no float weights)
    return ref_counts, ccs_counts


def _drop_lines_before(buf, start):
    # Drop the leading lines with START < start: records fetched only because
    # they overlap the left edge of the range (see _count_context_contig).
    # Returns (rest of buf, done); done = a line with START >= start was seen, so
    # later lines need no check (tabix-indexed files are sorted by START).
    u8, tabs, nl = _split_context_lines(buf)
    if nl.size == 0:
        return buf, False
    pos = _parse_int_fields(u8, tabs[:, 0] + 1, tabs[:, 1])
    keep = pos >= start
    if not keep.any():
        return b"", False
    first = int(np.argmax(keep))
    if not keep[first:].all():
        raise ValueError("shared.context is not sorted by START")
    line_start = 0 if first == 0 else int(nl[first - 1]) + 1
    return memoryview(buf)[line_start:], True


def _iter_tabix_lines_pysam(shared_context, contig, start, end):
    # Fallback reader with fastio.iter_tabix_chunks' interface: the records of
    # pysam's fetch (the query the original used) joined into byte chunks of
    # _PYSAM_BATCH complete lines. Only the line iteration is in Python; the
    # lines are tallied in bulk like fastio's chunks.
    tbx = pysam.TabixFile(shared_context)
    try:
        lines = tbx.fetch(contig, start, end)
        while True:
            batch = list(itertools.islice(lines, _PYSAM_BATCH))
            if not batch:
                break
            batch.append("")                   # -> trailing '\n'
            yield "\n".join(batch).encode()
    finally:
        tbx.close()


_fastio_failed = False        # set (per process) once a fastio query has failed


def _use_fastio():
    # fastio is used when importable, `tabix` is on PATH, and no fastio query
    # has failed in this process yet.
    return (_fastio_iter_tabix_chunks is not None and not _fastio_failed
            and shutil.which("tabix") is not None)


def _tally_query(chunks, start):
    # Tally the byte chunks of one tabix query; with start > 0 (a range of a
    # split contig, queried from start-1) first drop the leading rows with
    # START < start (see _count_context_contig).
    ref_counts = np.zeros(_N_TRI, dtype=np.int64)
    ccs_counts = np.zeros(_N_TRI, dtype=np.int64)
    trim = bool(start)                         # None or 0 -> nothing to drop
    for buf in chunks:
        if trim:
            buf, done = _drop_lines_before(buf, start)
            trim = not done
        r, c = tally_context_chunk(buf)
        ref_counts += r
        ccs_counts += c
    return ref_counts, ccs_counts


def _count_context_contig(contig, shared_context, start=None, end=None):
    # Per-task worker: tally the rows of one contig, or of the rows with START in
    # [start, end) of a split contig. Each worker streams its own tabix query,
    # so RSS stays O(one chunk). Returns int64[32] (ref, ccs) partial tallies.
    #
    # start=None: whole contig, the exact query of the original. For a range,
    # htslib returns records overlapping the query, and a zero-length record
    # (END == START) only when strictly inside it; querying [start-1, end) and
    # dropping rows with START < start therefore assigns every record the
    # whole-contig query returns to exactly one range (the one holding its
    # START). The first range starts at 0, as the whole-contig query does.
    global _fastio_failed
    qstart = start - 1 if start else start
    if _use_fastio():
        chunks = None
        try:
            chunks = _fastio_iter_tabix_chunks(shared_context, contig, qstart, end)
            return _tally_query(chunks, start)
        except Exception as exc:
            # Whatever went wrong (tabix missing/failing, fastio bug, malformed
            # rows), redo the whole task with pysam: partial tallies are
            # discarded, so nothing is counted twice, and a genuinely malformed
            # file raises the original's error from the pysam pass below.
            _fastio_failed = True
            print(f"[bistro sbs96] tabix CLI reader failed on {contig} "
                  f"({type(exc).__name__}: {exc}); retrying with pysam", file=sys.stderr)
        finally:
            try:                               # stops tabix if we bailed out early
                getattr(chunks, "close", lambda: None)()
            except Exception:                  # cleanup only; the tallies are complete
                pass
    return _tally_query(_iter_tabix_lines_pysam(shared_context, contig, qstart, end), start)


def _count_context_task(task):
    shared_context, contig, start, end = task
    return _count_context_contig(contig, shared_context, start, end)


def _tbi_spans(shared_context):
    # {contig: bound on its record STARTs} from the .tbi linear index (n_intv
    # 16-kb windows), used only to decide how to split contigs into ranges; the
    # last range of a contig is open-ended, so results never depend on it.
    # None if there is no readable .tbi (e.g. a .csi index): no splitting.
    try:
        with gzip.open(shared_context + ".tbi", "rb") as fh:
            data = fh.read()
        if data[:4] != b"TBI\x01":
            return None
        n_ref = struct.unpack_from("<i", data, 4)[0]
        l_nm = struct.unpack_from("<i", data, 32)[0]
        names = data[36:36 + l_nm].split(b"\0")[:n_ref]
        off = 36 + l_nm
        spans = {}
        for name in names:
            n_bin = struct.unpack_from("<i", data, off)[0]
            off += 4
            for _ in range(n_bin):             # bin uint32, n_chunk int32, n_chunk x 16 bytes
                off += 8 + 16 * struct.unpack_from("<i", data, off + 4)[0]
            n_intv = struct.unpack_from("<i", data, off)[0]
            off += 4 + 8 * n_intv
            spans[name.decode()] = n_intv << 14
        return spans
    except (OSError, EOFError, ValueError, struct.error, UnicodeDecodeError):
        return None


def _context_tasks(shared_context, contigs, nproc):
    # (shared_context path, contig, start, end) tasks, largest first. With more
    # than one worker, contigs longer than CONTEXT_RANGE_BP are split into ranges
    # so one big chromosome does not set the wall time.
    spans = _tbi_spans(shared_context) if nproc > 1 else None
    sized = []
    for contig in contigs:
        span = spans.get(contig) if spans else None
        if span is None or span <= CONTEXT_RANGE_BP:
            sized.append((span or 0, (shared_context, contig, None, None)))
            continue
        for s in range(0, span, CONTEXT_RANGE_BP):
            e = s + CONTEXT_RANGE_BP
            sized.append((min(e, span) - s,
                          (shared_context, contig, s, e if e < span else _OPEN_END)))
    sized.sort(key=lambda t: -t[0])            # stable: ties keep index order
    return [task for _, task in sized]


def _drop_comment_lines(buf):
    # Remove lines starting with '#' from a buffer of complete lines.
    if not (buf.startswith(b"#") or b"\n#" in buf):
        return buf
    u8 = np.frombuffer(buf, dtype=np.uint8)
    nl = np.flatnonzero(u8 == 10)
    starts = np.concatenate(([0], nl[:-1] + 1))
    keep = np.repeat(u8[starts] != ord("#"), nl - starts + 1)
    return u8[keep].tobytes()


def _iter_gzip_data_chunks(path):
    # Complete data lines of a gzip file in ~16 MB chunks, with the line
    # semantics of the original's text-mode loop (gzip.open(path, "rt")):
    # universal newlines ('\r\n' and a lone '\r' also end a line), a missing
    # final newline is supplied, lines starting with '#' are skipped.
    pending = b""
    with gzip.open(path, "rb") as fh:
        while True:
            block = fh.read(_GZIP_CHUNK)
            if not block:
                break
            data = pending + block
            hold = data.endswith(b"\r")        # may be the '\r' of a '\r\n' split across reads
            if hold:
                data = data[:-1]
            if b"\r" in data:
                data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            cut = data.rfind(b"\n") + 1
            if cut:
                yield _drop_comment_lines(data[:cut])
            pending = data[cut:] + (b"\r" if hold else b"")
    if pending:
        pending = pending.replace(b"\r", b"\n")
        if not pending.endswith(b"\n"):
            pending += b"\n"
        yield _drop_comment_lines(pending)


def _read_shared_context_serial(shared_context, ref_callable, ccs_callable):
    # Single-pass fallback used when no usable tabix index is present.
    ref_counts = np.zeros(_N_TRI, dtype=np.int64)
    ccs_counts = np.zeros(_N_TRI, dtype=np.int64)
    for buf in _iter_gzip_data_chunks(shared_context):
        r, c = tally_context_chunk(buf)
        ref_counts += r
        ccs_counts += c
    return _add_partials(ref_callable, ccs_callable, ref_counts, ccs_counts)


def read_shared_context(shared_context, tri_lst, nproc):
    # Accumulate, per canonical trinucleotide context:
    #   ref_callable[c] += 1        (position-unique: each callable base once)
    #   ccs_callable[c] += DEPTH    (read-weighted: each base by its coverage)
    #
    # The pass over shared.context.bed.gz is the heavy part (one row per callable
    # base, genome-wide), so it is split by contig (long contigs by range) and
    # the pieces are processed in parallel. BiStro tabix-indexes this file,
    # which lets each worker seek directly to its piece. Each piece is tallied
    # with numpy (tally_context_chunk), not line by line.
    ref_callable = {tri: 0 for tri in tri_lst}
    ccs_callable = {tri: 0 for tri in tri_lst}

    # Discover the contigs from the tabix index. If the index is missing or
    # unreadable, fall back to a plain serial scan of the gzip stream.
    try:
        tbx = pysam.TabixFile(shared_context)
        contigs = list(tbx.contigs)
        tbx.close()
    except (OSError, ValueError):
        contigs = []

    if not contigs:
        return _read_shared_context_serial(shared_context, ref_callable, ccs_callable)

    tasks = _context_tasks(shared_context, contigs, nproc)
    ref_counts = np.zeros(_N_TRI, dtype=np.int64)
    ccs_counts = np.zeros(_N_TRI, dtype=np.int64)
    if nproc > 1 and len(tasks) > 1:
        # Integer sums: the order in which tasks finish does not matter.
        with mp.Pool(processes=min(nproc, len(tasks))) as pool:
            for r, c in pool.imap_unordered(_count_context_task, tasks, chunksize=1):
                ref_counts += r
                ccs_counts += c
    else:
        for task in tasks:
            r, c = _count_context_task(task)
            ref_counts += r
            ccs_counts += c
    return _add_partials(ref_callable, ccs_callable, ref_counts, ccs_counts)


# ---------------------------------------------------------------------------
# Step 2 — genome composition G(c) from the reference FASTA
# ---------------------------------------------------------------------------
GENOME_RANGE_BP = 1 << 25     # windows per task (a task may hold pieces of several contigs)
_GENOME_BLOCK_BP = 1 << 22    # windows per FASTA fetch inside a task (bounds worker memory)


def _genome_range_counts(task):
    # Per-code counts (int64[126]) of the 3-base windows of a task: for each
    # piece (contig, start, end), the windows starting at i in [start, end)
    # (end <= contig length - 2), fetched block by block with a 2-base overlap
    # so every window is seen exactly once. One FASTA handle per task.
    reference, pieces, block = task
    counts = np.zeros(_BAD_LEN + 1, dtype=np.int64)
    fasta = pysam.FastaFile(reference)
    try:
        for contig, start, end in pieces:
            for s in range(start, end, block):
                e = min(s + block, end)
                seq = fasta.fetch(contig, s, e + 2).encode()
                if len(seq) != e + 2 - s:
                    raise ValueError(f"short FASTA fetch for {contig}:{s}-{e + 2} in {reference}")
                b = _CODE_ANYCASE[np.frombuffer(seq, dtype=np.uint8)]   # digits 0..4
                code = b[:-2] * 25 + b[1:-1] * 5 + b[2:]                # <= 124: fits uint8
                counts += np.bincount(code, minlength=_BAD_LEN + 1)
    finally:
        fasta.close()
    return counts


def _genome_tasks(reference, contigs, range_bp=None, block=None):
    # Split every contig's windows [0, length - 2) into pieces of at most
    # range_bp windows and pack them, largest first, into tasks of at most
    # range_bp windows (many small contigs share a task, so a FASTA with
    # thousands of scaffolds does not reopen the index per scaffold).
    range_bp = GENOME_RANGE_BP if range_bp is None else range_bp
    block = _GENOME_BLOCK_BP if block is None else block
    pieces = []
    for contig, length in contigs:
        n_win = length - 2
        for s in range(0, max(n_win, 0), range_bp):
            pieces.append((contig, s, min(s + range_bp, n_win)))
    pieces.sort(key=lambda p: p[1] - p[2])     # largest first (stable)
    tasks, cur, cur_bp = [], [], 0
    for piece in pieces:
        size = piece[2] - piece[1]
        if cur and cur_bp + size > range_bp:
            tasks.append((reference, cur, block))
            cur, cur_bp = [], 0
        cur.append(piece)
        cur_bp += size
    if cur:
        tasks.append((reference, cur, block))
    return tasks


def get_genome_tricounts(reference, tri_lst, nproc=1):
    # Count how many times each canonical trinucleotide occurs in the reference,
    # summed over every contig in the FASTA. This is the genome-wide
    # "opportunity" baseline that ref_tri_ratio compares the callable
    # composition against.
    #
    # Same windows as the original loop (i in range(len(seq) - 2) over the
    # upper-cased contig; windows with a non-ACGT base skipped), tallied with
    # numpy over fixed-size pieces of each contig, the pieces spread over
    # `nproc` worker processes; each worker holds one block of sequence at a
    # time. Purine-centred windows fold onto their pyrimidine strand through
    # _FOLD, exactly as canonical_tri() does. Integer sums, so the order in
    # which pieces finish does not matter.
    fasta = pysam.FastaFile(reference)
    try:
        contigs = list(zip(fasta.references, fasta.lengths))
    finally:
        fasta.close()

    tasks = _genome_tasks(reference, contigs)
    counts = np.zeros(_BAD_LEN + 1, dtype=np.int64)
    if nproc > 1 and len(tasks) > 1:
        with mp.Pool(processes=min(nproc, len(tasks))) as pool:
            for c in pool.imap_unordered(_genome_range_counts, tasks, chunksize=1):
                counts += c
    else:
        for task in tasks:
            counts += _genome_range_counts(task)

    folded = np.zeros(_N_TRI + 1, dtype=np.int64)
    np.add.at(folded, _FOLD, counts)           # 126 codes -> 32 contexts + skip
    return {tri: int(folded[_TRI_INDEX[tri]]) for tri in tri_lst}


# ---------------------------------------------------------------------------
# Step 3 — raw SBS96 numerator from shared.muts.bed.gz
# ---------------------------------------------------------------------------
def read_shared_muts(shared_muts, sbs96_lst):
    # Count true somatic mutations per SBS96 context. A row is somatic when its
    # ANNOTATION (column 14) is "DNM" — the de novo class that survives
    # BiStro's germline/artefact reclassification. REF and the reference triplet
    # (CTX) come straight from the row; ALT is the first ':'-subfield of
    # STRAND_1 (the fwd-strand call), falling back to STRAND_2 if empty.
    #
    # A single DNM position can appear as more than one row — once per
    # supporting read/ZMW — so counting rows would double-count a mutation
    # wherever DEPTH > 1. We instead count each (CONTIG, START, END, ALT)
    # combination at most once, regardless of how many reads support it. ALT
    # is part of the key, not just position, because two reads at the same
    # site can in principle report different alleles (sequencing noise or a
    # genuine multiallelic site); this matches the mutation-identity key
    # sharelib.py already uses for cross-sample deduplication.
    sbs96_counts = {sbs96: 0 for sbs96 in sbs96_lst}
    seen_muts = set()
    with gzip.open(shared_muts, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            # Columns: CONTIG START END ZMW CTX REF STRAND_1 STRAND_2
            #          NOB N_MIS TYPE DEPTH SHARED ANNOTATION [COUNTS]
            if len(f) < 14 or f[13] != "DNM":
                continue
            ctx, ref = f[4], f[5]
            alt = f[6].split(":")[0]              # STRAND_1 ALT
            if alt in ("", "."):                  # fall back to rev-strand call
                alt = f[7].split(":")[0]
            mut_key = (f[0], f[1], f[2], alt)     # CONTIG, START, END, ALT
            if mut_key in seen_muts:
                continue
            seen_muts.add(mut_key)
            sbs96 = mut_to_sbs96(ref, alt, ctx)
            if sbs96 is not None:
                sbs96_counts[sbs96] += 1
    return sbs96_counts


# ---------------------------------------------------------------------------
# Step 4 — frequencies and ratios
# ---------------------------------------------------------------------------
def get_trifreq(tri2count):
    # Convert absolute per-context counts into compositional fractions that sum
    # to 1 over the 32 contexts. This is what makes the ratios dimensionless
    # correction factors centred on ~1 rather than absolute rates.
    total = sum(tri2count.values())
    if total == 0:
        return {tri: 0.0 for tri in tri2count}
    return {tri: count / float(total) for tri, count in tri2count.items()}


# Human (GRCh38) whole-genome trinucleotide counts, pyrimidine-folded, generated
# once with get_human_trinuc.py from GCF_000001405.40_GRCh38.p14_genomic.fna.gz
# (3,136,815,317 counted trinucleotides). This is the fixed freq_human constant
# used to re-express the mouse spectrum on human-genome opportunity; it never
# depends on the sample, so it is hardcoded rather than recomputed per run.
HUMAN_TRINUC_COUNT = {
    "ACA": 126854213, "ACC": 72578891,  "ACG": 16496260,  "ACT": 100990627,
    "ATA": 127495608, "ATC": 83838683,  "ATG": 114451342, "ATT": 154599055,
    "CCA": 115686979, "CCC": 82550742,  "CCG": 17853807,  "CCT": 111388272,
    "CTA": 80291697,  "CTC": 106500910, "CTG": 127712228, "CTT": 125210558,
    "GCA": 90643198,  "GCC": 74544879,  "GCG": 15448975,  "GCT": 87637692,
    "GTA": 70255358,  "GTC": 59266286,  "GTG": 95195215,  "GTT": 92203097,
    "TCA": 123229713, "TCC": 97805448,  "TCG": 14374345,  "TCT": 139698895,
    "TTA": 127454269, "TTC": 125503042, "TTG": 119055336, "TTT": 239999697,
}

#Mouse (GRCm39) whole-genome trinucleotide counts, pyrimidine-folded, generated
# once with get_human_trinuc.py from GCF_000001635.27_GRCm39_genomic.fa.gz.

MOUSE_TRINUC_COUNT  = {
    "ACA": 121650020, "ACC": 62814809,  "ACG": 12036495, "ACT": 86300621,
    "ATA": 105283164, "ATC": 72060841,  "ATG": 99727719, "ATT": 114608299,
    "CCA": 97759775,  "CCC": 71314595,  "CCG": 11690957, "CCT": 96531716,
    "CTA": 72258737,  "CTC": 94079833,  "CTG": 110434398, "CTT": 112816987,
    "GCA": 74126008,  "GCC": 56103704,  "GCG": 9410763,  "GCT": 76061814,
    "GTA": 61141141,  "GTC": 55774985,  "GTG": 85619453, "GTT": 80266372,
    "TCA": 101795998, "TCC": 87063932,  "TCG": 10707178, "TCT": 130695796,
    "TTA": 99793970,  "TTC": 108347453, "TTG": 99550228, "TTT": 176793317,
}

# Assemblies whose whole-genome composition is already tabulated above. Samples
# on one of these can skip the FASTA scan entirely (--preset_genome), which is
# the slowest step of the sbs96 run; the tables are exact counts over the whole
# assembly, so the result is identical to scanning that same FASTA.
PRESET_TRINUC_COUNT = {
    "GRCh38": HUMAN_TRINUC_COUNT,
    "GRCm39": MOUSE_TRINUC_COUNT,
}

# ---------------------------------------------------------------------------
# Step 5 — assemble and write the corrected table
# ---------------------------------------------------------------------------
def apply_correction(sbs96_counts, sbs96_lst, sbs96_to_tri, ccs_freq, genome_freq):
    # normcount = count * freq_G(c) / freq_C(c), plus the fraction each corrected
    # count represents of the whole corrected spectrum (the 96 fractions sum to
    # 1). freq_G is a parameter so the same correction can be run against either
    # the sample's own genome composition or the fixed human one.
    #
    # The correction is per-trinucleotide (32 factors) and is applied uniformly
    # to all 3 substitution types that share the context.
    normcounts = {}
    for sbs96 in sbs96_lst:
        tri = sbs96_to_tri[sbs96]
        # A context with no callable coverage was never sampled: it stays 0
        # (there is no factor to rescale by).
        cf = ccs_freq[tri]
        normcounts[sbs96] = (
            sbs96_counts[sbs96] * genome_freq[tri] / cf
            if cf > 0 else 0.0
        )

    # The fraction is the coverage/opportunity-independent shape used to compare
    # mutational spectra.
    norm_total = sum(normcounts.values())
    normfracs = {
        sbs96: (nc / norm_total if norm_total > 0 else 0.0)
        for sbs96, nc in normcounts.items()
    }
    return normcounts, normfracs


def dump_normcounts(sbs96_counts, sbs96_lst, sbs96_to_tri,
                    genome, ref_callable, ccs_callable,
                    cmdline, out_file):
    # freq_C = compositional fraction of the read-weighted callable count
    # freq_G = compositional fraction of the whole-genome composition
    # Their ratio rescales each count onto the genome's own context composition,
    # undoing how over- or under-sampled the reads are at that context, so a
    # context that is genome-proportional AND at average depth gets a factor ~1
    # and passes through essentially unchanged.
    ccs_freq = get_trifreq(ccs_callable)
    genome_freq = get_trifreq(genome)

    # frac = raw counts expressed as a proportion of the total raw mutation
    # count (the 96 values sum to 1), i.e. the uncorrected counterpart of
    # normfrac.
    counts_freq = get_trifreq(sbs96_counts)

    # The same correction against the fixed GRCh38 composition instead of this
    # sample's genome. COSMIC signatures are defined on human-genome
    # opportunity, so the Human_* spectrum is the one to compare against them.
    human_freq = get_trifreq(HUMAN_TRINUC_COUNT)

    normcounts, normfracs = apply_correction(
        sbs96_counts, sbs96_lst, sbs96_to_tri, ccs_freq, genome_freq
    )
    human_normcounts, human_normfracs = apply_correction(
        sbs96_counts, sbs96_lst, sbs96_to_tri, ccs_freq, human_freq
    )

    # ---- Alternative freq_G: position-unique callable composition ----
    # freq_G(c)_new is built from R(c) — each callable position in
    # shared.context.bed.gz counted exactly once, depth ignored (ref_callable) —
    # instead of the whole-reference trinucleotide composition. It asks what the
    # spectrum would look like had the reads sampled every context in proportion
    # to how much callable sequence that context covers. No human-composition
    # variant is computed for this method.
    ref_freq = get_trifreq(ref_callable)
    new_normcounts, new_normfracs = apply_correction(
        sbs96_counts, sbs96_lst, sbs96_to_tri, ccs_freq, ref_freq
    )

    # Per-context correction factor freq_G(c)/freq_C(c) for each method, reported
    # as its own column. Undefined where the context has no read-weighted
    # coverage (freq_C == 0); written as 0.0 there, matching how those contexts
    # are already zeroed in normcounts.
    ratio_gc = {
        tri: (genome_freq[tri] / ccs_freq[tri] if ccs_freq[tri] > 0 else 0.0)
        for tri in genome
    }
    ratio_gc_new = {
        tri: (ref_freq[tri] / ccs_freq[tri] if ccs_freq[tri] > 0 else 0.0)
        for tri in genome
    }

    # Write the table now that the fraction denominators are known.
    with open(out_file, "w") as o:
        o.write(cmdline + "\n")
        o.write("\t".join([
            "sub", "tri", "sbs96", "counts", "frac", "normcounts", "normfrac",
            "normcounts_new", "normfrac_new",
            "freq_C(c)", "freq_G(c)", "freq_G(c)_new",
            "freq_G/freq_C(c)", "freq_G(c)_new/freq_C(c)",
            "genome_tri_count", "ref_callable_tri_count", "duplex_callable_tri_count",
            "Human_normcounts", "Human_normfrac", "freq_G_human", "human_tri_count",
        ]) + "\n")

        for sbs96 in sbs96_lst:
            tri = sbs96_to_tri[sbs96]
            # sub is the 6-class substitution label, e.g. "C>T"
            sub = "{}>{}".format(sbs96[2], sbs96[4])
            o.write("\t".join(str(x) for x in [
                sub, tri, sbs96, sbs96_counts[sbs96], counts_freq[sbs96], normcounts[sbs96], normfracs[sbs96],
                new_normcounts[sbs96], new_normfracs[sbs96],
                ccs_freq[tri], genome_freq[tri], ref_freq[tri],
                ratio_gc[tri], ratio_gc_new[tri],
                genome[tri], ref_callable[tri], ccs_callable[tri],
                human_normcounts[sbs96], human_normfracs[sbs96],
                human_freq[tri], HUMAN_TRINUC_COUNT[tri],
            ]) + "\n")
    # Hand back both the corrected spectrum and its fractions so the caller can
    # plot without re-parsing the TSV.
    return normcounts, normfracs


# ---------------------------------------------------------------------------
# Step 6 — plot the corrected SBS96 spectrum (matplotlib port of himut's plot)
# ---------------------------------------------------------------------------
def dump_norm_sbs96_plt(normfracs, sbs96_lst, sbs96_to_tri, sample, out_plot):
    # Render the corrected spectrum in the canonical COSMIC SBS96 style: a single
    # continuous axis of 96 bars laid out in six substitution blocks, a coloured
    # header strip per block, and full trinucleotide labels coloured per block.
    # We plot the fraction of normalised mutations (normfrac) so the y-axis is a
    # "Mutation fraction" in [0, ~], directly comparable to a COSMIC signature.
    from matplotlib.transforms import blended_transform_factory

    # Concatenate the 96 keys in block order (C>A, C>G, ... T>G) and, within each
    # block, in the by-context order already baked into sbs96_lst (5' then 3').
    ordered, colors, group_ranges = [], [], []
    for sub in SUB_ORDER:
        keys = [s for s in sbs96_lst if "{}>{}".format(s[2], s[4]) == sub]
        start = len(ordered)
        ordered.extend(keys)
        colors.extend([SUB_COLORS[sub]] * len(keys))
        group_ranges.append((sub, start, len(ordered) - 1))

    heights = [normfracs[k] for k in ordered]
    tris = [sbs96_to_tri[k] for k in ordered]   # full triplet, e.g. "ACA"
    n = len(ordered)
    x = list(range(n))

    # Two stacked axes sharing x: a thin colour strip on top, the bars below.
    fig, (ax_top, ax) = plt.subplots(
        2, 1, figsize=(20, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 22], "hspace": 0.06},
    )

    # ---- top colour strip with the substitution-class labels above it ----
    ax_top.set_xlim(-0.5, n - 0.5)
    ax_top.set_ylim(0, 1)
    # Blended transform: x in data units (bar centres), y in axes fraction, so
    # each label sits a fixed height above its strip regardless of data scale.
    trans = blended_transform_factory(ax_top.transData, ax_top.transAxes)
    for sub, start, end in group_ranges:
        ax_top.axvspan(start - 0.5, end + 0.5, color=SUB_COLORS[sub])
        ax_top.text((start + end) / 2.0, 1.5, sub, ha="center", va="bottom",
                    fontsize=12, transform=trans, clip_on=False)
    ax_top.axis("off")

    # ---- 96 bars on one continuous axis ----
    ax.bar(x, heights, color=colors, width=0.6)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tris, rotation=90, family="monospace", fontsize=8)
    # Colour every tick label to match its substitution block.
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)
    ax.tick_params(axis="x", length=0)          # hide x tick marks, keep labels
    ax.set_ylabel("Mutation fraction", fontsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Sample name inside the panel, top-left (like the reference figure).
    ax.text(0.01, 0.93, sample, transform=ax.transAxes, fontsize=13, va="top")

    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.13)
    fig.savefig(f"{out_plot}.pdf", dpi=200)
    fig.savefig(f"{out_plot}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(shared_context, shared_muts, reference, out_file,
         out_plot, sample, no_plot, nproc, preset_genome):
    # Record the exact invocation in the output header for reproducibility.
    cmdline = arglib.get_cmdline("##bistro_normcounts_command=")

    tri_lst, sbs96_lst, sbs96_to_tri = build_context_lists()

    # Step 1: callable denominators R(c), C(c), parallelised over contigs.
    ref_callable, ccs_callable = read_shared_context(shared_context, tri_lst, nproc)
    # Step 2: genome composition G(c). For an assembly we already have tabulated
    # this is a lookup; otherwise it is a full sliding-window scan of the FASTA.
    if preset_genome is not None:
        genome = dict(PRESET_TRINUC_COUNT[preset_genome])
        print("using hardcoded {} genome composition".format(preset_genome))
    else:
        genome = get_genome_tricounts(reference, tri_lst, nproc)
    # Step 3: raw somatic SBS96 numerator.
    sbs96_counts = read_shared_muts(shared_muts, sbs96_lst)
    # Steps 4-5: frequencies, correction, and write; keep the corrected spectrum
    # and its fractions.
    normcounts, normfracs = dump_normcounts(sbs96_counts, sbs96_lst, sbs96_to_tri,
                                            genome, ref_callable, ccs_callable,
                                            cmdline, out_file)
    print("wrote {}".format(out_file))

    # Step 6: SBS96 spectrum plot (unless suppressed).
    if not no_plot:
        # Image path base: use the caller-supplied --out-plot when given,
        # otherwise default to the TSV path with its extension stripped
        # (dump_norm_sbs96_plt appends .pdf/.png).
        base = out_plot if out_plot is not None else os.path.splitext(out_file)[0]
        # Default the sample label to the output basename.
        sample = sample or os.path.basename(out_file).split(".")[0]
        dump_norm_sbs96_plt(normfracs, sbs96_lst, sbs96_to_tri, sample, base)
        print("saved {} to image".format(base))
