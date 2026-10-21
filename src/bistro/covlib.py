"""
covlib — raw-read coverage and the excessive-coverage / low-complexity mask.

For each contig the maximum-depth threshold is ``round(m + SIGMA*sqrt(m), 2)``
(reportlib.get_maximum_coverage_per_contig) where ``m`` is the contig's
*raw-read* mean depth at MAPQ >= COV_MAPQ. Reference intervals whose raw depth
reaches that threshold are treated as collapsed-repeat / mis-mapping artefacts
and, together with the optional low-complexity-region (LCR) BED, are removed
from ``preprocess``'s ``.muts.bed.gz`` / ``.context.bed.gz``.

Historically the raw depth came from a separate ``mosdepth -x -Q 20`` pass over
the BAM, and the mask was applied with ``bedtools merge`` + ``bedtools subtract
-A``. The functions in the first half of this module reproduce those results
in-process, from the reads the per-contig worker already fetches:

* ``RawCoverage``            — per-read accumulator (call ``add`` for EVERY
                               fetched read, before BiStro's own filters).
* ``contig_raw_coverage``    — (mosdepth mean | None, cap | None, high-cov intervals).
* ``mosdepth_mean`` / ``covered_bases`` / ``high_coverage_intervals``.
* ``parse_lcr_bed``          — LCR BED -> per-contig int64 (starts, ends).
* ``mask_bool``              — per-contig bool mask of LCR ∪ high-cov, clipped
                               to the contig (``merge_intervals`` / ``in_intervals``:
                               the same set as sorted disjoint intervals).
* ``write_highcov_bed`` / ``bam_header_order`` — the ``{sample}.highcov.bed`` artefact.

Intended use (preprocess)::

    lcr = covlib.parse_lcr_bed(subtract_bed, contigs)      # parent, once; ship lcr[c] to c's worker
    # worker for contig c (length L from the BAM header == FASTA length):
    cov = covlib.RawCoverage()
    for read in bam.fetch(c, 0, L):
        cov.add(read)                                       # first statement of the loop
        ...                                                 # BiStro's filters / walk
    raw_mean, cap, hc_s, hc_e = covlib.contig_raw_coverage(cov, L, min_depth)
    mask = covlib.mask_bool(L, lcr[c], (hc_s, hc_e))       # row at position p survives iff not mask[p]
    # parent: raw_means[c] = raw_mean, max_depth[c] = cap only when raw_mean is not None;
    covlib.write_highcov_bed(out_dir, sample,
                             [(c, hc_s, hc_e) for c in covlib.bam_header_order(bam, contigs)])

mosdepth 0.3.10 semantics reproduced here (verified empirically, see
tests/test_covlib.py):

* a read counts iff ``flag & 1796 == 0`` (not unmapped / secondary / QC-fail /
  duplicate; supplementary and paired flags are irrelevant) and
  ``MAPQ >= COV_MAPQ`` (255 passes);
* fast mode (``-x``): the read covers ``[pos, bam_endpos)`` with no CIGAR
  splitting (D/N spans are covered); ``bam_endpos`` is ``pos + 1`` when the
  read has no CIGAR or its CIGAR consumes no reference (pysam reports
  ``reference_end=None`` in the former case);
* coverage past the contig end is clipped to the contig length (BAM header);
* a contig is listed in the summary / per-base output iff its index query
  returned at least one record (filtered or not); a contig with no record at
  all is omitted — then the original had no RAW_COVERAGE / MAX_DEPTH and no
  high-coverage intervals for it;
* summary mean = bases / length printed with ``%.2f`` (correctly rounded, the
  same as Python's ``format(x, ".2f")``).
"""

import gzip
import os
from array import array

import numpy as np

from . import reportlib, utilib

COV_MAPQ = 20   # mapping-quality floor for the coverage scan
SIGMA = 4       # threshold = mean + SIGMA * sqrt(mean)

# mosdepth's default exclude flag (-F 1796): UNMAP | SECONDARY | QCFAIL | DUP.
MOSDEPTH_EXCLUDE_FLAGS = 0x4 | 0x100 | 0x200 | 0x400

_EMPTY = np.zeros(0, dtype=np.int64)


def _empty_pair():
    return _EMPTY.copy(), _EMPTY.copy()


# ---------------------------------------------------------------------------
# Raw-read coverage (replaces `mosdepth -x -Q COV_MAPQ`)
# ---------------------------------------------------------------------------

class RawCoverage:
    """Collects ``[start, end)`` of the reads mosdepth ``-x -Q COV_MAPQ`` counts.

    Feed it every read returned by ``bam.fetch(contig, 0, contig_length)``
    (i.e. before any BiStro filter): ``n_seen`` must count ALL records, because
    mosdepth lists a contig iff its query returned any record.
    Cheap: two ``array('q')`` appends per qualifying read.
    """

    __slots__ = ("starts", "ends", "n_seen")

    def __init__(self):
        self.starts = array("q")
        self.ends = array("q")
        self.n_seen = 0

    def add(self, read):
        self.n_seen += 1
        if read.flag & MOSDEPTH_EXCLUDE_FLAGS or read.mapping_quality < COV_MAPQ:
            return
        start = read.reference_start
        end = read.reference_end   # == htslib bam_endpos, or None without CIGAR
        self.starts.append(start)
        self.ends.append(start + 1 if end is None else end)  # bam_endpos: pos + 1

    def arrays(self):
        """``(starts, ends)`` as int64 numpy arrays (copies, read order)."""
        return (np.array(self.starts, dtype=np.int64),
                np.array(self.ends, dtype=np.int64))

    def __len__(self):
        return len(self.starts)


def covered_bases(starts, ends, length):
    """mosdepth's summary ``bases``: Σ per-base depth over ``[0, length)``."""
    starts = np.minimum(np.asarray(starts, dtype=np.int64), length)
    ends = np.minimum(np.asarray(ends, dtype=np.int64), length)
    return int((ends - starts).sum())


def mosdepth_mean(bases, length):
    """Contig mean exactly as ``parse_contig_means`` read it back from mosdepth's
    summary: ``float("%.2f" % (bases / length))``."""
    return float(f"{bases / length:.2f}")


def high_coverage_intervals(starts, ends, cap, length=None):
    """Maximal runs of raw depth >= ``cap`` as int64 (starts, ends).

    Equals the original ``high_coverage_bed`` (merge of consecutive mosdepth
    per-base rows with ``float(depth) >= cap``) for one contig that mosdepth
    lists. Event sweep over the sorted read starts/ends: depth is constant on
    ``[pts[i], pts[i+1])`` and equals #starts <= pts[i] minus #ends <= pts[i].
    No contig-length array is built.

    ``length`` (contig length from the BAM header) clips reads that run past
    the contig end, as mosdepth does; pass it whenever it is known (always, in
    the caller). It is also required for ``cap <= 0`` (only reachable with a
    negative --min_depth), where every per-base row qualifies: the result is
    then the whole contig. With ``length=None`` reads are assumed to lie
    within the contig.
    """
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if length is not None:
        starts = np.minimum(starts, length)
        ends = np.minimum(ends, length)
    starts = np.sort(starts)
    ends = np.sort(ends)
    if cap <= 0:
        # every per-base row (depth >= 0) qualifies -> the whole contig
        if length is None:
            raise ValueError("high_coverage_intervals: cap <= 0 needs the contig length")
        return (np.array([0], dtype=np.int64), np.array([length], dtype=np.int64)) \
            if length > 0 else _empty_pair()
    if starts.size == 0:
        return _empty_pair()
    pts = np.union1d(starts, ends)
    depth = np.searchsorted(starts, pts, side="right") - np.searchsorted(ends, pts, side="right")
    hi = depth >= cap          # same test as `not float(depth) < cap` for int depths
    # depth at pts[-1] (= the largest end) is 0 < cap, so every run is closed.
    edge = np.diff(hi.astype(np.int8), prepend=np.int8(0))
    return pts[edge == 1], pts[edge == -1]


def contig_raw_coverage(cov, length, min_depth):
    """Per-contig replacement of mosdepth + parse_contig_means + the cap + high_coverage_bed.

    ``cov``: RawCoverage fed with every read fetched for the contig; ``length``:
    contig length from the BAM header (what mosdepth uses).
    Returns ``(raw_mean, cap, hc_starts, hc_ends)``; ``raw_mean`` and ``cap`` are
    ``None`` when mosdepth would not have listed the contig (no record at all):
    leave such contigs OUT of the raw_means / max_depth dicts.
    ``cap`` is ``reportlib.get_maximum_coverage_per_contig``'s value (np.float64,
    or int ``min_depth + 1``), so it prints exactly as before.
    """
    if cov.n_seen == 0:
        return None, None, *_empty_pair()
    starts, ends = cov.arrays()
    mean = mosdepth_mean(covered_bases(starts, ends, length), length)
    cap = reportlib.get_maximum_coverage_per_contig({0: mean}, min_depth=min_depth)[0]
    hc_starts, hc_ends = high_coverage_intervals(starts, ends, cap, length)
    return mean, cap, hc_starts, hc_ends


# ---------------------------------------------------------------------------
# Low-complexity-region BED (replaces build_mask's zcat | filter | sort)
# ---------------------------------------------------------------------------

def _open_maybe_gzip(path):
    """Like ``zcat -f``: gzip (incl. BGZF / multi-member) or plain bytes."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    return gzip.open(path, "rb") if magic == b"\x1f\x8b" else open(path, "rb")


def _header_lines(u8, line_starts, line_lens):
    """Lines starting with '#', 'track' or 'browser' (the build_mask skip list)."""
    first = u8[np.minimum(line_starts, len(u8) - 1)]
    hit = (first == ord("#")) & (line_lens > 0)
    for prefix in (b"track", b"browser"):
        cand = np.flatnonzero((first == prefix[0]) & (line_lens >= len(prefix)))
        for j in range(1, len(prefix)):
            cand = cand[u8[line_starts[cand] + j] == prefix[j]]
        hit[cand] = True
    return hit


def _parse_uint_fields(data, u8, fstart, fend):
    """int(field) for each byte range; vectorised for plain decimal fields,
    Python ``int()`` (which raises on garbage) for anything else."""
    n = fstart.size
    flen = fend - fstart
    val = np.zeros(n, dtype=np.int64)
    good = (flen >= 1) & (flen <= 18)
    width = int(flen[good].max()) if good.any() else 0
    last = len(u8) - 1
    for j in range(width):
        active = good & (j < flen)
        digit = u8[np.minimum(fstart + j, last)].astype(np.int64) - 48
        good &= ~active | ((digit >= 0) & (digit <= 9))
        active &= good
        val = np.where(active, val * 10 + digit, val)
    for i in np.flatnonzero(~good).tolist():
        val[i] = int(data[fstart[i]:fend[i]].decode())
    return val


def _parse_lcr_block(data, names, maxlen, acc):
    """Parse complete lines of ``data`` (bytes); append matching records to ``acc``."""
    if not data:
        return
    # build_mask read `zcat` through a text-mode pipe: universal newlines
    # ('\r\n' and lone '\r' end a line too).
    if b"\r" in data:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    u8 = np.frombuffer(data, dtype=np.uint8)
    line_end = np.flatnonzero(u8 == 10)
    if not data.endswith(b"\n"):                      # last line without '\n'
        line_end = np.append(line_end, len(data))
    line_start = np.empty_like(line_end)
    line_start[0] = 0
    line_start[1:] = line_end[:-1] + 1
    line_len = line_end - line_start

    tabs = np.flatnonzero(u8 == 9)
    k0 = np.searchsorted(tabs, line_start)            # first tab of the line
    ntab = np.searchsorted(tabs, line_end) - k0
    # skip '#'/'track'/'browser' lines; need >= 3 tab-separated fields
    ok = (ntab >= 2) & ~_header_lines(u8, line_start, line_len)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return
    s0 = line_start[idx]
    t1 = tabs[k0[idx]]
    t2 = tabs[k0[idx] + 1]
    f3_end = np.where(ntab[idx] >= 3, tabs[np.minimum(k0[idx] + 2, tabs.size - 1)], line_end[idx])

    # Contig of each line: group consecutive lines with identical first field
    # (compared on the first maxlen+1 bytes; longer names can never match) and
    # look the name up once per run (runs ~ contig switches in a sorted BED).
    width = maxlen + 1
    name_len = np.minimum(t1 - s0, width)
    m = idx.size
    brk = np.zeros(m, dtype=bool)
    brk[0] = True
    brk[1:] = name_len[1:] != name_len[:-1]
    last = len(u8) - 1
    for j in range(width):
        col = np.where(j < name_len, u8[np.minimum(s0 + j, last)], 0)
        brk[1:] |= col[1:] != col[:-1]
    run_start = np.flatnonzero(brk)
    run_end = np.append(run_start[1:], m)
    for a, b in zip(run_start.tolist(), run_end.tolist()):
        contig = names.get(data[s0[a]:t1[a]])
        if contig is None:
            continue
        st = _parse_uint_fields(data, u8, t1[a:b] + 1, t2[a:b])
        en = _parse_uint_fields(data, u8, t2[a:b] + 1, f3_end[a:b])   # int(f[2].rstrip())
        acc[contig][0].append(st)
        acc[contig][1].append(en)


def parse_lcr_bed(path, contigs, chunk_bytes=4 << 20):
    """``{contig: (starts, ends)}`` (int64, file order) for every contig in ``contigs``.

    Same records build_mask fed to ``sort | bedtools merge``: lines starting with
    '#', 'track' or 'browser' are skipped, lines need >= 3 tab-separated fields,
    first field must be an analysed contig; start = int(f[1]), end =
    int(f[2].rstrip()). gzip/BGZF or plain text, parsed with numpy in chunks of
    ``chunk_bytes`` (small chunks are markedly faster: fewer page faults on the
    temporaries). A missing or empty file yields
    empty arrays (build_mask silently ignored it). Records bedtools would reject
    (start > end, negative coordinates) raise ValueError, as the original
    pipeline failed there too. Zero-length records are returned as is (see
    merge_intervals for how bedtools applied them).
    """
    acc = {c: ([], []) for c in contigs}
    if path and os.path.isfile(path) and os.path.getsize(path) > 0:
        names = {c.encode(): c for c in acc}
        maxlen = max((len(k) for k in names), default=0)
        with _open_maybe_gzip(path) as fh:
            carry = b""
            while True:
                block = fh.read(chunk_bytes)
                if not block:
                    _parse_lcr_block(carry, names, maxlen, acc)
                    break
                data = carry + block
                cut = data.rfind(b"\n") + 1
                carry = data[cut:]
                _parse_lcr_block(data[:cut], names, maxlen, acc)
    out = {}
    for c, (st, en) in acc.items():
        st = np.concatenate(st) if st else _EMPTY.copy()
        en = np.concatenate(en) if en else _EMPTY.copy()
        bad = (st < 0) | (en < 0) | (st > en)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(f"{path}: invalid BED record {c}\t{st[i]}\t{en[i]} "
                             f"(bedtools would reject it)")
        out[c] = (st, en)
    return out


# ---------------------------------------------------------------------------
# Mask = LCR ∪ high-coverage (replaces bedtools merge + subtract -A)
# ---------------------------------------------------------------------------

_POW10 = np.array([10 ** k for k in range(20)], dtype=np.uint64)


def _sort_like_gnu_sort(s, e):
    """Order of ``sort -k1,1 -k2,2n`` on the 'contig\\tstart\\tend' lines of ONE
    contig: numeric start, ties broken by the last-resort whole-line comparison,
    i.e. the END field compared as a digit string ('129' < '99', '12' < '120').
    (Digit strings collate the same in the C and en_US.UTF-8 locales.)"""
    ue = np.maximum(e, 0).astype(np.uint64)
    ndig = np.maximum(np.searchsorted(_POW10, ue, side="right"), 1)
    left_aligned = ue * _POW10[19 - ndig]           # < 10**19 < 2**64
    return np.lexsort((ndig, left_aligned, s))


def merge_intervals(*interval_pairs, length=None):
    """Union of several ``(starts, ends)`` int arrays -> sorted, disjoint int64
    ``(starts, ends)``, clipped to ``[0, length)`` when ``length`` is given.

    Position p is in the result iff the original ``build_mask`` (``sort -k1,1
    -k2,2n | bedtools merge``) + ``bedtools subtract -A`` removed the 1-bp
    record ``[p, p+1)``. For ordinary intervals that is plain union membership
    (overlapping / book-ended intervals merge). Zero-length records ``[s, s)``
    are what make it subtle (verified with bedtools 2.31.1): bedtools reads them
    as ``[s-1, s+1)``, but ``merge`` starts a merged block at the first record's
    start in sort order, so an ``s-1`` that falls before the start of a block
    begun by an earlier-sorted record with the same start is dropped. Both the
    GNU sort tie order and that sequential rule are reproduced below.
    (For a zero-length record at 0 the original pipeline broke — bedtools
    subtract errored inside a pipe whose status was not checked; here position 0
    is simply masked.)
    """
    starts = [np.asarray(s, dtype=np.int64) for s, _ in interval_pairs]
    ends = [np.asarray(e, dtype=np.int64) for _, e in interval_pairs]
    s = np.concatenate(starts) if starts else _EMPTY.copy()
    e = np.concatenate(ends) if ends else _EMPTY.copy()
    keep = s <= e
    s, e = s[keep], e[keep]
    if s.size == 0:
        return _empty_pair()
    zero = s == e
    if zero.any():
        order = _sort_like_gnu_sort(s, e)
        s, e, zero = s[order], e[order], zero[order]
        s = np.where(zero, s - 1, s)
        e = np.where(zero, e + 1, e)
    else:                                   # tie order is irrelevant without zero-length records
        order = np.argsort(s, kind="stable")
        s, e = s[order], e[order]
    # bedtools merge, sequentially: a record joins the current block iff its
    # (adjusted) start <= the block's running max end; a block spans
    # [start of its FIRST record, running max end).
    reach = np.maximum.accumulate(e)
    new = np.empty(s.size, dtype=bool)
    new[0] = True
    new[1:] = s[1:] > reach[:-1]
    first = np.flatnonzero(new)
    last = np.append(first[1:] - 1, s.size - 1)
    ms, me = np.maximum(s[first], 0), reach[last]
    if length is not None:
        me = np.minimum(me, length)
    keep = ms < me
    return ms[keep], me[keep]


def mask_bool(length, *interval_pairs):
    """Boolean mask of ``[0, length)``: True where any interval covers the
    position (same semantics as merge_intervals). One byte per base."""
    ms, me = merge_intervals(*interval_pairs, length=length)
    bounds = np.empty(2 * ms.size + 2, dtype=np.int64)
    bounds[0] = 0
    bounds[1:-1:2] = ms
    bounds[2:-1:2] = me
    bounds[-1] = length
    vals = np.zeros(2 * ms.size + 1, dtype=bool)
    vals[1::2] = True
    return np.repeat(vals, np.diff(bounds))


def in_intervals(positions, merged_starts, merged_ends):
    """For sorted disjoint intervals (merge_intervals output): bool array, True
    where the position lies in an interval. Memory O(len(positions))."""
    pos = np.asarray(positions, dtype=np.int64)
    i = np.searchsorted(merged_starts, pos, side="right") - 1
    hit = i >= 0
    hit[hit] = pos[hit] < merged_ends[i[hit]]
    return hit


# ---------------------------------------------------------------------------
# {sample}.highcov.bed
# ---------------------------------------------------------------------------

def bam_header_order(bam, contigs):
    """``contigs`` sorted by their @SQ index in the BAM header.

    The original highcov.bed listed contigs in mosdepth's (= BAM header) order;
    this equals the FASTA order of the analysed contigs whenever the BAM was
    aligned to the same reference.
    """
    import pysam
    with pysam.AlignmentFile(bam, "rb") as fh:
        return sorted(contigs, key=fh.get_tid)


def write_highcov_bed(out_dir, sample, per_contig):
    """Write ``{out_dir}/{sample}.highcov.bed`` from ``[(contig, starts, ends), ...]``
    (in output order). Returns the path, or None (and writes nothing) when
    there is no interval at all — as the original high_coverage_bed did."""
    lines = []
    for contig, starts, ends in per_contig:
        lines.extend(f"{contig}\t{s}\t{e}\n"
                     for s, e in zip(np.asarray(starts).tolist(), np.asarray(ends).tolist()))
    if not lines:
        return None
    out = os.path.join(out_dir, f"{sample}.highcov.bed")
    with open(out, "w") as dst:
        dst.write("".join(lines))
    return out
