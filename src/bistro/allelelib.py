"""Per-site A/C/G/T read counts from a coordinate-sorted, indexed BAM.

Replaces the original ``sharelib.count_alleles`` (one ``pysam.pileup`` call per site) with
one ``fetch`` per cluster of nearby sites and a direct reference->query CIGAR mapping per read.
Results are identical to the original for every site; see EQUIVALENCE below.

ORIGINAL (bistro <= efficiency refactor), per 1-based site ``pos``::

    bam.pileup(contig, pos - 1, pos, truncate=True, min_base_quality=0,
               min_mapping_quality=0, max_depth=1_000_000)
    for read in column.pileups:
        skip is_del / is_refskip; skip secondary / supplementary / duplicate;
        skip mapq < min_mapq; skip query_qualities is None or qual[qpos] < min_baseq;
        count query_sequence[qpos].upper() if it is one of A/C/G/T.

EQUIVALENCE (pysam 0.23.3 + bundled htslib 1.21, verified by reading
``pysam/libcalignmentfile.pyx``, ``pysam/libcalignedsegment.pyx`` and ``htslib/sam.c``):

R1  Read set.  ``pileup(contig, p, p+1)`` builds an ``IteratorColumnRegion`` whose row iterator is
    ``sam_itr_queryi(tid, p, p+1)``: reads with ``pos <= p < bam_endpos`` (``bam_endpos`` = pos +
    reference length, or pos + 1 when that length is 0).  ``fetch(contig, a, b)`` uses the same
    query, so fetching a cluster span yields a superset of every member site's read set.
R2  Stepper.  pysam's real default stepper is ``"samtools"`` (``IteratorColumn.__cinit__``; the
    docstring saying ``"all"`` is stale).  Without a ``fastafile`` it does no BAQ / capQ, so it
    reduces to: drop ``flag & (UNMAP|SECONDARY|QCFAIL|DUP)`` (default ``flag_filter``),
    ``flag_require`` = 0, ``mapq < 0`` (never), and ``ignore_orphans`` (default True): drop
    ``flag & PAIRED and not flag & PROPER_PAIR``.  ``bam_plp_push`` additionally drops UNMAP.
R3  Column membership.  ``bam_plp64_next`` puts a buffered read in column p iff
    ``beg <= p < beg + raw reference length`` (raw ``bam_cigar2rlen``, no +1), so reads whose
    CIGAR consumes no reference never appear.
R4  ref->query.  ``resolve_cigar2`` walks the CIGAR: I and S advance the query offset, M/=/X
    advance both, D/N advance the reference only (column inside D/N -> ``is_del``/``is_refskip``,
    ``query_position`` None -> not counted), H/P/B advance nothing.  This equals a plain CIGAR walk
    EXCEPT when a reference-consuming op (M/=/X/D/N) has length 0: resolve_cigar2 only advances one
    such op per column, so it attributes one column to the empty op (spurious deletion / shifted
    qpos).  Reads containing such an op are handled by the exact fallback (R8).
R5  Base quality.  ``min_base_quality=0`` -> ``pileup_base_qual_skip`` never skips.  The original
    then skips reads whose ``query_qualities`` is None (``l_qseq == 0`` or ``qual[0] == 0xff`` -
    only the FIRST byte is tested) and bases with ``qual[qpos] < min_baseq``.
R6  Base.  ``query_sequence`` decodes nibbles with ``"=ACMGRSVTWYHKDBN"``; only A/C/G/T are
    counted (``.upper()`` is a no-op on that alphabet).
R7  Read filters added by the original loop: secondary, supplementary, duplicate,
    ``mapping_quality < min_mapq``.
R8  Exact fallback (sites recomputed with the original per-site pileup code, ``_pileup_site``):
    a) depth: ``bam_plp_push`` drops a read only when the buffer already holds > max_depth nodes
       (``mp->cnt > maxcnt``, cnt includes the empty tail node).  A per-site pileup only ever
       buffers reads of that site's R1+R2 read set, so if that set has <= max_depth reads nothing
       can be dropped.  Sites with more R1+R2 reads than max_depth fall back.  (pysam only calls
       ``bam_mplp_set_maxcnt`` when ``max_depth`` is truthy; otherwise htslib's default 8000.)
    b) mate overlap: ``bam_mplp_init_overlaps`` (ignore_overlaps=True) rewrites base qualities of
       overlapping mates; ``overlap_push`` only touches reads with PROPER_PAIR set (and mate
       mapped).  Any site overlapped by an R1+R2 read with PROPER_PAIR falls back.  PacBio CCS
       input is unpaired, so this never triggers on BiStro data.
    c) reads with a zero-length M/=/X/D/N CIGAR op (R4): every site of the cluster they overlap
       falls back.  (The scalar walk only notices an empty op that it reaches before the read's
       last site in the cluster; one lying further right is harmless, because the column
       resolve_cigar2 parks on an empty op is the op's own reference position, i.e. past every
       site already walked.)
R9  Contig absent from the BAM header -> all-zero rows (the original skipped the contig and its
    caller used ``counts.get(site, [0, 0, 0, 0])``), whatever the positions.  On a contig present
    in the header, positions < 1 raise ValueError like ``pileup(contig, -1, 0)``; positions past
    the contig end simply have no reads.
"""

from bisect import bisect_left
from itertools import chain

import numpy as np
import pysam

__all__ = [
    "BASE_ORDER", "CLUSTER_GAP", "DEFAULT_MAX_DEPTH",
    "count_alleles", "count_alleles_task", "split_positions",
]

BASE_ORDER = "ACGT"                 # column order of the returned (n, 4) arrays
DEFAULT_MAX_DEPTH = 1_000_000       # the original's pileup(max_depth=...)
_HTSLIB_DEFAULT_MAXCNT = 8000       # bam_plp_init(); used by pysam when max_depth is falsy

# Sites closer than this share one fetch.  A BAI fetch for one site decodes every record from the
# first read overlapping the site's 16-kb linear-index window, i.e. from ~(read length + 0..16 kb)
# before the site, so for gaps below about that distance one merged fetch decodes no more records
# than separate per-site fetches (the shared ones only once).  Per-site cost is dominated by this
# BGZF decoding (and by Lustre I/O when cold), not by Python.  Affects speed only, never results.
CLUSTER_GAP = 20_000

# Sites a single read must cover before the per-read numpy path beats the scalar CIGAR walk
# (measured crossover on the real HiFi subset: ~400-800 sites per read).  Speed only.
_DENSE_SITES = 512
# Flush (site, base) accumulators into the count array once they hold this many entries.
_FLUSH = 1 << 20

_FPAIRED, _FPROPER_PAIR, _FUNMAP = 0x1, 0x2, 0x4
_FSECONDARY, _FQCFAIL, _FDUP, _FSUPPLEMENTARY = 0x100, 0x200, 0x400, 0x800
_STEPPER_FILTER = _FUNMAP | _FSECONDARY | _FQCFAIL | _FDUP          # R2 default flag_filter

_BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
_BASE_LUT = np.full(256, -1, dtype=np.int64)                      # ASCII -> 0..3 / -1
for _b, _i in _BASE_CODE.items():
    _BASE_LUT[ord(_b)] = _i
# CIGAR op classes (BAM codes M=0 I=1 D=2 N=3 S=4 H=5 P=6 '='=7 X=8 B=9).
_MATCH = np.zeros(16, dtype=bool)
_MATCH[[0, 7, 8]] = True
_REF = np.zeros(16, dtype=bool)
_REF[[0, 2, 3, 7, 8]] = True
_QRY = np.zeros(16, dtype=bool)
_QRY[[0, 1, 4, 7, 8]] = True
_NO_CODES = np.zeros(0, dtype=np.int64)


def count_alleles(bam_path, contig, positions_1based_sorted_unique, min_mapq, min_baseq,
                  max_depth=DEFAULT_MAX_DEPTH, cluster_gap=CLUSTER_GAP, _dense_sites=_DENSE_SITES):
    """Return int64 array (n, 4) of A/C/G/T read counts at each 1-based position of ``contig``.

    ``positions_1based_sorted_unique``: strictly increasing ints >= 1.  ``bam_path`` may also be an
    open ``pysam.AlignmentFile`` (it is not closed).  Row i equals the original
    ``sharelib.count_alleles`` row for (contig, positions[i]) with the same min_mapq / min_baseq /
    max_depth.  ``cluster_gap`` / ``_dense_sites`` only tune speed (tests vary them).
    """
    pos = np.asarray(positions_1based_sorted_unique, dtype=np.int64).reshape(-1)
    out = np.zeros((pos.size, 4), dtype=np.int64)
    if pos.size == 0:
        return out
    if pos.size > 1 and not np.all(np.diff(pos) > 0):
        raise ValueError("positions must be sorted and unique")

    own = not isinstance(bam_path, pysam.AlignmentFile)
    bam = pysam.AlignmentFile(bam_path, "rb") if own else bam_path
    try:
        if contig not in bam.references:                              # R9
            return out
        if pos[0] < 1:      # R9: pileup(contig, -1, 0) raises in pysam.parse_region
            raise ValueError(f"start out of range ({int(pos[0]) - 1})")
        max_cnt = max_depth if max_depth else _HTSLIB_DEFAULT_MAXCNT  # R8a
        sites0 = pos - 1
        cuts = np.flatnonzero(np.diff(sites0) > cluster_gap) + 1
        bounds = np.concatenate(([0], cuts, [sites0.size]))
        for i0, i1 in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
            _count_cluster(bam, contig, sites0[i0:i1], min_mapq, min_baseq, max_cnt,
                           max_depth, _dense_sites, out[i0:i1])
    finally:
        if own:
            bam.close()
    return out


def count_alleles_task(*task):
    """Picklable pool worker.

    ``task`` = ``(bam_path, contig, positions, min_mapq, min_baseq[, max_depth])``, passed either as
    separate arguments (``starmap``) or as one tuple (``imap`` / ``imap_unordered``).
    Returns ``(contig, positions as int64 array, counts int64 (n, 4))``.
    """
    if len(task) == 1:
        task = task[0]
    bam_path, contig, positions, min_mapq, min_baseq, *rest = task
    positions = np.asarray(positions, dtype=np.int64).reshape(-1)
    counts = count_alleles(bam_path, contig, positions, min_mapq, min_baseq, *rest)
    return contig, positions, counts


def split_positions(positions, max_sites, cluster_gap=CLUSTER_GAP):
    """Split sorted positions into chunks of about ``max_sites`` for parallel tasks.

    Cuts are placed only between clusters (gap > cluster_gap) so no read is fetched twice, except
    that a single cluster larger than ``2 * max_sites`` is cut into equal pieces (a few boundary
    reads are then read twice; results are unaffected).  Returns a list of int64 arrays whose
    concatenation is ``positions``.
    """
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    if pos.size == 0:
        return []
    max_sites = max(1, int(max_sites))
    cuts = np.flatnonzero(np.diff(pos) > cluster_gap) + 1
    bounds = np.concatenate(([0], cuts, [pos.size])).tolist()
    chunks = []
    cur = None                                        # start index of the open chunk
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a > 2 * max_sites:                     # huge cluster: close open chunk, cut it
            if cur is not None:
                chunks.append(pos[cur:a])
                cur = None
            chunks.extend(np.array_split(pos[a:b], -(-(b - a) // max_sites)))
            continue
        if cur is None:
            cur = a
        if b - cur >= max_sites:
            chunks.append(pos[cur:b])
            cur = None
    if cur is not None:
        chunks.append(pos[cur:])
    return chunks


def _count_cluster(bam, contig, sites0, min_mapq, min_baseq, max_cnt, max_depth, dense, out):
    """Fill ``out`` (n, 4) for 0-based sorted ``sites0`` using one fetch over their span."""
    n = sites0.size
    sites = sites0.tolist()
    depth_diff = [0] * (n + 1)      # R1+R2 reads per site (R8a), as a difference array
    special_diff = [0] * (n + 1)    # sites needing the exact fallback (R8b, R8c)
    flat = np.zeros(4 * n, dtype=np.int64)
    acc = []                        # scalar path: site * 4 + base codes
    acc_np = []                     # dense path: arrays of site * 4 + base codes
    acc_np_len = 0
    acc_append = acc.append
    base_code = _BASE_CODE.get

    for r in bam.fetch(contig, sites[0], sites[-1] + 1):
        flag = r.flag
        if flag & _STEPPER_FILTER:                                    # R2
            continue
        if flag & _FPAIRED and not flag & _FPROPER_PAIR:              # R2 ignore_orphans
            continue
        start = r.reference_start
        endpos = r.reference_end                                      # bam_endpos (R1)
        if endpos is None:                                            # no CIGAR: endpos = pos+1
            endpos = start + 1
        lo = bisect_left(sites, start)
        hi = bisect_left(sites, endpos, lo)
        if lo == hi:
            continue
        depth_diff[lo] += 1
        depth_diff[hi] -= 1
        if flag & _FPROPER_PAIR:                                      # R8b
            special_diff[lo] += 1
            special_diff[hi] -= 1
        if flag & _FSUPPLEMENTARY:                                    # R7 (sec/dup gone in R2)
            continue
        if r.mapping_quality < min_mapq:                              # R7
            continue
        quals = r.query_qualities                                     # R5
        if quals is None:
            continue
        cig = r.cigartuples
        if not cig:                                                   # R3: no reference span
            continue

        if hi - lo >= dense:
            codes = _dense_read_codes(r, cig, start, sites0[lo:hi], lo, quals, min_baseq)
            if codes is None:                                         # R8c
                special_diff[lo] += 1
                special_diff[hi] -= 1
                continue
            acc_np.append(codes)
            acc_np_len += codes.size
            if acc_np_len >= _FLUSH:
                flat += np.bincount(np.concatenate(acc_np), minlength=4 * n)
                acc_np.clear()
                acc_np_len = 0
            continue

        # Scalar CIGAR walk (R4) over the few sites this read covers.
        seq = None
        rpos = start
        qpos = 0
        j = lo
        for op, ln in cig:
            if op == 0 or op == 7 or op == 8:                         # M = X
                if ln == 0:                                           # R8c
                    special_diff[lo] += 1
                    special_diff[hi] -= 1
                    break
                rend = rpos + ln
                while j < hi:
                    s = sites[j]
                    if s >= rend:
                        break
                    qp = qpos + s - rpos
                    if not quals[qp] < min_baseq:                     # R5
                        if seq is None:
                            seq = r.query_sequence
                        b = base_code(seq[qp])                        # R6
                        if b is not None:
                            acc_append(j * 4 + b)
                    j += 1
                if j >= hi:
                    break
                rpos = rend
                qpos += ln
            elif op == 2 or op == 3:                                  # D N: not counted
                if ln == 0:                                           # R8c
                    special_diff[lo] += 1
                    special_diff[hi] -= 1
                    break
                rend = rpos + ln
                while j < hi and sites[j] < rend:
                    j += 1
                if j >= hi:
                    break
                rpos = rend
            elif op == 1 or op == 4:                                  # I S
                qpos += ln
            # H P B: consume neither
        if len(acc) >= _FLUSH:
            flat += np.bincount(np.asarray(acc, dtype=np.int64), minlength=4 * n)
            acc.clear()

    if acc:
        flat += np.bincount(np.asarray(acc, dtype=np.int64), minlength=4 * n)
    if acc_np:
        flat += np.bincount(np.concatenate(acc_np), minlength=4 * n)
    out[:] = flat.reshape(n, 4)

    depth = np.cumsum(depth_diff[:-1])
    special = np.cumsum(special_diff[:-1]) > 0
    redo = np.flatnonzero(special | (depth > max_cnt))
    if redo.size:
        for i in redo.tolist():
            out[i] = _pileup_site(bam, contig, sites[i] + 1, min_mapq, min_baseq, max_depth)


def _dense_read_codes(r, cig, start, s, lo, quals, min_baseq):
    """Vectorised R4-R6 for one read covering many sites ``s`` (0-based, sorted, all in
    ``[start, bam_endpos)``).

    Returns int64 codes ``site_index * 4 + base`` of counted bases, or None if the read has a
    zero-length reference-consuming op (R8c).
    """
    c = np.fromiter(chain.from_iterable(cig), dtype=np.int64, count=2 * len(cig))
    op = c[0::2]
    ln = c[1::2]
    isref = _REF[op]
    if (isref & (ln == 0)).any():                      # R8c
        return None
    ref_end = np.cumsum(ln * isref)
    ref_end += start                                   # exclusive reference end of each op
    if ref_end[-1] == start:                           # no reference span: never piled (R3)
        return _NO_CODES
    # Every site is < raw end here (bam_endpos == raw end when rlen > 0), so k < len(op).
    k = np.searchsorted(ref_end, s, side="right")      # op holding each site
    idx = np.flatnonzero(_MATCH[op][k])                # D/N columns are not counted (R4)
    if idx.size == 0:
        return _NO_CODES
    # Inside an M/=/X op query and reference advance together: qpos = site + (q_end - ref_end).
    qp = s[idx] + (np.cumsum(ln * _QRY[op]) - ref_end)[k[idx]]
    q = np.frombuffer(quals, dtype=np.uint8)[qp].astype(np.int64)
    b = _BASE_LUT[np.frombuffer(r.query_sequence.encode("ascii"), dtype=np.uint8)[qp]]
    keep = ~(q < min_baseq) & (b >= 0)                 # same base-quality test as the original (R5, R6)
    return ((idx + lo) * 4 + b)[keep]


def _pileup_site(bam, contig, pos, min_mapq, min_baseq, max_depth):
    """Exact per-site recount with the ORIGINAL code path (used only for R8 sites)."""
    row = [0, 0, 0, 0]
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
            idx = _BASE_CODE.get(aln.query_sequence[qpos].upper())
            if idx is not None:
                row[idx] += 1
    return row
