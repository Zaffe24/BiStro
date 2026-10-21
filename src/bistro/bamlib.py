"""
Per-read helpers used by caller.py: quality-based rejection of individual CCS
reads, base-quality masking near read ends and indels, and the per-molecule
walk that classifies every aligned reference position of a fwd/rev duplex pair
(duplex_walk) or, with --collapsed, of a single read (collapsed_walk), and
formats the resulting mutation records.

The walks are numpy-vectorised over the read; only the (rare) mutation records
are formatted one by one. Each record is returned as its final text prefix
(everything up to and including TYPE, exactly as vcflib.write_mut_one prints it)
plus its reference position; the caller appends f"\\t{depth}\\n" at write time.

The list-based helpers of the original per-base implementation (mask_ends,
mask_indels, process_mutations, process_mutations_collapsed) are kept unchanged
for backward compatibility and as the reference the new functions are tested
against (tests/test_walk.py).
"""

from typing import NamedTuple

import pysam
import numpy as np

from . import reportlib, bqlib


def get_seq_identity(read: pysam.AlignedSegment):
    return round(1 - float(read.get_tag("NM")) / read.query_length, 3)


def get_softclipping(read: pysam.AlignedSegment):
    return 1 - float(read.query_alignment_length / read.query_length)


def discard_low_qual_read(read: pysam.AlignedSegment,
                    report:reportlib.METRICS,
                    min_mapq,
                    min_ec,
                    min_qlen,
                    max_qlen,
                    max_softclipping,
                    min_seq_identity):

    if read.is_secondary or read.is_supplementary:
        report.num_discarded += 1
        report.disc_no_primary += 1
        return True

    elif read.mapping_quality < min_mapq:
        report.num_discarded += 1
        report.disc_mapq_qual += 1
        return True

    elif read.get_tag("ec") < min_ec:
        report.num_discarded += 1
        report.disc_min_ec += 1
        return True

    elif read.query_length < min_qlen:
        report.num_discarded += 1
        report.disc_min_qlen += 1
        return True

    elif read.query_length > max_qlen:
        report.num_discarded += 1
        report.disc_max_qlen += 1
        return True

    elif get_softclipping(read) >= max_softclipping:
        report.num_discarded += 1
        report.disc_max_soft += 1
        return True

    elif get_seq_identity(read) < min_seq_identity:
        report.num_discarded += 1
        report.disc_min_identity += 1
        return True

    return False


def remove_useless_tags(read: pysam.AlignedSegment):
    for tag in ("sa", "sx", "sn", "ip","pw"):
        if read.has_tag(tag):
            read.set_tag(tag, None)
    return read


# ---------------------------------------------------------------------------
# CIGAR helpers
# ---------------------------------------------------------------------------
# indexed by CIGAR op code: M I D N S H P = X B
_CONSUMES_QUERY = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0], dtype=np.int64)
_CONSUMES_REF   = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 0], dtype=np.int64)
_IS_MATCH       = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0], dtype=bool)
_N = ord("N")
_EMPTY = np.zeros(0, dtype=np.int64)


def aligned_pairs_array(read: pysam.AlignedSegment):
    """(qpos, rpos) int64 arrays of the read's M/=/X positions: the same pairs as
    get_aligned_pairs(matches_only=True), built from the CIGAR without Python tuples."""
    ct = np.asarray(read.cigartuples, dtype=np.int64)
    ops, lens = ct[:, 0], ct[:, 1]
    qlens = lens * _CONSUMES_QUERY[ops]
    rlens = lens * _CONSUMES_REF[ops]
    qstart = np.cumsum(qlens) - qlens
    rstart = read.reference_start + np.cumsum(rlens) - rlens
    is_m = _IS_MATCH[ops]
    mlens = lens[is_m]
    offs = np.arange(mlens.sum()) - np.repeat(np.cumsum(mlens) - mlens, mlens)
    return np.repeat(qstart[is_m], mlens) + offs, np.repeat(rstart[is_m], mlens) + offs


# ---------------------------------------------------------------------------
# Base-quality masking
# ---------------------------------------------------------------------------
def masked_qualities(read: pysam.AlignedSegment, trim_ends, indels_window) -> np.ndarray:
    """The read's base qualities (np.uint8 array) after mask_ends(read, trim_ends) and
    then mask_indels(read, indels_window), WITHOUT writing them back into the read.

    Same results and same errors as the original pair of functions, including the
    quirk that a fractional trim_ends rounding to n == 0 zeroes the whole read
    (quals[-0:] is the whole array). Note indels_window == 0 still masks: the
    inserted bases plus the next base for an insertion, the base after a deletion.
    """
    m = trim_ends
    if m == 0:
        n = None
    elif 0 < m < 1:
        n = round(read.query_length * m)
    elif m >= 1:
        n = int(m)
    else:
        raise ValueError("Error in mask_ends: {} must be a positive percentage (0 < m < 1) or an integer >= 1".format(m))

    raw = read.query_qualities
    if n is not None:
        quals = np.array(raw, dtype=np.uint8)   # raises TypeError on None, as mask_ends did
        quals[:n] = 0
        quals[-n:] = 0                          # n == 0 -> quals[0:] = 0 (original quirk)

    p = indels_window
    if p < 0:
        raise ValueError("Error in mask_indels: {} must be a positive integer".format(p))
    cigar = read.cigartuples
    if cigar is None or raw is None:
        raise ValueError("Error in mask_indels: No cigartuples of Qualities present")
    if n is None:
        quals = np.array(raw, dtype=np.uint8)

    # mask_indels: the original loop over CIGAR *operations* (not bases), zeroing each
    # window directly instead of via a boolean mask (zeroing is idempotent, so the
    # union is the same). qpos = query offset before the op, advanced by M I S = X.
    L = quals.size
    qpos = 0
    for op, length in cigar:
        if op == 1:                                   # insertion
            quals[max(0, qpos - p):min(L, qpos + length + p + 1)] = 0
            qpos += length
        elif op == 2:                                 # deletion
            quals[max(0, qpos - p):min(L, qpos + p + 1)] = 0
        elif op == 0 or op == 7 or op == 8 or op == 4:
            qpos += length
    return quals


## mask terminal read ends by setting their quality to 0 ##
## (original implementation, kept for compatibility; see masked_qualities)
def mask_ends(read: pysam.AlignedSegment, m:float) -> pysam.AlignedSegment:
    if m == 0:
        return read
    elif 0 < m < 1:
        # m is a fraction: mask that percentage of bases at each end
        n = round(read.query_length * m)
    elif m >= 1:
        # m is an absolute number of bases to mask at each end
        n = int(m)
    else:
        raise ValueError("Error in mask_ends: {} must be a positive percentage (0 < m < 1) or an integer >= 1".format(m))

    quals = np.array(read.query_qualities, dtype=np.uint8)
    quals[:n] = 0
    quals[-n:] = 0
    read.query_qualities = quals.tolist()
    return read


## set base quality to 0 around indels ###
## (original implementation, kept for compatibility; see masked_qualities)
def mask_indels(read: pysam.AlignedSegment, p:int) -> pysam.AlignedSegment:
    if p<0:
        raise ValueError("Error in mask_indels: {} must be a positive integer".format(p))

    if read.cigartuples is None or read.query_qualities is None:
        raise ValueError("Error in mask_indels: No cigartuples of Qualities present")

    quals = np.array(read.query_qualities, dtype = int)
    qpos = 0
    mask = np.zeros(len(quals), dtype=bool)

    for op, length in read.cigartuples:
        if op == 1:  # insertion
            lo = max(0, qpos - p)
            hi = min(len(quals), qpos + length + p + 1)
            mask[lo:hi] = True
            qpos += length
        elif op == 2:  # deletion
            lo = max(0, qpos - p)
            hi = min(len(quals), qpos + p + 1)
            mask[lo:hi] = True
        elif op in (0, 7, 8, 4):
            qpos += length

    quals[mask] = 0
    read.query_qualities = quals.tolist()
    return read


# ---------------------------------------------------------------------------
# Shared walk helpers
# ---------------------------------------------------------------------------
class WalkResult(NamedTuple):
    """Everything the caller needs from one molecule (duplex pair or collapsed read)."""
    callable_pos: np.ndarray   # int64, ascending, unique: positions to add 1 to depth at
    rec_pos: np.ndarray        # int64, ascending: reference position of each record
    rec_text: list             # per record: text up to and including TYPE (no '\t{depth}\n')
    n_callable: int            # report.callable_bps increment (== callable_pos.size)


_EMPTY.flags.writeable = False       # shared empty result array: never written to


def _no_walk():
    return WalkResult(_EMPTY, _EMPTY, [], 0)


def _match_blocks(cigar, ref_start, pad_consumes_query):
    """Maximal runs of a read's aligned M/=/X positions as three parallel lists
    (reference start, length, qpos - rpos): ascending, non-overlapping, and runs that
    are contiguous on both sequences (e.g. '10=1X5=') merged. One pass over the CIGAR
    operations (tens per read), not over bases.

    pad_consumes_query=True follows pysam's get_aligned_pairs (the original duplex
    walk), which advances the query on P (padding) ops; False follows
    aligned_pairs_array (the original collapsed walk), which advances neither
    coordinate on P. Every other op is treated alike: M/=/X both, I/S query, D/N
    reference, H/B neither. (Aligners do not emit P for PacBio reads.)"""
    r = ref_start
    q = 0
    r0s, lens, offs = [], [], []
    for op, l in cigar:
        if op == 0 or op == 7 or op == 8:
            if l:                                     # a 0-length M yields no pair
                if r0s and r == r0s[-1] + lens[-1] and q - r == offs[-1]:
                    lens[-1] += l
                else:
                    r0s.append(r)
                    lens.append(l)
                    offs.append(q - r)
            r += l
            q += l
        elif op == 1 or op == 4 or (op == 6 and pad_consumes_query):
            q += l
        elif op == 2 or op == 3:
            r += l
    return r0s, lens, offs


def _project(blocks, lo, n, seq_u8, qual):
    """The read's bases and base qualities laid out on reference positions
    lo .. lo+n-1 (two uint8 arrays of length n), base 0 where the read has no M/=/X
    base (real bases are ASCII letters, never 0). One slice copy per aligned block
    replaces the per-position gathers of an aligned-pairs merge-join."""
    B = np.zeros(n, dtype=np.uint8)
    Q = np.zeros(n, dtype=np.uint8)
    for r0, l, o in zip(*blocks):
        a = r0 - lo
        if a >= n:
            break
        b = a + l
        if a < 0:
            a = 0
        if b > n:
            b = n
        if a < b:
            B[a:b] = seq_u8[lo + o + a:lo + o + b]
            Q[a:b] = qual[lo + o + a:lo + o + b]
    return B, Q


def _qpos_of(blocks, rpos):
    """Query positions of aligned reference positions rpos (int64 array, each inside
    one of the read's blocks): r + offset of the block containing r."""
    r0s = np.asarray(blocks[0], dtype=np.int64)
    offs = np.asarray(blocks[2], dtype=np.int64)
    return rpos + offs[np.searchsorted(r0s, rpos, "right") - 1]


def _window_mask(n, centers, w):
    """Boolean mask over 0..n-1, True within w of any center (|i - c| <= w): the
    original blacklist set().update(range(r - w, r + w + 1)); all False when w < 0
    (the ranges are then empty). centers: int64 array of in-span positions."""
    out = np.zeros(n, dtype=bool)
    if w < 0 or centers.size == 0:
        return out
    if centers.size <= 32:                            # usual case: a few mismatches
        for c in centers.tolist():
            out[max(c - w, 0):c + w + 1] = True
        return out
    lo = np.maximum(centers - w, 0)
    hi = np.minimum(centers + w + 1, n)
    d = np.bincount(lo, minlength=n + 1) - np.bincount(hi, minlength=n + 1)
    return np.cumsum(d[:n]) > 0


def _bernoulli_select(rng, n, z_prob):
    """Indices in [0, n), each kept independently with probability 1/z_prob (the
    original's random.randint(1, z_prob) == 1 per reference-matching position).
    Draws geometric gaps between successes, so the cost scales with the number of
    hits instead of n."""
    if n == 0:
        return _EMPTY
    if rng is None:                 # no per-task generator given: fresh OS-seeded one
        rng = np.random.default_rng()
    p = 1.0 / z_prob
    out = []
    last = -1                       # index of the last success (or -1)
    while True:
        k = int((n - 1 - last) * p) + 8
        idx = last + np.cumsum(rng.geometric(p, size=k))
        out.append(idx[idx < n])
        if idx[-1] >= n:
            break
        last = int(idx[-1])
    return np.concatenate(out) if len(out) > 1 else out[0]


def _window_mean_sd(qual, p, w, ln):
    """str(round(np.mean(x), 2)), str(round(np.std(x), 2)) of the original's
    x = qual[np.arange(max(p - w, 0), min(ln, p + w + 1))], as an (identical) slice.
    The stop is clamped at 0 so that an empty arange (negative w) never turns into a
    wrap-around slice. round() on np.float64 is numpy's rounding (kept on purpose)."""
    x = qual[max(p - w, 0):max(min(ln, p + w + 1), 0)]
    return str(round(np.mean(x), 2)), str(round(np.std(x), 2))


# ---------------------------------------------------------------------------
# Duplex mode: one fwd/rev read pair per ZMW
# ---------------------------------------------------------------------------
def duplex_walk(read1: pysam.AlignedSegment, read2: pysam.AlignedSegment,
                q1: np.ndarray, q2: np.ndarray,
                zmw, contig: str, ref_seq: str, ref_arr: np.ndarray,
                min_bq, z_prob, mismatch_window_len,
                rng=None, bq_hist=None) -> WalkResult:
    """Vectorised equivalent of the original caller.py per-ZMW duplex walk + blacklist +
    bamlib.process_mutations + vcflib.write_mut_one text formatting.

    read1/read2: the fwd / rev read; q1/q2: their masked_qualities(); zmw: the `zm` tag
    value the pair was matched on (printed in the ZMW column); ref_seq: the contig's
    upper-cased sequence (str) and ref_arr the same as a uint8 array; rng: a
    np.random.Generator for the TYPE z baseline (only used when z_prob > 0); bq_hist:
    if not None, the duplex BQ profile histogram, updated like bqlib.add_duplex.

    Over the reference positions where both reads have an M/=/X base (the original
    merge-join of get_aligned_pairs(matches_only=True)), with ok = both BQ >= min_bq
    and m1/m2 = read base equals reference:
      ok & m1 & m2                         -> callable context (+ TYPE z w.p. 1/z_prob)
      ok & (m1 ^ m2)                       -> context + TYPE m + single-strand mismatch
      ok & ~m1 & ~m2 & b1 == b2 & ref != N -> context + TYPE x  (no N test on read bases)
    Callable = context positions farther than mismatch_window_len from every mismatch.
    Records are in ascending reference order (z/m/x interleaved).

    Implementation: both reads are projected onto their common reference span (one
    slice per aligned block), so the classification is elementwise on that span.
    """
    blk1 = _match_blocks(read1.cigartuples, read1.reference_start, True)
    blk2 = _match_blocks(read2.cigartuples, read2.reference_start, True)
    r1, l1 = blk1[0], blk1[1]
    r2, l2 = blk2[0], blk2[1]
    if not r1 or not r2:
        return _no_walk()
    lo = max(r1[0], r2[0])
    hi = min(r1[-1] + l1[-1], r2[-1] + l2[-1])
    if hi <= lo:
        return _no_walk()
    n = hi - lo
    s1 = read1.query_sequence.upper()
    s2 = read2.query_sequence.upper()
    B1, Q1 = _project(blk1, lo, n, np.frombuffer(s1.encode("ascii"), dtype=np.uint8), q1)
    B2, Q2 = _project(blk2, lo, n, np.frombuffer(s2.encode("ascii"), dtype=np.uint8), q2)
    rb = ref_arr[lo:hi]

    m1 = B1 == rb                  # False where read 1 has no aligned base (0 != letter)
    m2 = B2 == rb
    both_ref = m1 & m2             # implies both reads aligned here
    bq = np.minimum(Q1, Q2)        # both >= min_bq  <=>  min >= min_bq
    if bq_hist is not None:        # bqlib.add_duplex: matched pairs, no BQ gate
        bqlib.add_counts(bq_hist, bq[both_ref])

    ok = (bq >= min_bq) & (B1 != 0) & (B2 != 0)
    any_m = m1 | m2
    ctx = ok & (any_m | ((B1 == B2) & (rb != _N)))    # == ref | ss-mismatch | x
    is_mm = ok & (m1 ^ m2)
    w = mismatch_window_len
    if is_mm.any():
        mm_rel = np.flatnonzero(is_mm)
        callable_pos = np.flatnonzero(ctx & ~_window_mask(n, mm_rel, w)) + lo
    else:
        mm_rel = _EMPTY
        callable_pos = np.flatnonzero(ctx) + lo

    is_x = ctx & ~any_m
    is_rec = is_mm | is_x
    if z_prob > 0:
        ref_rel = np.flatnonzero(ok & both_ref)
        is_rec[ref_rel[_bernoulli_select(rng, ref_rel.size, z_prob)]] = True
    if not is_rec.any():
        return WalkResult(callable_pos, _EMPTY, [], int(callable_pos.size))

    # ---- records (rare): process_mutations + write_mut_one formatting ----
    ri = np.flatnonzero(is_rec)
    rec_pos = ri + lo
    rec_mm = is_mm[ri]
    types = np.where(rec_mm, "m", np.where(is_x[ri], "x", "z")).tolist()
    # N_MIS = mismatches within [r-w, r+w] (the original intersect1d count), -1 for TYPE m
    mm_pos = mm_rel + lo
    n_mis = (np.maximum(np.searchsorted(mm_pos, rec_pos + w, "right")
                        - np.searchsorted(mm_pos, rec_pos - w, "left"), 0)
             - rec_mm).tolist()
    qp1 = _qpos_of(blk1, rec_pos)
    qp2 = _qpos_of(blk2, rec_pos)

    ln1, ln2 = read1.query_length, read2.query_length
    ec1, ec2 = str(int(read1.get_tag("ec"))), str(int(read2.get_tag("ec")))
    id1, id2 = str(get_seq_identity(read1)), str(get_seq_identity(read2))
    ln1s, ln2s = str(ln1), str(ln2)
    # first / last aligned pair of each read (aln[0][1], aln[-1][1] in the original)
    nob = str(abs(r1[0] - r2[0]) + abs((r1[-1] + l1[-1]) - (r2[-1] + l2[-1])))
    sm1 = read1.get_tag("sm")
    sm2 = read2.get_tag("sm")
    zmw_s = str(zmw)

    texts = []
    for r, p1, p2, bq1, bq2, t, nm in zip(rec_pos.tolist(), qp1.tolist(), qp2.tolist(),
                                          q1[qp1].tolist(), q2[qp2].tolist(), types, n_mis):
        mean1, sd1 = _window_mean_sd(q1, p1, w, ln1)
        mean2, sd2 = _window_mean_sd(q2, p2, w, ln2)
        texts.append("\t".join((
            contig, str(r), str(r + 1), zmw_s, ref_seq[r - 1:r + 2], ref_seq[r],
            ":".join((s1[p1], str(bq1), str(p1), ec1, str(sm1[p1]), ln1s, id1, mean1, sd1)),
            ":".join((s2[p2], str(bq2), str(p2), ec2, str(sm2[p2]), ln2s, id2, mean2, sd2)),
            nob, str(nm), t)))
    return WalkResult(callable_pos, rec_pos, texts, int(callable_pos.size))


# ---------------------------------------------------------------------------
# Collapsed mode (--collapsed): every read is its own molecule
# ---------------------------------------------------------------------------
def walk_collapsed(qpos, rpos, seq, qual, ref_arr, min_bq, z_prob, rng=None):
    """Collapsed-mode counterpart of the duplex walk, vectorised over one read.

    Over the aligned positions whose BQ >= min_bq: a base equal to the reference is
    callable context; a non-N base differing from a non-N reference base is a candidate
    substitution (TYPE x, also callable, as in the duplex walk). With z_prob > 0 each
    reference-matching position is also recorded as a random baseline (TYPE z) with
    probability 1/z_prob. With no second strand there is no single-strand mismatch
    (TYPE m), hence nothing to blacklist around.

    rng: np.random.Generator for the z draws. None keeps the original behaviour (global
    numpy RNG, one draw per aligned position) for callers not yet passing one.

    Returns (context_rpos, rec_qpos, rec_rpos, rec_type); records in reference order.
    """
    base = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)[qpos]
    ref = ref_arr[rpos]
    ok = qual[qpos] >= min_bq
    is_ref = base == ref
    match = ok & is_ref
    subst = ok & ~is_ref & (base != _N) & (ref != _N)
    keep = subst
    if z_prob > 0:
        if rng is None:
            keep = subst | (match & (np.random.randint(1, z_prob + 1, size=len(match)) == 1))
        else:
            keep = subst.copy()
            midx = np.flatnonzero(match)
            keep[midx[_bernoulli_select(rng, midx.size, z_prob)]] = True
    rec_type = np.where(subst[keep], "x", "z").tolist()
    return rpos[match | subst], qpos[keep], rpos[keep], rec_type


def collapsed_records(read: pysam.AlignedSegment, rec_qpos, rec_rpos, rec_type,
                      seq, qual, mismatch_window_len, contig, mol_id, ref_seq):
    """process_mutations_collapsed + write_mut_one(collapsed=True) formatting: one text
    prefix per record (STRAND_2 '.', NOB 0, N_MIS 0). Returns a list of str."""
    texts = []
    if len(rec_type) > 0:
        ln = read.query_length
        lns = str(ln)
        ec = str(int(read.get_tag("ec")))
        idn = str(get_seq_identity(read))
        sm = read.get_tag("sm") if read.has_tag("sm") else None
        w = mismatch_window_len
        mol = str(mol_id)
        rq = np.asarray(rec_qpos)
        for p, r, bq, t in zip(rq.tolist(), np.asarray(rec_rpos).tolist(),
                               qual[rq].tolist(), rec_type):
            mean, sd = _window_mean_sd(qual, p, w, ln)
            texts.append("\t".join((
                contig, str(r), str(r + 1), mol, ref_seq[r - 1:r + 2], ref_seq[r],
                ":".join((seq[p], str(bq), str(p), ec, str(sm[p]) if sm is not None else ".",
                          lns, idn, mean, sd)),
                ".", "0", "0", t)))
    return texts


def collapsed_walk(read: pysam.AlignedSegment, q: np.ndarray, contig: str,
                   ref_seq: str, ref_arr: np.ndarray, min_bq, z_prob, mismatch_window_len,
                   rng=None, bq_hist=None) -> WalkResult:
    """Everything the original collapsed branch of caller.py does for one read, given its
    masked_qualities() q: BQ profile (== bqlib.add_read) if bq_hist is not None, the
    walk_collapsed classification, and the records formatted like
    write_mut_one(collapsed=True) (MOL = read name). No blacklist in collapsed mode:
    callable = context. Same span projection as duplex_walk (aligned_pairs_array's
    CIGAR semantics, as the original collapsed path)."""
    blk = _match_blocks(read.cigartuples, read.reference_start, False)
    r0s, lens = blk[0], blk[1]
    if not r0s:
        return _no_walk()
    lo = r0s[0]
    n = r0s[-1] + lens[-1] - lo
    s = read.query_sequence.upper()
    B, Q = _project(blk, lo, n, np.frombuffer(s.encode("ascii"), dtype=np.uint8), q)
    rb = ref_arr[lo:lo + n]

    is_ref = B == rb                       # implies an aligned base
    if bq_hist is not None:                # bqlib.add_read: matching bases, no BQ gate
        bqlib.add_counts(bq_hist, Q[is_ref])
    ok = (Q >= min_bq) & (B != 0)
    match = ok & is_ref
    subst = ok & ~is_ref & (B != _N) & (rb != _N)
    ctx_pos = np.flatnonzero(match | subst) + lo
    keep = subst
    if z_prob > 0:
        if rng is None:                    # never fall back to the shared global numpy RNG
            rng = np.random.default_rng()
        keep = subst.copy()
        midx = np.flatnonzero(match)
        keep[midx[_bernoulli_select(rng, midx.size, z_prob)]] = True
    if not keep.any():
        return WalkResult(ctx_pos, _EMPTY, [], int(ctx_pos.size))
    ki = np.flatnonzero(keep)
    rec_rpos = ki + lo
    rec_type = np.where(subst[ki], "x", "z").tolist()
    texts = collapsed_records(read, _qpos_of(blk, rec_rpos), rec_rpos, rec_type, s, q,
                              mismatch_window_len, contig, read.query_name, ref_seq)
    return WalkResult(ctx_pos, rec_rpos, texts, int(ctx_pos.size))


# ---------------------------------------------------------------------------
# Original list-based record builders (kept for compatibility; see duplex_walk /
# collapsed_records for the text-producing equivalents)
# ---------------------------------------------------------------------------
def process_mutations_collapsed(read: pysam.AlignedSegment,
                                rec_qpos, rec_rpos, rec_type,
                                seq, qual,
                                mismatch_window_len, contig,
                                mol_id, fasta):
    """Mutation rows for one collapsed read, in process_mutations' layout with the
    read as STRAND_1 and no STRAND_2 sub-fields (written as "." by
    vcflib.write_mut_one); NOB and N_MIS are 0 (no second read, no ss mismatches).
    SM is "." when the read has no `sm` tag (not all HiFi BAMs carry it)."""
    output = []
    if len(rec_type) > 0:
        ln = read.query_length
        ec = int(read.get_tag("ec"))
        idn = get_seq_identity(read)
        sm = read.get_tag("sm") if read.has_tag("sm") else None

        for p, r, t in zip(rec_qpos.tolist(), rec_rpos.tolist(), rec_type):
            qual_range = qual[np.arange(max(p - mismatch_window_len, 0), min(ln, p+mismatch_window_len+1))]
            output.append([contig,
                           r,
                           r+1,
                           mol_id,
                           fasta[r -1 : r+2],
                           fasta[r],

                           seq[p],
                           qual[p],
                           p,
                           ec,
                           sm[p] if sm is not None else ".",
                           ln,
                           idn,
                           round(np.mean(qual_range),2),
                           round(np.std(qual_range),2),

                           0,     # NOB
                           0,     # N_MIS
                           t])
    return output


def process_mutations(read1:pysam.AlignedSegment,
                      read2:pysam.AlignedSegment,
                      list_qpos1:list,
                      list_qpos2:list,
                      list_ref1:list,
                      list_type:list,
                      mismatch_list:list,
                      seq1, seq2, qual1, qual2,
                      mismatch_window_len, contig,
                      zmw, fasta,
                      aln1, aln2
                      ):

    output = []
    if len(list_type)>0:

        ln1 = read1.query_length
        ln2 = read2.query_length
        ec1 = int(read1.get_tag("ec"))
        ec2 = int(read2.get_tag("ec"))
        id1 = get_seq_identity(read1)
        id2 = get_seq_identity(read2)
        non_overlapping_bases = abs(aln1[0][1] - aln2[0][1]) + abs(aln1[-1][1] - aln2[-1][1])

        for n, r in enumerate(list_ref1):
            ref_base = fasta[r]
            triplet = fasta[r -1 : r+2]
            t = list_type[n]

            p1 = list_qpos1[n]
            p2 = list_qpos2[n]

            q1 = qual1[p1]
            q2 = qual2[p2]
            b1 = seq1[p1]
            b2 = seq2[p2]

            sm1 = read1.get_tag("sm")[p1]
            sm2 = read2.get_tag("sm")[p2]

            ##compute mismatches within window from mutations
            mismatch_range = np.arange(r - mismatch_window_len, r+mismatch_window_len+1)
            n_mismatches = len(np.intersect1d(mismatch_range, mismatch_list))

            ## compute QB standard deviation around mutation
            qual_range1 = qual1[np.arange(max(p1 - mismatch_window_len, 0), min(ln1, p1+mismatch_window_len+1))]
            qual_range2 = qual2[np.arange(max(p2 - mismatch_window_len, 0), min(ln2, p2+mismatch_window_len+1))]

            mean1, mean2 = round(np.mean(qual_range1),2), round(np.mean(qual_range2),2)
            sd1, sd2 = round(np.std(qual_range1),2), round(np.std(qual_range2),2)

            mut = [contig,
                   r,
                   r+1,
                   zmw,
                   triplet,
                   ref_base,

                   b1,
                   q1,
                   p1,
                   ec1,
                   sm1,
                   ln1,
                   id1,
                   mean1,
                   sd1,

                   b2,
                   q2,
                   p2,
                   ec2,
                   sm2,
                   ln2,
                   id2,
                   mean2,
                   sd2,

                   non_overlapping_bases,
                   n_mismatches -1 if t == "m" else n_mismatches,
                   t]
            output.append(mut)

    return output
