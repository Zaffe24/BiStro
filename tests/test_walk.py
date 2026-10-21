#!/usr/bin/env python3
"""
Equivalence test of the vectorised per-molecule walk in bistro.bamlib
(masked_qualities, duplex_walk, collapsed_walk) against the ORIGINAL per-ZMW code
path of caller.call_somatic_mutations (pairing, mask_ends/mask_indels, while-loop
walk, blacklist, process_mutations, write_mut_one text formatting).

Plain script (no pytest); exits non-zero on any failure.

  python tests/test_walk.py                                   (~1-2 min, 1 CPU)
      tests/data BAMs, full parameter grid, both modes, z checks, whole-contig check.
  python tests/test_walk.py --bam X.bam [--bam Y.bam] --ref REF.fa.gz --grid real --procs N
      the same on real data (run through srun: it is CPU-heavy). Without --region every
      contig with reads is used WHOLE, so pass small sub-BAMs (e.g. samtools view of a
      2-Mbp window); the whole-contig check always processes entire contigs.
  python tests/test_walk.py --bam X.bam --ref REF.fa.gz --region C:S-E --timing
      per-molecule time of the original vs the new path (default parameters).

Original code is loaded from $BISTRO_ORIG_SRC (default: the validation snapshot); the
new code from this repo's src/.

What is compared, per molecule (ZMW duplex, or read with --collapsed), z_prob = 0:
  * masked qualities (np.uint8) == the original write-back result, and the read's own
    qualities are left untouched by masked_qualities;
  * aligned_pairs_array == get_aligned_pairs(matches_only=True);
  * callable positions (int64, ascending, unique) == the original's callable list;
  * record text prefixes (up to TYPE) == write_mut_one's text without '\\t{depth}\\n';
  * record positions, report counter (callable_bps), BQ histogram (bq profile on).
With z_prob = 1 every reference-matching position becomes a TYPE z record in both
implementations (randint(1, 1) == 1), so z rows are compared exactly on a subset of
molecules. With z_prob > 1 only the new path is run (seeded rng): non-z records and
callable positions must equal the z_prob = 0 run, z rows must be well formed and sit
on reference-matching positions, and the z rate must be consistent with 1/z_prob.
Whole-contig check: the original call_somatic_mutations() on a whole contig vs. the
new functions aggregated the same way (report, depth array, write_mut_one text, BQ
histogram) -- this guards the fidelity of the original-path copy used per molecule.
"""

import argparse
import gzip
import importlib.util
import itertools
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time
import warnings

sys.dont_write_bytecode = True          # never write __pycache__ into original_src
warnings.filterwarnings("ignore", category=RuntimeWarning)   # mean of empty slice (both sides)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ORIG_SRC = os.environ.get(
    "BISTRO_ORIG_SRC",
    "/home/csic/gpo/pza/lustre/bistro_refactor_validation/original_src/bistro")

import numpy as np
import pysam
pysam.set_verbosity(0)


def _load_original():
    spec = importlib.util.spec_from_file_location(
        "orig_bistro", os.path.join(ORIG_SRC, "__init__.py"),
        submodule_search_locations=[ORIG_SRC])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orig_bistro"] = mod
    spec.loader.exec_module(mod)
    import orig_bistro.caller, orig_bistro.bamlib, orig_bistro.bqlib   # noqa: E401
    import orig_bistro.reportlib, orig_bistro.vcflib                    # noqa: E401
    return sys.modules["orig_bistro"]


O = _load_original()
sys.path.insert(0, os.path.join(REPO, "src"))
from bistro import bamlib as NB, bqlib as NQ, reportlib as NR   # noqa: E402
assert os.path.dirname(NB.__file__) == os.path.join(REPO, "src", "bistro"), NB.__file__
OB, OQ, OR = O.bamlib, O.bqlib, O.reportlib

FILTER_RELAXED = dict(min_mapq=0, min_sequence_identity=0, min_qlen=0, max_qlen=10_000,
                      min_ec=0, max_softclipping=.2)
FILTER_DEFAULT = dict(min_mapq=60, min_sequence_identity=.99, min_qlen=800, max_qlen=10_000,
                      min_ec=5, max_softclipping=.2)


# ---------------------------------------------------------------------------
# ORIGINAL per-molecule path: verbatim copy of the loop body of the original
# caller.call_somatic_mutations (only renamed P[...] parameters), using the
# original bamlib / bqlib modules.
# ---------------------------------------------------------------------------
def orig_duplex(read1, read2, zmw, P, CONTIG, ref_seq, ref_arr, bq_hist, z_prob):
    min_bq, mismatch_window_len = P["min_bq"], P["mismatch_window_len"]
    read1, read2 = OB.mask_ends(read1, P["trim_ends"]), OB.mask_ends(read2, P["trim_ends"])
    read1, read2 = OB.mask_indels(read1, P["indels_window"]), OB.mask_indels(read2, P["indels_window"])

    s1, s2 = read1.query_sequence.upper(), read2.query_sequence.upper()
    q1, q2 = np.array(read1.query_qualities, dtype=np.uint8), np.array(read2.query_qualities, dtype=np.uint8)
    aln1, aln2 = read1.get_aligned_pairs(matches_only=True), read2.get_aligned_pairs(matches_only=True)
    len1, len2 = len(aln1), len(aln2)

    if bq_hist is not None:
        OQ.add_duplex(bq_hist, aln1, aln2, s1, s2, q1, q2, ref_arr)

    context_positions = []
    list_qpos1 = []
    list_qpos2 = []
    list_ref1 = []
    list_type = []
    mismatch_list = []
    i = j = 0

    while i < len1 and j < len2:
        qpos1, ref1 = aln1[i]
        qpos2, ref2 = aln2[j]

        if ref1 < ref2:
            i += 1
            continue
        if ref2 < ref1:
            j += 1
            continue

        if q1[qpos1] >= min_bq and q2[qpos2] >= min_bq:
            ref_base = ref_seq[ref1]
            b1 = s1[qpos1]
            b2 = s2[qpos2]
            m1 = (b1 == ref_base)
            m2 = (b2 == ref_base)

            if m1 and m2:
                context_positions += [ref1]
                if z_prob > 0:
                    if random.randint(1, z_prob) == 1:
                        list_qpos1 += [qpos1]
                        list_qpos2 += [qpos2]
                        list_ref1 += [ref1]
                        list_type += ["z"]

            elif m1 ^ m2:
                context_positions += [ref1]
                list_qpos1 += [qpos1]
                list_qpos2 += [qpos2]
                list_ref1 += [ref1]
                list_type += ["m"]
                mismatch_list += [ref1]

            elif b1 == b2:
                if ref_base == "N":
                    i += 1
                    j += 1
                    continue
                context_positions += [ref1]
                list_qpos1 += [qpos1]
                list_qpos2 += [qpos2]
                list_ref1 += [ref1]
                list_type += ["x"]
        i += 1
        j += 1

    if mismatch_list:
        blacklist = set()
        for r in mismatch_list:
            blacklist.update(range(r - mismatch_window_len, r + mismatch_window_len + 1))
        callable_positions = [p for p in context_positions if p not in blacklist]
    else:
        callable_positions = context_positions

    muts = OB.process_mutations(read1, read2, list_qpos1, list_qpos2, list_ref1, list_type,
                                mismatch_list, s1, s2, q1, q2, mismatch_window_len, CONTIG,
                                zmw, ref_seq, aln1, aln2)
    return callable_positions, muts, (q1, q2), (aln1, aln2)


def orig_collapsed(ss_strand, P, CONTIG, ref_seq, ref_arr, bq_hist, z_prob):
    read = OB.mask_indels(OB.mask_ends(ss_strand, P["trim_ends"]), P["indels_window"])
    s = read.query_sequence.upper()
    q = np.array(read.query_qualities, dtype=np.uint8)
    qpos, rpos = OB.aligned_pairs_array(read)
    if bq_hist is not None:
        OQ.add_read(bq_hist, qpos, rpos, s, q, ref_arr)
    context_positions, rec_qpos, rec_rpos, rec_type = OB.walk_collapsed(
        qpos, rpos, s, q, ref_arr, P["min_bq"], z_prob)
    muts = OB.process_mutations_collapsed(read, rec_qpos, rec_rpos, rec_type, s, q,
                                          P["mismatch_window_len"], CONTIG, read.query_name, ref_seq)
    return context_positions, muts, q


def orig_prefix(mut, collapsed):
    """vcflib.write_mut_one's line building (verbatim) for a row WITHOUT the depth."""
    str_mut = [str(x) for x in mut]
    block1 = ":".join(str_mut[6:15])
    if collapsed:
        block2, rest = ".", str_mut[15:]
    else:
        block2, rest = ":".join(str_mut[15:24]), str_mut[24:]
    return "\t".join(x for x in (str_mut[:6] + [block1] + [block2] + rest))


# ---------------------------------------------------------------------------
# Lockstep driver: original pairing + filters, both implementations per molecule
# ---------------------------------------------------------------------------
class Mismatch(Exception):
    pass


def _ref_match_count(read1, read2, q1, q2, ref_arr, min_bq):
    """Independent count of z candidates (both strands match ref, both BQ >= min_bq)."""
    a1 = np.asarray(read1.get_aligned_pairs(matches_only=True), dtype=np.int64).reshape(-1, 2)
    a2 = np.asarray(read2.get_aligned_pairs(matches_only=True), dtype=np.int64).reshape(-1, 2)
    rp, i1, i2 = np.intersect1d(a1[:, 1], a2[:, 1], assume_unique=True, return_indices=True)
    s1 = np.frombuffer(read1.query_sequence.upper().encode(), np.uint8)[a1[i1, 0]]
    s2 = np.frombuffer(read2.query_sequence.upper().encode(), np.uint8)[a2[i2, 0]]
    rb = ref_arr[rp]
    ok = (q1[a1[i1, 0]] >= min_bq) & (q2[a2[i2, 0]] >= min_bq) & (s1 == rb) & (s2 == rb)
    return set(rp[ok].tolist())


def _check_z_rows(texts, positions, cand, P, collapsed):
    for t, p in zip(texts, positions):
        f = t.split("\t")
        if len(f) != 11:
            raise Mismatch(f"malformed record ({len(f)} fields): {t!r}")
        if f[10] != "z":
            continue
        if int(f[1]) != p or int(f[2]) != p + 1 or p not in cand:
            raise Mismatch(f"z row not on a reference-matching position: {t!r}")
        s1 = f[6].split(":")
        if len(s1) != 9 or s1[0] != f[5] or int(s1[1]) < P["min_bq"]:
            raise Mismatch(f"bad z row strand 1: {t!r}")
        if collapsed:
            if f[7] != "." or f[8] != "0" or f[9] != "0":
                raise Mismatch(f"bad collapsed z row: {t!r}")
        else:
            s2 = f[7].split(":")
            if len(s2) != 9 or s2[0] != f[5] or int(s2[1]) < P["min_bq"] or int(f[9]) < 0:
                raise Mismatch(f"bad z row strand 2: {t!r}")


def run_case(case):
    """One (bam, regions, params, mode, z setting) case. Returns a stats dict;
    raises Mismatch (with details) on the first difference."""
    bam_path, ref_path, regions, P, collapsed, zcfg, opts = case
    z_mode = zcfg["mode"]            # "zero" | "one" | "stat"
    z_prob = {"zero": 0, "one": 1, "stat": zcfg.get("z_prob", 0)}[z_mode]
    rng = np.random.default_rng(zcfg.get("seed"))
    max_mol = opts.get("max_mol")
    timing = opts.get("timing", False)
    use_bq = not timing
    st = dict(mol=0, rec=0, rec_m=0, rec_x=0, rec_z=0, callable=0, t_orig=0.0, t_new=0.0,
              z_cand=0, z_obs=0, reads=0)
    rep_o, rep_n = OR.METRICS(), NR.METRICS()
    hist_o = OQ.new_histogram() if use_bq else None
    hist_n = NQ.new_histogram() if use_bq else None

    def compare(where, cal_o, muts_o, res, q_o=None, q_n=None):
        if q_o is not None:
            for a, b in zip(q_o, q_n):
                if a.dtype != np.uint8 or b.dtype != np.uint8 or not np.array_equal(a, b):
                    raise Mismatch(f"{where}: masked qualities differ")
        cal_n = res.callable_pos
        if cal_n.dtype != np.int64 or not np.array_equal(np.asarray(cal_o, dtype=np.int64), cal_n):
            raise Mismatch(f"{where}: callable positions differ: orig {len(cal_o)} new {cal_n.size}")
        if cal_n.size > 1 and not np.all(np.diff(cal_n) > 0):
            raise Mismatch(f"{where}: callable positions not strictly ascending")
        if res.n_callable != len(cal_o):
            raise Mismatch(f"{where}: n_callable differs")
        txt_o = [orig_prefix(m, collapsed) for m in muts_o]
        pos_o = [m[1] for m in muts_o]
        if txt_o != res.rec_text:
            for a, b in itertools.zip_longest(txt_o, res.rec_text):
                if a != b:
                    raise Mismatch(f"{where}: record text differs:\n  orig {a!r}\n  new  {b!r}")
        if res.rec_pos.dtype != np.int64 or pos_o != res.rec_pos.tolist():
            raise Mismatch(f"{where}: record positions differ")
        if use_bq and not np.array_equal(hist_o, hist_n):
            raise Mismatch(f"{where}: BQ histogram differs")
        st["rec"] += len(txt_o)
        for t in txt_o:
            st["rec_" + t[-1]] += 1

    with pysam.AlignmentFile(bam_path, "rb") as bam, pysam.FastaFile(ref_path) as fasta:
        for contig, start, end in regions:
            ref_seq = fasta.fetch(contig).upper()
            ref_arr = np.frombuffer(ref_seq.encode("ascii"), dtype=np.uint8)
            zmw_dict = {}
            for ss_strand in bam.fetch(contig, start, end):
                if max_mol is not None and st["mol"] >= max_mol:
                    break
                rep_o.num_reads += 1
                rep_n.num_reads += 1
                st["reads"] += 1
                d_o = OB.discard_low_qual_read(ss_strand, rep_o, P["min_mapq"], P["min_ec"], P["min_qlen"],
                                               P["max_qlen"], P["max_softclipping"], P["min_sequence_identity"])
                d_n = NB.discard_low_qual_read(ss_strand, rep_n, P["min_mapq"], P["min_ec"], P["min_qlen"],
                                               P["max_qlen"], P["max_softclipping"], P["min_sequence_identity"])
                if d_o != d_n:
                    raise Mismatch(f"{ss_strand.query_name}: discard decision differs")
                if d_o:
                    continue
                ss_strand = NB.remove_useless_tags(ss_strand)

                if collapsed:
                    rep_o.num_zmw += 1
                    rep_n.num_zmw += 1
                    st["mol"] += 1
                    where = f"{contig}:{ss_strand.query_name}"
                    qq_before = np.array(ss_strand.query_qualities, dtype=np.uint8)
                    t0 = time.perf_counter()
                    q_n = NB.masked_qualities(ss_strand, P["trim_ends"], P["indels_window"])
                    res = NB.collapsed_walk(ss_strand, q_n, contig, ref_seq, ref_arr, P["min_bq"],
                                            z_prob, P["mismatch_window_len"], rng, hist_n)
                    t1 = time.perf_counter()
                    if not np.array_equal(qq_before, np.array(ss_strand.query_qualities, dtype=np.uint8)):
                        raise Mismatch(f"{where}: masked_qualities modified the read")
                    if z_mode == "stat":
                        _check_stat(where, contig, ref_seq, ref_arr, ss_strand, None, q_n, None, res, P, st, collapsed)
                        continue
                    cal_o, muts_o, q_o = orig_collapsed(ss_strand, P, contig, ref_seq, ref_arr, hist_o, z_prob)
                    if timing:
                        _ = [orig_prefix(m, True) for m in muts_o]
                    t2 = time.perf_counter()
                    st["t_new"] += t1 - t0
                    st["t_orig"] += t2 - t1
                    rep_o.callable_bps += len(cal_o)
                    rep_n.callable_bps += res.n_callable
                    st["callable"] += res.n_callable
                    compare(where, cal_o, muts_o, res, (q_o,), (q_n,))
                    continue

                strand, zmw = ss_strand.query_name.split("/")[-1], ss_strand.get_tag("zm")
                pair = zmw_dict.get(zmw)
                if pair is None:
                    zmw_dict[zmw] = [ss_strand, None] if strand == "fwd" else [None, ss_strand]
                    continue
                pair[0 if strand == "fwd" else 1] = ss_strand
                if pair[0] is None or pair[1] is None:
                    continue
                rep_o.num_zmw += 1
                rep_n.num_zmw += 1
                st["mol"] += 1
                read1, read2 = pair
                del zmw_dict[zmw]
                where = f"{contig}:zmw {zmw}"

                if not timing:
                    for rd in (read1, read2):     # aligned_pairs_array == get_aligned_pairs
                        qa, ra = NB.aligned_pairs_array(rd)
                        ap = np.asarray(rd.get_aligned_pairs(matches_only=True), dtype=np.int64).reshape(-1, 2)
                        if not (np.array_equal(qa, ap[:, 0]) and np.array_equal(ra, ap[:, 1])):
                            raise Mismatch(f"{where}: aligned_pairs_array differs")
                qq1 = np.array(read1.query_qualities, dtype=np.uint8)
                qq2 = np.array(read2.query_qualities, dtype=np.uint8)
                t0 = time.perf_counter()
                q1 = NB.masked_qualities(read1, P["trim_ends"], P["indels_window"])
                q2 = NB.masked_qualities(read2, P["trim_ends"], P["indels_window"])
                res = NB.duplex_walk(read1, read2, q1, q2, zmw, contig, ref_seq, ref_arr,
                                     P["min_bq"], z_prob, P["mismatch_window_len"], rng, hist_n)
                t1 = time.perf_counter()
                if not (np.array_equal(qq1, np.array(read1.query_qualities, dtype=np.uint8))
                        and np.array_equal(qq2, np.array(read2.query_qualities, dtype=np.uint8))):
                    raise Mismatch(f"{where}: masked_qualities modified a read")
                if z_mode == "stat":
                    _check_stat(where, contig, ref_seq, ref_arr, read1, read2, q1, q2, res, P, st, collapsed)
                    continue
                cal_o, muts_o, q_o, _aln = orig_duplex(read1, read2, zmw, P, contig, ref_seq, ref_arr,
                                                       hist_o, z_prob)
                if timing:
                    _ = [orig_prefix(m, False) for m in muts_o]
                t2 = time.perf_counter()
                st["t_new"] += t1 - t0
                st["t_orig"] += t2 - t1
                rep_o.callable_bps += len(cal_o)
                rep_n.callable_bps += res.n_callable
                st["callable"] += res.n_callable
                compare(where, cal_o, muts_o, res, q_o, (q1, q2))

            for rep in (rep_o, rep_n):
                rep.num_discarded += len(zmw_dict)
                rep.disc_lack_compl += len(zmw_dict)
    if z_mode != "stat" and vars(rep_o) != vars(rep_n):
        raise Mismatch(f"report counters differ:\n{vars(rep_o)}\n{vars(rep_n)}")
    st["report"] = vars(rep_n)
    return st


def _check_stat(where, contig, ref_seq, ref_arr, read1, read2, q1, q2, res, P, st, collapsed):
    """z_prob > 1 (new path only): non-z records + callable == the z_prob=0 run;
    z rows well formed and on reference-matching positions; tally for the rate test."""
    w = P["mismatch_window_len"]
    if collapsed:
        res0 = NB.collapsed_walk(read1, q1, contig, ref_seq, ref_arr, P["min_bq"], 0, w, None, None)
        qpos, rpos = NB.aligned_pairs_array(read1)
        base = np.frombuffer(read1.query_sequence.upper().encode(), np.uint8)[qpos]
        cand = set(rpos[(q1[qpos] >= P["min_bq"]) & (base == ref_arr[rpos])].tolist())
    else:
        res0 = NB.duplex_walk(read1, read2, q1, q2, read1.get_tag("zm"), contig, ref_seq, ref_arr,
                              P["min_bq"], 0, w, None, None)
        cand = _ref_match_count(read1, read2, q1, q2, ref_arr, P["min_bq"])
    if not np.array_equal(res0.callable_pos, res.callable_pos):
        raise Mismatch(f"{where}: z sampling changed the callable positions")
    nz = [(t, p) for t, p in zip(res.rec_text, res.rec_pos.tolist()) if not t.endswith("\tz")]
    if [t for t, _ in nz] != res0.rec_text or [p for _, p in nz] != res0.rec_pos.tolist():
        raise Mismatch(f"{where}: z sampling changed the non-z records")
    if res.rec_pos.size > 1 and not np.all(np.diff(res.rec_pos) > 0):
        raise Mismatch(f"{where}: records not in ascending position order")
    _check_z_rows(res.rec_text, res.rec_pos.tolist(), cand, P, collapsed)
    st["z_cand"] += len(cand)
    st["z_obs"] += sum(t.endswith("\tz") for t in res.rec_text)
    st["callable"] += res.n_callable


# ---------------------------------------------------------------------------
# Whole-contig check against the original call_somatic_mutations()
# ---------------------------------------------------------------------------
def new_contig(bam_path, ref_path, contig, P, collapsed, min_depth):
    """The new functions driven like the original contig worker (z_prob = 0):
    returns (report, muts text as write_mut_one writes it, depth, bq_hist)."""
    rep = NR.METRICS()
    with pysam.FastaFile(ref_path) as fasta:
        ref_seq = fasta.fetch(contig).upper()
        L = fasta.get_reference_length(contig)
    rep.total_length_ref += L
    rep.chrom_name = contig
    ref_arr = np.frombuffer(ref_seq.encode("ascii"), dtype=np.uint8)
    depth = np.zeros(L, dtype=np.uint16)
    hist = NQ.new_histogram()
    rng = np.random.default_rng()
    pos_l, txt_l = [], []
    zmw_dict = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for rd in bam.fetch(contig, 0, L):
            rep.num_reads += 1
            if NB.discard_low_qual_read(rd, rep, P["min_mapq"], P["min_ec"], P["min_qlen"], P["max_qlen"],
                                        P["max_softclipping"], P["min_sequence_identity"]):
                continue
            rd = NB.remove_useless_tags(rd)
            if collapsed:
                rep.num_zmw += 1
                q = NB.masked_qualities(rd, P["trim_ends"], P["indels_window"])
                res = NB.collapsed_walk(rd, q, contig, ref_seq, ref_arr, P["min_bq"], 0,
                                        P["mismatch_window_len"], rng, hist)
            else:
                strand, zmw = rd.query_name.split("/")[-1], rd.get_tag("zm")
                pair = zmw_dict.get(zmw)
                if pair is None:
                    zmw_dict[zmw] = [rd, None] if strand == "fwd" else [None, rd]
                    continue
                pair[0 if strand == "fwd" else 1] = rd
                if pair[0] is None or pair[1] is None:
                    continue
                rep.num_zmw += 1
                r1, r2 = pair
                del zmw_dict[zmw]
                q1 = NB.masked_qualities(r1, P["trim_ends"], P["indels_window"])
                q2 = NB.masked_qualities(r2, P["trim_ends"], P["indels_window"])
                res = NB.duplex_walk(r1, r2, q1, q2, zmw, contig, ref_seq, ref_arr, P["min_bq"], 0,
                                     P["mismatch_window_len"], rng, hist)
            rep.callable_bps += res.n_callable
            depth[res.callable_pos] += 1
            pos_l.append(res.rec_pos)
            txt_l.extend(res.rec_text)
    rep.num_discarded += len(zmw_dict)
    rep.disc_lack_compl += len(zmw_dict)
    pos = np.concatenate(pos_l) if pos_l else np.zeros(0, np.int64)
    d = depth[pos]
    out = "".join(f"{t}\t{x}\n" for t, x, k in zip(txt_l, d.tolist(), (d >= min_depth).tolist()) if k)
    return rep, out, depth, hist


def contig_check(bam_path, ref_path, contig, P, collapsed, min_depth=3):
    with pysam.FastaFile(ref_path) as fasta:
        L = fasta.get_reference_length(contig)
    t0 = time.perf_counter()
    rep_o, muts_o, depth_o, hist_o = O.caller.call_somatic_mutations(
        (contig, 0, L), bam_path, ref_path, P["min_mapq"], P["min_sequence_identity"], P["min_bq"],
        P["min_qlen"], P["max_qlen"], P["trim_ends"], P["indels_window"], P["mismatch_window_len"],
        P["min_ec"], P["max_softclipping"], 0, [], collapsed, min_depth, contig)
    t1 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        frag = os.path.join(tmp, "m.bgz")
        O.vcflib.write_mut_one(frag, muts_o, min_depth, collapsed)
        with gzip.open(frag, "rt") as f:
            text_o = f.read()
    rep_n, text_n, depth_n, hist_n = new_contig(bam_path, ref_path, contig, P, collapsed, min_depth)
    t2 = time.perf_counter()
    rep_n.filtering_params = rep_o.filtering_params
    errs = []
    if vars(rep_o) != vars(rep_n):
        errs.append(f"report differs:\n{vars(rep_o)}\n{vars(rep_n)}")
    if not np.array_equal(depth_o, depth_n) or depth_o.dtype != depth_n.dtype:
        errs.append("depth differs")
    if text_o != text_n:
        errs.append(f"muts text differs ({text_o.count(chr(10))} vs {text_n.count(chr(10))} rows)")
    if not np.array_equal(hist_o, hist_n):
        errs.append("bq histogram differs")
    return errs, dict(rows=text_o.count("\n"), zmw=rep_o.num_zmw, reads=rep_o.num_reads,
                      callable=rep_o.callable_bps, t_orig=t1 - t0, t_new=t2 - t1)


# ---------------------------------------------------------------------------
# Unit checks of the helpers
# ---------------------------------------------------------------------------
def unit_checks():
    rng = np.random.default_rng(12345)
    # _match_blocks: expanded blocks == get_aligned_pairs(matches_only=True) (pysam CIGAR
    # semantics, P consumes query) and == the original aligned_pairs_array (P consumes
    # nothing), on random CIGARs using every op incl. P/H/B/N and 0-length ops
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "c", "LN": 100_000}]})
    n_p = 0
    for it in range(3000):
        k = int(rng.integers(1, 12))
        ops = rng.choice([0, 1, 2, 3, 6, 7, 8, 9], size=k, p=[.2, .1, .1, .05, .1, .2, .2, .05]).tolist()
        cig = [(op, int(rng.integers(0 if it % 7 == 0 else 1, 6))) for op in ops]
        if rng.random() < .3:
            cig = [(4, int(rng.integers(1, 5)))] + cig
        if rng.random() < .2:
            cig = [(5, 3)] + cig + [(5, 2)]
        n_p += any(op == 6 for op, _ in cig)
        qlen = sum(l for op, l in cig if op in (0, 1, 4, 7, 8))
        a = pysam.AlignedSegment(hdr)
        a.query_name = "r"
        a.reference_id = 0
        a.reference_start = int(rng.integers(0, 1000))
        a.query_sequence = "A" * qlen
        a.cigartuples = cig
        ap = np.asarray(a.get_aligned_pairs(matches_only=True), dtype=np.int64).reshape(-1, 2)
        for pad, (eq, er) in ((True, (ap[:, 0], ap[:, 1])), (False, OB.aligned_pairs_array(a))):
            r0s, lens, offs = NB._match_blocks(a.cigartuples, a.reference_start, pad)
            ln = np.asarray(lens, dtype=np.int64)
            rp = (np.repeat(np.asarray(r0s, dtype=np.int64), ln)
                  + np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln))
            qp = rp + np.repeat(np.asarray(offs, dtype=np.int64), ln)
            assert np.array_equal(rp, er) and np.array_equal(qp, eq), (cig, pad)
            assert all(x > 0 for x in lens) and all(np.diff(r0s) > 0)
            if rp.size:
                assert np.array_equal(NB._qpos_of((r0s, lens, offs), rp), qp)
                lo, hi = int(rp[0]) - int(rng.integers(0, 3)), int(rp[-1]) + int(rng.integers(1, 3))
                seq = rng.integers(65, 91, size=int(qp.max()) + 1).astype(np.uint8)
                qual = rng.integers(0, 94, size=seq.size).astype(np.uint8)
                B, Q = NB._project((r0s, lens, offs), lo, hi - lo, seq, qual)
                eB = np.zeros(hi - lo, np.uint8)
                eQ = np.zeros(hi - lo, np.uint8)
                eB[rp - lo], eQ[rp - lo] = seq[qp], qual[qp]
                assert np.array_equal(B, eB) and np.array_equal(Q, eQ)
    assert n_p > 100
    # _window_mask == the original blacklist set, incl. w < 0 and > 32 centers
    for _ in range(3000):
        n = int(rng.integers(1, 300))
        c = np.unique(rng.integers(0, n, size=int(rng.integers(0, 80)))).astype(np.int64)
        w = int(rng.integers(-3, 40))
        bl = set()
        for r in c.tolist():
            bl.update(range(r - w, r + w + 1))
        exp = np.array([i in bl for i in range(n)], dtype=bool)
        assert np.array_equal(NB._window_mask(n, c, w), exp), (n, c, w)
    # _bernoulli_select: z_prob = 1 keeps everything; rate and gaps for z_prob > 1
    assert np.array_equal(NB._bernoulli_select(rng, 1000, 1), np.arange(1000))
    assert NB._bernoulli_select(rng, 0, 5).size == 0
    for zp, n in ((2, 200_000), (37, 2_000_000), (1000, 5_000_000), (1_000_000, 50_000_000)):
        tot = 0
        for _ in range(4):
            s = NB._bernoulli_select(rng, n, zp)
            assert s.size == 0 or (s[0] >= 0 and s[-1] < n and np.all(np.diff(s) > 0))
            tot += s.size
        p = 1 / zp
        mu, sd = 4 * n * p, np.sqrt(4 * n * p * (1 - p))
        assert abs(tot - mu) < 5 * sd + 1, (zp, tot, mu)
    # small n many times (the per-molecule regime): total rate
    zp, n, reps = 50, 3000, 4000
    tot = sum(NB._bernoulli_select(rng, n, zp).size for _ in range(reps))
    mu = reps * n / zp
    assert abs(tot - mu) < 5 * np.sqrt(mu), (tot, mu)
    # positions uniform: kept-index histogram over 10 bins (chi-square, 9 dof)
    s = np.concatenate([NB._bernoulli_select(rng, n, zp) for _ in range(2000)])
    h = np.bincount(s * 10 // n, minlength=10)
    chi2 = (((h - h.mean()) ** 2) / h.mean()).sum()
    assert chi2 < 35, chi2
    # bqlib.add_counts == the histogram add_duplex builds
    h1, h2 = NQ.new_histogram(), NQ.new_histogram()
    v = rng.integers(0, 256, 5000).astype(np.uint8)
    NQ.add_counts(h1, v)
    h2 += np.bincount(v, minlength=256)
    assert np.array_equal(h1, h2)
    print("unit checks: OK", flush=True)


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------
def walk_params(min_bq, trim_ends, indels_window, mmw, filt):
    return dict(filt, min_bq=min_bq, trim_ends=trim_ends, indels_window=indels_window,
                mismatch_window_len=mmw)


def grid_small():
    """tests/data: full required grid + edge values, relaxed read filters."""
    out = [walk_params(b, t, iw, w, FILTER_RELAXED)
           for b, t, iw, w in itertools.product((0, 70, 93), (0, .025, 50), (0, 10), (0, 20, 50))]
    out += [walk_params(*x, FILTER_RELAXED) for x in (
        (0, 1e-5, 10, 20),       # fractional trim rounding to n == 0: whole read zeroed (quirk)
        (93, 1e-5, 0, 0),
        (0, .3, 3, 20), (0, 20000, 1, 5),
        (0, 0, 10, -1), (0, .025, 0, -30), (70, 50, 10, -3),   # negative windows (CLI allows them)
        (94, .025, 10, 20), (-1, 0, 0, 0), (300, 0, 0, 0))]
    out += [walk_params(93, .025, 10, 20, FILTER_DEFAULT)]
    return out


def grid_real():
    """Real data: every required value of each parameter, default and relaxed filters."""
    rows = [(93, .025, 10, 20), (0, 0, 0, 0), (70, 50, 10, 50), (0, .025, 10, 50),
            (93, 50, 0, 0), (70, 0, 0, 20), (0, 50, 0, 20), (93, 0, 10, 50)]
    out = [walk_params(*r, FILTER_DEFAULT) for r in rows]
    out += [walk_params(*r, FILTER_RELAXED) for r in ((93, .025, 10, 20), (0, .025, 10, 20), (0, 0, 0, 50))]
    return out


def parse_region(s, ref):
    with pysam.FastaFile(ref) as fa:
        if ":" in s and s.rsplit(":", 1)[1].replace(",", "").replace("-", "").isdigit():
            c, se = s.rsplit(":", 1)
            a, b = se.replace(",", "").split("-")
            return (c, int(a), int(b))
        return (s, 0, fa.get_reference_length(s))


def _fmt(p):
    keys = ("min_bq", "trim_ends", "indels_window", "mismatch_window_len", "min_mapq")
    return " ".join(f"{k}={p[k]}" for k in keys)


def _run_wrapped(case):
    try:
        if case[5]["mode"] == "contig":
            b, ref, regs, P, collapsed, _, _ = case
            errs, info = contig_check(b, ref, regs[0][0], P, collapsed)
            return case, info, "\n  ".join(errs) if errs else None
        return case, run_case(case), None
    except Mismatch as e:
        return case, None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bam", action="append")
    ap.add_argument("--ref")
    ap.add_argument("--region", action="append")
    ap.add_argument("--grid", choices=("small", "real"), default=None)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--timing", action="store_true", help="only time orig vs new per molecule")
    ap.add_argument("--max-z1-mol", type=int, default=150, help="molecules for the exact z_prob=1 check")
    ap.add_argument("--z-prob", type=int, default=37)
    ap.add_argument("--no-contig-check", action="store_true")
    ap.add_argument("--modes", default="duplex,collapsed")
    a = ap.parse_args()

    if not a.bam:
        a.bam = [os.path.join(HERE, "data", s + ".bam") for s in ("E01", "F01")]
        a.ref = os.path.join(HERE, "data", "reference.fa.gz")
    grid = {"small": grid_small, "real": grid_real}[a.grid or ("small" if not a.region else "real")]()
    modes = [m == "collapsed" for m in a.modes.split(",")]

    unit_checks()
    regions_by_bam = {}
    for b in a.bam:
        if a.region:
            regions_by_bam[b] = [parse_region(r, a.ref) for r in a.region]
        else:
            with pysam.AlignmentFile(b) as f:      # contigs with reads (whole contigs)
                regions_by_bam[b] = [(st.contig, 0, f.get_reference_length(st.contig))
                                     for st in f.get_index_statistics() if st.mapped > 0]

    cases = []
    if a.timing:
        for b in a.bam:
            for collapsed in modes:
                P = walk_params(93, .025, 10, 20, FILTER_DEFAULT)
                cases.append((b, a.ref, regions_by_bam[b], P, collapsed, {"mode": "zero"}, {"timing": True}))
    else:
        for b in a.bam:
            for collapsed in modes:
                for P in grid:
                    cases.append((b, a.ref, regions_by_bam[b], P, collapsed, {"mode": "zero"}, {}))
                for P in (grid[0], grid[1], grid[-1]):
                    cases.append((b, a.ref, regions_by_bam[b], P, collapsed, {"mode": "one"},
                                  {"max_mol": a.max_z1_mol}))
                for P in (grid[0], grid[1]):
                    for reg in regions_by_bam[b]:        # stat mode: one contig per case
                        cases.append((b, a.ref, [reg], P, collapsed,
                                      {"mode": "stat", "z_prob": a.z_prob, "seed": len(cases)}, {}))
                if not a.no_contig_check:
                    # whole contigs, through the original call_somatic_mutations (with --region
                    # on a big BAM this processes the WHOLE contig: give it a sub-BAM instead)
                    for P in (grid[0], grid[-1] if a.grid == "real" else grid[1]):
                        for c, _, _ in regions_by_bam[b]:
                            with pysam.FastaFile(a.ref) as fa:
                                L = fa.get_reference_length(c)
                            cases.append((b, a.ref, [(c, 0, L)], P, collapsed, {"mode": "contig"}, {}))

    # slowest first (whole-contig checks run the original contig worker end to end)
    cases.sort(key=lambda c: c[5]["mode"] != "contig")
    n_fail = 0
    t_start = time.time()
    ctx = mp.get_context("fork")
    agg = {}
    with ctx.Pool(a.procs) as pool:
        for case, st, err in pool.imap_unordered(_run_wrapped, cases):
            b, _, regs, P, collapsed, zcfg, opts = case
            tag = f"[{os.path.basename(b)} {'collapsed' if collapsed else 'duplex'} z={zcfg['mode']} {_fmt(P)}]"
            if err:
                n_fail += 1
                print(f"FAIL {tag}\n  {err}", flush=True)
                continue
            if zcfg["mode"] == "contig":
                print(f"ok   {tag} whole-contig {regs[0][0]}: {st['reads']} reads, {st['zmw']} molecules, "
                      f"{st['rows']} rows, callable {st['callable']} "
                      f"(orig {st['t_orig']:.1f}s, new {st['t_new']:.1f}s)", flush=True)
            elif zcfg["mode"] == "stat":
                k = (b, collapsed)
                c, o = agg.get(k, (0, 0))
                agg[k] = (c + st["z_cand"], o + st["z_obs"])
                print(f"ok   {tag} {regs[0][0]}: z rows {st['z_obs']} / candidates {st['z_cand']}", flush=True)
            elif opts.get("timing"):
                n = max(st["mol"], 1)
                print(f"TIMING {tag}: {st['mol']} molecules, {st['rec']} records "
                      f"(m {st['rec_m']} x {st['rec_x']} z {st['rec_z']}); "
                      f"orig {st['t_orig']:.2f}s ({1e3 * st['t_orig'] / n:.3f} ms/mol)  "
                      f"new {st['t_new']:.2f}s ({1e3 * st['t_new'] / n:.3f} ms/mol)  "
                      f"speed-up x{st['t_orig'] / max(st['t_new'], 1e-9):.1f}", flush=True)
            else:
                print(f"ok   {tag}: {st['mol']} molecules, {st['rec']} records "
                      f"(m {st['rec_m']} x {st['rec_x']} z {st['rec_z']}), callable {st['callable']}",
                      flush=True)

    # z rate: total z rows ~ Binomial(candidates, 1/z_prob)
    for (b, collapsed), (c, o) in sorted(agg.items()):
        p = 1 / a.z_prob
        mu, sd = c * p, np.sqrt(c * p * (1 - p))
        zsc = (o - mu) / sd if sd > 0 else 0.0
        ok = abs(zsc) < 4.5
        n_fail += not ok
        print(f"{'ok  ' if ok else 'FAIL'} z rate {os.path.basename(b)} {'collapsed' if collapsed else 'duplex'}: "
              f"{o} z rows / {c} candidates = 1/{c / max(o, 1):.2f} (expected 1/{a.z_prob}; z-score {zsc:+.2f})",
              flush=True)

    print(f"\n{len(cases)} cases, {n_fail} failure(s), {time.time() - t_start:.0f}s", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
