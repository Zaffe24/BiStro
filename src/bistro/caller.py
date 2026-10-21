"""
`BiStro preprocess` — extracts candidate mutations from CCS reads. For each
contig (in parallel), walks every ZMW's aligned fwd/rev CCS read pair,
classifies each jointly-covered position as reference, single-strand
mismatch, or double-strand (de novo) substitution, tracks per-position read
depth, and writes the resulting mutation and context records to per-contig
BED fragments that main() then merges into the sample's .muts.bed.gz /
.context.bed.gz outputs. The `somatic` subcommand (sharelib.py) later decides
which of these double-strand candidates (TYPE x) are de novo or germline.
On one contig the walk also records a duplex base-quality profile (bqlib.py).

With --collapsed, reads are not paired: every read that passes the filters is
its own molecule and is walked on its own (bamlib.collapsed_walk), so there are
no single-strand mismatches (TYPE m) and nothing is blacklisted around them.

Efficiency refactor (results identical to the original):
* One worker per contig does everything for that contig: the walk (vectorised,
  bamlib.duplex_walk / collapsed_walk), the raw-read coverage that mosdepth used
  to compute in a second pass over the BAM (covlib.RawCoverage, fed with the reads
  the worker fetches anyway), the low-complexity + high-coverage mask (applied in
  memory instead of `bedtools subtract` over the assembled files), and writing its
  .muts / .context fragments (vectorised text formatting, tabix offsets recorded
  while writing). Only small summaries go back to the parent; no per-base array
  is pickled, and there is no second pool.
* The .context.bed.gz index is written from those offsets (no re-scan by tabix).
* Contigs are dispatched largest first so the pool stays busy.
"""

import os
import time
import multiprocessing as mp

import numpy as np
import pysam
pysam.set_verbosity(0)

from . import reportlib, utilib, bamlib, vcflib, contiglib, covlib, bqlib, fastio

# Context rows are formatted/written in blocks of this many reference bases (bounds temporaries).
CTX_BLOCK_BP = 1 << 23
_N = ord("N")


def _write_context_fragment(path, contig, start, end, depth, ref_arr, min_depth, mask_s, mask_e):
    """Context rows of one contig, exactly as the original write_context_one + `bedtools
    subtract -A` produced them: positions start < p < end-1 with depth >= min_depth, a
    reference triplet without 'N', and p outside the (merged, sorted) mask intervals.
    Returns the fragment's FragmentIndex (None if no row)."""
    w = fastio.IndexedBgzfWriter(path, contig)
    try:
        lo_all, hi_all = start + 1, end - 1                # start < p < end - 1
        for a in range(lo_all, max(lo_all, hi_all), CTX_BLOCK_BP):
            b = min(a + CTX_BLOCK_BP, hi_all)
            d = depth[a:b]
            keep = d >= min_depth
            if not keep.any():
                continue
            # triplet ref[p-1:p+2] must not contain 'N' (other IUPAC letters pass, as before)
            keep &= ref_arr[a - 1:b - 1] != _N
            keep &= ref_arr[a:b] != _N
            keep &= ref_arr[a + 1:b + 1] != _N
            # mask intervals overlapping [a, b)
            i0 = np.searchsorted(mask_e, a, side="right")
            i1 = np.searchsorted(mask_s, b, side="left")
            for s, e in zip(mask_s[i0:i1].tolist(), mask_e[i0:i1].tolist()):
                keep[max(s, a) - a:min(e, b) - a] = False
            idx = np.flatnonzero(keep)
            if idx.size == 0:
                continue
            pos = idx + a
            tri = np.stack((ref_arr[pos - 1], ref_arr[pos], ref_arr[pos + 1]), axis=1)
            data, offs = fastio.format_context_lines(contig, pos, tri, d[idx])
            w.write_records(data, pos, offs)
    finally:
        fi = w.close()
    return fi


def call_somatic_mutations(task):
    """Per-contig worker. `task` = (index, region_tuple, params dict, lcr (starts, ends)).

    Returns a small dict: report, fragment paths, context FragmentIndex, bq histogram (only
    on the BQ-profile contig), raw-coverage mean / cap and the high-coverage intervals."""
    k, region_tuple, P, lcr = task
    min_mapq = P["min_mapq"]; min_sequence_identity = P["min_sequence_identity"]
    min_bq = P["min_bq"]; min_qlen = P["min_qlen"]; max_qlen = P["max_qlen"]
    trim_ends = P["trim_ends"]; indels_window = P["indels_window"]
    mismatch_window_len = P["mismatch_window_len"]; min_ec = P["min_ec"]
    max_softclipping = P["max_softclipping"]; z_prob = P["z_prob"]
    check_mem_usage = P["check_mem_usage"]; collapsed = P["collapsed"]
    min_depth = P["min_depth"]; bq_profile_contig = P["bq_profile_contig"]

    wrk = mp.current_process().name
    report = reportlib.METRICS()
    report.filtering_params = (min_mapq,
                           min_sequence_identity,
                           min_bq,
                           min_qlen,
                           max_qlen,
                           trim_ends,
                           indels_window,
                           mismatch_window_len,
                           min_ec,
                           max_softclipping,
                           min_depth,
                           "collapsed" if collapsed else "duplex"   # read_mode: keep last (METRICS.is_collapsed)
                           )

    CONTIG, START, END = region_tuple
    region_len = END - START
    report.total_length_ref += region_len
    report.chrom_name = CONTIG

    depth = np.zeros(region_len, dtype=np.uint16)
    zmw_dict = dict()
    rec_pos = []            # per molecule: int64 positions of its records
    rec_text = []           # record text up to TYPE (the DEPTH column is appended at write time)
    # duplex BQ profile: only the worker handling bq_profile_contig records it
    bq_hist = bqlib.new_histogram() if CONTIG == bq_profile_contig else None
    # one OS-seeded generator per task (z rows); never the global numpy state shared by forked workers
    rng = np.random.default_rng()
    raw = covlib.RawCoverage()

    with pysam.AlignmentFile(P["bam"], "rb") as bam_file, \
    pysam.FastaFile(P["reference"]) as fasta:
        ref_seq = fasta.fetch(CONTIG).upper()
        ref_arr = np.frombuffer(ref_seq.encode("ascii"), dtype=np.uint8)
        bam_len = bam_file.get_reference_length(CONTIG)

        for ss_strand in bam_file.fetch(CONTIG, START, END):
            # raw-read coverage (mosdepth -x -Q 20 equivalent) sees every fetched read
            raw.add(ss_strand)
            # ------------------------------------------------
            # Applying quality filters
            # ------------------------------------------------
            report.num_reads +=1

            if bamlib.discard_low_qual_read(ss_strand, report, min_mapq, min_ec,
                                     min_qlen, max_qlen, max_softclipping,
                                     min_sequence_identity):
                continue
            # ------------------------------------------------
            # decrease memory burden before storing reads
            # ------------------------------------------------
            ss_strand = bamlib.remove_useless_tags(ss_strand)

            if collapsed:
                # Collapsed mode: every read is its own molecule (no fwd/rev pairing)
                report.num_zmw += 1
                if report.num_zmw in check_mem_usage:
                    utilib.cprint(f"[{wrk}]\tMEM OCCUPIED: {utilib.get_stats()}\tRead counter for {CONTIG}: {report.num_zmw}")
                q = bamlib.masked_qualities(ss_strand, trim_ends, indels_window)
                res = bamlib.collapsed_walk(ss_strand, q, CONTIG, ref_seq, ref_arr, min_bq, z_prob,
                                            mismatch_window_len, rng, bq_hist)
            else:
                strand, zmw = ss_strand.query_name.split("/")[-1], ss_strand.get_tag("zm")
                # ------------------------------------------------
                # Core duplex-pairing logic
                # ------------------------------------------------
                pair = zmw_dict.get(zmw)
                if pair is None:
                    zmw_dict[zmw] = [ss_strand, None] if strand == "fwd" else [None, ss_strand]
                    continue
                pair[0 if strand == "fwd" else 1] = ss_strand
                if pair[0] is None or pair[1] is None:
                    continue
                report.num_zmw += 1

                if report.num_zmw in check_mem_usage:
                    utilib.cprint(f"[{wrk}]\tMEM OCCUPIED: {utilib.get_stats()}\tZMW counter for {CONTIG}: {report.num_zmw}")

                read1, read2 = pair
                del zmw_dict[zmw]
                # mask read ends + indel neighbourhoods (BQ -> 0), then walk the duplex
                q1 = bamlib.masked_qualities(read1, trim_ends, indels_window)
                q2 = bamlib.masked_qualities(read2, trim_ends, indels_window)
                res = bamlib.duplex_walk(read1, read2, q1, q2, zmw, CONTIG, ref_seq, ref_arr,
                                         min_bq, z_prob, mismatch_window_len, rng, bq_hist)

            # positions within mismatch_window_len of a ss-mismatch are already excluded
            report.callable_bps += res.n_callable
            depth[res.callable_pos] += 1 ## update count of all investigable genome positions
            if res.rec_text:
                rec_pos.append(res.rec_pos)
                rec_text.extend(res.rec_text)

        report.num_discarded += len(zmw_dict.keys())
        report.disc_lack_compl += len(zmw_dict.keys())
        del zmw_dict

    # ---- raw-read coverage cap + mask (LCR ∪ high coverage) for this contig ----
    raw_mean, cap, hc_s, hc_e = covlib.contig_raw_coverage(raw, bam_len, min_depth)
    del raw
    mask_s, mask_e = covlib.merge_intervals(lcr, (hc_s, hc_e), length=region_len)

    # ---- .muts fragment: rows with depth >= min_depth outside the mask, DEPTH appended ----
    out_dir, sample = P["out_dir"], P["sample"]
    mut_frag = os.path.join(out_dir, f".{sample}.muts.{k:05d}.frag.bgz")
    ctx_frag = os.path.join(out_dir, f".{sample}.context.{k:05d}.frag.bgz")
    pos = np.concatenate(rec_pos) if rec_pos else np.zeros(0, np.int64)
    d = depth[pos]
    keep = (d >= min_depth) & ~covlib.in_intervals(pos, mask_s, mask_e)
    n_muts = int(np.count_nonzero(keep))
    with pysam.BGZFile(mut_frag, "wb") as bed:
        step = 200_000
        for a in range(0, len(rec_text), step):
            sl = slice(a, a + step)
            bed.write("".join(f"{t}\t{x}\n" for t, x, kk in
                              zip(rec_text[sl], d[sl].tolist(), keep[sl].tolist()) if kk).encode())
    del rec_text, rec_pos, pos, d, keep

    # ---- .context fragment (indexed) ----
    ctx_fi = _write_context_fragment(ctx_frag, CONTIG, START, END, depth, ref_arr, min_depth,
                                     mask_s, mask_e)

    utilib.cprint(f"[{wrk}]\tMEM OCCUPIED: {utilib.get_stats()}\tZMW counter for {CONTIG}: {report.num_zmw}")
    utilib.cprint(f"{report}\n", color="yellow")

    return dict(k=k, report=report, mut_frag=mut_frag, n_muts=n_muts, ctx_frag=ctx_frag,
                ctx_fi=ctx_fi, bq_hist=bq_hist, raw_mean=raw_mean, cap=cap,
                hc=(CONTIG, hc_s, hc_e))


def main(bam,
            reference,
            out_dir,
            sample,
            region,
            exclude,
            nproc,
            min_mapq,
            min_sequence_identity,
            min_bq,
            min_qlen,
            max_qlen,
            trim_ends,
            indels_window,
            mismatch_window_len,
            min_ec,
            max_softclipping,
            z_prob,
            check_mem_usage,
            collapsed,
            min_depth,
            subtract_bed,
            bq_profile_contig=None):

    checkpoints = range(0, 1000_000, check_mem_usage) if check_mem_usage>0 else []
    clock_start = time.time() / 60

    # [(contig, start, end), ...] to call mutations on
    contigs_list = contiglib.get_contig_list(reference, region, exclude)
    contigs = [c for c, _, _ in contigs_list]
    if region:
        utilib.cprint(f"Calling Mutations only on Chromosomes: {region}.", color="green")
    if exclude:
        utilib.cprint(f"Excluding Chromosomes: {exclude}", color="green")
    if collapsed:
        utilib.cprint("Collapsed mode: every read is treated as an independent molecule (no ZMW duplex pairing).", color="green")

    # Duplex BQ profile defaults to the first analysed contig (FASTA/.fai order).
    if bq_profile_contig is None:
        bq_profile_contig = contigs[0]
    elif bq_profile_contig not in contigs:
        utilib.cprint(f"ERROR: --bq_profile_contig '{bq_profile_contig}' is not among the analysed contigs (check --region / --exclude).", color="red")
        utilib.exit(1)

    # Low-complexity regions, parsed once and handed to each worker for its own contig only.
    lcr = covlib.parse_lcr_bed(subtract_bed or None, contigs)

    params = dict(bam=bam, reference=reference, out_dir=out_dir, sample=sample,
                  min_mapq=min_mapq, min_sequence_identity=min_sequence_identity,
                  min_bq=min_bq, min_qlen=min_qlen, max_qlen=max_qlen, trim_ends=trim_ends,
                  indels_window=indels_window, mismatch_window_len=mismatch_window_len,
                  min_ec=min_ec, max_softclipping=max_softclipping, z_prob=z_prob,
                  check_mem_usage=checkpoints, collapsed=collapsed, min_depth=min_depth,
                  bq_profile_contig=bq_profile_contig)
    tasks = [(k, rt, params, lcr[rt[0]]) for k, rt in enumerate(contigs_list)]
    # largest contigs first: the longest task starts immediately
    tasks.sort(key=lambda t: -(t[1][2] - t[1][1]))

    results = [None] * len(tasks)
    ctx = mp.get_context("forkserver")
    with ctx.Pool(processes=max(1, min(nproc, len(tasks)))) as pool:
        for r in pool.imap_unordered(call_somatic_mutations, tasks, chunksize=1):
            results[r["k"]] = r       # back to contigs_list order
    del tasks, lcr

    # ---- merge ----
    reports = [r["report"] for r in results]
    bq_hist = next(r["bq_hist"] for r in results if r["bq_hist"] is not None)
    merged_report = reportlib.merge_reports(reports)
    utilib.cprint(merged_report)
    print("")

    bqlib.write_bq_profile(bq_hist, out_dir, sample, bq_profile_contig, min_bq, collapsed)

    # ---- assemble outputs from the per-contig fragments ----
    # mut-bed: prepend the shared header, concat fragments (no tabix, as before).
    mut_file = os.path.join(out_dir, f"{sample}.muts.bed.gz")
    raw_unit = "read" if collapsed else "single-strand read"
    max_depth_hdr = f"per-contig raw ({raw_unit}) mean+{covlib.SIGMA}*sqrt(mean) @MAPQ{covlib.COV_MAPQ} (high-coverage regions subtracted)"
    mut_header = "\n".join(vcflib.mut_bed_header(sample, reference, merged_report.filtering_params + (max_depth_hdr,) ))
    uno = time.time() / 60
    fastio.assemble_bgzf(mut_file, mut_header.encode(), [(r["mut_frag"], None) for r in results],
                         index=False)
    dos = time.time() / 60
    utilib.cprint(f"Writing {os.path.basename(mut_file)} completed. Time elapsed {round(dos - uno,2)} min.")

    vcflib.write_report(out_dir, sample, merged_report)

    # context: two-line header, concat fragments, .tbi from the offsets recorded while writing.
    context_file = os.path.join(out_dir, f"{sample}.context.bed.gz")
    header_ctx = f"##Minimum depth: {min_depth}; Maximum depth: per-contig raw ({raw_unit}) mean+{covlib.SIGMA}*sqrt(mean) @MAPQ{covlib.COV_MAPQ} (high-coverage regions subtracted)\n##CONTIG\tSTART\tEND\tREF\tDEPTH\n"
    uno = time.time() / 60
    fastio.assemble_bgzf(context_file, header_ctx.encode(),
                         [(r["ctx_frag"], r["ctx_fi"]) for r in results], index=True)
    dos = time.time() / 60
    utilib.cprint(f"Writing {os.path.basename(context_file)} completed. Time elapsed {round(dos - uno,2)} min.")

    # ---- excessive-coverage report artefacts ----------------------------------
    # Raw-read depth (MAPQ>=20) per contig; intervals reaching round(mean + 4*sqrt(mean))
    # are collapsed-repeat / mis-mapping artefacts. Together with --low_complexity_regions
    # they were already removed from the .muts / .context rows by the workers.
    contig_means = {r["hc"][0]: r["raw_mean"] for r in results if r["raw_mean"] is not None}
    max_thr = {r["hc"][0]: r["cap"] for r in results if r["cap"] is not None}
    hc_by_contig = {r["hc"][0]: r["hc"] for r in results}
    covlib.write_highcov_bed(out_dir, sample,
                             [hc_by_contig[c] for c in covlib.bam_header_order(bam, contigs)])
    utilib.cprint(f"Masked low-complexity + high-coverage regions from "
                  f"{os.path.basename(mut_file)} / {os.path.basename(context_file)}.", color="green")

    vcflib.write_merged_coverages(merged_report, out_dir, sample,
                                  raw_means=contig_means, max_depth=max_thr)

    clock_end = time.time() / 60
    utilib.cprint(f"BiStro took {round(clock_end - clock_start, 2)} minutes to pre-process candidate mutations in sample {sample}", color="green")
