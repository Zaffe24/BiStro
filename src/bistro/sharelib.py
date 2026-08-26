"""
`BiStro somatic` — cross-sample post-processing of `preprocess`'s
.muts.bed.gz / .context.bed.gz outputs. Restricts every sample to the genomic
space callable in ALL samples, reclassifies de novo (DNM) calls as
germline/false-positive when they recur across samples, across reads of the
same ZMW, or have raw read support in other samples' BAMs, and reports each
sample's resulting somatic mutation rate with a Poisson confidence interval.
"""

import gzip
import os
import tempfile
import pandas as pd
import subprocess
import pysam
from functools import partial
from collections import defaultdict
from . import vcflib, utilib, contiglib
import multiprocessing as mp
from scipy.stats import chi2
import time
import numpy as np
pysam.set_verbosity(0)

# Column order of .muts.bed.gz files (no header line in the file)
COLS = [
    "contig", "start", "end", "zmw", "triplet", "ref",
    "strand_1", "strand_2", "nob", "n_mis", "type", "depth",
]


# Mutation type codes from the bistro caller
ANN_DICT = {"z":"RDM",   # random non-mutated site
            "m":"MSM",   # mismatch (QB artefact)
            "g":"GRM",   # germline variant
            "d":"DNM"}   # de novo / somatic mutation




def _sort_bed(bed_path, genome_fai):
    # Sort a BED file in reference chromosome order using Python.
    # bedtools sort SIGABRT'd on bgzipped input with ## headers; Python avoids that entirely.
    # Reads chromosome order from the FAI, sorts by (chrom_index, start), strips comment lines.
    # Returns path to an uncompressed temp file; caller is responsible for deleting it.
    chrom_idx = {}
    with open(genome_fai) as fai:
        for i, line in enumerate(fai):
            chrom_idx[line.split("\t")[0]] = i

    rows = []
    with gzip.open(bed_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split("\t", 3)
            rows.append(((chrom_idx.get(fields[0], len(chrom_idx)), int(fields[1])), line))

    rows.sort(key=lambda x: x[0])

    out = tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False)
    try:
        for _, line in rows:
            out.write(line)
        out.close()
        return out.name
    except Exception:
        out.close()
        try: os.unlink(out.name)
        except OSError: pass
        raise


def subset_mutations_by_shared_intervals(mut_bed, reference_bed, genome_file):
    # Streams bedtools output directly into pandas to avoid a double-copy through a temp file.
    # -sorted requires mut_bed to be coordinate-sorted (caller must ensure this).
    # -g is required so bedtools tolerates contigs present in -b but absent from -a.
    proc = subprocess.Popen(
        ["bedtools", "intersect", "-sorted", "-g", genome_file, "-a", mut_bed, "-b", reference_bed],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        try:
            df = pd.read_csv(proc.stdout, sep="\t", header=None, names=COLS)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=COLS)
    finally:
        proc.stdout.close()
        proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
    # Tuple key used for set-lookup in get_shared_muts; dropped before writing output.
    # Includes the alternate allele (ALT = first colon-sub-field of STRAND_1) so that two
    # samples only count as sharing a site when they carry the same substitution.
    alt = df.strand_1.astype(str).str.split(":").str[0]
    df["contig_start"] = list(zip(df.contig.astype(str), df.start, alt))
    return df


def get_positions(file_path):
    # Build a position lookup set for one sample.
    # Keeps only high-confidence somatic candidates: not random (z) or mismatch (m) type,
    # and zero mismatches in the QB window (n_mis == 0).
    # Used by get_shared_muts to detect the same position appearing in multiple samples.
    positions = set()
    with gzip.open(file_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[10].rstrip() not in ("m", "z") and int(fields[9]) == 0:
                alt = fields[6].split(":")[0]
                positions.add((fields[0], int(fields[1]), alt))
    return positions


def get_shared_muts(sample_df, sample_mut, all_positions):
    # For each mutation, count how many OTHER samples carry a call at the same position.
    # sum() over boolean Series yields an integer Series (0 = private, N = shared in N others).
    sample_df["shared"] = sum(
        sample_df.contig_start.isin(positions)
        for path, positions in all_positions.items()
        if path != sample_mut
    )
    sample_df = sample_df.drop(columns="contig_start")
    return sample_df


def get_zmw_with_multiple_dnm(df,max_muts_per_duplex):

    only_dnm = df.loc[df.annotation == "DNM"]
    counts = only_dnm.groupby("zmw").count()
    zmw_discarded = counts.loc[counts["annotation"] >= max_muts_per_duplex].index
    return zmw_discarded.tolist()


def flag_zmw_with_multiple_dnm(df, max_muts_per_duplex):
    # Flag putative de novo mutations found in reads carrying a suspiciously high no. of DNMs.
    #
    # Runs BEFORE the pileup-based check (flag_fp_denovo), as a cheap early filter: a duplex
    # read reporting an implausible number of independent DNMs is itself evidence of an
    # artefact (e.g. a chimeric or mismapped read), so it is worth flagging ahead of the more
    # expensive pileup. Trade-off: since this runs before flag_fp_denovo has demoted its own
    # rejects, a ZMW's DNM tally can include calls that the pileup would separately reject as
    # FPD, which can pull that ZMW's genuine DNMs down with them.
    to_flag = get_zmw_with_multiple_dnm(df, max_muts_per_duplex)
    df.loc[(df["zmw"].isin(to_flag)) & (df["annotation"] == ANN_DICT["d"]), "annotation"] = "FPD"
    return df


def annotate_muts(df, thr_shared):
    # Default annotation from type code; then upgrade de novo (d) mutations that appear
    # in >= thr_shared other samples to GRM — recurrence across organisms implies germline.
    # The single-sample path has no "shared" column: there is nothing to be shared with.
    df["annotation"] = df["type"].map(ANN_DICT)
    if "shared" in df.columns:
        df.loc[(df.type == "d") & (df.shared >= thr_shared),"annotation"] = ANN_DICT["g"]

    ### flag de novo mutations with any mismatch in the QB window (n_mis > 0) as false positives
    df.loc[(df["n_mis"] > 0) & (df["annotation"] == ANN_DICT["d"]), "annotation"] = "FPD"
    return df


# Index into the [A, C, G, T] count rows returned by count_alleles.
BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def count_alleles(bam_path, positions_by_contig, min_mapq, min_baseq, max_depth=1_000_000):
    # Count high-quality A/C/G/T reads at each requested 1-based position.
    # One pileup call per position: truncate=True yields only the column we asked for, so
    # nothing is walked between distant mutations. Filtering is done per-read rather than
    # through pileup()'s own min_base_quality / min_mapping_quality so that
    # secondary/supplementary/duplicate exclusion is explicit and cannot drift with pysam's
    # default stepper.
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


def flag_fp_denovo(df, other_bams, min_mapq, min_baseq, min_alt_support, min_alt_samples):
    # Final safeguard against false-positive de novo calls.
    #
    # A true somatic mutation is private to one organism, so its ALT allele should be absent
    # from every other sample's raw reads. get_shared_muts only sees positions the caller
    # actually *called* elsewhere; this check goes underneath that detection threshold by
    # counting ALT-supporting reads directly in the other samples' BAMs. A DNM whose ALT is
    # supported by >= min_alt_support reads in >= min_alt_samples other BAMs is reclassified
    # as FPD (germline variant, or a recurrent artefact at a systematically error-prone site).
    #
    # min_alt_samples is thr_shared: both ask "in how many other samples must this variant
    # appear before we stop believing it is private?" — one counts called sites, one counts
    # raw read support.
    #
    # other_bams: [(sample_id, bam_path), ...]. The per-sample A:C:G:T counts behind the
    # decision are kept in the COUNTS column so a flagged call can be inspected afterwards.
    # "." marks rows where no pileup was done (non-DNM rows, or --bams not supplied).
    df["counts"] = "."
    if not other_bams:
        return df

    dnm = df.loc[df.annotation == ANN_DICT["d"]]
    if dnm.empty:
        return df

    # BED end column holds the 1-based coordinate of the substituted base.
    contigs = dnm.contig.astype(str)
    positions = dnm.end.astype(int)
    positions_by_contig = defaultdict(set)
    for contig, pos in zip(contigs, positions):
        positions_by_contig[contig].add(pos)

    # One [A,C,G,T] table per other sample, in the order they appear in --bams.
    per_sample = [count_alleles(bam, positions_by_contig, min_mapq, min_baseq)
                  for _, bam in other_bams]

    # A single ALT read is the strictest this check can be, so never let a 0 threshold
    # make every base "supported" and flag every de novo.
    support_thr = max(1, min_alt_support)
    zeros = [0, 0, 0, 0]

    sites = list(zip(contigs, positions))
    alts = dnm.strand_1.astype(str).str.split(":").str[0]

    # COUNTS: one A:C:G:T field per other sample, comma-separated, order given in the header.
    df.loc[dnm.index, "counts"] = [
        ",".join(":".join(map(str, counts.get(site, zeros))) for counts in per_sample)
        for site in sites
    ]

    is_fp = []
    for site, alt in zip(sites, alts):
        idx = BASE_INDEX.get(alt)
        supporting = 0 if idx is None else sum(
            counts.get(site, zeros)[idx] >= support_thr for counts in per_sample
        )
        is_fp.append(supporting >= min_alt_samples)
    df.loc[dnm.index[is_fp], "annotation"] = "FPD"
    return df


def _bed_has_records(bed_path):
    # True if the (optionally bgzipped) BED has at least one non-comment data line.
    # Short-circuits on the first record, so it does not read the whole file.
    with gzip.open(bed_path, "rt") as f:
        return any(not line.startswith("#") for line in f)


def subset_context_by_shared_intervals(query_bed, reference_bed_path):
    # Streams bedtools intersect line by line; never loads the full context into RAM.
    # stdout.close() sends EOF/SIGPIPE to bedtools if the caller breaks early.
    #
    # Guard against bedtools 2.31.1's empty-input quirk: when -a is an empty bgzipped file
    # it echoes the raw BGZF EOF block to stdout instead of emitting zero records, and
    # text=True then chokes trying to UTF-8 decode the gzip magic (0x1f 0x8b). An empty
    # query intersects to nothing anyway, so yield nothing without invoking bedtools.
    if not _bed_has_records(query_bed):
        return
    proc = subprocess.Popen(
        ["bedtools", "intersect", "-sorted", "-a", query_bed, "-b", reference_bed_path],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        yield from proc.stdout
    finally:
        proc.stdout.close()  # closes read-end of pipe; signals bedtools if it's still writing
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)


def get_sample_id(mut_bed):
    return os.path.basename(mut_bed).split(".")[0]

def poisson_rate_cl(k, exposure, alpha=0.05):
    # exact Poisson confidence interval via chi-squared quantiles.
    if exposure > 0:
        lower = 0.0 if k == 0 else chi2.ppf(alpha/2, 2*k) / 2
        upper = chi2.ppf(1-alpha/2, 2*(k+1)) / 2
        return lower/exposure, upper/exposure
    return 0, 0


def wrap_worker(mut_bed, intersection, samples_id, all_positions, thr_shared, reference,
                max_muts_per_duplex, bam_by_mut, min_mapq, min_baseq, min_alt_support):
    wrk = mp.current_process().name
    sample_id = get_sample_id(mut_bed)

    uno = time.time() / 60
    utilib.cprint(f"[MEM][{wrk}][{sample_id}] worker start: {utilib.get_stats()} GB", color="yellow")

    context_bed = mut_bed.replace("muts.bed.gz", "context.bed.gz")
    sample_context_out=mut_bed.replace("muts.bed.gz", "shared.context.bed.gz")
    sample_mut_out=mut_bed.replace("muts.bed.gz", "shared.muts.bed.gz")

    # Step 1: filter this sample's context to the shared genomic space and sum the denominator.
    # col 4 of context.bed.gz holds the base count for that interval (rate denominator).
    denominator = 0
    with pysam.BGZFile(sample_context_out, "wb") as out_context:
        write=out_context.write
        write(f"##INTERVALS FROM {sample_id} SHARED ACROSS: {" ".join(samples_id)}\n".encode())
        write(b"".join(vcflib.get_context_header(context_bed)))
        for line in subset_context_by_shared_intervals(context_bed, intersection):
            write(line.encode())
            fields = line.split("\t")
            denominator += int(fields[4])

    # tabix index required so the shared context file can be used as -b in bedtools below
    pysam.tabix_index(sample_context_out, preset="bed", force=True)

    dos = time.time() / 60
    utilib.cprint(f"[MEM][{wrk}][{sample_id}] after context filter: {utilib.get_stats()} GB | denominator={denominator:,} bp | time={round(dos-uno,2)} min", color="yellow")

    # Step 2: restrict mutations to the shared context, annotate, and count true de novos.
    # mut_bed is not coordinate-sorted, so sort it first to enable -sorted in bedtools intersect.
    # Use the reference .fai to enforce reference chromosome order, matching the context/intersection files.
    sorted_mut = _sort_bed(mut_bed, reference + ".fai")
    try:
        mut_df = subset_mutations_by_shared_intervals(sorted_mut, sample_context_out, reference + ".fai")
    finally:
        os.unlink(sorted_mut)
    utilib.cprint(f"[MEM][{wrk}][{sample_id}] after mut_df load: {utilib.get_stats()} GB | rows={len(mut_df)}", color="yellow")

    # WARNING: iterating all_positions here touches every Python object in the inherited dict,
    # breaking copy-on-write and causing each worker to copy the full dict into its own RSS.
    mut_df = get_shared_muts(mut_df, mut_bed, all_positions)
    utilib.cprint(f"[MEM][{wrk}][{sample_id}] after get_shared_muts: {utilib.get_stats()} GB", color="yellow")

    mut_df = annotate_muts(mut_df, thr_shared)

    # Step 3: per-read check, ahead of the pileup, to flag ZMWs with an implausible number of
    # DNMs before spending time piling up their positions in other samples' BAMs.
    mut_df = flag_zmw_with_multiple_dnm(mut_df, max_muts_per_duplex)

    # Step 4: final safeguard — drop de novos whose ALT allele has read support in other BAMs.
    # thr_shared doubles as the number of supporting samples required (see flag_fp_denovo).
    # bam_by_mut preserves the -i order, so the COUNTS fields line up with the header's
    # OTHER SAMPLES list for every sample.
    other_bams = [(get_sample_id(m), bam) for m, bam in bam_by_mut.items() if m != mut_bed]
    tres = time.time() / 60
    mut_df = flag_fp_denovo(mut_df, other_bams, min_mapq, min_baseq, min_alt_support, thr_shared)
    quattro = time.time() / 60
    utilib.cprint(f"[MEM][{wrk}][{sample_id}] after flag_fp_denovo ({len(other_bams)} BAMs): {utilib.get_stats()} GB | time={round(quattro-tres,2)} min", color="yellow")

    # numerator = mutations that survive germline reclassification (true de novos)
    numerator = len(mut_df.loc[mut_df.annotation == "DNM"])

    vcflib.write_mut_shared(mut_df, mut_bed, samples_id, sample_mut_out)
    utilib.cprint(f"[MEM][{wrk}][{sample_id}] done: {utilib.get_stats()} GB | numerator={numerator}", color="yellow")

    cl1, cl2 = poisson_rate_cl(numerator,denominator)
    rate =  numerator/denominator if denominator > 0 else 0
    output = "\t".join([sample_id, str(rate), str(cl1), str(cl2), str(numerator), str(denominator)])
    return output


# Module-level globals used to share large data with forked workers without pickling.
# _g_all_positions must be assigned BEFORE the Pool is created so the fork() call
# captures it in every child; _worker_init then copies it into a per-process dict
# alongside the smaller per-run arguments (intersection path, sample list, threshold).
_worker_state = {}
_g_all_positions = None


def _worker_init(intersection_path, samples_id, thr_shared, reference, max_muts_per_duplex,
                 bam_by_mut, min_mapq, min_baseq, min_alt_support):
    _worker_state['all_positions'] = _g_all_positions  # no pickling; inherited via fork
    _worker_state['intersection'] = intersection_path
    _worker_state['samples_id'] = samples_id
    _worker_state['thr_shared'] = thr_shared
    _worker_state['reference'] = reference
    _worker_state['max_muts_per_duplex'] = max_muts_per_duplex
    _worker_state['bam_by_mut'] = bam_by_mut
    _worker_state['min_mapq'] = min_mapq
    _worker_state['min_baseq'] = min_baseq
    _worker_state['min_alt_support'] = min_alt_support


def _wrap_worker_top(mut_bed):
    # Pool.map requires a single-argument callable; this unpacks the shared worker state.
    return wrap_worker(
        mut_bed,
        intersection=_worker_state['intersection'],
        samples_id=_worker_state['samples_id'],
        all_positions=_worker_state['all_positions'],
        thr_shared=_worker_state['thr_shared'],
        reference=_worker_state['reference'],
        max_muts_per_duplex=_worker_state['max_muts_per_duplex'],
        bam_by_mut=_worker_state['bam_by_mut'],
        min_mapq=_worker_state['min_mapq'],
        min_baseq=_worker_state['min_baseq'],
        min_alt_support=_worker_state['min_alt_support'],
    )

def get_single_sample_depth(context_bed):
    out=0
    with gzip.open(context_bed, "rb") as src:
        for line in src:
            if not line.startswith(b"#"):
                out+=int(line.split("\t")[4])
    return out

def get_single_sample_muts(mut_bed):
    df = pd.read_csv(mut_bed, sep="\t", compression="gzip", comment="#", header=None)
    df.columns = COLS
    df["annotation"] = df["type"].map(ANN_DICT)
    return df

###########
############
#########


def _intersect_chrom(region_tuple, beds, n):
    # Workers write filtered lines to a temp file and return its path so that only a short
    # string travels through the multiprocessing pipe, bounding RSS to O(1 chromosome) at a time.
    CONTIG, _START, _END = region_tuple
    tmp_files = []
    out_tmp = None
    try:
        for bed in beds:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as tmp:
                subprocess.run(["tabix", bed, CONTIG], stdout=tmp, stderr=subprocess.DEVNULL, text=True, check=True)
                tmp_files.append(tmp.name)

        # If any sample has no coverage on this chromosome, exclude the whole chromosome
        # from the shared space rather than producing a partial intersection.
        if any(os.path.getsize(f) == 0 for f in tmp_files):
            return None

        with tempfile.NamedTemporaryFile(mode="w", suffix=".out.bed", delete=False) as out:
            out_tmp = out.name
            proc = subprocess.Popen(["bedtools", "multiinter", "-i", *tmp_files],
                                    stdout=subprocess.PIPE, text=True)
            for line in proc.stdout:
                fields = line.split("\t")
                # fields[3] = number of input files overlapping this interval;
                # keep only intervals covered in ALL n samples
                if int(fields[3]) == n:
                    out.write(f"{fields[0]}\t{fields[1]}\t{fields[2]}\n")
            proc.stdout.close()
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, proc.args)
        return out_tmp

    except Exception:
        if out_tmp and os.path.exists(out_tmp):
            os.unlink(out_tmp)
        raise
    finally:
        for f in tmp_files:
            try: os.unlink(f)
            except OSError: pass


def intersect_context_beds(beds, reference, nproc):
    # imap preserves chromosome order and lets up to nproc workers run ahead, but since
    # each result is now a file path (not a list), buffered results consume negligible
    # memory while waiting. Each file is streamed then deleted before moving to the next.
    n = len(beds)
    contigs_list = contiglib.get_contig_list(reference, None, None)
    worker = partial(_intersect_chrom, beds=beds, n=n)
    with mp.Pool(processes=nproc) as pool:
        for i, path in enumerate(pool.imap(worker, contigs_list)):
            contig = contigs_list[i][0]
            if path is None:
                utilib.cprint(f"[MEM] [{contig}] no shared coverage, skipped. Mem: {utilib.get_stats()} GB")
                continue
            try:
                size_mb = os.path.getsize(path) / 1e6
                utilib.cprint(f"[MEM] [{contig}] shared context: {size_mb:.1f} MB. Mem: {utilib.get_stats()} GB")
                with open(path) as f:
                    yield from f
            finally:
                os.unlink(path)

##############
###########
########
def main(all_muts, thr_shared, max_muts_per_duplex, nproc, reference,
         bams=None, min_mapq=30, min_baseq=70, min_alt_support=1):

    # --bams is positionally matched to -i. With a single sample there are no other BAMs to
    # cross-check against, so the safeguard is a no-op and the list is not required.
    bams = bams or []
    if bams and len(bams) != len(all_muts):
        utilib.cprint(f"ERROR: --bams has {len(bams)} file(s) but -i has {len(all_muts)}; they must match in number and order.", color="red")
        utilib.exit()
    bam_by_mut = dict(zip(all_muts, bams))

    if len(all_muts)<2:
        # Single-sample path: no cross-sample intersection needed.
        # Annotate mutations from the full context without shared-interval filtering.
        mut = all_muts[0]
        mut_out = mut.replace("muts.bed.gz", "revised.muts.bed.gz")
        context_bed = mut.replace("muts.bed.gz", "context.bed.gz")
        sample_id = get_sample_id(mut)
        out_dir = os.path.dirname(mut)
        out_file = os.path.join(out_dir,f"{sample_id}.mutation_rates.tsv")
        df_mut =  get_single_sample_muts(mut)

        # No other BAMs to pile up against here, so the ZMW check follows annotate_muts directly.
        df_mut = annotate_muts(df_mut, thr_shared)
        df_mut = flag_zmw_with_multiple_dnm(df_mut, max_muts_per_duplex)

        denominator = get_single_sample_depth(context_bed)
        numerator = len(df_mut.loc[df_mut.annotation == ANN_DICT["d"]])

        cl1, cl2 = poisson_rate_cl(numerator,denominator)
        rate =  numerator/denominator if denominator > 0 else 0

        vcflib.write_mut_shared(df_mut, mut, sample_id, mut_out, single=True)
        output = "\t".join([sample_id, str(rate), str(cl1), str(cl2), str(numerator), str(denominator)])
        vcflib.write_mut_rate([output], out_file)

    else:
        global _g_all_positions
        out_dir = os.path.dirname(all_muts[0])

        all_contexts = [m.replace("muts.bed.gz", "context.bed.gz") for m in all_muts]
        samples_id = [get_sample_id(b) for b in all_muts]

        # Load mutation positions for all samples BEFORE forking so workers inherit the
        # dict via copy-on-write without any pickling overhead.
        utilib.cprint(f"[MEM] before get_positions: {utilib.get_stats()} GB")
        all_positions = {m: get_positions(m) for m in all_muts}
        _g_all_positions = all_positions  # must be set before Pool() so workers inherit via fork
        total_entries = sum(len(v) for v in all_positions.values())
        utilib.cprint(f"[MEM] all_positions loaded: {total_entries:,} entries across {len(all_muts)} samples. Mem: {utilib.get_stats()} GB")

        out_file = os.path.join(out_dir, "_".join(samples_id) + ".mutation_rates.tsv")

        # Phase 1: compute the genomic space observable in ALL samples simultaneously.
        # Result written to a single temp BED on disk; used as -b reference in every worker.
        # intersection_path assigned before the loop so the finally block can always clean up.
        utilib.cprint(f"[MEM] starting bedtools multiinter. Mem: {utilib.get_stats()} GB")
        uno = time.time() / 60

        intersection_path = None
        intersection_gz = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as tmp:
                intersection_path = tmp.name
                for line in intersect_context_beds(all_contexts, reference, nproc):
                    tmp.write(line)

            dos = time.time() / 60
            intersection_size_mb = os.path.getsize(intersection_path) / 1e6
            utilib.cprint(f"[MEM] bedtools multiinter done. Intersection file: {intersection_size_mb:.1f} MB. Time: {round(dos-uno,2)} min. Mem: {utilib.get_stats()} GB")

            # Compress and index the intersection file so each worker streams it via the
            # tabix index instead of loading the full file into memory per subprocess.
            intersection_gz = intersection_path + ".gz"
            pysam.tabix_compress(intersection_path, intersection_gz, force=True)
            pysam.tabix_index(intersection_gz, preset="bed", force=True)
            os.unlink(intersection_path)
            intersection_path = None  # plain file gone; cleanup targets .gz and .gz.tbi
            utilib.cprint(f"[MEM] intersection indexed. Mem: {utilib.get_stats()} GB")

            # Phase 2: per-sample parallel processing — filter context + mutations to shared
            # intervals, reclassify shared de novos as germline, compute rate per sample.
            # fork context required so workers inherit _g_all_positions without pickling.
            utilib.cprint(f"[MEM] before forking pool (nproc={nproc}): {utilib.get_stats()} GB")
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=nproc,
                        initializer=_worker_init,
                        initargs=(intersection_gz, samples_id, thr_shared, reference, max_muts_per_duplex,
                                  bam_by_mut, min_mapq, min_baseq, min_alt_support)) as pool:
                results = pool.map(_wrap_worker_top, all_muts)
        finally:
            if intersection_path:
                try: os.unlink(intersection_path)
                except OSError: pass
            if intersection_gz:
                for f in (intersection_gz, intersection_gz + ".tbi"):
                    try: os.unlink(f)
                    except OSError: pass
            _g_all_positions = None

        vcflib.write_mut_rate(results, out_file)
        utilib.cprint(f"MUTATION RATE FILE SAVED TO: {out_file}")

        # Each worker has written its shared.context.bed.gz; where that output exists, the
        # original per-sample context input is redundant, so reclaim its disk space.
        for ctx in all_contexts:
            shared_context = ctx.replace("context.bed.gz", "shared.context.bed.gz")
            if os.path.exists(shared_context):
                for f in (ctx, ctx + ".tbi"):
                    try:
                        os.unlink(f)
                        utilib.cprint(f"{ctx} DELETED", color="yellow")
                    except OSError: pass
