"""
`BiStro somatic` — cross-sample post-processing of `preprocess`'s
.muts.bed.gz / .context.bed.gz outputs. Restricts every sample to the genomic
space callable in ALL samples, reclassifies de novo (DNM) calls as
germline/false-positive when they recur across samples, across reads of the
same ZMW, or have raw read support in other samples' BAMs, and reports each
sample's resulting somatic mutation rate with a Poisson confidence interval.

Implementation (efficiency refactor; results identical to the original):

* Shared callable space. The original extracted every sample's per-base context
  per chromosome into uncompressed temp files, ran `bedtools multiinter`, and
  then streamed each sample's whole context through `bedtools intersect` in one
  worker per sample. Here the genome is cut into ranges of RANGE_BP; for every
  (sample, range) a worker streams that range of the sample's context via tabix
  and returns a packed presence bitmap (pass 1); the parent ANDs the bitmaps of
  all samples into the shared-space bitmap of the range; a second task per
  (sample, range) re-streams the range, keeps the lines whose position is shared
  (verbatim text, as `bedtools intersect` reported them), writes them to an
  indexed BGZF fragment and sums their DEPTH (pass 2). Fragments are
  concatenated per sample and the .tbi is written from the offsets recorded
  while writing (no re-scan). Nothing per-base crosses a process boundary
  except one bit per base of the range being processed, and nothing is written
  to TMPDIR.
* Mutations. Each .muts.bed.gz is read once; every row keeps its original text
  line (the original round-tripped the 12 columns through pandas, which prints
  them back unchanged) plus the few parsed columns the logic needs.
* Pileup safeguard. Every (BAM, site) pair is counted once, sites sorted and
  clustered, in parallel tasks, by allelelib.count_alleles (same per-site result
  as the original per-site pysam pileup, much faster).
"""

import gzip
import os
import time
import multiprocessing as mp
from collections import OrderedDict

import numpy as np
import pandas as pd
import pysam
from scipy.stats import chi2

from . import vcflib, utilib, contiglib, fastio, allelelib
pysam.set_verbosity(0)

# Column order of .muts.bed.gz files (no header line in the file)
COLS = [
    "contig", "start", "end", "zmw", "triplet", "ref",
    "strand_1", "strand_2", "nob", "n_mis", "type", "depth",
]


# Mutation type codes from the bistro caller, mapped to their default annotation
ANN_DICT = {"z":"RDM",   # random non-mutated site
            "m":"MSM",   # mismatch (QB artefact)
            "g":"GRM",   # germline variant (assigned by cross-sample recurrence in annotate_muts)
            "x":"DNM"}   # ds-mutation: de novo unless annotate_muts reclassifies it

# Index into the [A, C, G, T] count rows returned by count_alleles.
BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}

# Genome ranges processed per task for the shared-space computation. A multiple of the
# 16-kb tabix window so no index window is split between two fragments.
RANGE_BP = 1 << 24
# Target number of pileup sites per allele-counting task.
PILEUP_SITES_PER_TASK = 2000
CTX_NCOLS = 5          # CONTIG START END TRIPLET DEPTH


def get_sample_id(mut_bed):
    return os.path.basename(mut_bed).split(".")[0]


def poisson_rate_cl(k, exposure, alpha=0.05):
    # exact Poisson confidence interval via chi-squared quantiles.
    if exposure > 0:
        lower = 0.0 if k == 0 else chi2.ppf(alpha/2, 2*k) / 2
        upper = chi2.ppf(1-alpha/2, 2*(k+1)) / 2
        return lower/exposure, upper/exposure
    return 0, 0


def is_collapsed(mut_bed):
    # preprocess writes read_mode=collapsed into the ##filters header line of collapsed-mode files.
    with gzip.open(mut_bed, "rt") as f:
        for line in f:
            if not line.startswith("#"):
                return False
            if "read_mode=collapsed" in line:
                return True
    return False


# ---------------------------------------------------------------------------
# Mutation table
# ---------------------------------------------------------------------------
def load_muts(mut_bed):
    """All data rows of a .muts.bed.gz as a slim DataFrame.

    Columns: line (the row's original text, no newline), contig, start, end, zmw, type,
    n_mis, alt (STRAND_1 up to the first ':'). Text columns are kept exactly as written,
    numeric ones parsed as int64. Header ('#') lines are skipped, like bedtools did.
    """
    with gzip.open(mut_bed, "rt") as fh:
        # bedtools skipped '#'/track/browser header lines and blank lines
        lines = [l[:-1] if l.endswith("\n") else l for l in fh
                 if l.strip() and not l.startswith(("#", "track", "browser"))]
    if not lines:
        return pd.DataFrame({"line": pd.Series([], dtype=object),
                             "contig": pd.Series([], dtype=object),
                             "start": pd.Series([], dtype=np.int64),
                             "end": pd.Series([], dtype=np.int64),
                             "zmw": pd.Series([], dtype=object),
                             "type": pd.Series([], dtype=object),
                             "n_mis": pd.Series([], dtype=np.int64),
                             "alt": pd.Series([], dtype=object)})
    from io import StringIO
    # Every field as its exact text (no NA conversion), then parse what the logic needs.
    tab = pd.read_csv(StringIO("\n".join(lines)), sep="\t", header=None, names=COLS,
                      usecols=["contig", "start", "end", "zmw", "strand_1", "n_mis", "type"],
                      dtype=str, na_filter=False, quoting=3)
    df = pd.DataFrame({
        "line": pd.Series(lines, dtype=object),
        "contig": tab["contig"].astype(object),
        "start": tab["start"].astype(np.int64),
        "end": tab["end"].astype(np.int64),
        "zmw": tab["zmw"].astype(object),
        "type": tab["type"].astype(object),
        "n_mis": tab["n_mis"].astype(np.int64),
        "alt": tab["strand_1"].str.split(":", n=1).str[0].astype(object),
    })
    return df


def position_keys(df):
    # get_positions() of the original, from the FULL file: high-confidence somatic candidates,
    # i.e. TYPE not random (z) / mismatch (m) and zero mismatches in the QB window.
    keep = ~df["type"].str.rstrip().isin(["m", "z"]) & (df["n_mis"] == 0)
    return df.loc[keep, ["contig", "start", "alt"]].drop_duplicates()


def sort_like_original(df, chrom_idx):
    # The original sorted rows by (reference .fai index of the contig, START) with a stable
    # sort (unknown contigs last) before `bedtools intersect -sorted`, which keeps that order.
    idx = df["contig"].map(chrom_idx).fillna(len(chrom_idx)).to_numpy(dtype=np.int64)
    key = (idx << 32) | df["start"].to_numpy(dtype=np.int64)
    order = np.argsort(key, kind="stable")
    return df.iloc[order].reset_index(drop=True)


def collapsed_alt_support_filter(mut_df, min_alt_support):
    # Collapsed mode only: drop mutation calls whose ALT is carried by fewer than min_alt_support
    # of this sample's reads, so unconfirmed single-read calls never reach the pileup.
    is_x = mut_df["type"] == "x"
    support = mut_df.loc[is_x].groupby(["contig", "start", "alt"]).zmw.transform("size")
    return mut_df.drop(support.index[support < min_alt_support])


def get_shared_counts(mut_df, n_total, own_keys):
    # For each mutation, count how many OTHER samples carry a call at the same (contig,start,ALT):
    # (number of input files whose key set holds it) - (1 if this file's own set holds it).
    keys = mut_df[["contig", "start", "alt"]]
    tot = keys.merge(n_total, on=["contig", "start", "alt"], how="left")["n"].fillna(0)
    own = keys.merge(own_keys.assign(o=1), on=["contig", "start", "alt"], how="left")["o"].fillna(0)
    return (tot.to_numpy(dtype=np.int64) - own.to_numpy(dtype=np.int64))


def get_zmw_with_multiple_dnm(df,max_muts_per_duplex):

    only_dnm = df.loc[df.annotation == "DNM"]
    counts = only_dnm.groupby("zmw").size()
    zmw_discarded = counts.loc[counts >= max_muts_per_duplex].index
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
    df.loc[(df["zmw"].isin(to_flag)) & (df["annotation"] == ANN_DICT["x"]), "annotation"] = "FPD"
    return df


def annotate_muts(df, thr_shared):
    # Default annotation from type code (ds-mutation x -> DNM); then upgrade x mutations that appear
    # in >= thr_shared other samples to GRM — recurrence across organisms implies germline.
    # The single-sample path has no "shared" column: there is nothing to be shared with.
    df["annotation"] = df["type"].map(ANN_DICT)
    if "shared" in df.columns:
        df.loc[(df.type == "x") & (df.shared >= thr_shared),"annotation"] = ANN_DICT["g"]

    ### flag de novo mutations with any mismatch in the QB window (n_mis > 0) as false positives
    df.loc[(df["n_mis"] > 0) & (df["annotation"] == ANN_DICT["x"]), "annotation"] = "FPD"
    return df


def dnm_sites(df):
    # {contig: sorted unique 1-based positions} of this sample's DNM rows (BED END column).
    dnm = df.loc[df.annotation == ANN_DICT["x"], ["contig", "end"]]
    return {c: np.unique(g["end"].to_numpy(dtype=np.int64)) for c, g in dnm.groupby("contig", sort=False)}


def flag_fp_denovo(df, other_counts, min_alt_support, min_alt_samples):
    # Final safeguard against false-positive de novo calls.
    #
    # A true somatic mutation is private to one organism, so its ALT allele should be absent
    # from every other sample's raw reads. get_shared_counts only sees positions the caller
    # actually *called* elsewhere; this check goes underneath that detection threshold by
    # counting ALT-supporting reads directly in the other samples' BAMs. A DNM whose ALT is
    # supported by >= min_alt_support reads in >= min_alt_samples other BAMs is reclassified
    # as FPD (germline variant, or a recurrent artefact at a systematically error-prone site).
    #
    # min_alt_samples is thr_shared: both ask "in how many other samples must this variant
    # appear before we stop believing it is private?" — one counts called sites, one counts
    # raw read support.
    #
    # other_counts: one entry per other BAM, in --bams order: {contig: (positions, counts[n,4])}
    # from allelelib.count_alleles. The per-sample A:C:G:T counts behind the decision are kept
    # in the COUNTS column so a flagged call can be inspected afterwards.
    # "." marks rows where no pileup was done (non-DNM rows, or --bams not supplied).
    df["counts"] = "."
    if not other_counts:
        return df

    dnm = df.loc[df.annotation == ANN_DICT["x"]]
    if dnm.empty:
        return df

    # BED end column holds the 1-based coordinate of the substituted base.
    contigs = dnm["contig"].to_numpy(dtype=object)
    positions = dnm["end"].to_numpy(dtype=np.int64)

    # [n_dnm, 4] count table per other sample, rows aligned with dnm (zeros where not counted,
    # e.g. contig absent from that BAM).
    per_sample = []
    for table in other_counts:
        rows = np.zeros((len(dnm), 4), dtype=np.int64)
        for c in pd.unique(contigs):
            sel = np.flatnonzero(contigs == c)
            if c not in table:
                continue
            pos_c, cnt_c = table[c]
            k = np.searchsorted(pos_c, positions[sel])
            k_ok = k < pos_c.size
            hit = np.zeros(sel.size, dtype=bool)
            hit[k_ok] = pos_c[k[k_ok]] == positions[sel][k_ok]
            rows[sel[hit]] = cnt_c[k[hit]]
        per_sample.append(rows)

    # A single ALT read is the strictest this check can be, so never let a 0 threshold
    # make every base "supported" and flag every de novo.
    support_thr = max(1, min_alt_support)

    # COUNTS: one A:C:G:T field per other sample, comma-separated, order given in the header.
    fields = [[":".join(map(str, r)) for r in rows.tolist()] for rows in per_sample]
    df.loc[dnm.index, "counts"] = [",".join(f) for f in zip(*fields)]

    alts = dnm["alt"].to_numpy(dtype=object)
    idx = np.array([BASE_INDEX.get(a, -1) for a in alts], dtype=np.int64)
    supporting = np.zeros(len(dnm), dtype=np.int64)
    known = idx >= 0
    for rows in per_sample:
        supporting[known] += rows[np.flatnonzero(known), idx[known]] >= support_thr
    is_fp = supporting >= min_alt_samples
    df.loc[dnm.index[is_fp], "annotation"] = "FPD"
    return df


def write_mut_shared(df, input, samples, output, single=False):
    # Same header as the original vcflib.write_mut_shared; each row = its original 12-column
    # text + the added columns (SHARED ANNOTATION COUNTS, or ANNOTATION alone in single mode),
    # i.e. exactly what df.to_csv(sep="\t") printed. A NaN annotation prints as ''.
    with gzip.open(input, "rb") as src, pysam.BGZFile(output, "wb") as dst:
        collapsed = any(b"read_mode=collapsed" in line for line in vcflib.copy_header(src, dst))
        if single:
            dst.write(("\n".join(vcflib.mut_header_single(samples, collapsed))).encode())
        else:
            dst.write(("\n".join(vcflib.mut_header_shared(samples, collapsed))).encode())
        ann = df["annotation"].astype(object).where(df["annotation"].notna(), "")
        if single:
            cols = [df["line"], ann]
        else:
            cols = [df["line"], df["shared"].astype(np.int64).astype(str), ann, df["counts"]]
        step = 200_000
        n = len(df)
        for a in range(0, n, step):
            block = zip(*(c.iloc[a:a + step].tolist() for c in cols))
            dst.write(("".join("\t".join(r) + "\n" for r in block)).encode())


# ---------------------------------------------------------------------------
# Shared callable space (pool tasks)
# ---------------------------------------------------------------------------
def _presence_task(task):
    # Pass 1: packed bitmap of the context positions of one sample in [a, b) of one contig.
    k, s, ctx, contig, a, b = task
    present = np.zeros(b - a, dtype=bool)
    for buf in fastio.iter_tabix_chunks(ctx, contig, a, b):
        ch = fastio.parse_bed_chunk(buf, CTX_NCOLS)
        pos = fastio.parse_uint_field(ch, 1)
        present[pos - a] = True
    return k, s, np.packbits(present)


def _shared_task(task):
    # Pass 2: keep this sample's context lines of [a, b) whose position is in the shared space;
    # write them verbatim to an indexed BGZF fragment and sum their DEPTH.
    k, s, ctx, contig, a, b, common_packed, frag = task
    common = np.unpackbits(common_packed, count=b - a).view(bool)
    depth_sum = 0
    n = 0
    w = fastio.IndexedBgzfWriter(frag, contig)
    try:
        for buf in fastio.iter_tabix_chunks(ctx, contig, a, b):
            ch = fastio.parse_bed_chunk(buf, CTX_NCOLS)
            pos = fastio.parse_uint_field(ch, 1)
            keep = common[pos - a]
            nk = int(np.count_nonzero(keep))
            if nk == 0:
                continue
            depth_sum += int(fastio.parse_uint_field(ch, 4)[keep].sum())
            data, offs = fastio.select_lines(ch, keep)
            w.write_records(data, pos[keep], offs)
            n += nk
    finally:
        fi = w.close()
    return k, s, frag, fi, n, depth_sum


def _assemble_task(task):
    final, header, frags = task
    fastio.assemble_bgzf(final, header, frags, index=True)
    return final


def _context_depth_task(task):
    # Single-sample path: sum of DEPTH over one contig of a context file.
    ctx, contig = task
    tot = 0
    for buf in fastio.iter_tabix_chunks(ctx, contig):
        tot += int(fastio.parse_uint_field(fastio.parse_bed_chunk(buf, CTX_NCOLS), 4).sum())
    return tot


def _make_ranges(reference, contexts):
    # [(contig, a, b)] in reference (.fai) order over the contigs present in EVERY sample's context
    # index. A contig missing from any sample has an empty shared space (as with multiinter == n).
    present = None
    for ctx in contexts:
        cs = set(fastio.tabix_contigs(ctx))
        present = cs if present is None else present & cs
    ranges = []
    for contig, _, length in contiglib.get_contig_list(reference, None, None):
        if contig not in present:
            continue
        for a in range(0, length, RANGE_BP):
            ranges.append((contig, a, min(a + RANGE_BP, length)))
    return ranges


def shared_space(pool, paths_id, contexts, header_ids, reference, out_dir):
    """Compute the shared callable space and every sample's shared-context fragments.

    paths_id/contexts: one entry per distinct input file; header_ids: the sample list printed in
    the shared-context header (every -i entry, as in the original). Returns (denominators per
    file, {contig: [(a, b, common_packed)]}, assembly tasks for _assemble_task)."""
    n = len(contexts)
    ranges = _make_ranges(reference, contexts)
    utilib.cprint(f"[MEM] shared space: {len(ranges)} range(s) x {n} samples. Mem: {utilib.get_stats()} GB")

    # Largest ranges first keeps the pool busy until the end.
    order = sorted(range(len(ranges)), key=lambda k: -(ranges[k][2] - ranges[k][1]))
    tasks1 = [(k, s, contexts[s], *ranges[k]) for k in order for s in range(n)]
    common = [None] * len(ranges)
    seen = [0] * len(ranges)
    for k, s, bits in pool.imap_unordered(_presence_task, tasks1):
        common[k] = bits if common[k] is None else np.bitwise_and(common[k], bits, out=common[k])
        seen[k] += 1
    assert all(c == n for c in seen)
    utilib.cprint(f"[MEM] shared-space bitmaps done. Mem: {utilib.get_stats()} GB")

    frag_dir = out_dir
    tasks2 = []
    for k in order:
        if not common[k].any():
            continue
        for s in range(n):
            frag = os.path.join(frag_dir, f".{paths_id[s]}.shared.context.r{k:06d}.frag.bgz")
            tasks2.append((k, s, contexts[s], *ranges[k], common[k], frag))
    frags = [dict() for _ in range(n)]           # per sample: k -> (frag path, FragmentIndex)
    denominators = [0] * n
    for k, s, frag, fi, nk, dsum in pool.imap_unordered(_shared_task, tasks2):
        frags[s][k] = (frag, fi)
        denominators[s] += dsum
    utilib.cprint(f"[MEM] shared-context fragments written. Mem: {utilib.get_stats()} GB")

    by_contig = OrderedDict()
    for k, (contig, a, b) in enumerate(ranges):
        by_contig.setdefault(contig, []).append((a, b, common[k]))

    assembly = []
    for s in range(n):
        out = contexts[s].replace("context.bed.gz", "shared.context.bed.gz")
        header = (f"##INTERVALS FROM {paths_id[s]} SHARED ACROSS: {' '.join(header_ids)}\n".encode()
                  + b"".join(vcflib.get_context_header(contexts[s])))
        # fragments in reference order; an empty fragment (no kept line) carries no index
        fl = [frags[s][k] for k in range(len(ranges)) if k in frags[s]]
        assembly.append((out, header, fl))
    return denominators, by_contig, assembly


def in_shared_space(df, by_contig):
    # Boolean mask: row START lies in the shared space (bit lookup in the packed range bitmaps).
    keep = np.zeros(len(df), dtype=bool)
    contigs = df["contig"].to_numpy(dtype=object)
    starts = df["start"].to_numpy(dtype=np.int64)
    for c in pd.unique(contigs):
        if c not in by_contig:
            continue
        sel = np.flatnonzero(contigs == c)
        st = starts[sel]
        for a, b, packed in by_contig[c]:
            inr = (st >= a) & (st < b)
            if not inr.any():
                continue
            off = st[inr] - a
            bits = (packed[off >> 3] >> (7 - (off & 7))) & 1
            keep[sel[inr]] = bits.astype(bool)
    return keep


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _pool(nproc):
    # forkserver: workers start lean (no copy of the parent's tables) on every Python version.
    return mp.get_context("forkserver").Pool(processes=max(1, nproc))


def single_sample(mut, thr_shared, max_muts_per_duplex, nproc):
    # Single-sample path: no cross-sample intersection needed.
    # Annotate mutations from the full context without shared-interval filtering.
    mut_out = mut.replace("muts.bed.gz", "revised.muts.bed.gz")
    context_bed = mut.replace("muts.bed.gz", "context.bed.gz")
    sample_id = get_sample_id(mut)
    out_dir = os.path.dirname(mut)
    out_file = os.path.join(out_dir,f"{sample_id}.mutation_rates.tsv")
    df_mut = load_muts(mut)

    # No other BAMs to pile up against here, so the ZMW check follows annotate_muts directly.
    df_mut = annotate_muts(df_mut, thr_shared)
    df_mut = flag_zmw_with_multiple_dnm(df_mut, max_muts_per_duplex)

    # Denominator = sum of DEPTH over the whole context (the original's per-line loop crashed:
    # it split bytes lines by a str separator).
    with _pool(nproc) as pool:
        denominator = sum(pool.map(_context_depth_task,
                                   [(context_bed, c) for c in fastio.tabix_contigs(context_bed)]))
    numerator = int((df_mut.annotation == ANN_DICT["x"]).sum())

    cl1, cl2 = poisson_rate_cl(numerator,denominator)
    rate =  numerator/denominator if denominator > 0 else 0

    # [sample_id]: the original passed the bare string, so the header listed its characters.
    write_mut_shared(df_mut, mut, [sample_id], mut_out, single=True)
    output = "\t".join([sample_id, str(rate), str(cl1), str(cl2), str(numerator), str(denominator)])
    vcflib.write_mut_rate([output], out_file)


def main(all_muts, thr_shared, max_muts_per_duplex, nproc, reference,
         bams=None, min_mapq=30, min_baseq=70, min_alt_support=1):

    # --bams is positionally matched to -i. With a single sample there are no other BAMs to
    # cross-check against, so the safeguard is a no-op and the list is not required.
    bams = bams or []
    if bams and len(bams) != len(all_muts):
        utilib.cprint(f"ERROR: --bams has {len(bams)} file(s) but -i has {len(all_muts)}; they must match in number and order.", color="red")
        utilib.exit(1)
    bam_by_mut = dict(zip(all_muts, bams))

    if len(all_muts)<2:
        single_sample(all_muts[0], thr_shared, max_muts_per_duplex, nproc)
        return

    t0 = time.time()
    out_dir = os.path.dirname(all_muts[0])
    # One entry per distinct input file (the original keyed its per-file tables by path).
    paths = list(OrderedDict.fromkeys(all_muts))
    all_contexts = [m.replace("muts.bed.gz", "context.bed.gz") for m in paths]
    samples_id = [get_sample_id(b) for b in all_muts]
    paths_id = [get_sample_id(b) for b in paths]
    out_file = os.path.join(out_dir, "_".join(samples_id) + ".mutation_rates.tsv")

    chrom_idx = {}
    with open(reference + ".fai") as fai:
        for i, line in enumerate(fai):
            chrom_idx[line.split("\t")[0]] = i

    with _pool(nproc) as pool:
        # ---- Phase 1: shared callable space + shared context files -------------------------
        utilib.cprint(f"[MEM] computing shared callable space (nproc={nproc}). Mem: {utilib.get_stats()} GB")
        denominators, by_contig, assembly = shared_space(pool, paths_id, all_contexts, samples_id,
                                                         reference, out_dir)
        # Concatenating + indexing each sample's shared context only needs the fragments: run it
        # in the background while the mutations are processed.
        assembled = pool.map_async(_assemble_task, assembly)
        utilib.cprint(f"[MEM] shared space done in {round((time.time()-t0)/60, 2)} min. Mem: {utilib.get_stats()} GB")

        # ---- Phase 2: mutations ------------------------------------------------------------
        # Cross-sample key sets from the FULL files (before any filter), as get_positions did.
        tables = {}
        key_sets = {}
        for m in paths:
            df = load_muts(m)
            key_sets[m] = position_keys(df)
            tables[m] = df
        all_keys = pd.concat(key_sets.values(), ignore_index=True)
        n_total = all_keys.groupby(["contig", "start", "alt"]).size().rename("n").reset_index()
        utilib.cprint(f"[MEM] position keys loaded: {len(all_keys):,} entries across {len(paths)} samples. Mem: {utilib.get_stats()} GB")
        del all_keys

        sites_by_path = {}
        for m in paths:
            sid = get_sample_id(m)
            df = tables[m]
            df = sort_like_original(df.loc[in_shared_space(df, by_contig)], chrom_idx)
            utilib.cprint(f"[MEM][{sid}] mut rows in shared space: {len(df)}", color="yellow")
            if is_collapsed(m):
                df = collapsed_alt_support_filter(df, min_alt_support)
                utilib.cprint(f"[MEM][{sid}] after collapsed ALT-support filter (>= {min_alt_support} reads): rows={len(df)}", color="yellow")
            df["shared"] = get_shared_counts(df, n_total, key_sets[m])
            df = annotate_muts(df, thr_shared)
            # per-read check, ahead of the pileup, to flag ZMWs with an implausible number of DNMs
            df = flag_zmw_with_multiple_dnm(df, max_muts_per_duplex)
            tables[m] = df
            sites_by_path[m] = dnm_sites(df)
        del key_sets, n_total

        # ---- Phase 3: pileup safeguard, each (BAM, site) counted once -------------------------
        counts_by_path = {}
        if bam_by_mut:
            t1 = time.time()
            tasks = []
            for m in paths:
                need = {}
                for other in paths:
                    if other == m:
                        continue
                    for c, pos in sites_by_path[other].items():
                        need.setdefault(c, []).append(pos)
                for c, plist in need.items():
                    pos = np.unique(np.concatenate(plist))
                    for chunk in allelelib.split_positions(pos, PILEUP_SITES_PER_TASK):
                        tasks.append((m, (bam_by_mut[m], c, chunk, min_mapq, min_baseq)))
            n_sites = sum(len(t[1][2]) for t in tasks)
            utilib.cprint(f"[MEM] pileup safeguard: {n_sites:,} (BAM, site) pairs in {len(tasks)} task(s)")
            # biggest tasks first
            tasks.sort(key=lambda t: -len(t[1][2]))
            results = pool.map(allelelib.count_alleles_task, [t[1] for t in tasks], chunksize=1)
            parts = {}
            for (m, _), (c, pos, cnt) in zip(tasks, results):
                parts.setdefault(m, {}).setdefault(c, []).append((pos, cnt))
            for m, per_c in parts.items():
                table = {}
                for c, lst in per_c.items():
                    pos = np.concatenate([p for p, _ in lst])
                    cnt = np.concatenate([x for _, x in lst])
                    o = np.argsort(pos, kind="stable")
                    table[c] = (pos[o], cnt[o])
                counts_by_path[m] = table
            utilib.cprint(f"[MEM] pileups done in {round((time.time()-t1)/60, 2)} min. Mem: {utilib.get_stats()} GB")

        # ---- Phase 4: final annotation, outputs ----------------------------------------------
        rate_by_path = {}
        for i, m in enumerate(paths):
            sid = get_sample_id(m)
            df = tables.pop(m)
            # bam_by_mut preserves the -i order, so the COUNTS fields line up with the header's
            # TOTAL SAMPLES list (minus this sample) for every sample.
            other = [counts_by_path.get(o, {}) for o in bam_by_mut if o != m]
            df = flag_fp_denovo(df, other, min_alt_support, thr_shared)
            # numerator = mutations that survive germline reclassification (true de novos)
            numerator = int((df.annotation == "DNM").sum())
            denominator = int(denominators[i])
            write_mut_shared(df, m, samples_id, m.replace("muts.bed.gz", "shared.muts.bed.gz"))
            utilib.cprint(f"[MEM][{sid}] done: numerator={numerator} denominator={denominator:,} bp. Mem: {utilib.get_stats()} GB", color="yellow")
            cl1, cl2 = poisson_rate_cl(numerator, denominator)
            rate = numerator/denominator if denominator > 0 else 0
            rate_by_path[m] = "\t".join([sid, str(rate), str(cl1), str(cl2), str(numerator), str(denominator)])
            del df

        assembled.get()
        utilib.cprint(f"[MEM] shared context files assembled + indexed. Mem: {utilib.get_stats()} GB")

    results = [rate_by_path[m] for m in all_muts]
    vcflib.write_mut_rate(results, out_file)
    utilib.cprint(f"MUTATION RATE FILE SAVED TO: {out_file}")
    utilib.cprint(f"somatic took {round((time.time()-t0)/60, 2)} minutes")

    # Each sample's shared.context.bed.gz is written; where that output exists, the
    # original per-sample context input is redundant, so reclaim its disk space.
    for ctx in all_contexts:
        shared_context = ctx.replace("context.bed.gz", "shared.context.bed.gz")
        if os.path.exists(shared_context):
            for f in (ctx, ctx + ".tbi"):
                try:
                    os.unlink(f)
                    utilib.cprint(f"{ctx} DELETED", color="yellow")
                except OSError: pass
