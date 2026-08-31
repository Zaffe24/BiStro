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
human-opportunity variant is produced for this method. The coverage mask (below)
is reused unchanged: it flags contexts whose freq_C is too low to trust, which
does not depend on the freq_G target.

The freq_G/freq_C(c) and freq_G(c)_new/freq_C(c) columns report the per-context
correction factor freq_G(c)/freq_C(c) for the genome method and the alternative
method respectively.

A context this sample barely covered (freq_C(c) near zero) makes freq_G/freq_C
huge, so even a single raw count there gets amplified into an outsized,
unreliable share of the corrected spectrum. Contexts whose factor exceeds 4x
this sample's own median freq_G/freq_C ratio (over the 32 contexts) are
masked: their normcount/Human_normcounts are zeroed and normfrac/Human_normfrac
are renormalized over the surviving contexts. The masked column in the output
flags which contexts (and thus which sbs96 rows) were excluded this way.

Usage:
    BiStro sbs96 \
        --context   SAMPLE.shared.context.bed.gz \
        --muts      SAMPLE.shared.muts.bed.gz \
        --ref       genome.fa \
        --out       SAMPLE.normcounts.tsv
"""

import gzip
import os
import statistics
import multiprocessing as mp
from functools import partial

import pysam  # already a BiStro dependency; used only for random FASTA access

from . import arglib

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
# Step 1 — callable denominators R(c) and C(c) from shared.context.bed.gz
# ---------------------------------------------------------------------------
def _count_context_contig(contig, shared_context):
    # Per-contig worker (mirrors caller.py's one-worker-per-contig fan-out).
    # Each worker opens its OWN TabixFile handle — pysam handles are not shareable
    # across processes — and streams only this contig's rows via the .tbi index,
    # so no worker touches another's data and RSS stays at O(one contig).
    # Returns partial {canonical_tri -> count} dicts to be merged by the parent.
    ref_partial = {}
    ccs_partial = {}
    tbx = pysam.TabixFile(shared_context)
    try:
        for line in tbx.fetch(contig):
            # Columns: CONTIG  START  END  TRIPLET  DEPTH
            _, _, _, triplet, depth = line.rstrip("\n").split("\t")
            ctri = canonical_tri(triplet)
            if ctri is None:              # triplet touched an N / non-ACGT base
                continue
            ref_partial[ctri] = ref_partial.get(ctri, 0) + 1
            ccs_partial[ctri] = ccs_partial.get(ctri, 0) + int(depth)
    finally:
        tbx.close()
    return ref_partial, ccs_partial


def _read_shared_context_serial(shared_context, ref_callable, ccs_callable):
    # Single-pass fallback used when no usable tabix index is present.
    with gzip.open(shared_context, "rt") as fh:
        for line in fh:
            if line.startswith("#"):      # skip the two '##' header lines
                continue
            # Columns: CONTIG  START  END  TRIPLET  DEPTH
            _, _, _, triplet, depth = line.rstrip("\n").split("\t")
            ctri = canonical_tri(triplet)
            if ctri is None:
                continue
            ref_callable[ctri] += 1
            ccs_callable[ctri] += int(depth)
    return ref_callable, ccs_callable


def read_shared_context(shared_context, tri_lst, nproc):
    # Accumulate, per canonical trinucleotide context:
    #   ref_callable[c] += 1        (position-unique: each callable base once)
    #   ccs_callable[c] += DEPTH    (read-weighted: each base by its coverage)
    #
    # The pass over shared.context.bed.gz is the heavy part (one row per callable
    # base, genome-wide), so we split it by contig and process contigs in
    # parallel. BiStro tabix-indexes this file, which lets each worker seek
    # directly to its contig instead of every worker scanning the whole file.
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

    # One task per contig; cap the pool at the number of contigs so we never
    # spawn idle workers. Bind the file path; the contig is the mapped argument.
    worker = partial(_count_context_contig, shared_context=shared_context)
    with mp.Pool(processes=min(nproc, len(contigs))) as pool:
        results = pool.map(worker, contigs)

    # Merge the per-contig partial tallies into the full 32-context dicts.
    for ref_partial, ccs_partial in results:
        for tri, n in ref_partial.items():
            ref_callable[tri] += n
        for tri, n in ccs_partial.items():
            ccs_callable[tri] += n
    return ref_callable, ccs_callable


# ---------------------------------------------------------------------------
# Step 2 — genome composition G(c) from the reference FASTA
# ---------------------------------------------------------------------------
def get_genome_tricounts(reference, tri_lst):
    # Count how many times each canonical trinucleotide occurs in the reference,
    # summed over every contig in the FASTA. This is the genome-wide
    # "opportunity" baseline that ref_tri_ratio compares the callable
    # composition against.
    genome = {tri: 0 for tri in tri_lst}
    fasta = pysam.FastaFile(reference)
    
    try:
        for contig in fasta.references:
            seq = fasta.fetch(contig).upper()
            # Slide a 3-base window across the whole contig. Canonicalisation
            # folds purine-centred windows onto their pyrimidine strand, so
            # complementary contexts are pooled exactly as for the callable set.
            for i in range(len(seq) - 2):
                ctri = canonical_tri(seq[i:i + 3])
                if ctri is not None:
                    genome[ctri] += 1
    finally:
        fasta.close()
    return genome


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
# How many times the sample's median freq_G/freq_C ratio a context's own ratio
# may reach before it is treated as an unreliable outlier rather than genuine
# signal.
RATIO_OUTLIER_K = 4


def compute_coverage_mask(ccs_freq, genome_freq, tri_lst, k=RATIO_OUTLIER_K):
    # Flag contexts whose correction factor freq_G(c)/freq_C(c) exceeds k times
    # this sample's own median ratio (over the contexts that have any coverage
    # at all). A context with zero coverage has no ratio to compare and is
    # masked outright, same as apply_correction already treats it (nothing to
    # rescale).
    ratios = {
        tri: genome_freq[tri] / ccs_freq[tri]
        for tri in tri_lst
        if ccs_freq[tri] > 0
    }
    if not ratios:
        return {tri: True for tri in tri_lst}

    threshold = k * statistics.median(ratios.values())

    return {
        tri: (tri not in ratios) or (ratios[tri] > threshold)
        for tri in tri_lst
    }


def apply_correction(sbs96_counts, sbs96_lst, sbs96_to_tri, ccs_freq, genome_freq, masked):
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
        # A context with no callable coverage was never sampled, and a context
        # flagged by compute_coverage_mask was sampled too little to trust its
        # correction factor: both stay 0 rather than rescaling a raw count into
        # an unreliable outsized one.
        cf = ccs_freq[tri]
        normcounts[sbs96] = (
            sbs96_counts[sbs96] * genome_freq[tri] / cf
            if cf > 0 and not masked[tri] else 0.0
        )

    # The fraction is the coverage/opportunity-independent shape used to compare
    # mutational spectra. Masked contexts are 0 in normcounts, so they drop out
    # of norm_total on their own: this renormalizes over the surviving contexts.
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

    # Computed once against this sample's own genome composition: whether a
    # context's coverage was reliable enough to correct at all doesn't depend
    # on which target composition (native or human) the correction rescales
    # to, so the same mask is applied to both.
    masked = compute_coverage_mask(ccs_freq, genome_freq, list(genome.keys()))

    normcounts, normfracs = apply_correction(
        sbs96_counts, sbs96_lst, sbs96_to_tri, ccs_freq, genome_freq, masked
    )
    human_normcounts, human_normfracs = apply_correction(
        sbs96_counts, sbs96_lst, sbs96_to_tri, ccs_freq, human_freq, masked
    )

    # ---- Alternative freq_G: position-unique callable composition ----
    # freq_G(c)_new is built from R(c) — each callable position in
    # shared.context.bed.gz counted exactly once, depth ignored (ref_callable) —
    # instead of the whole-reference trinucleotide composition. It asks what the
    # spectrum would look like had the reads sampled every context in proportion
    # to how much callable sequence that context covers. No human-composition
    # variant is computed for this method; the same coverage mask is reused.
    ref_freq = get_trifreq(ref_callable)
    new_normcounts, new_normfracs = apply_correction(
        sbs96_counts, sbs96_lst, sbs96_to_tri, ccs_freq, ref_freq, masked
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
            "masked",
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
                masked[tri],
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
        genome = get_genome_tricounts(reference, tri_lst)
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
