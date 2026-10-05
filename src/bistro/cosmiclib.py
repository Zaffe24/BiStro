#!/usr/bin/env python
"""
`BiStro cosmic` — two-tier COSMIC SBS signature attribution across samples,
modified from the "Mutational signature attribution" step of Pham et al.,
"A comprehensive atlas of somatic mutation rates and mutational signatures in
normal human cells", bioRxiv 10.64898/2026.08.28.747772.

Attribution (refitting) takes a FIXED set of reference COSMIC signatures and
finds, per sample, the non-negative combination that best reconstructs its
SBS96 spectrum (SigProfilerAssignment: NNLS + forward/backward selection).
Plain refitting over-assigns spurious low exposures and is unstable for
samples with few mutations, so it runs in two tiers:

  TIER 1 (per group): bootstrap every sample's spectrum --nboot times, refit
    every replicate against the full COSMIC set, and validate a signature for
    a group when its lower 95% CI (2.5th percentile) is > 0 in at least
    --min_samples samples of that group. With --merge_groups_tier1, tier 1 runs
    on one pooled pseudo-sample per group (sum of its samples) instead.
  TIER 2 (final): refit every real sample against ONLY its group's validated
    signatures (connected_sigs=False, so nothing else can be assigned).

Groups ("cell types") come from the pipeline's samples TSV
(BAM<TAB>SAMPLE[<TAB>GROUP]); without a GROUP column all samples form one
group "all". SigProfilerAssignment 1.1.x has no bootstrap option, so the
replicates are generated here and fitted in one cosmic_fit call (every input
column is fitted independently), then split back by sample.
"""

import os
import itertools

import numpy as np
import pandas as pd

from . import utilib, sharelib

# The canonical 96 SBS mutation types, in the pyrimidine-centred order every
# SigProfiler tool expects: 6 substitutions x 4 x 4 flanking bases, e.g. "A[C>T]G".
SUBS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
SBS96 = [f"{u}[{s}]{d}" for s in SUBS
         for u, d in itertools.product("ACGT", repeat=2)]


def fail(msg):
    utilib.cprint(f"ERROR: {msg}", color="red")
    utilib.exit(1)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_matrix(normcounts_files, column):
    # MutationType x samples matrix from each sample's normcounts.tsv `column`,
    # reindexed to the canonical 96-type order. Sample name = file prefix
    # (A01.normcounts.tsv -> A01), the same convention `somatic` uses.
    cols = {}
    for path in normcounts_files:
        df = pd.read_csv(path, sep="\t", comment="#")
        if column not in df.columns:
            fail(f"column '{column}' not in {path} (available: {', '.join(df.columns)})")
        sample = sharelib.get_sample_id(path)
        if sample in cols:
            fail(f"sample '{sample}' given twice in -i")
        cols[sample] = df.set_index("sbs96")[column].reindex(SBS96)
    mat = pd.DataFrame(cols)
    mat.index.name = "MutationType"
    # a hole in the spectrum would silently corrupt every fit
    if mat.isna().any().any():
        bad = mat.columns[mat.isna().any()].tolist()
        fail(f"column '{column}' is missing SBS96 types or holds NA values for sample(s): {bad}")
    return mat


def load_groups(groups_tsv, samples):
    # {sample -> group} from the pipeline's samples TSV (BAM<TAB>SAMPLE[<TAB>GROUP],
    # no header, '#' comments allowed). No file or no GROUP column -> one group "all".
    if not groups_tsv:
        return {s: "all" for s in samples}
    tab = pd.read_csv(groups_tsv, sep="\t", header=None, comment="#", dtype=str)
    if tab.shape[1] < 3:
        utilib.cprint(f"No GROUP column in {groups_tsv}: all samples form one group 'all'.", color="yellow")
        return {s: "all" for s in samples}
    sample2group = dict(zip(tab[1].str.strip(), tab[2].str.strip()))
    missing = [s for s in samples if pd.isna(sample2group.get(s))]
    if missing:
        fail(f"no GROUP in {groups_tsv} for sample(s): {missing}")
    return {s: sample2group[s] for s in samples}


# ---------------------------------------------------------------------------
# SigProfilerAssignment wrappers (imported lazily: only `cosmic` needs it)
# ---------------------------------------------------------------------------
def ref_signature_file(genome, cosmic_version):
    # The COSMIC table bundled with SigProfilerAssignment, e.g.
    # .../data/Reference_Signatures/mm39/COSMIC_v3.6_SBS_mm39.txt. Tier 1 needs
    # the full signature list; tier 2 passes a column subset as signature_database.
    import SigProfilerAssignment
    base = os.path.join(os.path.dirname(SigProfilerAssignment.__file__), "data", "Reference_Signatures", genome)
    p = os.path.join(base, f"COSMIC_v{cosmic_version}_SBS_{genome}.txt")
    if not os.path.exists(p):
        fail(f"reference signature file not found: {p} (check --genome / --cosmic_version, or list {base})")
    return p


def cosmic_fit(samples_tsv, outdir, genome, cosmic_version,
               signature_database=None, connected_sigs=True, cpu=-1, plotting=False):
    # One SigProfilerAssignment refit; EVERY column of samples_tsv is fitted
    # independently, so a whole bootstrap ensemble goes in one call.
    #   signature_database: None -> full COSMIC set for (genome, cosmic_version);
    #                       path -> ONLY the signatures in that file (tier 2).
    #   connected_sigs:     True for tier 1; False for tier 2 so nothing sneaks back in.
    # Returns the Activities table: rows = input columns, cols = signatures (counts).
    from SigProfilerAssignment import Analyzer
    Analyzer.cosmic_fit(
        samples=samples_tsv,
        output=outdir,
        input_type="matrix",          # a 96 x N table, not VCFs
        context_type="96",            # SBS96
        genome_build=genome,
        cosmic_version=float(cosmic_version),
        signature_database=signature_database,
        connected_sigs=connected_sigs,
        cpu=cpu,
        make_plots=plotting,          # skip the (slow) plotting for bootstrap runs
        sample_reconstruction_plots=plotting,
        verbose=False,
    )
    act = os.path.join(outdir, "Assignment_Solution", "Activities",
                       "Assignment_Solution_Activities.txt")
    return pd.read_csv(act, sep="\t", index_col=0)


def bootstrap_columns(mat, nboot, seed, poisson=False):
    # 96 x S matrix -> 96 x (S*nboot) bootstrap replicates; replicate k of sample S
    # is column "S@@b{k}" ("@@" maps replicates back to their parent sample).
    #   multinomial (default): c* ~ Multinomial(N, c/N), N = c.sum() fixed
    #   poisson   (--poisson): c*_i ~ Poisson(c_i) per channel, total not fixed
    # One Generator seeded once, so the whole run is reproducible.
    rng = np.random.default_rng(seed)
    out = {}
    for s in mat.columns:
        c = mat[s].to_numpy(dtype=float)
        tot = c.sum()
        # a sample with 0 mutations gets a flat spectrum so multinomial() does not
        # divide by zero (its fits are still meaningless)
        p = c / tot if tot > 0 else np.full(len(c), 1.0 / len(c))
        for k in range(nboot):
            if poisson:
                out[f"{s}@@b{k}"] = rng.poisson(c)
            else:
                out[f"{s}@@b{k}"] = rng.multinomial(int(round(tot)), p)
    b = pd.DataFrame(out, index=mat.index)
    b.index.name = "MutationType"
    return b


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(normcounts_files, groups_tsv, out_dir, genome, cosmic_version, column,
         nboot, min_samples, merge_groups_tier1, bootstrap_tier2, poisson, seed, cpu):

    os.makedirs(out_dir, exist_ok=True)

    mat = load_matrix(normcounts_files, column)
    sample2group = load_groups(groups_tsv, mat.columns)
    groups = sorted(set(sample2group.values()))
    utilib.cprint(f"{mat.shape[1]} samples ('{column}' column), {len(groups)} group(s): {groups}", color="green")

    # --merge_groups_tier1: tier 1 runs on one pooled pseudo-sample per group (the
    # sum of its samples). Tier 2 below still refits every real sample.
    if merge_groups_tier1:
        group_labels = [sample2group[s] for s in mat.columns]
        tier1_mat = mat.T.groupby(group_labels).sum().T
        tier1_mat.index.name = "MutationType"
        tier1_sample2group = {g: g for g in tier1_mat.columns}
        utilib.cprint(f"--merge_groups_tier1: tier 1 runs on {tier1_mat.shape[1]} pooled "
                      f"pseudo-sample(s); tier 2 stays per-sample", color="green")
    else:
        tier1_mat = mat
        tier1_sample2group = sample2group

    # reference signature table: index = 96 types, columns = signature names
    ref = pd.read_csv(ref_signature_file(genome, cosmic_version), sep="\t", index_col=0)

    tier1_matrix_tsv = os.path.join(out_dir, "tier1_input_matrix.txt")
    tier1_mat.to_csv(tier1_matrix_tsv, sep="\t")

    # ======================================================================
    # POINT ESTIMATE - one plain refit of the real tier-1 input. Not used to
    # validate (that is purely CI-based); reported next to the CIs.
    # ======================================================================
    point = cosmic_fit(tier1_matrix_tsv, os.path.join(out_dir, "tier1_point"),
                       genome, cosmic_version, cpu=cpu, plotting=True)
    point = point.reindex(columns=ref.columns, fill_value=0.0)

    # ======================================================================
    # TIER 1 - bootstrap every tier-1 column, refit the ensemble in one call,
    #          95% CI per (sample-or-pool, signature).
    # ======================================================================
    boot = bootstrap_columns(tier1_mat, nboot, seed, poisson=poisson)
    boot_tsv = os.path.join(out_dir, "tier1_bootstrap_matrix.txt")
    boot.to_csv(boot_tsv, sep="\t")

    bact = cosmic_fit(boot_tsv, os.path.join(out_dir, "tier1_boot"),
                      genome, cosmic_version, cpu=cpu)
    # cosmic_fit only emits signatures it assigned at least once; pad the rest
    # with zeros so every signature has a full distribution.
    bact = bact.reindex(columns=ref.columns, fill_value=0.0)
    rep_sample = [c.split("@@")[0] for c in bact.index]

    rows = []
    detected = {}       # sample/pool -> signatures whose lower 95% CI > 0
    for s in tier1_mat.columns:
        sub = bact.loc[[i for i, rs in zip(bact.index, rep_sample) if rs == s]]
        lo = sub.quantile(0.025)        # lower 95% CI bound
        hi = sub.quantile(0.975)        # upper 95% CI bound
        mean = sub.mean()
        # lo > 0 <=> non-zero exposure in >= 97.5% of the bootstrap refits
        detected[s] = set(lo.index[lo > 0])
        for sig in ref.columns:
            rows.append(dict(
                sample=s, group=tier1_sample2group[s], signature=sig,
                point=float(point.loc[s, sig]),
                boot_mean=float(mean[sig]),
                lo95=float(lo[sig]), hi95=float(hi[sig]),
                detected=bool(lo[sig] > 0),
            ))
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "tier1_signature_CIs.tsv"), sep="\t", index=False)

    # Per-group validated list: detected in >= --min_samples samples of the group
    # (with --merge_groups_tier1 each group has one pool, so its own flag decides).
    validated = {}
    sel_rows = []
    for grp in groups:
        gsamples = [s for s in tier1_mat.columns if tier1_sample2group[s] == grp]
        counts = {sig: sum(sig in detected[s] for s in gsamples) for sig in ref.columns}
        need = 1 if merge_groups_tier1 else min_samples
        keep = [sig for sig in ref.columns if counts[sig] >= need]
        validated[grp] = keep
        for sig in ref.columns:
            sel_rows.append(dict(group=grp, signature=sig,
                                 n_samples_detected=counts[sig],
                                 n_samples=len(gsamples),
                                 validated=sig in keep))
        # same layout as the COSMIC reference, kept columns only: tier 2's signature_database
        ref[keep].to_csv(os.path.join(out_dir, f"validated_signatures.{grp}.txt"), sep="\t")
        utilib.cprint(f"  group {grp}: {len(keep)} validated -> {keep}", color="green")

    pd.DataFrame(sel_rows).to_csv(
        os.path.join(out_dir, "tier1_validation_summary.tsv"), sep="\t", index=False)

    # ======================================================================
    # TIER 2 - refit each group's real samples against ONLY its validated
    #          signatures (connected_sigs=False: the restriction is absolute).
    # ======================================================================
    final = []
    for grp in groups:
        gsamples = [s for s in mat.columns if sample2group[s] == grp]
        if not validated[grp]:
            utilib.cprint(f"  group {grp}: no validated signatures - skipping tier 2", color="yellow")
            continue

        vfile = os.path.join(out_dir, f"validated_signatures.{grp}.txt")
        gmat_tsv = os.path.join(out_dir, f"tier2_input.{grp}.txt")
        m = mat[gsamples].copy()
        m.index.name = "MutationType"
        m.to_csv(gmat_tsv, sep="\t")

        if bootstrap_tier2:
            # final exposures with 95% CIs; seed + 1 so tier-1 and tier-2 noise are independent
            b = bootstrap_columns(mat[gsamples], nboot, seed + 1, poisson=poisson)
            btsv = gmat_tsv.replace(".txt", ".boot.txt")
            b.to_csv(btsv, sep="\t")
            t2 = cosmic_fit(btsv, os.path.join(out_dir, f"tier2_boot.{grp}"),
                            genome, cosmic_version, signature_database=vfile,
                            connected_sigs=False, cpu=cpu)
            rs = [c.split("@@")[0] for c in t2.index]
            for s in gsamples:
                sub = t2.loc[[i for i, r in zip(t2.index, rs) if r == s]]
                for sig in sub.columns:
                    final.append(dict(sample=s, group=grp, signature=sig,
                                      exposure=float(sub[sig].mean()),
                                      lo95=float(sub[sig].quantile(0.025)),
                                      hi95=float(sub[sig].quantile(0.975))))
        else:
            # one deterministic refit, exposure = absolute counts
            t2 = cosmic_fit(gmat_tsv, os.path.join(out_dir, f"tier2.{grp}"),
                            genome, cosmic_version, signature_database=vfile,
                            connected_sigs=False, cpu=cpu, plotting=True)
            for s in gsamples:
                for sig in t2.columns:
                    final.append(dict(sample=s, group=grp, signature=sig,
                                      exposure=float(t2.loc[s, sig])))

    final_cols = ["sample", "group", "signature", "exposure"] + (["lo95", "hi95"] if bootstrap_tier2 else [])
    final_tsv = os.path.join(out_dir, "final_attribution.tsv")
    pd.DataFrame(final, columns=final_cols).to_csv(final_tsv, sep="\t", index=False)
    utilib.cprint(f"Signature attribution saved to: {final_tsv}", color="green")
