"""
`BiStro preprocess` — extracts candidate mutations from CCS reads. For each
contig (in parallel), walks every ZMW's aligned fwd/rev CCS read pair,
classifies each jointly-covered position as reference, single-strand
mismatch, or double-strand (de novo) substitution, tracks per-position read
depth, and writes the resulting mutation and context records to per-contig
BED fragments that main() then merges into the sample's .muts.bed.gz /
.context.bed.gz outputs. The `somatic` subcommand (sharelib.py) later decides
which of these candidates are true de novo mutations.
"""

# import sys
# import os

import pysam
#import pandas as pd
import numpy as np
#import psutil
#from datetime import datetime
#from array import array
#from dataclasses import dataclass
import os
import random
import time
from functools import partial
import multiprocessing as mp
pysam.set_verbosity(0)

from . import reportlib, utilib, bamlib, vcflib, contiglib
#from reportlib import METRICS

def call_somatic_mutations(region_tuple,
                           bam,
                           reference,
                           germline_vcf,
                           min_mapq,
                           min_sequence_identity,
                           min_bq,
                           min_qlen,
                           max_qlen,
                           trim_ends,
                           indels_window,
                           mismatch_window_len,
                           min_ec,
                           min_rq,
                           max_softclipping, 
                           z_prob,
                           check_mem_usage,
                           do_not_collapse,
                           min_gq,
                           min_depth):
    
    #output_dict = dict()
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
                           min_rq,
                           max_softclipping,
                           min_gq, min_depth
                           )
    
    CONTIG, START, END = region_tuple
    region_len = END - START
    report.total_length_ref += region_len
    report.chrom_name = CONTIG

    depth = np.zeros(region_len, dtype=np.uint16)
    zmw_dict = dict()
    chr_mutations = []

    germ_positions = set()
    if germline_vcf:
        germ_positions = vcflib.process_germline_vcf(germline_vcf, CONTIG, min_gq)

    with pysam.AlignmentFile(bam, "rb") as bam_file, \
    pysam.FastaFile(reference) as fasta:
        ref_seq = fasta.fetch(CONTIG).upper()

        for ss_strand in bam_file.fetch(CONTIG, START, END):
            # ------------------------------------------------
            # Applying quality filters
            # ------------------------------------------------
            report.num_reads +=1
            #strand, zmw= ss_strand.query_name.split("/")[-1], ss_strand.get_tag("zm")
            #print(ss_strand.query_name, strand, zmw)

            if bamlib.discard_low_qual_read(ss_strand, report, min_mapq, min_ec, 
                                     min_rq, min_qlen, max_qlen, max_softclipping, 
                                     min_sequence_identity):
                continue
            # ------------------------------------------------
            # decrease memory burden before storing reads
            # ------------------------------------------------
            ss_strand = bamlib.remove_useless_tags(ss_strand)
            strand, zmw= ss_strand.query_name.split("/")[-1], ss_strand.get_tag("zm")
            #print(strand, zmw)
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

            # ------------------------------------------------
            # Mask read regions (around indels & extremeties)
            # ------------------------------------------------
            read1, read2 = bamlib.mask_ends(read1, trim_ends), bamlib.mask_ends(read2, trim_ends)
            read1, read2 = bamlib.mask_indels(read1, indels_window), bamlib.mask_indels(read2, indels_window)

            s1, s2 = read1.query_sequence.upper(), read2.query_sequence.upper()
            q1, q2 = np.array(read1.query_qualities, dtype=np.uint8), np.array(read2.query_qualities, dtype=np.uint8)
            #ln1, ln2 = read1.query_length, read2.query_length
            aln1, aln2 = read1.get_aligned_pairs(matches_only=True), read2.get_aligned_pairs(matches_only=True)
            len1, len2 = len(aln1), len(aln2)

            ####################
            # if report.num_zmw in np.arange(49000, 50000):
            #     if CONTIG == "NC_000069.7":
            #         print(report.num_zmw)
            #         print(read1.to_string())
            #         print(read2.to_string())

            ######################
            # ------------------------------------------------
            # Walking through the duplex and record positions of interest
            # ------------------------------------------------
        #if do_not_collapse:
            context_positions = []
            list_qpos1=[]
            list_qpos2=[]
            list_ref1=[]
            list_type=[]
            mismatch_list=[]
            i=j=0

            while i < len1 and j < len2:
                qpos1, ref1 = aln1[i]
                qpos2, ref2 = aln2[j]
                
                if ref1 < ref2:
                    i += 1
                    continue
                if ref2 < ref1:
                    j += 1
                    continue
                
                #idx = ref1 - dupl_start  # offset into masked_region

                if q1[qpos1] >= min_bq and q2[qpos2] >= min_bq:
                    ref_base = ref_seq[ref1]
                    b1 = s1[qpos1]
                    b2 = s2[qpos2]
                    m1 = (b1 == ref_base)
                    m2 = (b2 == ref_base)

                    if m1 and m2: ## reference bp
                        context_positions += [ref1] ## only consider positions within the target region
                        ### randomly save non-mutated position to have a baseline for cofounding factors
                        if z_prob >0 :
                            if random.randint(1, z_prob) == 1:
                                list_qpos1+=[qpos1]
                                list_qpos2+=[qpos2]
                                list_ref1+=[ref1]
                                list_type+=["z"]
                            ## store all this info after closing the for loop[CONTIG,ref1,ref1+1,zmw,triplet,ref_base,b1,q1[qpos1],qpos1,ln1,b2, q2[qpos2], qpos2, ln2, "z"])
                        
                    elif m1 ^ m2: ##ss mutation
                        context_positions += [ref1]
                        list_qpos1+=[qpos1]
                        list_qpos2+=[qpos2]
                        list_ref1+=[ref1]
                        list_type+=["m"]

                        mismatch_list += [ref1]
                    #triplet = ref_seq[ref1 -1 : ref1+2] 
                        #mut_dict_per_read[ref1]=[CONTIG,ref1,ref1+1,zmw,triplet,ref_base,b1,q1[qpos1],qpos1,ln1,b2, q2[qpos2], qpos2, ln2, "f" if b2 else "r"]

                    elif b1 == b2: #ds mutation
                        if ref_base == "N":
                            i += 1
                            j += 1
                            continue
                        context_positions += [ref1]
                        list_qpos1+=[qpos1]
                        list_qpos2+=[qpos2]
                        list_ref1+=[ref1]
                        list_type+=["d"]

                        #triplet = ref_seq[ref1 -1 : ref1+2] 
                        #mut_dict_per_read[ref1]=[CONTIG,ref1,ref1+1,zmw,triplet,ref_base,b1,q1[qpos1],qpos1,ln1,b2, q2[qpos2], qpos2, ln2, "d"]
                i += 1
                j += 1

            # ------------------------------------------------
            # Positions within mismatch_window_len of a mismatch are
            # error-prone and must not be counted as callable.
            # ------------------------------------------------
            if mismatch_list:
                blacklist = set()
                for r in mismatch_list:
                    blacklist.update(range(r - mismatch_window_len, r + mismatch_window_len + 1))
                callable_positions = [p for p in context_positions if p not in blacklist]
            else:
                callable_positions = context_positions

            report.callable_bps += len(callable_positions)
            depth[callable_positions] += 1 ## update count of all investigable genome positions
            
            mutations_in_read = bamlib.process_mutations(read1, read2, 
                                                    list_qpos1, list_qpos2,
                                                    list_ref1, list_type, mismatch_list,
                                                    s1, s2, q1, q2, 
                                                    mismatch_window_len, CONTIG,
                                                    zmw, ref_seq,
                                                    aln1, aln2,
                                                    germ_positions)
            
            if mutations_in_read:
                chr_mutations += mutations_in_read


    #vcflib.write_mut_bed(mut_file, chr_mutations)
    #vcflib.write_report(report)


        report.num_discarded += len(zmw_dict.keys())
        report.disc_lack_compl += len(zmw_dict.keys())
        del zmw_dict

        ## add depth to each mutation
        for mut in chr_mutations:
            r = mut[1]
            mut += [depth[r]]

        utilib.cprint(f"[{wrk}]\tMEM OCCUPIED: {utilib.get_stats()}\tZMW counter for {CONTIG}: {report.num_zmw}")
        #utilib.cprint(f"REPORT OF {CONTIG}\n", color="yellow")
        utilib.cprint(f"{report}\n", color="yellow")

    return report, chr_mutations, depth


def write_frags(mut_list, ctx_list, region_tuple, reference, out_dir, sample, min_depth, max_depth):

    CONTIG, START, END = region_tuple
    mut_frag = os.path.join(out_dir, f"{sample}.muts.{CONTIG}.frag.bgz")
    ctx_frag = os.path.join(out_dir, f"{sample}.context.{CONTIG}.frag.bgz")

    vcflib.write_mut_one(mut_frag, mut_list, min_depth, max_depth)
    vcflib.write_context_one(ctx_frag, CONTIG, START, END, ctx_list, reference, min_depth, max_depth)

    return mut_frag, ctx_frag


def main(bam,
            reference,
            germline_vcf,
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
            min_rq,
            max_softclipping,
            z_prob,
            check_mem_usage,
            do_not_collapse,
            min_gq,
            min_depth,
            subtract_bed):

    checkpoints = range(0, 1000_000, check_mem_usage) if check_mem_usage>0 else []
    clock_start = time.time() / 60

    # [(contig, start, end), ...] to call mutations on
    contigs_list = contiglib.get_contig_list(reference, region, exclude)
    if region:
        utilib.cprint(f"Calling Mutations only on Chromosomes: {region}.", color="green")
    if exclude:
        utilib.cprint(f"Excluding Chromosomes: {exclude}", color="green")

    # Bind everything except region_tuple; map the rest over the contigs.
    # out_dir/sample are bound too, so each worker can write its own fragments.
    worker = partial(call_somatic_mutations,
                    bam=bam,
                    reference=reference,
                    germline_vcf=germline_vcf,
                    min_mapq=min_mapq,
                    min_sequence_identity=min_sequence_identity,
                    min_bq=min_bq,
                    min_qlen=min_qlen,
                    max_qlen=max_qlen,
                    trim_ends=trim_ends,
                    indels_window=indels_window,
                    mismatch_window_len=mismatch_window_len,
                    min_ec=min_ec,
                    min_rq=min_rq,
                    max_softclipping=max_softclipping,
                    z_prob=z_prob,
                    check_mem_usage=checkpoints,
                    do_not_collapse=do_not_collapse,
                    min_gq=min_gq,
                    min_depth = min_depth
                    )

    with mp.Pool(processes=nproc) as pool:
        # results are in the SAME ORDER as contigs_list (pool.map preserves order)
        results = pool.map(worker, contigs_list)

    # ---- merge ----
    reports   = [r[0] for r in results]
    mut_list = [r[1] for r in results]   # already in contig order
    ctx_list = [r[2] for r in results]
    merged_report = reportlib.merge_reports(reports)
    utilib.cprint(merged_report)
    print("")
    
    del results

    max_thr = reportlib.get_maximum_coverage(merged_report.callable_bps, merged_report.total_length_ref)
    
    worker2 = partial(write_frags,
                      min_depth = min_depth,
                      max_depth = max_thr,
                      sample=sample,
                      reference=reference,
                      out_dir=out_dir)
    
    with mp.Pool(processes=nproc) as pool2:
        # results are in the SAME ORDER as contigs_list (pool.map preserves order)
        results2 = pool2.starmap(worker2, zip(mut_list, ctx_list, contigs_list))


    mut_frags = [r[0] for r in results2]
    ctx_frags = [r[1] for r in results2]

    # ---- assemble outputs from the per-contig fragments ----
    # mut-bed: prepend the shared header, concat fragments (no tabix, matching write_mut_bed).
    mut_file = os.path.join(out_dir, f"{sample}.muts.bed.gz")
    mut_header = "\n".join(vcflib.mut_bed_header(sample, reference, merged_report.filtering_params + (max_thr,) ))
    uno = time.time() / 60
    vcflib.finalize_bgzf(mut_file, mut_header, mut_frags, tabix_bed=False)
    dos = time.time() / 60
    utilib.cprint(f"Writing {os.path.basename(mut_file)} completed. Time elapsed {round(dos - uno,2)} min.")
    if subtract_bed:
        vcflib.apply_subtract_bed(mut_file, subtract_bed, tabix_bed=False)


    vcflib.write_report(out_dir, sample, merged_report)
    vcflib.write_merged_coverages(merged_report, out_dir, sample)

    # context: single-line header, concat fragments, then tabix-index.
    context_file = os.path.join(out_dir, f"{sample}.context.bed.gz")
    header_ctx = f"##Minimum depth: {min_depth}; Maximum depth (AVG+4√AVG): {max_thr}\n##CONTIG\tSTART\tEND\tREF\tDEPTH\n"
    uno = time.time() / 60
    vcflib.finalize_bgzf(context_file, header_ctx, ctx_frags, tabix_bed=True)
    dos = time.time() / 60
    utilib.cprint(f"Writing {os.path.basename(context_file)} completed. Time elapsed {round(dos - uno,2)} min.")
    if subtract_bed:
        vcflib.apply_subtract_bed(context_file, subtract_bed, tabix_bed=True)

    clock_end = time.time() / 60
    utilib.cprint(f"BiStro took {round(clock_end - clock_start, 2)} minutes to pre-process candidate mutations in sample {sample}", color="green")
