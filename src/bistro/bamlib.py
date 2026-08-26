"""
Per-read helpers used by caller.py: quality-based rejection of individual CCS
reads, base-quality masking near read ends and indels, and building the
mutation record rows for every candidate position found in a fwd/rev duplex
pair.
"""

#import sys
import pysam
#import os
#import pandas as pd
import numpy as np
# import argparse
# import psutil
from . import reportlib
#reportlib import METRICS

def get_seq_identity(read: pysam.AlignedSegment):
    return round(1 - float(read.get_tag("NM")) / read.query_length, 3)
    
def get_softclipping(read: pysam.AlignedSegment):
    return 1 - float(read.query_alignment_length / read.query_length)

def discard_low_qual_read(read: pysam.AlignedSegment,
                    report:reportlib.METRICS,
                    min_mapq,
                    min_ec,
                    min_rq,
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
    
    elif read.get_tag("rq") < min_rq:
        report.num_discarded += 1
        report.disc_mean_qual += 1
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
    
    
## mask terminal read ends by setting their quality to 0 ##
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


def remove_useless_tags(read: pysam.AlignedSegment):
    for tag in ("sa", "sx", "sn", "ip","pw"):
        if read.has_tag(tag):
            read.set_tag(tag, None)
        #else:
            #print(f"TAG {tag} NOT PRESENT")
    return read


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
                      aln1, aln2,
                      germ_positions
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

            if r in germ_positions:
                if t == "m":
                    continue
                else:
                    t = "g"
                    
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
            #print(np.arange(max(p1 - mismatch_window_len, 0), min(ln1, p1+mismatch_window_len+1), dtype=int))
            ##print(p1, p2, mismatch_window_len, ln1, ln2)
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
            #print(mut)
            output.append(mut)

    return output