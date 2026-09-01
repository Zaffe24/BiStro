"""
Output I/O for BiStro: reads germline VCF calls to filter out during
preprocessing, builds the BED headers/column glossaries for the preprocess
and somatic outputs, and provides the bgzf writers that assemble caller.py's
per-contig fragments into the final .muts.bed.gz / .context.bed.gz /
.shared.muts.bed.gz files.
"""

import os
#import sys
import shutil
import subprocess
import pysam
#import numpy as np
import json
from datetime import date
from dataclasses import asdict
from . import utilib, __main__
import time
import gzip
import pandas as pd
def process_germline_vcf(germline_vcf, region, min_gq):
    """Reads Germline VCF from DeepVariant and returns a set of positions with germline mutations, to be used for filtering somatic calls."""
    germline_positions = set()
    if germline_vcf:
        with pysam.VariantFile(germline_vcf) as vcf:
            for record in vcf.fetch(region=region):
                if record.qual >= min_gq:
                    germline_positions.add(record.pos - 1) # VCF are 1-based
    return germline_positions



def mut_bed_header(sample, reference, filtering_params):
    
    min_mapq, min_sequence_identity, min_bq, min_qlen, max_qlen, \
        trim_percentage, indels_window, mismatch_window_len, min_ec,min_rq, \
            max_softclipping, min_gq,min_depth, max_depth = filtering_params

    filters = (f"min_mapq={min_mapq};min_rq={min_rq};min_ec={min_ec};"
               f"min_qlen={min_qlen};max_qlen={max_qlen};"
               f"max_softclip={max_softclipping};min_identity={min_sequence_identity};"
               f"min_bq={min_bq};trim_pct={trim_percentage};"
               f"indels_window={indels_window};mismatch_window={mismatch_window_len};min_gq={min_gq};min_depth={min_depth};max_depth={max_depth}")

    return [f"##fileformat=BiStro-bed-v{__main__.__version__}",
            f"##sample={sample}",
            f"##reference={os.path.basename(reference)}",
            f"##date={date.today().isoformat()}",
             "##coordinates=0-based,half-open (BED)",
            f"##filters={filters}",
             "##",
             "## Column glossary (in output order)",
             "## 1  CONTIG   reference contig name",
             "## 2  START    0-based start of the position",
             "## 3  END      half-open end (START+1)",
             "## 4  ZMW      Zero-Mode Waveguide id (duplex pair)",
             "## 5  CTX      reference triplet centered on the position",
             "## 6  REF      reference base",
             "## 7  STRAND_1 fwd-strand call, colon-delimited (see sub-fields below)",
             "## 8  STRAND_2 rev-strand call, colon-delimited (see sub-fields below)",
             "## 9  NOB      number of non-overlapping bases between the two reads",
             "## 10 N_MIS    mismatches within the QB window (default 20)",
             "## 11 TYPE     g=ds-germline s=ds-somatic  z=non-mutated random  m=mismatch",
             "## 12 DEPTH    read coverage at that position",
             "##",
             "## STRAND_1 / STRAND_2 sub-fields (colon-delimited):",
             "##   ALT:QUAL:QPOS:EC:SM:QLEN:SIM:MEAN:SD",
             "##   ALT=alt base  QUAL=base quality  QPOS=position in read  EC=avg full subpasses",
             "##   SM=subpasses supporting position  QLEN=read length  SIM=similarity to template",
             "##   MEAN=mean QB in window  SD=stddev QB in window",
             "##",
             "\t".join(["#CONTIG", "START", "END", "ZMW", "CTX", "REF",
                        "STRAND_1", "STRAND_2", "NOB", "N_MIS", "TYPE", "DEPTH"]) + "\n"
        ]


def write_report(dir, sample, report):
    jsonF= os.path.join(dir, f"{sample}.report.json")
    txtF = os.path.join(dir, f"{sample}.report.txt")
    
    with open(jsonF, "w") as f:
        json.dump(asdict(report), f)
    
    with open(txtF, "w") as f:
        f.write(str(report))    

    utilib.cprint(f"Somatic Mutation Calling report saved to: {os.path.basename(txtF)} {os.path.basename(jsonF)}")


def write_merged_coverages(merged, out_dir, sample):
    file_name = os.path.join(out_dir, f"{sample}_coverage_report.tsv")
    lengths = merged.total_length_ref
    callable = merged.callable_bps
    chroms = merged.chrom_name

    with open(file_name, "w") as f:
        f.write("## CHROM\tLENGTH\tCALLABLE_BPS\tCOVERAGE\tINCLUDED\n")
        for n, c in enumerate(chroms):
            coverage = round(callable[n]/lengths[n],2) if lengths[n] > 0 else 0
            #f.write(f"{c}\t{lengths[n]}\t{callable[n]}\t{coverage}\t{"PASS" if lengths[n]>= min_chr_len and callable[n] > 0 else "NO" }\n")
            f.write(f"{c}\t{lengths[n]}\t{callable[n]}\t{coverage}\n")
    
    utilib.cprint(f"Report of Chromosome Coverages saved to: {os.path.basename(file_name)}")


def write_mut_one(path, mut_list, min_depth, max_depth):
    """Write one contig's mutations as a headerless bgzf fragment."""
    with pysam.BGZFile(path, "wb") as bed:
        for mut in mut_list:
            if mut[-1] >= min_depth and mut[-1] <= max_depth:
                str_mut = [str(x) for x in mut]
                block1 = ":".join(str_mut[6:15])
                block2 = ":".join(str_mut[15:24])
                bed.write(("\t".join(x for x in (str_mut[:6]+[block1]+[block2]+str_mut[24:])) + "\n").encode())


def write_context_one(path, contig, start, end, depth, reference, min_depth, max_depth):
    """Write one contig's per-position context as a headerless bgzf fragment."""
    if min_depth >= max_depth:
        raise ValueError(f"Minimum depth threshold ({min_depth}) cannot be higher than maximum threhsold ({max_depth})")
        utilib.exit(1)
    with pysam.FastaFile(reference) as fasta, \
         pysam.BGZFile(path, "wb") as bed:
        ref_seq = fasta.fetch(contig).upper()
        for pos, d in enumerate(depth):
            if pos < end - 1 and pos > start:
                if d >= min_depth and d <= max_depth:
                    triplet = ref_seq[pos - 1:pos + 2]
                    if "N" not in triplet:
                        bed.write(f"{contig}\t{pos}\t{pos+1}\t{triplet}\t{d}\n".encode())


def apply_subtract_bed(final_path, subtract_bed, tabix_bed=False):
    """Remove intervals in subtract_bed from a bgzf BED file using bedtools subtract."""
    tmp = final_path + ".subtract.tmp.gz"
    cmd = ["bash", "-c",
           f"bedtools subtract -header -a {final_path} -b {subtract_bed} | bgzip -c > {tmp}"]
    subprocess.run(cmd, check=True)
    os.replace(tmp, final_path)
    if tabix_bed:
        pysam.tabix_index(final_path, preset="bed", force=True)


def finalize_bgzf(final_path, header, frag_paths, tabix_bed=False):
    """Assemble a single bgzf file from per-contig fragments.

    Writes `header` (str) as the first bgzf member, then appends each fragment's
    raw bgzf bytes in the given order. Fragment files are deleted after merging.
    Set tabix_bed=True to build a .tbi index on the result.
    """
    with pysam.BGZFile(final_path, "wb") as bed:
        bed.write(header.encode())
    with open(final_path, "ab") as out:
        for p in frag_paths:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(p)
    if tabix_bed:
        pysam.tabix_index(final_path, preset="bed", force=True)




def mut_header_shared(samples):

    return ["##",
             "## Column glossary (in output order)",
             "## 1  CONTIG   reference contig name",
             "## 2  START    0-based start of the position",
             "## 3  END      half-open end (START+1)",
             "## 4  ZMW      Zero-Mode Waveguide id (duplex pair)",
             "## 5  CTX      reference triplet centered on the position",
             "## 6  REF      reference base",
             "## 7  STRAND_1 fwd-strand call, colon-delimited (see sub-fields below)",
             "## 8  STRAND_2 rev-strand call, colon-delimited (see sub-fields below)",
             "## 9  NOB      number of non-overlapping bases between the two reads",
             "## 10 N_MIS    mismatches within the QB window (default 20)",
             "## 11 TYPE     g=ds-germline s=ds-somatic  z=non-mutated random  m=mismatch",
             "## 12 DEPTH    read coverage at that position",
             "## 13 SHARED    Number of samples in which the mutation is observed",
             "## 14 ANNOTATION    Revised annotation for TYPE categories",
             "## 15 COUNTS    Allele counts in the OTHER samples' BAMs at this position",
             "##",
             "## STRAND_1 / STRAND_2 sub-fields (colon-delimited):",
             "##   ALT:QUAL:QPOS:EC:SM:QLEN:SIM:MEAN:SD",
             "##   ALT=alt base  QUAL=base quality  QPOS=position in read  EC=avg full subpasses",
             "##   SM=subpasses supporting position  QLEN=read length  SIM=similarity to template",
             "##   MEAN=mean QB in window  SD=stddev QB in window",
             "##",
             "## COUNTS sub-fields: one A:C:G:T field per other sample, comma-delimited,",
             "##   in TOTAL SAMPLES order excluding this sample. Counts pass --min_mapq /",
             "##   --min_baseq and exclude secondary, supplementary and duplicate reads.",
             "##   '.' = not computed (non-DNM row, or --bams not supplied).",
             "##   A de novo whose ALT is supported here is re-annotated FPD.",
             "##",
            f"##  TOTAL SAMPLES: {" ".join(samples)}",
             "##",
             "\t".join(["#CONTIG", "START", "END", "ZMW", "CTX", "REF",
                        "STRAND_1", "STRAND_2", "NOB", "N_MIS", "TYPE", "DEPTH", "SHARED","ANNOTATION",
                        "COUNTS"]) + "\n"
        ]

def mut_header_single(samples):

    return ["##",
             "## Column glossary (in output order)",
             "## 1  CONTIG   reference contig name",
             "## 2  START    0-based start of the position",
             "## 3  END      half-open end (START+1)",
             "## 4  ZMW      Zero-Mode Waveguide id (duplex pair)",
             "## 5  CTX      reference triplet centered on the position",
             "## 6  REF      reference base",
             "## 7  STRAND_1 fwd-strand call, colon-delimited (see sub-fields below)",
             "## 8  STRAND_2 rev-strand call, colon-delimited (see sub-fields below)",
             "## 9  NOB      number of non-overlapping bases between the two reads",
             "## 10 N_MIS    mismatches within the QB window (default 20)",
             "## 11 TYPE     g=ds-germline s=ds-somatic  z=non-mutated random  m=mismatch",
             "## 12 DEPTH    read coverage at that position",
             "## 13 ANNOTATION    Revised annotation for TYPE categories",
             "##",
             "## STRAND_1 / STRAND_2 sub-fields (colon-delimited):",
             "##   ALT:QUAL:QPOS:EC:SM:QLEN:SIM:MEAN:SD",
             "##   ALT=alt base  QUAL=base quality  QPOS=position in read  EC=avg full subpasses",
             "##   SM=subpasses supporting position  QLEN=read length  SIM=similarity to template",
             "##   MEAN=mean QB in window  SD=stddev QB in window",
             "##",
            f"##  TOTAL SAMPLES: {" ".join(samples)}",
             "##",
             "\t".join(["#CONTIG", "START", "END", "ZMW", "CTX", "REF",
                        "STRAND_1", "STRAND_2", "NOB", "N_MIS", "TYPE", "DEPTH","ANNOTATION"]) + "\n"
        ]


def copy_header(src, dst, max_lines=5):
    i = 0
    for line in src:
        if not line.startswith(b"#") or i > max_lines:
            break
        i += 1
        dst.write(line)


def write_mut_shared(df, input, samples, output, single=False):

    with gzip.open(input, "rb") as src, pysam.BGZFile(output, "wb") as dst:
        if single:
            copy_header(src, dst)
            dst.write(("\n".join(mut_header_single(samples))).encode())
        else:
            copy_header(src, dst)
            dst.write(("\n".join(mut_header_shared(samples))).encode())
        dst.write(df.to_csv(sep="\t", header=False, index=False).encode())


def get_context_header(context_bed, max_lines=1):
    i=0
    out=[]
    with gzip.open(context_bed, "rb") as src:
        for line in src:
            if not line.startswith(b"#") or i > max_lines:
                break
            out+=[line]
            i += 1
    return out


def write_mut_rate(lines, out_file):
    header ="##SAMPLE\tMUT_RATE\tPOISSON_CI_LO\tPOISSON_CI_HI\tTOT_MUTS\tTOT_BPS\n"
    with open(out_file,"w") as f:
        f.write(header)
        for l in lines:
            f.write(l+"\n")