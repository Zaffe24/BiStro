#!/bin/bash

### small script to save BED file of low-complexity regions from FASTA genome ###
### sdusts is required ###

conda activate bistro

set -euo pipefail

## example for GCRm39 genome
IN="path/to/GCF_000001635.27_GRCm39_genomic.fa.gz"
FAI="path/to/GCF_000001635.27_GRCm39_genomic.fa.gz.fai"

# output name
OUT="path/to/GRCm39_genomic.sdust.bed.gz"
OUT2="path/to/GRCm39_genomic.sdust.3bp_slop.bed.gz"

sdust $IN | bgzip > $OUT
bedtools slop -i $OUT -g $FAI -b 3 | bgzip > $OUT2 

if [ -e $OUT2 ]; then
    rm -f $OUT
fi