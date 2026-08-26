# BiStro test data

`tests/data/` holds a tiny, real-read smoke-test dataset for BiStro, built from
two production PacBio Revio single-strand CCS duplex samples (`E01`, `F01`).

It is **not** a synthetic/simulated dataset: every read is a real, unmodified
alignment record (sequence, CIGAR, quality scores, and the BAM tags BiStro
actually reads: `rq`, `ec`, `np`, `zm`, `sm`, `NM`). Only two things were
changed relative to the source BAMs:

1. Reads were subset to a handful of ZMWs (duplexes) around two small genomic
   windows, to keep the data small.
2. Both the reads and the reference were remapped onto two short synthetic
   contigs (`test_dnm_locus`, `test_grm_locus`) starting at position 1, so the
   data carries no real chromosome coordinates. `reference.fa.gz` contains the
   real reference bases for exactly those two windows.

| File | Contents |
|---|---|
| `E01.bam` / `F01.bam` (+ `.bai`) | Reads for the two synthetic loci, per sample. |
| `reference.fa.gz` (+ `.fai`, `.gzi`) | The matching mini reference (2 contigs). |

## What's in each locus

- **`test_dnm_locus`**: contains 2 de novo mutations (DNMs), both private to
  `E01` (ZMWs `24641657` and `44568641`), plus background duplex coverage from
  both samples. Running `preprocess` + `somatic` on `E01` alone (or on
  `E01`+`F01` together) should classify both as `DNM`.
- **`test_grm_locus`**: contains a cluster of germline SNVs shared by both
  `E01` and `F01` (real ZMWs carrying the alt allele in both samples), plus
  background duplex coverage. Running `somatic` on both samples together
  should classify these as `GRM`.

Note the dataset is tiny and calling it with BiStro's default QC thresholds
requires relaxing several of them (mapping quality, base quality, read
quality, etc.) -- it's meant as a pipeline wiring check, not a statistical
validation.
