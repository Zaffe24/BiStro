# Changelog

## v1.1.0 — 2026-10-21

Efficiency release: same algorithm and identical outputs (verified file by file against v1.0.0 on
real PacBio data, duplex and `--collapsed` modes; only `##date=` lines and the random TYPE `z` rows
differ), much lower run time and memory, aimed at HPC use.

### preprocess
- Vectorised duplex walk (~18x faster per ZMW); masked base qualities no longer written back into reads.
- One worker per contig does the whole contig (walk, mask, fragment writing); nothing per-base is
  pickled back to the parent and the second process pool is gone. Contigs are dispatched largest first.
- Raw-read coverage cap computed in-process from the reads already fetched (identical to
  `mosdepth -x -Q 20`); the low-complexity + high-coverage mask is applied in memory instead of
  `bedtools subtract` over the assembled files.
- `.context.bed.gz.tbi` written from offsets recorded while writing (no tabix re-scan).

### somatic
- Shared callable space computed per (sample, 16-Mbp range) in parallel from tabix streams, with no
  uncompressed per-base temp files (the cause of the >100 GB memory peaks) and all cores busy.
- Pileup safeguard: each (BAM, site) counted once by a fast CIGAR-based allele counter
  (`allelelib`), identical to the previous per-site `pysam.pileup`.
- Mutation tables loaded once; shared-sample counts computed with vectorised joins.
- Fixed: the single-sample path crashed, and its header listed the sample name letter by letter.

### sbs96
- Vectorised shared-context tally and genome trinucleotide counts (the FASTA scan no longer takes hours).

### other
- New module `fastio` (fast BED parsing/formatting, indexed BGZF writer, `.tbi` builder).
- Subcommands import only their own modules.
- mosdepth and bedtools are no longer required (htslib's `tabix` is).
- Snakefile: removed the `cd $HOME` lines (they could shadow the installed package);
  `call_somatic` default memory lowered to 32 GB.
