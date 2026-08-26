# BiStro

Somatic single-base substitution caller for PacBio Revio single-strand CCS duplex reads.

BiStro is both a standalone command-line tool and a Snakemake pipeline that chains
its four stages together across a whole cohort of samples.

## Installation

```bash
git clone https://github.com/Zaffe24/BiStro.git
cd BiStro
conda env create -f environment.yml
conda activate bistro
```

This installs BiStro itself (in editable mode) plus its Python dependencies
(`pysam`, `pandas`, `numpy`, `scipy`, `matplotlib`, `psutil`) and the external
tools the Snakemake pipeline shells out to (`bgzip`, `tabix`, `snakemake`).

Requires **Python >= 3.12**.

Alternatively, install just the package with pip:

```bash
pip install .
```

Either way, this exposes a `bistro` command (equivalent to `python -m bistro`).

## Usage

BiStro has four subcommands, run in sequence:

| Subcommand | Purpose |
|---|---|
| `bistro preprocess` | Pre-process single-strand CCS reads into candidate mutation calls. |
| `bistro somatic` | Call true somatic mutations and rates from pre-processed candidate calls across samples. |
| `bistro sbs96` | Correct the somatic SBS96 spectrum for trinucleotide opportunity biases. |
| `bistro cosmic` | Cosine similarity of a BiStro SBS96 spectrum against COSMIC signatures. |

Run `bistro <subcommand> --help` for the full list of options for each stage.

### Example: single sample, by hand

```bash
bistro preprocess --bam sample.bam --ref reference.fa.gz --out_dir out/ --sample S01
bistro somatic -i out/S01.muts.bed.gz --bams sample.bam --ref reference.fa.gz
bistro sbs96 --context out/S01.shared.context.bed.gz --muts out/S01.shared.muts.bed.gz \
    --ref reference.fa.gz --out out/S01.normcounts.tsv
bistro cosmic -i out/S01.normcounts.tsv --out out/S01.SBS_cosine_similarity.tsv
```

### Running the full pipeline with Snakemake

The `workflow/` directory holds a Snakemake pipeline that runs all four stages
(plus an optional DeepVariant germline-calling step) across every sample listed
in a samples TSV, ending in a per-sample COSMIC cosine-similarity table.

1. Copy the template config and fill in your paths:
   ```bash
   cp config/config.yaml config/my_run.yaml
   ```
2. Run it directly:
   ```bash
   snakemake --snakefile workflow/Snakefile --configfile config/my_run.yaml --cores 8
   ```
   or, on a SLURM cluster, adapt and submit `workflow/main_Snakemake.sh` (written
   for CESGA; update the module/conda activation lines for your own cluster):
   ```bash
   sbatch workflow/main_Snakemake.sh my_run config/my_run.yaml
   ```

`workflow/scripts/get_human_trinuc.py` is a standalone helper for regenerating
the whole-genome trinucleotide composition table used by the `sbs96` correction,
should you need it for a genome build other than the ones already hardcoded in
`normcountlib.py`.

## Reference data

BiStro ships with COSMIC v3.6 SBS signature files (`GRCh38` and `mm10`) used as
the default input to `bistro cosmic`; override with `--sign_file` for another
COSMIC release or genome build.

## License

MIT - see [LICENSE](LICENSE).
