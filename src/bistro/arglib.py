"""
Command-line parsing for BiStro. Defines the `preprocess`, `somatic`, `sbs96`,
and `cosmic` subcommands and every option they accept, via argparse.
"""

import sys
import warnings
import argparse
from importlib.resources import files

DEFAULT_SIGN_FILE = str(files("bistro").joinpath("data", "COSMIC_v3.6_SBS_GRCh38.txt"))


def make_wide(formatter, w=120, h=36):
    """Return a wider HelpFormatter, if possible."""
    try:
        kwargs = {"width": w, "max_help_position": h}
        formatter(None, **kwargs)
        return lambda prog: formatter(prog, **kwargs)
    except TypeError:
        warnings.warn("argparse help formatter failed, falling back.")
        return formatter


def get_cmdline(prefix):
    """Return the exact invocation as a '##'-prefixed header line for output files."""
    return prefix + " ".join(sys.argv)


def parse_args(program_version, arguments=sys.argv[1:]):
    # main_arguments
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=make_wide(argparse.ArgumentDefaultsHelpFormatter),
        description="BiStro calls somatic mutations on single-strand CCS reads from PacBio Revio technology.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=program_version),
    )
    # subcommands: init
    subparsers = parser.add_subparsers(dest="sub", metavar="")

    # subcommands: call
    parser_preprocess = subparsers.add_parser(
        "preprocess",
        formatter_class=make_wide(argparse.ArgumentDefaultsHelpFormatter, w=180, h=60),
        help="Pre-process single-strand CCS reads into candidate mutation calls.",
    )
    parser_preprocess.add_argument(
        "-i",
        "--bam",
        type=str,
        required=True,
        help="BAM file with single-strand CCS reads aligned via 'pbmm2 align --preset CCS'.",
    )
    parser_preprocess.add_argument(
        "-r",
        "--ref",
        type=str,
        required=True,
        help="Reference genome FASTA file",
    )
    parser_preprocess.add_argument(
        "-g",
        "--germline_vcf",
        required=False,
        nargs='?',
        default=None,
        const=None,
        help="VCF file with germline mutations. Somatic mutations will not be called at these positions.",
    )
    parser_preprocess.add_argument(
        "--min_gq",
        type=float,
        required=False,
        default=20,
        help="Minimum threshold for Genotype Quality to be used to flag postions with Germline variants."
        )
    parser_preprocess.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=True,
        help="Output directory to write the results.",
    )
    parser_preprocess.add_argument(
        "-s","--sample",
        type=str,
        required=False,
        default="TrySample",
        help="Sample name to be used in the output files.",
    )
    parser_preprocess.add_argument(
        "--region",
        type=str,
        required=False,
        default="",
        nargs='+',
        help="Target Chromosome(s) to call mutations on. Example: 'chr1 chr2 chr3'",
    )
    parser_preprocess.add_argument(
        "--exclude",
        type=str,
        required=False,
        default="",
        nargs='+',
        help="Chromosome(s) to exclude from mutation calling. Example: 'chr1 chr2 chr3'",
    )
    parser_preprocess.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=1,
        help="Number of threads to use for parallel processing.",
    )
    parser_preprocess.add_argument(
        "--min_mapq",
        type=int,
        required=False,
        default=60,
        help="Minimum mapping quality score for reads to be considered in mutation calling.",
    )
    parser_preprocess.add_argument(
        "--min_sequence_identity",
        type=float,
        default=.99,
        required=False,
        help="Minimum sequence identity threshold for reads to be considered in mutation calling.",
    )
    parser_preprocess.add_argument(
        "--min_bq",
        type=int,
        default=93,
        required=False,
        help="Minimum base quality score threshold for both postion of a base-pair to be considered in mutation calling.",
    )
    parser_preprocess.add_argument(
        "--min_qlen",
        type=int,
        default=800,
        required=False,
        help="Minimum read length for reads to be considered in mutation calling.",
    )
    parser_preprocess.add_argument(
        "--max_qlen",
        type=int,
        default=10_000,
        required=False,
        help="Maximum read length for reads to be considered in mutation calling.",
    )
    parser_preprocess.add_argument(
        "--min_depth",
        type=int,
        default=3,
        required=False,
        help="Minimum read coverage to be considered in mutation calling.",
    )
    parser_preprocess.add_argument(
        "--trim_ends",
        type=float,
        default=.025,
        required=False,
        help="Number of bases to trim from the start and end of each read. If < 1, interpreted as a proportion of the read length; if >= 1, interpreted as an absolute number of bases.",
    )
    parser_preprocess.add_argument(
        "--indels_window",
        type=int,
        default=10,
        required=False,
        help="Window size for indels filtering. Indels within this window around a candidate single-base substitution will be used to filter the candidate.",
    )
    parser_preprocess.add_argument(
        "--mismatch_window_len",
        type=int,
        default=20,
        required=False,
        help="Window size for mismatches filtering. Mismatches within this window around a candidate variant will be recorded.",
    )
    parser_preprocess.add_argument(
        "--min_ec",
        type=int,
        default=5,
        required=False,
        help="Minimum number of effective coverage (EC) for a candidate variant to be considered.",
    )
    parser_preprocess.add_argument(
        "--min_rq",
        type=float,
        default=0.99,
        required=False,
        help="Minimum read quality (RQ) for reads to be considered in mutation calling.",
    )
    parser_preprocess.add_argument(
        "--max_softclipping",
        type=float,
        default=.2,
        required=False,
        help="Maximum proportion of soft-clipped bases allowed in a read for it to be considered in mutation calling.",
    )
    parser_preprocess.add_argument(
        "--z_prob",
        type=int,
        default=0,
        required=False,
        help="Debug_feature: set to 1000_000 to randomly record a reference base-pair with a probability of 1/1000_000 for error model estimation. Set to 0 to disable this feature.",
    )
    parser_preprocess.add_argument(
        "--check_mem_usage",
        type=int,
        default=10_000,
        required=False,
        help="How often (in number of processed ZMW duplexes) to check memory usage. If set to 0, memory usage will not be checked during the run.",
    )
    parser_preprocess.add_argument(
        "--do_not_collapse",
        action="store_true",
        default=True,
        help="If set, mutations will be reported at the read level without collapsing to ZMW duplexes. This is useful for debugging and error model estimation.",
    )
    parser_preprocess.add_argument(
        "--low_complexity_regions",
        type=str,
        required=False,
        default=None,
        help="BED file of intervals to exclude from the mutation and context outputs (e.g. repeat/low-complexity regions).",
    )
    # parser_preprocess.add_argument(
    #     "--min_chr_len",
    #     type=int,
    #     required=False,
    #     default=200_000,
    #     help="Minimum Length for a contig to be included in the computation of the maximum depth threshold [MEAN(contigs) + 2SD]",
    # )
    # subcommand: share
    parser_somatic = subparsers.add_parser(
        "somatic",
        formatter_class=make_wide(argparse.ArgumentDefaultsHelpFormatter, w=180, h=60),
        help="Call true somatic mutations and rates from pre-processed candidate calls across samples.",
    )
    parser_somatic.add_argument(
        "-i",
        type=str,
        required=True,
        nargs="+",
        metavar="BED",
        help="Series of .muts.bed.gz files produced by the 'preprocess' subcommand.",
    )
    parser_somatic.add_argument(
        "--thr",
        type=int,
        required=False,
        default=1,
        help="Minimum number of other samples that must show a mutation before it stops being treated as private: either by carrying a call at that position (annotated germline), or -- when --bams is given -- by carrying read support for its ALT allele (annotated FPD).",
    )
    parser_somatic.add_argument(
        "--max_muts_per_duplex",
        type=int,
        required=False,
        default=5,
        help="Maximum number of DNMs allowed per ZMW duplex. ZMWs exceeding this threshold are flagged as FPD (false positive de novo).",
    )
    parser_somatic.add_argument(
        "--ref",
        type=str,
        required=True,
        metavar="FASTA",
        help="Reference genome FASTA file (must be indexed) used to determine the chromosome list.",
    )
    parser_somatic.add_argument(
        "--bams",
        type=str,
        required=False,
        nargs="+",
        default=None,
        metavar="BAM",
        help="Indexed BAM file per sample, in the SAME ORDER as -i. Enables the final safeguard: a de novo mutation whose ALT allele is supported by reads in other samples' BAMs is re-flagged as FPD. Omit to skip the check.",
    )
    parser_somatic.add_argument(
        "--min_alt_support",
        type=int,
        required=False,
        default=1,
        help="Minimum number of high-quality reads carrying the ALT allele for another sample's BAM to count as supporting that allele.",
    )
    parser_somatic.add_argument(
        "--min_mapq",
        type=int,
        required=False,
        default=30,
        help="Minimum mapping quality for a read to be counted in the --bams pileup safeguard.",
    )
    parser_somatic.add_argument(
        "--min_baseq",
        type=int,
        required=False,
        default=70,
        help="Minimum base quality for a base to be counted in the --bams pileup safeguard.",
    )
    parser_somatic.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=1,
        help="Number of parallel worker processes.",
    )

    # subcommand: sbs96
    parser_sbs96 = subparsers.add_parser(
        "sbs96",
        formatter_class=make_wide(argparse.ArgumentDefaultsHelpFormatter, w=180, h=60),
        help="Correct the somatic SBS96 spectrum for trinucleotide opportunity biases.",
    )
    parser_sbs96.add_argument(
        "--context",
        type=str,
        required=True,
        metavar="BED",
        help="SAMPLE.shared.context.bed.gz (already-cleaned callable bases).",
    )
    parser_sbs96.add_argument(
        "--muts",
        type=str,
        required=True,
        metavar="BED",
        help="SAMPLE.shared.muts.bed.gz (annotated somatic calls).",
    )
    # The genome trinucleotide composition G(c) comes from scanning this FASTA,
    # unless --preset_genome supplies it from a hardcoded table instead.
    parser_sbs96.add_argument(
        "-r",
        "--ref",
        type=str,
        default=None,
        metavar="FASTA",
        help="Reference genome FASTA file (must be indexed) for genome composition. Required unless --preset_genome is given.",
    )
    parser_sbs96.add_argument(
        "--preset_genome",
        type=str,
        default=None,
        choices=["GRCh38", "GRCm39"],
        help="Take the genome composition from the hardcoded table for this assembly, skipping the --ref scan (overrides --ref).",
    )
    parser_sbs96.add_argument(
        "-o",
        "--out",
        type=str,
        required=True,
        help="Output normcounts TSV.",
    )
    parser_sbs96.add_argument(
        "--plot",
        type=str,
        required=False,
        default=None,
        help="Output SBS96 spectrum image (PDF/PNG). Defaults to --out with its extension replaced by .pdf.",
    )
    parser_sbs96.add_argument(
        "-s",
        "--sample",
        type=str,
        required=False,
        default=None,
        help="Sample name for the plot title. Defaults to the --out basename.",
    )
    parser_sbs96.add_argument(
        "--no_plot",
        action="store_true",
        help="Write only the TSV; skip the spectrum image.",
    )
    parser_sbs96.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=1,
        help="Worker processes for the shared.context scan (one contig per worker).",
    )

    # subcommand: cosmic
    parser_cosmic = subparsers.add_parser(
        "cosmic",
        formatter_class=make_wide(argparse.ArgumentDefaultsHelpFormatter, w=180, h=60),
        help="Cosine similarity of a BiStro SBS96 spectrum against COSMIC signatures.",
    )
    parser_cosmic.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        metavar="TSV",
        help="SAMPLE.normcounts.tsv produced by the 'sbs96' subcommand.",
    )
    parser_cosmic.add_argument(
        "--sign_file",
        type=str,
        required=False,
        default=DEFAULT_SIGN_FILE,
        metavar="TSV",
        help="COSMIC SBS96 signature file (SigProfiler 'Type\\tSBS1\\tSBS2...' layout, e.g. COSMIC_v3.6_SBS_GRCh38.txt).",
    )
    parser_cosmic.add_argument(
        "--signatures",
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="Optional subset of signatures to compare against (e.g. --signatures SBS1 SBS5 SBS40). Default: every signature in the COSMIC file.",
    )
    parser_cosmic.add_argument(
        "--column",
        type=str,
        required=False,
        default="Human_normfrac",
        help="normcounts.tsv column to use as the sample spectrum (Human_normfrac is the spectrum corrected on human-genome opportunity, which is what COSMIC signatures are defined on).",
    )
    parser_cosmic.add_argument(
        "-o",
        "--out",
        type=str,
        required=False,
        default=None,
        help="Output TSV. Default: print to stdout.",
    )
    parser_cosmic.add_argument(
        "--top",
        type=int,
        required=False,
        default=None,
        help="Report only the N best-matching signatures.",
    )

    if len(arguments) == 0:
            parser.print_help()
            parser.exit()
    else:
        return parser, parser.parse_args(arguments)