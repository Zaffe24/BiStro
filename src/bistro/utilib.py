"""
Shared helpers used across all BiStro subcommands: colored console logging,
memory-usage reporting, and the up-front input validation (BAM/FASTA/VCF/BED
existence, indexing, and readability) that runs before each subcommand.
"""

import sys
import os
import gzip
from datetime import datetime
import psutil
import pysam
pysam.set_verbosity(0)

def date():
    return f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]"


# Utility function for colored console output.
def cprint(text="", color="cyan", sep="", file=sys.stderr):
    colors = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "purple": "\033[35m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }

    reset = "\033[0m"

    # Only use colors if output is a real terminal
    if hasattr(file, "isatty") and file.isatty():
        color_code = colors.get(color, "")
        print(f"{color_code}{text}{reset}", sep=sep, file=file, flush=True)
    else:
        print(text, sep=sep, file=file, flush=True)


def get_GB(size):
    return round(int(size) / 1024**3, 3)

def get_stats():
    process = psutil.Process(os.getpid())
    return get_GB(process.memory_info().rss)

def exit(code=0):
    # A non-zero code makes Snakemake register the job as failed; use it on
    # every validation-error path so the pipeline stops instead of marching on.
    if code == 0:
        cprint("exiting BiStro...", color="green")
    else:
        cprint("exiting BiStro (error)...", color="red")
    sys.exit(code)

def check_num_threads(thread_count: int):
    system_thread_count = psutil.cpu_count()
    if thread_count > system_thread_count:
        cprint(f"System does not have {thread_count} number of threads", color="red")
        exit(1)
    else:
        cprint(f"Using {thread_count} threads", color="green")


def check_inputs_preprocess(bam, ref, out_dir, germline_vcf, low_complexity_regions):
    """
    Validate that the BAM, REF, OUT and (optional) GERM inputs exist and are
    not corrupted before launching the pipeline. Exits BiStro on any failure.

    When a low-complexity BED is supplied it is subtracted from the outputs via
    `bedtools subtract`; a missing/unreadable file makes that step fail silently
    and wipe the outputs, so reject it up front. An empty BED is a harmless
    no-op (bedtools returns -a unchanged), so warn but proceed.
    """

    # --- BAM ---------------------------------------------------------------
    if not os.path.isfile(bam):
        cprint(f"ERROR: BAM file not found: {bam}", color="red")
        exit(1)
    try:
        with pysam.AlignmentFile(bam, "rb") as bam_file:
            if not bam_file.has_index():
                cprint(f"ERROR: BAM index (.bai/.csi) not found for: {bam}", color="red")
                exit(1)
            # touch the header / first records to confirm it is readable
            _ = bam_file.header
    except (ValueError, OSError) as e:
        cprint(f"ERROR: BAM file is corrupted or unreadable: {bam} ({e})", color="red")
        exit(1)

    # --- REF ---------------------------------------------------------------
    if not os.path.isfile(ref):
        cprint(f"ERROR: Reference FASTA not found: {ref}", color="red")
        exit(1)
    try:
        with pysam.FastaFile(ref) as fasta:
            if len(fasta.references) == 0:
                cprint(f"ERROR: Reference FASTA has no sequences: {ref}", color="red")
                exit(1)
    except (ValueError, OSError) as e:
        cprint(f"ERROR: Reference FASTA is corrupted or missing its .fai index: {ref} ({e})", color="red")
        exit(1)

    # --- OUT ---------------------------------------------------------------
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        cprint(f"Creating Output directory: {out_dir}", color="yellow")

    if not os.access(out_dir, os.W_OK):
        cprint(f"ERROR: Output directory is not writable: {out_dir}", color="red")
        exit(1)

    # --- GERM (optional) ---------------------------------------------------
    if germline_vcf:
        if not os.path.isfile(germline_vcf):
            cprint(f"ERROR: Germline VCF not found: {germline_vcf}", color="red")
            exit(1)
        try:
            with pysam.VariantFile(germline_vcf) as vcf:
                if vcf.index is None:
                    cprint(f"ERROR: Germline VCF index (.tbi/.csi) not found for: {germline_vcf}", color="red")
                    exit(1)
                _ = vcf.header
        except (ValueError, OSError) as e:
            cprint(f"ERROR: Germline VCF is corrupted or unreadable: {germline_vcf} ({e})", color="red")
            exit(1)

    # --- LOW-COMPLEXITY / SUBTRACT BED (optional) --------------------------
    if low_complexity_regions:
        if not os.path.isfile(low_complexity_regions):
            cprint(f"ERROR: Low-complexity BED not found: {low_complexity_regions}", color="red")
            exit(1)
        try:
            opener = gzip.open if low_complexity_regions.endswith(".gz") else open
            with opener(low_complexity_regions, "rt") as f:
                has_records = any(line.strip() and not line.startswith("#") for line in f)
        except OSError as e:
            cprint(f"ERROR: Low-complexity BED is corrupted or unreadable: {low_complexity_regions} ({e})", color="red")
            exit(1)
        if not has_records:
            cprint(f"WARNING: Low-complexity BED has no intervals; nothing will be subtracted: {low_complexity_regions}", color="yellow")

    cprint("All input files validated successfully.", color="green")


def check_inputs_somatic(bed_files: list, bam_files: list = None):
    """
    Validate that every BED file in the list exists before launching the
    somatic pipeline. Exits BiStro on the first missing file.

    When --bams is supplied it must line up one-to-one with the BEDs, since the
    de novo safeguard pairs them positionally. Each BAM is pileup'd at candidate
    positions, so an unindexed or corrupted BAM must be rejected up front rather
    than crashing a worker deep into the run.
    """
    for bed in bed_files:
        if not os.path.isfile(bed):
            cprint(f"ERROR: BED file not found: {bed}", color="red")
            exit(1)
    cprint(f"All {len(bed_files)} BED file(s) validated successfully.", color="green")

    if not bam_files:
        cprint("No --bams supplied: skipping the de novo pileup safeguard.", color="yellow")
        return

    if len(bam_files) != len(bed_files):
        cprint(f"ERROR: --bams has {len(bam_files)} file(s) but -i has {len(bed_files)}; they must match in number and order.", color="red")
        exit(1)

    for bam in bam_files:
        if not os.path.isfile(bam):
            cprint(f"ERROR: BAM file not found: {bam}", color="red")
            exit(1)
        try:
            with pysam.AlignmentFile(bam, "rb") as bam_file:
                if not bam_file.has_index():
                    cprint(f"ERROR: BAM index (.bai/.csi) not found for: {bam}", color="red")
                    exit(1)
                _ = bam_file.header
        except (ValueError, OSError) as e:
            cprint(f"ERROR: BAM file is corrupted or unreadable: {bam} ({e})", color="red")
            exit(1)

    for bed, bam in zip(bed_files, bam_files):
        cprint(f"  pairing {os.path.basename(bed)} <-> {os.path.basename(bam)}", color="cyan")
    cprint(f"All {len(bam_files)} BAM file(s) validated successfully.", color="green")


def check_inputs_sbs96(context_bed, muts_bed, ref, preset_genome=None):
    """
    Validate the shared.context / shared.muts BEDs and the reference FASTA
    before launching the sbs96 pipeline. Exits BiStro on any failure.

    --preset_genome overrides --ref: the composition comes from the hardcoded
    table, so the FASTA is never opened and there is nothing to validate.
    """
    for bed in (context_bed, muts_bed):
        if not os.path.isfile(bed):
            cprint(f"ERROR: BED file not found: {bed}", color="red")
            exit(1)

    if preset_genome is not None:
        cprint("All input files validated successfully.", color="green")
        return

    if ref is None:
        cprint(
            "ERROR: --ref FASTA is required unless --preset_genome is given",
            color="red",
        )
        exit(1)

    if not os.path.isfile(ref):
        cprint(f"ERROR: Reference FASTA not found: {ref}", color="red")
        exit(1)
    try:
        with pysam.FastaFile(ref) as fasta:
            if len(fasta.references) == 0:
                cprint(f"ERROR: Reference FASTA has no sequences: {ref}", color="red")
                exit(1)
    except (ValueError, OSError) as e:
        cprint(f"ERROR: Reference FASTA is corrupted or missing its .fai index: {ref} ({e})", color="red")
        exit(1)

    cprint("All input files validated successfully.", color="green")


def check_inputs_cosmic(normcounts_tsv, cosmic_file):
    """
    Validate the normcounts TSV and the COSMIC signature table before launching
    the cosmic pipeline. Exits BiStro on the first missing file.
    """
    if not os.path.isfile(normcounts_tsv):
        cprint(f"ERROR: normcounts TSV not found: {normcounts_tsv}", color="red")
        exit(1)
    if not os.path.isfile(cosmic_file):
        cprint(f"ERROR: COSMIC signature file not found: {cosmic_file}", color="red")
        exit(1)
    cprint("All input files validated successfully.", color="green")
