"""
covlib — raw-read coverage scan (mosdepth) and the excessive-coverage mask.

For each contig the maximum-depth threshold is ``round(m + SIGMA*sqrt(m))``
where ``m`` is the contig's *raw-read* mean depth (mosdepth, MAPQ >= COV_MAPQ).
Reference intervals whose raw depth reaches that threshold are treated as
collapsed-repeat / mis-mapping artefacts and get scrubbed out of
``preprocess``'s ``.muts.bed.gz`` / ``.context.bed.gz``.

Everything here runs as a post-pass over the already-assembled BED files; the
mutation caller (``caller.call_somatic_mutations``) is not touched.
"""

import gzip
import os
import shutil
import subprocess

from . import utilib

COV_MAPQ = 20   # mapping-quality floor for the coverage scan
SIGMA = 4       # threshold = mean + SIGMA * sqrt(mean)


def _require(tool):
    path = shutil.which(tool)
    if path is None:
        raise RuntimeError(f"'{tool}' not found on PATH — required for the coverage scan")
    return path


class _MosdepthRun:
    def __init__(self, proc, prefix):
        self.proc = proc
        self.prefix = prefix


def start_mosdepth(bam, out_dir, sample, contigs, threads=1):
    """Launch mosdepth in the background (fast-mode, MAPQ>=COV_MAPQ).

    Returns a handle for finish_mosdepth(). The scan only needs the BAM, so it
    is meant to run concurrently with the fragment-writing / assembly steps.
    When ``contigs`` is a single contig the scan is restricted to it
    (``mosdepth -c`` takes exactly one); otherwise mosdepth scans the whole BAM
    and the callers below filter every derived quantity down to ``contigs``.
    """
    mosdepth = _require("mosdepth")
    prefix = os.path.join(out_dir, f"{sample}.covscan")
    cmd = [mosdepth, "-x", "-Q", str(COV_MAPQ), "-t", str(max(1, min(4, int(threads))))]
    if len(contigs) == 1:
        cmd += ["-c", contigs[0]]
    cmd += [prefix, bam]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)
    return _MosdepthRun(proc, prefix)


def finish_mosdepth(run):
    """Block until mosdepth finishes; return ``(summary_txt, per_base_bed_gz)``."""
    _, err = run.proc.communicate()
    if run.proc.returncode != 0:
        raise RuntimeError(f"mosdepth failed (exit {run.proc.returncode}): {(err or '').strip()}")
    return f"{run.prefix}.mosdepth.summary.txt", f"{run.prefix}.per-base.bed.gz"


def parse_contig_means(summary_txt, contigs):
    """``{contig: raw mean depth}`` for contigs in ``contigs`` (mosdepth summary)."""
    want = set(contigs)
    means = {}
    with open(summary_txt) as fh:
        next(fh, None)  # header: chrom length bases mean min max
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 4 and f[0] in want:
                means[f[0]] = float(f[3])
    return means


def high_coverage_bed(per_base_bed_gz, caps, out_dir, sample, contigs):
    """BED of intervals whose raw depth reaches the per-contig cap.

    mosdepth's per-base output is contiguous and position-sorted per contig, so
    qualifying intervals are merged on the fly. Returns the BED path, or ``None``
    when nothing qualifies.
    """
    want = set(contigs)
    merged = []          # list of [chrom, start, end]
    cur = None
    with gzip.open(per_base_bed_gz, "rt") as src:
        for line in src:
            chrom, start, end, depth = line.rstrip("\n").split("\t")
            cap = caps.get(chrom)
            if cap is None or chrom not in want or float(depth) < cap:
                continue
            s, e = int(start), int(end)
            if cur is not None and cur[0] == chrom and s <= cur[2]:
                cur[2] = max(cur[2], e)
            else:
                if cur is not None:
                    merged.append(cur)
                cur = [chrom, s, e]
    if cur is not None:
        merged.append(cur)

    if not merged:
        return None
    out = os.path.join(out_dir, f"{sample}.highcov.bed")
    with open(out, "w") as dst:
        for chrom, s, e in merged:
            dst.write(f"{chrom}\t{s}\t{e}\n")
    return out


def build_mask(out_dir, sample, contigs, *bed_paths):
    """Union the given BED files (any may be gzipped, missing, empty or None),
    restrict to ``contigs``, sort and merge -> ``{sample}.mask.bed``.

    Returns the path, or ``None`` when the union is empty. Folds the
    low-complexity-region BED and the high-coverage BED into a single mask so
    each output file is subtracted in one pass rather than two.
    """
    beds = [b for b in bed_paths if b and os.path.isfile(b) and os.path.getsize(b) > 0]
    if not beds:
        return None
    bedtools = _require("bedtools")
    keep = set(contigs)
    out = os.path.join(out_dir, f"{sample}.mask.bed")
    cat = subprocess.Popen(["zcat", "-f", *beds], stdout=subprocess.PIPE, text=True)
    srt = subprocess.Popen(["sort", "-k1,1", "-k2,2n"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for line in cat.stdout:
        if line.startswith(("#", "track", "browser")):
            continue
        f = line.split("\t")
        if len(f) >= 3 and f[0] in keep:
            srt.stdin.write(f"{f[0]}\t{f[1]}\t{f[2].rstrip()}\n")
    cat.stdout.close()
    cat.wait()
    srt.stdin.close()
    with open(out, "w") as fh:
        subprocess.run([bedtools, "merge", "-i", "-"], stdin=srt.stdout,
                       stdout=fh, check=True)
    srt.stdout.close()
    srt.wait()
    if os.path.getsize(out) == 0:
        os.remove(out)
        return None
    return out


def cleanup(summary_txt, per_base_bed_gz):
    """Delete the mosdepth intermediates (the .highcov.bed is kept as an artefact)."""
    prefix = summary_txt[: -len(".mosdepth.summary.txt")]
    for f in (summary_txt, per_base_bed_gz, per_base_bed_gz + ".csi",
              f"{prefix}.mosdepth.global.dist.txt"):
        try:
            os.remove(f)
        except OSError:
            pass
