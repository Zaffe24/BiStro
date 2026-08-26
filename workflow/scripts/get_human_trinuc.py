#!/usr/bin/env python
"""
get_human_trinuc.py — count reference trinucleotides once, write a small table.

Scans a FASTA (e.g. GRCh38) and emits the 32 pyrimidine-centred trinucleotide
counts as a two-column "trinucleotide<TAB>count" table. Run this ONCE against
the human genome that matches your COSMIC build; the resulting table is a fixed
constant reused for free by normcounts.py / cosmiclib.py (the freq_human term in
the sigfit-style human-relative conversion). Counting is vectorised, so a whole
genome is a few passes, not a per-base Python loop.

    python get_human_trinuc.py --fasta GRCh38.fa[.gz] --out GRCh38.trinuc.tsv

The folding convention (purine-centred contexts reverse-complemented onto the
pyrimidine strand) matches bistro's own G(c), so the table is directly
comparable to the genome_tri_count column in a normcounts.tsv.
"""

import sys
import gzip
import argparse

import numpy as np


COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
PURINES = {"A", "G"}
TRI_LST = [up + centre + down
           for up in "ACGT" for centre in "CT" for down in "ACGT"]


def revcomp(tri):
    return "".join(COMPLEMENT[b] for b in reversed(tri))


def canonical_tri(tri):
    # Fold onto the pyrimidine-centred strand; None on any non-ACGT base.
    if any(b not in COMPLEMENT for b in tri):
        return None
    if tri[1] in PURINES:
        return revcomp(tri)
    return tri


def _count_block(block, code):
    # Count the 64 forward 3-mers in a sequence block, dropping any triple that
    # touches a non-ACGT base. Returns a length-64 bincount (index = 16a+4b+c).
    arr = code[np.frombuffer(block.encode("ascii"), dtype=np.uint8)].astype(np.int64)
    if arr.size < 3:
        return np.zeros(64, dtype=np.int64)
    a, b, c = arr[:-2], arr[1:-1], arr[2:]
    valid = (a < 4) & (b < 4) & (c < 4)
    idx = (a * 16 + b * 4 + c)[valid]
    return np.bincount(idx, minlength=64)


def count_trinucleotides(fasta):
    # Stream the FASTA (plain or gzip — no bgzip/index needed) and count forward
    # 3-mers with numpy. Sequence is processed in ~8 MB blocks; the last 2 bases
    # of each block are carried into the next so triples spanning a block/line
    # boundary are counted exactly once — but the carry is reset at every contig
    # header so no triple is counted across two contigs.
    code = np.full(256, 4, dtype=np.uint8)      # non-ACGT -> 4 (dropped)
    for i, base in enumerate(b"ACGT"):
        code[base] = i

    counts64 = np.zeros(64, dtype=np.int64)
    opener = gzip.open if fasta.endswith(".gz") else open
    tail = ""                                    # last 2 bases of the current contig
    buf, buflen = [], 0
    with opener(fasta, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if buf:                          # flush the tail of the finished contig
                    counts64 += _count_block(tail + "".join(buf), code)
                buf, buflen, tail = [], 0, ""    # never carry across contigs
                print("  counting {}".format(line[1:].split()[0]), file=sys.stderr)
                continue
            seq = line.strip().upper()
            if not seq:
                continue
            buf.append(seq)
            buflen += len(seq)
            if buflen >= 8_000_000:
                block = tail + "".join(buf)
                counts64 += _count_block(block, code)
                tail = block[-2:]                # overlap into the next block
                buf, buflen = [], 0
        if buf:
            counts64 += _count_block(tail + "".join(buf), code)

    bases = "ACGT"
    tri_counts = {tri: 0 for tri in TRI_LST}
    for i in range(64):
        tri = bases[i // 16] + bases[(i // 4) % 4] + bases[i % 4]
        ctri = canonical_tri(tri)
        tri_counts[ctri] += int(counts64[i])
    return tri_counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True,
                    help="Reference FASTA (bgzipped + .fai/.gzi indexed, or plain).")
    ap.add_argument("--out", required=True,
                    help="Output trinucleotide table (trinucleotide<TAB>count).")
    args = ap.parse_args()

    tri_counts = count_trinucleotides(args.fasta)
    total = sum(tri_counts.values())
    with open(args.out, "w") as fh:
        fh.write("#trinucleotide\tcount\n")
        for tri in TRI_LST:
            fh.write("{}\t{}\n".format(tri, tri_counts[tri]))
    print("wrote {} ({} contexts, {} counted trinucleotides)".format(
        args.out, len(TRI_LST), total))


if __name__ == "__main__":
    main()
