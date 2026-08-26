#!/usr/bin/env python
"""
Takes the SBS96 spectrum written by the ``BiStro sbs96`` subcommand and measures
how close it is to each reference COSMIC SBS signature, using cosine similarity.
"""

import sys
import itertools
from numpy import dot, asarray
from numpy.linalg import norm

from . import arglib


# ---------------------------------------------------------------------------
# Canonical SBS96 vocabulary (pyrimidine-centred), shared by both inputs.
# ---------------------------------------------------------------------------
def build_sbs96_list():
    # The 96 SBS types in a fixed order, so both spectra are vectorised the
    # same way regardless of the row order in either file.
    sub_lst = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
    sbs96_lst = []
    for sub in sub_lst:
        ref, alt = sub.split(">")
        for upstream, downstream in itertools.product("ACGT", repeat=2):
            sbs96_lst.append("{}[{}>{}]{}".format(upstream, ref, alt, downstream))
    return sbs96_lst


SBS96_LST = build_sbs96_list()
SBS96_SET = set(SBS96_LST)

# The 32 pyrimidine-centred trinucleotide contexts, and the context of each
# SBS96 type read straight off its label ("A[C>A]A" -> "ACA").
TRI_LST = [up + centre + down
           for up in "ACGT" for centre in "CT" for down in "ACGT"]
TRI_SET = set(TRI_LST)
SBS96_TO_TRI = {sbs96: sbs96[0] + sbs96[2] + sbs96[6] for sbs96 in SBS96_LST}

COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
PURINES = {"A", "G"}


def revcomp(tri):
    return "".join(COMPLEMENT[b] for b in reversed(tri))


def canonical_tri(tri):
    # Fold a trinucleotide onto its pyrimidine-centred strand; None on any
    # non-ACGT base (mirrors bistro normcounts.canonical_tri so that a human
    # G(c) computed here matches how the mouse G(c) column was built).
    if any(b not in COMPLEMENT for b in tri):
        return None
    if tri[1] in PURINES:
        return revcomp(tri)
    return tri


# ---------------------------------------------------------------------------
# Similarity metric (himut scripts/cos_sim.py)
# ---------------------------------------------------------------------------
def get_cos_sim(va, vb):
    # Cosine similarity between two non-negative spectra; 1.0 == identical shape.
    va = asarray(va, dtype=float)
    vb = asarray(vb, dtype=float)
    denom = norm(va) * norm(vb)
    if denom == 0:
        return 0.0
    return float(dot(va, vb) / denom)


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------
def load_normcounts(infile, column):
    # Read a BiStro normcounts.tsv and return {sbs96 -> value} for the chosen
    # column. Lines beginning with '##' are the reproducibility header; the
    # first non-comment line is the column header.
    sbs96_to_value = {}
    header = None
    for line in open(infile):
        if line.startswith("##"):
            continue
        arr = line.rstrip("\n").split("\t")
        if header is None:
            header = arr
            if column not in header:
                sys.exit(
                    "error: column '{}' not in {} (available: {})".format(
                        column, infile, ", ".join(header)
                    )
                )
            sbs96_idx = header.index("sbs96")
            value_idx = header.index(column)
            continue
        sbs96 = arr[sbs96_idx]
        if sbs96 in SBS96_SET:
            raw = arr[value_idx]
            if raw == "NA":
                sys.exit(
                    "error: column '{}' is NA in {} — re-run normcounts.py with "
                    "--human-trinuc to populate it".format(column, infile)
                )
            sbs96_to_value[sbs96] = float(raw)

    missing = SBS96_SET - set(sbs96_to_value)
    if missing:
        sys.exit(
            "error: {} incomplete — missing {} of 96 SBS96 types".format(
                infile, len(missing)
            )
        )
    return sbs96_to_value


def load_cosmic(infile, signatures=None):
    # Read a COSMIC/SigProfiler signature table and return an ordered list of
    # (signature_name, {sbs96 -> probability}). The first column is the mutation
    # type; every other column is one signature. If `signatures` is given, only
    # those columns are kept (and their presence is verified).
    sig_names = []
    sig_to_spectrum = {}
    type_idx = 0
    with open(infile) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col_names = header[1:]
        for name in col_names:
            sig_to_spectrum[name] = {}
        for line in fh:
            arr = line.rstrip("\n").split("\t")
            if len(arr) < 2:
                continue
            sbs96 = arr[type_idx].strip()
            if sbs96 not in SBS96_SET:
                continue
            for name, value in zip(col_names, arr[1:]):
                sig_to_spectrum[name][sbs96] = float(value)

    if signatures:
        unknown = [s for s in signatures if s not in sig_to_spectrum]
        if unknown:
            sys.exit(
                "error: signature(s) not found in {}: {}\navailable: {}".format(
                    infile, ", ".join(unknown), ", ".join(col_names)
                )
            )
        sig_names = list(signatures)
    else:
        sig_names = list(col_names)

    # Verify every requested signature is a complete 96-type spectrum.
    for name in sig_names:
        missing = SBS96_SET - set(sig_to_spectrum[name])
        if missing:
            sys.exit(
                "error: signature {} in {} is missing {} SBS96 types".format(
                    name, infile, len(missing)
                )
            )
    return [(name, sig_to_spectrum[name]) for name in sig_names]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def compute_similarities(sample_spectrum, cosmic_signatures):
    # Vectorise the sample once in canonical order, then score every signature.
    sample_vec = [sample_spectrum[sbs96] for sbs96 in SBS96_LST]
    results = []
    for name, spectrum in cosmic_signatures:
        sig_vec = [spectrum[sbs96] for sbs96 in SBS96_LST]
        results.append((name, get_cos_sim(sample_vec, sig_vec)))
    # Rank by descending similarity (best match first).
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def dump_similarities(sample, results, out, top):
    if top is not None:
        results = results[:top]
    lines = ["signature\tcosine_similarity"]
    for name, sim in results:
        lines.append("{}\t{:.6f}".format(name, sim))
    text = "\n".join(lines) + "\n"
    if out:
        with open(out, "w") as fh:
            fh.write(arglib.get_cmdline("##cosmiclib_command=") + "\n")
            fh.write("##sample=" + sample + "\n")
            fh.write(text)
        print("wrote {}".format(out))
    else:
        sys.stdout.write(text)


def main(in_file, cosmic_file, signatures, column, out_file, top):
    sample_spectrum = load_normcounts(in_file, column)
    cosmic_signatures = load_cosmic(cosmic_file, signatures)
    results = compute_similarities(sample_spectrum, cosmic_signatures)
    dump_similarities(in_file, results, out_file, top)
