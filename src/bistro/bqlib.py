"""
Duplex base-quality (BQ) profile written by `BiStro preprocess`. On one contig
(the first analysed one, or --bq_profile_contig), every duplex base-pair where
BOTH strands match the reference is binned by min(fwd BQ, rev BQ), taken after
mask_ends / mask_indels and before the --min_bq gate. The result is reported as
the fraction of matched base-pairs whose two strands both reach BQ >= X
(cumulative curve modelled on duplex_bq_profile.py's plot_cumulative).
With --collapsed there is no second strand: each read base matching the
reference is binned by its own BQ (same files, labels say "read").
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless backend: render straight to a file
import matplotlib.pyplot as plt
from . import utilib

N_BQ = 256      # BAM base qualities are uint8
MIN_HI = 94     # always show the axis up to the PacBio BQ cap (93)

# Style: dataviz default palette, light surface (same as duplex_bq_profile.py)
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"

STYLE = {
    "figure.facecolor": COLOR_SURFACE,
    "axes.facecolor": COLOR_SURFACE,
    "savefig.facecolor": COLOR_SURFACE,
    "text.color": COLOR_TEXT_PRIMARY,
    "axes.edgecolor": COLOR_BASELINE,
    "axes.labelcolor": COLOR_TEXT_PRIMARY,
    "xtick.color": COLOR_MUTED,
    "ytick.color": COLOR_MUTED,
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "font.family": "sans-serif",
}


def new_histogram():
    return np.zeros(N_BQ, dtype=np.int64)


def add_duplex(hist, aln1, aln2, s1, s2, q1, q2, ref_arr):
    """Add one duplex's matched base-pairs to hist, indexed by min(fwd BQ, rev BQ).

    aln1/aln2: get_aligned_pairs(matches_only=True) of each read; s1/s2: upper-case
    query sequences; q1/q2: masked qualities (uint8 arrays); ref_arr: the contig's
    upper-case reference as a uint8 array. Same joint positions as caller.py's walk.
    """
    if not aln1 or not aln2:
        return
    a1 = np.asarray(aln1, dtype=np.int64)
    a2 = np.asarray(aln2, dtype=np.int64)
    # reference positions are unique and sorted within each read
    rp, i1, i2 = np.intersect1d(a1[:, 1], a2[:, 1], assume_unique=True, return_indices=True)
    if rp.size == 0:
        return
    qp1, qp2 = a1[i1, 0], a2[i2, 0]
    ref_base = ref_arr[rp]
    b1 = np.frombuffer(s1.encode("ascii"), dtype=np.uint8)[qp1]
    b2 = np.frombuffer(s2.encode("ascii"), dtype=np.uint8)[qp2]
    matched = (b1 == ref_base) & (b2 == ref_base)
    bq = np.minimum(q1[qp1], q2[qp2])[matched]
    hist += np.bincount(bq, minlength=N_BQ)


def add_read(hist, qpos, rpos, seq, qual, ref_arr):
    """Collapsed mode: add one read's reference-matching bases to hist, by BQ.
    qpos/rpos: bamlib.aligned_pairs_array of the read; qual: masked qualities."""
    base = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)[qpos]
    matched = base == ref_arr[rpos]
    hist += np.bincount(qual[qpos[matched]], minlength=N_BQ)


def write_bq_profile(hist, out_dir, sample, contig, min_bq, collapsed=False):
    """Write {sample}.duplex_bq_cumulative.tsv + .pdf; returns (tsv, pdf)."""
    tsv = os.path.join(out_dir, f"{sample}.duplex_bq_cumulative.tsv")
    pdf = os.path.join(out_dir, f"{sample}.duplex_bq_cumulative.pdf")

    n = int(hist.sum())
    nz = np.flatnonzero(hist)
    hi = max(int(nz.max()) + 1 if nz.size else 0, MIN_HI)
    # frac_ge[x] = fraction of matched base-pairs with BQ >= x on BOTH strands
    tail = hist[::-1].cumsum()[::-1]
    frac_ge = tail[:hi] / n if n > 0 else np.zeros(hi)

    with open(tsv, "w") as f:
        f.write(f"##sample={sample}\n")
        f.write(f"##contig={contig}\n")
        if collapsed:
            f.write(f"##matched_read_bases={n}\n")
            f.write("## BQ      = BQ of a read base (collapsed mode: one read per molecule) that matches the\n")
            f.write("##           reference; taken after --trim_ends/--indels_window masking (masked\n")
            f.write("##           bases have BQ 0) and before the --min_bq gate.\n")
            f.write("## COUNT   = matched bases with exactly this BQ.\n")
            f.write("## FRAC_GE = fraction of matched bases with BQ >= this value.\n")
        else:
            f.write(f"##matched_duplex_bp={n}\n")
            f.write("## BQ      = min(fwd-strand BQ, rev-strand BQ) of a duplex base-pair whose two strands both\n")
            f.write("##           match the reference; taken after --trim_ends/--indels_window masking (masked\n")
            f.write("##           bases have BQ 0) and before the --min_bq gate.\n")
            f.write("## COUNT   = matched base-pairs with exactly this BQ.\n")
            f.write("## FRAC_GE = fraction of matched base-pairs with BQ >= this value on both strands.\n")
        f.write("BQ\tCOUNT\tFRAC_GE\n")
        for bq in range(hi):
            f.write(f"{bq}\t{hist[bq]}\t{round(float(frac_ge[bq]), 6)}\n")

    plot_bq_cumulative(frac_ge, n, pdf, sample, contig, min_bq, collapsed)
    what = "read bases" if collapsed else "base-pairs"
    utilib.cprint(f"{'Read' if collapsed else 'Duplex'} BQ profile ({contig}, {n:,} matched {what}) saved to: "
                  f"{os.path.basename(tsv)} {os.path.basename(pdf)}")
    return tsv, pdf


def plot_bq_cumulative(frac_ge, n, out_pdf, sample, contig, min_bq, collapsed=False):
    """Single curve: fraction of matched duplex base-pairs with both strands BQ >= X
    (collapsed: fraction of matched read bases with BQ >= X)."""
    hi = len(frac_ge)
    x = np.arange(hi)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.5))

        # subtle reference lines: above grid, below the curve
        ref_style = dict(color=COLOR_BASELINE, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1.9)
        for bq in (20, 50, 70):
            ax.axvline(bq, **ref_style)
        for frac in (0.5, 0.8):
            ax.axhline(frac, **ref_style)

        if n > 0:
            ax.plot(x, frac_ge, color=COLOR_TEXT_PRIMARY, linewidth=2,
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)
            # the one value worth labelling: what this run's --min_bq keeps
            if 0 <= min_bq < hi:
                f = float(frac_ge[min_bq])
                ax.plot([min_bq], [f], marker="o", markersize=8, color=COLOR_TEXT_PRIMARY,
                        markeredgecolor=COLOR_SURFACE, markeredgewidth=2, zorder=4, clip_on=False)
                # The curve never rises, so below-left of the point is always free of it;
                # with no room on the left, go above-right (or below-right near the top).
                if min_bq >= 20:
                    offset, ha, va = (-8, -8), "right", "top"
                elif f <= 0.9:
                    offset, ha, va = (8, 8), "left", "bottom"
                else:
                    offset, ha, va = (8, -8), "left", "top"
                ax.annotate(f"--min_bq {min_bq}: {f:.1%}", xy=(min_bq, f), xytext=offset,
                            textcoords="offset points", ha=ha, va=va,
                            fontsize=10, color=COLOR_TEXT_SECONDARY)
        else:
            ax.text(0.5, 0.5, f"No matched {'read bases' if collapsed else 'duplex base-pairs'} on {contig}",
                    transform=ax.transAxes, ha="center", va="center", color=COLOR_TEXT_SECONDARY)

        ax.set_xlabel("BQ threshold (X)")
        if collapsed:
            ax.set_ylabel("Fraction with BQ ≥ X")
            ax.set_title(f"{sample} -- cumulative read BQ ({contig}, matched bases, n={n:,})",
                         color=COLOR_TEXT_PRIMARY, fontsize=12)
        else:
            ax.set_ylabel("Fraction with both strands BQ ≥ X")
            ax.set_title(f"{sample} -- cumulative duplex BQ ({contig}, matched base-pairs, n={n:,})",
                         color=COLOR_TEXT_PRIMARY, fontsize=12)
        ax.set_xlim(0, hi - 1)
        ax.set_xticks(list(range(0, 85, 10)) + list(range(85, hi, 5)))
        ax.tick_params(axis="x", labelsize=9, labelrotation=90)
        ax.set_ylim(0, 1.02)

        fig.tight_layout()
        fig.savefig(out_pdf)
        plt.close(fig)
