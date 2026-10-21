#!/bin/bash
# Smoke test: run preprocess + somatic on tests/data and check that the known
# DNM (private to E01) and GRM (shared E01/F01) calls come back out.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/data"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "[smoke test] preprocessing E01..."
bistro preprocess --bam "$DATA/E01.bam" --ref "$DATA/reference.fa.gz" \
    --out_dir "$OUT" --sample E01 -t 1 \
    --min_mapq 0 --min_bq 0 --min_sequence_identity 0 --min_qlen 0 --min_ec 0 --min_depth 1

echo "[smoke test] preprocessing F01..."
bistro preprocess --bam "$DATA/F01.bam" --ref "$DATA/reference.fa.gz" \
    --out_dir "$OUT" --sample F01 -t 1 \
    --min_mapq 0 --min_bq 0 --min_sequence_identity 0 --min_qlen 0 --min_ec 0 --min_depth 1

echo "[smoke test] calling somatic mutations across E01+F01..."
bistro somatic -i "$OUT/E01.muts.bed.gz" "$OUT/F01.muts.bed.gz" \
    --bams "$DATA/E01.bam" "$DATA/F01.bam" --ref "$DATA/reference.fa.gz" \
    --thr 1 --max_muts_per_duplex 5 -t 1 --min_mapq 0 --min_baseq 0 --min_alt_support 1

n_dnm=$(zcat "$OUT/E01.shared.muts.bed.gz" | awk -F'\t' '$14=="DNM"' | wc -l)
n_grm_e01=$(zcat "$OUT/E01.shared.muts.bed.gz" | awk -F'\t' '$14=="GRM"' | wc -l)
n_grm_f01=$(zcat "$OUT/F01.shared.muts.bed.gz" | awk -F'\t' '$14=="GRM"' | wc -l)

echo "E01 DNM calls: $n_dnm (expected 2)"
echo "E01 GRM calls: $n_grm_e01 (expected > 0)"
echo "F01 GRM calls: $n_grm_f01 (expected > 0)"

if [[ "$n_dnm" -eq 2 && "$n_grm_e01" -gt 0 && "$n_grm_f01" -gt 0 ]]; then
    echo "[smoke test] PASSED"
else
    echo "[smoke test] FAILED"
    exit 1
fi
