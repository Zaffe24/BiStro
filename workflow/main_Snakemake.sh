#!/bin/bash
# Example SLURM submission script (written for CESGA -- adapt module/conda
# names to your own cluster). Replace --mail-user with your own address.
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=pietro.zaffe@gmail.com
#SBATCH --job-name=Snk_BiStro
#SBATCH --error=snk_BiStro_%j.err
#SBATCH --output=snk_BiStro_%j.err
#SBATCH --partition=long
#SBATCH --mem=1000
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00

# ---------------------------------------------------------------------------
# Usage:
#   sbatch main_Snakemake.sh [run_name] <config.yaml> [extra snakemake args...]
#
#   <run_name>     Unique label for this run. It gets its OWN working
#                  directory (runs/<run_name>/) with its OWN .snakemake/
#                  state, so concurrent runs can NEVER collide.
#                  OPTIONAL -- if omitted, defaults to ".snakemake".
#   <config.yaml>  The config file for THIS run (copy config/config.yaml and
#                  fill in your paths first). It is COPIED into the run
#                  directory, so later edits to your template config can no
#                  longer change a run that is already queued/running.
#
# Example (two runs at once, no conflict):
#   sbatch main_Snakemake.sh bq93 ../config/config_bq93.yaml
#   sbatch main_Snakemake.sh bq70 ../config/config_bq70.yaml
#
# Example (run_name omitted, defaults to ".snakemake"):
#   sbatch main_Snakemake.sh ../config/config.yaml
# ---------------------------------------------------------------------------

# Strict mode only for argument handling / setup (conda activate trips on it).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch main_Snakemake.sh [run_name] <config.yaml> [extra snakemake args...]" >&2
    exit 1
fi

# If $1 is itself an existing file, no run_name was given -- treat $1 as the
# config and fall back to the default run_name.
if [[ -f "$1" ]]; then
    RUN_NAME=".snakemake"
    CONFIG_SRC="$1"
    shift 1
else
    if [[ $# -lt 2 ]]; then
        echo "Usage: sbatch main_Snakemake.sh [run_name] <config.yaml> [extra snakemake args...]" >&2
        exit 1
    fi
    RUN_NAME="$1"
    CONFIG_SRC="$2"
    shift 2
fi

if [[ ! -f "$CONFIG_SRC" ]]; then
    echo "ERROR: config file not found: $CONFIG_SRC" >&2
    exit 1
fi

# Absolute path to the pipeline source (where this script + Snakefile live).
# Under sbatch the script runs from a private SLURM spool dir, so DON'T use
# ${BASH_SOURCE[0]} -- use the submission directory instead (falls back to the
# script's own dir when run directly, not via sbatch).
PIPELINE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SNAKEFILE="$PIPELINE_DIR/Snakefile"

if [[ ! -f "$SNAKEFILE" ]]; then
    echo "ERROR: Snakefile not found at $SNAKEFILE" >&2
    echo "       (submit sbatch from bistro/workflow/)" >&2
    exit 1
fi

# Per-run isolated working directory -> its own .snakemake/ state + past_runs/.
RUN_DIR="$PIPELINE_DIR/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

# Freeze this run's config inside its working dir (skip if it's already there).
if [[ "$(readlink -f "$CONFIG_SRC")" != "$(readlink -f "$RUN_DIR/config.yaml")" ]]; then
    cp "$CONFIG_SRC" "$RUN_DIR/config.yaml"
fi

# ---- environment setup (relax strict mode; conda/module scripts need it) ---
# CESGA-specific example -- replace with whatever activates your own
# snakemake + bistro environment (e.g. `conda activate bistro`).
set +eu

module load cesga/2025 miniconda3 2>/dev/null
module unload snakemake 2>/dev/null  # remove CESGA snakemake module, use conda version
conda activate snakemake_env

# Clear leaked PYTHONPATH/PYTHONHOME so conda's own packages (pandas/numpy) are used,
# not the Python 3.11 system modules inherited via sbatch's environment export.
unset PYTHONPATH PYTHONHOME

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$RUN_DIR" \
    --configfile "$RUN_DIR/config.yaml" \
    --printshellcmds --executor slurm --cores 36 --jobs 10 \
    --latency-wait 30 --keep-going "$@"
