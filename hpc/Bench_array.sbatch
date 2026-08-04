#!/bin/bash -l
#SBATCH --job-name=bench-array
#SBATCH --partition=cpu-standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-22
#SBATCH --output=/home/kg1111r/multilevel-partition-refinement/logs/%x-%A_%a.out
#SBATCH --error=/home/kg1111r/multilevel-partition-refinement/logs/%x-%A_%a.err
#SBATCH --chdir=/home/kg1111r/multilevel-partition-refinement
# =============================================================================
# PARALLEL benchmark: ONE graph per array task (shard I/N over the filtered list).
# Why: the serial Bench.sbatch hit the wall-clock limit after ~3 graphs
# because seeds x global-guard(2x) x evolutionary scaffold is heavy. Here each
# graph is its own task, so seeds=3 + the guard finish comfortably in parallel.
#
#   sbatch hpc/Bench_array.sbatch                       # all 23 Walshaw graphs
#   MAX_N=15000 SEEDS=3 sbatch hpc/Bench_array.sbatch   # filtered + 3 seeds/graph
#   sbatch --array=0-11 hpc/Bench_array.sbatch          # if only 12 pass the filter
#
# The array size is the shard COUNT (SLURM_ARRAY_TASK_COUNT); files[I::N] maps one
# graph to each task. Extra tasks (beyond the filtered count) no-op harmlessly.
# After the array finishes, MERGE the per-shard files into one JSONL (see end).
# =============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kg1111r/multilevel-partition-refinement}"
MODEL="${MODEL:-${PROJECT_DIR}/models/spectral_refiner_k${K_VALUE}_eps_0_03.pt}"
GRAPH_DIR="${GRAPH_DIR:-${PROJECT_DIR}/graphs}"   # override per job
K_VALUE="${K_VALUE:-4}"; EPSILON="${EPSILON:-0.03}"; SEEDS="${SEEDS:-3}"
MAX_N="${MAX_N:-20000}"
STOCHASTIC="${STOCHASTIC:-0}"
ML_REFINE_MAX_N="${ML_REFINE_MAX_N:-}"     # empty => use the checkpoint's stored value
RUN_TAG="${RUN_TAG:-walshaw}"; EXTRA="${EXTRA:-}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/results"
cd "${PROJECT_DIR}"

module purge
module load slurm/25.11.3 || true
module load Python/3.11.10-GCCcore-12.3.0
module load libffi/3.4.4-GCCcore-12.3.0
source "${PROJECT_DIR}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" PYTHONUNBUFFERED=1

python -c "import pymetis" 2>/dev/null || pip install pymetis 2>/dev/null \
  || echo "NOTE: pymetis unavailable; the METIS column is skipped."

ML_ARG=""; [ -n "${ML_REFINE_MAX_N}" ] && ML_ARG="--ml-refine-max-n ${ML_REFINE_MAX_N}"
SHARD="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
OUT="results/bench_${RUN_TAG}_k${K_VALUE}.shard${SLURM_ARRAY_TASK_ID}.jsonl"

echo "array bench | task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} | shard ${SHARD} | seeds=${SEEDS} max_n=${MAX_N} | $(date)"
python -u src/benchmark.py \
    --model "${MODEL}" --graph-dir "${GRAPH_DIR}" \
    --k "${K_VALUE}" --epsilon "${EPSILON}" --seeds "${SEEDS}" --max-n "${MAX_N}" \
    --stochastic "${STOCHASTIC}" ${ML_ARG} --metis ${EXTRA} \
    --shard "${SHARD}" --resume \
    --output "${OUT}"
echo "Done task ${SLURM_ARRAY_TASK_ID} $(date). Shard results: ${OUT}"

# -----------------------------------------------------------------------------
# AFTER the whole array finishes, merge shards -> one JSONL, then compare:
#   cat results/bench_${RUN_TAG}_k${K_VALUE}.shard*.jsonl > results/bench_${RUN_TAG}_k${K_VALUE}.jsonl
# -----------------------------------------------------------------------------
