#!/bin/bash -l
#SBATCH --job-name=bench
#SBATCH --partition=cpu-standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/kg1111r/multilevel-partition-refinement/logs/%x-%j.out
#SBATCH --error=/home/kg1111r/multilevel-partition-refinement/logs/%x-%j.err
#SBATCH --chdir=/home/kg1111r/multilevel-partition-refinement
# =============================================================================
# Benchmark a trained model (refinement at every level) on a test archive.
# The benchmark pipeline with the METIS baseline, so cuts are directly
# comparable. The mlref column shows how many levels the multi-level refiner acted on.
#   sbatch hpc/Bench.sbatch
#   MODEL=... GRAPH_DIR=... MAX_N=6000 ML_REFINE_MAX_N=4000 sbatch hpc/Bench.sbatch
# =============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kg1111r/multilevel-partition-refinement}"
MODEL="${MODEL:-${PROJECT_DIR}/models/spectral_refiner_k4_eps_0_03.pt}"
GRAPH_DIR="${GRAPH_DIR:-${PROJECT_DIR}/graphs}"   # override per job
K_VALUE="${K_VALUE:-4}"; EPSILON="${EPSILON:-0.03}"; SEEDS="${SEEDS:-1}"
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

python -c "import pymetis" 2>/dev/null || pip install pymetis 2>/dev/null || echo "NOTE: pymetis unavailable; the METIS column is skipped."

ML_ARG=""; [ -n "${ML_REFINE_MAX_N}" ] && ML_ARG="--ml-refine-max-n ${ML_REFINE_MAX_N}"

echo "benchmark | model ${MODEL} | graphs ${GRAPH_DIR} | k=${K_VALUE} eps=${EPSILON} seeds=${SEEDS} | $(date)"
python -u src/benchmark.py \
    --model "${MODEL}" --graph-dir "${GRAPH_DIR}" \
    --k "${K_VALUE}" --epsilon "${EPSILON}" --seeds "${SEEDS}" --max-n "${MAX_N}" \
    --stochastic "${STOCHASTIC}" ${ML_ARG} --metis ${EXTRA} \
    --resume \
    --output "results/bench_${RUN_TAG}_k${K_VALUE}.jsonl"
echo "Done $(date). Results: results/bench_${RUN_TAG}_k${K_VALUE}.jsonl"
