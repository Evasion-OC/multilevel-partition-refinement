#!/bin/bash -l
#SBATCH --job-name=bench-quant
#SBATCH --partition=cpu-standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/kg1111r/multilevel-partition-refinement/logs/%x-%A_%a.out
#SBATCH --error=/home/kg1111r/multilevel-partition-refinement/logs/%x-%A_%a.err
#SBATCH --chdir=/home/kg1111r/multilevel-partition-refinement
# =============================================================================
# Task-level quantization sensitivity + the rewrite inside the real pipeline.
# Array index: 0 fp32 | 1 int8 per-tensor | 2 int8 per-channel | 3 int4 per-tensor
#              4 int4 per-channel | 5 fp32 with --spectral-rewrite (same cuts, timing)
# Same protocol as hpc/Bench.sbatch so the rows are comparable. First run once:
#   python quantization/export_quantized.py --ckpt models/spectral_refiner_k16_eps_0_03.pt
#   CKPT_STEM=spectral_refiner_k16_eps_0_03 K_VALUE=16 sbatch hpc/Bench_quant.sbatch
# =============================================================================
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/kg1111r/multilevel-partition-refinement}"
CKPT_STEM="${CKPT_STEM:-spectral_refiner_k16_eps_0_03}"
GRAPH_DIR="${GRAPH_DIR:-${PROJECT_DIR}/graphs}"
K_VALUE="${K_VALUE:-16}"; EPSILON="${EPSILON:-0.03}"; SEEDS="${SEEDS:-1}"
MAX_N="${MAX_N:-20000}"; STOCHASTIC="${STOCHASTIC:-0}"; EXTRA="${EXTRA:-}"
case "${SLURM_ARRAY_TASK_ID}" in
  0) MODEL="models/${CKPT_STEM}.pt";          TAG="fp32";    FLAG="" ;;
  1) MODEL="models/quantized/${CKPT_STEM}_int8_pt.pt"; TAG="int8_pt"; FLAG="" ;;
  2) MODEL="models/quantized/${CKPT_STEM}_int8_pc.pt"; TAG="int8_pc"; FLAG="" ;;
  3) MODEL="models/quantized/${CKPT_STEM}_int4_pt.pt"; TAG="int4_pt"; FLAG="" ;;
  4) MODEL="models/quantized/${CKPT_STEM}_int4_pc.pt"; TAG="int4_pc"; FLAG="" ;;
  5) MODEL="models/${CKPT_STEM}.pt";          TAG="fp32_rewrite"; FLAG="--spectral-rewrite" ;;
esac
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
echo "bench-quant | ${TAG} | model ${MODEL} | graphs ${GRAPH_DIR} | k=${K_VALUE} | $(date)"
python -u src/benchmark.py \
    --model "${MODEL}" --graph-dir "${GRAPH_DIR}" \
    --k "${K_VALUE}" --epsilon "${EPSILON}" --seeds "${SEEDS}" --max-n "${MAX_N}" \
    --stochastic "${STOCHASTIC}" --metis ${FLAG} ${EXTRA} --resume \
    --output "results/bench_quant_${CKPT_STEM}_${TAG}.jsonl"
echo "Done $(date). Results: results/bench_quant_${CKPT_STEM}_${TAG}.jsonl"
