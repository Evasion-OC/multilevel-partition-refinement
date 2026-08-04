#!/bin/bash -l
#SBATCH --job-name=train-cpu
#SBATCH --partition=cpu-standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/kg1111r/multilevel-partition-refinement/logs/%x-%j.out
#SBATCH --error=/home/kg1111r/multilevel-partition-refinement/logs/%x-%j.err
#SBATCH --chdir=/home/kg1111r/multilevel-partition-refinement
# =============================================================================
# Multi-depth RL training on CPU (fp32). Same output as the GPU job.
#   sbatch hpc/Train_CPU.sbatch
# =============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kg1111r/multilevel-partition-refinement}"
MODEL_DIR="${PROJECT_DIR}/Models"
GRAPH_DIR="${GRAPH_DIR:-${PROJECT_DIR}/corpus/train}"
RUN_TAG="${RUN_TAG:-synth}"
COARSEN_SEEDS="${COARSEN_SEEDS:-3}"
N_EIGS="${N_EIGS:-8}"
ML_REFINE_MAX_N="${ML_REFINE_MAX_N:-4000}"
mkdir -p "${MODEL_DIR}" "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

module purge
module load slurm/25.11.3 || true
module load Python/3.11.10-GCCcore-12.3.0
module load libffi/3.4.4-GCCcore-12.3.0
source "${PROJECT_DIR}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

K_VALUE="${K_VALUE:-4}"
EPSILON="${EPSILON:-0.03}"
EPISODES="${EPISODES:-2000}"
eps_tag=$(echo "${EPSILON}" | sed 's/\./_/g')
SAVE_PATH="${MODEL_DIR}/refiner_${RUN_TAG}_k${K_VALUE}_eps_${eps_tag}.pt"

echo "CPU training | node $(hostname) | job ${SLURM_JOB_ID} | k=${K_VALUE} eps=${EPSILON} ml_refine_max_n=${ML_REFINE_MAX_N} | $(date)"
if [ -f "${SAVE_PATH}" ]; then
    if python -c "from refiner import load_spectral_actor_critic; load_spectral_actor_critic('${SAVE_PATH}')" 2>/dev/null; then
        echo "Checkpoint exists and reload-verifies; skipping. Remove it to retrain."
        exit 0
    fi
    echo "Existing checkpoint failed reload-verify (corrupt/partial); deleting and retraining."
    rm -f "${SAVE_PATH}"
fi

python -u src/train.py \
    --graph-dir "${GRAPH_DIR}" \
    --k "${K_VALUE}" --epsilon "${EPSILON}" --episodes "${EPISODES}" \
    --device cpu --no-amp \
    --coarsen-min 100 --coarsen-seeds "${COARSEN_SEEDS}" --ml-refine-max-n "${ML_REFINE_MAX_N}" \
    --d-model 64 --n-heads 4 --n-layers 2 --n-eigs "${N_EIGS}" \
    --max-steps 60 --min-steps 8 --ppo-epochs 3 \
    --log-every 50 --ckpt-every 200 \
    --save "${SAVE_PATH}"

echo "Done $(date). Models:"
ls -la "${MODEL_DIR}"/*.pt 2>/dev/null || true
