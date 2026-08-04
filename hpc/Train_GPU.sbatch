#!/bin/bash -l
#SBATCH --job-name=train-gpu
#SBATCH --partition=gpu-standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/kg1111r/multilevel-partition-refinement/logs/%x-%j.out
#SBATCH --error=/home/kg1111r/multilevel-partition-refinement/logs/%x-%j.err
#SBATCH --chdir=/home/kg1111r/multilevel-partition-refinement
# =============================================================================
# Multi-depth RL training on GPU. Trains the GPS-hybrid SpectralActorCritic
# on graphs at EVERY coarsening depth (2k <= n <= ml_refine_max_n), so the policy
# is competent at all the level sizes the multi-level refiner acts on.
#   sbatch hpc/Train_GPU.sbatch
#   ML_REFINE_MAX_N=4000 EPISODES=4000 sbatch hpc/Train_GPU.sbatch
# =============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kg1111r/multilevel-partition-refinement}"
MODEL_DIR="${PROJECT_DIR}/Models"
# prefer the full uploaded corpus (data/pool/train, ~205 sources up to n=8000); fall back to the
# smaller setup_env corpus (corpus/train, 72 sources up to n=5000) only on a fresh checkout.
GRAPH_DIR="${GRAPH_DIR:-$([ -d "${PROJECT_DIR}/data/pool/train" ] && echo "${PROJECT_DIR}/data/pool/train" || echo "${PROJECT_DIR}/corpus/train")}"
RUN_TAG="${RUN_TAG:-synth}"
COARSEN_SEEDS="${COARSEN_SEEDS:-3}"
N_EIGS="${N_EIGS:-8}"
ML_REFINE_MAX_N="${ML_REFINE_MAX_N:-4000}"                 # depth cap = largest level trained on
GLOBAL_KIND="${GLOBAL_KIND:-attn}"                         # attn (baseline) | lanczos (Module A)
PE_KIND="${PE_KIND:-stable}"                               # stable (baseline) | specformer (Module B)
mkdir -p "${MODEL_DIR}" "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

module purge
module load slurm/25.11.3 || true
module load Python/3.11.10-GCCcore-12.3.0
module load libffi/3.4.4-GCCcore-12.3.0
source "${PROJECT_DIR}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

K_VALUE="${K_VALUE:-4}"
EPSILON="${EPSILON:-0.03}"
EPISODES="${EPISODES:-2000}"
eps_tag=$(echo "${EPSILON}" | sed 's/\./_/g')
SAVE_PATH="${MODEL_DIR}/refiner_${RUN_TAG}_k${K_VALUE}_eps_${eps_tag}.pt"

echo "============================================================"
echo "  GPU training (multi-depth)"
echo "  Node    : $(hostname)"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  k=${K_VALUE} eps=${EPSILON} episodes=${EPISODES} ml_refine_max_n=${ML_REFINE_MAX_N}"
echo "  Graphs  : ${GRAPH_DIR} ($(ls "${GRAPH_DIR}" 2>/dev/null | wc -l) files)"
echo "  Save    : ${SAVE_PATH}"
echo "  Started : $(date)"
echo "============================================================"

if [ -f "${SAVE_PATH}" ]; then
    if python -c "from refiner import load_spectral_actor_critic; load_spectral_actor_critic('${SAVE_PATH}')" 2>/dev/null; then
        echo "Checkpoint exists and reload-verifies; skipping. Remove it to retrain."
        exit 0
    fi
    echo "Existing checkpoint failed reload-verify (corrupt/partial); deleting and retraining."
    rm -f "${SAVE_PATH}"
fi

# reduce CUDA fragmentation OOMs (esp. the lanczos/Module-A run, which varies graph sizes per episode)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available -- check torch build vs cluster driver'; print('CUDA preflight OK:', torch.cuda.get_device_name(0))"

python -u src/train.py \
    --graph-dir "${GRAPH_DIR}" \
    --k "${K_VALUE}" --epsilon "${EPSILON}" --episodes "${EPISODES}" \
    --device cuda --amp \
    --coarsen-min 100 --coarsen-seeds "${COARSEN_SEEDS}" --ml-refine-max-n "${ML_REFINE_MAX_N}" \
    --d-model 64 --n-heads 4 --n-layers 2 --n-eigs "${N_EIGS}" \
    --global-kind "${GLOBAL_KIND}" --pe-kind "${PE_KIND}" \
    --max-steps 60 --min-steps 8 --ppo-epochs 3 \
    --log-every 50 --ckpt-every 200 \
    --save "${SAVE_PATH}"

echo "Done $(date). Models:"
ls -la "${MODEL_DIR}"/*.pt 2>/dev/null || true
