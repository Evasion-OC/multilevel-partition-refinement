#!/bin/bash -l
#SBATCH --job-name=ssl-transfer
#SBATCH --partition=cpu-standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/kg1111r/multilevel-partition-refinement/logs/%x-%j.out
#SBATCH --error=/home/kg1111r/multilevel-partition-refinement/logs/%x-%j.err
#SBATCH --chdir=/home/kg1111r/multilevel-partition-refinement
# =============================================================================
# The Huang-bridge experiment: does SSL pretraining transfer to an unseen family?
# Pretrain the spectral encoder (masked + contrastive) on families EXCLUDING the
# held-out one, transfer the backbone into the RL refiner, fine-tune on the held-out
# family, and compare to from-scratch over many seeds. Honest axis = transfer /
# sample-efficiency, NOT raw cut. Threads pinned to 1 for reproducibility.
#
#   sbatch hpc/Train_SSL_Transfer.sbatch
#   HELDOUT=ba,geometric SEEDS=5 PRE_EPOCHS=40 EPISODES=40 sbatch hpc/Train_SSL_Transfer.sbatch
# =============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kg1111r/multilevel-partition-refinement}"
POOL="${POOL:-${PROJECT_DIR}/data/pool/train}"
SCALE="${SCALE:-small}"                 # pool size if it must be generated
HELDOUT="${HELDOUT:-ba,geometric}"      # families tested separately (held out of pretraining)
PRETRAIN="${PRETRAIN:-auto}"            # 'auto' = all families minus the held-out one
SEEDS="${SEEDS:-5}"; PRE_EPOCHS="${PRE_EPOCHS:-40}"; EPISODES="${EPISODES:-40}"
LIMIT_CORPUS="${LIMIT_CORPUS:-6}"       # cap graphs/pretrain-family (keeps the SSL batch light)
LIMIT_HELDOUT="${LIMIT_HELDOUT:-8}"
RUN_TAG="${RUN_TAG:-xfer}"; EXTRA="${EXTRA:-}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/results"
cd "${PROJECT_DIR}"

module purge
module load slurm/25.11.3 || true
module load Python/3.11.10-GCCcore-12.3.0
module load libffi/3.4.4-GCCcore-12.3.0
source "${PROJECT_DIR}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

# self-contained: generate the family-tagged pool if it is empty (graphs are gitignored).
if ! ls "${POOL}"/*.graph >/dev/null 2>&1; then
    echo "pool ${POOL} empty -> generating (scale=${SCALE})"
    python data/prepare_data.py --scale "${SCALE}" --skip-views
fi

echo "SSL transfer | held-out=${HELDOUT} pretrain=${PRETRAIN} seeds=${SEEDS} | $(date)"
python -u src/train_ssl_transfer.py \
    --pool "${POOL}" --heldout "${HELDOUT}" --pretrain "${PRETRAIN}" \
    --seeds "${SEEDS}" --pre-epochs "${PRE_EPOCHS}" --episodes "${EPISODES}" \
    --limit-corpus "${LIMIT_CORPUS}" --limit-heldout "${LIMIT_HELDOUT}" \
    --threads 1 ${EXTRA} \
    --output "results/ssl_transfer_${RUN_TAG}.json"
echo "Done $(date). Results: results/ssl_transfer_${RUN_TAG}.json"
