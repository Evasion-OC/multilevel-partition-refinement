#!/bin/bash -l
# =============================================================================
# One-time setup on the cluster login node.
#   Creates the virtualenv, installs deps (numpy/scipy + a CUDA-matched torch),
#   generates the training/val corpora, and runs a tiny CPU smoke test.
#   Safe + idempotent to paste on the login node; self-heals a half-built venv.
#
#   bash hpc/setup_env.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kg1111r/multilevel-partition-refinement}"
# torch: cu121 matches the cluster's known-good PyTorch/2.1.2-...-CUDA-12.1.1 module
# (driver supports CUDA 12.1). Override TORCH_SPEC / TORCH_INDEX for a different cluster.
TORCH_SPEC="${TORCH_SPEC:-torch==2.4.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
cd "${PROJECT_DIR}"

SMOKE_CKPT="/tmp/smoke_${USER}.pt"
trap 'rm -f "${SMOKE_CKPT}"' EXIT          # always clean up, even if the smoke test fails

# ---- known-good module stack (Python 3.11.10 + libffi, no SciPy-bundle) ----
module purge
module load slurm/25.11.3 || true
module load Python/3.11.10-GCCcore-12.3.0
module load libffi/3.4.4-GCCcore-12.3.0

echo ">> home/project disk space (torch+CUDA deps are ~5GB -- ensure quota has room):"
df -h "${PROJECT_DIR}" 2>/dev/null || true

# ---- virtualenv (recreate if missing OR half-built) ----
venv_ok() { [ -f "${PROJECT_DIR}/.venv/bin/activate" ] && \
            "${PROJECT_DIR}/.venv/bin/python" -c "import sys" >/dev/null 2>&1; }
if ! venv_ok; then
    echo ">> creating virtualenv at ${PROJECT_DIR}/.venv"
    rm -rf "${PROJECT_DIR}/.venv"
    python -m venv "${PROJECT_DIR}/.venv"
fi
source "${PROJECT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_DIR}/requirements.txt"
echo ">> installing ${TORCH_SPEC} from ${TORCH_INDEX}"
python -m pip install "${TORCH_SPEC}" --index-url "${TORCH_INDEX}"

echo ">> dependency check (cuda=False here is EXPECTED -- the login node has no GPU;"
echo "   GPU availability is verified inside the GPU job / preflight, not here):"
python - <<'PY'
import numpy, scipy, torch
print(f"   numpy {numpy.__version__} | scipy {scipy.__version__} | "
      f"torch {torch.__version__} (cuda build {torch.version.cuda}) | cuda_available {torch.cuda.is_available()}")
PY

# ---- generate corpora (METIS .graph; no data transfer needed; each guarded independently) ----
mkdir -p "${PROJECT_DIR}/corpus" "${PROJECT_DIR}/logs" "${PROJECT_DIR}/Models"
if [ -z "$(ls -A "${PROJECT_DIR}/corpus/train" 2>/dev/null)" ]; then
    echo ">> generating the training pool (n up to 5000)"
    python src/make_train_corpus.py --output-dir corpus/train --ns 500 1000 2000 5000 --seeds 0 1 2
else
    echo ">> corpus/train already populated; skipping"
fi
if [ -z "$(ls -A "${PROJECT_DIR}/corpus/val" 2>/dev/null)" ]; then
    echo ">> generating validation corpus"
    python src/make_train_corpus.py --output-dir corpus/val --ns 500 1000 --seeds 7 8
else
    echo ">> corpus/val already populated; skipping"
fi

# ---- tiny smoke test (a few CPU episodes; ~30s -- confirms the venv + the multi-depth code) ----
echo ">> smoke test (5 CPU episodes, fp32, multi-depth trainer on the first few val graphs):"
python -u src/train.py --graph-dir corpus/val --k 4 --epsilon 0.03 --episodes 5 \
    --device cpu --no-amp --ml-refine-max-n 1200 --coarsen-seeds 1 --max-steps 20 \
    --log-every 5 --ckpt-every 0 --limit-graphs 3 --save "${SMOKE_CKPT}"

echo ""
echo "Setup complete. Next:"
echo "  (GPU only) verify CUDA on a compute node before the long run:"
echo "    srun --partition=gpu-standard --gres=gpu:1 --time=00:05:00 bash -lc \\"
echo "      'source .venv/bin/activate && python -c \"import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))\"'"
echo "  Submit training (multi-depth):"
echo "    sbatch hpc/Train_GPU.sbatch     # GPU jumps the queue if cpu-standard is busy"
echo "    sbatch hpc/Train_CPU.sbatch     # CPU (comparable speed at these sizes)"
