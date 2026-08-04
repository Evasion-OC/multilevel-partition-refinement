#!/bin/bash
#SBATCH --job-name=witness-2x2
#SBATCH --partition=cpu-standard
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/kg1111r/multilevel-partition-refinement/logs/witness_2x2_%j.out
#SBATCH --error=/home/kg1111r/multilevel-partition-refinement/logs/witness_2x2_%j.out
# =============================================================================
# Complete the four-arm control table of Appendix H, through the SAME harness that
# produced its numbers (src/lev1.py, guard OFF, best_of_k=4, full config), so the new
# arm is directly comparable rather than merely similar.
#
# What the earlier lever runs already record on add32:
#     trained   stock  = 45 / 45 / 45      (seeds 0,1,2)
#     trained   stoch8 = 38 / 38 / 33      (seeds 0,1,2)   best 33, median 38
#     untrained stoch8 = 33 / 33           (two seeds only)
#     untrained stock  = NOT RUN  <-- the "---" cell of tab:app-c-add32
#
# The missing arm is a SINGLE rollout, so it is the cheapest configuration here
# (~140 s/seed against ~390 s/seed for stoch8). This job reruns all four arms over a
# common seed set, which also repairs the 3-seeds-vs-2-seeds mismatch that Appendix H
# lines 112 and 114 assert did not happen.
#
# A local probe (1 seed, best_of_k=4, deployed Lanczos backbone) put the missing cell
# at 33. If that holds here then Delta^srch_8 = c^0_1 - c^0_8 = 0, and the trained
# policy is 12 edges WORSE than its own random initialisation at single rollout.
#
# MODEL should be the ATTENTION-BACKBONE baseline checkpoint of the module ablation,
# which is what Appendix H says the control used. That matters here: lev1.py:41-46
# builds its untrained mirror WITHOUT passing global_kind/pe_kind, so the mirror always
# takes the attention defaults. Against a Lanczos checkpoint the arms would differ by
# backbone as well as by weights, which breaks "the policy weights are the only factor
# that changes" (Appendix H line 112).
#
#   MODEL=/path/to/attn_baseline.pt GRAPHS=add32 sbatch hpc/Ctrl_Witness_2x2.sbatch
# =============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kg1111r/multilevel-partition-refinement}"
MODEL="${MODEL:-${PROJECT_DIR}/models/spectral_refiner_k4_eps_0_03.pt}"
GRAPH_DIR="${GRAPH_DIR:-${PROJECT_DIR}/graphs}"   # override per job
GRAPHS="${GRAPHS:-add32}"
SEEDS="${SEEDS:-0 1 2}"
BEST_OF_K="${BEST_OF_K:-4}"
K_VALUE="${K_VALUE:-4}"; EPSILON="${EPSILON:-0.03}"
mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

module purge
module load slurm/25.11.3 || true
module load Python/3.11.10-GCCcore-12.3.0
module load libffi/3.4.4-GCCcore-12.3.0
source "${PROJECT_DIR}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" PYTHONUNBUFFERED=1

echo "model     ${MODEL}"
echo "graphs    ${GRAPHS}"
echo "seeds     ${SEEDS}"
echo "best_of_k ${BEST_OF_K}  (the witness value; the benchmark uses 12)"
echo "guard OFF, evolutionary ON -- the recorded witness conditions"
echo

for GRAPH in ${GRAPHS}; do
  for SEED in ${SEEDS}; do
    for ARM in "trained 0" "trained 8" "untrained 0" "untrained 8"; do
      set -- ${ARM}
      WHICH=$1; STOCH=$2
      UNTRAINED_FLAG=""
      [ "${WHICH}" = "untrained" ] && UNTRAINED_FLAG="--untrained"
      echo "--- ${GRAPH} seed=${SEED} ${WHICH} stoch=${STOCH} ---"
      python src/lev1.py \
          --model "${MODEL}" \
          --graph-dir "${GRAPH_DIR}" \
          --graph "${GRAPH}" \
          --use-rl 1 \
          --stoch "${STOCH}" \
          --seed "${SEED}" \
          --k "${K_VALUE}" --epsilon "${EPSILON}" \
          --best-of-k "${BEST_OF_K}" \
          --no-guard \
          ${UNTRAINED_FLAG}
    done
  done
done

echo
echo "############ read off ############"
echo "Group the RESULT lines by untrained= and stoch=."
echo "  c^theta_1 = trained  stoch=0     c^theta_8 = trained  stoch=8"
echo "  c^0_1     = untrained stoch=0    c^0_8     = untrained stoch=8   <-- new cell"
echo "  Delta^srch_8 = c^0_1 - c^0_8      Delta^lrn_8 = c^0_8 - c^theta_8"
echo "Report BOTH the median and the best over seeds: Appendix H uses best-over-seeds,"
echo "and on the recorded data the median tells a stronger story than the best does."
echo "NOTE lev1.py:42 seeds the untrained policy from --seed, so each seed is a"
echo "DIFFERENT random initialisation. That is a feature: agreement across seeds is"
echo "agreement across independent random policies, not one lucky draw."
