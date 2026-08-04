# Running on the cluster

Every command here is meant for the login node; the compute goes through
`sbatch`. The `#SBATCH` directives hold absolute paths and cannot read shell
variables, so they assume the repository sits at
`/home/<user>/multilevel-partition-refinement`. Clone it
there, or edit the directive lines at the top of each script. Everything below
those lines reads `PROJECT_DIR` and `GRAPH_DIR` from the environment, so those
can be overridden per job.

## 1. Set up, once

```bash
bash hpc/setup_env.sh
```

This builds the virtual environment (Python 3.11 with `libffi`, no SciPy bundle,
torch matched to CUDA 12.1), generates the training pool and the validation pool
under `corpus/`, and runs a short CPU smoke test.

## 2. Train

```bash
sbatch hpc/Train_GPU.sbatch
```

Training draws graphs from every coarsening depth rather than from the coarsest
level alone, so the policy is competent at the sizes the refiner actually meets
during uncoarsening. The log line to watch reports how many source graphs
expanded into how many level graphs. The depth cap and the episode count come
from the environment:

```bash
ML_REFINE_MAX_N=4000 EPISODES=2000 sbatch hpc/Train_GPU.sbatch
```

`Train_CPU.sbatch` is the same job on standard nodes. At these graph sizes
the GPU is worth using mainly when the CPU queue is busy.

## 3. Benchmark

```bash
GRAPH_DIR=/path/to/graphs MAX_N=6000 sbatch hpc/Bench.sbatch
```

`Bench_array.sbatch` is the job array version, one task per value of k,
which is how the full table was produced. Both write JSONL to `results/`, one
record per graph, including the count of level refinements adopted against those
attempted.

## 4. Controls

```bash
sbatch hpc/Ctrl_Witness_2x2.sbatch
```

This completes the four-arm table of Appendix H through the same harness that
produced the rest of it (`src/lev1.py`, guard off, best of 4, full config), so
the arms come out directly comparable rather than merely similar.

```bash
sbatch hpc/Train_SSL_Transfer.sbatch
```

This runs the pretraining pilot.

## 5. Watch

```bash
squeue -u $USER
tail -f logs/*.out
```
