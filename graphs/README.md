# Demo graphs

Two benchmark graphs from the SuiteSparse collection, converted to METIS
graph format and shipped so that `src/demo.py` can partition real benchmark
graphs end to end without any external data. Both are headline wins of
Chapter 5:

- `rdb3200l.graph`: reaction-diffusion matrix (Bai group), n = 3,200.
  About 15 s at the demo budget.
- `conf5_0-4x4-14.graph`: lattice QCD configuration, n = 3,072.
  About 95 s at the demo budget; runs under `--wins`.
