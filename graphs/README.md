# Demo graphs

Six benchmark graphs shipped so that `src/demo.py` can run a benchmark in
miniature end to end without any external data: two headline wins, a mesh
win near parity, a tie, and two known parity-regime losses, in that order.

From the SuiteSparse collection (converted to METIS graph format):

- `rdb3200l.graph`: reaction-diffusion matrix, n = 3,200. Headline win.
- `conf5_0-4x4-14.graph`: lattice QCD configuration, n = 3,072. Headline win.
- `power.graph`: Western US power grid, n = 4,941.

From the Walshaw graph partitioning archive:

- `data.graph`: 3D FEM mesh, n = 2,851.
- `uk.graph`: near-planar mesh of Great Britain, n = 4,824.
- `3elt.graph`: 2D FEM airfoil mesh, n = 4,720.
- `add20.graph`: 20-bit adder circuit, n = 2,395. The slowest of the seven
  (a matching stall in coarsening makes the hierarchy deep), several minutes.
