# Demo graphs

Nine benchmark graphs shipped so that `src/demo.py` can run a benchmark in
miniature end to end without any external data. The default run has two
waves: a six-graph roster (two headline wins, a mesh win near parity, a
tie, and two known parity-regime losses, in that order), then an extended
tier of three larger meshes in the separator-ceiling regime. add20 runs
on request only.

From the SuiteSparse collection (converted to METIS graph format):

- `rdb3200l.graph`: reaction-diffusion matrix, n = 3,200. Headline win.
- `conf5_0-4x4-14.graph`: lattice QCD configuration, n = 3,072. Headline win.
- `power.graph`: Western US power grid, n = 4,941.

From the Walshaw graph partitioning archive:

- `data.graph`: 3D FEM mesh, n = 2,851.
- `uk.graph`: near-planar mesh of Great Britain, n = 4,824.
- `3elt.graph`: 2D FEM airfoil mesh, n = 4,720.
- `crack.graph`: 2D fracture-propagation mesh, n = 10,240. Extended tier.
- `whitaker3.graph`: 2D FEM mesh, n = 9,800. Extended tier.
- `fe_4elt2.graph`: 2D FEM airfoil mesh, n = 11,143. Extended tier.
- `add20.graph`: 20-bit adder circuit, n = 2,395. On request only
  (`--graph graphs/add20.graph`): a matching stall in coarsening makes the
  hierarchy deep, about nine minutes.
