# Spectral Graph Transformers for Multilevel Partition Refinement: Expressivity, Invariance, and Linear-Time Encoding

Code for the MSc dissertation of that title, University of Greenwich, 2026.

The object of study is a spectral transformer acting as a partition refiner
inside the multilevel V-cycle. Two questions shape it. Which group actions must
the encoder respect for a refiner on graphs to be well defined, and how far can
such an encoder tell graphs apart, measured against the Weisfeiler–Leman
hierarchy. The partitioning benchmark then asks whether an encoder built to
those requirements is any good at the task it was built for.

## The design

One refiner, its parameters shared across levels, runs at **every** uncoarsening
level of the V-cycle, on the level graphs whose size falls in the gate
2k ≤ n ≤ 4000. Earlier learned refiners act only at the coarsest level, where
the graph is small enough for a network but where the fine-level losses have not
been made yet. A per-level guard and a global guard compare realised cuts alone,
which keeps cut(refined) ≤ cut(FM-only) true by construction, so the benchmark
reads as a measurement of what learning adds rather than of what it risks.

## Invariance

Three group actions leave the encoded state fixed.

| action on the input | group |
|---|---|
| vertex relabelling, G ↦ PGPᵀ | Sₙ |
| eigenvector sign flip, V ↦ VS | {±1}ᵈ |
| change of basis inside a degenerate eigenspace | O(m) |

These are symmetries of the construction, not fitted behaviour. The learned
filters read eigenvalues only, and eigenvectors enter through the eigenprojector
V diag(g(λ)) Vᵀ, which is fixed both by the sign group and by rotations inside
an eigenspace. Measured end to end they hold to about 10⁻⁶
(`src/verify_ab.py`, `src/demo.py`).

## Expressive power

The spectral path separates pairs that 1-WL cannot separate, and the encoder
classifies CSL at 100%. Its power stays incomparable to 1-WL rather than sitting
above it: co-spectral pairs defeat it, and it is not 3-WL-complete. The probes
are `src/csl_probe.py`, `src/cfi_probe.py` and `src/substructure_probe.py`.

## Cost

The global branch is a `LanczosSpectralMix` after LanczosNet. Its cost is linear
in the number of vertices where dense attention is quadratic, which is what
makes a refiner at every level affordable at all. At n = 50,000 the Lanczos
branch completes and dense attention exhausts memory (`src/scaling_test.py`).

## What was measured

The `SpectralActorCritic` has 180,731 parameters and trains by PPO with GAE on
205 synthetic source graphs, which expand to 2,584 level graphs across depths.
The test set is 67 held-out real and control graphs, none of them seen in
training. The baseline is METIS run at a matched number of trials, so both sides
of every ratio work under the same budget.

| protocol | wins | geometric mean cut ratio |
|---|---|---|
| seed median, equal trial budget | 45 of 67 | 0.979 |
| best of three seeds | 50 of 67 | 0.964 |

A scale study on the Walshaw large tier (62k to 156k vertices) is still running
and sits outside the dissertation's established results; its numbers will land
here once the runs complete.

## What the evidence does not support

Two controls qualify the gain, and the dissertation reports them beside the
positive results. The stochastic rollout lever turns out to be a search effect,
since an untrained policy given a best-of-8 budget matches the trained one. A
learned community-based coarsener loses to classical label propagation. Eight
graphs lose systematically, most of them near-planar circuit and stiffness
matrices whose optimal cuts are very small.

## Layout

| Path | Contents |
|---|---|
| `src/refiner.py`, `src/graph_transformer.py`, `src/spectral_pe.py` | The `SpectralActorCritic`, the encoder with its Lanczos global branch, and the basis-invariant spectral positional encoding |
| `src/multilevel_partitioner.py`, `src/partitioner.py` | The V-cycle carrying the refiner at every level, and the two guards |
| `src/srmp/` | Multilevel scaffold: graph container, coarsening, initial partitioning, FM refinement, pipeline, spectral routines |
| `src/train.py` | PPO training with GAE over the multi-depth level graphs |
| `src/benchmark.py` | Benchmark against METIS at a matched trial budget |
| `src/verify_ab.py`, `src/test_ab_train_ckpt.py` | Invariance measurement, backbone parity, and the trainability and checkpoint round-trip gate |
| `src/scaling_test.py` | Encode time against n, out to n = 50,000 |
| `src/csl_probe.py`, `src/cfi_probe.py`, `src/substructure_probe.py` | Expressivity probes: CSL, CFI pairs, substructure counts |
| `src/analyze_ceiling.py`, `src/struct_descriptors.py` | The structural descriptors τ and σ_R, and the ceiling analysis of where a win can come from |
| `src/lev1.py`, `src/make_untrained_ckpt.py` | The search against learning control, running trained and untrained policies through one harness |
| `src/probe_coarsening.py`, `src/community_ssl.py` | The learned coarsening control, against label propagation |
| `src/pretrain.py`, `src/train_ssl_transfer.py` | The pretraining pilot |
| `src/make_train_corpus.py`, `data/prepare_data.py`, `data/make_social_benchmark.py` | Building the training pool and the test sets |
| `src/demo.py` | Short demonstration: the checkpoint reports itself, the three invariances are measured live, and the two global branches are timed against each other |
| `hpc/` | Slurm scripts for training, for the benchmark, and for the controls |
| `models/` | The four deployed checkpoints, k = 4, 8, 16 and 32, each storing its own constructor arguments |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Measure the invariances and time the two global branches, which takes about a
minute and picks up `models/spectral_refiner_k4_eps_0_03.pt` on its own:

```bash
python src/demo.py
```

Benchmark a checkpoint against METIS on a directory of graphs in METIS format:

```bash
python src/benchmark.py --model models/spectral_refiner_k4_eps_0_03.pt \
    --graph-dir /path/to/graphs --k 4 --epsilon 0.03 --seeds 3 --metis \
    --output results/bench.jsonl
```

Train from scratch:

```bash
python src/train.py --help
```

The trained checkpoints are in `models/`, described in `models/README.md`. Graph
collections are not stored here, for size reasons, so the scripts that need them take
an explicit `--graph-dir`. `hpc/README.md` gives the workflow that produced the
dissertation numbers on the cluster.
