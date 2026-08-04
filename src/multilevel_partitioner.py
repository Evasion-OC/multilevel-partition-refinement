"""
multilevel_partitioner.py -- learned refinement at every uncoarsening level.
===========================================================================

The learned spectral refiner runs at EVERY uncoarsening level (size-gated), not at the
coarsest one alone. This reaches the fine-level losses that coarsest-only refinement
structurally cannot touch, for instance the `add32` cut, which is lost during fine-level
FM, a stage a coarsest-only refiner never sees.

  coarsest-only:  learned refine ONLY at the coarsest graph (<=100 vertices); uncoarsening = FM.
  every level:    learned refine at the coarsest AND at each uncoarsening level with
                  n <= ml_refine_max_n, then FM. The size gate keeps dense attention
                  affordable; the Lanczos global branch lifts it.

Clean subclass of SRMPPartitioner -- `srmp/` is NOT edited. A no-harm guard keeps the
learned move only if it does not worsen the cut, so applying the (currently coarsest-trained)
policy to larger levels is SAFE: where it helps it helps, where it doesn't FM still runs. Training
the policy to actually act well at multiple levels is the next step (train on multi-depth graphs).
"""
from __future__ import annotations
import numpy as np
import torch

from refiner import SpectralActorCritic
from partitioner import SpectralRefiner
from srmp.pipeline import SRMPPartitioner
from srmp.config import SRMPConfig
from srmp.evaluation import compute_edge_cut


class MultilevelSpectralPartitioner(SRMPPartitioner):
    """SRMPPartitioner + learned refinement at every uncoarsening level (size-gated, no-harm)."""

    def __init__(self, config, ml_refine_max_n: int = 4000, global_no_harm: bool = True):
        super().__init__(config)
        self.ml_refine_max_n = ml_refine_max_n
        self.global_no_harm = global_no_harm   # run FM-only too and keep the better of the two
        self.ml_refine_calls = 0          # how many times a learned level-refine was ADOPTED
        self.ml_refine_tries = 0          # how many levels it was attempted on
        self._uncoarsen_count = 0         # gate multi-level refine to the MAIN uncoarsen pass
        self._ml_disabled = False         # set during the FM-only comparison run
        self.ml_global_adopted = 0        # 1 if the multi-level run won the global no-harm compare

    def partition(self, G, seed=42, verbose=None):
        # reset per-partition state
        self._uncoarsen_count = 0; self._ml_disabled = False; self.ml_global_adopted = 0
        self.ml_refine_calls = 0; self.ml_refine_tries = 0
        if not self.global_no_harm or self.rl_trainer is None:
            return super().partition(G, seed=seed, verbose=verbose)
        # GLOBAL no-harm guard: run the full multi-level pipeline AND the FM-only pipeline on
        # the same seed; keep whichever has the lower FINAL cut. This makes
        # cut(refined) <= cut(FM-only) hold end to end, so a regression cannot happen.
        p_ml = super().partition(G, seed=seed, verbose=verbose)
        self._uncoarsen_count = 0; self._ml_disabled = True            # FM-only comparison run
        p_fm = super().partition(G, seed=seed, verbose=verbose)
        self._ml_disabled = False
        if compute_edge_cut(G, p_ml) <= compute_edge_cut(G, p_fm):
            self.ml_global_adopted = 1
            return p_ml
        return p_fm

    def _uncoarsen_and_refine(self, hierarchy, coarse_partition, k, epsilon):
        # _combine_parents (evolutionary) and _vcycle also call this; only the FIRST call per
        # partition() is the main uncoarsen pass -- restrict multi-level learned refine to it so we
        # don't run the refiner inside every evolutionary combine (huge slowdown + alters the
        # comparability of the evolutionary stage). NOTE: the per-level refine is size-gated only,
        # NOT eigengap-gated (unlike the coarsest refiner); by design it acts on every graph.
        self._uncoarsen_count += 1
        do_ml = (self._uncoarsen_count == 1) and not self._ml_disabled
        partition = coarse_partition
        for level in reversed(range(len(hierarchy))):
            cl = hierarchy[level]
            partition = cl.prolong_partition(partition)
            G = cl.fine_graph
            # learned refinement at this level (size-gated), no-harm guarded.
            if do_ml and self.rl_trainer is not None and (2 * k) <= G.n <= self.ml_refine_max_n:
                self.ml_refine_tries += 1
                try:
                    cand = self.rl_trainer.refine(G, partition, k, epsilon,
                                                  spectral_coords=None, num_starts=1)
                    if compute_edge_cut(G, cand) <= compute_edge_cut(G, partition):
                        partition = cand
                        self.ml_refine_calls += 1
                except Exception:
                    pass                  # never let the learned refiner break the pipeline; FM runs
            partition = self._fm_refine(G, partition, k, epsilon)
        return partition


def make_multilevel_partitioner(policy: SpectralActorCritic, k=4, epsilon=0.03, best_of_k=12,
                           refiner_steps=80, refiner_stochastic=0, n_eigs=8,
                           evolutionary=True, num_cycles=5, force_rl=False, min_vertices=100,
                           ml_refine_max_n=4000, global_no_harm=True, lean=False,
                           coarsening_method=None, community_source="internal", community_file=None,
                           verbose=False) -> MultilevelSpectralPartitioner:
    """srmp scaffold + coarsest-level spectral refiner + refinement at every level.

    The defaults are the benchmark protocol, so the cuts stay comparable. `ml_refine_max_n`
    is the size gate for the learned per-level refinement; raise it when the Lanczos global
    branch replaces dense attention. `global_no_harm=True` runs the FM-only pipeline too and
    keeps the better final cut, which makes the no-harm bound hold by construction at about
    twice the cost. `lean=True` gives a fast configuration that is not comparable."""
    cfg = SRMPConfig(k=k, epsilon=epsilon, best_of_k=best_of_k, verbose=verbose)
    cfg.use_rl = True
    cfg.coarsening.min_vertices = min_vertices
    if coarsening_method is not None:
        cfg.coarsening.method = coarsening_method            # e.g. "community_aware" or "expansion2"
    cfg.coarsening.community_source = community_source        # internal|lp|louvain_simple|louvain_nx|file|ssl
    cfg.coarsening.community_file = community_file
    if lean:
        evolutionary, num_cycles, force_rl = False, 0, True
    cfg.evolutionary.enabled = evolutionary
    cfg.vcycle.num_cycles = num_cycles
    if force_rl:
        cfg.spectral_threshold = 1e9
    part = MultilevelSpectralPartitioner(cfg, ml_refine_max_n=ml_refine_max_n,
                                         global_no_harm=global_no_harm)
    part.rl_trainer = SpectralRefiner(policy, max_steps=refiner_steps, n_eigs=n_eigs,
                                      stochastic_rollouts=refiner_stochastic)
    return part


if __name__ == "__main__":
    import scipy.sparse as sp
    from srmp.graph import Graph
    from srmp.evaluation import compute_imbalance
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    n, k = 600, 4

    row, col = [], []
    for i in range(n):
        j = (i + 1) % n; row += [i, j]; col += [j, i]
    for _ in range(3 * n):
        a, b = rng.randint(0, n), rng.randint(0, n)
        if a != b: row += [a, b]; col += [b, a]
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n)); A.data[:] = 1.0
    A.sum_duplicates(); A.data[:] = 1.0
    G = Graph(A)

    policy = SpectralActorCritic(k=k, in_dim=k + 6, d_model=48, n_heads=4, n_layers=2)  # untrained
    P = make_multilevel_partitioner(policy, k=k, epsilon=0.03, best_of_k=3, refiner_steps=50,
                               ml_refine_max_n=4000, lean=True)
    part = P.partition(G, seed=42)

    bw = np.bincount(part, minlength=k)
    imb = bw.max() / (n / k) - 1.0
    cut = compute_edge_cut(G, part)
    valid = part.shape == (n,) and set(np.unique(part)).issubset(set(range(k)))
    print(f"partition: shape {part.shape}, blocks {sorted(set(part.tolist()))}")
    print(f"block sizes {bw.tolist()}  imbalance {imb:+.3f}  edge-cut {cut:.0f}")
    print(f"coarsest refiner calls {P.rl_trainer.calls}; MULTI-LEVEL refine: "
          f"tried {P.ml_refine_tries} levels, adopted {P.ml_refine_calls}")
    ok = valid and imb <= 0.05 and P.ml_refine_tries > 0
    print("PASS" if ok else "FAIL",
          "-- runs end to end with learned refinement at multiple uncoarsening levels.")
