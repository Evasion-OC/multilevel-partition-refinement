"""
SRMP Pipeline -- Algorithm-Structure Matching Multilevel Partitioner
=====================================================================
Multilevel partitioning pipeline used as the experimental substrate for
the thesis programme (T1/C2/C3; see README). The pipeline wires:
- Best-of-K coarse initialization
- RL refinement at coarsest level (GATv2 + SignNet; T1 depends on SignNet)
- Advanced FM knobs (multi-try, flow, alpha scaling, unconstrained)
- Expansion*2 + GPA coarsening controls (C2 compares this against LP)
- V-cycle / F-cycle iteration
- KaFFPaE-style evolutionary combine phase

Thesis linkage:
  * T1: SignNet-based refinement acts on CFI pairs here; tested via
        scripts/cfi_separation_experiment.py.
  * C2: coarsening method choice (expansion2 vs LP) is the C2 phase-transition
        variable; landscaped via scripts/coarsening_phase_landscape.py.
  * C3: thesis_features extracted per run feed the classifier consistency
        experiments in scripts/classifier_{learning_curve,feature_ablation}.py.
"""
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Tuple, List

from .graph import Graph
from .config import SRMPConfig
from .spectral import compute_eigenvectors, eigengap
from .coarsening import (
    build_hierarchy,
    CoarseningLevel,
    matching_to_mapping,
    contract_graph,
    evolutionary_combine_matching,
)
from .initial_partition import multi_start_partition, enforce_balance
from .fm_refinement import fm_refine_2way, fm_refine_kway
from .evaluation import compute_edge_cut, compute_imbalance, evaluate_partition


class SRMPPartitioner:
    def __init__(self, config: SRMPConfig):
        self.config = config
        self.rl_trainer = None

    def _compute_coarsest_spectral(self, coarsest: Graph, k: int) -> Optional[np.ndarray]:
        try:
            spec_dim = self.config.rl.spectral_dim
            n_eigs = min(spec_dim + 1, coarsest.n - 1)
            if n_eigs < 2:
                return None
            _, evecs = compute_eigenvectors(coarsest, n_eigs, normalized=True)
            coords = evecs[:, 1:min(spec_dim + 1, evecs.shape[1])]
            if coords.shape[1] < spec_dim:
                pad = np.zeros((coarsest.n, spec_dim - coords.shape[1]))
                coords = np.hstack([coords, pad])
            elif coords.shape[1] > spec_dim:
                coords = coords[:, :spec_dim]
            return coords
        except Exception:
            return None

    def _fm_refine(self, G: Graph, partition: np.ndarray, k: int, epsilon: float) -> np.ndarray:
        if k == 2:
            part, _ = fm_refine_2way(
                G,
                partition,
                epsilon,
                max_passes=self.config.fm.max_passes,
                early_stop_passes=self.config.fm.early_stop_passes,
                unconstrained=self.config.fm.unconstrained,
                no_harm_rollback=self.config.fm.no_harm_rollback,
            )
            return part

        part, _ = fm_refine_kway(
            G,
            partition,
            k,
            epsilon,
            max_passes=self.config.fm.max_passes,
            early_stop_passes=self.config.fm.early_stop_passes,
            use_lp=self.config.fm.use_lp_refinement,
            use_multi_try=self.config.fm.use_multi_try,
            use_flow=self.config.fm.use_flow_refinement,
            multi_try_rounds=self.config.fm.multi_try_rounds,
            flow_alpha=self.config.fm.flow_alpha,
            lp_iterations=self.config.fm.lp_refine_iterations,
            flow_alpha_scale=self.config.fm.flow_alpha_scale,
            flow_alpha_min=self.config.fm.flow_alpha_min,
            flow_alpha_max=self.config.fm.flow_alpha_max,
            unconstrained=self.config.fm.unconstrained,
            no_harm_rollback=self.config.fm.no_harm_rollback,
        )
        return part

    def _uncoarsen_and_refine(
        self,
        hierarchy: List[CoarseningLevel],
        coarse_partition: np.ndarray,
        k: int,
        epsilon: float,
    ) -> np.ndarray:
        partition = coarse_partition
        for level in reversed(range(len(hierarchy))):
            cl = hierarchy[level]
            partition = cl.prolong_partition(partition)
            partition = self._fm_refine(cl.fine_graph, partition, k, epsilon)
        return partition

    def _best_of_k_coarse(
        self,
        coarsest: Graph,
        k: int,
        epsilon: float,
        seed: int,
        use_rl: bool,
        spectral_coords: Optional[np.ndarray],
    ) -> np.ndarray:
        K = max(1, min(int(self.config.best_of_k), 64))
        best_part = None
        best_score = float("inf")

        for i in range(K):
            local_seed = seed + 97 * i
            init = multi_start_partition(
                coarsest,
                k,
                epsilon,
                num_starts=max(10, min(32, K)),
                seed=local_seed,
            )
            init = enforce_balance(coarsest, init, k, epsilon)

            cand = init
            if use_rl and self.rl_trainer is not None:
                try:
                    rl_cand = self.rl_trainer.refine(
                        coarsest,
                        init,
                        k,
                        epsilon,
                        spectral_coords=spectral_coords,
                        num_starts=K,
                    )
                    # No-harm guard: only adopt RL output if it does not worsen the cut
                    if compute_edge_cut(coarsest, rl_cand) <= compute_edge_cut(coarsest, init):
                        cand = rl_cand
                except Exception as _rl_err:
                    import traceback as _tb
                    print(f'[RL FALLBACK] {type(_rl_err).__name__}: {_rl_err}', flush=True)
                    _tb.print_exc()
                    cand = init

            cut = compute_edge_cut(coarsest, cand)
            imb = compute_imbalance(coarsest, cand, k)
            score = cut if imb <= epsilon + 1e-6 else cut * (1.0 + 10.0 * imb)
            if score < best_score:
                best_score = score
                best_part = cand.copy()

        if best_part is None:
            best_part = multi_start_partition(coarsest, k, epsilon, num_starts=10, seed=seed)
        return enforce_balance(coarsest, best_part, k, epsilon)

    def _combine_parents(
        self,
        G: Graph,
        parent1: np.ndarray,
        parent2: np.ndarray,
        k: int,
        epsilon: float,
        seed: int,
    ) -> np.ndarray:
        rng = np.random.RandomState(seed)
        current = G
        p1 = parent1.copy()
        p2 = parent2.copy()
        hierarchy: List[CoarseningLevel] = []

        for _ in range(self.config.coarsening.max_levels):
            if current.n <= max(self.config.coarsening.min_vertices, 2 * k):
                break

            match = evolutionary_combine_matching(
                current,
                p1,
                p2,
                rng,
                use_expansion2=self.config.coarsening.use_expansion2,
            )
            mapping = matching_to_mapping(match)
            coarse_graph, mapping = contract_graph(current, mapping)

            if coarse_graph.n >= 0.95 * current.n:
                break

            p1_coarse = np.zeros(coarse_graph.n, dtype=np.int64)
            p2_coarse = np.zeros(coarse_graph.n, dtype=np.int64)
            for v in range(current.n):
                p1_coarse[mapping[v]] = p1[v]
                p2_coarse[mapping[v]] = p2[v]

            hierarchy.append(CoarseningLevel(current, coarse_graph, mapping))
            current = coarse_graph
            p1 = p1_coarse
            p2 = p2_coarse

        # Resolve disagreements using connectivity to agreed-upon neighbors
        # rather than random assignment (improves combine quality significantly)
        score1 = compute_edge_cut(current, p1)
        score2 = compute_edge_cut(current, p2)
        coarse = p1.copy() if score1 <= score2 else p2.copy()
        other = p2 if score1 <= score2 else p1
        disagree = p1 != p2
        if np.any(disagree):
            disagree_verts = np.where(disagree)[0]
            # For each disagreement vertex, pick the block with more
            # agreed-upon neighbor connectivity
            for v in disagree_verts:
                nbrs, weights = current.neighbors(v)
                # Count weighted connectivity to each block from agreed neighbors
                block_conn = np.zeros(k, dtype=np.float64)
                for nb, w in zip(nbrs, weights):
                    if not disagree[nb]:  # agreed-upon neighbor
                        block_conn[coarse[nb]] += w
                if block_conn.sum() > 0:
                    coarse[v] = int(np.argmax(block_conn))
                # else: keep the better parent's assignment
        coarse = enforce_balance(current, coarse, k, epsilon)

        return self._uncoarsen_and_refine(hierarchy, coarse, k, epsilon)

    def _evolutionary_improve(
        self,
        G: Graph,
        base_partition: np.ndarray,
        k: int,
        epsilon: float,
        seed: int,
    ) -> np.ndarray:
        evo = self.config.evolutionary
        if not evo.enabled:
            return base_partition

        rng = np.random.RandomState(seed)
        pop_size = max(2, evo.population_size)

        population: List[np.ndarray] = [base_partition.copy()]
        while len(population) < pop_size:
            candidate = base_partition.copy()
            boundary = G.boundary_vertices_fast(candidate)
            if len(boundary) > 0:
                n_move = max(1, len(boundary) // 20)
                move_set = rng.choice(boundary, size=min(n_move, len(boundary)), replace=False)
                for v in move_set:
                    nbrs, _ = G.neighbors(v)
                    adj_blocks = [candidate[nb] for nb in nbrs if candidate[nb] != candidate[v]]
                    if adj_blocks:
                        candidate[v] = adj_blocks[rng.randint(len(adj_blocks))]
            candidate = enforce_balance(G, candidate, k, epsilon)
            candidate = self._fm_refine(G, candidate, k, epsilon)
            population.append(candidate)

        def score(part: np.ndarray) -> float:
            imb = compute_imbalance(G, part, k)
            cut = compute_edge_cut(G, part)
            return cut if imb <= epsilon + 1e-6 else cut * (1.0 + 10.0 * imb)

        for gen in range(evo.num_generations):
            i, j = rng.choice(len(population), size=2, replace=False)
            child = self._combine_parents(
                G,
                population[i],
                population[j],
                k,
                epsilon,
                seed + 1000 + gen,
            )
            population.append(child)
            population = sorted(population, key=score)[:pop_size]

        return population[0]

    def partition(
        self,
        G: Graph,
        seed: int = 42,
        verbose: Optional[bool] = None,
    ) -> np.ndarray:
        if verbose is None:
            verbose = self.config.verbose

        k = self.config.k
        epsilon = self.config.epsilon
        t_start = time.time()

        if verbose:
            print(f"Phase 1: Spectral analysis (n={G.n}, m={G.m}, k={k})")

        num_eigs = min(2 * k + 2, G.n - 1)
        eigenvalues, _ = compute_eigenvectors(G, num_eigs, normalized=True)
        delta_k = eigengap(eigenvalues, k)
        lambda_k = eigenvalues[min(k - 1, len(eigenvalues) - 1)]
        eigengap_ratio = delta_k / max(lambda_k, 1e-12)

        if verbose:
            print(
                f"  lambda2={eigenvalues[1]:.6f}, delta_k={delta_k:.6f}, "
                f"eigengap_ratio={eigengap_ratio:.4f}"
            )

        if verbose:
            print("Phase 2: Multilevel coarsening")

        coarsening_method = self.config.coarsening.method or "auto"

        # community-aware coarsening: resolve communities ONCE from the configured source
        # (internal Louvain by default; or LP / nx-Louvain / a .comm file / the SSL head)
        communities = None
        if coarsening_method == "community_aware":
            from .coarsening import resolve_communities
            communities = resolve_communities(
                G, self.config.coarsening.community_source, seed,
                self.config.coarsening.community_file)

        hierarchy = build_hierarchy(
            G,
            min_vertices=max(self.config.coarsening.min_vertices, 2 * k),
            max_levels=self.config.coarsening.max_levels,
            method=coarsening_method,
            seed=seed,
            lp_iterations=self.config.coarsening.lp_iterations,
            use_expansion2=self.config.coarsening.use_expansion2,
            use_gpa=self.config.coarsening.use_gpa_matching,
            communities=communities,
        )

        coarsest = hierarchy[-1].coarse_graph if hierarchy else G
        if verbose:
            print(f"  {len(hierarchy)} levels: {G.n} -> {coarsest.n} vertices")

        if verbose:
            print("Phase 3: Best-of-K coarse initialization")

        use_rl = (
            self.config.use_rl
            and self.rl_trainer is not None
            and eigengap_ratio < self.config.spectral_threshold
        )
        spec_coords = self._compute_coarsest_spectral(coarsest, k) if use_rl else None
        coarse_partition = self._best_of_k_coarse(
            coarsest,
            k,
            epsilon,
            seed,
            use_rl=use_rl,
            spectral_coords=spec_coords,
        )

        if verbose:
            cut = compute_edge_cut(coarsest, coarse_partition)
            imb = compute_imbalance(coarsest, coarse_partition, k)
            print(f"  Best-of-K coarse cut={cut:.0f}, imb={imb:.4f}, K={self.config.best_of_k}")

        if verbose:
            print("Phase 4: Uncoarsening + advanced FM refinement")
        partition = self._uncoarsen_and_refine(hierarchy, coarse_partition, k, epsilon)

        if self.config.vcycle.num_cycles > 0:
            if verbose:
                cycle_kind = "F-cycle" if self.config.vcycle.use_fcycle else "V-cycle"
                print(f"Phase 5: {cycle_kind} iteration")

            for cycle in range(self.config.vcycle.num_cycles):
                prev_cut = compute_edge_cut(G, partition)
                if self.config.vcycle.use_fcycle:
                    partition = self._fcycle(
                        G,
                        partition,
                        k,
                        epsilon,
                        seed + cycle + 100,
                        depth=self.config.vcycle.fcycle_depth,
                    )
                else:
                    partition = self._vcycle(G, partition, k, epsilon, seed + cycle + 100)

                new_cut = compute_edge_cut(G, partition)
                if verbose:
                    print(f"  cycle {cycle + 1}: cut {prev_cut:.0f} -> {new_cut:.0f}")
                if self.config.vcycle.early_stop and new_cut >= prev_cut:
                    break

        if self.config.evolutionary.enabled:
            if verbose:
                print("Phase 6: Evolutionary combine")
            partition = self._evolutionary_improve(
                G,
                partition,
                k,
                epsilon,
                seed + 5000,
            )

        # Final FM polish after evolutionary phase
        partition = self._fm_refine(G, partition, k, epsilon)

        final_imb = compute_imbalance(G, partition, k)
        if final_imb > epsilon + 1e-6:
            partition = enforce_balance(G, partition, k, epsilon)

        if verbose:
            final_cut = compute_edge_cut(G, partition)
            final_imb = compute_imbalance(G, partition, k)
            print(f"\nResult: cut={final_cut:.0f}, imbalance={final_imb:.4f}, time={time.time() - t_start:.2f}s")

        return partition

    def _vcycle(
        self,
        G: Graph,
        partition: np.ndarray,
        k: int,
        epsilon: float,
        seed: int,
    ) -> np.ndarray:
        hierarchy: List[CoarseningLevel] = []
        current_graph = G
        current_partition = partition.copy()

        for _ in range(self.config.coarsening.max_levels):
            if current_graph.n <= max(self.config.coarsening.min_vertices, 2 * k):
                break

            cl = build_hierarchy(
                current_graph,
                min_vertices=max(self.config.coarsening.min_vertices, 2 * k),
                max_levels=1,
                method=self.config.coarsening.method or "auto",
                seed=seed,
                use_expansion2=self.config.coarsening.use_expansion2,
                use_gpa=self.config.coarsening.use_gpa_matching,
            )
            if not cl:
                break
            lvl = cl[0]

            coarse_partition = np.zeros(lvl.coarse_graph.n, dtype=np.int64)
            for v in range(lvl.fine_graph.n):
                coarse_partition[lvl.mapping[v]] = current_partition[v]

            hierarchy.append(lvl)
            current_graph = lvl.coarse_graph
            current_partition = coarse_partition

            if current_graph.n >= 0.95 * lvl.fine_graph.n:
                break

        return self._uncoarsen_and_refine(hierarchy, current_partition, k, epsilon)

    def _fcycle(
        self,
        G: Graph,
        partition: np.ndarray,
        k: int,
        epsilon: float,
        seed: int,
        depth: int,
    ) -> np.ndarray:
        refined = self._vcycle(G, partition, k, epsilon, seed)
        if depth <= 1:
            return refined
        return self._fcycle(G, refined, k, epsilon, seed + 17, depth - 1)

    def multi_run(
        self,
        G: Graph,
        num_runs: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        if num_runs is None:
            num_runs = self.config.num_runs

        k = self.config.k
        epsilon = self.config.epsilon

        cuts = []
        best_cut = float("inf")
        best_partition = None
        best_cut_any = float("inf")
        best_partition_any = None

        use_parallel = (
            bool(getattr(self.config, "parallel_runs", False))
            and num_runs > 1
            and int(getattr(self.config, "num_workers", 1)) > 1
        )
        num_workers = min(num_runs, max(1, int(getattr(self.config, "num_workers", 1))))

        if use_parallel:
            completed = 0
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = [
                    ex.submit(self.partition, G, self.config.seed + run, False)
                    for run in range(num_runs)
                ]
                for fut in as_completed(futures):
                    partition = fut.result()
                    cut = compute_edge_cut(G, partition)
                    imbalance = compute_imbalance(G, partition, k)
                    cuts.append(cut)
                    completed += 1

                    if cut < best_cut_any:
                        best_cut_any = cut
                        best_partition_any = partition.copy()

                    if cut < best_cut and imbalance <= epsilon + 1e-6:
                        best_cut = cut
                        best_partition = partition.copy()

                    if verbose and completed % 5 == 0:
                        bal_str = "ok" if imbalance <= epsilon + 1e-6 else "imb"
                        print(f"  Run {completed}/{num_runs}: cut={cut:.0f} [{bal_str}], best={best_cut:.0f}")
        else:
            for run in range(num_runs):
                partition = self.partition(G, seed=self.config.seed + run, verbose=False)
                cut = compute_edge_cut(G, partition)
                imbalance = compute_imbalance(G, partition, k)
                cuts.append(cut)

                if cut < best_cut_any:
                    best_cut_any = cut
                    best_partition_any = partition.copy()

                if cut < best_cut and imbalance <= epsilon + 1e-6:
                    best_cut = cut
                    best_partition = partition.copy()

                if verbose and (run + 1) % 5 == 0:
                    bal_str = "ok" if imbalance <= epsilon + 1e-6 else "imb"
                    print(f"  Run {run + 1}/{num_runs}: cut={cut:.0f} [{bal_str}], best={best_cut:.0f}")

        if best_partition is None:
            best_partition = enforce_balance(G, best_partition_any, k, epsilon)

        stats = {
            "min_cut": min(cuts),
            "mean_cut": np.mean(cuts),
            "std_cut": np.std(cuts),
            "all_cuts": cuts,
            "best_imbalance": compute_imbalance(G, best_partition, k),
        }

        if verbose:
            print(f"\n{num_runs} runs: best={stats['min_cut']:.0f}, avg={stats['mean_cut']:.1f}+-{stats['std_cut']:.1f}")

        return best_partition, stats

    def evaluate(self, G: Graph, partition: np.ndarray) -> Dict[str, float]:
        return evaluate_partition(G, partition, self.config.k, self.config.epsilon)
