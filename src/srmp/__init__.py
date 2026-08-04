"""
SRMP: Spectral-RL Multilevel Partitioner (v2)
===============================================
A hybrid framework for balanced k-way graph partitioning combining
spectral graph theory, GNN-based reinforcement learning, and classical
Fiduccia-Mattheyses refinement within the multilevel paradigm.

v2 changes:
  - Balance enforcement throughout the pipeline
  - RL restricted to coarsest level only
  - Rollback mechanism in RL episodes
  - Batched PPO updates
  - Higher entropy coefficient to prevent policy collapse
  - Soft balance in RL env with penalty shaping

Quick start:
    from srmp import SRMPPartitioner, SRMPConfig, read_metis

    G = read_metis("add20.graph")
    config = SRMPConfig(k=4, epsilon=0.03)
    partitioner = SRMPPartitioner(config)
    partition = partitioner.partition(G)
    metrics = partitioner.evaluate(G, partition)
"""

from .config import SRMPConfig, CoarseningConfig, FMConfig, RLConfig, VCycleConfig
from .graph import Graph, read_metis, write_metis
from .graph import generate_ba_graph, generate_er_graph, generate_delaunay_graph
from .graph import generate_sbm_graph, generate_geometric_graph, generate_grid_graph
from .pipeline import SRMPPartitioner
from .evaluation import (
    compute_edge_cut,
    compute_normalized_cut,
    compute_imbalance,
    evaluate_partition,
    validate_partition,
    performance_profile,
    geometric_mean_ratio,
)
from .spectral import (
    compute_eigenvectors,
    fiedler_vector,
    spectral_embedding,
    estimate_conductance,
    donath_hoffman_bound,
)
from .structural_features import extract_structural_features, predict_rl_advantage
from .fm_refinement import fm_refine_2way, fm_refine_kway
from .coarsening import build_hierarchy, coarsen_graph
from .initial_partition import (
    spectral_partition, greedy_graph_growing, enforce_balance,
)

__version__ = "0.3.0-full"
__all__ = [
    "SRMPConfig", "SRMPPartitioner", "Graph",
    "read_metis", "write_metis",
    "compute_edge_cut", "compute_normalized_cut", "compute_imbalance",
    "evaluate_partition", "validate_partition",
    "extract_structural_features", "enforce_balance",
]
