"""
SRMP Configuration (v3 - full improvements)
=============================================
All hyperparameters and settings for the SRMP pipeline.

v3 changes:
  - SignNet spectral PE processing config
  - Multi-try FM and flow-based refinement config
  - Expansion*2 edge rating config
  - F-cycle and evolutionary framework config
  - POMO-style multi-start sampling config
  - Cosine entropy schedule parameters
  - Reward normalization settings
  - GATv2 attention option
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class CoarseningConfig:
    """Configuration for the coarsening phase."""
    method: str = "expansion2"         # "hem", "shem", "label_propagation", "expansion2"
    min_vertices: int = 100            # Stop coarsening when |V| <= this
    max_levels: int = 30               # Maximum coarsening levels
    lp_iterations: int = 10            # Label propagation iterations
    contraction_ratio: float = 0.5     # Target ratio |V_{i+1}|/|V_i| per level
    use_expansion2: bool = True        # Use expansion*2 edge rating (KaHIP)
    use_gpa_matching: bool = True      # Use Global Path Algorithm matching
    community_aware: bool = False      # Restrict contractions within communities
    max_cluster_weight_factor: float = 1.5  # Factor for LP cluster weight limit
    # Source of communities when method=="community_aware": internal|lp|louvain_simple|louvain_nx|file|ssl
    community_source: str = "internal"
    community_file: Optional[str] = None    # path to a .comm file when community_source=="file"


@dataclass
class FMConfig:
    """Configuration for Fiduccia-Mattheyses refinement."""
    max_passes: int = 10               # Maximum FM passes per level
    max_negative_moves: int = -1       # -1 = unlimited (full rollback)
    early_stop_passes: int = 3         # Stop if no improvement for this many passes
    balance_factor: float = 1.0        # Tightness of balance constraint
    unconstrained: bool = False        # If true, allow temporary imbalance during FM

    # When True (default, classical FM), each pass rolls back to the
    # best-prefix of moves seen so far. This is a "no-harm" guard: any
    # sequence of moves with non-positive net gain is discarded entirely,
    # which can mask exploratory RL move sequences whose value comes from
    # non-monotone trajectories. Set False to keep the full pass result
    # (only useful for ablation; default behavior preserved for compat).
    no_harm_rollback: bool = True

    # Multi-try FM (KaHIP-style)
    use_multi_try: bool = True         # Enable multi-try localized FM
    multi_try_rounds: int = 10         # Number of multi-try rounds per level
    adaptive_stopping: bool = True     # Use adaptive stopping rule

    # Flow-based refinement
    use_flow_refinement: bool = True   # Enable max-flow/min-cut refinement
    flow_alpha: int = 16               # BFS corridor width for flow region
    flow_max_pairs: int = 10           # Max block pairs to refine per level
    flow_alpha_scale: bool = True      # Scale alpha by boundary activity
    flow_alpha_min: int = 4            # Minimum scaled alpha
    flow_alpha_max: int = 48           # Maximum scaled alpha

    # Label propagation pre-refinement
    use_lp_refinement: bool = True     # LP quick-pass before FM
    lp_refine_iterations: int = 3      # LP iterations for refinement


@dataclass
class RLConfig:
    """Configuration for the RL policy and training."""
    # Architecture
    hidden_dim: int = 128              # GNN hidden dimension
    num_gnn_layers: int = 3            # Number of GNN message-passing layers
    spectral_dim: int = 8              # Number of spectral coordinates to use
    use_gatv2: bool = True             # Use GATv2 attention layers
    use_edge_features: bool = True     # Include edge features in message passing
    num_attention_heads: int = 4       # Number of attention heads for GATv2
    use_signnet: bool = True           # Use SignNet for spectral PE processing
    signnet_hidden: int = 64           # SignNet MLP hidden dimension

    # Optimiser
    lr_actor: float = 3e-4             # Actor learning rate
    lr_critic: float = 1e-3            # Critic learning rate
    max_grad_norm: float = 0.5         # Gradient clipping norm
    use_cosine_lr: bool = True         # Cosine annealing for learning rate
    warmup_steps: int = 1000           # LR warmup steps

    # PPO
    gamma: float = 0.95                # Discount factor (changed from 0.99)
    gae_lambda: float = 0.98           # GAE lambda (changed from 0.95)
    clip_epsilon: float = 0.2          # PPO clipping
    ppo_epochs: int = 3                # PPO update epochs per collected batch
    entropy_coeff: float = 0.05        # Initial entropy coefficient
    entropy_coeff_final: float = 0.005 # Final entropy coefficient (cosine schedule)
    value_coeff: float = 0.5           # Value loss coefficient

    # Episode structure
    max_episode_steps: int = 300       # Max steps per refinement episode
    min_episode_steps: int = 30        # Force at least this many moves
    num_episodes: int = 1000           # Total training episodes

    # Batching
    episodes_per_batch: int = 8        # Collect this many episodes before PPO update
    batch_size: int = 64               # Minibatch size within PPO epoch
    episode_workers: int = 1           # Parallel episode collection workers

    # Rollback
    use_rollback: bool = True          # Track best partition in episode

    # Reward
    use_spectral_features: bool = True # Include spectral positional encodings
    use_local_reward: bool = True      # Use local reward shaping
    balance_penalty: float = 0.5       # Penalty coefficient for balance violations
    normalize_reward: bool = True      # Normalize reward by graph volume
    use_potential_shaping: bool = True  # Potential-based reward shaping

    # POMO-style multi-start
    num_pomo_starts: int = 12          # Number of parallel rollouts per graph
    use_pomo_baseline: bool = True     # Use shared mean baseline

    # Greedy rollout baseline
    use_greedy_baseline: bool = True   # Use greedy rollout as baseline
    greedy_baseline_update: int = 50   # Episodes between greedy baseline updates

    # GPU acceleration
    use_amp: bool = True               # Mixed precision training on CUDA

    # Performance optimizations (ablation toggles)
    use_batched_ppo: bool = True       # Disjoint-graph batched PPO updates (GPU optimization)
    use_process_workers: bool = True   # ProcessPoolExecutor for episode collection (CPU parallelism)
    ppo_mini_batch_graphs: int = 16   # Max graphs per disjoint batch in batched PPO


@dataclass
class VCycleConfig:
    """Configuration for V-cycle and F-cycle iteration."""
    num_cycles: int = 5                # Number of V/F-cycles
    early_stop: bool = False           # Don't early stop (FM is non-monotone)
    use_fcycle: bool = True            # Use F-cycles (2 recursive calls per level)
    fcycle_depth: int = 2              # Number of recursive calls per F-cycle level


@dataclass
class EvolutionaryConfig:
    """Configuration for KaFFPaE-style evolutionary framework."""
    enabled: bool = True               # Enable evolutionary combine
    population_size: int = 16          # Population size
    num_generations: int = 30          # Number of evolutionary generations
    combine_method: str = "kaffpae"    # "kaffpae" or "random_cross"
    tournament_size: int = 2           # Tournament selection size
    mutation_rate: float = 0.1         # Probability of random perturbation


@dataclass
class SRMPConfig:
    """Master configuration for the SRMP pipeline."""
    k: int = 2                         # Number of partitions
    epsilon: float = 0.03              # Imbalance tolerance (3%)
    num_runs: int = 20                 # Number of independent runs
    seed: int = 42                     # Random seed
    use_rl: bool = True                # Enable RL component
    spectral_threshold: float = 2.0    # tau_spec: eigengap ratio threshold
    best_of_k: int = 12               # Best-of-K sampling at each pipeline stage
    coarsening: CoarseningConfig = field(default_factory=CoarseningConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    vcycle: VCycleConfig = field(default_factory=VCycleConfig)
    evolutionary: EvolutionaryConfig = field(default_factory=EvolutionaryConfig)
    device: str = "cpu"                # "cpu" or "cuda"
    parallel_runs: bool = False        # Parallelize independent multi-run restarts
    num_workers: int = 1               # Worker count for parallel benchmark runs
    verbose: bool = True
