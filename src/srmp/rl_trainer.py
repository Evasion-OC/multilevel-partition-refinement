"""
PPO Trainer for Partition Refinement (v4 - batched PPO + process workers)
==========================================================================
Trains the GNN-PPO policy with:
  - Reward normalization by graph volume (20-40% variance reduction)
  - Potential-based reward shaping (Ng, Harada & Russell, 1999)
  - POMO-style multi-start sampling (Kwon et al., NeurIPS 2020)
  - Cosine entropy schedule (0.05 -> 0.005)
  - Greedy rollout baseline (Kool et al., ICLR 2019)
  - Advantage normalization across full batch

v4 changes:
  - Disjoint-graph batched PPO updates for GPU utilization (use_batched_ppo)
  - ProcessPoolExecutor for true parallel episode collection (use_process_workers)
  - Eliminated redundant GNN forward passes in collect_episode and batched_update
  - Both optimizations independently toggleable for ablation

References:
  - Schulman et al. (2017): PPO
  - Ng, Harada & Russell (1999): Potential-based reward shaping
  - Kwon et al. (NeurIPS, 2020): POMO
  - Kool et al. (ICLR, 2019): Attention model with greedy rollout baseline
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import time
import io
import multiprocessing as mp
from dataclasses import fields as dataclass_fields
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
from collections import namedtuple
from .graph import Graph
from .config import RLConfig
from .rl_policy import (
    ActorCritic, build_features, build_edge_features,
    graph_to_torch, compute_feature_dim, compute_edge_feature_dim,
)
from .evaluation import compute_edge_cut, compute_normalized_cut_fast, compute_imbalance

Transition = namedtuple('Transition', [
    'features', 'adj_indices', 'adj_weights', 'deg_inv',
    'edge_features', 'boundary_mask',
    # valid_vertex_mask = boundary cap balance-feasible at action time.
    # This is the ACTUAL support over which the rollout sampled the vertex.
    # It must be used at PPO update time so that old and new log-probs are
    # over the same action distribution. boundary_mask is kept for GNN-side
    # masking (it's the broader set the GNN attends over).
    'valid_vertex_mask',
    'action_vertex', 'action_block',
    'candidate_blocks', 'log_prob', 'reward', 'value', 'done',
    # pomo_group_id ties together POMO trajectories that share an instance.
    # Used by the per-instance baseline subtraction; -1 means "no group"
    # (no POMO baseline to subtract for this transition).
    'pomo_group_id',
])

TransitionNumpy = namedtuple('TransitionNumpy', [
    'features', 'adj_indices', 'adj_weights', 'deg_inv',
    'edge_features', 'boundary_mask', 'valid_vertex_mask',
    'action_vertex', 'action_block',
    'candidate_blocks', 'log_prob', 'reward', 'value', 'done',
    'pomo_group_id',
])


# =========================================================================
# Serialization utilities for cross-process transfer
# =========================================================================

def _transition_to_numpy(t: Transition) -> TransitionNumpy:
    """Convert a Transition with torch tensors to numpy for pickling."""
    return TransitionNumpy(
        features=t.features.cpu().numpy() if isinstance(t.features, torch.Tensor) else t.features,
        adj_indices=t.adj_indices.cpu().numpy() if isinstance(t.adj_indices, torch.Tensor) else t.adj_indices,
        adj_weights=t.adj_weights.cpu().numpy() if isinstance(t.adj_weights, torch.Tensor) else t.adj_weights,
        deg_inv=t.deg_inv.cpu().numpy() if isinstance(t.deg_inv, torch.Tensor) else t.deg_inv,
        edge_features=(t.edge_features.cpu().numpy() if isinstance(t.edge_features, torch.Tensor) else t.edge_features) if t.edge_features is not None else None,
        boundary_mask=t.boundary_mask.cpu().numpy() if isinstance(t.boundary_mask, torch.Tensor) else t.boundary_mask,
        valid_vertex_mask=t.valid_vertex_mask.cpu().numpy() if isinstance(t.valid_vertex_mask, torch.Tensor) else t.valid_vertex_mask,
        action_vertex=t.action_vertex,
        action_block=t.action_block,
        candidate_blocks=list(t.candidate_blocks),
        log_prob=t.log_prob,
        reward=t.reward,
        value=t.value,
        done=t.done,
        pomo_group_id=int(t.pomo_group_id),
    )


def _transition_from_numpy(tn: TransitionNumpy, device: str) -> Transition:
    """Convert a TransitionNumpy back to a Transition with torch tensors."""
    return Transition(
        features=torch.tensor(tn.features, dtype=torch.float32, device=device),
        adj_indices=torch.tensor(tn.adj_indices, dtype=torch.long, device=device),
        adj_weights=torch.tensor(tn.adj_weights, dtype=torch.float32, device=device),
        deg_inv=torch.tensor(tn.deg_inv, dtype=torch.float32, device=device),
        edge_features=torch.tensor(tn.edge_features, dtype=torch.float32, device=device) if tn.edge_features is not None else None,
        boundary_mask=torch.tensor(tn.boundary_mask, device=device),
        valid_vertex_mask=torch.tensor(tn.valid_vertex_mask, device=device),
        action_vertex=tn.action_vertex,
        action_block=tn.action_block,
        candidate_blocks=list(tn.candidate_blocks),
        log_prob=tn.log_prob,
        reward=tn.reward,
        value=tn.value,
        done=tn.done,
        pomo_group_id=int(tn.pomo_group_id),
    )


def _serialize_graph(G: Graph) -> tuple:
    """Serialize a Graph to picklable CSR components."""
    csr = G.adj
    return (csr.data.copy(), csr.indices.copy(), csr.indptr.copy(),
            csr.shape, G.vertex_weights.copy())


def _deserialize_graph(data: tuple) -> Graph:
    """Reconstruct a Graph from serialized CSR components."""
    from scipy import sparse
    adj_data, adj_indices, adj_indptr, adj_shape, vw = data
    adj = sparse.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)
    return Graph(adj, vw)


def _config_to_dict(config: RLConfig) -> dict:
    """Serialize RLConfig to a picklable dict."""
    return {f.name: getattr(config, f.name) for f in dataclass_fields(config)}


# =========================================================================
# Lightweight episode collector for child processes
# =========================================================================

class _EpisodeCollector:
    """
    Lightweight episode collector for use in child processes.
    Holds only model + config (no optimizer, no gradients, no AMP scaler).
    Always runs on CPU for cross-process compatibility.
    """

    def __init__(self, model: ActorCritic, config: RLConfig, k: int, device: str = "cpu"):
        self.model = model
        self.config = config
        self.k = k
        self.device = device

    def collect_episode(
        self,
        env: 'PartitionEnv',
    ) -> Tuple[List[Transition], float, np.ndarray]:
        """Collect one complete episode -- single GNN call per step."""
        state = env.reset()
        transitions = []
        total_reward = 0.0

        for step in range(self.config.max_episode_steps):
            valid_actions = env.get_valid_actions()
            if len(valid_actions) == 0:
                if step < self.config.min_episode_steps:
                    block_weights = env._get_block_weights()
                    boundary = env.G.boundary_vertices_fast(env.partition)
                    for v in boundary:
                        nbrs, _ = env.G.neighbors(v)
                        for nb in nbrs:
                            b = env.partition[nb]
                            if b != env.partition[v]:
                                new_weight = block_weights[b] + env.G.vertex_weights[v]
                                if new_weight <= env.soft_block_weight:
                                    valid_actions.append((int(v), int(b)))
                        if valid_actions:
                            break
                if len(valid_actions) == 0:
                    break

            with torch.no_grad():
                # Single GNN call -- compute node embeddings once
                node_emb_all = self.model.gnn(
                    state['features'],
                    state['adj_indices'],
                    state['adj_weights'],
                    state['deg_inv'],
                    state.get('edge_features'),
                )
                # Vertex logits from cached embeddings
                vertex_scores = self.model.actor_vertex(node_emb_all).squeeze(-1)
                mask_value = torch.finfo(vertex_scores.dtype).min
                vertex_logits = vertex_scores.masked_fill(~state['boundary_mask'], mask_value)
                # Value from mean-pooled embeddings
                value = self.model.critic(node_emb_all.mean(dim=0, keepdim=True)).squeeze()

            valid_vertices = sorted(set(v for v, b in valid_actions))
            if len(valid_vertices) == 0:
                break

            valid_vtx_t = torch.tensor(valid_vertices, dtype=torch.long, device=self.device)
            # Build the valid_vertex_mask from valid_vertices. This is the
            # ACTUAL action support (boundary cap balance-feasible) the policy
            # sampled from. PPO update uses this mask, NOT boundary_mask.
            vv_mask = torch.zeros_like(state['boundary_mask'])
            vv_mask[valid_vtx_t] = True
            vertex_scores_valid = vertex_logits[valid_vtx_t].float()
            vertex_dist = torch.distributions.Categorical(logits=vertex_scores_valid)
            vertex_sample = vertex_dist.sample()
            selected_vertex = valid_vertices[int(vertex_sample)]

            candidate_blocks = sorted(set(
                b for v, b in valid_actions if v == selected_vertex
            ))
            if len(candidate_blocks) == 0:
                break

            candidates_tensor = torch.tensor(
                candidate_blocks, dtype=torch.long, device=self.device
            )

            # Reuse cached node embedding -- no second GNN call
            with torch.no_grad():
                node_emb = node_emb_all[selected_vertex]

            block_logits = self.model.get_block_logits(node_emb, candidates_tensor).float()
            block_dist = torch.distributions.Categorical(logits=block_logits)
            block_sample = block_dist.sample()
            selected_block = candidate_blocks[int(block_sample)]

            log_prob = vertex_dist.log_prob(vertex_sample) + block_dist.log_prob(block_sample)

            next_state, reward, done = env.step(selected_vertex, selected_block)
            total_reward += reward

            transitions.append(Transition(
                features=state['features'],
                adj_indices=state['adj_indices'],
                adj_weights=state['adj_weights'],
                deg_inv=state['deg_inv'],
                edge_features=state.get('edge_features'),
                boundary_mask=state['boundary_mask'],
                valid_vertex_mask=vv_mask,
                action_vertex=selected_vertex,
                action_block=selected_block,
                candidate_blocks=candidate_blocks,
                log_prob=log_prob.item(),
                reward=reward,
                value=value.item(),
                done=done,
                pomo_group_id=-1,  # caller will tag if part of a POMO group
            ))

            state = next_state
            if done:
                break

        return transitions, total_reward, env.get_best_partition()

    def compute_gae(
        self,
        transitions: List[Transition],
        last_value: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns.

        If ``last_value`` is None and the final transition is not done,
        bootstrap with ``transitions[-1].value`` (truncation, not termination).
        Passing ``last_value=0.0`` explicitly forces a terminal bootstrap.
        """
        T = len(transitions)
        advantages = np.zeros(T, dtype=np.float64)
        returns = np.zeros(T, dtype=np.float64)
        if T == 0:
            return advantages, returns

        if last_value is None:
            last_value = 0.0 if transitions[-1].done else float(transitions[-1].value)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(T)):
            if transitions[t].done:
                next_value = 0.0
                gae = 0.0
            delta = (transitions[t].reward +
                     self.config.gamma * next_value -
                     transitions[t].value)
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + transitions[t].value
            next_value = transitions[t].value

        return advantages, returns

    def collect_batch_item(
        self,
        episode_idx: int,
        G: Graph,
        init_part: np.ndarray,
        spec_coords: Optional[np.ndarray],
        k: int,
        epsilon: float,
        entropy_coeff: float,
    ) -> Tuple[List[Transition], List[np.ndarray], List[np.ndarray], Dict]:
        """Run one episode with POMO multi-start, return transitions + GAE."""
        env = PartitionEnv(
            G, init_part, k, epsilon, spec_coords,
            use_local_reward=self.config.use_local_reward,
            balance_penalty=self.config.balance_penalty,
            max_steps=self.config.max_episode_steps,
            min_steps=self.config.min_episode_steps,
            normalize_reward=self.config.normalize_reward,
            use_potential_shaping=self.config.use_potential_shaping,
            use_edge_features=self.config.use_edge_features,
            gamma=self.config.gamma,
            device=self.device,
        )

        num_starts = self.config.num_pomo_starts if self.config.use_pomo_baseline else 1
        pomo_rewards = []
        pomo_transitions_list = []
        best_pomo_part = None
        best_pomo_cut = float('inf')

        for pomo_idx in range(num_starts):
            if pomo_idx > 0:
                perturbed = init_part.copy()
                boundary = G.boundary_vertices_fast(perturbed)
                if len(boundary) > 0:
                    rng = np.random.RandomState(episode_idx * 100 + pomo_idx)
                    n_perturb = max(1, len(boundary) // 10)
                    to_move = rng.choice(
                        boundary, size=min(n_perturb, len(boundary)), replace=False
                    )
                    for v in to_move:
                        nbrs, _ = G.neighbors(v)
                        adj_blocks = set()
                        for nb in nbrs:
                            if perturbed[nb] != perturbed[v]:
                                adj_blocks.add(perturbed[nb])
                        if adj_blocks:
                            perturbed[v] = rng.choice(list(adj_blocks))
                env_pomo = PartitionEnv(
                    G, perturbed, k, epsilon, spec_coords,
                    use_local_reward=self.config.use_local_reward,
                    balance_penalty=self.config.balance_penalty,
                    max_steps=self.config.max_episode_steps,
                    min_steps=self.config.min_episode_steps,
                    normalize_reward=self.config.normalize_reward,
                    use_potential_shaping=self.config.use_potential_shaping,
                    use_edge_features=self.config.use_edge_features,
                    gamma=self.config.gamma,
                    device=self.device,
                )
            else:
                env_pomo = env

            transitions, total_reward, best_part = self.collect_episode(env_pomo)
            pomo_rewards.append(total_reward)
            pomo_transitions_list.append(transitions)

            part_cut = compute_edge_cut(G, best_part)
            if part_cut < best_pomo_cut:
                best_pomo_cut = part_cut
                best_pomo_part = best_part

        episode_transitions: List[Transition] = []
        episode_advantages: List[np.ndarray] = []
        episode_returns: List[np.ndarray] = []

        # POMO-baseline tagging: every trajectory from the same instance shares
        # one pomo_group_id so the PPO update can subtract a per-instance mean.
        # Use episode_idx as the group key (each call to collect_batch_item
        # corresponds to one instance).
        for transitions in pomo_transitions_list:
            if len(transitions) == 0:
                continue
            advantages, returns = self.compute_gae(transitions)
            tagged = [t._replace(pomo_group_id=int(episode_idx)) for t in transitions]
            episode_transitions.extend(tagged)
            episode_advantages.append(advantages)
            episode_returns.append(returns)

        init_cut = compute_edge_cut(G, init_part)
        best_cut = best_pomo_cut if best_pomo_part is not None else init_cut

        max_weight = (1.0 + epsilon) * np.ceil(G.total_weight / k)
        if best_pomo_part is not None:
            bw = np.bincount(best_pomo_part, weights=G.vertex_weights, minlength=k)
            balance_ok = bool(bw.max() <= max_weight + 1e-9)
        else:
            balance_ok = False

        stats = {
            'episode': episode_idx,
            'episode_reward': pomo_rewards[0] if pomo_rewards else 0.0,
            'episode_length': len(pomo_transitions_list[0]) if pomo_transitions_list else 0,
            'initial_cut': init_cut,
            'final_cut': best_cut,
            'improvement_pct': (
                (init_cut - best_cut) / init_cut * 100
                if init_cut > 0 else 0
            ),
            'balance_ok': balance_ok,
            'entropy_coeff': entropy_coeff,
            'num_pomo_starts': num_starts,
        }
        return episode_transitions, episode_advantages, episode_returns, stats


# =========================================================================
# Module-level worker function for ProcessPoolExecutor
# =========================================================================

def _collect_episode_worker(args_tuple: tuple) -> tuple:
    """
    Top-level picklable function for ProcessPoolExecutor.
    Reconstructs model on CPU, runs episode collection, returns numpy transitions.
    """
    (episode_idx, graphs_data, initial_partitions, spectral_coords_list,
     k, epsilon, entropy_coeff, config_dict, model_state_bytes,
     feature_dim, edge_dim) = args_tuple

    # Reconstruct config
    config = RLConfig(**config_dict)

    # Reconstruct model on CPU
    model = ActorCritic(config, k, feature_dim, edge_dim)
    buffer = io.BytesIO(model_state_bytes)
    state_dict = torch.load(buffer, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Reconstruct graph
    idx = episode_idx % len(graphs_data)
    G = _deserialize_graph(graphs_data[idx])
    init_part = initial_partitions[idx]
    spec_coords = spectral_coords_list[idx] if spectral_coords_list else None

    # Collect using lightweight collector on CPU
    collector = _EpisodeCollector(model, config, k, device="cpu")
    ep_transitions, ep_advantages, ep_returns, stats = collector.collect_batch_item(
        episode_idx, G, init_part, spec_coords, k, epsilon, entropy_coeff
    )

    # Convert transitions to numpy for cross-process transfer
    numpy_transitions = [_transition_to_numpy(t) for t in ep_transitions]

    return numpy_transitions, ep_advantages, ep_returns, stats


# =========================================================================
# Partition Environment
# =========================================================================

class PartitionEnv:
    """
    Environment for partition refinement (v3).

    v3 changes:
      - Reward normalization by graph volume
      - Potential-based reward shaping
      - Edge feature computation for GATv2
    """

    def __init__(
        self,
        G: Graph,
        initial_partition: np.ndarray,
        k: int,
        epsilon: float = 0.03,
        spectral_coords: Optional[np.ndarray] = None,
        use_local_reward: bool = True,
        balance_penalty: float = 0.5,
        max_steps: int = 300,
        min_steps: int = 30,
        normalize_reward: bool = True,
        use_potential_shaping: bool = True,
        use_edge_features: bool = True,
        gamma: float = 0.95,
        device: str = "cpu",
    ):
        self.G = G
        self.k = k
        self.epsilon = epsilon
        self.spectral_coords = spectral_coords
        self.use_local_reward = use_local_reward
        self.balance_penalty = balance_penalty
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.normalize_reward = normalize_reward
        self.use_potential_shaping = use_potential_shaping
        self.use_edge_features = use_edge_features
        self.gamma = gamma
        self.device = device
        self.initial_partition = initial_partition.copy()

        # Precompute graph tensors
        self.adj_indices, self.adj_weights, self.deg_inv = graph_to_torch(G, device)

        self.max_block_weight = (1.0 + epsilon) * np.ceil(G.total_weight / k)
        self.soft_block_weight = (1.0 + 3.0 * epsilon) * np.ceil(G.total_weight / k)

        # Graph volume for reward normalization
        self.graph_volume = float(G.degrees.sum())
        if self.graph_volume < 1e-12:
            self.graph_volume = 1.0

        self.reset()

    def reset(self) -> Dict:
        self.partition = self.initial_partition.copy()
        self.step_count = 0
        self.current_ncut = compute_normalized_cut_fast(self.G, self.partition)
        self.current_cut = compute_edge_cut(self.G, self.partition)

        # Rollback tracking
        self.best_partition = self.partition.copy()
        self.best_cut = self.current_cut
        self.best_is_balanced = self._check_balance(self.partition)

        return self._get_state()

    def _check_balance(self, partition: np.ndarray) -> bool:
        block_weights = np.bincount(partition, weights=self.G.vertex_weights, minlength=self.k)
        return bool(block_weights.max() <= self.max_block_weight + 1e-9)

    def _get_block_weights(self) -> np.ndarray:
        return np.bincount(self.partition, weights=self.G.vertex_weights, minlength=self.k).astype(np.float64)

    def _get_state(self) -> Dict:
        features = build_features(
            self.G, self.partition, self.k, self.epsilon, self.spectral_coords
        )
        boundary_mask = np.zeros(self.G.n, dtype=bool)
        boundary_verts = self.G.boundary_vertices_fast(self.partition)
        boundary_mask[boundary_verts] = True

        state = {
            'features': torch.tensor(features, dtype=torch.float32, device=self.device),
            'boundary_mask': torch.tensor(boundary_mask, device=self.device),
            'adj_indices': self.adj_indices,
            'adj_weights': self.adj_weights,
            'deg_inv': self.deg_inv,
        }

        # Edge features for GATv2
        if self.use_edge_features:
            edge_feat = build_edge_features(self.G, self.partition)
            state['edge_features'] = torch.tensor(
                edge_feat, dtype=torch.float32, device=self.device
            )
        else:
            state['edge_features'] = None

        return state

    def get_valid_actions(self) -> List[Tuple[int, int]]:
        block_weights = self._get_block_weights()
        limit = self.soft_block_weight if self.step_count < self.min_steps else self.max_block_weight

        # Vectorized: find all cross edges
        rows, cols = self.G.adj.nonzero()
        cross = self.partition[rows] != self.partition[cols]
        cross_verts = rows[cross]
        cross_targets = self.partition[cols[cross]]

        if len(cross_verts) == 0:
            return []

        # Get unique (vertex, target_block) pairs
        pairs = np.stack([cross_verts, cross_targets], axis=1)
        unique_pairs = np.unique(pairs, axis=0)

        # Filter by balance constraint
        actions = []
        for i in range(unique_pairs.shape[0]):
            v, target = int(unique_pairs[i, 0]), int(unique_pairs[i, 1])
            if block_weights[target] + self.G.vertex_weights[v] <= limit:
                actions.append((v, target))

        return actions

    def step(self, vertex: int, target_block: int) -> Tuple[Dict, float, bool]:
        """Execute action with improved reward computation."""
        self.step_count += 1
        old_block = self.partition[vertex]
        old_ncut = self.current_ncut
        old_block_weights = self._get_block_weights()

        # Execute move
        self.partition[vertex] = target_block

        # Compute new state
        new_ncut = compute_normalized_cut_fast(self.G, self.partition)
        new_cut = compute_edge_cut(self.G, self.partition)
        new_block_weights = self._get_block_weights()

        # ---- Reward Component 1: NCut improvement ----
        ncut_reward = old_ncut - new_ncut

        # ---- Reward Component 2: Potential-based shaping ----
        potential_reward = 0.0
        if self.use_potential_shaping:
            phi_old = -old_ncut
            phi_new = -new_ncut
            potential_reward = self.gamma * phi_new - phi_old

        # ---- Reward Component 3: Local reward ----
        local_reward = 0.0
        if self.use_local_reward:
            nbrs, weights = self.G.neighbors(vertex)
            ext_new = sum(
                w for nb, w in zip(nbrs, weights)
                if self.partition[nb] != target_block
            )
            self.partition[vertex] = old_block
            ext_old = sum(
                w for nb, w in zip(nbrs, weights)
                if self.partition[nb] != old_block
            )
            self.partition[vertex] = target_block
            deg_v = max(self.G.degrees[vertex], 1.0)
            local_reward = (ext_old - ext_new) / deg_v

        # ---- Reward Component 4: Balance penalty ----
        bal_penalty = 0.0
        old_max_overweight = max(0.0, old_block_weights.max() - self.max_block_weight)
        new_max_overweight = max(0.0, new_block_weights.max() - self.max_block_weight)
        if new_max_overweight > old_max_overweight:
            bal_penalty = self.balance_penalty * (new_max_overweight - old_max_overweight) / self.max_block_weight
        elif new_max_overweight < old_max_overweight:
            bal_penalty = -0.1 * self.balance_penalty * (old_max_overweight - new_max_overweight) / self.max_block_weight

        # ---- Combine and normalize ----
        raw_reward = 0.4 * ncut_reward + 0.2 * potential_reward + 0.3 * local_reward - bal_penalty

        if self.normalize_reward:
            reward = raw_reward / (self.graph_volume / self.G.n)
        else:
            reward = raw_reward

        self.current_ncut = new_ncut
        self.current_cut = new_cut

        # Rollback tracking
        is_balanced = self._check_balance(self.partition)
        if is_balanced and new_cut < self.best_cut:
            self.best_cut = new_cut
            self.best_partition = self.partition.copy()
            self.best_is_balanced = True
        elif not self.best_is_balanced and new_cut < self.best_cut:
            self.best_cut = new_cut
            self.best_partition = self.partition.copy()

        # Termination check
        if self.step_count >= self.max_steps:
            done = True
        elif self.step_count >= self.min_steps:
            done = True
            block_weights = self._get_block_weights()
            boundary = self.G.boundary_vertices_fast(self.partition)
            for v in boundary:
                nbrs, _ = self.G.neighbors(v)
                for nb in nbrs:
                    b = self.partition[nb]
                    if b != self.partition[v]:
                        if block_weights[b] + self.G.vertex_weights[v] <= self.max_block_weight:
                            done = False
                            break
                if not done:
                    break
        else:
            done = False

        return self._get_state(), reward, done

    def get_best_partition(self) -> np.ndarray:
        return self.best_partition.copy()


class PPOTrainer:
    """
    PPO trainer (v4) with POMO multi-start, cosine entropy schedule,
    disjoint-graph batched PPO updates, and ProcessPoolExecutor episode collection.
    """

    def __init__(
        self,
        config: RLConfig,
        k: int,
        device: str = "cpu",
    ):
        self.config = config
        self.k = k
        self.device = device
        self.use_amp = bool(getattr(config, "use_amp", True)) and device == "cuda" and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.feature_dim = compute_feature_dim(
            k, config.spectral_dim, config.use_spectral_features
        )
        self.edge_dim = compute_edge_feature_dim(config.use_edge_features)

        # Create actor-critic
        self.model = ActorCritic(config, k, self.feature_dim, self.edge_dim).to(device)
        self.optimizer = optim.Adam([
            {'params': self.model.gnn.parameters(), 'lr': config.lr_actor, 'name': 'gnn'},
            {'params': self.model.actor_vertex.parameters(), 'lr': config.lr_actor, 'name': 'actor_vertex'},
            {'params': self.model.actor_block.parameters(), 'lr': config.lr_actor, 'name': 'actor_block'},
            {'params': self.model.block_embedding.parameters(), 'lr': config.lr_actor, 'name': 'block_embedding'},
            {'params': self.model.critic.parameters(), 'lr': config.lr_critic, 'name': 'critic'},
        ])

        # Greedy rollout baseline model (frozen copy)
        self.greedy_model = None
        if config.use_greedy_baseline:
            self._update_greedy_baseline()

        self.training_stats: List[Dict] = []
        self.global_step = 0

    def _autocast_context(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _update_greedy_baseline(self):
        """Update greedy rollout baseline with current policy."""
        import copy
        self.greedy_model = copy.deepcopy(self.model)
        self.greedy_model.eval()
        for p in self.greedy_model.parameters():
            p.requires_grad = False

    def _get_entropy_coeff(self, episode: int, total_episodes: int) -> float:
        """Cosine annealing entropy coefficient: 0.05 -> 0.005."""
        progress = min(episode / max(total_episodes, 1), 1.0)
        start = self.config.entropy_coeff
        end = self.config.entropy_coeff_final
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))

    def _get_learning_rate(self, step: int, total_steps: int) -> float:
        """Cosine annealing with warmup."""
        if step < self.config.warmup_steps:
            return self.config.lr_actor * step / max(self.config.warmup_steps, 1)
        progress = (step - self.config.warmup_steps) / max(total_steps - self.config.warmup_steps, 1)
        return self.config.lr_actor * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _serialize_model_state(self) -> bytes:
        """Serialize model state_dict to bytes for cross-process transfer."""
        # Move to CPU for serialization to avoid CUDA IPC issues
        cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        buffer = io.BytesIO()
        torch.save(cpu_state, buffer)
        return buffer.getvalue()

    def collect_episode(
        self,
        env: PartitionEnv,
    ) -> Tuple[List[Transition], float, np.ndarray]:
        """Collect one complete episode -- single GNN call per step."""
        state = env.reset()
        transitions = []
        total_reward = 0.0

        for step in range(self.config.max_episode_steps):
            valid_actions = env.get_valid_actions()
            if len(valid_actions) == 0:
                if step < self.config.min_episode_steps:
                    block_weights = env._get_block_weights()
                    boundary = env.G.boundary_vertices_fast(env.partition)
                    for v in boundary:
                        nbrs, _ = env.G.neighbors(v)
                        for nb in nbrs:
                            b = env.partition[nb]
                            if b != env.partition[v]:
                                new_weight = block_weights[b] + env.G.vertex_weights[v]
                                if new_weight <= env.soft_block_weight:
                                    valid_actions.append((int(v), int(b)))
                        if valid_actions:
                            break
                if len(valid_actions) == 0:
                    break

            with torch.no_grad():
                with self._autocast_context():
                    # Single GNN call -- compute node embeddings once
                    node_emb_all = self.model.gnn(
                        state['features'],
                        state['adj_indices'],
                        state['adj_weights'],
                        state['deg_inv'],
                        state.get('edge_features'),
                    )
                    # Vertex logits from cached embeddings
                    vertex_scores = self.model.actor_vertex(node_emb_all).squeeze(-1)
                    mask_value = torch.finfo(vertex_scores.dtype).min
                    vertex_logits = vertex_scores.masked_fill(~state['boundary_mask'], mask_value)
                    # Value from mean-pooled embeddings
                    value = self.model.critic(node_emb_all.mean(dim=0, keepdim=True)).squeeze()

            valid_vertices = sorted(set(v for v, b in valid_actions))
            if len(valid_vertices) == 0:
                break

            valid_vtx_t = torch.tensor(valid_vertices, dtype=torch.long, device=self.device)
            # Build the valid_vertex_mask. PPO update uses this; see Transition docstring.
            vv_mask = torch.zeros_like(state['boundary_mask'])
            vv_mask[valid_vtx_t] = True
            vertex_scores_valid = vertex_logits[valid_vtx_t].float()
            vertex_dist = torch.distributions.Categorical(logits=vertex_scores_valid)
            vertex_sample = vertex_dist.sample()
            selected_vertex = valid_vertices[int(vertex_sample)]

            candidate_blocks = sorted(set(
                b for v, b in valid_actions if v == selected_vertex
            ))
            if len(candidate_blocks) == 0:
                break

            candidates_tensor = torch.tensor(
                candidate_blocks, dtype=torch.long, device=self.device
            )

            # Reuse cached node embedding -- no second GNN call
            with torch.no_grad():
                with self._autocast_context():
                    node_emb = node_emb_all[selected_vertex]

            block_logits = self.model.get_block_logits(node_emb, candidates_tensor).float()
            block_dist = torch.distributions.Categorical(logits=block_logits)
            block_sample = block_dist.sample()
            selected_block = candidate_blocks[int(block_sample)]

            log_prob = vertex_dist.log_prob(vertex_sample) + block_dist.log_prob(block_sample)

            next_state, reward, done = env.step(selected_vertex, selected_block)
            total_reward += reward

            transitions.append(Transition(
                features=state['features'],
                adj_indices=state['adj_indices'],
                adj_weights=state['adj_weights'],
                deg_inv=state['deg_inv'],
                edge_features=state.get('edge_features'),
                boundary_mask=state['boundary_mask'],
                valid_vertex_mask=vv_mask,
                action_vertex=selected_vertex,
                action_block=selected_block,
                candidate_blocks=candidate_blocks,
                log_prob=log_prob.item(),
                reward=reward,
                value=value.item(),
                done=done,
                pomo_group_id=-1,
            ))

            state = next_state
            if done:
                break

        return transitions, total_reward, env.get_best_partition()

    def compute_gae(
        self,
        transitions: List[Transition],
        last_value: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns.

        If ``last_value`` is None and the final transition is not done,
        bootstrap with ``transitions[-1].value`` (truncation, not termination).
        Passing ``last_value=0.0`` explicitly forces a terminal bootstrap.
        """
        T = len(transitions)
        advantages = np.zeros(T, dtype=np.float64)
        returns = np.zeros(T, dtype=np.float64)
        if T == 0:
            return advantages, returns

        if last_value is None:
            last_value = 0.0 if transitions[-1].done else float(transitions[-1].value)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(T)):
            if transitions[t].done:
                next_value = 0.0
                gae = 0.0
            delta = (transitions[t].reward +
                     self.config.gamma * next_value -
                     transitions[t].value)
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + transitions[t].value
            next_value = transitions[t].value

        return advantages, returns

    def _pomo_baseline_subtract(
        self,
        all_transitions: List[Transition],
        all_advantages: np.ndarray,
    ) -> np.ndarray:
        """POMO per-instance baseline subtraction (Kwon et al., ICLR 2020).

        For each pomo_group_id (one instance / one graph, multiple trajectories),
        subtract the per-instance mean advantage from every transition in that
        group. Transitions with pomo_group_id == -1 (no group) are left alone.

        This is the headline POMO variance-reduction trick: graphs that happen
        to have systematically larger reward magnitudes don't dominate the
        gradient just because of scale.

        Returns a NEW advantages array (does not mutate the input).
        """
        if not getattr(self.config, "use_pomo_baseline", False):
            return all_advantages
        if len(all_transitions) == 0:
            return all_advantages

        adv = np.asarray(all_advantages, dtype=np.float64).copy()
        # Index transitions by pomo_group_id
        groups: Dict[int, List[int]] = {}
        for i, tr in enumerate(all_transitions):
            gid = int(getattr(tr, "pomo_group_id", -1))
            if gid < 0:
                continue
            groups.setdefault(gid, []).append(i)

        for gid, indices in groups.items():
            if len(indices) < 2:
                # Single-trajectory group: subtraction would be zero anyway.
                continue
            idx_arr = np.asarray(indices, dtype=np.int64)
            baseline = adv[idx_arr].mean()
            adv[idx_arr] -= baseline
        return adv

    def batched_update(
        self,
        all_transitions: List[Transition],
        all_advantages: np.ndarray,
        all_returns: np.ndarray,
        entropy_coeff: float,
    ) -> Dict[str, float]:
        """One-at-a-time PPO update (fallback when use_batched_ppo=False).
        Fixed: redundant GNN call eliminated."""
        T = len(all_transitions)
        if T == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

        # POMO per-instance baseline (no-op if use_pomo_baseline=False or no
        # multi-trajectory groups present).
        all_advantages = self._pomo_baseline_subtract(all_transitions, all_advantages)
        # Normalize advantages across the whole batch
        adv = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(
            [t.log_prob for t in all_transitions],
            dtype=torch.float32, device=self.device
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_valid_count = 0

        for epoch in range(self.config.ppo_epochs):
            indices = np.random.permutation(T)
            batch_size = min(self.config.batch_size, T)

            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch_idx = indices[start:end]

                losses = []
                valid_count = 0

                for t_idx in batch_idx:
                    tr = all_transitions[t_idx]

                    with self._autocast_context():
                        # Single GNN call (fixed: was two calls before)
                        node_emb_all = self.model.gnn(
                            tr.features, tr.adj_indices, tr.adj_weights,
                            tr.deg_inv, tr.edge_features,
                        )

                        # Vertex logits from cached embeddings.
                        # Mask using valid_vertex_mask (boundary cap balance-feasible
                        # at action time) so the support matches what the rollout
                        # sampled from. boundary_mask is broader and would let
                        # balance-infeasible vertices contribute to the softmax,
                        # producing an old/new log-prob ratio that is not a valid
                        # importance ratio.
                        vertex_scores = self.model.actor_vertex(node_emb_all).squeeze(-1)
                        mask_value = torch.finfo(vertex_scores.dtype).min
                        vertex_logits = vertex_scores.masked_fill(~tr.valid_vertex_mask, mask_value)

                        # Value from mean-pooled embeddings
                        value = self.model.critic(node_emb_all.mean(dim=0, keepdim=True)).squeeze()

                        valid_idx = torch.where(tr.valid_vertex_mask)[0]
                        if len(valid_idx) == 0:
                            continue

                        vertex_match = (valid_idx == tr.action_vertex).nonzero()
                        if len(vertex_match) == 0:
                            continue
                        vertex_idx_in_valid = vertex_match[0, 0]

                        vertex_probs = torch.softmax(vertex_logits[valid_idx].float(), dim=0)
                        vertex_log_prob = torch.log(vertex_probs[vertex_idx_in_valid] + 1e-10)

                        # Block logits -- reuse cached node embedding
                        node_emb = node_emb_all[tr.action_vertex]

                        candidates_tensor = torch.tensor(
                            tr.candidate_blocks, dtype=torch.long, device=self.device
                        )
                        block_logits = self.model.get_block_logits(node_emb, candidates_tensor).float()
                        block_probs = torch.softmax(block_logits, dim=0)

                        block_idx_in_candidates = tr.candidate_blocks.index(tr.action_block)
                        block_log_prob = torch.log(block_probs[block_idx_in_candidates] + 1e-10)

                        new_log_prob = vertex_log_prob + block_log_prob

                        # PPO clipped objective
                        ratio = torch.exp(new_log_prob - old_log_probs[t_idx])
                        surr1 = ratio * adv[t_idx]
                        surr2 = torch.clamp(
                            ratio,
                            1 - self.config.clip_epsilon,
                            1 + self.config.clip_epsilon
                        ) * adv[t_idx]
                        policy_loss = -torch.min(surr1, surr2)

                        # Value loss
                        value_loss = self.config.value_coeff * (value - ret[t_idx]) ** 2

                        # Entropy with cosine-scheduled coefficient
                        vertex_entropy = -(vertex_probs * torch.log(vertex_probs + 1e-10)).sum()
                        block_entropy = -(block_probs * torch.log(block_probs + 1e-10)).sum()
                        entropy = vertex_entropy + block_entropy
                        entropy_loss = -entropy_coeff * entropy

                        loss = policy_loss + value_loss + entropy_loss

                    losses.append(loss)
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    valid_count += 1

                total_valid_count += valid_count

                if len(losses) > 0:
                    total_loss = torch.stack(losses).mean()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.use_amp:
                        self.scaler.scale(total_loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        total_loss.backward()
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        self.optimizer.step()

        divisor = max(total_valid_count, 1)
        return {
            'policy_loss': total_policy_loss / divisor,
            'value_loss': total_value_loss / divisor,
            'entropy': total_entropy / divisor,
        }

    def batched_update_gpu(
        self,
        all_transitions: List[Transition],
        all_advantages: np.ndarray,
        all_returns: np.ndarray,
        entropy_coeff: float,
    ) -> Dict[str, float]:
        """
        Batched PPO update using disjoint-graph batching for GPU efficiency.

        Instead of one GNN forward pass per transition, batches multiple
        transitions into a single disjoint mega-graph and runs one forward pass.
        Also eliminates the redundant second GNN call for node embeddings.
        """
        T = len(all_transitions)
        if T == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

        # POMO per-instance baseline (no-op if disabled or no groups present)
        all_advantages = self._pomo_baseline_subtract(all_transitions, all_advantages)
        adv = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(
            [t.log_prob for t in all_transitions],
            dtype=torch.float32, device=self.device
        )

        max_batch_graphs = max(1, self.config.ppo_mini_batch_graphs)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_valid_count = 0

        for epoch in range(self.config.ppo_epochs):
            indices = np.random.permutation(T)
            batch_size = min(self.config.batch_size, T)

            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch_idx = indices[start:end]

                all_losses = []

                # Process in sub-batches of max_batch_graphs for VRAM control
                for sub_start in range(0, len(batch_idx), max_batch_graphs):
                    sub_end = min(sub_start + max_batch_graphs, len(batch_idx))
                    sub_idx = batch_idx[sub_start:sub_end]

                    # --- Build disjoint graph batch ---
                    features_list = []
                    adj_indices_list = []
                    adj_weights_list = []
                    deg_inv_list = []
                    edge_features_list = []
                    boundary_masks = []
                    node_counts = []
                    cumulative_nodes = 0
                    valid_sub_indices = []

                    valid_vertex_masks = []
                    for t_idx in sub_idx:
                        tr = all_transitions[t_idx]
                        n_i = tr.features.size(0)

                        features_list.append(tr.features)

                        # Offset adjacency indices for disjoint batching
                        offset_adj = tr.adj_indices + cumulative_nodes
                        adj_indices_list.append(offset_adj)
                        adj_weights_list.append(tr.adj_weights)
                        deg_inv_list.append(tr.deg_inv)

                        if tr.edge_features is not None:
                            edge_features_list.append(tr.edge_features)

                        boundary_masks.append(tr.boundary_mask)
                        valid_vertex_masks.append(tr.valid_vertex_mask)
                        node_counts.append(n_i)
                        valid_sub_indices.append(t_idx)
                        cumulative_nodes += n_i

                    if len(features_list) == 0:
                        continue

                    # Concatenate into mega-graph tensors
                    mega_features = torch.cat(features_list, dim=0)
                    mega_adj_indices = torch.cat(adj_indices_list, dim=1)
                    mega_adj_weights = torch.cat(adj_weights_list, dim=0)
                    mega_deg_inv = torch.cat(deg_inv_list, dim=0)
                    mega_edge_features = (
                        torch.cat(edge_features_list, dim=0)
                        if edge_features_list else None
                    )

                    with self._autocast_context():
                        # SINGLE GNN forward pass for entire sub-batch
                        all_node_emb = self.model.gnn(
                            mega_features, mega_adj_indices, mega_adj_weights,
                            mega_deg_inv, mega_edge_features,
                        )

                        # Extract per-graph results and compute losses
                        offset = 0
                        for i, t_idx in enumerate(valid_sub_indices):
                            tr = all_transitions[t_idx]
                            n_i = node_counts[i]

                            graph_emb = all_node_emb[offset: offset + n_i]

                            # Vertex logits from cached embeddings.
                            # Mask using valid_vertex_mask (boundary cap balance-feasible
                            # at action time) so the support matches what the rollout
                            # sampled from. See _EpisodeCollector.collect_episode for
                            # the rationale on why boundary_mask is too broad.
                            vv_mask = valid_vertex_masks[i]
                            vertex_scores = self.model.actor_vertex(graph_emb).squeeze(-1)
                            mask_value = torch.finfo(vertex_scores.dtype).min
                            vertex_logits = vertex_scores.masked_fill(~vv_mask, mask_value)

                            # Value from mean-pooled embeddings
                            value = self.model.critic(
                                graph_emb.mean(dim=0, keepdim=True)
                            ).squeeze()

                            valid_idx = torch.where(vv_mask)[0]
                            if len(valid_idx) == 0:
                                offset += n_i
                                continue

                            vertex_match = (valid_idx == tr.action_vertex).nonzero()
                            if len(vertex_match) == 0:
                                offset += n_i
                                continue
                            vertex_idx_in_valid = vertex_match[0, 0]

                            vertex_probs = torch.softmax(
                                vertex_logits[valid_idx].float(), dim=0
                            )
                            vertex_log_prob = torch.log(
                                vertex_probs[vertex_idx_in_valid] + 1e-10
                            )

                            # Block logits -- using CACHED node embedding
                            node_emb = graph_emb[tr.action_vertex]
                            candidates_tensor = torch.tensor(
                                tr.candidate_blocks, dtype=torch.long,
                                device=self.device
                            )
                            block_logits = self.model.get_block_logits(
                                node_emb, candidates_tensor
                            ).float()
                            block_probs = torch.softmax(block_logits, dim=0)

                            block_idx_in_candidates = tr.candidate_blocks.index(
                                tr.action_block
                            )
                            block_log_prob = torch.log(
                                block_probs[block_idx_in_candidates] + 1e-10
                            )

                            new_log_prob = vertex_log_prob + block_log_prob

                            # PPO clipped objective
                            ratio = torch.exp(new_log_prob - old_log_probs[t_idx])
                            surr1 = ratio * adv[t_idx]
                            surr2 = torch.clamp(
                                ratio,
                                1 - self.config.clip_epsilon,
                                1 + self.config.clip_epsilon,
                            ) * adv[t_idx]
                            policy_loss = -torch.min(surr1, surr2)

                            value_loss = self.config.value_coeff * (
                                value - ret[t_idx]
                            ) ** 2

                            vertex_entropy = -(
                                vertex_probs * torch.log(vertex_probs + 1e-10)
                            ).sum()
                            block_entropy = -(
                                block_probs * torch.log(block_probs + 1e-10)
                            ).sum()
                            entropy = vertex_entropy + block_entropy
                            entropy_loss = -entropy_coeff * entropy

                            loss = policy_loss + value_loss + entropy_loss
                            all_losses.append(loss)

                            total_policy_loss += policy_loss.item()
                            total_value_loss += value_loss.item()
                            total_entropy += entropy.item()
                            total_valid_count += 1

                            offset += n_i

                # Gradient step for the entire PPO minibatch
                if len(all_losses) > 0:
                    total_loss = torch.stack(all_losses).mean()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.use_amp:
                        self.scaler.scale(total_loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        total_loss.backward()
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        self.optimizer.step()

        divisor = max(total_valid_count, 1)
        return {
            'policy_loss': total_policy_loss / divisor,
            'value_loss': total_value_loss / divisor,
            'entropy': total_entropy / divisor,
        }

    def _collect_episode_batch_item(
        self,
        episode_idx: int,
        graphs: List[Graph],
        initial_partitions: List[np.ndarray],
        spectral_coords_list: Optional[List[np.ndarray]],
        k: int,
        epsilon: float,
        entropy_coeff: float,
    ) -> Tuple[List[Transition], List[np.ndarray], List[np.ndarray], Dict]:
        idx = episode_idx % len(graphs)
        G = graphs[idx]
        init_part = initial_partitions[idx]
        spec_coords = spectral_coords_list[idx] if spectral_coords_list else None

        env = PartitionEnv(
            G, init_part, k, epsilon, spec_coords,
            use_local_reward=self.config.use_local_reward,
            balance_penalty=self.config.balance_penalty,
            max_steps=self.config.max_episode_steps,
            min_steps=self.config.min_episode_steps,
            normalize_reward=self.config.normalize_reward,
            use_potential_shaping=self.config.use_potential_shaping,
            use_edge_features=self.config.use_edge_features,
            gamma=self.config.gamma,
            device=self.device,
        )

        num_starts = self.config.num_pomo_starts if self.config.use_pomo_baseline else 1
        pomo_rewards = []
        pomo_transitions_list = []
        best_pomo_part = None
        best_pomo_cut = float('inf')

        for pomo_idx in range(num_starts):
            if pomo_idx > 0:
                perturbed = init_part.copy()
                boundary = G.boundary_vertices_fast(perturbed)
                if len(boundary) > 0:
                    rng = np.random.RandomState(episode_idx * 100 + pomo_idx)
                    n_perturb = max(1, len(boundary) // 10)
                    to_move = rng.choice(
                        boundary, size=min(n_perturb, len(boundary)), replace=False
                    )
                    for v in to_move:
                        nbrs, _ = G.neighbors(v)
                        adj_blocks = set()
                        for nb in nbrs:
                            if perturbed[nb] != perturbed[v]:
                                adj_blocks.add(perturbed[nb])
                        if adj_blocks:
                            perturbed[v] = rng.choice(list(adj_blocks))
                env_pomo = PartitionEnv(
                    G, perturbed, k, epsilon, spec_coords,
                    use_local_reward=self.config.use_local_reward,
                    balance_penalty=self.config.balance_penalty,
                    max_steps=self.config.max_episode_steps,
                    min_steps=self.config.min_episode_steps,
                    normalize_reward=self.config.normalize_reward,
                    use_potential_shaping=self.config.use_potential_shaping,
                    use_edge_features=self.config.use_edge_features,
                    gamma=self.config.gamma,
                    device=self.device,
                )
            else:
                env_pomo = env

            transitions, total_reward, best_part = self.collect_episode(env_pomo)
            pomo_rewards.append(total_reward)
            pomo_transitions_list.append(transitions)

            part_cut = compute_edge_cut(G, best_part)
            if part_cut < best_pomo_cut:
                best_pomo_cut = part_cut
                best_pomo_part = best_part

        episode_transitions: List[Transition] = []
        episode_advantages: List[np.ndarray] = []
        episode_returns: List[np.ndarray] = []

        # POMO-baseline tagging: see collect_batch_item.
        for transitions in pomo_transitions_list:
            if len(transitions) == 0:
                continue
            advantages, returns = self.compute_gae(transitions)
            tagged = [t._replace(pomo_group_id=int(episode_idx)) for t in transitions]
            episode_transitions.extend(tagged)
            episode_advantages.append(advantages)
            episode_returns.append(returns)

        init_cut = compute_edge_cut(G, init_part)
        best_cut = best_pomo_cut if best_pomo_part is not None else init_cut

        stats = {
            'episode': episode_idx,
            'episode_reward': pomo_rewards[0] if pomo_rewards else 0.0,
            'episode_length': len(pomo_transitions_list[0]) if pomo_transitions_list else 0,
            'initial_cut': init_cut,
            'final_cut': best_cut,
            'improvement_pct': (
                (init_cut - best_cut) / init_cut * 100
                if init_cut > 0 else 0
            ),
            'balance_ok': self._check_balance_static(G, best_pomo_part, k, epsilon) if best_pomo_part is not None else False,
            'entropy_coeff': entropy_coeff,
            'num_pomo_starts': num_starts,
        }
        return episode_transitions, episode_advantages, episode_returns, stats

    def train(
        self,
        graphs: List[Graph],
        initial_partitions: List[np.ndarray],
        k: int,
        epsilon: float = 0.03,
        spectral_coords_list: Optional[List[np.ndarray]] = None,
        num_episodes: int = 500,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Train with POMO multi-start and cosine entropy schedule.
        Supports batched PPO and process-based episode collection.
        """
        self.model.train()
        training_log = []
        episodes_per_batch = self.config.episodes_per_batch
        episode_counter = 0
        total_transitions_collected = 0
        train_start = time.perf_counter()

        use_batched = getattr(self.config, 'use_batched_ppo', True)
        use_processes = getattr(self.config, 'use_process_workers', True)

        if verbose:
            mode_parts = []
            if use_batched:
                mode_parts.append("batched-PPO")
            if use_processes:
                mode_parts.append("process-workers")
            mode_str = " + ".join(mode_parts) if mode_parts else "baseline"
            print(f"  Training mode: {mode_str}")

        # Pre-serialize graphs for process workers (once, not per batch)
        graphs_data = None
        config_dict = None
        if use_processes:
            graphs_data = [_serialize_graph(G) for G in graphs]
            config_dict = _config_to_dict(self.config)
            # Convert numpy arrays to lists for pickling
            init_parts_serial = [p.copy() for p in initial_partitions]
            spec_coords_serial = (
                [s.copy() if s is not None else None for s in spectral_coords_list]
                if spectral_coords_list else None
            )

        while episode_counter < num_episodes:
            batch_start = time.perf_counter()
            batch_transitions: List[Transition] = []
            batch_advantages_list: List[np.ndarray] = []
            batch_returns_list: List[np.ndarray] = []
            batch_stats: List[Dict] = []
            update_stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

            # Get current entropy coefficient
            entropy_coeff = self._get_entropy_coeff(episode_counter, num_episodes)

            # Update learning rate (cosine annealing)
            if self.config.use_cosine_lr:
                lr = self._get_learning_rate(episode_counter, num_episodes)
                for param_group in self.optimizer.param_groups:
                    if param_group.get('name') != 'critic':
                        param_group['lr'] = lr

            batch_count = min(episodes_per_batch, num_episodes - episode_counter)
            episode_ids = [episode_counter + i for i in range(batch_count)]
            episode_workers = max(1, int(getattr(self.config, "episode_workers", 1)))

            if episode_workers > 1 and batch_count > 1 and use_processes:
                # === ProcessPoolExecutor path ===
                model_state_bytes = self._serialize_model_state()

                ctx = mp.get_context('spawn')
                with ProcessPoolExecutor(
                    max_workers=min(episode_workers, batch_count),
                    mp_context=ctx,
                ) as ex:
                    futures = {}
                    for ep_idx in episode_ids:
                        args = (
                            ep_idx, graphs_data, init_parts_serial,
                            spec_coords_serial, k, epsilon, entropy_coeff,
                            config_dict, model_state_bytes,
                            self.feature_dim, self.edge_dim,
                        )
                        fut = ex.submit(_collect_episode_worker, args)
                        futures[fut] = ep_idx

                    episode_outputs = []
                    for fut in as_completed(futures):
                        ep_idx = futures[fut]
                        numpy_transitions, ep_advantages, ep_returns, stats = fut.result()
                        # Convert back to torch tensors on training device
                        torch_transitions = [
                            _transition_from_numpy(tn, self.device)
                            for tn in numpy_transitions
                        ]
                        episode_outputs.append((
                            ep_idx,
                            (torch_transitions, ep_advantages, ep_returns, stats)
                        ))
                    episode_outputs.sort(key=lambda x: x[0])

            elif episode_workers > 1 and batch_count > 1:
                # === ThreadPoolExecutor path (fallback) ===
                futures = {}
                with ThreadPoolExecutor(max_workers=min(episode_workers, batch_count)) as ex:
                    for ep_idx in episode_ids:
                        fut = ex.submit(
                            self._collect_episode_batch_item,
                            ep_idx,
                            graphs,
                            initial_partitions,
                            spectral_coords_list,
                            k,
                            epsilon,
                            entropy_coeff,
                        )
                        futures[fut] = ep_idx

                    episode_outputs = []
                    for fut in as_completed(futures):
                        episode_outputs.append((futures[fut], fut.result()))
                    episode_outputs.sort(key=lambda x: x[0])
            else:
                # === Sequential path ===
                episode_outputs = [
                    (
                        ep_idx,
                        self._collect_episode_batch_item(
                            ep_idx,
                            graphs,
                            initial_partitions,
                            spectral_coords_list,
                            k,
                            epsilon,
                            entropy_coeff,
                        ),
                    )
                    for ep_idx in episode_ids
                ]

            for _, output in episode_outputs:
                episode_transitions, episode_advantages, episode_returns, stats = output
                batch_transitions.extend(episode_transitions)
                batch_advantages_list.extend(episode_advantages)
                batch_returns_list.extend(episode_returns)
                batch_stats.append(stats)

            episode_counter += batch_count

            # PPO update -- dispatch to batched or one-at-a-time
            if len(batch_transitions) > 0:
                all_advantages = np.concatenate(batch_advantages_list)
                all_returns = np.concatenate(batch_returns_list)
                total_transitions_collected += len(batch_transitions)

                if use_batched:
                    update_stats = self.batched_update_gpu(
                        batch_transitions, all_advantages, all_returns, entropy_coeff
                    )
                else:
                    update_stats = self.batched_update(
                        batch_transitions, all_advantages, all_returns, entropy_coeff
                    )

                batch_seconds = max(time.perf_counter() - batch_start, 1e-12)
                total_seconds = max(time.perf_counter() - train_start, 1e-12)
                batch_eps = len(batch_stats)
                batch_trans_per_sec = len(batch_transitions) / batch_seconds
                batch_eps_per_sec = batch_eps / batch_seconds
                total_eps_per_sec = episode_counter / total_seconds
                total_trans_per_sec = total_transitions_collected / total_seconds

                for s in batch_stats:
                    s.update(update_stats)
                    s['batch_seconds'] = batch_seconds
                    s['batch_episodes_per_sec'] = batch_eps_per_sec
                    s['batch_transitions_per_sec'] = batch_trans_per_sec
                    s['total_episodes_per_sec'] = total_eps_per_sec
                    s['total_transitions_per_sec'] = total_trans_per_sec
                    training_log.append(s)

            # Update greedy baseline periodically
            if (self.config.use_greedy_baseline and
                episode_counter % self.config.greedy_baseline_update == 0):
                self._update_greedy_baseline()

            # Print progress
            if verbose and batch_stats:
                latest = batch_stats[-1]
                avg_reward = np.mean([s['episode_reward'] for s in batch_stats])
                avg_improv = np.mean([s['improvement_pct'] for s in batch_stats])
                avg_len = np.mean([s['episode_length'] for s in batch_stats])
                wins = sum(1 for s in batch_stats if s['improvement_pct'] > 0)
                ep_num = latest['episode']

                if ep_num % 50 < episodes_per_batch or ep_num < episodes_per_batch:
                    print(
                        f"Ep {ep_num:>4d} | batch: reward={avg_reward:+.4f}, "
                        f"improv={avg_improv:+.2f}%, len={avg_len:.1f}, "
                        f"wins={wins}/{len(batch_stats)} | "
                        f"entropy={update_stats.get('entropy', 0):.3f}, "
                        f"ent_coeff={entropy_coeff:.4f} | "
                        f"eps/s={latest.get('batch_episodes_per_sec', 0.0):.2f}, "
                        f"trans/s={latest.get('batch_transitions_per_sec', 0.0):.1f}"
                    )

        return training_log

    @staticmethod
    def _check_balance_static(G, partition, k, epsilon):
        if partition is None:
            return False
        max_weight = (1.0 + epsilon) * np.ceil(G.total_weight / k)
        block_weights = np.zeros(k, dtype=np.float64)
        for v in range(G.n):
            block_weights[partition[v]] += G.vertex_weights[v]
        return bool(block_weights.max() <= max_weight + 1e-9)

    def refine(
        self,
        G: Graph,
        partition: np.ndarray,
        k: int,
        epsilon: float = 0.03,
        spectral_coords: Optional[np.ndarray] = None,
        num_starts: int = 1,
    ) -> np.ndarray:
        """
        Use trained policy to refine a partition (inference mode).
        Supports multi-start (best-of-K) at inference.
        Fixed: single GNN call per step (was two before).
        """
        self.model.eval()
        # [patched] align spectral coord dim to what the loaded model was trained on
        _model_sd = getattr(getattr(self.model, 'gnn', None), 'spectral_dim', None)
        if spectral_coords is not None and _model_sd is not None and spectral_coords.shape[-1] > _model_sd:
            spectral_coords = spectral_coords[:, :_model_sd]
        best_partition = partition.copy()
        best_cut = compute_edge_cut(G, partition)

        for start_idx in range(num_starts):
            # Vary initial conditions
            if start_idx > 0:
                init = partition.copy()
                boundary = G.boundary_vertices_fast(init)
                if len(boundary) > 0:
                    rng = np.random.RandomState(start_idx * 31)
                    n_perturb = max(1, len(boundary) // 10)
                    to_move = rng.choice(boundary, size=min(n_perturb, len(boundary)), replace=False)
                    for v in to_move:
                        nbrs, _ = G.neighbors(v)
                        adj_blocks = set()
                        for nb in nbrs:
                            if init[nb] != init[v]:
                                adj_blocks.add(init[nb])
                        if adj_blocks:
                            init[v] = rng.choice(list(adj_blocks))
            else:
                init = partition

            env = PartitionEnv(
                G, init, k, epsilon, spectral_coords,
                use_local_reward=False,
                balance_penalty=self.config.balance_penalty,
                max_steps=self.config.max_episode_steps,
                min_steps=0,
                normalize_reward=False,
                use_potential_shaping=False,
                use_edge_features=self.config.use_edge_features,
                gamma=self.config.gamma,
                device=self.device,
            )

            state = env.reset()

            with torch.no_grad():
                for step in range(self.config.max_episode_steps):
                    valid_actions = env.get_valid_actions()
                    if len(valid_actions) == 0:
                        break

                    with self._autocast_context():
                        # Single GNN call (fixed: was two calls before)
                        node_emb_all = self.model.gnn(
                            state['features'], state['adj_indices'],
                            state['adj_weights'], state['deg_inv'],
                            state.get('edge_features'),
                        )
                        vertex_scores = self.model.actor_vertex(node_emb_all).squeeze(-1)
                        mask_value = torch.finfo(vertex_scores.dtype).min
                        vertex_logits = vertex_scores.masked_fill(
                            ~state['boundary_mask'], mask_value
                        )

                    valid_vertices = sorted(set(v for v, b in valid_actions))
                    if len(valid_vertices) == 0:
                        break

                    valid_vtx_t = torch.tensor(
                        valid_vertices, dtype=torch.long, device=self.device
                    )
                    vertex_scores_sel = vertex_logits[valid_vtx_t]
                    best_vtx_idx = vertex_scores_sel.argmax()
                    selected_vertex = valid_vertices[int(best_vtx_idx)]

                    candidate_blocks = sorted(set(
                        b for v, b in valid_actions if v == selected_vertex
                    ))
                    if len(candidate_blocks) == 0:
                        break

                    candidates_tensor = torch.tensor(
                        candidate_blocks, dtype=torch.long, device=self.device
                    )
                    with self._autocast_context():
                        # Reuse cached node embedding
                        node_emb = node_emb_all[selected_vertex]
                        block_logits = self.model.get_block_logits(node_emb, candidates_tensor)
                    best_block_idx = block_logits.argmax()
                    selected_block = candidate_blocks[int(best_block_idx)]

                    state, _, done = env.step(selected_vertex, selected_block)
                    if done:
                        break

            result = env.get_best_partition()
            result_cut = compute_edge_cut(G, result)
            if result_cut < best_cut:
                best_cut = result_cut
                best_partition = result

        return best_partition

    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict() if self.use_amp else None,
            'config': self.config,
            'k': self.k,
            'feature_dim': self.feature_dim,
            'edge_dim': self.edge_dim,
            'spectral_dim': self.config.spectral_dim,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        ckpt_k = checkpoint.get('k')
        if ckpt_k is not None and ckpt_k != self.k:
            raise ValueError(
                f"Checkpoint k={ckpt_k} does not match trainer k={self.k}"
            )
        ckpt_fd = checkpoint.get('feature_dim')
        if ckpt_fd is not None and ckpt_fd != self.feature_dim:
            raise ValueError(
                f"Checkpoint feature_dim={ckpt_fd} does not match trainer "
                f"feature_dim={self.feature_dim} (likely spectral_dim or "
                f"use_spectral_features mismatch)"
            )
        ckpt_ed = checkpoint.get('edge_dim')
        if ckpt_ed is not None and ckpt_ed != self.edge_dim:
            raise ValueError(
                f"Checkpoint edge_dim={ckpt_ed} does not match trainer "
                f"edge_dim={self.edge_dim} (use_edge_features mismatch)"
            )
        ckpt_sd = checkpoint.get('spectral_dim')
        if ckpt_sd is not None and ckpt_sd != self.config.spectral_dim:
            raise ValueError(
                f"Checkpoint spectral_dim={ckpt_sd} does not match trainer "
                f"spectral_dim={self.config.spectral_dim}"
            )
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        scaler_state = checkpoint.get('scaler_state')
        if self.use_amp and scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)
