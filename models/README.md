# Trained checkpoints

The four deployed `SpectralActorCritic` policies, one per number of blocks. Each was
trained by PPO with GAE on the synthetic pool only, over graphs drawn from every
coarsening depth, and each is the checkpoint behind the corresponding column of the
benchmark. None of the test graphs appears in training.

| file | k | parameters | episodes |
|---|---|---|---|
| `spectral_refiner_k4_eps_0_03.pt` | 4 | 180,731 | 6,000 |
| `spectral_refiner_k8_eps_0_03.pt` | 8 | 181,115 | 6,000 |
| `spectral_refiner_k16_eps_0_03.pt` | 16 | 181,883 | 6,000 |
| `spectral_refiner_k32_eps_0_03.pt` | 32 | 183,419 | 6,000 |

The parameter count grows with k only through the block embedding and the k-dependent
partition-state features; the encoder is the same size throughout.

Shared configuration: `epsilon = 0.03`, `d_model = 64`, `n_heads = 4`, `n_layers = 2`,
`pe_dim = 32`, `n_eigs = 8`, `num_filters = 8`, `global_kind = lanczos`,
`pe_kind = stable`, GPS local branch on, `coarsen_min = 100`, `ml_refine_max_n = 4000`.

Every checkpoint stores its own constructor arguments, so it reloads without any
hyperparameter being supplied a second time:

```python
import sys; sys.path.insert(0, "src")
from refiner import load_spectral_actor_critic
model, ck = load_spectral_actor_critic("models/spectral_refiner_k4_eps_0_03.pt")
print(ck["k"], ck["global_kind"], ck["param_count"])
```

The load is strict, so a checkpoint that did not match the current architecture would
fail rather than load silently with missing tensors.

The k = 4 file is the one the headline numbers report: 45 wins over 67 held-out graphs
at a geometric mean cut ratio of 0.979. Coverage is complete at k = 4 and k = 8, reaches
66 of 67 at k = 16, and is partial at k = 32.
