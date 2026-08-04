#!/usr/bin/env python3
"""A+B trainability + checkpoint round-trip + backward-compat (completes the local 'fully verified' gate).
- existing trained checkpoint still loads (defaults global_kind=attn, pe_kind=stable)
- each new mode: gradients flow through the encoder (trainable) AND save->strict-reload reproduces outputs
"""
import os, sys, tempfile
import numpy as np
import scipy.sparse as sp
import torch
SRC_DIR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SRC_DIR)
from srmp.graph import Graph
from refiner import SpectralActorCritic, load_spectral_actor_critic, build_state

torch.manual_seed(0)
k = 4

# small graph
rng = np.random.RandomState(0); n = 200; row, col = [], []
for i in range(n):
    j = (i + 1) % n; row += [i, j]; col += [j, i]
for _ in range(3 * n):
    a, b = rng.randint(0, n), rng.randint(0, n)
    if a != b: row += [a, b]; col += [b, a]
A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n)); A.data[:] = 1.0
A.sum_duplicates(); A.data[:] = 1.0
G = Graph(A)
part = rng.randint(0, k, n)
x, ev, V, bmask, ai, aw, di = build_state(G, part, k, n_eigs=8)
in_dim = x.shape[1]

# --- backward-compat: existing trained checkpoint still loads ---
print("=== backward-compat ===")
ckpath = "models/spectral_refiner_k4_eps_0_03.pt"
m0, ck0 = load_spectral_actor_critic(ckpath)
gk0 = ck0.get("global_kind", "attn"); pk0 = ck0.get("pe_kind", "stable")
with torch.no_grad():
    v0, val0 = m0(x, ev, V, bmask, ai, aw, di)
print(f"loaded {os.path.basename(ckpath)}: global_kind={gk0} pe_kind={pk0} "
      f"-> forward OK (v shape {tuple(v0.shape)}, value {val0.item():.3f})")
compat_ok = (gk0 == "attn" and pk0 == "stable" and v0.shape == (n,))

# --- each new mode: grads flow + ckpt round-trip ---
print("\n=== new modes: trainable + checkpoint round-trip ===")
print(f"{'mode':>18} {'enc_grad':>10} {'reload_maxdiff':>14} {'ok':>5}")
all_ok = compat_ok
for gk, pk in [("lanczos", "stable"), ("attn", "specformer"), ("lanczos", "specformer")]:
    torch.manual_seed(1)
    pol = SpectralActorCritic(k=k, in_dim=in_dim, d_model=64, n_heads=4, n_layers=2,
                              global_kind=gk, pe_kind=pk)
    # trainability: forward + backward, encoder gradient mass > 0
    pol.train()
    vsc, val = pol(x, ev, V, bmask, ai, aw, di)
    h = pol.encode(x, ev, V, ai, aw, di)
    bnode = int(torch.nonzero(bmask)[0])
    bl = pol.get_block_logits(h[bnode], torch.arange(k))
    loss = val + vsc[bmask].sum() + bl.sum()
    loss.backward()
    enc_grad = sum(p.grad.abs().sum().item() for p in pol.encoder.parameters() if p.grad is not None)
    # ckpt round-trip: save with the new args, strict reload, outputs match
    pol.eval()
    with torch.no_grad():
        ref = pol(x, ev, V, bmask, ai, aw, di)[0]
    ck = {"model_state": {kk: vv.cpu() for kk, vv in pol.state_dict().items()},
          "k": k, "in_dim": in_dim, "d_model": 64, "n_heads": 4, "n_layers": 2, "pe_dim": 32,
          "num_filters": 8, "pe_mode": "diag", "use_local": True, "global_kind": gk, "pe_kind": pk}
    tmp = tempfile.mktemp(suffix=".pt"); torch.save(ck, tmp)
    try:
        m2, _ = load_spectral_actor_critic(tmp)
        with torch.no_grad():
            out2 = m2(x, ev, V, bmask, ai, aw, di)[0]
        maxdiff = (ref - out2).abs().max().item()
    finally:
        os.remove(tmp)
    ok = enc_grad > 0 and maxdiff < 1e-6
    all_ok = all_ok and ok
    print(f"{gk+'/'+pk:>18} {enc_grad:>10.1f} {maxdiff:>14.2e} {str(ok):>5}")

print("\nPASS" if all_ok else "\nFAIL",
      "-- A+B trainable (grads flow), checkpoints strict-reload, existing checkpoint still loads.")
