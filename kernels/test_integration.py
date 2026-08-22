"""The rewrite inside the model: flag on == flag off, on random inputs and on the shipped weights.

    python kernels/test_integration.py
"""
import copy, glob, os, sys
import torch
sys.path.insert(0, "src")
from graph_transformer import LanczosSpectralMix, set_spectral_rewrite
from refiner import load_spectral_actor_critic

torch.manual_seed(0)

def inputs(n, d, c, dtype):
    V, _ = torch.linalg.qr(torch.randn(n, d))
    return torch.randn(n, c, dtype=dtype), (torch.rand(d) * 2).to(dtype), V.to(dtype)

def check(mod, n, d, dtype, tol):
    # the eager path applies proj as an nn.Linear, so for bf16 inputs the module itself must be
    # bf16 (as in the refiner-perf bf16 arms); test on a cast copy and leave `mod` untouched
    m = copy.deepcopy(mod).to(dtype) if dtype != torch.float32 else mod
    h, ev, V = inputs(n, d, mod.proj.out_features, dtype)
    m.use_rewrite = False; ref = m(h, ev, V)
    m.use_rewrite = True;  out = m(h, ev, V)
    m.use_rewrite = False
    err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    assert err < tol, f"rel err {err:.2e} > {tol} (n={n}, d={d}, {dtype})"
    return err

# 1) fresh module, realistic shapes, both dtypes
m = LanczosSpectralMix(128, num_filters=8)
for n, d in [(500, 8), (5000, 16), (20000, 16)]:
    e32 = check(m, n, d, torch.float32, 1e-5)
    e16 = check(m, n, d, torch.bfloat16, 3e-2)
    print(f"fresh module n={n:>6} d={d}: fp32 rel err {e32:.1e}, bf16 rel err {e16:.1e}")

# 2) every mixer inside every shipped checkpoint, with its trained weights
for ck in sorted(glob.glob("models/spectral_refiner_k*_eps_0_03.pt")):
    model, meta = load_spectral_actor_critic(ck)
    mixers = [mod for mod in model.modules() if isinstance(mod, LanczosSpectralMix)]
    worst = max(check(mod, 3000, meta.get("n_eigs", 8), torch.float32, 1e-5) for mod in mixers)
    assert set_spectral_rewrite(model, True) == len(mixers)
    assert all(mod.use_rewrite for mod in mixers)
    set_spectral_rewrite(model, False)
    print(f"{os.path.basename(ck)}: {len(mixers)} mixers, worst fp32 rel err {worst:.1e}")
print("OK: rewrite is the same map inside the model")
