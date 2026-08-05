#!/usr/bin/env python3
"""Live demo: the four claims that can be shown in a couple of minutes.

The full benchmark is an HPC artefact and cannot be reproduced live (roughly
four minutes per graph per seed on a laptop, times 67 graphs, times 3 seeds).
What CAN be shown live:

  (1) the deployed checkpoint reloads through the real architecture, strictly,
      and reports its own configuration;
  (2) the encoder's three invariances hold end to end on the DEPLOYED weights,
      measured now rather than quoted -- permutation of the vertices, sign flip
      of the eigenvectors, rotation inside a degenerate eigenspace;
  (3) the global branch is linear where dense attention is quadratic, on the
      same encoder object the pipeline uses;
  (4) the pipeline runs end to end on six real archive graphs: the refiner
      engages at every gated level, the guard returns its verdict, the block
      sizes are printed with their sum checked against n, and the cut
      is compared with METIS at the same (reduced) trial budget, and with
      KaHIP eco, KaHIP strong and Scotch at their own presets. Everything
      printed is computed in this run; nothing is quoted from stored results.

Everything below imports the real repository modules. Nothing is reimplemented
for the demo, which is the point: a marker can follow any number here back into
the code that produced the thesis.

    python src/demo.py                # the whole demonstration, ~4 min
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from srmp.graph import Graph, read_metis                      # noqa: E402
from srmp.spectral import compute_eigenvectors                # noqa: E402
from srmp.evaluation import compute_edge_cut, compute_imbalance  # noqa: E402
from graph_transformer import SpectralGraphTransformer        # noqa: E402
from refiner import build_state, load_spectral_actor_critic   # noqa: E402
from multilevel_partitioner import make_multilevel_partitioner  # noqa: E402

RULE = "=" * 66

# The live end-to-end run uses a REDUCED budget so it finishes in ~30 s:
# best-of-4 coarsest starts, a single V-cycle, no evolutionary stage, one
# fixed seed -- and METIS gets the SAME reduced budget (ncuts = 4, same
# seed). The thesis numbers use best-of-12, the evolutionary stage and three
# seeds per graph (Chapter 5); this section demonstrates the machinery, and
# the benchmark chapter carries the claims.
LIVE_BOK = 4
LIVE_SEED = 1


def rand_graph(n, deg=6, seed=0):
    """A connected random graph, the same construction verify_ab.py uses."""
    rng = np.random.RandomState(seed)
    row, col = [], []
    for i in range(n):
        j = (i + 1) % n
        row += [i, j]
        col += [j, i]
    for _ in range(deg * n // 2):
        a, b = rng.randint(0, n), rng.randint(0, n)
        if a != b:
            row += [a, b]
            col += [b, a]
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    return Graph(A)


def run_kahip(path, k, eps, seed, preconfig):
    """KaHIP (kaffpa) on the METIS file, one run at the named preset."""
    import shutil as _sh, subprocess, tempfile
    if _sh.which("kaffpa") is None:
        return None
    outp = tempfile.mktemp(suffix=".part")
    try:
        subprocess.run(["kaffpa", path, f"--k={k}", f"--imbalance={round(eps * 100)}",
                        f"--preconfiguration={preconfig}", f"--seed={seed}",
                        f"--output_filename={outp}"],
                       capture_output=True, text=True, check=True)
        return np.loadtxt(outp, dtype=int)
    except Exception:
        return None
    finally:
        if os.path.exists(outp):
            os.remove(outp)


def run_scotch(path, k, eps, n):
    """Scotch (gcv + gpart) on the METIS file, one run at the default strategy."""
    import shutil as _sh, subprocess, tempfile
    if _sh.which("gpart") is None or _sh.which("gcv") is None:
        return None
    grf, mp = tempfile.mktemp(suffix=".grf"), tempfile.mktemp(suffix=".map")
    try:
        subprocess.run(["gcv", "-ic", "-os", path, grf],
                       capture_output=True, text=True, check=True)
        subprocess.run(["gpart", f"-b{eps}", str(k), grf, mp],
                       capture_output=True, text=True, check=True)
        part = np.empty(n, dtype=int)
        pairs = np.array(open(mp).read().split()[1:], dtype=int).reshape(-1, 2)
        part[pairs[:, 0] - pairs[:, 0].min()] = pairs[:, 1]
        return part
    except Exception:
        return None
    finally:
        for f in (grf, mp):
            if os.path.exists(f):
                os.remove(f)


def find_checkpoint():
    """A checkpoint from models/, so the demo runs without --model.
    Prefers a k=4 checkpoint, which is the configuration Chapter 5 reports."""
    import glob as _glob
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hits = sorted(_glob.glob(os.path.join(here, "models", "*.pt"))) or \
        sorted(_glob.glob(os.path.expanduser("~/Downloads/Models/*.pt")))
    k4 = [h for h in hits if "k4" in os.path.basename(h)]
    return (k4 or hits or [""])[0]


def find_graphs():
    """The shipped benchmark graphs: a benchmark in miniature, in this order.
    Two headline wins (rdb3200l, conf5_0-4x4-14), a mesh win near parity
    (data), a tie (power), two known parity-regime losses (uk, 3elt), and
    add20, the archive's circuit graph, whose deep hierarchy makes it the
    slowest of the seven by far (several minutes on a laptop)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = ["rdb3200l.graph", "conf5_0-4x4-14.graph", "data.graph",
             "power.graph", "uk.graph", "3elt.graph", "add20.graph"]
    return [p for p in (os.path.join(here, "graphs", n) for n in names)
            if os.path.exists(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="trained checkpoint; defaults to models/spectral_refiner_k4_*.pt")
    ap.add_argument("--graph", nargs="+", default=None,
                    help="METIS-format graph(s) for the end-to-end section; "
                         "defaults to the two shipped headline wins")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-benchmark", action="store_true",
                    help="skip section 4 (the live end-to-end partitions)")
    a = ap.parse_args()
    if a.model is None:
        a.model = find_checkpoint()
    if a.graph is None:
        a.graph = find_graphs()
    torch.manual_seed(0)

    # ---------------- (1) the deployed checkpoint, strictly reloaded ----------------
    print(RULE)
    print(" (1) THE DEPLOYED CHECKPOINT, RELOADED THROUGH THE ARCHITECTURE")
    print(RULE)
    policy = None
    k, n_eigs = 4, 8
    if os.path.exists(a.model):
        policy, ck = load_spectral_actor_critic(a.model)
        k, n_eigs = int(ck["k"]), int(ck.get("n_eigs", 8))
        n_par = sum(p.numel() for p in policy.parameters())
        print(f"  file            {os.path.basename(a.model)}")
        print(f"  parameters      {n_par:,}   (instantiated model, strict load)")
        for key in ("k", "epsilon", "global_kind", "pe_kind", "d_model"):
            if key in ck:
                print(f"  {key:15s} {ck[key]}")
        print("  -> the checkpoint stores its own constructor arguments, and the")
        print("     load is strict: a mismatch with the architecture would fail")
        print("     rather than load silently. k=4 is the Chapter 5 configuration.")
    else:
        print(f"  (checkpoint not found at {a.model!r}; sections 2 and 3 use a")
        print("   fresh encoder, and section 4 is skipped)")

    # ---------------- (2) the invariances, measured on the deployed weights ----------------
    print()
    print(RULE)
    print(" (2) THE THREE INVARIANCES, MEASURED NOW (not quoted)")
    print(RULE)
    n = 300 if a.quick else 600
    G = rand_graph(n, seed=3)
    pi = np.zeros(G.n, dtype=np.int64)
    x, ev_t, V_t, _, ai, aw, di = build_state(G, pi, k, n_eigs=n_eigs)

    if policy is not None:
        enc = policy.encoder.eval()
        src = "the DEPLOYED trained encoder"
    else:
        enc = SpectralGraphTransformer(in_dim=x.shape[1], d_model=64, n_heads=4,
                                       n_layers=2, use_local=True,
                                       global_kind="lanczos").eval()
        src = "a fresh encoder (no checkpoint found)"
    with torch.no_grad():
        H0 = enc(x, ev_t, V_t, ai, aw, di)

        # (a) eigenvector SIGN flip: V -> V S, S = diag(+-1)
        signs = torch.tensor(np.random.RandomState(0).choice([-1.0, 1.0], n_eigs),
                             dtype=V_t.dtype)
        H_sign = enc(x, ev_t, V_t * signs, ai, aw, di)
        d_sign = (H0 - H_sign).abs().max().item()

        # (b) BASIS rotation inside a genuinely DEGENERATE eigenspace.
        #     A random graph has generically distinct eigenvalues, so it has no
        #     degenerate subspace and rotating any pair is NOT a symmetry. The
        #     cycle C_n does: its normalised-Laplacian eigenvalues 1 - cos(2 pi j / n)
        #     are exactly doubled, with a sine/cosine pair spanning each eigenspace.
        ring = sp.csr_matrix((np.ones(2 * n),
                              (np.arange(n).repeat(2),
                               np.stack([(np.arange(n) + 1) % n,
                                         (np.arange(n) - 1) % n], 1).ravel())),
                             shape=(n, n))
        Gc = Graph(ring)
        xc, evc, Vc, _, aic, awc, dic = build_state(
            Gc, np.zeros(n, dtype=np.int64), k, n_eigs=n_eigs)
        gaps = (evc[1:] - evc[:-1]).abs()
        j = int(torch.argmin(gaps))                   # the tightest pair
        gap = float(gaps[j])
        Hc = enc(xc, evc, Vc, aic, awc, dic)
        Vr = Vc.clone()
        th = 0.7853981633974483                       # pi/4
        c, s = float(np.cos(th)), float(np.sin(th))
        Vr[:, j], Vr[:, j + 1] = (c * Vc[:, j] - s * Vc[:, j + 1],
                                  s * Vc[:, j] + c * Vc[:, j + 1])
        H_basis = enc(xc, evc, Vr, aic, awc, dic)
        d_basis = (Hc - H_basis).abs().max().item()

        # (c) vertex PERMUTATION: relabel the graph, encode, unpermute
        perm = np.random.RandomState(1).permutation(G.n)
        P = sp.csr_matrix((np.ones(G.n), (np.arange(G.n), perm)), shape=(G.n, G.n))
        Gp = Graph((P @ G.adj @ P.T).tocsr())
        xp, evp, Vp, _, aip, awp, dip = build_state(Gp, pi[perm], k, n_eigs=n_eigs)
        Hp = enc(xp, evp, Vp, aip, awp, dip)
        inv = np.argsort(perm)
        d_perm = (H0 - Hp[torch.as_tensor(inv)]).abs().max().item()

    print(f"  measured on {src}")
    print(f"  sign / permutation on a random graph: n = {G.n}, m = {G.adj.nnz // 2}")
    print(f"  basis rotation on the cycle C_{n}, whose eigenvalues are exactly")
    print(f"  doubled: the rotated pair has eigenvalue gap {gap:.2e}")
    print()
    print(f"  {'symmetry':34s} {'max |dH|':>12}")
    print(f"  {'-' * 34} {'-' * 12}")
    print(f"  {'eigenvector sign flip  V -> VS':34s} {d_sign:12.2e}")
    print(f"  {'basis rotation, degenerate pair':34s} {d_basis:12.2e}")
    print(f"  {'vertex permutation  G -> PGP^T':34s} {d_perm:12.2e}")
    worst = max(d_sign, d_basis, d_perm)
    print()
    print(f"  worst case {worst:.2e} -- the thesis claims ~1e-6 end to end.")
    print("  These are exact symmetries of the construction, not fitted behaviour:")
    print("  the filters see only eigenVALUES, and eigenvectors enter only through")
    print("  the eigenprojector V diag(.) V^T, which is invariant to both.")

    # ---------------- (3) linear vs quadratic global branch ----------------
    print()
    print(RULE)
    print(" (3) GLOBAL BRANCH COST: LANCZOS O(n) vs DENSE ATTENTION O(n^2)")
    print(RULE)
    ns = [500, 1000, 2000] if a.quick else [500, 1000, 2000, 4000]
    print(f"  {'n':>6} {'attention ms':>13} {'lanczos ms':>12} {'speedup':>9}")
    print(f"  {'-' * 6} {'-' * 13} {'-' * 12} {'-' * 9}")
    for nn in ns:
        Gn = rand_graph(nn, seed=1)
        xn, evn, Vn, _, ain, awn, din = build_state(
            Gn, np.zeros(nn, dtype=np.int64), k, n_eigs=n_eigs)

        def ms(kind):
            try:
                e = SpectralGraphTransformer(in_dim=xn.shape[1], d_model=64,
                                             n_heads=4, n_layers=2,
                                             use_local=True,
                                             global_kind=kind).eval()
                with torch.no_grad():
                    e(xn, evn, Vn, ain, awn, din)             # warm up
                    t0 = time.time()
                    for _ in range(2):
                        e(xn, evn, Vn, ain, awn, din)
                return (time.time() - t0) / 2 * 1000
            except RuntimeError:
                return float("nan")

        ta, tl = ms("attn"), ms("lanczos")
        sp_s = f"{ta / tl:.2f}x" if (ta == ta and tl == tl and tl > 0) else "attn OOM"
        print(f"  {nn:>6} {ta:>13.1f} {tl:>12.1f} {sp_s:>9}")
    print()
    print("  The gap widens with n. Appendix I carries the same measurement to")
    print("  n = 50,000, where dense attention runs out of memory and Lanczos")
    print("  completes -- which is what makes a refiner at EVERY level feasible.")

    # ---------------- (4) the pipeline end to end, on real graphs ----------------
    graphs = [g for g in (a.graph or []) if os.path.exists(g)]
    if not a.no_benchmark and policy is not None and graphs:
        print()
        print(RULE)
        print(" (4) END TO END, LIVE: PARTITION REAL ARCHIVE GRAPHS")
        print(RULE)
        print(f"  live budget     best-of-{LIVE_BOK} starts, one V-cycle, seed {LIVE_SEED}")
        print(f"                  (the thesis protocol is best-of-12, evolutionary")
        print(f"                   search on, three seeds; Chapter 5 carries those claims)")
        summary = []
        for gpath in graphs:
            Gr = read_metis(gpath)
            gname = os.path.basename(gpath)
            print()
            print(f"  graph           {gname}   n = {Gr.n:,}   m = {Gr.adj.nnz // 2:,}")
            Pm = make_multilevel_partitioner(policy, k=k, epsilon=0.03, best_of_k=LIVE_BOK,
                                             num_cycles=1, evolutionary=False,
                                             ml_refine_max_n=4000, global_no_harm=True)
            t0 = time.time()
            part = Pm.partition(Gr, seed=LIVE_SEED)
            t_ml = time.time() - t0
            cut = compute_edge_cut(Gr, part)
            imb = compute_imbalance(Gr, part, k)
            arm = "multi-level" if Pm.ml_global_adopted else "FM-only"
            print(f"  pipeline        {t_ml:5.1f} s: refiner engaged at "
                  f"{Pm.ml_refine_tries} gated levels, kept at {Pm.ml_refine_calls}")
            print(f"  guard verdict   both arms ran; the {arm} arm had the lower final cut")
            print(f"  result          cut = {cut:.0f}   imbalance = {imb:+.4f}  (eps = 0.03)")
            sizes = np.bincount(np.asarray(part, dtype=int), minlength=k)
            ok = (len(sizes) == k and int(sizes.sum()) == Gr.n
                  and int(np.asarray(part).min()) >= 0)
            print(f"  blocks          sizes {[int(s) for s in sizes[:k]]}   "
                  f"sum = {int(sizes.sum()):,} = n ({'valid partition' if ok else 'INVALID'})")
            try:
                from benchmark import run_metis
                m = run_metis(Gr, k, ncuts=LIVE_BOK, seeds=(LIVE_SEED,))
                mc, r = m["cut"], cut / m["cut"]
                verdict = "win" if r < 0.995 else "tie" if r <= 1.005 else "loss"
                print(f"  METIS           cut = {mc:.0f}  at the same budget "
                      f"(ncuts = {LIVE_BOK}, seed {LIVE_SEED})")
                print(f"  ratio           r = {r:.3f}  at the live budget ({verdict})")
            except Exception as e:                    # pymetis absent: still a full run
                print(f"  (METIS comparison unavailable: {e})")
                mc, r, verdict = None, None, "n/a"
            comps = {}
            for label, pc in (("eco", run_kahip(gpath, k, 0.03, LIVE_SEED, "eco")),
                              ("strong", run_kahip(gpath, k, 0.03, LIVE_SEED, "strong")),
                              ("Scotch", run_scotch(gpath, k, 0.03, Gr.n))):
                comps[label] = float(compute_edge_cut(Gr, pc)) if pc is not None else None
            if any(v is not None for v in comps.values()):
                shown = "  ·  ".join(f"KaHIP {kk} {v:.0f}" if kk != "Scotch" else f"Scotch {v:.0f}"
                                     for kk, v in comps.items() if v is not None)
                print(f"  comparators     {shown}   (one run each, own presets)")
            summary.append((gname, cut, mc, comps, r, verdict))
        if len(summary) > 1:
            print()
            print(f"  {'graph':24s} {'ours':>7} {'METIS':>7} {'eco':>7} {'strong':>7} {'Scotch':>7} {'r':>7}  verdict")
            print(f"  {'-' * 24} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}  {'-' * 7}")
            for gname, cut, mc, comps, r, verdict in summary:
                cuts_all = [cut] + [v for v in [mc, comps.get("eco"), comps.get("strong"),
                                                comps.get("Scotch")] if v is not None]
                lo = min(cuts_all)
                def cell(v):
                    if v is None:
                        return f"{'-':>7}"
                    mark = "*" if v <= lo else " "
                    return f"{v:>6.0f}{mark}"
                r_s = f"{r:.3f}" if r is not None else "-"
                print(f"  {gname:24s} {cell(cut)} {cell(mc)} {cell(comps.get('eco'))} "
                      f"{cell(comps.get('strong'))} {cell(comps.get('Scotch'))} {r_s:>7}  {verdict}")
            print("  * lowest cut in the row. r and the verdict are against trials-matched")
            print("    METIS, the thesis baseline; KaHIP eco, KaHIP strong and Scotch run")
            print("    once each at their own presets, as in the Appendix G gallery.")
        print()
        print("  This section demonstrates the machinery on one fixed seed; the")
        print("  benchmark claims are the 67-graph, 3-seed tables of Chapter 5.")
    elif not a.no_benchmark:
        print()
        print(RULE)
        print(" (4) END TO END, LIVE: skipped "
              + ("(no checkpoint)" if policy is None else "(no graph found)"))
        print(RULE)

    print()
    print(RULE)
    print(" The full benchmark (67 graphs x 3 seeds x trials-matched METIS) is")
    print(" an HPC artefact: roughly 4 minutes per graph per seed on a laptop.")
    print(" Its records are in runs/*.jsonl and bench_data/*.tsv, and every figure")
    print(" script re-derives its numbers from them and asserts them before drawing.")
    print(RULE)


if __name__ == "__main__":
    main()
