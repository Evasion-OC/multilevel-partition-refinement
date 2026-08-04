"""
train_ssl_transfer.py -- the Huang-bridge experiment, made rigorous + HPC-runnable.
==================================================================================

Question (Lang Huang's self-supervision angle, on a Liao-style spectral encoder):
  *Does SSL pretraining of the spectral encoder help an RL refiner generalise to a
   graph family it never saw during fine-tuning?*

Controlled A/B, per held-out family, over many seeds:
  1. SSL-pretrain the encoder (masked + contrastive, partition-free degree features)
     on a corpus of families that EXCLUDES the held-out family.
  2. Transfer the encoder BACKBONE into a fresh SpectralActorCritic (only `in_proj`
     re-inits -- different input features). Verified clean: SpectralActorCritic.encoder
     IS a SpectralGraphTransformer, GPS local branch included.
  3. Fine-tune with RL on the HELD-OUT family, and compare to a FROM-SCRATCH refiner
     with the IDENTICAL init/graphs/seed/PPO -- so the only difference is whether the
     encoder body started pretrained.

Honest axis: transfer / sample-efficiency on the unseen family, NOT raw final cut. The
deliverable is the controlled result either way (a clean negative is a real finding).
This isolates the SSL effect on the single-level refiner (fewest moving parts); fold into
the multilevel method only if it transfers here.

Reproducible: pins threads (so the add32-style nondeterminism can't masquerade as an
effect) and reports cross-seed mean +/- std; an effect only counts if it clears the noise.

Usage (local smoke):
    python src/train_ssl_transfer.py --pool data/pool/train --heldout ba \
        --seeds 2 --pre-epochs 8 --episodes 8 --limit-corpus 2 --limit-heldout 2
HPC (real):
    python src/train_ssl_transfer.py --pool data/pool/train --heldout ba,geometric \
        --seeds 5 --pre-epochs 40 --episodes 40 --output results/ssl_transfer.json
"""
from __future__ import annotations
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from scipy import stats as sstats

from refiner import SpectralActorCritic
from graph_transformer import SpectralGraphTransformer
from pretrain import SSLObjective
from srmp.graph import read_metis
from srmp.spectral import compute_eigenvectors
import train_refiner as T   # reuse the single-level PPO loop (identical algorithm)

ALL_FAMILIES = ["ba", "er", "delaunay", "sbm", "geometric", "grid"]


def eigp(G, n_eigs):
    ev, V = compute_eigenvectors(G, num_eigenvectors=min(n_eigs, G.n - 1), normalized=True)
    return (torch.tensor(np.asarray(ev), dtype=torch.float32),
            torch.tensor(np.asarray(V), dtype=torch.float32))


def degree_feat(G):
    d = np.asarray(G.degrees, dtype=np.float32)
    return torch.tensor(d / max(d.max(), 1.0), dtype=torch.float32).reshape(-1, 1)


def family_files(pool, families, limit=None):
    files = []
    for fam in families:
        fs = sorted(glob.glob(os.path.join(pool, f"{fam}_*.graph")))
        files += fs[:limit] if limit else fs
    return files


def ssl_pretrain(corpus_files, d_model, n_eigs, epochs, lr, seed):
    """Pretrain the spectral encoder (masked + contrastive). Returns the encoder state_dict
    WITHOUT in_proj (re-inits downstream for the refiner's partition features)."""
    torch.manual_seed(seed); np.random.seed(seed)
    corpus = []
    for f in corpus_files:
        G = read_metis(f); ev, V = eigp(G, n_eigs)
        corpus.append((degree_feat(G), ev, V))
    enc = SpectralGraphTransformer(in_dim=1, d_model=d_model, n_heads=4, n_layers=2,
                                   pe_dim=32, n_eigs=n_eigs)
    ssl = SSLObjective(enc, in_dim=1, d_model=d_model)
    opt = torch.optim.Adam(ssl.parameters(), lr=lr)
    first = last = None
    for epoch in range(epochs):
        loss, parts = ssl(corpus)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ssl.parameters(), 1.0); opt.step()
        if epoch == 0:
            first = parts["total"]
        last = parts["total"]
    return ({k: v for k, v in enc.state_dict().items() if not k.startswith("in_proj")},
            first, last)


def run_refiner(pretrained_sd, graphs, eigs, episodes, seed, k, d_model, lr):
    """Fine-tune the refiner on `graphs`. Scratch and SSL share the IDENTICAL init (same seed) --
    the ONLY difference is whether the encoder body is overwritten with pretrained weights."""
    torch.manual_seed(seed); np.random.seed(seed)
    policy = SpectralActorCritic(k=k, in_dim=k + 6, d_model=d_model, n_heads=4, n_layers=2)
    if pretrained_sd is not None:
        miss = policy.encoder.load_state_dict(pretrained_sd, strict=False)
        bad = [m for m in miss.missing_keys if not m.startswith("in_proj")]
        assert not bad, f"unexpected missing keys (encoder mismatch): {bad}"
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    imps = []
    for ep in range(episodes):
        gi = ep % len(graphs); G = graphs[gi]
        env = T.make_env("spectral", G, T.balanced_partition(G.n, k, seed=100 + ep))
        ctx = {"adj_indices": env.adj_indices, "adj_weights": env.adj_weights,
               "deg_inv": env.deg_inv, "ev": eigs[gi][0], "V": eigs[gi][1]}
        init_cut = env.current_cut
        trans = T.run_episode(policy, env, ctx)
        if not trans:
            imps.append(0.0); continue
        adv, ret = T.gae([t["reward"] for t in trans], [t["value"] for t in trans])
        T.ppo_update(policy, opt, trans, ctx, adv, ret)
        imps.append(float(init_cut - env.best_cut))
    return imps


def windows(imps, nwin=3):
    w = max(1, len(imps) // nwin)
    return [float(np.mean(imps[i * w:(i + 1) * w])) for i in range(nwin)]


def paired_test(diffs):
    """Paired difference (SSL-minus-scratch) across seeds: mean, sample std, two-sided t-test p.
    Degenerate (n<2 or zero variance) -> p reflects whether the constant effect is non-zero; the
    *magnitude* is judged separately by a minimum-effect gate (so a tiny-but-consistent effect is
    NOT mistaken for a real one)."""
    d = np.asarray(diffs, dtype=float); n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else 0.0
    if n < 2 or sd < 1e-12:
        p = 0.0 if abs(m) > 1e-12 else 1.0
    else:
        p = float(2 * sstats.t.sf(abs(m / (sd / np.sqrt(n))), df=n - 1))
    return m, sd, p


def verdict(m, p, baseline_mag, label):
    """Symmetric verdict: significant (paired t p<0.05) AND effect >= 5% of the scratch baseline."""
    min_eff = 0.05 * max(abs(baseline_mag), 1e-9)
    if abs(m) < min_eff:
        return f"NEGLIGIBLE {label} (|effect| {abs(m):.2f} < 5% of baseline {baseline_mag:.2f})"
    if p < 0.05:
        return (f"POSITIVE {label}" if m > 0 else f"NEGATIVE {label}") + f" (paired t p={p:.3f})"
    return f"INCONCLUSIVE {label} (p={p:.3f} >= 0.05) -- need more seeds / larger corpus"


def run_family(heldout, pretrain_fams, pool, seeds, pre_epochs, episodes, d_model,
               n_eigs, lr_pre, lr_ft, limit_corpus, limit_heldout):
    corpus_files = family_files(pool, pretrain_fams, limit_corpus)
    heldout_files = family_files(pool, [heldout], limit_heldout)
    assert corpus_files, f"no pretrain graphs for {pretrain_fams} in {pool}"
    assert heldout_files, f"no held-out graphs for {heldout} in {pool}"
    # leakage guard: the held-out family must NOT be in the pretraining corpus.
    assert heldout not in pretrain_fams, f"{heldout} is in pretrain set -- would leak"
    graphs = [read_metis(f) for f in heldout_files]
    eigs = [eigp(G, n_eigs) for G in graphs]
    print(f"\n=== held-out: {heldout} ({len(graphs)} graphs, n~{graphs[0].n}) | "
          f"pretrain on {pretrain_fams} ({len(corpus_files)} graphs) | {len(seeds)} seeds ===",
          flush=True)

    rows = {"scratch": [], "ssl": []}          # per-seed [early, mid, late] windows
    mean_imp = {"scratch": [], "ssl": []}      # per-seed mean improvement over ALL episodes (whole-run)
    curves = {"scratch": [], "ssl": []}        # per-seed full per-episode improvement (for figures)
    for s in seeds:
        pre_sd, l0, l1 = ssl_pretrain(corpus_files, d_model, n_eigs, pre_epochs, lr_pre, s)
        imp_sc = run_refiner(None, graphs, eigs, episodes, s, T.K, d_model, lr_ft)
        imp_ss = run_refiner(pre_sd, graphs, eigs, episodes, s, T.K, d_model, lr_ft)
        rows["scratch"].append(windows(imp_sc)); rows["ssl"].append(windows(imp_ss))
        mean_imp["scratch"].append(float(np.mean(imp_sc))); mean_imp["ssl"].append(float(np.mean(imp_ss)))
        curves["scratch"].append(imp_sc); curves["ssl"].append(imp_ss)
        print(f"  seed {s}: SSL loss {l0:.2f}->{l1:.2f} | early(scratch,ssl)="
              f"({windows(imp_sc)[0]:+.2f},{windows(imp_ss)[0]:+.2f}) | "
              f"mean=({np.mean(imp_sc):+.2f},{np.mean(imp_ss):+.2f})", flush=True)

    sc, ss = np.array(rows["scratch"]), np.array(rows["ssl"])     # (seeds, 3)
    early_diff = ss[:, 0] - sc[:, 0]                              # paired SSL-minus-scratch, early window
    whole_diff = np.array(mean_imp["ssl"]) - np.array(mean_imp["scratch"])   # paired, whole run
    em, esd, ep = paired_test(early_diff)        # early window = the SAMPLE-EFFICIENCY claim
    wm, wsd, wp = paired_test(whole_diff)         # whole run = does the benefit persist
    base_early = float(sc[:, 0].mean())           # scratch baseline magnitude (for the 5% effect gate)
    base_whole = float(np.mean(mean_imp["scratch"]))
    res = {
        "heldout": heldout, "pretrain": pretrain_fams, "seeds": list(seeds),
        "early_scratch_mean": base_early, "early_ssl_mean": float(ss[:, 0].mean()),
        "early_diff_mean": em, "early_diff_std": esd, "early_p": ep,
        "whole_scratch_mean": base_whole, "whole_ssl_mean": float(np.mean(mean_imp["ssl"])),
        "whole_diff_mean": wm, "whole_diff_std": wsd, "whole_p": wp,
        "seeds_favoring_ssl_early": int(np.sum(early_diff > 0)),
        "verdict_early": verdict(em, ep, base_early, "sample-efficiency (early window)"),
        "verdict_whole": verdict(wm, wp, base_whole, "overall (whole run)"),
        "curves": curves,                       # per-seed per-episode improvement, for learning-curve figures
    }
    res["verdict"] = res["verdict_early"]         # headline = the sample-efficiency claim
    print(f"  -> early(sample-eff): {em:+.2f}+/-{esd:.2f} p={ep:.3f} "
          f"({res['seeds_favoring_ssl_early']}/{len(seeds)} favour SSL) | {res['verdict_early']}", flush=True)
    print(f"  -> whole(persist)   : {wm:+.2f}+/-{wsd:.2f} p={wp:.3f} | {res['verdict_whole']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="data/pool/train", help="dir of family-prefixed .graph files")
    ap.add_argument("--heldout", default="ba", help="comma list of held-out families (tested separately)")
    ap.add_argument("--pretrain", default="auto", help="comma list, or 'auto' = ALL families minus held-out")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pre-epochs", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--d-model", type=int, default=T.SPECTRAL_DMODEL)
    ap.add_argument("--n-eigs", type=int, default=8)
    ap.add_argument("--lr-pre", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=3e-4)
    ap.add_argument("--limit-corpus", type=int, default=None, help="cap graphs per pretrain family")
    ap.add_argument("--limit-heldout", type=int, default=None, help="cap held-out graphs")
    ap.add_argument("--threads", type=int, default=1, help="torch threads (1 = reproducible; 0 = leave default)")
    ap.add_argument("--output", default=None, help="write per-family results JSON")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    heldouts = [f.strip() for f in args.heldout.split(",") if f.strip()]
    seeds = list(range(args.seeds))
    print(f"Huang-bridge SSL transfer | pool={args.pool} | held-out={heldouts} | "
          f"seeds={args.seeds} pre_epochs={args.pre_epochs} episodes={args.episodes} "
          f"threads={args.threads} (reproducible={'yes' if args.threads==1 else 'no'})", flush=True)

    results = []
    for ho in heldouts:
        pretrain_fams = ([f for f in ALL_FAMILIES if f != ho] if args.pretrain == "auto"
                         else [f.strip() for f in args.pretrain.split(",") if f.strip() and f.strip() != ho])
        results.append(run_family(ho, pretrain_fams, args.pool, seeds, args.pre_epochs,
                                  args.episodes, args.d_model, args.n_eigs, args.lr_pre,
                                  args.lr_ft, args.limit_corpus, args.limit_heldout))

    print("\n=== SUMMARY (paired SSL-minus-scratch; early window = sample-efficiency) ===", flush=True)
    for r in results:
        print(f"  {r['heldout']:10s}: early {r['early_diff_mean']:+.2f}+/-{r['early_diff_std']:.2f} "
              f"(p={r['early_p']:.3f}) | {r['verdict_early']}", flush=True)
    if len(results) > 1:
        a = 0.05 / len(results)
        print(f"Multiple-comparisons caveat: {len(results)} families tested -- for family-wise alpha=0.05 "
              f"use Bonferroni p<{a:.3f}, not 0.05. Do NOT cherry-pick a single POSITIVE.", flush=True)
    print("Honest: transfer/sample-efficiency on the unseen family, not raw cut. "
          "Single-level refiner testbed; fold into the multilevel method only if positive.", flush=True)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        json.dump(results, open(args.output, "w"), indent=2)
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
