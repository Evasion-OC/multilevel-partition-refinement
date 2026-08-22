"""
benchmark.py -- benchmark a trained partitioner against METIS.
=============================================================

The benchmark protocol (pymetis kway/ufactor=30 baseline) with
`make_multilevel_partitioner`, so the learned refiner runs at EVERY uncoarsening level
(size-gated by ml_refine_max_n) rather than at the coarsest alone. Records how many levels
the refiner acted on (`ml_refine_tries`/`ml_refine_calls`), which shows where it engages.

Usage:
    python benchmark.py --model models/spectral_refiner_k4_eps_0_03.pt \
        --graph-dir /home/kg1111r/graphs/walshaw \
        --k 4 --epsilon 0.03 --seeds 1 --max-n 6000 --metis --output results/walshaw_k4.jsonl
"""
from __future__ import annotations
import os, sys, glob, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from refiner import load_spectral_actor_critic
from multilevel_partitioner import make_multilevel_partitioner
from srmp.graph import read_metis
from srmp.evaluation import compute_edge_cut, compute_imbalance


def run_metis(G, k, ufactor=30, ncuts=12, niter=10, seeds=(1, 2, 3)):
    """METIS via pymetis -- kway (recursive=False), ufactor=30, BEST-OF-ncuts, MEDIAN over seeds.

    ncuts is matched to the method's best_of_k so the baselines are TRIALS-MATCHED: the method reports the
    best of best_of_k initial partitions, so METIS must likewise return the best of ncuts
    partitionings. METIS is seed-sensitive (a single run can be ~20% worse than best-of-12, e.g.
    3elt 261/219 vs 209), so a single-run baseline understates METIS and spuriously inflates
    the win rate, which is what stops the cut-ratio comparison from overselling.

    The seed budget is matched too. The method is reported as the median over `seeds` seeds, so METIS
    is likewise run once per seed and the MEDIAN cut returned; a single unseeded METIS call left
    the two sides on different protocols (one sampled run against a median of three). This is the
    "3 seeds x 12 trials" protocol documented in the thesis solver table. The returned imbalance
    is the one belonging to the median-cut seed, not a median of imbalances, so the reported pair
    describes a single realised partition."""
    import pymetis
    adj = G.adj.tocsr()
    csr = pymetis.CSRAdjacency(adj.indptr, adj.indices)
    seeds = tuple(seeds) or (1,)
    cuts, imbs = [], []
    t0 = time.time()
    for s in seeds:
        opts = pymetis.Options(ufactor=ufactor, ncuts=ncuts, niter=niter, seed=int(s))
        result = pymetis.part_graph(k, adjacency=csr, recursive=False, options=opts)
        part = np.asarray(result.vertex_part, dtype=np.int32)
        cuts.append(float(compute_edge_cut(G, part)))
        imbs.append(float(compute_imbalance(G, part, k)))
    mid = int(np.argsort(cuts)[len(cuts) // 2])    # index of the median-cut seed
    return {"cut": float(cuts[mid]), "imbalance": imbs[mid], "time": time.time() - t0,
            "cuts": cuts, "seeds": list(seeds)}


def main():
    ap = argparse.ArgumentParser(description="Benchmark a trained partitioner against METIS.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--graph-dir", required=True, help="dir of METIS .graph test graphs (e.g. Walshaw)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--best-of-k", type=int, default=12, help="benchmark-protocol default")
    ap.add_argument("--metis-seeds", type=int, default=None,
                    help="seeds for the METIS baseline; defaults to --seeds so the two sides "
                         "share a seed budget (median over seeds, best-of-ncuts within a seed)")
    ap.add_argument("--refiner-steps", type=int, default=80)
    ap.add_argument("--stochastic", type=int, default=0, help="N sampled refiner rollouts (no retrain)")
    ap.add_argument("--ml-refine-max-n", type=int, default=None,
                    help="size gate for multi-level refinement (default: the checkpoint's value)")
    ap.add_argument("--lean", action="store_true", help="fast non-comparable config for iteration")
    ap.add_argument("--force-rl", action="store_true", help="bypass the eigengap gate")
    ap.add_argument("--no-global-guard", dest="global_no_harm", action="store_false", default=True,
                    help="disable the global no-harm guard (faster ~1x, but the cut may regress)")
    ap.add_argument("--metis", action="store_true", help="also run METIS baseline (needs pymetis)")
    ap.add_argument("--max-n", type=int, default=None, help="skip graphs with more than this many nodes")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=str, default=None,
                    help="I/N: run only files[I::N] of the (filtered) list -- for Slurm array jobs")
    ap.add_argument("--resume", action="store_true",
                    help="skip graphs already present in --output and append (survive time-limit kills)")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--spectral-rewrite", action="store_true",
                    help="evaluate the spectral mixer via the reassociated forward (same map, faster)")
    args = ap.parse_args()

    policy, ck = load_spectral_actor_critic(args.model, map_location="cpu")
    if args.spectral_rewrite:
        from graph_transformer import set_spectral_rewrite
        print(f"spectral rewrite enabled on {set_spectral_rewrite(policy, True)} mixer(s)", flush=True)
    min_vertices = ck.get("coarsen_min", 100)
    n_eigs = ck.get("n_eigs", 8)
    ml_max = args.ml_refine_max_n if args.ml_refine_max_n is not None else ck.get("ml_refine_max_n", 4000)
    print(f"loaded model {args.model}  (k={ck['k']} d_model={ck['d_model']} gps={ck['gps_hybrid']} "
          f"coarsen_min={min_vertices} n_eigs={n_eigs} ml_refine_max_n={ml_max} "
          f"multidepth={ck.get('trained_multidepth')} params={ck['param_count']:,})  "
          f"stochastic={args.stochastic}", flush=True)
    if ck["k"] != args.k:
        print(f"WARNING: checkpoint k={ck['k']} != --k {args.k}", flush=True)

    files = sorted(glob.glob(os.path.join(args.graph_dir, "*.graph")))
    if not files:
        raise SystemExit(f"no .graph files in {args.graph_dir}")
    if args.max_n:
        def peek_n(p):
            with open(p) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln and not ln.startswith("%"):
                        return int(ln.split()[0])
            return 0
        kept = [f for f in files if peek_n(f) <= args.max_n]
        print(f"size filter: {len(kept)}/{len(files)} graphs have n <= {args.max_n} "
              f"(skipped {len(files) - len(kept)} larger)", flush=True)
        files = kept
    if args.limit:
        files = files[:args.limit]
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        files = files[si::sn]
        print(f"shard {si}/{sn}: {len(files)} graph(s) this task", flush=True)
    if not files:
        print("no graphs for this task (filtered/sharded out); nothing to do.", flush=True)
        return

    metis_ok = args.metis
    n_metis_seeds = args.metis_seeds if args.metis_seeds is not None else max(1, args.seeds)
    metis_seeds = tuple(range(1, n_metis_seeds + 1))
    if args.metis:
        try:
            import pymetis  # noqa
        except Exception as e:
            print(f"NOTE: --metis requested but pymetis unavailable ({e}); skipping METIS column.", flush=True)
            metis_ok = False

    done = set()
    out_f = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        if args.resume and os.path.exists(args.output):
            for ln in open(args.output):
                ln = ln.strip()
                if ln:
                    try:
                        done.add(json.loads(ln)["graph"])
                    except Exception:
                        pass
            print(f"resume: {len(done)} graph(s) already in {args.output}; will append.", flush=True)
        out_f = open(args.output, "a" if (args.resume and done) else "w")

    print(f"\n{'graph':22s} {'n':>8} {'m':>9} {'best':>11} {'med':>10} {'imb':>6} {'mlref':>6}"
          + (f" {'metis':>10} {'r_med':>10} {'r_best':>10}" if metis_ok else ""), flush=True)
    print("-" * (86 + (33 if metis_ok else 0)), flush=True)

    ratios_med, ratios_best = [], []
    for f in files:
        name = os.path.basename(f).replace(".graph", "")
        if name in done:
            print(f"{name:22s}  already done (resume) -- skipping", flush=True)
            continue
        G = read_metis(f)
        cuts, imbs, times, refiner_calls, ml_calls, ml_tries, ml_global = [], [], [], 0, 0, 0, 0
        for s in range(max(1, args.seeds)):
            P = make_multilevel_partitioner(policy, k=args.k, epsilon=args.epsilon,
                                       best_of_k=args.best_of_k, refiner_steps=args.refiner_steps,
                                       refiner_stochastic=args.stochastic, n_eigs=n_eigs,
                                       min_vertices=min_vertices, ml_refine_max_n=ml_max,
                                       global_no_harm=args.global_no_harm,
                                       lean=args.lean, force_rl=args.force_rl)
            t0 = time.time()
            part = P.partition(G, seed=s)
            times.append(time.time() - t0)
            cuts.append(float(compute_edge_cut(G, part)))
            imbs.append(float(compute_imbalance(G, part, args.k)))
            refiner_calls += int(P.rl_trainer.calls)   # ALL refine() calls (coarsest + multi-level)
            ml_calls += int(P.ml_refine_calls); ml_tries += int(P.ml_refine_tries)
            ml_global += int(P.ml_global_adopted)      # seeds where multi-level won the global guard
        best_cut, med_cut = float(np.min(cuts)), float(np.median(cuts))
        rec = {"graph": name, "n": int(G.n), "m": int(G.m), "k": args.k, "epsilon": args.epsilon,
               "best_cut": best_cut, "median_cut": med_cut,
               "imbalance_median": float(np.median(imbs)),
               "time_median": float(np.median(times)), "seeds": args.seeds,
               "refiner_calls_total": refiner_calls,
               "ml_refine_tries": ml_tries, "ml_refine_calls": ml_calls,
               "ml_global_adopted": ml_global}
        line = (f"{name:22s} {G.n:>8} {G.m:>9} {best_cut:>11.0f} {med_cut:>10.0f} "
                f"{np.median(imbs):>6.3f} {str(ml_calls)+'/'+str(ml_tries):>6}")
        if metis_ok:
            # trials- AND seed-matched to the method: best-of-best_of_k within a seed, median across seeds
            m = run_metis(G, args.k, ncuts=args.best_of_k, seeds=metis_seeds)
            rec["metis_seeds"] = m["seeds"]; rec["metis_cuts"] = m["cuts"]
            rec["metis_cut"] = m["cut"]; rec["metis_imbalance"] = m["imbalance"]
            # Two protocols, printed side by side and never conflated: r_med pairs with the
            # med column (the primary, seed-median protocol) and r_best with best.
            # Previously only best_cut/metis was emitted, under the unqualified header
            # "e04/metis", so a reader pairing it with the median cut could not reproduce it
            # and the run summary reported the more favourable protocol as the headline.
            ok = m["cut"] > 0
            ratio_med = med_cut / m["cut"] if ok else float("nan")
            ratio_best = best_cut / m["cut"] if ok else float("nan")
            rec["over_metis_median"] = ratio_med
            rec["over_metis_best"] = ratio_best
            rec["over_metis"] = ratio_best   # retained (== *_best) for existing readers
            ratios_med.append(ratio_med); ratios_best.append(ratio_best)
            line += f" {m['cut']:>10.0f} {ratio_med:>10.3f} {ratio_best:>10.3f}"
        print(line, flush=True)
        if out_f:
            out_f.write(json.dumps(rec) + "\n"); out_f.flush()

    if out_f:
        out_f.close()
        print(f"\nwrote {len(files)} records -> {args.output}", flush=True)
    if metis_ok and ratios_med:
        for tag, arr in (("seed-median (primary)", np.array(ratios_med)),
                         ("best-of-seeds", np.array(ratios_best))):
            good = arr[np.isfinite(arr) & (arr > 0)]
            gm = float(np.exp(np.mean(np.log(good)))) if len(good) else float("nan")
            print(f"\nmethod / METIS edge-cut ratio, {tag} protocol (lower=better): "
                  f"median {np.median(arr):.4f}  gmean {gm:.4f}  "
                  f"win-rate(<1.0) {float(np.mean(arr < 1.0)):.1%}  over {len(arr)} graphs",
                  flush=True)
        print("(mlref column = level refinements adopted/tried)",
              flush=True)


if __name__ == "__main__":
    main()
