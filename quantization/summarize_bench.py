"""Summarise hpc/Bench_quant.sbatch results: task-level quantization sensitivity + the rewrite.

    python quantization/summarize_bench.py --stem spectral_refiner_k16_eps_0_03

Reads results/bench_quant_<stem>_<tag>.jsonl (tags: fp32, int8_pt, int8_pc, int4_pt, int4_pc,
fp32_rewrite) written by src/benchmark.py and prints two markdown tables:
  1. per scheme: graphs, geometric-mean cut ratio vs METIS (median protocol and best-of protocol),
     wins, mean time per graph;
  2. per graph: median-protocol cut relative to fp32 for every scheme, and the rewrite's cut
     equality + time ratio (same model, reassociated forward).
"""
import argparse, glob, json, math, os

TAGS = ["fp32", "int8_pt", "int8_pc", "int4_pt", "int4_pc", "fp32_rewrite"]

def load(path):
    rows = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line); rows[r["graph"]] = r
    return rows

def gmean(xs):
    xs = [x for x in xs if x and not math.isnan(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="spectral_refiner_k16_eps_0_03")
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    data = {t: load(os.path.join(a.results_dir, f"bench_quant_{a.stem}_{t}.jsonl")) for t in TAGS}
    base = data["fp32"]
    print(f"## Task-level results for {a.stem}\n")
    print("| scheme | graphs | gmean cut/METIS (median protocol) | gmean cut/METIS (best-of) | wins (median < METIS) | mean time/graph (s) |")
    print("|---|---|---|---|---|---|")
    for t in TAGS:
        rows = data[t]
        if not rows:
            print(f"| {t} | (no results yet) | | | | |"); continue
        rm = [r.get("over_metis_median", float("nan")) for r in rows.values()]
        rb = [r.get("over_metis_best", float("nan")) for r in rows.values()]
        wins = sum(1 for x in rm if x and x < 1)
        tm = sum(r.get("time_median", 0) for r in rows.values()) / len(rows)
        print(f"| {t} | {len(rows)} | {gmean(rm):.4f} | {gmean(rb):.4f} | {wins}/{len(rows)} | {tm:.1f} |")
    if not base:
        return
    print("\n### Per graph, median-protocol cut relative to fp32 (1.000 = identical)\n")
    qtags = [t for t in TAGS if t not in ("fp32", "fp32_rewrite") and data[t]]
    head = "| graph | n | fp32 cut | " + " | ".join(qtags) + " | rewrite cut ratio | rewrite time ratio |"
    print(head); print("|" + "---|" * (head.count("|") - 1))
    for g, r0 in sorted(base.items(), key=lambda kv: kv[1]["n"]):
        cells = [g, str(r0["n"]), f"{r0['median_cut']:.0f}"]
        for t in qtags:
            r = data[t].get(g)
            cells.append(f"{r['median_cut'] / r0['median_cut']:.3f}" if r and r0["median_cut"] else "—")
        rw = data["fp32_rewrite"].get(g)
        if rw and r0["median_cut"] and r0.get("time_median"):
            cells += [f"{rw['median_cut'] / r0['median_cut']:.3f}", f"{rw['time_median'] / r0['time_median']:.2f}"]
        else:
            cells += ["—", "—"]
        print("| " + " | ".join(cells) + " |")
    print("\nReading: a quantized column at 1.000 means the cut is unchanged on that graph; >1 worse, <1 better. "
          "The rewrite's cut ratio should be 1.000 (same map) and its time ratio is the pipeline-level speedup "
          "(<1 = faster); the rewrite only touches the spectral mixer, so expect a modest ratio unless the mixer dominates.")

if __name__ == "__main__":
    main()
