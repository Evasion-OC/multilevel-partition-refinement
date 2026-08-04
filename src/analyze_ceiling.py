#!/usr/bin/env python3
"""Ceiling-distance analysis: pair per-graph benchmark cuts with the structural descriptors,
then quantify whether the cut ratio against METIS is a function of graph structure.

Inputs (all small text files):
  --descriptors results/descriptors_test_all.csv         (from struct_descriptors.py)
  --jsonl baseline_a=results/baseline_a_k4.jsonl ... (one or more model=path pairs)

Outputs (printed tables + an optional scatter PDF):
  - cut-ratio summary on REAL graphs (median, win-rate) for each model
  - Spearman(ratio, tau / lambda2 / deg_cv) -- the ceiling-distance correlations
  - median ratio per tau bin (binned validation)
  - FALSIFICATION: high-tau IRREGULAR (social, deg_cv>1) should win; high-tau REGULAR
    (expander/d-regular + dense chemistry, deg_cv<0.3) should tie -- tau alone must not predict wins
  - A/B parity: each module vs its baseline, against the baseline_a-vs-baseline_b noise floor

  python src/analyze_ceiling.py --descriptors results/descriptors_test_all.csv \
      --jsonl baseline_a=results/baseline_a_k4.jsonl \
      --jsonl specformer=results/specformer_k4.jsonl --out results/ceiling
"""
import os, csv, json, argparse, math
from collections import defaultdict

CONTROL_PREFIXES = ("expander", "dreg")          # synthetic high-lambda2 controls (not real-graph headline)


def is_control(name):
    return any(name.startswith(p) for p in CONTROL_PREFIXES)


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def spearman(xs, ys):
    """Spearman rho via Pearson on ranks; pairs with a None in either are dropped."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 4:
        return None, n
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals); i = 0
        while i < len(vals):                       # average ties
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx); vy = sum((b - my) ** 2 for b in ry)
    return (cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else None), n


def median(xs):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def load_descriptors(path):
    d = {}
    for row in csv.DictReader(open(path)):
        d[row["graph"]] = {"tau": fnum(row.get("tau")), "lambda2": fnum(row.get("lambda2")),
                           "deg_cv": fnum(row.get("deg_cv")), "family": row.get("family", ""),
                           "planar": row.get("planar", ""), "n": fnum(row.get("n"))}
    return d


def load_jsonl(path):
    cuts = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        ratio = fnum(r.get("over_metis"))
        if ratio is None:
            e, m = fnum(r.get("best_cut")), fnum(r.get("metis_cut"))
            ratio = (e / m) if (e is not None and m) else None
        cuts[r["graph"]] = {"ratio": ratio, "cut": fnum(r.get("best_cut")),
                            "metis": fnum(r.get("metis_cut"))}
    return cuts


TAU_BINS = [(0, 0.05), (0.05, 0.15), (0.15, 0.35), (0.35, 0.7), (0.7, 1e9)]


def analyse_model(tag, cuts, desc):
    rows = []
    for g, c in cuts.items():
        if g not in desc or c["ratio"] is None:
            continue
        rows.append((g, c["ratio"], desc[g], is_control(g)))
    real = [r for r in rows if not r[3]]
    ctrl = [r for r in rows if r[3]]
    print(f"\n===== {tag}: {len(rows)} graphs ({len(real)} real, {len(ctrl)} control) =====")
    rr = [r[1] for r in real]
    print(f"  REAL graphs: median ratio = {median(rr):.3f} | win-rate (ratio<0.99) = "
          f"{sum(1 for x in rr if x < 0.99)}/{len(rr)} | tie (0.99-1.01) = "
          f"{sum(1 for x in rr if 0.99 <= x <= 1.01)} | loss = {sum(1 for x in rr if x > 1.01)}")
    # ceiling-distance correlations (real only)
    for axis in ("tau", "lambda2", "deg_cv"):
        rho, n = spearman([r[2][axis] for r in real], [r[1] for r in real])
        print(f"  Spearman(ratio, {axis:7}) = {rho:+.3f}  (n={n})" if rho is not None
              else f"  Spearman(ratio, {axis:7}) = n/a")
    # binned median ratio by tau (real only)
    print("  median ratio by tau bin (real):")
    for lo, hi in TAU_BINS:
        b = [r[1] for r in real if r[2]["tau"] is not None and lo <= r[2]["tau"] < hi]
        if b:
            print(f"    tau in [{lo:>4}, {hi:>4}): n={len(b):>3}  median ratio = {median(b):.3f}")
    # FALSIFICATION: high-tau irregular (win) vs high-tau regular (tie)
    hi_irr = [r[1] for r in real if r[2]["tau"] and r[2]["tau"] > 0.5 and r[2]["deg_cv"] and r[2]["deg_cv"] > 1.0]
    hi_reg = [r[1] for r in rows if r[2]["tau"] and r[2]["tau"] > 0.5 and r[2]["deg_cv"] is not None and r[2]["deg_cv"] < 0.3]
    print(f"  FALSIFICATION (tau>0.5): irregular deg_cv>1 -> median {median(hi_irr) if hi_irr else float('nan'):.3f} "
          f"(n={len(hi_irr)}, expect <1 WIN) | regular deg_cv<0.3 -> median "
          f"{median(hi_reg) if hi_reg else float('nan'):.3f} (n={len(hi_reg)}, expect ~1 TIE)")
    return {g: c["ratio"] for g, c in cuts.items() if c["ratio"] is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptors", required=True)
    ap.add_argument("--jsonl", action="append", default=[], help="tag=path (repeatable)")
    ap.add_argument("--out", default="results/ceiling")
    a = ap.parse_args()
    desc = load_descriptors(a.descriptors)
    models = {}
    for spec in a.jsonl:
        tag, path = spec.split("=", 1)
        models[tag] = load_jsonl(path)
    ratios_by_model = {tag: analyse_model(tag, cuts, desc) for tag, cuts in models.items()}

    # A/B parity: module vs its baseline, against the baseline_a-vs-baseline_b noise floor
    if "baseline_a" in ratios_by_model and "baseline_b" in ratios_by_model:
        a_, b_ = ratios_by_model["baseline_a"], ratios_by_model["baseline_b"]
        common = [g for g in a_ if g in b_ and not is_control(g)]
        noise = median([abs(a_[g] - b_[g]) for g in common])
        print(f"\n===== A/B parity (real graphs) =====")
        print(f"  noise floor |baseline_a - baseline_b| median = {noise:.4f}  (n={len(common)})")
        for tag in ("lanczos", "specformer"):
            if tag in ratios_by_model:
                m = ratios_by_model[tag]
                cg = [g for g in m if g in a_ and not is_control(g)]
                dmed = median([m[g] - a_[g] for g in cg])
                within = sum(1 for g in cg if abs(m[g] - a_[g]) <= max(noise, 0.005))
                print(f"  {tag:10} vs baseline_a: median delta-ratio = {dmed:+.4f} | "
                      f"within noise floor {within}/{len(cg)}  (parity if ~all)")

    # optional scatter
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        for tag, m in ratios_by_model.items():
            xs = [desc[g]["tau"] for g in m if g in desc and desc[g]["tau"]]
            ys = [m[g] for g in m if g in desc and desc[g]["tau"]]
            cs = ["crimson" if is_control(g) else "royalblue" for g in m if g in desc and desc[g]["tau"]]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
            ax.scatter(xs, ys, c=cs, s=14, alpha=0.7, linewidths=0)
            ax.set_xscale("log"); ax.set_xlabel(r"$\tau = \mathrm{degen}/\sqrt{n}$")
            ax.set_ylabel(r"$c_{\mathrm{method}}/c_{\mathrm{METIS}}$"); ax.set_title(f"{tag}: cut ratio vs structure")
            fig.tight_layout(); fig.savefig(f"{a.out}_{tag}_ratio_vs_tau.pdf"); plt.close(fig)
        print(f"\nwrote scatter PDFs -> {a.out}_<model>_ratio_vs_tau.pdf")
    except Exception as e:
        print(f"\n(scatter skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
