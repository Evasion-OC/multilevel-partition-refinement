#!/usr/bin/env python3
"""Build a social/citation benchmark set (Cora, CiteSeer, Actor, Facebook) as METIS .graph files,
for testing the partitioner on non-mesh held-out graphs (the NeuroCUT general-graph instances).

Each graph is made undirected, self-loops removed, restricted to its largest connected component,
and relabelled 0..n-1 -- matching NeuroCUT's instances (Cora 2485, CiteSeer 2120, Actor 6198 LCC).
Output: data/social/{cora,citeseer,actor,facebook}.graph (standard unweighted METIS).

  python data/make_social_benchmark.py
"""
import os, sys, shutil
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "social"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(HERE, ".pyg_cache")
FACEBOOK_SRC = os.environ.get(
    "FACEBOOK_SRC", os.path.join(HERE, "raw", "facebook_combined.graph"))


def write_metis(G, path):
    """G: nx.Graph with integer nodes; relabel to 0..n-1 and write 1-indexed METIS."""
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering="default")
    n, m = G.number_of_nodes(), G.number_of_edges()
    with open(path, "w") as f:
        f.write(f"{n} {m}\n")
        for i in range(n):
            f.write(" ".join(str(j + 1) for j in sorted(G.neighbors(i))) + "\n")
    return n, m


def lcc_undirected(edge_index):
    """edge_index: torch [2,E] -> nx.Graph, undirected, no self-loops, largest CC."""
    G = nx.Graph()
    src, dst = edge_index.tolist()
    G.add_edges_from(zip(src, dst))
    G.remove_edges_from(nx.selfloop_edges(G))
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()


def main():
    from torch_geometric.datasets import Planetoid, Actor
    specs = [
        ("cora",     lambda: Planetoid(os.path.join(CACHE, "Cora"), "Cora")[0]),
        ("citeseer", lambda: Planetoid(os.path.join(CACHE, "CiteSeer"), "CiteSeer")[0]),
        ("actor",    lambda: Actor(os.path.join(CACHE, "Actor"))[0]),
    ]
    print(f"{'graph':10} {'n':>7} {'m':>8}   (largest connected component)")
    for name, load in specs:
        data = load()
        G = lcc_undirected(data.edge_index)
        n, m = write_metis(G, os.path.join(OUT, f"{name}.graph"))
        print(f"{name:10} {n:>7} {m:>8}")

    # Facebook: reuse the canonical SNAP ego-Facebook combined graph (already METIS).
    if os.path.exists(FACEBOOK_SRC):
        shutil.copy(FACEBOOK_SRC, os.path.join(OUT, "facebook.graph"))
        with open(FACEBOOK_SRC) as f:
            n, m = f.readline().split()[:2]
        print(f"{'facebook':10} {int(n):>7} {int(m):>8}   (SNAP ego-Facebook combined)")
    else:
        print("facebook   MISSING facebook_combined.graph source")

    print(f"\nWritten to {OUT}/")


if __name__ == "__main__":
    main()
