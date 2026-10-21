
from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import normalize

from taxonomy.embed import load_texts, name_cluster
from taxonomy.explore import projection
from taxonomy.embed import dedup_index
from taxonomy.plots import ART, THEME, apply_theme, frame, pe_stroke, save, shorten, titles

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("taxonomy.plot_embeddings")

# Diverging blue <-> red around a gray midpoint. The midpoint is a step darker
# than the reference neutral so silhouette-0 points stay visible on the surface.
DIVERGING = ["#e34948", "#c3c2b7", "#2a78d6"]


# ---------------------------------------------------------------- data
def load(db: str, dedup: bool):
    X = np.load(ART / "embeddings.npy")
    ids = pd.read_parquet(ART / "embedding_ids.parquet")["conversation_id"].tolist()
    con = duckdb.connect(db, read_only=True)
    texts = load_texts(con).set_index("conversation_id").loc[ids, "text"].tolist()
    con.close()

    if not dedup:
        return X, texts, np.arange(len(X)), np.arange(len(X))
    return X, texts, *dedup_index(texts)


# ---------------------------------------------------------------- one k
def fit_k(U: np.ndarray, k: int, seeds: int, n_init: int):
    runs = [KMeans(n_clusters=k, n_init=n_init, random_state=s).fit(U) for s in range(seeds)]
    ref = min(runs, key=lambda r: r.inertia_)
    others = [r for r in runs if r is not ref]

    ari = [adjusted_rand_score(a.labels_, b.labels_) for a, b in combinations(runs, 2)]

    # Clusterwise stability (Hennig): for each reference cluster, the best
    # Jaccard overlap with any cluster of another seed, averaged over seeds.
    jac = np.zeros(k)
    for o in others:
        C = contingency_matrix(ref.labels_, o.labels_)
        union = C.sum(1)[:, None] + C.sum(0)[None, :] - C
        jac += (C / union).max(axis=1)
    jac /= max(len(others), 1)
    return ref, np.array(ari), jac


def score_k(U, texts_u, ref, jac, k, merge_sim):
    labels = ref.labels_
    sil = silhouette_samples(U, labels, metric="cosine")
    cent = normalize(ref.cluster_centers_)
    S = cent @ cent.T
    np.fill_diagonal(S, -1)
    nearest = S.argmax(axis=1)
    keywords = name_cluster(texts_u, labels, k, top_n=5)

    rows = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        typical = idx[np.argmax(U[idx] @ cent[c])]
        rows.append({"cluster": c, "unique_msgs": len(idx),
                     "silhouette": sil[idx].mean(), "stability": jac[c],
                     "nearest": int(nearest[c]), "nearest_sim": S[c, nearest[c]],
                     "keywords": ", ".join(keywords[c]), "example": shorten(texts_u[typical], 160)})
    clusters = pd.DataFrame(rows)

    pairs = [(a, b, S[a, b]) for a, b in combinations(range(k), 2) if S[a, b] >= merge_sim]
    merges = pd.DataFrame(pairs, columns=["a", "b", "centroid_sim"]).sort_values(
        "centroid_sim", ascending=False)
    merges["a_keywords"] = merges["a"].map(clusters["keywords"])
    merges["b_keywords"] = merges["b"].map(clusters["keywords"])
    return sil, clusters, merges


# ---------------------------------------------------------------- charts
def chart_k(t, P, lab_all, sil_all, clusters, k, ari, out: Path):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": [1.25, 1]})

    # Left: every conversation, coloured by how well it sits in its cluster.
    lim = max(0.05, float(np.quantile(np.abs(sil_all), 0.98)))
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    order = np.argsort(np.abs(sil_all))  # confident points on top
    sc = ax.scatter(P[order, 0], P[order, 1], c=sil_all[order], cmap=cmap,
                    norm=TwoSlopeNorm(0, -lim, lim), s=2.2, linewidths=0, rasterized=True)
    for c in range(k):
        m = lab_all == c
        if m.any():
            ax.annotate(str(c), (np.median(P[m, 0]), np.median(P[m, 1])), ha="center", va="center",
                        fontsize=8, fontweight="bold", color=t["ink"], path_effects=[pe_stroke(t)])
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(sc, ax=ax, shrink=0.45, aspect=18, pad=0.01)
    cb.set_label("silhouette (cosine)", fontsize=9, color=t["ink_secondary"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8.5, colors=t["muted"])
    titles(ax, f"k = {k}: all {len(P):,} conversations",
           f"t-SNE · blue = fits its cluster, red = closer to another cluster · numbers are cluster ids")

    # Right: one dot per cluster. Bottom-left = unstable and poorly separated,
    # the first place to look for merges or clusters to dissolve.
    frame(bx, t, xgrid=True, ygrid=True)
    size = clusters["unique_msgs"].to_numpy()
    bx.scatter(clusters["stability"], clusters["silhouette"], s=18 + 180 * size / size.max(),
               color=t["series"][0], alpha=0.75, edgecolors=t["surface"], linewidths=1.5, zorder=3)
    for _, r in clusters.iterrows():
        bx.annotate(str(int(r.cluster)), (r.stability, r.silhouette), xytext=(5, 3),
                    textcoords="offset points", fontsize=8, color=t["ink_secondary"])
    for v, axis in ((clusters["stability"].median(), bx.axvline),
                    (clusters["silhouette"].median(), bx.axhline)):
        axis(v, color=t["muted"], linewidth=1, linestyle=(0, (3, 3)), zorder=2)
    bx.set_xlabel("stability: mean best Jaccard across seeds")
    bx.set_ylabel("mean silhouette of the cluster")
    titles(bx, "Clusters: separation vs stability",
           f"dot size = unique messages · dashed = medians · seed ARI {ari.mean():.2f}")
    fig.tight_layout()
    save(fig, out / f"map_k{k}.png")
    plt.close(fig)


def chart_selection(t, metrics: pd.DataFrame, out: Path):
    panels = [
        ("silhouette", "Mean silhouette", "higher = clusters better separated", None),
        ("stability_ari", "Stability across seeds (ARI)", "higher = same clusters every run",
         ("stability_ari_min", "stability_ari_max")),
        ("merge_pairs", "Merge-candidate pairs", "centroid pairs above --merge-sim", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    ks = metrics["k"].to_numpy()
    for ax, (col, title, sub, band) in zip(axes, panels):
        frame(ax, t, ygrid=True)
        y = metrics[col].to_numpy()
        if band:
            ax.fill_between(ks, metrics[band[0]], metrics[band[1]], color=t["series"][0],
                            alpha=0.15, linewidth=0)
        ax.plot(ks, y, color=t["series"][0], linewidth=2, zorder=3)
        ax.scatter(ks, y, s=36, color=t["series"][0], edgecolors=t["surface"], linewidths=2, zorder=4)
        if col != "merge_pairs":  # mark the best k for the two "higher is better" metrics
            b = int(np.argmax(y))
            ax.annotate(f"k={ks[b]}", (ks[b], y[b]), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold", color=t["ink"])
        ax.set_xticks(ks)
        ax.set_xlabel("k")
        titles(ax, title, sub)
    fig.tight_layout()
    save(fig, out / "k_selection.png")
    plt.close(fig)


# ---------------------------------------------------------------- entrypoint
def main():
    p = argparse.ArgumentParser(description="plot all embeddings and score k-means for several k")
    p.add_argument("--db", default="data/wildchat.duckdb")
    p.add_argument("--out", default=str(ART / "k_sweep"))
    p.add_argument("--ks", default="10,15,20,25,30,35,40,50,60,80",
                   help="comma-separated k values")
    p.add_argument("--seeds", type=int, default=5, help="k-means runs per k, for stability")
    p.add_argument("--n-init", type=int, default=3, help="k-means restarts inside each run")
    p.add_argument("--merge-sim", type=float, default=0.8,
                   help="centroid cosine at or above which two clusters are flagged to merge")
    p.add_argument("--no-dedup", action="store_true", help="fit on every row, duplicates included")
    p.add_argument("--projection", choices=["tsne", "pca"], default="tsne")
    args = p.parse_args()

    matplotlib.use("Agg")
    t = THEME
    apply_theme(t)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, texts, first, inverse = load(args.db, dedup=not args.no_dedup)
    U, texts_u = X[first], [texts[i] for i in first]
    log.info(f"fitting on {len(U):,} unique messages, assigning all {len(X):,} conversations")
    P = projection(X, args.projection, recompute=False)

    metrics = []
    for k in [int(v) for v in args.ks.split(",")]:
        log.info(f"k={k}: {args.seeds} seeds × {args.n_init} inits")
        ref, ari, jac = fit_k(U, k, args.seeds, args.n_init)
        sil, clusters, merges = score_k(U, texts_u, ref, jac, k, args.merge_sim)

        # Every conversation, duplicates included, takes its unique message's values.
        lab_all, sil_all = ref.labels_[inverse], sil[inverse]
        clusters.insert(2, "conversations", np.bincount(lab_all, minlength=k))
        clusters.round(4).to_csv(out / f"clusters_k{k}.csv", index=False)
        merges.round(4).to_csv(out / f"merge_candidates_k{k}.csv", index=False)
        chart_k(t, P, lab_all, sil_all, clusters, k, ari, out)

        metrics.append({"k": k, "silhouette": sil.mean(),
                        "stability_ari": ari.mean(), "stability_ari_min": ari.min(),
                        "stability_ari_max": ari.max(),
                        "median_cluster_stability": float(np.median(jac)),
                        "unstable_clusters": int((jac < 0.6).sum()),
                        "negative_silhouette_share": float((sil < 0).mean()),
                        "smallest_cluster": int(clusters["unique_msgs"].min()),
                        "merge_pairs": len(merges)})
        log.info(f"  silhouette {sil.mean():.3f} · ARI {ari.mean():.3f} · "
                 f"{len(merges)} merge candidates")

    metrics = pd.DataFrame(metrics)
    metrics.round(4).to_csv(out / "metrics.csv", index=False)
    chart_selection(t, metrics, out)
    print("\n" + metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
