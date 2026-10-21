
from __future__ import annotations

import argparse
import json
import logging
import webbrowser
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from taxonomy.plots import ART, load_artifacts

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("taxonomy.explore")


def projection(X: np.ndarray, method: str, recompute: bool, seed: int = 42) -> np.ndarray:
    cache = ART / f"projection_{method}.npy"
    if cache.exists() and not recompute:
        P = np.load(cache)
        if len(P) == len(X):
            log.info(f"using cached {cache}")
            return P

    from sklearn.decomposition import PCA

    if method == "pca":
        P = PCA(n_components=2, random_state=seed).fit_transform(X)
    else:
        from sklearn.manifold import TSNE

        log.info(f"t-SNE on {len(X):,} points — a minute or two")
        pre = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
        P = TSNE(n_components=2, init="pca", perplexity=30, metric="cosine",
                 random_state=seed).fit_transform(pre)
    np.save(cache, P.astype(np.float32))
    return P


def cluster_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """k-means assignment: nearest centroid in Euclidean distance."""
    return np.argmax(2 * X @ centroids.T - (centroids ** 2).sum(axis=1), axis=1)


def build_frame(args) -> tuple[pd.DataFrame, list[dict]]:
    X, centroids, clusters, _, ids = load_artifacts()
    P = projection(X, args.projection, args.recompute)
    lab = cluster_labels(X, centroids)

    con = duckdb.connect(args.db, read_only=True)
    meta = con.execute("""
        SELECT c.conversation_id, c.team, c.model, c.country, c.n_turns,
               c.likely_failure, c.first_user_msg,
               coalesce(t.category, 'Unclassified') AS category
        FROM conversations c
        LEFT JOIN conversation_topics t
          ON t.conversation_id = c.conversation_id AND t.is_primary
    """).df().drop_duplicates("conversation_id").set_index("conversation_id")
    con.close()

    df = meta.loc[ids].reset_index()
    df["x"], df["y"] = P[:, 0], P[:, 1]
    df["cluster"] = [clusters[i]["name"] for i in lab]
    df["failure"] = np.where(df["likely_failure"], "Likely failure", "No failure signal")
    df["text"] = df["first_user_msg"].fillna("").str.replace(r"\s+", " ", regex=True).str.slice(0, args.chars)

    # Label each cluster at the median of its own points, so the label sits on
    # the blob rather than wherever the mean happens to fall between two lobes.
    labels = [{"name": c["name"], "x": float(np.median(P[lab == i, 0])),
               "y": float(np.median(P[lab == i, 1])), "n": int((lab == i).sum())}
              for i, c in enumerate(clusters) if (lab == i).any()]
    return df, labels


DIMS = ("cluster", "category", "team", "model", "failure", "country")


def encode(df: pd.DataFrame, labels: list[dict], method: str) -> str:
    """Columnar JSON: each dimension as a value list + per-point int codes."""
    dims = {}
    for d in DIMS:
        df[d] = df[d].fillna("Unknown")
        counts = df[d].value_counts()
        values = counts.index.tolist()
        index = {v: i for i, v in enumerate(values)}
        dims[d] = {"values": values, "counts": counts.tolist(),
                   "codes": df[d].map(index).astype(int).tolist()}
    return json.dumps({
        "method": "t-SNE (cosine, PCA-50)" if method == "tsne" else "PCA",
        "x": np.round(df["x"].to_numpy(np.float64), 2).tolist(),
        "y": np.round(df["y"].to_numpy(np.float64), 2).tolist(),
        "text": df["text"].tolist(),
        "turns": df["n_turns"].astype(int).tolist(),
        "dims": dims,
        "labels": labels,
    }, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def main():
    p = argparse.ArgumentParser(description="interactive embedding explorer")
    p.add_argument("--db", default="data/wildchat.duckdb")
    p.add_argument("--out", default=str(ART / "plots" / "embedding_explorer.html"))
    p.add_argument("--projection", choices=["tsne", "pca"], default="tsne")
    p.add_argument("--recompute", action="store_true", help="ignore the cached projection")
    p.add_argument("--chars", type=int, default=400, help="message characters kept for hover")
    p.add_argument("--open", action="store_true", help="open the page in a browser")
    args = p.parse_args()

    df, labels = build_frame(args)
    template = (Path(__file__).parent / "explorer.html").read_text()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.replace("/*__DATA__*/null", encode(df, labels, args.projection)))
    log.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(df):,} points)")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
