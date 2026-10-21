
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("taxonomy.plots")

ART = Path("artifacts")

# ---------------------------------------------------------------- palette
# Validated categorical slots (see dataviz reference palette). Only the first
# two slots are used anywhere here; everything else is single-hue or sequential.
THEME = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": ("#2a78d6", "#eb6834"),
    "critical": "#d03b3b",
    # sequential blue, light -> dark
    "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
}


def apply_theme(t: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "font.family": ["DejaVu Sans"],
        "font.size": 10,
        "text.color": t["ink"],
        "axes.labelcolor": t["ink_secondary"],
        "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink_secondary"],
        "ytick.labelcolor": t["ink_secondary"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 160,
    })


def frame(ax, t: dict, *, xgrid=False, ygrid=False) -> None:
    """Recessive chrome: hairline grid on one axis only, no box."""
    ax.set_axisbelow(True)
    if xgrid:
        ax.xaxis.grid(True, color=t["grid"], linewidth=1, linestyle="-")
    if ygrid:
        ax.yaxis.grid(True, color=t["grid"], linewidth=1, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1)


def titles(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=plt.rcParams["axes.labelcolor"], va="bottom")


def rounded_barh(ax, y, widths, *, height, color, radius_px=4.0):
    """Horizontal bars with a rounded data-end, square at the baseline.

    The radius has to be circular in *pixels*, so the cap is an Ellipse whose
    data-space width and height are both converted from the same pixel radius.
    Call once the axes limits are final.
    """
    ax.figure.canvas.draw()  # need the rendered box for the px-per-data scale
    bbox = ax.get_window_extent()
    xs, ys = ax.get_xlim(), ax.get_ylim()
    px_per_x = bbox.width / (xs[1] - xs[0])
    px_per_y = bbox.height / (ys[1] - ys[0])

    # Never let the cap exceed half the bar thickness, or it stops being a bar.
    r_px = min(radius_px, height * px_per_y / 2)
    rx, ry = r_px / px_per_x, r_px / px_per_y

    for yi, w in zip(y, widths):
        if w <= 0:
            continue
        body = max(w - rx, 0)
        # Body (square at the baseline), then the two right-hand corner rounds
        # with the strip between them.
        ax.add_patch(plt.Rectangle((0, yi - height / 2), body, height,
                          linewidth=0, facecolor=color))
        if body > 0 and height > 2 * ry:
            ax.add_patch(plt.Rectangle((body, yi - height / 2 + ry), rx, height - 2 * ry,
                              linewidth=0, facecolor=color))
            for cy in (yi + height / 2 - ry, yi - height / 2 + ry):
                ax.add_patch(Ellipse((body, cy), 2 * rx, 2 * ry, linewidth=0, facecolor=color))


def compact(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n / 1_000:.1f}K".replace(".0K", "K")
    return f"{n:,.0f}"


def shorten(s: str, n: int = 34) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def fold_other(df: pd.DataFrame, label_col: str, value_col: str, top: int):
    """Top-N rows by value, the rest summed into one 'Other' row."""
    df = df.sort_values(value_col, ascending=False)
    head, tail = df.iloc[:top], df.iloc[top:]
    if tail.empty:
        return head, 0
    other = pd.DataFrame({label_col: [f"Other ({len(tail)} more)"],
                          value_col: [tail[value_col].sum()]})
    return pd.concat([head, other], ignore_index=True), len(tail)


# ---------------------------------------------------------------- loaders
def load_artifacts() -> tuple[np.ndarray, np.ndarray, list[dict], list[dict], list[str]]:
    missing = [p for p in ("embeddings.npy", "centroids.npy", "clusters.json",
                           "taxonomy.json", "embedding_ids.parquet")
               if not (ART / p).exists()]
    if missing:
        raise SystemExit(f"missing artifacts: {', '.join(missing)} — run `discover` first")
    X = np.load(ART / "embeddings.npy")
    centroids = np.load(ART / "centroids.npy")
    clusters = json.loads((ART / "clusters.json").read_text())
    tax = json.loads((ART / "taxonomy.json").read_text())["categories"]
    ids = pd.read_parquet(ART / "embedding_ids.parquet")["conversation_id"].tolist()
    return X, centroids, clusters, tax, ids


def project(X: np.ndarray, centroids: np.ndarray, method: str, seed: int = 42):
    """2D projection of the embedding space, with centroids in the same frame."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=seed).fit(X)
    P, Pc = pca.transform(X), pca.transform(centroids)
    if method == "pca":
        return P, Pc, f"PCA ({100 * pca.explained_variance_ratio_.sum():.0f}% of variance)"

    # t-SNE on a PCA-50 pre-reduction, the usual recipe; centroids ride along so
    # they land in the same frame rather than being transformed separately.
    from sklearn.manifold import TSNE

    pre = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit(X)
    Z = np.vstack([pre.transform(X), pre.transform(centroids)])
    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(Z)
    return emb[: len(X)], emb[len(X):], "t-SNE of PCA-50"


# ---------------------------------------------------------------- charts
def chart_cluster_sizes(t, clusters, out: Path, top=15):
    df = pd.DataFrame([{"name": c["name"], "size": c["size"]} for c in clusters])
    df, n_folded = fold_other(df, "name", "size", top)
    df = df.iloc[::-1]  # largest at the top of a barh

    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(df) + 1.9))
    y = np.arange(len(df))
    ax.set_xlim(0, df["size"].max() * 1.14)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_yticks(y, [shorten(s) for s in df["name"]])
    frame(ax, t, xgrid=True)
    rounded_barh(ax, y, df["size"].to_numpy(), height=0.62, color=t["series"][0])

    # Value at the tip — one label per bar is the whole point of a ranked bar.
    for yi, w in zip(y, df["size"]):
        ax.text(w + df["size"].max() * 0.015, yi, compact(w), va="center", fontsize=9,
                color=t["ink_secondary"])
    ax.xaxis.set_major_formatter(lambda v, _: compact(v))
    titles(ax, "Conversations per cluster",
           f"{len(clusters)} k-means clusters" + (f", smallest {n_folded} folded into Other" if n_folded else ""))
    save(fig, out / "cluster_sizes.png")


def chart_map(t, X, centroids, clusters, out: Path, method: str):
    P, Pc, how = project(X, centroids, method)
    cmap = LinearSegmentedColormap.from_list("seq_blue", t["ramp"])

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.margins(0.09)
    hb = ax.hexbin(P[:, 0], P[:, 1], gridsize=58, cmap=cmap, mincnt=1, linewidths=0)
    frame(ax, t)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    cb = fig.colorbar(hb, ax=ax, shrink=0.42, aspect=18, pad=0.02)
    cb.set_label("conversations per cell", fontsize=9, color=t["ink_secondary"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8.5, colors=t["muted"])

    # Density alone has no identity channel, so name the biggest regions in ink.
    order = np.argsort([-c["size"] for c in clusters])[:8]
    fig.canvas.draw()
    placed = []  # display coords of labels so far; near-coincident centroids stack downward
    line = 1.4 * 8.5 * fig.dpi / 72  # one label height, in display pixels
    for i in order:
        x, y = ax.transData.transform(Pc[i])
        dy = 0
        while any(abs(x - px) < 160 * fig.dpi / 72 and abs(y - dy - py) < line for px, py in placed):
            dy += line
        placed.append((x, y - dy))
        ax.annotate(shorten(clusters[i]["name"], 26), Pc[i], xytext=(0, -dy * 72 / fig.dpi),
                    textcoords="offset points", fontsize=8.5, ha="center",
                    color=t["ink"],
                    path_effects=[pe_stroke(t)],
                    zorder=5)

    titles(ax, "Where the conversations sit in embedding space",
           f"{len(X):,} first user messages · {how} · 8 largest clusters labelled")
    save(fig, out / "embedding_map.png")


def pe_stroke(t):
    """A surface-colored halo so ink labels stay legible over dense hexes."""
    from matplotlib import patheffects
    return patheffects.withStroke(linewidth=2.6, foreground=t["surface"])


def chart_category_volume(t, con, out: Path, top=15):
    df = con.execute("""
        SELECT category, count(*) AS conversations
        FROM conversation_topics WHERE is_primary
        GROUP BY 1 ORDER BY conversations DESC
    """).df()
    review = df.loc[df.category == "Needs review", "conversations"].sum()
    df, n_folded = fold_other(df, "category", "conversations", top)
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(df) + 1.9))
    y = np.arange(len(df))
    ax.set_xlim(0, df["conversations"].max() * 1.14)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_yticks(y, [shorten(s) for s in df["category"]])
    frame(ax, t, xgrid=True)

    # "Needs review" is a state, not a series — status color, and it keeps its
    # own axis label so hue is never carrying the meaning alone.
    for yi, (name, w) in enumerate(zip(df["category"], df["conversations"])):
        color = t["critical"] if name == "Needs review" else t["series"][0]
        rounded_barh(ax, [yi], [w], height=0.62, color=color)
        ax.text(w + df["conversations"].max() * 0.015, yi, compact(w), va="center",
                fontsize=9, color=t["ink_secondary"])
    ax.xaxis.set_major_formatter(lambda v, _: compact(v))
    total = int(con.execute("SELECT count(DISTINCT conversation_id) FROM conversation_topics").fetchone()[0])
    titles(ax, "Primary category by volume",
           f"{total:,} conversations · {review:,} ({100 * review / max(total, 1):.1f}%) below the "
           f"similarity floor, shown in red as Needs review"
           + (f" · {n_folded} smaller categories folded into Other" if n_folded else ""))
    save(fig, out / "category_volume.png")


def chart_confidence(t, con, out: Path, min_sim: float):
    df = con.execute("""
        SELECT score, category = 'Needs review' AS needs_review
        FROM conversation_topics WHERE is_primary
    """).df()
    lo = df.loc[df.needs_review, "score"].to_numpy()
    hi = df.loc[~df.needs_review, "score"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bins = np.linspace(0, max(df["score"].max(), min_sim) * 1.02, 56)
    ax.hist([hi, lo], bins=bins, stacked=True, color=[t["series"][0], t["critical"]],
            label=["Classified", "Needs review"],
            edgecolor=t["surface"], linewidth=0.8)
    frame(ax, t, ygrid=True)
    ax.axvline(min_sim, color=t["ink_secondary"], linewidth=1.4, linestyle="-", zorder=4)
    ax.annotate(f"similarity floor {min_sim:g}", (min_sim, ax.get_ylim()[1] * 0.94),
                xytext=(6, 0), textcoords="offset points", fontsize=9, color=t["ink_secondary"])
    ax.set_xlabel("cosine similarity to the assigned category centroid")
    ax.set_ylabel("conversations")
    ax.yaxis.set_major_formatter(lambda v, _: compact(v))
    ax.legend(frameon=False, loc="upper right", fontsize=9.5,
              labelcolor=t["ink_secondary"])
    titles(ax, "How confident the assignments are",
           f"median {np.median(df['score']):.2f} · {len(lo):,} of {len(df):,} fall below the floor")
    save(fig, out / "confidence.png")


def chart_failure_rate(t, con, out: Path, top=15, min_n=50):
    df = con.execute(f"""
        SELECT t.category,
               count(*)                                AS conversations,
               100 * avg(c.likely_failure::INT)        AS pct_likely_failure
        FROM conversation_topics t JOIN conversations c USING (conversation_id)
        WHERE t.is_primary
        GROUP BY 1 HAVING count(*) >= {min_n}
        ORDER BY pct_likely_failure DESC
        LIMIT {top}
    """).df()
    overall = float(con.execute(
        "SELECT 100 * avg(likely_failure::INT) FROM conversations").fetchone()[0])
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.6, 0.34 * len(df) + 2.1))
    y = np.arange(len(df))
    ax.set_xlim(0, max(df["pct_likely_failure"].max(), overall) * 1.2)
    # Extra headroom at the top so the reference-line label sits in clear space
    # instead of over the longest bar.
    ax.set_ylim(-0.6, len(df) + 0.1)
    ax.set_yticks(y, [shorten(f"{c}  (n={compact(n)})", 40)
                      for c, n in zip(df["category"], df["conversations"])])
    frame(ax, t, xgrid=True)
    rounded_barh(ax, y, df["pct_likely_failure"].to_numpy(), height=0.62, color=t["series"][0])
    for yi, w in zip(y, df["pct_likely_failure"]):
        ax.text(w + ax.get_xlim()[1] * 0.012, yi, f"{w:.1f}%", va="center", fontsize=9,
                color=t["ink_secondary"])
    ax.axvline(overall, color=t["ink_secondary"], linewidth=1.4, zorder=4)
    ax.annotate(f"all conversations {overall:.1f}%", (overall, len(df) - 0.25),
                xytext=(6, 0), textcoords="offset points", fontsize=9,
                color=t["ink_secondary"], va="center", zorder=6)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    titles(ax, "Likely-failure rate by category",
           f"baseline heuristic (rephrase + correction, no thanks) · categories with n >= {min_n}")
    save(fig, out / "failure_by_category.png")


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    log.info(f"wrote {path}")


# ---------------------------------------------------------------- entrypoint
CHARTS = ("clusters", "map", "volume", "confidence", "failure")


def main():
    p = argparse.ArgumentParser(description="render the taxonomy charts")
    p.add_argument("--db", default="data/wildchat.duckdb")
    p.add_argument("--out", default=str(ART / "plots"))
    p.add_argument("--only", choices=CHARTS, action="append",
                   help="render just this chart (repeatable); default is all")
    p.add_argument("--projection", choices=["pca", "tsne"], default="pca",
                   help="tsne separates better but takes minutes at 20k rows")
    p.add_argument("--min-sim", type=float, default=0.3,
                   help="the floor classify used, drawn as a reference line")
    p.add_argument("--top", type=int, default=15, help="bars before folding into Other")
    p.add_argument("--show", action="store_true", help="also open the figures interactively")
    args = p.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    t = THEME
    apply_theme(t)
    want = set(args.only or CHARTS)
    out = Path(args.out)

    X, centroids, clusters, tax, ids = load_artifacts()
    if "clusters" in want:
        chart_cluster_sizes(t, clusters, out, top=args.top)
    if "map" in want:
        chart_map(t, X, centroids, clusters, out, args.projection)

    if want & {"volume", "confidence", "failure"}:
        con = duckdb.connect(args.db, read_only=True)
        have = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "conversation_topics" not in have:
            log.warning("no conversation_topics table — run `classify` for the remaining charts")
        else:
            if "volume" in want:
                chart_category_volume(t, con, out, top=args.top)
            if "confidence" in want:
                chart_confidence(t, con, out, args.min_sim)
            if "failure" in want:
                chart_failure_rate(t, con, out, top=args.top)
        con.close()

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
