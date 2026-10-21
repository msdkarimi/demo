import duckdb
import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer, normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from pathlib import Path
import json
import joblib
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)

ART = Path("artifacts")

def load_texts(con) -> pd.DataFrame:
    df = con.execute("SELECT conversation_id, first_user_msg FROM conversations").df()
    df["text"] = df["first_user_msg"].fillna("").str.slice(0, 1000)
    return df

def dedup_index(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(first, inverse): row of each unique message, and each row's unique index.

    Messages count as the same after lowercasing and collapsing whitespace.
    Pasted templates repeated hundreds of times would otherwise pull whole
    clusters toward themselves, so clustering is fitted on unique messages.
    """
    keys = pd.Series(texts).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    _, first, inverse = np.unique(keys.to_numpy(), return_index=True, return_inverse=True)
    return first, inverse

# ---------------------------------------------------------------- embeddings
def fit_embedder(texts: list[str], method: str):
    """Returns (embeddings, embedder) with L2-normalised rows (dot = cosine)."""
    if method == "st":
        from sentence_transformers import SentenceTransformer

        model_name = "all-MiniLM-L6-v2"  # small, fast, good enough for v0
        model = SentenceTransformer(model_name)
        X = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        return X, {"type": "st", "model": model_name}

    # pipe = make_pipeline(
    #     TfidfVectorizer(stop_words="english", max_features=30_000, ngram_range=(1, 2), min_df=2),
    #     TruncatedSVD(n_components=min(100, max(2, len(texts) // 5)), random_state=42),
    #     Normalizer(),
    # )
    # return pipe.fit_transform(texts), {"type": "tfidf", "pipe": pipe}

def name_cluster(texts, labels, k, top_n=8):
    """Class-based TF-IDF (the BERTopic trick): words frequent in one cluster, rare in others."""
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=30_000)
    counts = cv.fit_transform(texts)
    n = len(texts)
    member = sp.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(k, n))
    M = (member @ counts).astype(float)  # k x vocab
    tf = normalize(M, norm="l1", axis=1)
    idf = np.log(1 + (M.sum() / k) / (np.asarray(M.sum(axis=0)).ravel() + 1e-9))
    scores = tf.multiply(idf).toarray()
    vocab = np.array(cv.get_feature_names_out())
    return [vocab[np.argsort(-scores[c])[:top_n]].tolist() for c in range(k)]

def embed(args):

    con = duckdb.connect(args.db, read_only=True)
    df = load_texts(con)
    con.close()
    texts = df["text"].tolist()
    first, inverse = dedup_index(texts)
    uniq = [texts[i] for i in first]
    logging.info(f"Embedding {len(uniq):,} unique of {len(texts):,} messages ({args.embedder})...")
    emb_u, embedder = fit_embedder(uniq, args.embedder)
    embeddings = emb_u[inverse]  # every conversation, duplicates share their message's vector

    k = min(args.k, len(uniq))
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(emb_u)
    labels_u, labels = km.labels_, km.labels_[inverse]
    centroids = normalize(km.cluster_centers_)
    keywords = name_cluster(uniq, labels_u, k)

    clusters = []
    for c in range(k):
        idx = np.where(labels_u == c)[0]
        closest = idx[np.argsort(-(emb_u[idx] @ centroids[c]))[:8]]  # most typical members
        clusters.append({"id": c, "size": int((labels == c).sum()), "keywords": keywords[c],
                         "examples": [uniq[i] for i in closest],
                         "name": " / ".join(keywords[c][:3]), "description": ""})

    categories = [{"name": c["name"], "description": "", "clusters": [c["id"]],
                       "est_minutes_manual": None} for c in clusters]

    for cat in categories:
        cat.setdefault("keywords", [])  # optional regex rules, fill in during review

    ART.mkdir(exist_ok=True)
    np.save(ART / "embeddings.npy", embeddings)
    np.save(ART / "centroids.npy", centroids)
    df[["conversation_id"]].to_parquet(ART / "embedding_ids.parquet", index=False)
    joblib.dump(embedder, ART / "embedder.joblib")
    (ART / "clusters.json").write_text(json.dumps(clusters, indent=2, ensure_ascii=False))
    (ART / "taxonomy.json").write_text(json.dumps({"categories": categories}, indent=2, ensure_ascii=False))

    print(f"\n{k} clusters -> {len(categories)} categories. Review {ART / 'taxonomy.json'} "
          f"(examples per cluster in {ART / 'clusters.json'}).\n")
    for cat in sorted(categories, key=lambda c: -sum(clusters[i]["size"] for i in c["clusters"])):
        size = sum(clusters[i]["size"] for i in cat["clusters"])
        print(f"  {size:>6}  {cat['name']}")


