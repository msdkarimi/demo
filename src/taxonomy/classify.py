import json
import re
import numpy as np
import pandas as pd
import duckdb
from taxonomy.embed import load_texts, ART
from taxonomy.task import detect, key as task_key
from sklearn.preprocessing import Normalizer, normalize


def keyword_pattern(k: str) -> str:
    """Literal keyword, matched as a whole word wherever it starts/ends with a letter
    or digit, so "api" skips "capital" while "c++" and "```" still match."""
    return (r"\b" if k[0].isalnum() else "") + re.escape(k) + (r"\b" if k[-1].isalnum() else "")


def label(ids, X, texts, centroids, tax, min_sim=0.3, margin=0.05, max_labels=3,
          tasks=None, full_texts=None) -> pd.DataFrame:

    tasks = tasks or {}
    task_map, protect = tasks.get("map", {}), set(tasks.get("protect", []))
    index = {c["name"]: j for j, c in enumerate(tax)}
    full_texts = texts if full_texts is None else full_texts
    # A category scores as its closest member cluster. Averaging the centroids
    # instead lands between distant clusters (Python vs web code) and pushes
    # their members under the floor.
    C = X @ centroids.T  # cosine similarity, n x clusters
    S = np.stack([C[:, c["clusters"]].max(axis=1) for c in tax], axis=1)  # n x categories
    rules = [re.compile("|".join(map(keyword_pattern, c["keywords"])), re.I) if c["keywords"] else None
             for c in tax]

    rows = []
    for i, cid in enumerate(ids):
        order = np.argsort(-S[i])
        best = S[i, order[0]]
        labels = {}
        for j in order[: max_labels]:
            # multi-label: keep categories close to the best one and above a floor
            if S[i, j] >= min_sim and S[i, j] >= best - margin:
                labels[j] = (float(S[i, j]), "embedding")
        has_rule = False
        for j, rx in enumerate(rules):  # rules override: high precision, explicit
            if rx and rx.search(texts[i]):
                labels[j] = (1.0, "rule")
                has_rule = True

        # Task: the embedding sees the topic, the taxonomy is defined by the task.
        t = task_key(*detect(full_texts[i]))
        target = task_map.get(t)
        if target and not has_rule:
            name, unless = (target, ()) if isinstance(target, str) else (target["category"], target.get("unless", ()))
            current = tax[max(labels, key=lambda j: labels[j][0])]["name"] if labels else None
            if current not in protect and current not in unless:
                labels[index[name]] = (1.0, "task")

        if not labels:
            rows.append((cid, "Needs review", float(best), "low_confidence", True, t))
            continue
        primary = max(labels, key=lambda j: labels[j][0])
        for j, (score, src) in labels.items():
            rows.append((cid, tax[j]["name"], score, src, j == primary, t))

    return pd.DataFrame(rows, columns=["conversation_id", "category", "score", "source", "is_primary", "task"])


def classify(args):
    taxonomy = json.loads((ART / "taxonomy.json").read_text())
    tax = taxonomy["categories"]
    X = np.load(ART / "embeddings.npy")
    centroids = np.load(ART / "centroids.npy")
    ids = pd.read_parquet(ART / "embedding_ids.parquet")["conversation_id"].tolist()

    con = duckdb.connect(args.db)
    df = load_texts(con).set_index("conversation_id").loc[ids]

    out = label(ids, X, df["text"].tolist(), centroids, tax, args.min_sim, args.margin, args.max_labels,
                tasks=taxonomy.get("tasks"), full_texts=df["first_user_msg"].fillna("").tolist())
    con.execute("CREATE OR REPLACE TABLE conversation_topics AS SELECT * FROM out")
    tax_df = pd.DataFrame([{k: v for k, v in c.items() if k != "clusters"} | {"keywords": ", ".join(c["keywords"])}
                           for c in tax])
    con.execute("CREATE OR REPLACE TABLE taxonomy AS SELECT * FROM tax_df")

    n_multi = out.groupby("conversation_id").size().gt(1).sum()
    prim = out[out.is_primary]
    print("task (what the user asked to have done):")
    print(prim.task.value_counts().to_string(), "\n")
    print(f"{(prim.source == 'task').sum():,} primary labels set by the task layer")
    print(f"Classified {len(ids):,} conversations; {n_multi:,} multi-label, "
          f"{(out.category == 'Needs review').sum():,} need review.\n")
    print(con.execute("""
        SELECT t.category,
               count(*)                                         AS conversations,
               round(100 * avg(c.likely_failure::INT), 1)       AS pct_likely_failure,
               round(avg(t.score), 2)                           AS avg_confidence
        FROM conversation_topics t JOIN conversations c USING (conversation_id)
        WHERE t.is_primary
        GROUP BY 1 ORDER BY conversations DESC
    """).df().to_string(index=False))
    con.close()
