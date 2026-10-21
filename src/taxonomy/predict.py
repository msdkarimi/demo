
from __future__ import annotations

import json
import logging

import duckdb
import joblib
import numpy as np
import pandas as pd

from taxonomy.classify import label
from taxonomy.embed import ART, load_texts

log = logging.getLogger("taxonomy.predict")


class Classifier:
    """The current taxonomy as a reusable classifier, loaded from artifacts/."""

    def __init__(self, art=ART, min_sim=0.3, margin=0.05, max_labels=3):
        info = joblib.load(art / "embedder.joblib")
        if info["type"] != "st":
            raise SystemExit(f"predict supports sentence-transformers embedders, not {info['type']!r}")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(info["model"])
        self.centroids = np.load(art / "centroids.npy")
        self.taxonomy = json.loads((art / "taxonomy.json").read_text())
        self.params = {"min_sim": min_sim, "margin": margin, "max_labels": max_labels}

    def predict(self, texts: list[str], ids: list | None = None) -> pd.DataFrame:
        """One row per label: conversation_id, category, score, source, is_primary."""
        # Same input preparation as load_texts, so scores match classify's;
        # the task detector reads the full message, as in classify.
        full = [(t or "")[:2000] for t in texts]
        texts = [t[:1000] for t in full]
        ids = list(range(len(texts))) if ids is None else list(ids)
        X = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                              show_progress_bar=len(texts) > 500)
        return label(ids, X, texts, self.centroids, self.taxonomy["categories"], **self.params,
                     tasks=self.taxonomy.get("tasks"), full_texts=full)


def predict_db(args) -> None:
    con = duckdb.connect(args.db)
    df = load_texts(con)
    clf = Classifier(min_sim=args.min_sim, margin=args.margin, max_labels=args.max_labels)
    log.info(f"predicting {len(df):,} conversations in {args.db} "
             f"(taxonomy {clf.taxonomy.get('version', '?')})")
    out = clf.predict(df["first_user_msg"].fillna("").tolist(), df["conversation_id"].tolist())
    con.execute("CREATE OR REPLACE TABLE conversation_topics AS SELECT * FROM out")
    prim = out[out.is_primary]
    print(prim.category.value_counts().to_string())
    print(f"\n{(prim.source == 'task').sum():,} primary labels set by the task layer")
    con.close()
