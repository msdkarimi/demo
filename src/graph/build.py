
from __future__ import annotations

import argparse
import json
import logging
import time

import duckdb
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from graph import config
from graph.entities import extract
from taxonomy.embed import ART, dedup_index
from taxonomy.explore import cluster_labels

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("graph.build")
logging.getLogger("neo4j").setLevel(logging.WARNING)

NEEDS_REVIEW = "Needs review"


def connect(wait: float = 120):
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    deadline = time.time() + wait
    while True:
        try:
            driver.verify_connectivity()
            return driver
        except ServiceUnavailable:
            if time.time() > deadline:
                raise SystemExit(f"Neo4j not reachable at {config.NEO4J_URI}: run `docker compose up -d`")
            time.sleep(2)


def run(driver, query: str, **params):
    return driver.execute_query(query, params, database_=config.NEO4J_DATABASE).records


def write(driver, query: str, rows: list[dict], what: str, batch: int = 2000) -> None:
    for i in range(0, len(rows), batch):
        run(driver, query, rows=rows[i:i + batch])
    log.info(f"  {what}: {len(rows):,}")


# ---------------------------------------------------------------- schema
def reset(driver) -> None:
    log.info("wiping database")
    with driver.session(database=config.NEO4J_DATABASE) as s:
        s.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS").consume()
    for r in run(driver, "SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name"):
        run(driver, f"DROP INDEX {r['name']}")


def schema(driver) -> None:
    for label, key in [("Conversation", "id"), ("User", "id"), ("Intent", "name"), ("Topic", "id"),
                       ("Task", "key"), ("Entity", "name"), ("Model", "name"), ("Team", "name")]:
        run(driver, f"CREATE CONSTRAINT {label.lower()}_{key} IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE")
    run(driver, "CREATE INDEX conversation_failure IF NOT EXISTS FOR (c:Conversation) ON (c.likely_failure)")


def vector_index(driver, dim: int) -> None:
    run(driver, f"""
        CREATE VECTOR INDEX {config.VECTOR_INDEX} IF NOT EXISTS
        FOR (c:Conversation) ON c.embedding
        OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}""")
    run(driver, "CALL db.awaitIndexes(600)")


# ---------------------------------------------------------------- data
def similar_pairs(X, texts, k: int, min_score: float, chunk: int = 2048):
    """Top-k nearest *distinct* messages per conversation (exact repeats skipped)."""
    first, inverse = dedup_index(texts)
    U = X[first]
    nbr = np.full((len(U), k), -1)
    sim = np.zeros((len(U), k), dtype=np.float32)
    for a in range(0, len(U), chunk):
        S = U[a:a + chunk] @ U.T
        S[np.arange(len(S)), np.arange(a, a + len(S))] = -1  # not yourself
        top = np.argpartition(-S, k, axis=1)[:, :k]
        val = np.take_along_axis(S, top, axis=1)
        order = np.argsort(-val, axis=1)
        nbr[a:a + chunk] = np.take_along_axis(top, order, axis=1)
        sim[a:a + chunk] = np.take_along_axis(val, order, axis=1)
    return [(i, int(first[v]), float(s)) for i, u in enumerate(inverse)
            for v, s in zip(nbr[u], sim[u]) if s >= min_score]


def load(args):
    X = np.load(ART / "embeddings.npy")
    C = np.load(ART / "centroids.npy")
    ids = pd.read_parquet(ART / "embedding_ids.parquet")["conversation_id"].tolist()
    clusters = json.loads((ART / "clusters.json").read_text())
    tax = json.loads((ART / "taxonomy.json").read_text())

    con = duckdb.connect(args.db, read_only=True)
    conv = con.execute("SELECT * FROM conversations").df().set_index("conversation_id").loc[ids].reset_index()
    topics = con.execute("SELECT * FROM conversation_topics").df()
    turns = con.execute(f"SELECT conversation_id, idx, role, content FROM turns WHERE idx < {args.max_turns}").df()
    con.close()

    lab = cluster_labels(X, C)
    topic_sim = (X @ C.T)[np.arange(len(X)), lab]
    names = {c["id"]: c["name"] for c in clusters}
    prim = topics[topics["is_primary"]].set_index("conversation_id")
    conv["intent"] = conv["conversation_id"].map(prim["category"])
    conv["task"] = conv["conversation_id"].map(prim["task"]).fillna("answer")
    conv["topic_id"] = lab
    conv["sensitive"] = [names[t] in config.SENSITIVE_TOPICS or bool(config.SENSITIVE_RE.search(m or ""))
                         for t, m in zip(lab, conv["first_user_msg"])]
    texts = conv["first_user_msg"].fillna("").str.slice(0, 1000).tolist()
    return dict(X=X, conv=conv, topics=topics, turns=turns, clusters=clusters, tax=tax,
                topic_sim=topic_sim, texts=texts)


def build(driver, d, args) -> None:
    conv, tax = d["conv"], d["tax"]
    preview = lambda t, n: " ".join((t or "").split())[:n]

    intents = [{"name": c["name"], "description": c.get("description", ""),
                "est_minutes_manual": c.get("est_minutes_manual"), "unclassified": False}
               for c in tax["categories"]]
    intents.append({"name": NEEDS_REVIEW, "description": "No category scored above the similarity floor.",
                    "est_minutes_manual": None, "unclassified": True})
    write(driver, """UNWIND $rows AS r MERGE (i:Intent {name: r.name})
                     SET i.description = r.description, i.est_minutes_manual = r.est_minutes_manual,
                         i.unclassified = r.unclassified""", intents, "intents")

    part_of = {i: c["name"] for c in tax["categories"] for i in c["clusters"]}
    write(driver, """UNWIND $rows AS r MERGE (t:Topic {id: r.id})
                     SET t.name = r.name, t.description = r.description, t.keywords = r.keywords,
                         t.sensitive = r.sensitive
                     WITH t, r MATCH (i:Intent {name: r.intent}) MERGE (t)-[:PART_OF]->(i)""",
          [{"id": c["id"], "name": c["name"], "description": c.get("description", ""),
            "keywords": c.get("keywords", [])[:8], "sensitive": c["name"] in config.SENSITIVE_TOPICS,
            "intent": part_of[c["id"]]} for c in d["clusters"]], "topics")

    users = conv.groupby("user_id")["team"].first().reset_index()
    write(driver, """UNWIND $rows AS r MERGE (u:User {id: r.user_id})
                     MERGE (t:Team {name: r.team}) MERGE (u)-[:MEMBER_OF]->(t)""",
          users.to_dict("records"), "users")

    rows = [{"id": r.conversation_id, "user_id": r.user_id, "model": r.model, "topic_id": int(r.topic_id),
             "topic_sim": float(s), "task": r.task, "intent": r.intent,
             "preview": preview(r.first_user_msg, 280), "first_message": (r.first_user_msg or "")[:2000],
             "n_turns": int(r.n_turns), "n_user_msgs": int(r.n_user_msgs),
             "likely_failure": bool(r.likely_failure), "n_rephrase": int(r.n_rephrase),
             "n_correction": int(r.n_correction), "thanked": bool(r.thanked), "redacted": bool(r.redacted),
             "country": r.country if isinstance(r.country, str) else None,
             "timestamp": r.timestamp.isoformat(), "sensitive": bool(r.sensitive),
             "embedding": e.tolist()}
            for r, s, e in zip(conv.itertuples(), d["topic_sim"], d["X"])]
    write(driver, """
        UNWIND $rows AS r
        MERGE (c:Conversation {id: r.id})
        SET c.preview = r.preview, c.first_message = r.first_message, c.n_turns = r.n_turns,
            c.n_user_msgs = r.n_user_msgs, c.likely_failure = r.likely_failure, c.n_rephrase = r.n_rephrase,
            c.n_correction = r.n_correction, c.thanked = r.thanked, c.redacted = r.redacted,
            c.country = r.country, c.timestamp = datetime(r.timestamp), c.sensitive = r.sensitive,
            c.intent = r.intent, c.task = r.task, c.embedding = r.embedding
        WITH c, r
        MATCH (u:User {id: r.user_id}) MERGE (u)-[:STARTED]->(c)
        WITH c, r
        MATCH (t:Topic {id: r.topic_id}) MERGE (c)-[it:IN_TOPIC]->(t) SET it.similarity = r.topic_sim
        MERGE (m:Model {name: r.model}) MERGE (c)-[:USED_MODEL]->(m)
        MERGE (k:Task {key: r.task})
          ON CREATE SET k.action = split(r.task, ':')[0], k.doc = coalesce(split(r.task, ':')[1], '')
        MERGE (c)-[:HAS_TASK]->(k)""", rows, "conversations", batch=1000)

    write(driver, """UNWIND $rows AS r MATCH (c:Conversation {id: r.conversation_id})
                     MATCH (i:Intent {name: r.category})
                     MERGE (c)-[h:HAS_INTENT]->(i)
                     SET h.primary = r.is_primary, h.score = r.score, h.source = r.source""",
          d["topics"][["conversation_id", "category", "score", "source", "is_primary"]].to_dict("records"),
          "intent labels")

    mentions = [{"conversation_id": cid, "name": n, "kind": k}
                for cid, m in zip(conv["conversation_id"], conv["first_user_msg"]) for n, k in extract(m)]
    write(driver, """UNWIND $rows AS r MATCH (c:Conversation {id: r.conversation_id})
                     MERGE (e:Entity {name: r.name}) ON CREATE SET e.kind = r.kind
                     MERGE (c)-[:MENTIONS]->(e)""", mentions, "entity mentions")

    t = d["turns"]
    t = t.assign(truncated=t["content"].str.len() > args.max_chars, content=t["content"].str.slice(0, args.max_chars))
    write(driver, """UNWIND $rows AS r MATCH (c:Conversation {id: r.conversation_id})
                     CREATE (c)-[:HAS_TURN]->(:Turn {idx: r.idx, role: r.role, content: r.content,
                                                     truncated: r.truncated})""",
          t.to_dict("records"), "turns")

    ids = conv["conversation_id"].tolist()
    pairs = similar_pairs(d["X"], d["texts"], args.k_similar, args.min_similar)
    write(driver, """UNWIND $rows AS r MATCH (a:Conversation {id: r.a}) MATCH (b:Conversation {id: r.b})
                     MERGE (a)-[s:SIMILAR_TO]->(b) SET s.score = r.score""",
          [{"a": ids[i], "b": ids[j], "score": s} for i, j, s in pairs], "similarity edges")

    vector_index(driver, d["X"].shape[1])


def summary(driver) -> None:
    nodes = run(driver, "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC")
    rels = run(driver, "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC")
    print("\nnodes:", ", ".join(f"{r['label']} {r['n']:,}" for r in nodes))
    print("relationships:", ", ".join(f"{r['type']} {r['n']:,}" for r in rels))


def main():
    p = argparse.ArgumentParser(description="load the conversation knowledge graph into Neo4j")
    p.add_argument("--db", default="data/wildchat.duckdb")
    p.add_argument("--max-turns", type=int, default=12, help="turns stored per conversation")
    p.add_argument("--max-chars", type=int, default=2000, help="characters stored per turn")
    p.add_argument("--k-similar", type=int, default=5, help="SIMILAR_TO edges per conversation")
    p.add_argument("--min-similar", type=float, default=0.6, help="cosine floor for SIMILAR_TO")
    p.add_argument("--keep", action="store_true", help="don't wipe the database first")
    args = p.parse_args()

    t0 = time.time()
    data = load(args)
    driver = connect()
    try:
        if not args.keep:
            reset(driver)
        schema(driver)
        build(driver, data, args)
        summary(driver)
    finally:
        driver.close()
    print(f"\ndone in {time.time() - t0:.0f}s · browser: http://localhost:7474")


if __name__ == "__main__":
    main()
