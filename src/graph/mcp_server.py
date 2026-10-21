
from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

import joblib
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from neo4j import READ_ACCESS, GraphDatabase, unit_of_work
from neo4j.graph import Node, Path as GPath, Relationship
from neo4j.time import Date, DateTime, Duration, Time

from graph import config
from taxonomy.embed import ART

logging.getLogger("neo4j").setLevel(logging.ERROR)

GOLD = Path(os.getenv("GOLD_DIR", "gold"))
QUERY_TIMEOUT = 15  # seconds per query
MAX_ROWS = 200

INSTRUCTIONS = """\
Knowledge graph of 20,000 real ChatGPT conversations (WildChat, English), classified into
user intents. Schema:
  (:User)-[:STARTED]->(:Conversation)-[:HAS_INTENT {primary, score, source}]->(:Intent)
  (:User)-[:MEMBER_OF]->(:Team)
  (:Conversation)-[:IN_TOPIC]->(:Topic)-[:PART_OF]->(:Intent)
  (:Conversation)-[:HAS_TASK]->(:Task)        task: translate | write:<doc> | rewrite:<doc> | answer
  (:Conversation)-[:MENTIONS]->(:Entity)      languages, tools, platforms, AI products, franchises
  (:Conversation)-[:USED_MODEL]->(:Model)
  (:Conversation)-[:HAS_TURN]->(:Turn {idx, role, content})
  (:Conversation)-[:SIMILAR_TO {score}]->(:Conversation)
Conversation.likely_failure is an implicit-feedback heuristic: the user rephrased or corrected
the assistant at least twice and never thanked it.

Start broad (intent_overview, failure_hotspots), then drill down (search_conversations,
get_conversation, related, user_activity). Before stating per-intent numbers as facts, check
classifier_reliability. Team names are synthetic. Conversations flagged sensitive (explicit
role-play) are hidden from content tools unless include_sensitive=true.
"""

server = MCPServer("conversation-graph", instructions=INSTRUCTIONS)
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

_driver = None
_model = None


# ---------------------------------------------------------------- plumbing
def driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    return _driver


def embed(text: str) -> list[float]:
    """Same model and normalisation as the classifier, so queries live in the graph's vector space."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(joblib.load(ART / "embedder.joblib")["model"])
    return _model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()


def clean(v: Any, max_chars: int = 500) -> Any:
    """Neo4j values -> JSON-safe; never return embeddings; cap long strings."""
    if isinstance(v, Node):
        return {"labels": sorted(v.labels), **clean(dict(v), max_chars)}
    if isinstance(v, Relationship):
        return {"type": v.type, **clean(dict(v), max_chars)}
    if isinstance(v, GPath):
        return {"nodes": [clean(n, max_chars) for n in v.nodes],
                "relationships": [clean(r, max_chars) for r in v.relationships]}
    if isinstance(v, dict):
        return {k: clean(x, max_chars) for k, x in v.items() if k != "embedding"}
    if isinstance(v, (list, tuple)):
        return [clean(x, max_chars) for x in v]
    if isinstance(v, (DateTime, Date, Time, Duration)):
        return str(v)
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, str) and len(v) > max_chars:
        return v[:max_chars] + "…"
    return v


def read(query: str, **params) -> list[dict]:
    @unit_of_work(timeout=QUERY_TIMEOUT)
    def work(tx):
        return [r.data() for r in tx.run(query, params)]

    with driver().session(database=config.NEO4J_DATABASE, default_access_mode=READ_ACCESS) as s:
        return s.execute_read(work)


def baseline_failure() -> float:
    return read("MATCH (c:Conversation) RETURN avg(CASE WHEN c.likely_failure THEN 1.0 ELSE 0.0 END) AS r")[0]["r"]


# ---------------------------------------------------------------- tools
@server.tool(annotations=READ_ONLY)
def graph_schema() -> dict:
    """Node labels and relationship types with counts. Call first if unsure what the graph holds."""
    nodes = read("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC")
    rels = read("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC")
    return {"nodes": nodes, "relationships": rels, "schema": INSTRUCTIONS}


@server.tool(annotations=READ_ONLY)
def intent_overview(include_needs_review: bool = True) -> dict:
    """What users come to the assistant for: per intent, conversations, distinct users,
    likely-failure rate, estimated manual effort (hours) and the topics it groups."""
    rows = read("""
        MATCH (u:User)-[:STARTED]->(c:Conversation)-[:HAS_INTENT {primary: true}]->(i:Intent)
        WHERE $nr OR NOT i.unclassified
        WITH i, count(c) AS conversations, count(DISTINCT u) AS users,
             avg(CASE WHEN c.likely_failure THEN 1.0 ELSE 0.0 END) AS failure_rate
        OPTIONAL MATCH (t:Topic)-[:PART_OF]->(i)
        RETURN i.name AS intent, i.description AS description, conversations, users, failure_rate,
               i.est_minutes_manual AS est_minutes_per_task,
               CASE WHEN i.est_minutes_manual IS NULL THEN null
                    ELSE round(conversations * i.est_minutes_manual / 60.0) END AS est_hours,
               [n IN collect(t.name) WHERE n IS NOT NULL] AS topics
        ORDER BY conversations DESC""", nr=include_needs_review)
    total = sum(r["conversations"] for r in rows)
    for r in rows:
        r["share"] = r["conversations"] / total
        r["users_per_conversation"] = r["users"] / r["conversations"]
    return {"baseline_failure_rate": baseline_failure(), "total_conversations": total,
            "est_hours_total": sum(r["est_hours"] or 0 for r in rows),
            "note": "users_per_conversation well below 1 means a few heavy users drive the volume.",
            "intents": clean(rows)}


HOTSPOT_MATCH = {
    "intent": "MATCH (c)-[:HAS_INTENT {primary: true}]->(g:Intent) WITH c, g.name AS grp, false AS hidden",
    "topic": "MATCH (c)-[:IN_TOPIC]->(g:Topic) WITH c, g.name AS grp, g.sensitive AS hidden",
    "entity": "MATCH (c)-[:MENTIONS]->(g:Entity) WITH c, g.name AS grp, false AS hidden",
    "model": "MATCH (c)-[:USED_MODEL]->(g:Model) WITH c, g.name AS grp, false AS hidden",
    "task": "MATCH (c)-[:HAS_TASK]->(g:Task) WITH c, g.key AS grp, false AS hidden",
    "team": "MATCH (u0:User)-[:STARTED]->(c), (u0)-[:MEMBER_OF]->(g:Team) WITH c, g.name AS grp, false AS hidden",
}


@server.tool(annotations=READ_ONLY)
def failure_hotspots(by: Literal["intent", "topic", "entity", "model", "task", "team"] = "topic",
                     min_conversations: int = 30, limit: int = 10, include_sensitive: bool = False) -> dict:
    """Where the assistant fails users: groups ranked by likely-failure rate, with lift over the
    baseline. `concentrated` marks groups driven by few users (fewer than 1 user per 5 conversations)."""
    rows = read(f"""
        MATCH (u:User)-[:STARTED]->(c:Conversation)
        {HOTSPOT_MATCH[by]}, u
        WHERE $sens OR NOT coalesce(hidden, false)
        WITH grp, count(DISTINCT c) AS conversations, count(DISTINCT u) AS users,
             count(DISTINCT CASE WHEN c.likely_failure THEN c END) AS failures
        WHERE conversations >= $min
        RETURN grp AS {by}, conversations, users, failures, toFloat(failures) / conversations AS failure_rate
        ORDER BY failure_rate DESC, conversations DESC LIMIT $limit""",
                sens=include_sensitive, min=min_conversations, limit=limit)
    base = baseline_failure()
    for r in rows:
        r["lift"] = r["failure_rate"] / base if base else None
        r["concentrated"] = r["users"] / r["conversations"] < 0.2
    return {"by": by, "baseline_failure_rate": base, "groups": clean(rows),
            "note": "likely_failure is a heuristic (repeat, reject, refusal or apology, no thanks; "
                            "64% precision on held-out users); "
                    "rates on small groups are noisy."}


@server.tool(annotations=READ_ONLY)
def search_conversations(query: str, k: int = 10, intent: str | None = None, only_failures: bool = False,
                         include_sensitive: bool = False) -> dict:
    """Semantic search over first user messages (vector index), each hit expanded through the graph
    with its intent, topic, task, entities and user. Filter by intent name or to likely failures."""
    rows = read(f"""
        CALL db.index.vector.queryNodes('{config.VECTOR_INDEX}', $candidates, $vec) YIELD node AS c, score
        WHERE ($intent IS NULL OR c.intent = $intent) AND (NOT $failures OR c.likely_failure)
              AND ($sens OR NOT c.sensitive)
        WITH c, score ORDER BY score DESC LIMIT $k
        MATCH (u:User)-[:STARTED]->(c)-[:IN_TOPIC]->(t:Topic)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        RETURN c.id AS conversation_id, score, c.preview AS first_message, c.intent AS intent,
               t.name AS topic, c.task AS task, c.likely_failure AS likely_failure,
               c.n_turns AS turns, collect(e.name) AS entities, u.id AS user_id
        ORDER BY score DESC""",
                # Filters are applied after the vector lookup, and failures are ~4% of
                # conversations, so filtered searches need a much deeper candidate pool.
                vec=embed(query), candidates=min(5000, k * (250 if only_failures or intent else 25)),
                k=k, intent=intent, failures=only_failures, sens=include_sensitive)
    return {"query": query, "results": clean(rows)}


@server.tool(annotations=READ_ONLY)
def get_conversation(conversation_id: str, max_turns: int = 8, max_chars: int = 800,
                     include_sensitive: bool = False) -> dict:
    """One conversation in full context: its turns, all intent labels (with how each was assigned),
    topic, task, entities, model, failure signals, and the user's overall activity."""
    rows = read("""
        MATCH (u:User)-[:STARTED]->(c:Conversation {id: $id})-[:IN_TOPIC]->(t:Topic)
        MATCH (c)-[:USED_MODEL]->(m:Model), (u)-[:MEMBER_OF]->(team:Team)
        OPTIONAL MATCH (c)-[h:HAS_INTENT]->(i:Intent)
        WITH u, c, t, m, team, collect({intent: i.name, primary: h.primary, score: h.score,
                                         source: h.source}) AS intents
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        WITH u, c, t, m, team, intents, collect(e.name) AS entities
        OPTIONAL MATCH (u)-[:STARTED]->(o:Conversation)
        WITH u, c, t, m, team, intents, entities, count(o) AS n_conv,
             avg(CASE WHEN o.likely_failure THEN 1.0 ELSE 0.0 END) AS user_failure
        RETURN c {.id, .n_turns, .n_user_msgs, .likely_failure, .n_rephrase, .n_correction, .thanked,
                  .task, .sensitive, .country, .timestamp} AS conversation,
               intents, t.name AS topic, entities, m.name AS model,
               {id: u.id, team: team.name, conversations: n_conv, failure_rate: user_failure} AS user""",
                id=conversation_id)
    if not rows:
        return {"error": f"no conversation {conversation_id!r}"}
    out = rows[0]
    if out["conversation"]["sensitive"] and not include_sensitive:
        out["turns"] = "hidden: flagged sensitive (pass include_sensitive=true to show)"
    else:
        turns = read("""MATCH (:Conversation {id: $id})-[:HAS_TURN]->(t:Turn) WHERE t.idx < $n
                        RETURN t.idx AS idx, t.role AS role, t.content AS content ORDER BY t.idx""",
                     id=conversation_id, n=max_turns)
        out["turns"] = clean(turns, max_chars)
    return clean(out)


@server.tool(annotations=READ_ONLY)
def related(conversation_id: str | None = None, entity: str | None = None, k: int = 5,
            include_sensitive: bool = False) -> dict:
    """Multi-hop context. For a conversation: its most similar conversations (is the failure a
    pattern?) and the same user's other conversations. For an entity: where it shows up (intents,
    topics), its failure rate, the entities mentioned with it, and example conversations."""
    if bool(conversation_id) == bool(entity):
        return {"error": "pass exactly one of conversation_id or entity"}
    if conversation_id:
        similar = read("""
            MATCH (:Conversation {id: $id})-[s:SIMILAR_TO]-(o:Conversation)
            WHERE $sens OR NOT o.sensitive
            WITH DISTINCT o, max(s.score) AS score ORDER BY score DESC LIMIT $k
            RETURN o.id AS conversation_id, score, o.preview AS first_message, o.intent AS intent,
                   o.likely_failure AS likely_failure""", id=conversation_id, k=k, sens=include_sensitive)
        same_user = read("""
            MATCH (u:User)-[:STARTED]->(:Conversation {id: $id}), (u)-[:STARTED]->(o:Conversation)
            WHERE o.id <> $id AND ($sens OR NOT o.sensitive)
            RETURN o.id AS conversation_id, o.timestamp AS timestamp, o.preview AS first_message,
                   o.intent AS intent, o.likely_failure AS likely_failure
            ORDER BY o.timestamp LIMIT $k""", id=conversation_id, k=k, sens=include_sensitive)
        n = len(similar)
        return clean({"conversation_id": conversation_id, "similar": similar,
                      "similar_failure_rate": sum(r["likely_failure"] for r in similar) / n if n else None,
                      "same_user_other_conversations": same_user}, 300)

    stats = read("""
        MATCH (e:Entity) WHERE toLower(e.name) = toLower($name)
        MATCH (u:User)-[:STARTED]->(c:Conversation)-[:MENTIONS]->(e)
        RETURN e.name AS entity, e.kind AS kind, count(c) AS conversations, count(DISTINCT u) AS users,
               avg(CASE WHEN c.likely_failure THEN 1.0 ELSE 0.0 END) AS failure_rate""", name=entity)
    if not stats:
        return {"error": f"no entity {entity!r}; try graph_query('MATCH (e:Entity) RETURN e.name, e.kind')"}
    name = stats[0]["entity"]
    intents = read("""MATCH (:Entity {name: $n})<-[:MENTIONS]-(c:Conversation)
                      RETURN c.intent AS intent, count(*) AS conversations ORDER BY conversations DESC LIMIT 8""",
                   n=name)
    together = read("""MATCH (:Entity {name: $n})<-[:MENTIONS]-(c:Conversation)-[:MENTIONS]->(o:Entity)
                       WHERE o.name <> $n
                       RETURN o.name AS entity, count(*) AS conversations ORDER BY conversations DESC LIMIT 8""",
                    n=name)
    examples = read("""MATCH (:Entity {name: $n})<-[:MENTIONS]-(c:Conversation)
                       WHERE $sens OR NOT c.sensitive
                       RETURN c.id AS conversation_id, c.preview AS first_message, c.intent AS intent,
                              c.likely_failure AS likely_failure
                       ORDER BY c.likely_failure DESC, c.n_turns DESC LIMIT $k""",
                    n=name, k=k, sens=include_sensitive)
    return clean({**stats[0], "baseline_failure_rate": baseline_failure(), "intents": intents,
                  "mentioned_with": together, "examples": examples}, 300)


@server.tool(annotations=READ_ONLY)
def user_activity(user_id: str, limit: int = 20, include_sensitive: bool = False) -> dict:
    """A user's journey: their conversations in time order with intent, task and failure signal,
    plus totals. Use it to see whether failing users come back, and with what."""
    rows = read("""
        MATCH (u:User {id: $id})-[:STARTED]->(c:Conversation), (u)-[:MEMBER_OF]->(t:Team)
        WITH u, t, c ORDER BY c.timestamp
        WITH u, t, collect(c) AS cs
        RETURN u.id AS user_id, t.name AS team, size(cs) AS conversations,
               size([c IN cs WHERE c.likely_failure]) AS failures,
               [c IN cs WHERE $sens OR NOT c.sensitive |
                c {.id, .timestamp, .intent, .task, .likely_failure, .preview}][..$limit] AS timeline""",
                id=user_id, limit=limit, sens=include_sensitive)
    return clean(rows[0], 200) if rows else {"error": f"no user {user_id!r}"}


@server.tool(annotations=READ_ONLY)
def classifier_reliability() -> dict:
    """How far to trust the intent labels: accuracy on a held-out gold set (users never seen in
    training) and precision / recall per intent. Check before presenting per-intent numbers."""
    summary = json.loads((GOLD / "summary.json").read_text()) if (GOLD / "summary.json").exists() else None
    per = []
    if (GOLD / "per_category.csv").exists():
        with open(GOLD / "per_category.csv") as f:
            per = [{k: (float(v) if v and k in ("precision", "recall_weighted") else v) for k, v in r.items()}
                   for r in csv.DictReader(f)]
    if not summary and not per:
        return {"error": "no evaluation found; run `python -m taxonomy.main gold-eval`"}
    failure = GOLD / "failure" / "summary.json"
    return clean({"overall": summary, "per_intent": per,
                  "failure_signal": json.loads(failure.read_text()) if failure.exists() else None,
                  "caveats": ["Gold labels were made by a model (Claude), not a person.",
                              "About 20 rows per intent: per-intent figures are ±20 points.",
                              "likely_failure (v2) is right about two times in three on held-out data "
                              "(see failure_signal); refusals count as failures, including policy-correct ones."]})


WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|ALTER|GRANT|DENY|REVOKE|"
    r"TERMINATE|CALL\s+(dbms|apoc)|IN\s+TRANSACTIONS)\b", re.I)


@server.tool(annotations=READ_ONLY)
def graph_query(cypher: str, params: dict[str, Any] | None = None) -> dict:
    """Read-only Cypher for questions the other tools don't cover. Runs in a read transaction with a
    timeout; results are capped at 200 rows; embeddings are never returned. Use $params, not
    string formatting."""
    if WRITE_CLAUSES.search(cypher):
        return {"error": "read-only: write and admin clauses are not allowed"}
    try:
        rows = read(cypher, **(params or {}))
    except Exception as e:  # surface Cypher errors to the model so it can fix the query
        return {"error": f"{type(e).__name__}: {e}"}
    return {"rows": clean(rows[:MAX_ROWS]), "truncated": len(rows) > MAX_ROWS}


def main():
    server.run()


if __name__ == "__main__":
    main()
