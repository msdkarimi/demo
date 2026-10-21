
from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path

import matplotlib
import networkx as nx
from matplotlib import pyplot as plt
from neo4j import READ_ACCESS, GraphDatabase

from graph import config

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Neo4j Browser-like palette, one colour per label
COLORS = {
    "Conversation": "#8DCC93", "User": "#F79767", "Team": "#C990C0",
    "Intent": "#57C7E3", "Topic": "#4C8EDA", "Entity": "#FFC454", "Model": "#D9C8AE",
}
FAILED = "#F16667"

# Similar conversations from the same user are mostly one person re-sending a
# prompt; ranking by other users' failures shows a pattern instead.
CENTER = """
MATCH (u0:User)-[:STARTED]->(c:Conversation)-[:MENTIONS]->(:Entity {name: $entity})
WHERE c.likely_failure AND NOT c.sensitive
MATCH (c)-[:SIMILAR_TO]-(s:Conversation)<-[:STARTED]-(u:User)
WHERE u <> u0 AND NOT s.sensitive
WITH c, count(DISTINCT CASE WHEN s.likely_failure THEN s END) AS failed_sims,
     count(DISTINCT u) AS other_users
RETURN c.id AS id ORDER BY failed_sims DESC, other_users DESC LIMIT 1
"""

SIMILAR = """
MATCH (c:Conversation {id: $id})
OPTIONAL MATCH (c)-[:SIMILAR_TO]-(s:Conversation) WHERE NOT s.sensitive
RETURN [c.id] + collect(DISTINCT s.id)[..$k] AS ids
"""

USER = """
MATCH (:User {id: $user})-[:STARTED]->(c:Conversation) WHERE NOT c.sensitive
WITH c ORDER BY c.timestamp
RETURN collect(c.id)[..$k] AS ids
"""

NEIGHBOURHOOD = """
MATCH (u:User)-[:STARTED]->(conv:Conversation) WHERE conv.id IN $ids
OPTIONAL MATCH (u)-[:MEMBER_OF]->(t:Team)
OPTIONAL MATCH (conv)-[:HAS_INTENT {primary: true}]->(i:Intent)
OPTIONAL MATCH (conv)-[:IN_TOPIC]->(tp:Topic)
OPTIONAL MATCH (conv)-[:USED_MODEL]->(m:Model)
RETURN conv.id AS id, conv.first_message AS msg, conv.likely_failure AS failed,
       u.id AS user, t.name AS team, i.name AS intent, tp.name AS topic, m.name AS model,
       [(conv)-[:MENTIONS]->(e:Entity) | e.name] AS entities,
       [(conv)-[:SIMILAR_TO]-(o:Conversation) WHERE o.id IN $ids | o.id] AS similar
"""


def fetch(entity: str | None, user: str | None, k: int) -> list[dict]:
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    with driver, driver.session(database=config.NEO4J_DATABASE, default_access_mode=READ_ACCESS) as s:
        if user:
            ids = s.run(USER, user=user, k=k).single()["ids"]
        else:
            center = s.run(CENTER, entity=entity).single()
            if center is None:
                raise SystemExit(f"no failed conversation mentions {entity!r}")
            log.info("center conversation: %s", center["id"])
            ids = s.run(SIMILAR, id=center["id"], k=k).single()["ids"]
        if not ids:
            raise SystemExit("no conversations found")
        return [r.data() for r in s.run(NEIGHBOURHOOD, ids=ids)]


def build(rows: list[dict]) -> nx.Graph:
    g = nx.Graph()

    def node(key, label, text, failed=False):
        g.add_node(key, label=label, text=text, failed=failed)

    for r in rows:
        conv = ("Conversation", r["id"])
        node(conv, "Conversation", textwrap.shorten(r["msg"] or "", 28, placeholder="…"), r["failed"])
        user = ("User", r["user"])
        node(user, "User", f"user {r['user'][:6]}")
        g.add_edge(user, conv, rel="STARTED")
        if r["team"]:
            node(("Team", r["team"]), "Team", r["team"])
            g.add_edge(user, ("Team", r["team"]), rel="MEMBER_OF")
        if r["intent"]:
            node(("Intent", r["intent"]), "Intent", r["intent"])
            g.add_edge(conv, ("Intent", r["intent"]), rel="HAS_INTENT")
        if r["topic"]:
            node(("Topic", r["topic"]), "Topic", r["topic"])
            g.add_edge(conv, ("Topic", r["topic"]), rel="IN_TOPIC")
            if r["intent"]:
                g.add_edge(("Topic", r["topic"]), ("Intent", r["intent"]), rel="PART_OF")
        if r["model"]:
            node(("Model", r["model"]), "Model", r["model"])
            g.add_edge(conv, ("Model", r["model"]), rel="USED_MODEL")
        for e in r["entities"]:
            node(("Entity", e), "Entity", e)
            g.add_edge(conv, ("Entity", e), rel="MENTIONS")
        for o in r["similar"]:
            g.add_edge(conv, ("Conversation", o), rel="SIMILAR_TO")
    return g


def draw(g: nx.Graph, title: str, out: Path) -> None:
    # Rings: users in the middle, their conversations around them, everything
    # the conversations point to on the outside (grouped by label, so edges of
    # one kind fan out together)
    by_label = lambda *labels: [n for l in labels for n, d in g.nodes(data=True) if d["label"] == l]
    pos = nx.shell_layout(g, [by_label("User"), by_label("Conversation"),
                              by_label("Team", "Model", "Intent", "Topic", "Entity")])
    fig, ax = plt.subplots(figsize=(14, 13))
    ax.set_facecolor("white")

    similar = [(a, b) for a, b, d in g.edges(data=True) if d["rel"] == "SIMILAR_TO"]
    other = [(a, b) for a, b, d in g.edges(data=True) if d["rel"] != "SIMILAR_TO"]
    nx.draw_networkx_edges(g, pos, edgelist=other, edge_color="#A5ABB6", width=0.9, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=similar, edge_color="#F16667", width=1.6, style="dashed", ax=ax)
    # Edge labels on every line are unreadable at this density; list the
    # relationship types with counts instead
    counts: dict[str, int] = {}
    for _, _, d in g.edges(data=True):
        counts[d["rel"]] = counts.get(d["rel"], 0) + 1
    ax.text(1.0, 0.0, "Relationships\n" + "\n".join(f"{r}  {n}" for r, n in sorted(counts.items(), key=lambda x: -x[1])),
            transform=ax.transAxes, ha="right", va="bottom", multialignment="left", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#ccc"))

    for label, color in COLORS.items():
        nodes = [n for n, d in g.nodes(data=True) if d["label"] == label]
        colors = [FAILED if g.nodes[n]["failed"] else color for n in nodes]
        size = 1100 if label == "Conversation" else 1600
        nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors, node_size=size,
                               edgecolors="#444", linewidths=0.8, ax=ax)
    for n, (x, y) in pos.items():
        d = g.nodes[n]
        text = d["text"] if d["label"] != "Conversation" else "\n".join(textwrap.wrap(d["text"], 16))
        ax.text(x, y - 0.075, text, ha="center", va="top", fontsize=7.5,
                fontweight="bold" if d["label"] != "Conversation" else "normal")

    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, mfc=c, mec="#444", label=l) for l, c in COLORS.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", ms=10, mfc=FAILED, mec="#444", label="Failed conversation"))
    if similar:
        handles.append(plt.Line2D([], [], color="#F16667", ls="--", label="SIMILAR_TO"))
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=False)
    n_conv = sum(1 for _, d in g.nodes(data=True) if d["label"] == "Conversation")
    n_fail = sum(1 for _, d in g.nodes(data=True) if d["failed"])
    ax.set_title(f"{title}: {n_fail} of {n_conv} conversations failed", fontsize=13, loc="left")
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info("wrote %s (%d nodes, %d relationships)", out, g.number_of_nodes(), g.number_of_edges())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", help="draw this user's conversations")
    p.add_argument("--entity", default="pandas", help="without --user: start from a failed conversation mentioning this")
    p.add_argument("--k", type=int, default=10, help="conversations to include")
    p.add_argument("--out", type=Path, default=Path("artifacts/plots/graph_example.png"))
    args = p.parse_args()
    title = (f"One user's conversations" if args.user
             else f"A failed {args.entity} conversation and its similar conversations")
    draw(build(fetch(args.entity, args.user, args.k)), title, args.out)


if __name__ == "__main__":
    main()
