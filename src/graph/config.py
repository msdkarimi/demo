
import os
import re

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "conversation-graph")  # matches docker-compose.yml
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

VECTOR_INDEX = "conversation_embedding"

SENSITIVE_TOPICS = {"Comedic nude-bathing fanfic", "Hypnosis role-play stories"}
SENSITIVE_RE = re.compile(
    r"\b(nude|naked|au naturel|nsfw|erotic|fetish|sex(ual|y)?|porn|yaoi|harem|bulges?|boobs?|breasts?|"
    r"butt|fart\w*|diapers?|poop\w*|pee(ing)?|wet(ting)? (himself|herself|themselves)|hypnoti\w*|"
    r"pregnan\w*|belly|horny|seduc\w*|lewd|genitals?|vore)\b",
    re.I)
