
from __future__ import annotations

import re

HEAD, TAIL = 220, 200

# "translate" alone is also figurative ("translate program goals"), so it needs a
# direction or an object; the non-English verbs are unambiguous.
TRANSLATE = re.compile(
    r"\btranslat(e|ion|ing)\b\s*(it|this|that|these|the (following|text|sentence|word|paragraph)s?|:)"
    r"|\btranslat(e|ion|ing)\b[^.?!\n]{0,40}?\b(to|into|from)\s+[A-Z]"
    r"|\b(translate|translation)\b[^.?!\n]{0,20}\b(to|into|in) (english|chinese|arabic|french|spanish|german|"
    r"russian|turkish|korean|japanese|hindi|persian|farsi|urdu|filipino|portuguese|italian|vietnamese)\b"
    r"|\btraduc|\bübersetz|翻译|翻譯|中译英|英译中|中翻英|переве|ترجم|번역|\bdịch\b",
    re.I)

REWRITE = re.compile(
    r"\b(re-?write|re-?phrase|paraphra[sz]e?|reword|polish|proofread|summari[sz]e|shorten|condense|"
    r"improve (this|the|my|it)|correct (this|the|my|it|grammar)|fix (the |my )?grammar|"
    r"make (this|it) (sound )?(more|better|excellent|professional|formal|concise|shorter)|"
    r"edit( this| the| my)?:|edit (this|the following|my))\b"
    r"|润色|改写|перепиш|перефраз|сократи|улучши текст",
    re.I)

WRITE = re.compile(
    r"\b(write|draft|compose|create|generate|prepare|produce|make|give me|need)\b", re.I)

# Document types, as nouns that follow the verb. Order matters only for ties.
DOCS = {
    "cv": r"cv|resume|résumé|cover ?letter|linkedin (profile|summary|headline|about)|motivation letter|"
          r"statement of purpose|personal statement",
    "email": r"e-?mail(?! (marketing|verification|address|server|list))|mail|letter|reply|memo|text message|notice",
    "marketing": r"ad|advert(isement)?|slogan|tagline|caption|hashtags?|product description|"
                 r"(product|etsy|amazon) listing|(instagram|facebook|linkedin|twitter|social media) post|"
                 r"tweet|landing page|about us|website content|headlines?|seo",
    # "script" and "scene" alone are mostly code and image prompts, so only their fiction forms count
    "story": r"story|stories|screenplay|shooting script|script (for|of) (a|an|the) (scene|episode|movie|film|"
             r"show|play|skit|sketch)|poem|song|lyrics|dialogue|fan ?fic(tion)?|monologue|limerick|haiku|"
             r"novel|fable",
    "essay": r"essay|article|blog( post)?|report|paper|thesis|dissertation|conclusion|introduction|"
             r"literature review|lecture|presentation|speech|proposal|abstract|summary|research",
}
WRITE_DOC = re.compile(
    r"\b(write|draft|compose|create|generate|prepare|produce|make|give me|need)\b"
    r"[^.?!\n]{0,60}?\b(?:" + "|".join(f"(?P<{k}>{v})" for k, v in DOCS.items()) + r")\b",
    re.I)
DOC_ONLY = re.compile(r"\b(?:" + "|".join(f"(?P<{k}>{v})" for k, v in DOCS.items()) + r")\b", re.I)


def _window(text: str) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= HEAD + TAIL else t[:HEAD] + " \n " + t[-TAIL:]


def _doc(m: re.Match | None) -> str:
    if not m:
        return "text"
    return next((k for k, v in m.groupdict().items() if v), "text")


def detect(text: str) -> tuple[str, str]:
    """(task, doc) for one first user message."""
    w = _window(text)
    if TRANSLATE.search(w):
        return "translate", ""
    if REWRITE.search(w):
        # the thing being rewritten: the first document noun in the instruction
        return "rewrite", _doc(DOC_ONLY.search(w))
    m = WRITE_DOC.search(w)
    if m:
        return "write", _doc(m)
    return "answer", ""


def key(task: str, doc: str) -> str:
    return f"{task}:{doc}" if doc else task
