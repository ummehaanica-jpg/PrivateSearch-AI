"""Small, dependency-free text helpers shared by search, QA and discovery."""
from __future__ import annotations

import re
from collections import Counter

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its itself just
    me more most my no nor not now of off on once only or other our out over own same she should
    so some such than that the their theirs them then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would you
    your yours yourself may might must shall one two use used using via per""".split()
)

# Everyday words that make poor "concepts" (used only when extracting terms for discovery).
GENERIC_WORDS = frozenset(
    """many much often well like make made makes also even still every whether without within another
    since however therefore common simple good better best large larger small smaller newer older new
    early late long short high low different similar various several single whole way ways thing
    things case cases example examples part parts kind type types number level based built often
    always never usually sometimes already often around among either neither whose whatever each
    help helps keep keeps let lets take takes give gives need needs needed want wants become becomes
    first second third next last end term""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def stem(word: str) -> str:
    """Very light suffix stripper so that 'agents' and 'agent' match."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
    return word


def is_content_word(word: str) -> bool:
    return len(word) >= 3 and word not in STOPWORDS and not word.isdigit()


def content_stems(text: str) -> set[str]:
    return {stem(w) for w in tokenize(text) if is_content_word(w)}


def extract_terms(text: str) -> dict[str, str]:
    """Return {term_key: surface_form} for unigrams and adjacent-word bigrams."""
    tokens = tokenize(text)
    terms: dict[str, str] = {}
    previous: str | None = None
    for token in tokens:
        if not is_content_word(token) or token in GENERIC_WORDS:
            previous = None
            continue
        terms.setdefault(stem(token), token)
        if previous is not None:
            terms.setdefault(f"{stem(previous)} {stem(token)}", f"{previous} {token}")
        previous = token
    return terms


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}|\n(?=\s*[-*\u2022]\s)", text)
    return [p.strip() for p in parts if p and p.strip()]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def top_terms(texts: list[str], n: int = 10) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(w for w in tokenize(text) if is_content_word(w))
    return [w for w, _ in counts.most_common(n)]
