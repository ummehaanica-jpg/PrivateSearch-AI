"""HTML snippets. Every piece of document text is escaped before it reaches the page."""
from __future__ import annotations

import html
import re

from privatesearch.models import SearchResult
from privatesearch.textutil import stem

WORD = re.compile(r"[A-Za-z0-9]+")


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def highlight(text: str, terms: list[str] | set[str], css_class: str = "") -> str:
    """Escape `text` and wrap words whose stem matches one of `terms` in <mark>."""
    stems = {stem(t.lower()) for term in terms for t in WORD.findall(term)}
    out, last = [], 0
    for m in WORD.finditer(text):
        if stem(m.group().lower()) in stems and len(m.group()) > 2:
            out.append(esc(text[last:m.start()]))
            cls = f' class="{css_class}"' if css_class else ""
            out.append(f"<mark{cls}>{esc(m.group())}</mark>")
            last = m.end()
    out.append(esc(text[last:]))
    return "".join(out).replace("\n", "<br>")


def chip(text: str, tone: str = "") -> str:
    return f'<span class="ps-chip {tone}">{esc(text)}</span>'


def page_head(title: str, subtitle: str) -> None:
    import streamlit as st
    st.markdown(f'<div class="ps-page-head"><h2>{esc(title)}</h2><p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> str:
    return f'<div class="ps-empty"><h3>{esc(title)}</h3><p>{esc(body)}</p></div>'


def stat(value: object, label: str) -> str:
    return f'<div class="ps-stat"><b>{esc(value)}</b><span>{esc(label)}</span></div>'


def passage_card(result: SearchResult, terms: list[str], rank: int | None = None) -> str:
    c = result.chunk
    chips = [chip(f"Page {c.page}", "violet")] if c.page else []
    if result.cosine is not None:
        chips.append(chip(f"meaning {result.cosine:.2f}"))
    if result.keyword_rank:
        chips.append(chip(f"keywords #{result.keyword_rank}"))
    elif result.matched_terms == []:
        chips.append(chip("found by meaning", "good"))
    prefix = f"{rank}. " if rank else ""
    return (
        f'<div class="ps-card"><div class="ps-head"><span class="ps-doc">{esc(prefix)}{esc(c.doc_name)}</span>'
        f'{"".join(chips)}</div><div class="ps-text">{highlight(c.text, terms)}</div>'
        f'<div class="ps-bar"><span style="width:{max(4, round(result.score * 100))}%"></span></div></div>'
    )


def passage_block(text: str, terms: list[str], css_class: str = "") -> str:
    return f'<div class="ps-text">{highlight(text, terms, css_class)}</div>'
