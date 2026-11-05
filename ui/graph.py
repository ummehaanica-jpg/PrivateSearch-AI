"""Plotly knowledge map for the Discovery page."""
from __future__ import annotations

import textwrap

import networkx as nx
import plotly.graph_objects as go

from privatesearch.discovery import DiscoveryResult
from ui.components import esc

PALETTE = ["#8B7CFF", "#4DA3FF", "#3DDBC0", "#FF8FB1", "#F5B84A", "#9BE564", "#C792EA", "#FF9F6E"]
MUTED, TEXT = "#9AA3C7", "#E6E8F5"


def _wrap(text: str, width: int = 58, lines: int = 5) -> str:
    wrapped = textwrap.wrap(text, width)
    if len(wrapped) > lines:
        wrapped = wrapped[:lines]
        wrapped[-1] += "…"
    return "<br>".join(esc(w) for w in wrapped)


def doc_colors(result: DiscoveryResult) -> dict[str, str]:
    names = list(dict.fromkeys(n.chunk.doc_name for n in result.passages))
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)}


def build_figure(result: DiscoveryResult, show_concepts: bool = True, selected: str | None = None) -> go.Figure:
    graph = nx.Graph()
    nodes = [n for n in result.nodes if show_concepts or n.kind != "concept"]
    keep = {n.id for n in nodes}
    for n in nodes:
        graph.add_node(n.id)
    edges = [e for e in result.edges if e.source in keep and e.target in keep]
    for e in edges:
        graph.add_edge(e.source, e.target, weight=max(e.weight, 0.05))
    pos = nx.spring_layout(graph, seed=11, k=1.4 / max(len(nodes), 1) ** 0.5, iterations=200,
                           weight="weight", pos={"q": (0.0, 0.0)}, fixed=["q"])
    colors = doc_colors(result)
    fig = go.Figure()

    def line_trace(subset, color, width, dash, name, opacity):
        xs, ys = [], []
        for e in subset:
            (x0, y0), (x1, y1) = pos[e.source], pos[e.target]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False, name=name,
                                     line=dict(color=color, width=width, dash=dash), opacity=opacity))

    line_trace([e for e in edges if e.kind == "query-match"], MUTED, 1, "dot", "query", 0.45)
    line_trace([e for e in edges if e.kind == "concept-mention"], "#5A6396", 1, "dash", "concept", 0.35)
    links = [e for e in edges if e.kind == "semantic-link"]
    for lo, hi, width in ((0.0, 0.45, 1.6), (0.45, 0.6, 2.8), (0.6, 2.0, 4.2)):
        line_trace([e for e in links if lo <= e.weight < hi], "#A99DFF", width, "solid", "link", 0.75)
    if links:  # invisible midpoints give each connection a hover explanation
        fig.add_trace(go.Scatter(
            x=[(pos[e.source][0] + pos[e.target][0]) / 2 for e in links],
            y=[(pos[e.source][1] + pos[e.target][1]) / 2 for e in links],
            mode="markers", marker=dict(size=14, opacity=0), showlegend=False,
            hovertext=[_wrap(e.explanation, 64, 6) for e in links], hoverinfo="text"))

    for doc_name, color in colors.items():
        group = [n for n in result.passages if n.chunk.doc_name == doc_name]
        fig.add_trace(go.Scatter(
            x=[pos[n.id][0] for n in group], y=[pos[n.id][1] for n in group], mode="markers+text",
            name=doc_name, customdata=[n.id for n in group],
            text=[n.label for n in group], textposition="top center", textfont=dict(size=10, color=MUTED),
            marker=dict(size=[22 if n.role == "seed" else 15 for n in group], color=color,
                        line=dict(width=[3 if n.id == selected else 1 for n in group],
                                  color=["#FFFFFF" if n.id == selected else "#0F1220" for n in group])),
            hovertext=[f"<b>{esc(n.chunk.location)}</b><br>{_wrap(n.chunk.text)}<br><i>{esc(n.reason)}</i>" for n in group],
            hoverinfo="text"))
    concept_nodes = [n for n in nodes if n.kind == "concept"]
    if concept_nodes:
        fig.add_trace(go.Scatter(
            x=[pos[n.id][0] for n in concept_nodes], y=[pos[n.id][1] for n in concept_nodes],
            mode="markers+text", name="Shared concept", text=[n.label for n in concept_nodes],
            textposition="bottom center", textfont=dict(size=11, color=TEXT), customdata=[n.id for n in concept_nodes],
            marker=dict(symbol="diamond", size=12, color="#2A3052", line=dict(color="#9AA3C7", width=1.5)),
            hovertext=[esc(n.reason) for n in concept_nodes], hoverinfo="text"))
    query = result.node("q")
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", name="Your search", text=[query.label],
                             textposition="bottom center", textfont=dict(size=13, color="#FFFFFF"),
                             marker=dict(symbol="star", size=30, color="#FFFFFF", line=dict(color="#8B7CFF", width=2)),
                             hoverinfo="skip", customdata=["q"]))
    fig.update_layout(
        height=580, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False), clickmode="event+select", dragmode="pan",
        font=dict(color=TEXT), legend=dict(orientation="h", y=-0.04, font=dict(size=11, color=MUTED)),
        hoverlabel=dict(bgcolor="#1F2444", bordercolor="#2A3052", font=dict(color=TEXT, size=12)))
    return fig
