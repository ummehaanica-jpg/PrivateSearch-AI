"""PrivateSearch AI - entry point.  Run with:  streamlit run app.py"""
import streamlit as st

from privatesearch.config import Settings, apply_privacy_environment
from ui.pages import (PAGES, render_ask, render_discover, render_documents, render_privacy, render_search,
                      render_sidebar)
from ui.theme import inject_theme

st.set_page_config(page_title="PrivateSearch AI", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")
apply_privacy_environment(Settings.from_env())
inject_theme()


@st.cache_resource(show_spinner="Starting the local engine. The first run downloads a small embedding model…")
def get_engine():
    from privatesearch.engine import PrivateSearchEngine
    return PrivateSearchEngine()


try:
    engine = get_engine()
except Exception as exc:  # show a readable message instead of a stack trace
    st.error("PrivateSearch AI could not start.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("search_mode_label", "Hybrid")
render_sidebar(engine)

{
    "Search": render_search,
    "Ask": render_ask,
    "Discover": render_discover,
    "Documents": render_documents,
    "Privacy & settings": render_privacy,
}[st.session_state["page"]](engine)
