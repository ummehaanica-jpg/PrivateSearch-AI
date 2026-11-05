"""Headless UI check using Streamlit's AppTest (no browser needed). Run: python scripts/smoke_ui.py"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["PSAI_DATA_DIR"] = tempfile.mkdtemp()
os.environ["PSAI_EMBEDDER"] = "hash"
os.environ["PSAI_OLLAMA_HOST"] = "http://localhost:9"  # nothing listens here: exercises the no-LLM path

from streamlit.testing.v1 import AppTest  # noqa: E402


def submit(at, label):
    button = next(b for b in at.button if b.label == label)
    button.click().run()
    assert not at.exception, (label, at.exception)


at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
assert not at.exception, at.exception
print("ok: first run shows empty state")

next(b for b in at.button if b.label == "Try the sample documents").click().run()
assert not at.exception, at.exception
print("ok: sample documents indexed")

at.text_input(key="search_text").set_value("AI agents")
submit(at, "Search")
assert at.session_state["search_state"]["results"], "search returned nothing"
print("ok: search ->", len(at.session_state["search_state"]["results"]), "results")

at.session_state["page"] = "Ask"
at.run()
at.text_input(key="ask_text").set_value("What does a supervisor agent do?")
submit(at, "Get answer")
answer = at.session_state["answer"]
print("ok: ask ->", answer.mode, "| sources:", len(answer.sources))

at.session_state["page"] = "Discover"
at.run()
at.text_input(key="disc_text").set_value("AI agents")
submit(at, "Find connections")
disc = at.session_state["disc"]
print("ok: discover ->", len(disc.passages), "passages,", len(disc.connections), "links")

for page in ["Documents", "Privacy & settings", "Search"]:
    at.session_state["page"] = page
    at.run()
    assert not at.exception, (page, at.exception)
    print("ok:", page)
print("UI smoke test passed")
