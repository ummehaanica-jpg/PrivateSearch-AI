import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from privatesearch.config import Settings  # noqa: E402
from privatesearch.embeddings import HashingEmbedder  # noqa: E402
from privatesearch.engine import PrivateSearchEngine  # noqa: E402

SAMPLES = ROOT / "sample_docs"


class FakeLLM:
    """Stands in for Ollama so tests need no model and no network."""

    model = "fake-model"

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


@pytest.fixture()
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", embedder_backend="hash")


@pytest.fixture()
def engine(settings):
    return PrivateSearchEngine(settings, embedder=HashingEmbedder())


@pytest.fixture()
def loaded_engine(engine):
    for path in sorted(SAMPLES.iterdir()):
        result = engine.ingest_path(path)
        assert result.status == "added", result.message
    return engine
