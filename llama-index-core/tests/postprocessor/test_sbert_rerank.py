import sys
import types
from typing import Any, Dict

import pytest
from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank


@pytest.fixture
def fake_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}

    class FakeCrossEncoder:
        def __init__(self, model: str, **kwargs: Any) -> None:
            captured["model"] = model
            captured.update(kwargs)

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return captured


def test_trust_remote_code_disabled_by_default(
    fake_cross_encoder: Dict[str, Any],
) -> None:
    reranker = SentenceTransformerRerank(device="cpu")

    assert fake_cross_encoder["trust_remote_code"] is False
    assert reranker.trust_remote_code is False


def test_trust_remote_code_opt_in(fake_cross_encoder: Dict[str, Any]) -> None:
    reranker = SentenceTransformerRerank(device="cpu", trust_remote_code=True)

    assert fake_cross_encoder["trust_remote_code"] is True
    assert reranker.trust_remote_code is True
