from typing import List
from unittest.mock import MagicMock
import pytest

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.analyticdb import AnalyticDBVectorStore

try:
    from alibabacloud_gpdb20160503.client import Client

    analyticdb_installed = True
except ImportError:
    analyticdb_installed = False


def _create_mock_vector_store(client: Client):
    return AnalyticDBVectorStore(
        client=client,
        region_id="cn-hangzhou",
        instance_id="adb-instance-id",
        account="adb-account",
        account_password="adb-account-password",
    )


def _create_sample_documents(n: int) -> List[TextNode]:
    return [
        TextNode(
            text=f"text {i}",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=f"test doc id {i}")
            },
            embedding=[0.5, 0.5, 0.5, 0.5],
        )
        for i in range(n)
    ]


@pytest.mark.skipif(
    not analyticdb_installed, reason="alibaba-adb-client package not installed"
)
def test_adbvector_add() -> None:
    client = MagicMock(spec=Client)
    vector_store = _create_mock_vector_store(client)

    nodes = _create_sample_documents(2)
    ids = vector_store.add(nodes)

    assert len(ids) == 2
    assert client.upsert_collection_data.call_count == 1


@pytest.mark.skipif(
    not analyticdb_installed, reason="alibaba-adb-client package not installed"
)
def test_adbvector_delete() -> None:
    client = MagicMock(spec=Client)
    vector_store = _create_mock_vector_store(client)

    vector_store.delete(ref_doc_id="31d00b73-41ca-4d5a-a5b4-85a014b2c924")

    assert client.delete_collection_data.call_count == 1


def test_build_filter_clause_escapes_values() -> None:
    from llama_index.core.vector_stores.types import (
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )
    from llama_index.vector_stores.analyticdb.base import (
        _build_filter_clause,
        _recursively_parse_adb_filter,
    )

    payload = "x' OR '1'='1"
    assert (
        _build_filter_clause(MetadataFilter(key="tenant", value=payload))
        == "metadata_->>E'tenant' = E'x'' OR ''1''=''1'"
    )
    assert (
        _build_filter_clause(MetadataFilter(key="n", value=5, operator=">"))
        == "metadata_->>E'n' > E'5'"
    )
    assert (
        _build_filter_clause(
            MetadataFilter(key="k", value=["a", "b'\"\\"], operator=FilterOperator.IN)
        )
        == "metadata_->>E'k' IN (E'a', E'b''\"\\\\')"
    )
    assert (
        _build_filter_clause(
            MetadataFilter(key="tags", value=payload, operator=FilterOperator.CONTAINS)
        )
        == "metadata_::jsonb->E'tags' @> E'[\"x'' OR ''1''=''1\"]'::jsonb"
    )
    assert (
        _recursively_parse_adb_filter(
            MetadataFilters(
                filters=[
                    MetadataFilter(key="a", value="1"),
                    MetadataFilter(key="b", value="2"),
                ]
            )
        )
        is not None
    )


@pytest.mark.parametrize("key", ["a' OR '1'='1", "a b", "a;--", "a\\b", ""])
def test_build_filter_clause_rejects_bad_keys(key: str) -> None:
    from llama_index.core.vector_stores.types import MetadataFilter
    from llama_index.vector_stores.analyticdb.base import _build_filter_clause

    with pytest.raises(ValueError):
        _build_filter_clause(MetadataFilter(key=key, value="v"))


def test_build_filter_clause_rejects_bad_values() -> None:
    from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter
    from llama_index.vector_stores.analyticdb.base import _build_filter_clause

    with pytest.raises(ValueError):
        _build_filter_clause(MetadataFilter(key="k", value="a\x00b"))
    with pytest.raises(ValueError):
        _build_filter_clause(
            MetadataFilter(key="k", value=[], operator=FilterOperator.IN)
        )
    with pytest.raises(ValueError):
        _build_filter_clause(
            MetadataFilter(key="k", value="a", operator=FilterOperator.NIN)
        )
