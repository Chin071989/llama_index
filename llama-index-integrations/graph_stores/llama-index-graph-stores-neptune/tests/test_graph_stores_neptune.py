from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.neptune import (
    NeptuneAnalyticsGraphStore,
    NeptuneDatabaseGraphStore,
)
from llama_index.graph_stores.neptune.base import NeptuneBaseGraphStore


def test_neptune_analytics_graph_store():
    names_of_bases = [b.__name__ for b in NeptuneAnalyticsGraphStore.__bases__]
    assert NeptuneBaseGraphStore.__name__ in names_of_bases


def test_neptune_database_graph_store():
    names_of_bases = [b.__name__ for b in NeptuneDatabaseGraphStore.__bases__]
    assert NeptuneBaseGraphStore.__name__ in names_of_bases


def test_neptune_base_graph_store():
    names_of_bases = [b.__name__ for b in NeptuneBaseGraphStore.__bases__]
    assert GraphStore.__name__ in names_of_bases


def test_quote_identifier_strips_injection_characters():
    from llama_index.graph_stores.neptune.neptune import quote_identifier

    assert quote_identifier("WORKS_AT") == "`WORKS_AT`"
    assert quote_identifier("has part") == "`has part`"
    payload = "X`]->(target) MATCH (n) DETACH DELETE n MERGE (source)-[r:`X"
    quoted = quote_identifier(payload)
    assert quoted.count("`") == 2
    assert quoted.startswith("`") and quoted.endswith("`")
    assert "\\" not in quote_identifier("A\\u0060B")


def test_quote_identifier_rejects_empty():
    import pytest
    from llama_index.graph_stores.neptune.neptune import quote_identifier

    for bad in ["", "``", "   "]:
        with pytest.raises(ValueError):
            quote_identifier(bad)


def test_property_graph_queries_quote_labels():
    from unittest.mock import MagicMock

    from llama_index.core.graph_stores.types import EntityNode, Relation
    from llama_index.graph_stores.neptune import NeptuneDatabasePropertyGraphStore

    store = NeptuneDatabasePropertyGraphStore.__new__(NeptuneDatabasePropertyGraphStore)
    store.structured_query = MagicMock(return_value=[])
    payload = "X`]->(t) MATCH (n) DETACH DELETE n //"

    store.upsert_relations([Relation(label=payload, source_id="a", target_id="b")])
    store.upsert_nodes([EntityNode(name="n", label=payload)])
    store.delete(relation_names=[payload], properties={payload: 1})
    store.get_triplets(relation_names=[payload], properties={payload: 1})

    for call in store.structured_query.call_args_list:
        query = call.args[0]
        assert "`X]->(t) MATCH (n) DETACH DELETE n //`" in query
        assert "X`]" not in query


def test_refresh_schema_quotes_labels():
    from llama_index.graph_stores.neptune.neptune import refresh_schema

    queries = []

    def query(q):
        queries.append(q)
        return []

    payload = "L`) MATCH (n) DETACH DELETE n //"
    refresh_schema(query, {"nodeLabels": [payload], "edgeLabels": [payload]})
    assert len(queries) == 3
    for q in queries:
        assert "`L) MATCH (n) DETACH DELETE n //`" in q
