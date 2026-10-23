"""Unit tests for Cypher identifier escaping in Neo4jPropertyGraphStore."""

from unittest.mock import MagicMock

from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.graph_stores.neo4j.neo4j_property_graph import (
    Neo4jPropertyGraphStore,
    escape_cypher_identifier,
)

PAYLOAD = "X`) MATCH (m) DETACH DELETE m WITH count(*) AS c MATCH (n:`X"


def _store() -> Neo4jPropertyGraphStore:
    store = Neo4jPropertyGraphStore.__new__(Neo4jPropertyGraphStore)
    store.structured_schema = {"metadata": {"index": []}}
    store.structured_query = MagicMock(return_value=[])
    return store


def _queries(store) -> list:
    return [c.args[0] for c in store.structured_query.call_args_list]


def _strip_identifiers(cypher: str) -> str:
    """Remove every well-formed backtick-quoted identifier from the query."""
    out, i = [], 0
    while i < len(cypher):
        if cypher[i] == "`":
            i += 1
            while i < len(cypher):
                if cypher[i] == "`":
                    if i + 1 < len(cypher) and cypher[i + 1] == "`":
                        i += 2
                        continue
                    break
                i += 1
            assert i < len(cypher), "unterminated identifier"
            out.append("ID")
            i += 1
        else:
            out.append(cypher[i])
            i += 1
    return "".join(out)


def test_escape_cypher_identifier():
    assert escape_cypher_identifier("Person") == "`Person`"
    assert escape_cypher_identifier("a`b") == "`a``b`"
    assert escape_cypher_identifier("a\\u0060b") == "`a``b`"
    assert escape_cypher_identifier(PAYLOAD).count("`") == 2 * PAYLOAD.count("`") + 2


def test_enhanced_schema_cypher_escapes_label_and_props():
    store = _store()
    props = [
        {"property": PAYLOAD, "type": "STRING"},
        {"property": PAYLOAD, "type": "INTEGER"},
        {"property": PAYLOAD, "type": "LIST"},
    ]
    for is_rel in (False, True):
        for exhaustive in (False, True):
            cypher, params = store._enhanced_schema_cypher(
                PAYLOAD, props, exhaustive, is_relationship=is_rel
            )
            assert params == {}
            assert "DETACH DELETE" not in _strip_identifiers(cypher)


def test_enhanced_schema_cypher_parameterizes_apoc_distinct():
    store = _store()
    store.structured_schema = {
        "metadata": {
            "index": [
                {
                    "label": PAYLOAD,
                    "properties": [PAYLOAD],
                    "type": "RANGE",
                    "size": 1,
                    "distinctValues": 1,
                }
            ]
        }
    }
    store.structured_query = MagicMock(return_value=[{"value": ["x' }) RETURN 1 //"]}])
    cypher, params = store._enhanced_schema_cypher(
        PAYLOAD, [{"property": PAYLOAD, "type": "STRING"}], False
    )
    call = store.structured_query.call_args
    assert "$label" in call.args[0] and "$prop" in call.args[0]
    assert PAYLOAD not in call.args[0]
    assert call.kwargs["param_map"] == {"label": PAYLOAD, "prop": PAYLOAD}
    assert list(params.values()) == [["x' }) RETURN 1 //"]]
    assert "RETURN 1" not in cypher


def test_get_triplets_delete_get_and_vector_query_escape_identifiers():
    store = _store()
    store.get(properties={PAYLOAD: 1})
    store.get_triplets(relation_names=[PAYLOAD], properties={PAYLOAD: 1})
    store.delete(relation_names=[PAYLOAD], properties={PAYLOAD: 1})
    store._supports_vector_index = False
    store.vector_query(
        VectorStoreQuery(
            query_embedding=[0.1],
            similarity_top_k=1,
            filters=MetadataFilters(filters=[MetadataFilter(key=PAYLOAD, value=1)]),
        )
    )
    queries = _queries(store)
    assert len(queries) == 5
    for cypher in queries:
        stripped = _strip_identifiers(cypher)
        assert "MATCH (m)" not in stripped
