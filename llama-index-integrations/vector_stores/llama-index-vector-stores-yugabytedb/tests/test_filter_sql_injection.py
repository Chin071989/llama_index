import pytest
from sqlalchemy.dialects import postgresql

from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.yugabytedb import YBVectorStore

PAYLOAD = "x' OR '1'='1"


def _compile(clause):
    return clause.compile(dialect=postgresql.psycopg2.dialect())


@pytest.fixture
def store() -> YBVectorStore:
    # Filter building does not need a live connection.
    return YBVectorStore.__new__(YBVectorStore)


@pytest.mark.parametrize(
    ("operator", "value"),
    [
        (FilterOperator.EQ, PAYLOAD),
        (FilterOperator.NE, PAYLOAD),
        (FilterOperator.GT, 1.5),
        (FilterOperator.IN, [PAYLOAD, "b"]),
        (FilterOperator.NIN, [PAYLOAD]),
        (FilterOperator.CONTAINS, PAYLOAD),
        (FilterOperator.TEXT_MATCH, PAYLOAD),
        (FilterOperator.TEXT_MATCH_INSENSITIVE, PAYLOAD),
        (FilterOperator.IS_EMPTY, None),
    ],
)
def test_filter_key_and_value_are_bound(
    store: YBVectorStore, operator: FilterOperator, value
) -> None:
    key = "k' OR 1=1 --"
    clause = store._build_filter_clause(
        MetadataFilter(key=key, value=value, operator=operator)
    )
    compiled = _compile(clause)
    sql = str(compiled)

    assert "OR" not in sql
    assert "1=1" not in sql
    assert key in compiled.params.values()


def test_nested_filters_use_unique_bind_names(store: YBVectorStore) -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value="1"),
            MetadataFilter(key="b", value="2"),
            MetadataFilters(
                filters=[MetadataFilter(key="c", value="3")], condition="or"
            ),
        ]
    )
    compiled = _compile(store._recursively_apply_filters(filters))

    assert sorted(v for v in compiled.params.values() if isinstance(v, str)) == [
        "a",
        "b",
        "c",
    ]
    assert sorted(v for v in compiled.params.values() if isinstance(v, float)) == [
        1.0,
        2.0,
        3.0,
    ]


def test_string_value_bound_verbatim(store: YBVectorStore) -> None:
    compiled = _compile(
        store._build_filter_clause(MetadataFilter(key="author", value=PAYLOAD))
    )

    assert PAYLOAD in compiled.params.values()
    assert PAYLOAD not in str(compiled)
