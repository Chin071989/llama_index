from typing import Any, Dict

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.mariadb import MariaDBVectorStore

PAYLOAD = "x' OR '1'='1"


def _store() -> MariaDBVectorStore:
    # Building the WHERE clause does not need a live connection.
    return MariaDBVectorStore.__new__(MariaDBVectorStore)


def test_filter_keys_and_values_are_bound() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="director",
                value=[PAYLOAD, "Christopher Nolan"],
                operator=FilterOperator.IN,
            ),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="k') OR 1=1 #", value=PAYLOAD),
                    MetadataFilter(key="pages", value=1000),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )
    params: Dict[str, Any] = {}

    where = _store()._filters_to_where_clause(filters, params)

    assert PAYLOAD not in where
    assert "1=1" not in where
    assert where == (
        "JSON_VALUE(metadata, :filter_0_key) IN (:filter_0_value_0, :filter_0_value_1)"
        " AND (JSON_VALUE(metadata, :filter_3_key) = :filter_3_value"
        " OR JSON_VALUE(metadata, :filter_5_key) = :filter_5_value)"
    )
    assert params == {
        "filter_0_key": "$.director",
        "filter_0_value_0": PAYLOAD,
        "filter_0_value_1": "Christopher Nolan",
        "filter_3_key": "$.k') OR 1=1 #",
        "filter_3_value": PAYLOAD,
        "filter_5_key": "$.pages",
        "filter_5_value": 1000,
    }
