import pytest
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.bigquery.utils import build_where_clause_and_params

PAYLOAD = "x' OR 1=1 --"


@pytest.mark.parametrize(
    "operator",
    [FilterOperator.TEXT_MATCH, FilterOperator.TEXT_MATCH_INSENSITIVE],
)
def test_text_match_value_is_bound_as_parameter(operator: FilterOperator) -> None:
    filters = MetadataFilters(
        filters=[MetadataFilter(key="author", value=PAYLOAD, operator=operator)]
    )

    clause, params = build_where_clause_and_params(filters=filters)

    assert PAYLOAD not in clause
    assert clause.count("?") == 1
    assert len(params) == 1
    assert params[0].value == PAYLOAD


@pytest.mark.parametrize(
    "key", ["a\"') OR TRUE OR ('", "a'b", 'a"b', "a\\b", "a`b", "a\nb", ""]
)
def test_malicious_keys_are_rejected(key: str) -> None:
    filters = MetadataFilters(filters=[MetadataFilter(key=key, value="v")])
    with pytest.raises(ValueError):
        build_where_clause_and_params(filters=filters)


def test_regular_keys_are_accepted() -> None:
    filters = MetadataFilters(filters=[MetadataFilter(key="file name-1.x", value="v")])
    clause, params = build_where_clause_and_params(filters=filters)
    assert clause == "(SAFE.JSON_VALUE(metadata, '$.\"file name-1.x\"') = ?)"
    assert params[0].value == "v"
