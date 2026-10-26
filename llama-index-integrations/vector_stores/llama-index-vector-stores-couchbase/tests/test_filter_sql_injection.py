import pytest
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.couchbase.base import _convert_llamaindex_filters_to_sql


def test_filter_values_are_bound_as_named_parameters() -> None:
    payload = "x' OR 1=1 --"
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="author", value=payload),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="year", value=2020, operator=FilterOperator.GT),
                    MetadataFilter(
                        key="tag", value=["a", "b'"], operator=FilterOperator.IN
                    ),
                ],
                condition=FilterCondition.OR,
            ),
        ]
    )

    clause, params = _convert_llamaindex_filters_to_sql(filters, "metadata")

    assert payload not in clause
    assert clause == (
        "d.`metadata`.`author` = $filter_param_0 AND "
        "(d.`metadata`.`year` > $filter_param_1 OR "
        "d.`metadata`.`tag` IN $filter_param_2)"
    )
    assert params == {
        "filter_param_0": payload,
        "filter_param_1": 2020,
        "filter_param_2": ["a", "b'"],
    }


def test_nested_key_path_is_escaped() -> None:
    filters = MetadataFilters(filters=[MetadataFilter(key="a.b", value="v")])
    clause, _ = _convert_llamaindex_filters_to_sql(filters, "metadata")
    assert clause == "d.`metadata`.`a`.`b` = $filter_param_0"


@pytest.mark.parametrize("key", ["a` = 1 OR `b", "", "a..b"])
def test_malicious_keys_are_rejected(key: str) -> None:
    filters = MetadataFilters(filters=[MetadataFilter(key=key, value="v")])
    with pytest.raises(ValueError):
        _convert_llamaindex_filters_to_sql(filters, "metadata")
