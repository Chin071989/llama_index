import pytest
from llama_index.core.vector_stores.types import MetadataFilter
from llama_index.vector_stores.alibabacloud_mysql import AlibabaCloudMySQLVectorStore
from llama_index.vector_stores.alibabacloud_mysql.base import _validate_metadata_key

MALICIOUS_KEYS = [
    "a') OR 1=1 -- ",
    'a"b',
    "a\\b",
    "a`b",
    "a\nb",
    "",
]


@pytest.mark.parametrize("key", MALICIOUS_KEYS)
def test_malicious_filter_keys_are_rejected(key: str) -> None:
    with pytest.raises(ValueError):
        _validate_metadata_key(key)
    with pytest.raises(ValueError):
        AlibabaCloudMySQLVectorStore._build_filter_clause(
            None, MetadataFilter(key=key, value="v"), [0]
        )


@pytest.mark.parametrize("key", ["category", "file_name", "author.name", "a-b"])
def test_regular_filter_keys_are_accepted(key: str) -> None:
    assert _validate_metadata_key(key) == key
