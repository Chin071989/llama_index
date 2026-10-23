from types import SimpleNamespace
from unittest.mock import MagicMock

from llama_index.vector_stores.azurecosmosnosql import AzureCosmosDBNoSqlVectorSearch


def test_delete_binds_ref_doc_id_as_parameter() -> None:
    container = MagicMock()
    container.query_items.return_value = [{"id": "n1", "partitionKey": "n1"}]
    store = SimpleNamespace(_container=container, _metadata_key="metadata")
    payload = "x' OR 1=1 OR c.id != '"

    AzureCosmosDBNoSqlVectorSearch.delete(store, payload)

    kwargs = container.query_items.call_args.kwargs
    assert payload not in kwargs["query"]
    assert "@ref_doc_id" in kwargs["query"]
    assert kwargs["parameters"] == [{"name": "@ref_doc_id", "value": payload}]
    container.delete_item.assert_called_once_with("n1", partition_key="n1")
