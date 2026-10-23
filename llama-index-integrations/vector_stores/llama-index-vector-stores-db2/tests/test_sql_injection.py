from types import SimpleNamespace
from unittest.mock import MagicMock

from llama_index.core.vector_stores.types import ExactMatchFilter
from llama_index.vector_stores.db2 import DB2LlamaVS

PAYLOAD = "x' OR '1'='1"


def test_meta_filter_values_are_bound() -> None:
    store = SimpleNamespace(metadata_column="metadata")
    params: list = ["doc-1"]

    where = DB2LlamaVS._append_meta_filter_condition(
        store,
        "doc_id in (?)",
        [
            ExactMatchFilter(key="author", value=PAYLOAD),
            ExactMatchFilter(key="k') = 'a' OR ('1", value="v"),
        ],
        params,
    )

    assert PAYLOAD not in where
    assert where == (
        "doc_id in (?) AND JSON_VALUE(metadata, '$.author') = ?"
        " AND JSON_VALUE(metadata, '$.k'') = ''a'' OR (''1') = ?"
    )
    assert params == ["doc-1", PAYLOAD, "v"]


def test_delete_binds_ref_doc_id() -> None:
    cursor = MagicMock()
    client = MagicMock()
    client.cursor.return_value = cursor
    store = SimpleNamespace(_client=client, table_name="nodes")

    DB2LlamaVS.delete(store, PAYLOAD)

    sql, args = cursor.execute.call_args_list[0].args
    assert PAYLOAD not in sql
    assert sql == "DELETE FROM nodes WHERE doc_id = ?"
    assert args == [PAYLOAD]
