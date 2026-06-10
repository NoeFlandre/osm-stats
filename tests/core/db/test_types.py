from src.core.db.types import DB


def test_db_is_a_protocol_with_execute_query():
    # Structural: a class with execute_query(query, params=None) -> DataFrame
    # satisfies the protocol. We just check the surface exists.
    members = set(dir(DB))
    assert "execute_query" in members
