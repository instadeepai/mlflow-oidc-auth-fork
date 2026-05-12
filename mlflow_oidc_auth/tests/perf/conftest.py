"""Perf-test fixtures: SQLAlchemy query counter + seed helpers.

All fixtures are explicitly ``scope="function"`` to guarantee a clean
in-memory SQLite store per test. Cross-test seed pollution would silently
shift query counts (Postgres/SQLite cache state, autoflush ordering) and
turn the perf-regression suite into a coin toss; per-function isolation
keeps the assertions deterministic.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterator, List
from unittest.mock import patch

import pytest
import sqlalchemy
from sqlalchemy.engine import Engine

from mlflow_oidc_auth.sqlalchemy_store import SqlAlchemyStore


_COUNTED_PREFIXES = ("SELECT", "INSERT", "UPDATE", "DELETE")


class QueryCounter:
    """Counts SELECT/INSERT/UPDATE/DELETE statements executed on an engine.

    Engine-level ``BEGIN`` / ``COMMIT`` / ``ROLLBACK`` / ``SAVEPOINT`` /
    ``RELEASE`` / ``PRAGMA`` traffic is filtered out so the counter reflects
    actual data operations, not transaction noise.
    """

    def __init__(self) -> None:
        self.total: int = 0
        self.by_op: dict[str, int] = defaultdict(int)
        self.statements: List[str] = []

    def record(self, statement: str) -> None:
        normalized = statement.lstrip()
        head = normalized.split(None, 1)[0].upper() if normalized else ""
        if head not in _COUNTED_PREFIXES:
            return
        self.total += 1
        self.by_op[head] += 1
        self.statements.append(normalized)

    def reset(self) -> None:
        self.total = 0
        self.by_op.clear()
        self.statements.clear()


@pytest.fixture(scope="function")
def store() -> Iterator[SqlAlchemyStore]:
    """A fresh in-memory ``SqlAlchemyStore`` per test.

    Matches the pattern in ``mlflow_oidc_auth/tests/test_sqlalchemy_store.py``
    but explicitly function-scoped to avoid seed leakage across perf tests.
    """
    with patch("mlflow_oidc_auth.sqlalchemy_store.dbutils.migrate_if_needed"):
        s = SqlAlchemyStore()
        s.init_db("sqlite:///:memory:")
        # Create schema manually since migrations are mocked out.
        from mlflow_oidc_auth.db.models._base import Base

        Base.metadata.create_all(s.engine)
        yield s


@pytest.fixture(scope="function")
def query_counter(store: SqlAlchemyStore) -> Iterator[QueryCounter]:
    """Subscribe to ``before_cursor_execute`` on the store engine.

    Yields a :class:`QueryCounter`. Listener is removed on teardown so the
    next test starts with a clean engine event registry.
    """
    counter = QueryCounter()

    def _listener(conn, cursor, statement, parameters, context, executemany):  # noqa: D401, ANN001
        counter.record(statement)

    engine: Engine = store.engine
    sqlalchemy.event.listen(engine, "before_cursor_execute", _listener)
    try:
        yield counter
    finally:
        sqlalchemy.event.remove(engine, "before_cursor_execute", _listener)


def seed_user_with_groups(
    store: SqlAlchemyStore,
    *,
    username: str,
    group_names: List[str],
    is_admin: bool = False,
) -> None:
    """Create a user + N groups + membership rows in one helper.

    Used by perf tests to share setup without duplicating the same 4-5
    store calls in every test body.

    :param store: An initialized :class:`SqlAlchemyStore`.
    :param username: Username to create.
    :param group_names: Group names to create and add the user to.
    :param is_admin: Whether the user is an admin.
    """
    store.create_user(
        username=username,
        password="pw",
        display_name=username,
        is_admin=is_admin,
        is_service_account=False,
    )
    # ``populate_groups`` skips already-existing groups, so it is safe to
    # call with the full list at once without per-name uniqueness handling.
    store.populate_groups(list(group_names))
    for group_name in group_names:
        store.add_user_to_group(username, group_name)


# Re-export the seed helper as a fixture-style callable so tests can request
# it via fixture injection if preferred. Tests can also just import it.
@pytest.fixture(scope="function")
def seed_user_with_groups_fixture() -> Callable[..., None]:
    return seed_user_with_groups
