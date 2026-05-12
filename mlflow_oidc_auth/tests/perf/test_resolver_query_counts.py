"""Resolver-chain query-count regression tests.

These tests pin EXACT query counts for the permission-resolution hot path:
``build_user_permission_context``, the experiment-group resolver, and the
workspace cache cold/warm lookup paths. They start as a baseline lock
(captured at the post-Task-1 commit on 2026-05-12) and are updated in
Task 6 to assert the post-optimization numbers.

Why exact-equality (not ``<=``)? Because a silent improvement is just as
interesting to know about as a regression - we want both to fail loud so
the maintainer can decide whether the new number is the new baseline.

See the quick task 260512-o78 for the full optimization plan.
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import patch

import pytest

from mlflow_oidc_auth.config import config
from mlflow_oidc_auth.sqlalchemy_store import SqlAlchemyStore
from mlflow_oidc_auth.tests.perf.conftest import QueryCounter, seed_user_with_groups


@pytest.fixture
def alice_with_three_groups(store: SqlAlchemyStore) -> SqlAlchemyStore:
    """Seed an alice user with 3 groups and yield the store."""
    seed_user_with_groups(store, username="alice", group_names=["g1", "g2", "g3"])
    return store


@pytest.fixture
def workspaces_enabled() -> Iterator[None]:
    """Temporarily enable workspaces + a deterministic source order on the
    real ``config`` singleton.

    We mutate the real object (instead of ``patch.object``) because the
    workspace_cache reads ``config.WORKSPACE_CACHE_MAX_SIZE`` / TTL when it
    lazily builds its backend, and mocking the whole module makes those
    numeric attrs return MagicMocks that break ``cachetools``.
    """
    orig_workspaces = config.MLFLOW_ENABLE_WORKSPACES
    orig_order = config.PERMISSION_SOURCE_ORDER
    config.MLFLOW_ENABLE_WORKSPACES = True
    config.PERMISSION_SOURCE_ORDER = ["user", "group", "regex", "group-regex"]
    try:
        yield
    finally:
        config.MLFLOW_ENABLE_WORKSPACES = orig_workspaces
        config.PERMISSION_SOURCE_ORDER = orig_order


def test_build_user_permission_context_query_count(alice_with_three_groups: SqlAlchemyStore, query_counter: QueryCounter):
    """Lock in current query count for ``build_user_permission_context``.

    baseline 2026-05-12 @ f3ac331 — see quick task 260512-o78
    """
    # Point the module-level ``store`` singleton at our per-test store.
    with patch("mlflow_oidc_auth.utils.batch_permissions.store", alice_with_three_groups):
        from mlflow_oidc_auth.utils.batch_permissions import build_user_permission_context

        query_counter.reset()
        ctx = build_user_permission_context("alice")

    assert ctx.username == "alice"
    # baseline 2026-05-12 @ f3ac331 — see quick task 260512-o78
    # Today's shape: 1 get_groups_ids + 1 list_experiment_perms (each goes
    # through get_user inside the repo = 2 queries each in many cases, then
    # the resolver re-fetches user_groups inside list_user_groups_*).
    assert query_counter.total == 21


def test_get_group_permission_for_user_experiment_query_count(alice_with_three_groups: SqlAlchemyStore, query_counter: QueryCounter):
    """Lock in current query count for the per-resource group resolver.

    Tests ``ExperimentPermissionGroupRepository.get_group_permission_for_user_experiment``,
    which is on the hot path for every protected experiment access.

    baseline 2026-05-12 @ f3ac331 (post-Task-1 JOIN collapse) — see quick task 260512-o78
    """
    alice_with_three_groups.create_group_experiment_permission("g1", "exp-1", "READ")

    query_counter.reset()
    perm = alice_with_three_groups.get_user_groups_experiment_permission("exp-1", "alice")

    assert perm.permission == "READ"
    # baseline 2026-05-12 @ f3ac331 — Post-Task-1, the chain is:
    #   1 get_user(alice) -> 1 query
    #   1 JOIN-collapsed _list_user_groups -> 1 query  (was 2 pre-Task-1)
    #   3 groups × 2 queries (_get_group_permission_or_none does
    #     SqlGroup lookup + permission lookup) -> 6 queries
    # Total: 8. See quick task 260512-o78.
    assert query_counter.total == 8


def test_workspace_cache_cold_lookup_query_count(
    alice_with_three_groups: SqlAlchemyStore,
    query_counter: QueryCounter,
    workspaces_enabled: None,
):
    """Lock in current query count for ``get_workspace_permission_cached`` cold path.

    With workspaces enabled and a user-direct workspace permission seeded,
    the resolver short-circuits at the first source ("user").

    baseline 2026-05-12 @ f3ac331 — see quick task 260512-o78
    """
    alice_with_three_groups.create_workspace_permission("ws-1", "alice", "READ")

    from mlflow_oidc_auth.utils import workspace_cache

    # Force the workspace cache backend to be re-initialized so this test
    # starts with a clean cache namespace.
    workspace_cache._cache = None

    with patch("mlflow_oidc_auth.store.store", alice_with_three_groups):
        workspace_cache.invalidate_workspace_permission("alice", "ws-1")

        query_counter.reset()
        result = workspace_cache.get_workspace_permission_cached("alice", "ws-1")

    assert result is not None
    assert result.name == "READ"
    # baseline 2026-05-12 @ f3ac331 — User-direct source is first in the
    # default PERMISSION_SOURCE_ORDER and resolves in 3 queries
    # (get_workspace_permission does a workspace+username lookup via JOIN
    # but the repo internally re-fetches the user). See quick task 260512-o78.
    assert query_counter.total == 3


def test_workspace_cache_warm_lookup_query_count(
    alice_with_three_groups: SqlAlchemyStore,
    query_counter: QueryCounter,
    workspaces_enabled: None,
):
    """Lock in current query count for the warm-cache path: the second
    lookup after a cold lookup with a known permission must be 0 queries.

    baseline 2026-05-12 @ f3ac331 — see quick task 260512-o78
    """
    alice_with_three_groups.create_workspace_permission("ws-1", "alice", "READ")

    from mlflow_oidc_auth.utils import workspace_cache

    workspace_cache._cache = None

    with patch("mlflow_oidc_auth.store.store", alice_with_three_groups):
        workspace_cache.invalidate_workspace_permission("alice", "ws-1")
        # Cold: populates cache.
        first = workspace_cache.get_workspace_permission_cached("alice", "ws-1")
        assert first is not None

        query_counter.reset()
        # Warm: must hit cache, 0 queries.
        second = workspace_cache.get_workspace_permission_cached("alice", "ws-1")
        assert second is not None

    # baseline 2026-05-12 @ f3ac331 — pure cache hit. See quick task 260512-o78.
    assert query_counter.total == 0
