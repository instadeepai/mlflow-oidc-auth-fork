"""Resolver-chain query-count regression tests.

These tests pin EXACT query counts for the permission-resolution hot path:
``build_user_permission_context``, the experiment-group resolver, and the
workspace cache cold/warm lookup paths. Numbers below were captured at
post-Task-5 (commit 06e458d on branch perf/resolver-context-caching,
2026-05-12) with one user in 3 groups against an in-memory SQLite store.

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


@pytest.fixture
def reset_user_context_cache() -> Iterator[None]:
    """Reset the per-test user_context cache backend so each test starts cold."""
    from mlflow_oidc_auth.utils import batch_permissions, workspace_cache

    batch_permissions._user_context_cache = None
    workspace_cache._cache = None
    yield
    batch_permissions._user_context_cache = None
    workspace_cache._cache = None


def test_build_user_permission_context_query_count(
    alice_with_three_groups: SqlAlchemyStore,
    query_counter: QueryCounter,
):
    """Query count for ``build_user_permission_context`` with workspaces OFF.

    No change vs. baseline because workspaces are off and the workspace
    branch is skipped entirely. See quick task 260512-o78.

    was 21, now 21 — workspaces disabled in this test
    """
    with patch("mlflow_oidc_auth.utils.batch_permissions.store", alice_with_three_groups):
        from mlflow_oidc_auth.utils.batch_permissions import build_user_permission_context

        query_counter.reset()
        ctx = build_user_permission_context("alice")

    assert ctx.username == "alice"
    # was 21, now 21 (workspaces off, no behavior change) — quick task 260512-o78
    assert query_counter.total == 21


def test_get_group_permission_for_user_experiment_query_count(
    alice_with_three_groups: SqlAlchemyStore,
    query_counter: QueryCounter,
):
    """Query count for the per-resource group resolver.

    Post-Task-1 JOIN collapse already dropped this from the pre-task-1
    baseline; no Task 5 change applies here. See quick task 260512-o78.

    was 9 (pre-Task-1, 2-step user-groups fetch), now 8 (post-Task-1 JOIN)
    """
    alice_with_three_groups.create_group_experiment_permission("g1", "exp-1", "READ")

    query_counter.reset()
    perm = alice_with_three_groups.get_user_groups_experiment_permission("exp-1", "alice")

    assert perm.permission == "READ"
    # was 9 (pre-Task-1), now 8 — quick task 260512-o78
    assert query_counter.total == 8


def test_workspace_cache_cold_lookup_query_count(
    alice_with_three_groups: SqlAlchemyStore,
    query_counter: QueryCounter,
    workspaces_enabled: None,
    reset_user_context_cache: None,
):
    """Cold workspace lookup with the new context cache pre-fetches everything.

    The shape changed in Task 5: the resolver now does one full
    ``build_user_permission_context`` (which when workspaces are on
    fetches 4 extra branches) and then reads in-process. Single-lookup
    cold cost rose (3 -> 27) but the entire context is now reusable for
    all subsequent permission checks in the same 30s window at 0 queries
    each, so any request that issues 3+ permission checks comes out
    ahead. See quick task 260512-o78.

    was 3, now 27 — see quick task 260512-o78
    """
    alice_with_three_groups.create_workspace_permission("ws-1", "alice", "READ")

    from mlflow_oidc_auth.utils import workspace_cache

    with (
        patch("mlflow_oidc_auth.store.store", alice_with_three_groups),
        patch("mlflow_oidc_auth.utils.batch_permissions.store", alice_with_three_groups),
    ):
        workspace_cache.invalidate_workspace_permission("alice", "ws-1")

        query_counter.reset()
        result = workspace_cache.get_workspace_permission_cached("alice", "ws-1")

    assert result is not None
    assert result.name == "READ"
    # was 3, now 27 — full context pre-fetch on first call (see quick task 260512-o78)
    assert query_counter.total == 27


def test_workspace_cache_warm_lookup_query_count(
    alice_with_three_groups: SqlAlchemyStore,
    query_counter: QueryCounter,
    workspaces_enabled: None,
    reset_user_context_cache: None,
):
    """Warm path: second lookup must be 0 queries.

    Cache-hit on both the workspace cache AND the user_context cache.

    was 0, now 0 — no change. See quick task 260512-o78.
    """
    alice_with_three_groups.create_workspace_permission("ws-1", "alice", "READ")

    from mlflow_oidc_auth.utils import workspace_cache

    with (
        patch("mlflow_oidc_auth.store.store", alice_with_three_groups),
        patch("mlflow_oidc_auth.utils.batch_permissions.store", alice_with_three_groups),
    ):
        workspace_cache.invalidate_workspace_permission("alice", "ws-1")
        first = workspace_cache.get_workspace_permission_cached("alice", "ws-1")
        assert first is not None

        query_counter.reset()
        second = workspace_cache.get_workspace_permission_cached("alice", "ws-1")
        assert second is not None

    # was 0, now 0 — see quick task 260512-o78
    assert query_counter.total == 0


def test_workspace_cache_second_workspace_warm_via_user_context(
    alice_with_three_groups: SqlAlchemyStore,
    query_counter: QueryCounter,
    workspaces_enabled: None,
    reset_user_context_cache: None,
):
    """Second workspace lookup for the same user should be 0 queries.

    Even though it's a different workspace key (so the workspace_cache
    misses), the user_context cache HITs and the resolver reads in-process.
    This is the primary perf win — N permission checks for one user cost
    O(1) DB after the first build. See quick task 260512-o78.
    """
    alice_with_three_groups.create_workspace_permission("ws-1", "alice", "READ")
    alice_with_three_groups.create_workspace_permission("ws-2", "alice", "EDIT")

    from mlflow_oidc_auth.utils import workspace_cache

    with (
        patch("mlflow_oidc_auth.store.store", alice_with_three_groups),
        patch("mlflow_oidc_auth.utils.batch_permissions.store", alice_with_three_groups),
    ):
        workspace_cache.invalidate_workspace_permission("alice", "ws-1")
        workspace_cache.invalidate_workspace_permission("alice", "ws-2")
        first = workspace_cache.get_workspace_permission_cached("alice", "ws-1")
        assert first is not None

        # Different workspace, same user: workspace cache misses but
        # user_context cache hits.
        query_counter.reset()
        second = workspace_cache.get_workspace_permission_cached("alice", "ws-2")
        assert second is not None

    assert query_counter.total == 0


def test_resolver_does_not_regress_when_context_missing(
    alice_with_three_groups: SqlAlchemyStore,
    query_counter: QueryCounter,
    workspaces_enabled: None,
    reset_user_context_cache: None,
):
    """Guards against future refactors that accidentally bypass the cache.

    Calls the resolver chain twice in the same simulated request (same
    username, no intervening mutation) and asserts the second call's
    query count is 0 (cache-hit path).
    """
    alice_with_three_groups.create_workspace_permission("ws-1", "alice", "READ")

    from mlflow_oidc_auth.utils import workspace_cache

    with (
        patch("mlflow_oidc_auth.store.store", alice_with_three_groups),
        patch("mlflow_oidc_auth.utils.batch_permissions.store", alice_with_three_groups),
    ):
        workspace_cache.invalidate_workspace_permission("alice", "ws-1")
        workspace_cache.get_workspace_permission_cached("alice", "ws-1")  # cold

        query_counter.reset()
        # Force a fresh workspace_cache key check by invalidating that one
        # entry, but leave the user_context cache populated.
        workspace_cache.invalidate_workspace_permission("alice", "ws-1")
        workspace_cache.get_workspace_permission_cached("alice", "ws-1")  # context-cache hit

    assert query_counter.total == 0


def test_invalidation_via_add_user_to_group_forces_rebuild(
    alice_with_three_groups: SqlAlchemyStore,
    query_counter: QueryCounter,
    workspaces_enabled: None,
    reset_user_context_cache: None,
):
    """Cache invalidation reaches the new user_context cache.

    After a mutation that goes through ``_PERMISSION_CUD_METHODS``
    (add_user_to_group), the next resolver call must rebuild the context
    (counter > 0), proving invalidation propagates.
    """
    alice_with_three_groups.create_workspace_permission("ws-1", "alice", "READ")

    from mlflow_oidc_auth.utils import workspace_cache

    with (
        patch("mlflow_oidc_auth.store.store", alice_with_three_groups),
        patch("mlflow_oidc_auth.utils.batch_permissions.store", alice_with_three_groups),
    ):
        workspace_cache.invalidate_workspace_permission("alice", "ws-1")
        first = workspace_cache.get_workspace_permission_cached("alice", "ws-1")
        assert first is not None

        # Confirm we're warm.
        query_counter.reset()
        workspace_cache.invalidate_workspace_permission("alice", "ws-1")
        workspace_cache.get_workspace_permission_cached("alice", "ws-1")
        warm = query_counter.total

        # Mutate -> _PERMISSION_CUD_METHODS wrapper flushes the user_context
        # cache (via flush_user_context_cache).
        alice_with_three_groups.populate_groups(["g4"])
        alice_with_three_groups.add_user_to_group("alice", "g4")

        query_counter.reset()
        workspace_cache.invalidate_workspace_permission("alice", "ws-1")
        workspace_cache.get_workspace_permission_cached("alice", "ws-1")
        after_mutation = query_counter.total

    assert warm == 0, "Sanity check: warm path should be 0 queries"
    assert after_mutation > 0, "After CUD mutation, the resolver must rebuild the context"
