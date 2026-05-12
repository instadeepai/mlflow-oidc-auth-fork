"""Tests for workspace permission cache module."""

import pytest
from unittest.mock import MagicMock, patch

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from mlflow_oidc_auth.permissions import EDIT, MANAGE, READ, USE, get_permission


class TestGetWorkspacePermissionCached:
    """Test get_workspace_permission_cached() function."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset module-level cache between tests."""
        import mlflow_oidc_auth.utils.workspace_cache as wc

        wc._cache = None
        yield
        wc._cache = None

    def test_returns_none_when_workspaces_disabled(self):
        """get_workspace_permission_cached() returns None when MLFLOW_ENABLE_WORKSPACES is False."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            get_workspace_permission_cached,
        )

        mock_config = MagicMock()
        mock_config.MLFLOW_ENABLE_WORKSPACES = False

        with patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config):
            result = get_workspace_permission_cached("user1", "ws1")
            assert result is None

    def test_default_workspace_resolved_via_normal_lookup(self):
        """get_workspace_permission_cached('user1', 'default') uses normal lookup — no implicit grant."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            get_workspace_permission_cached,
        )

        mock_config = MagicMock()
        mock_config.MLFLOW_ENABLE_WORKSPACES = True
        mock_config.WORKSPACE_CACHE_MAX_SIZE = 1024
        mock_config.WORKSPACE_CACHE_TTL_SECONDS = 300

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch("mlflow_oidc_auth.utils.workspace_cache._lookup_workspace_permission") as mock_lookup,
        ):
            mock_lookup.return_value = READ
            result = get_workspace_permission_cached("user1", "default")
            assert result == READ
            mock_lookup.assert_called_once_with("user1", "default")

    def test_default_workspace_returns_none_when_no_permission(self):
        """get_workspace_permission_cached('user1', 'default') returns None when user has no permission — no implicit grant."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            get_workspace_permission_cached,
        )

        mock_config = MagicMock()
        mock_config.MLFLOW_ENABLE_WORKSPACES = True
        mock_config.WORKSPACE_CACHE_MAX_SIZE = 1024
        mock_config.WORKSPACE_CACHE_TTL_SECONDS = 300

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch("mlflow_oidc_auth.utils.workspace_cache._lookup_workspace_permission") as mock_lookup,
        ):
            mock_lookup.return_value = None
            result = get_workspace_permission_cached("user1", "default")
            assert result is None
            mock_lookup.assert_called_once_with("user1", "default")

    def test_calls_lookup_on_cache_miss(self):
        """get_workspace_permission_cached() calls _lookup_workspace_permission on cache miss."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            get_workspace_permission_cached,
        )

        mock_config = MagicMock()
        mock_config.MLFLOW_ENABLE_WORKSPACES = True
        mock_config.WORKSPACE_CACHE_MAX_SIZE = 1024
        mock_config.WORKSPACE_CACHE_TTL_SECONDS = 300

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch("mlflow_oidc_auth.utils.workspace_cache._lookup_workspace_permission") as mock_lookup,
        ):
            mock_lookup.return_value = READ
            result = get_workspace_permission_cached("user1", "ws1")
            mock_lookup.assert_called_once_with("user1", "ws1")
            assert result == READ

    def test_returns_cached_value_on_hit(self):
        """get_workspace_permission_cached() returns cached value on cache hit without DB query."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            get_workspace_permission_cached,
        )

        mock_config = MagicMock()
        mock_config.MLFLOW_ENABLE_WORKSPACES = True
        mock_config.WORKSPACE_CACHE_MAX_SIZE = 1024
        mock_config.WORKSPACE_CACHE_TTL_SECONDS = 300

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch("mlflow_oidc_auth.utils.workspace_cache._lookup_workspace_permission") as mock_lookup,
        ):
            mock_lookup.return_value = MANAGE
            # First call — cache miss
            get_workspace_permission_cached("user1", "ws1")
            # Second call — cache hit
            result = get_workspace_permission_cached("user1", "ws1")
            assert result == MANAGE
            # lookup should only be called once (not twice)
            assert mock_lookup.call_count == 1

    def test_does_not_cache_none_results(self):
        """get_workspace_permission_cached() does NOT cache None results."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            get_workspace_permission_cached,
        )

        mock_config = MagicMock()
        mock_config.MLFLOW_ENABLE_WORKSPACES = True
        mock_config.WORKSPACE_CACHE_MAX_SIZE = 1024
        mock_config.WORKSPACE_CACHE_TTL_SECONDS = 300

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch("mlflow_oidc_auth.utils.workspace_cache._lookup_workspace_permission") as mock_lookup,
        ):
            mock_lookup.return_value = None
            # First call — returns None, should not cache
            result1 = get_workspace_permission_cached("user1", "ws1")
            assert result1 is None
            # Second call — should call lookup again because None was not cached
            result2 = get_workspace_permission_cached("user1", "ws1")
            assert result2 is None
            assert mock_lookup.call_count == 2


def _make_user_ctx(
    *,
    user_workspace=None,
    group_workspace=None,
    user_regex=None,
    group_regex=None,
):
    """Build a UserPermissionContext stub with workspace fields populated.

    Non-workspace fields default to the dataclass defaults (empty containers),
    which is correct for these tests since the resolver only reads the
    workspace branch.
    """
    from mlflow_oidc_auth.utils.batch_permissions import UserPermissionContext

    return UserPermissionContext(
        username="user1",
        group_ids=[1] if (group_workspace or group_regex) else [],
        user_experiment_permissions={},
        group_experiment_permissions={},
        experiment_regex_permissions=[],
        group_experiment_regex_permissions=[],
        user_model_permissions={},
        group_model_permissions={},
        model_regex_permissions=[],
        group_model_regex_permissions=[],
        prompt_regex_permissions=[],
        group_prompt_regex_permissions=[],
        user_workspace_permissions=user_workspace or {},
        group_workspace_permissions=group_workspace or {},
        workspace_regex_permissions=user_regex or [],
        group_workspace_regex_permissions=group_regex or [],
    )


class TestLookupWorkspacePermission:
    """Test _lookup_workspace_permission() function.

    Post-quick-task-260512-o78, the resolver reads from a cached
    ``UserPermissionContext`` instead of issuing per-source DB queries,
    so these tests stub the context fetch.
    """

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset module-level cache between tests."""
        import mlflow_oidc_auth.utils.workspace_cache as wc

        wc._cache = None
        yield
        wc._cache = None

    def test_tries_user_permission_first(self):
        """_lookup_workspace_permission() tries user-level permission first."""
        from mlflow_oidc_auth.utils.workspace_cache import _lookup_workspace_permission

        mock_config = MagicMock()
        mock_config.PERMISSION_SOURCE_ORDER = ["user", "group", "regex", "group-regex"]

        ctx = _make_user_ctx(user_workspace={"ws1": "MANAGE"})

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch(
                "mlflow_oidc_auth.utils.batch_permissions.get_or_build_user_permission_context",
                return_value=ctx,
            ),
        ):
            result = _lookup_workspace_permission("user1", "ws1")
            assert result == MANAGE

    def test_falls_through_to_group_permission(self):
        """_lookup_workspace_permission() tries group-level when user-level is missing."""
        from mlflow_oidc_auth.utils.workspace_cache import _lookup_workspace_permission

        mock_config = MagicMock()
        mock_config.PERMISSION_SOURCE_ORDER = ["user", "group", "regex", "group-regex"]

        ctx = _make_user_ctx(group_workspace={"ws1": "READ"})

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch(
                "mlflow_oidc_auth.utils.batch_permissions.get_or_build_user_permission_context",
                return_value=ctx,
            ),
        ):
            result = _lookup_workspace_permission("user1", "ws1")
            assert result == READ

    def test_returns_none_when_no_permission(self):
        """_lookup_workspace_permission() returns None when context has no relevant entries."""
        from mlflow_oidc_auth.utils.workspace_cache import _lookup_workspace_permission

        mock_config = MagicMock()
        mock_config.PERMISSION_SOURCE_ORDER = ["user", "group", "regex", "group-regex"]

        ctx = _make_user_ctx()  # All workspace branches empty.

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch(
                "mlflow_oidc_auth.utils.batch_permissions.get_or_build_user_permission_context",
                return_value=ctx,
            ),
        ):
            result = _lookup_workspace_permission("user1", "ws1")
            assert result is None

    def test_resolver_reads_context_not_store(self):
        """The resolver must not call any store.* method when the context is provided."""
        from mlflow_oidc_auth.utils.workspace_cache import _lookup_workspace_permission

        mock_config = MagicMock()
        mock_config.PERMISSION_SOURCE_ORDER = ["user", "group", "regex", "group-regex"]

        ctx = _make_user_ctx(user_workspace={"ws1": "MANAGE"})
        mock_store = MagicMock()  # Every attribute access raises on .return_value being unset is fine - we check call_count below.

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch("mlflow_oidc_auth.store.store", mock_store),
            patch(
                "mlflow_oidc_auth.utils.batch_permissions.get_or_build_user_permission_context",
                return_value=ctx,
            ),
        ):
            result = _lookup_workspace_permission("user1", "ws1")

        assert result == MANAGE
        # None of the resolver's old store methods should have been called.
        mock_store.get_workspace_permission.assert_not_called()
        mock_store.get_user_groups_workspace_permission.assert_not_called()
        mock_store.list_workspace_regex_permissions.assert_not_called()
        mock_store.get_groups_ids_for_user.assert_not_called()
        mock_store.list_workspace_group_regex_permissions_for_groups_ids.assert_not_called()


class TestFlushWorkspaceCache:
    """Test flush_workspace_cache() function."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset module-level cache between tests."""
        import mlflow_oidc_auth.utils.workspace_cache as wc

        wc._cache = None
        yield
        wc._cache = None

    def test_flush_clears_all_entries(self):
        """flush_workspace_cache() clears the entire cache."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            flush_workspace_cache,
            _get_cache,
        )

        mock_config = MagicMock()
        mock_config.WORKSPACE_CACHE_MAX_SIZE = 1024
        mock_config.WORKSPACE_CACHE_TTL_SECONDS = 300
        mock_config.CACHE_BACKEND = "local"

        with patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config):
            cache = _get_cache()
            cache.set("user1:ws1", MANAGE)
            cache.set("user2:ws2", READ)
            assert cache.get("user1:ws1") == MANAGE
            assert cache.get("user2:ws2") == READ

            flush_workspace_cache()
            assert cache.get("user1:ws1") is None
            assert cache.get("user2:ws2") is None

    def test_flush_on_empty_cache(self):
        """flush_workspace_cache() succeeds even on empty cache."""
        from mlflow_oidc_auth.utils.workspace_cache import flush_workspace_cache

        mock_config = MagicMock()
        mock_config.WORKSPACE_CACHE_MAX_SIZE = 1024
        mock_config.WORKSPACE_CACHE_TTL_SECONDS = 300
        mock_config.CACHE_BACKEND = "local"

        with patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config):
            flush_workspace_cache()  # Should not raise


class TestMatchWorkspaceRegexPermission:
    """Test _match_workspace_regex_permission() function."""

    def _make_regex_perm(self, regex, priority, permission):
        """Helper to create a mock regex permission entity."""
        perm = MagicMock()
        perm.regex = regex
        perm.priority = priority
        perm.permission = permission
        return perm

    def test_returns_none_for_empty_list(self):
        """Returns None when no regexes provided."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            _match_workspace_regex_permission,
        )

        result = _match_workspace_regex_permission([], "prod-workspace")
        assert result is None

    def test_returns_permission_for_matching_regex(self):
        """Returns permission for the first matching regex."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            _match_workspace_regex_permission,
        )

        regexes = [self._make_regex_perm("^prod-.*", 10, "MANAGE")]
        result = _match_workspace_regex_permission(regexes, "prod-workspace")
        assert result == MANAGE

    def test_returns_none_when_no_match(self):
        """Returns None when no regex matches the workspace name."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            _match_workspace_regex_permission,
        )

        regexes = [self._make_regex_perm("^staging-.*", 10, "READ")]
        result = _match_workspace_regex_permission(regexes, "prod-workspace")
        assert result is None

    def test_priority_ordering_lower_wins(self):
        """Lower priority number wins (D-07)."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            _match_workspace_regex_permission,
        )

        regexes = [
            self._make_regex_perm("^prod-.*", 5, "READ"),
            self._make_regex_perm("^prod-.*", 10, "MANAGE"),
        ]
        result = _match_workspace_regex_permission(regexes, "prod-workspace")
        assert result == READ

    def test_same_priority_most_permissive_wins(self):
        """When multiple regexes match at the same priority, return most permissive (D-08)."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            _match_workspace_regex_permission,
        )

        regexes = [
            self._make_regex_perm("^prod-.*", 10, "READ"),
            self._make_regex_perm("^prod-work.*", 10, "MANAGE"),
        ]
        result = _match_workspace_regex_permission(regexes, "prod-workspace")
        assert result == MANAGE

    def test_short_circuits_on_higher_priority_match(self):
        """Stops checking once a higher-priority tier is exhausted."""
        from mlflow_oidc_auth.utils.workspace_cache import (
            _match_workspace_regex_permission,
        )

        regexes = [
            self._make_regex_perm("^prod-.*", 5, "EDIT"),
            self._make_regex_perm("^prod-work.*", 10, "MANAGE"),
        ]
        # Even though the second regex also matches and has higher permission,
        # it's at lower priority tier so should be ignored
        result = _match_workspace_regex_permission(regexes, "prod-workspace")
        assert result == EDIT


class TestResolveUserRegex:
    """Test _resolve_user_regex() function."""

    def test_matches_workspace_against_user_regexes(self):
        """Calls store.list_workspace_regex_permissions and matches against workspace name."""
        from mlflow_oidc_auth.utils.workspace_cache import _resolve_user_regex

        mock_store = MagicMock()
        regex_perm = MagicMock()
        regex_perm.regex = "^prod-.*"
        regex_perm.priority = 10
        regex_perm.permission = "MANAGE"
        mock_store.list_workspace_regex_permissions.return_value = [regex_perm]

        result = _resolve_user_regex(mock_store, "user1", "prod-workspace")
        assert result == MANAGE
        mock_store.list_workspace_regex_permissions.assert_called_once_with("user1")

    def test_returns_none_when_store_returns_empty_list(self):
        """Returns None when store returns empty list."""
        from mlflow_oidc_auth.utils.workspace_cache import _resolve_user_regex

        mock_store = MagicMock()
        mock_store.list_workspace_regex_permissions.return_value = []

        result = _resolve_user_regex(mock_store, "user1", "prod-workspace")
        assert result is None

    def test_returns_none_on_store_exception(self):
        """Returns None on store exception."""
        from mlflow_oidc_auth.utils.workspace_cache import _resolve_user_regex

        mock_store = MagicMock()
        mock_store.list_workspace_regex_permissions.side_effect = Exception("DB error")

        result = _resolve_user_regex(mock_store, "user1", "prod-workspace")
        assert result is None


class TestResolveGroupRegex:
    """Test _resolve_group_regex() function."""

    def test_matches_workspace_against_group_regexes(self):
        """Gets user's group IDs and matches workspace against group regex permissions."""
        from mlflow_oidc_auth.utils.workspace_cache import _resolve_group_regex

        mock_store = MagicMock()
        mock_store.get_groups_ids_for_user.return_value = [1, 2]
        regex_perm = MagicMock()
        regex_perm.regex = "^team-.*"
        regex_perm.priority = 5
        regex_perm.permission = "EDIT"
        mock_store.list_workspace_group_regex_permissions_for_groups_ids.return_value = [regex_perm]

        result = _resolve_group_regex(mock_store, "user1", "team-workspace")
        assert result == EDIT
        mock_store.get_groups_ids_for_user.assert_called_once_with("user1")
        mock_store.list_workspace_group_regex_permissions_for_groups_ids.assert_called_once_with([1, 2])

    def test_returns_none_when_user_has_no_groups(self):
        """Returns None when user has no groups."""
        from mlflow_oidc_auth.utils.workspace_cache import _resolve_group_regex

        mock_store = MagicMock()
        mock_store.get_groups_ids_for_user.return_value = []

        result = _resolve_group_regex(mock_store, "user1", "team-workspace")
        assert result is None

    def test_returns_none_when_no_group_regex_matches(self):
        """Returns None when no group regex matches the workspace."""
        from mlflow_oidc_auth.utils.workspace_cache import _resolve_group_regex

        mock_store = MagicMock()
        mock_store.get_groups_ids_for_user.return_value = [1]
        regex_perm = MagicMock()
        regex_perm.regex = "^staging-.*"
        regex_perm.priority = 10
        regex_perm.permission = "READ"
        mock_store.list_workspace_group_regex_permissions_for_groups_ids.return_value = [regex_perm]

        result = _resolve_group_regex(mock_store, "user1", "prod-workspace")
        assert result is None

    def test_returns_none_on_store_exception(self):
        """Returns None on store exception."""
        from mlflow_oidc_auth.utils.workspace_cache import _resolve_group_regex

        mock_store = MagicMock()
        mock_store.get_groups_ids_for_user.side_effect = Exception("DB error")

        result = _resolve_group_regex(mock_store, "user1", "team-workspace")
        assert result is None


class TestLookupWithRegexSources:
    """Integration-level tests for _lookup_workspace_permission() with regex sources.

    Post-quick-task-260512-o78, the resolver reads regex permissions out of
    the cached ``UserPermissionContext`` rather than calling the regex
    listers directly. These tests build a context with the regex branch
    populated and assert the priority/short-circuit semantics survive.
    """

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset module-level cache between tests."""
        import mlflow_oidc_auth.utils.workspace_cache as wc

        wc._cache = None
        yield
        wc._cache = None

    def test_falls_through_to_user_regex(self):
        """When user-direct and group-direct are absent, user-regex matches."""
        from mlflow_oidc_auth.utils.workspace_cache import _lookup_workspace_permission

        mock_config = MagicMock()
        mock_config.PERMISSION_SOURCE_ORDER = ["user", "group", "regex", "group-regex"]

        regex_perm = MagicMock()
        regex_perm.regex = "^prod-.*"
        regex_perm.priority = 10
        regex_perm.permission = "EDIT"
        ctx = _make_user_ctx(user_regex=[regex_perm])

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch(
                "mlflow_oidc_auth.utils.batch_permissions.get_or_build_user_permission_context",
                return_value=ctx,
            ),
        ):
            result = _lookup_workspace_permission("user1", "prod-workspace")
            assert result == EDIT

    def test_falls_through_to_group_regex(self):
        """When all earlier sources are empty, group-regex matches."""
        from mlflow_oidc_auth.utils.workspace_cache import _lookup_workspace_permission

        mock_config = MagicMock()
        mock_config.PERMISSION_SOURCE_ORDER = ["user", "group", "regex", "group-regex"]

        group_regex_perm = MagicMock()
        group_regex_perm.regex = "^team-.*"
        group_regex_perm.priority = 5
        group_regex_perm.permission = "USE"
        ctx = _make_user_ctx(group_regex=[group_regex_perm])

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch(
                "mlflow_oidc_auth.utils.batch_permissions.get_or_build_user_permission_context",
                return_value=ctx,
            ),
        ):
            result = _lookup_workspace_permission("user1", "team-workspace")
            assert result == USE

    def test_respects_custom_source_order(self):
        """When source order puts regex before user, regex wins over user-direct."""
        from mlflow_oidc_auth.utils.workspace_cache import _lookup_workspace_permission

        mock_config = MagicMock()
        mock_config.PERMISSION_SOURCE_ORDER = ["regex", "user"]

        regex_perm = MagicMock()
        regex_perm.regex = "^prod-.*"
        regex_perm.priority = 10
        regex_perm.permission = "READ"
        ctx = _make_user_ctx(
            user_workspace={"prod-workspace": "MANAGE"},
            user_regex=[regex_perm],
        )

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch(
                "mlflow_oidc_auth.utils.batch_permissions.get_or_build_user_permission_context",
                return_value=ctx,
            ),
        ):
            result = _lookup_workspace_permission("user1", "prod-workspace")
            # regex is first in order, so READ wins even though user-direct has MANAGE.
            assert result == READ

    def test_invalid_source_name_skipped(self):
        """Invalid source names are skipped without error."""
        from mlflow_oidc_auth.utils.workspace_cache import _lookup_workspace_permission

        mock_config = MagicMock()
        mock_config.PERMISSION_SOURCE_ORDER = ["invalid-source", "user"]

        ctx = _make_user_ctx(user_workspace={"ws1": "MANAGE"})

        with (
            patch("mlflow_oidc_auth.utils.workspace_cache.config", mock_config),
            patch(
                "mlflow_oidc_auth.utils.batch_permissions.get_or_build_user_permission_context",
                return_value=ctx,
            ),
        ):
            result = _lookup_workspace_permission("user1", "ws1")
            assert result == MANAGE
