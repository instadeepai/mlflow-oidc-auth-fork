---
quick_id: 260512-o78
slug: reduce-per-request-db-load-for-permissio
date: 2026-05-12
status: planned
---

# PLAN — Reduce per-request DB load for permission resolution

## Goal

Reduce per-request resolver-chain DB queries from "4-6+ per resolution" to "≤2 in the cache-hit path / ≤N+1 in the cold path (N = #groups for user)" by collapsing the two-step `_list_user_groups` fetch into one JOIN and by extending the per-request `UserPermissionContext` to pre-fetch and hold the workspace branch of the chain, with query-count regression tests locking in both before and after.

## Approach summary

We take the surgical-and-additive route: (1) collapse the two-step fetch at `repository/_base.py:255` into a single `SqlGroup ↔ SqlUserGroup` JOIN — purely additive, drops one query off **every** group-permission check across all `BaseGroupPermissionRepository` subclasses (experiment, registered model, prompt, scorer, gateway endpoint, gateway secret, gateway model definition) without touching any caller. (2) Extend `UserPermissionContext` in `utils/batch_permissions.py` with workspace direct / regex maps (additive fields only — signature of `build_user_permission_context()` unchanged). (3) Add a thin `get_or_build_user_permission_context(username)` helper that caches the built context in the existing `permissions` `CacheBackend` (so cache invalidation is automatic via the existing `_PERMISSION_CUD_METHODS` wrapper at `sqlalchemy_store.py:1075-1217` — no new invalidation path needed). (4) Rewire `workspace_cache._resolve_group_direct` / `_resolve_group_regex` / `_resolve_user_direct` / `_resolve_user_regex` to read from the context when one is available, falling back to today's direct store calls otherwise. (5) Add a `tests/perf/` directory with a SQLAlchemy `before_cursor_execute` query-counter fixture (plain pytest, no new deps), record CURRENT counts first, then prove the optimization drops them. The `utils/permissions.py` resolver (`PERMISSION_REGISTRY` / `resolve_permission`) is **not refactored** in this quick task — its existing TTL cache + the new `_base.py` JOIN already cover its hot path. `repository/group.py:115` has the same two-step pattern but is admin-only — fixed opportunistically in the same diff as task 1 since it's a 4-line change.

Non-obvious decisions locked in:

- **No signature change.** `build_user_permission_context()` keeps its current signature. New fields on `UserPermissionContext` are added as `Optional` (default `None`/empty list/empty dict) so existing callers and tests aren't perturbed.
- **Cache key shape:** `f"user_ctx:{username}"` in the existing `permissions` namespace (same `CacheBackend` instance as `resolve_permission`). Reuses the existing flush — no separate invalidation surface.
- **Tests location:** `mlflow_oidc_auth/tests/perf/` (sibling to `mlflow_oidc_auth/tests/utils/`) so they pick up the existing `tox` discovery without config changes. Marker-free; runs as part of the default `pytest` invocation.
- **JOIN style:** stay on Core-style `session.query(...).join(...).filter(...)` to match the rest of the file. No SQLAlchemy 2.x `select()` rewrite in this diff.
- **Workspace context fields are populated lazily.** `build_user_permission_context()` only fetches workspace fields when `config.MLFLOW_ENABLE_WORKSPACES` is true, to keep cold-path cost the same for non-workspace deployments.

## Tasks

### Task 1: Collapse the two-step user-groups fetch into a single JOIN

- **Files**: `mlflow_oidc_auth/repository/_base.py`, `mlflow_oidc_auth/repository/group.py`
- **What**:
  - In `_base.py` rewrite `_list_user_groups()` (lines 246-256) so the body becomes a single query: `session.query(SqlGroup.group_name).join(SqlUserGroup, SqlUserGroup.group_id == SqlGroup.id).filter(SqlUserGroup.user_id == user.id).all()` and return `[name for (name,) in rows]`. The `get_user(session, username)` call stays (still needed to translate username → user.id and to keep "user not found" exception semantics identical).
  - In `group.py` apply the same JOIN collapse to `GroupRepository.list_groups_for_user()` (lines 106-116) since it has the identical anti-pattern. This is the "trivial in-same-diff cleanup" carve-out from CONTEXT.md "Out of scope".
  - Do NOT touch `GroupRepository.list_group_ids_for_user()` (lines 118-127) — it only reads from `user_groups` and already filters at the DB.
- **Verify**: Existing tests in `mlflow_oidc_auth/tests/repository/test_experiment_permission_group.py::test__list_user_groups` still pass; run `pytest mlflow_oidc_auth/tests/repository -k "list_user_groups or list_groups_for_user"`. No behavior change expected.
- **Commit message**: `perf(repository): collapse two-step user-groups fetch into one JOIN`

### Task 2: Add `tests/perf/` scaffolding with a SQLAlchemy query-counter fixture

- **Files**: `mlflow_oidc_auth/tests/perf/__init__.py` (new, empty), `mlflow_oidc_auth/tests/perf/conftest.py` (new)
- **What**:
  - Create the `perf/` package and a `conftest.py` that exposes a `query_counter` fixture. Implementation: subscribe to `sqlalchemy.event.listen(engine, "before_cursor_execute", ...)` against the session-maker's engine and return a context manager / counter object with `.total`, `.by_table`, and a `reset()` method.
  - Fixture must work against the project's existing in-memory test store (look at how `mlflow_oidc_auth/tests/test_sqlalchemy_store.py` builds its store — reuse the same `SqlAlchemyStore` construction so tests don't need their own DB setup).
  - Provide a sibling helper `seed_user_with_groups(store, *, username, group_names, ...)` that creates a user, N groups, and membership rows — needed by Task 3 so multiple perf tests share the same setup without duplication.
  - No new dependencies — pure stdlib + SQLAlchemy event API + pytest.
- **Verify**: Run `pytest mlflow_oidc_auth/tests/perf -x`. With no real test files yet, pytest reports "collected 0 items"; fixture import must succeed (no `ImportError`).
- **Commit message**: `test(perf): add query-count fixture and seed helpers for resolver perf tests`

### Task 3: Add baseline perf tests asserting CURRENT (pre-optimization) query counts

- **Files**: `mlflow_oidc_auth/tests/perf/test_resolver_query_counts.py` (new)
- **What**:
  - Add tests that exercise the hot paths and assert query counts at TODAY's levels. Concretely:
    - `test_build_user_permission_context_query_count`: seed 1 user + 3 groups, call `build_user_permission_context("alice")` inside `query_counter`, assert the count is whatever it currently is (run once locally first to capture; record the literal number in the test with a comment `# baseline 2026-05-12 — drops in task 6`).
    - `test_get_group_permission_for_user_resource_query_count`: invoke `ExperimentPermissionGroupRepository.get_group_permission_for_user_experiment(experiment_id, "alice")` and capture the count (post-task-1 number).
    - `test_workspace_cache_cold_lookup_query_count`: with `MLFLOW_ENABLE_WORKSPACES=True`, call `get_workspace_permission_cached("alice", "ws-1")` after flushing the workspace cache, capture the count (this is the `_resolve_group_direct` + `_resolve_group_regex` chain — currently makes multiple queries per source).
  - Each assertion uses `assert counter.total == <baseline>` (exact equality, NOT `<=`) so a regression OR an improvement both fail loud. Task 6 will edit the expected values.
  - Reuse the `query_counter` and `seed_user_with_groups` fixtures from Task 2.
- **Verify**: `pytest mlflow_oidc_auth/tests/perf/test_resolver_query_counts.py -v` — all three tests pass on the current main code.
- **Commit message**: `test(perf): lock in current resolver query counts as regression baseline`

### Task 4: Extend `UserPermissionContext` with pre-fetched workspace chain (additive)

- **Files**: `mlflow_oidc_auth/utils/batch_permissions.py`, `mlflow_oidc_auth/tests/utils/test_batch_permissions.py`
- **What**:
  - Add optional fields to the `UserPermissionContext` dataclass (lines 31-64): `user_workspace_permissions: Dict[str, str]`, `group_workspace_permissions: Dict[str, str]`, `workspace_regex_permissions: List[WorkspaceRegexPermission]`, `group_workspace_regex_permissions: List[WorkspaceGroupRegexPermission]`. All default to empty containers via `field(default_factory=...)` so existing call sites and `__init__` orderings keep working.
  - Extend `build_user_permission_context()` (lines 67-123): when `config.MLFLOW_ENABLE_WORKSPACES` is true, fetch the four workspace branches in one query each via existing store methods (`store.list_workspace_permissions_for_user(username)` if it exists — otherwise add a thin store method that returns `[(workspace, permission), ...]` for a user; `store.list_workspace_regex_permissions(username)`; `store.list_workspace_group_regex_permissions_for_groups_ids(group_ids)`; group-direct via a new `store.list_user_groups_workspace_permissions(username)` that JOINs `workspace_group_permissions` to `user_groups`). When `MLFLOW_ENABLE_WORKSPACES` is false, leave the new fields at their defaults.
  - If any of those store methods don't already exist in `sqlalchemy_store.py`, add the missing ones (using JOINs, not Python-side filtering) as part of this task. Search before adding — the workspace regex listers do exist at lines ~1006 and ~1036.
  - Update `mlflow_oidc_auth/tests/utils/test_batch_permissions.py` `TestBuildUserPermissionContext` (line 244+) with at least one test asserting the new fields are populated when `MLFLOW_ENABLE_WORKSPACES=True` and stay empty when false.
- **Verify**: `pytest mlflow_oidc_auth/tests/utils/test_batch_permissions.py -v` — all existing tests still pass, new test passes.
- **Commit message**: `feat(batch-permissions): pre-fetch workspace branch into UserPermissionContext`

### Task 5: Add per-request context caching helper and wire workspace_cache to use it

- **Files**: `mlflow_oidc_auth/utils/batch_permissions.py`, `mlflow_oidc_auth/utils/workspace_cache.py`, `mlflow_oidc_auth/tests/utils/test_workspace_cache.py`
- **What**:
  - In `batch_permissions.py` add `get_or_build_user_permission_context(username: str) -> UserPermissionContext`. It looks up the existing `permissions` cache backend via `_get_permission_cache()` (imported from `utils.permissions`) with key `f"user_ctx:{username}"`; on miss it calls `build_user_permission_context(username)` and stores the result.
  - In `workspace_cache.py` modify `_lookup_workspace_permission()` (lines 234-293) to obtain the context once at the top via `get_or_build_user_permission_context(username)`, and rewrite the four resolver closures so:
    - `_resolve_user_direct` reads `ctx.user_workspace_permissions.get(workspace)` instead of calling `store.get_workspace_permission`.
    - `_resolve_group_direct` reads `ctx.group_workspace_permissions.get(workspace)` instead of `store.get_user_groups_workspace_permission` (which today goes through `WorkspaceGroupPermissionRepository.get_highest_for_user` at `workspace_group_permission.py:183-225`).
    - `_resolve_user_regex` reads `ctx.workspace_regex_permissions` instead of `store.list_workspace_regex_permissions`.
    - `_resolve_group_regex` reads `ctx.group_workspace_regex_permissions` instead of calling `store.get_groups_ids_for_user` followed by `store.list_workspace_group_regex_permissions_for_groups_ids`.
  - The resolvers must still call `_match_workspace_regex_permission(...)` on the regex lists — only the **fetch** is moved into the context.
  - Preserve the existing per-source ordering / short-circuit / logging behavior — only the data source changes.
  - In `test_workspace_cache.py` add a test (or extend an existing one) that monkeypatches `get_or_build_user_permission_context` to return a pre-built context and asserts the resolvers don't call the underlying `store.*` methods at all.
- **Verify**: `pytest mlflow_oidc_auth/tests/utils/test_workspace_cache.py mlflow_oidc_auth/tests/utils/test_batch_permissions.py -v` — all pass.
- **Commit message**: `perf(workspace-cache): resolve via cached UserPermissionContext instead of per-source DB calls`

### Task 6: Update perf tests to assert NEW (lower) query counts; prove regression-fail safety

- **Files**: `mlflow_oidc_auth/tests/perf/test_resolver_query_counts.py`
- **What**:
  - Run the perf tests added in Task 3 against the post-task-5 code, capture the new counts, and rewrite each `assert counter.total == <old>` to `assert counter.total == <new>` with a comment showing both numbers and the delta (e.g. `# was 6, now 2 — see quick task 260512-o78`).
  - Add one negative test: `test_resolver_does_not_regress_when_context_missing` — calls the resolver chain twice in the same simulated request (same username, no intervening mutation) and asserts the second call's query count is 0 (cache-hit path). This guards against future refactors that accidentally bypass the context cache.
  - Add one cache-invalidation test in this file: mutate via `store.add_user_to_group("alice", "g4")` (which goes through `_PERMISSION_CUD_METHODS` → `flush_permission_cache`), then re-call the resolver and assert the query count is back to the cold-path number — proving invalidation reaches the new `user_ctx:` cache entries.
- **Verify**: `pytest mlflow_oidc_auth/tests/perf -v` — all pass. Manually revert any single line in Task 5's `_lookup_workspace_permission` rewiring to confirm a regression fails the suite (then re-apply).
- **Commit message**: `test(perf): assert post-optimization query counts and invalidation propagation`

### Task 7: Cache-invalidation correctness tests for membership and workspace-permission CUD

- **Files**: `mlflow_oidc_auth/tests/utils/test_batch_permissions.py` (new test class), `mlflow_oidc_auth/tests/perf/test_resolver_query_counts.py` (extend Task 6's invalidation test)
- **What**:
  - Add `TestUserContextCacheInvalidation` class in `test_batch_permissions.py` with cases that, for each of the five mutation surfaces listed in CONTEXT.md success criterion 3, verify the `user_ctx:<username>` entry is cleared:
    1. `add_user_to_group` flushes the entry (already in `_PERMISSION_CUD_METHODS` at sqlalchemy_store.py:1195).
    2. `remove_user_from_group` flushes (line 1196).
    3. `set_user_groups` flushes (line 1194).
    4. `create_workspace_permission` — verify whether this is or is not in `_PERMISSION_CUD_METHODS` (per the comment at sqlalchemy_store.py:1068-1069, workspace permissions are intentionally excluded today because they had their own cache). With the new context surface, that exclusion is now incorrect — **add the workspace permission CUD methods to `_PERMISSION_CUD_METHODS`** (`create_workspace_permission`, `update_workspace_permission`, `delete_workspace_permission`, `create_workspace_group_permission`, `update_workspace_group_permission`, `delete_workspace_group_permission`, plus the regex variants and `wipe_workspace_permissions`). This is the actual code change in this task — the tests prove it.
    5. `create_workspace_group_permission` flushes (after the addition above).
  - Each test: build context for "alice", verify cache hit on second call, perform the mutation, verify the next call rebuilds (counter > 0).
- **Verify**: `pytest mlflow_oidc_auth/tests/utils/test_batch_permissions.py::TestUserContextCacheInvalidation mlflow_oidc_auth/tests/perf -v` — all pass.
- **Commit message**: `fix(cache): include workspace-permission CUD methods in invalidation wrapper`

## Files modified (summary)

**Modified**

- `mlflow_oidc_auth/repository/_base.py` (Task 1)
- `mlflow_oidc_auth/repository/group.py` (Task 1)
- `mlflow_oidc_auth/utils/batch_permissions.py` (Tasks 4, 5)
- `mlflow_oidc_auth/utils/workspace_cache.py` (Task 5)
- `mlflow_oidc_auth/sqlalchemy_store.py` (Task 4 — may add small store methods; Task 7 — extends `_PERMISSION_CUD_METHODS`)
- `mlflow_oidc_auth/tests/utils/test_batch_permissions.py` (Tasks 4, 7)
- `mlflow_oidc_auth/tests/utils/test_workspace_cache.py` (Task 5)

**New**

- `mlflow_oidc_auth/tests/perf/__init__.py` (Task 2)
- `mlflow_oidc_auth/tests/perf/conftest.py` (Task 2)
- `mlflow_oidc_auth/tests/perf/test_resolver_query_counts.py` (Task 3, edited in Tasks 6 & 7)

## Risks and mitigations

- **Stale cached context after permission CUD.** Mitigation: Task 7 explicitly verifies invalidation for every mutation surface listed in the success criteria, and extends `_PERMISSION_CUD_METHODS` to cover workspace CUD which the existing comment at `sqlalchemy_store.py:1068-1069` intentionally excludes. Tests fail loud if the wrapper misses a surface.
- **Behavior drift from rewiring `workspace_cache` resolvers.** Mitigation: keep existing `_match_workspace_regex_permission`, source-order loop, and logging exactly as-is; only the **fetch** moves into the context. Existing `test_workspace_cache.py` cases run unmodified as a contract.
- **Perf-test flakiness from differing in-memory store seeds.** Mitigation: each perf test uses `seed_user_with_groups` with explicit counts; `query_counter` filters out engine `BEGIN`/`COMMIT` noise (event listener only counts `SELECT`/`INSERT`/`UPDATE`/`DELETE` statements via the SQL prefix). Tests assert exact equality, not ranges — flakiness manifests as a hard failure to investigate, not a silent drift.
- **`_list_user_groups` is shared by base group repo and scorer group repo.** Mitigation: the JOIN collapse is semantically identical (same input `username` → same set of group names). Re-running the existing scorer tests (`mlflow_oidc_auth/tests/test_sqlalchemy_store_scorer.py`) in Task 1's verify step covers this.
- **Tasks 4 may need new store methods.** Mitigation: search `sqlalchemy_store.py` first; only add methods that don't exist, and use JOIN-based queries so we don't reintroduce the very anti-pattern we're removing.
- **Memory ceiling:** `user_context` cache capped at 256 × ~100KB = ~25MB worst case per pod (vs 200MB if it shared the `permissions` cache). Acceptable for 7-replica deployment.
- **Cross-pod staleness:** local in-process cache; mutation on replica A only flushes replica A. TTL=30s bounds staleness window. Cross-pod consistency is Redis territory (out of scope per CONTEXT.md).

## Acceptance criteria

1. `repository/_base.py:255` becomes a single JOIN query — proved by Task 1's diff and by `pytest mlflow_oidc_auth/tests/repository -k "list_user_groups"` staying green.
2. Per-request resolver-chain query count for `build_user_permission_context` + a follow-up workspace resolution drops from the Task-3-baseline to ≤2 in the cache-hit path and ≤N+1 in the cold path — proved by `mlflow_oidc_auth/tests/perf/test_resolver_query_counts.py` (Task 6's exact-equality assertions).
3. Every mutation enumerated in CONTEXT.md success criterion 3 (`add_user_to_group`, `remove_user_from_group`, `set_groups_for_user`, workspace permission CUD, workspace group permission CUD) invalidates the new `user_ctx:` cache entry — proved by `TestUserContextCacheInvalidation` (Task 7).
4. `tox` (`pytest mlflow_oidc_auth/tests`) is green at every commit — verified at each task's verify step.
5. `build_user_permission_context()` signature is unchanged and public API (`resolve_*_permission_from_context`, `batch_resolve_*`, `filter_manageable_*`, `effective_*_permission`, `can_*`) is unchanged — proved by the existing unit tests in `test_batch_permissions.py` and `test_permissions.py` passing without modification beyond additive new tests.

## Out of scope (reminders)

- Redis backend rollout (separate user-led decision).
- Sticky sessions / LB changes.
- JWKS / auth-flow changes (covered separately by `plans/plan-authentication-perfs.md`).
- FK index changes (already in prod via the sibling indexes brief).
- Adding cached lookup for tables outside the 4 hot ones (`groups`, `user_groups`, `workspace_permissions`, `workspace_group_permissions`).
- `repository/group.py:50` admin `list_groups()` cleanup — the `.all()` there is genuinely full-table and admin-only; not worth a query rewrite (only called from admin UI; would need an actual filter to matter).
- Refactoring `utils/permissions.py` `PERMISSION_REGISTRY` to read from `UserPermissionContext` — its existing TTL cache and the new `_base.py` JOIN already address its share of the seq-scan load; a deeper unification can be a follow-up task.
