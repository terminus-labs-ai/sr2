# Plan: X-SR2-Agent-Name Header for Per-Request Memory Isolation

## Problem
`MemoryScopeConfig.agent_name` is set once at SR2 construction time (from YAML config). When multiple agents (Edi, Mordin, Tali) share the same bridge instance, they all get the same `agent_name` — so private memories (`scope_ref = "agent:<name>"`) aren't isolated per-agent.

## Architecture Insight
The `scope_config` object is shared by the retriever and extractor:
- **Retriever** (`retrieval.py:64`): `f"agent:{self._scope_config.agent_name}"` → used in `_build_scope_params()` for filtering
- **Extractor** (`extraction.py:150`): `f"agent:{self._scope_config.agent_name}"` → stamps `scope_ref` on new memories
- **Scope Detector** (`scope_detector.py:47`): `f"agent:{self._scope_config.agent_name}"` → deterministic private scope

These are constructed once in `SR2.__init__()` (sr2.py:109-125) and wired with scope_config at line 204-206.

## Design: Per-Request Override via `proxy_optimize()`

Rather than modifying the shared `_scope_config` object (thread-unsafe), we thread `agent_name` as an optional parameter through the proxy call chain. When present, it **temporarily overrides** `scope_config.agent_name` for the duration of that request.

### Changes (4 files, ~30 lines)

#### 1. `app.py` — Extract header, pass to engine
- In `proxy_messages()` (line 219) and `proxy_chat_completions()` (line 308): read `X-SR2-Agent-Name` header
- Pass `agent_name` to `engine.optimize()`

#### 2. `engine.py` — Thread agent_name to SR2
- `optimize()` and `_optimize_locked()`: accept optional `agent_name: str | None`
- Pass through to `self._sr2.proxy_optimize(agent_name=agent_name)`

#### 3. `sr2.py` — Apply override in proxy_optimize
- `proxy_optimize()`: accept optional `agent_name: str | None`
- If provided and `self._scope_config_resolved` exists, create a **shallow copy** of scope_config with the overridden agent_name
- Temporarily set `self._retriever._scope_config` and `self._extractor._scope_config` to the overridden copy for the duration of the call, restoring afterward
- This is safe because `proxy_optimize` runs under a per-session lock in the bridge

#### 4. `session_tracker.py` — No changes needed
- `X-SR2-Agent-Name` is request-scoped (not session-scoped), so it bypasses the session tracker entirely

### Flow
```
Request header: X-SR2-Agent-Name: edi
    → app.py: agent_name = headers.get("x-sr2-agent-name")
    → engine.optimize(agent_name="edi")
    → sr2.proxy_optimize(agent_name="edi")
    → scope_config_override = scope_config.model_copy(update={"agent_name": "edi"})
    → retriever uses scope_ref = "agent:edi"
    → extractor stamps scope_ref = "agent:edi" on new memories
```

### Backward Compatibility
- Header is optional — omitting it uses the YAML-configured `agent_name` (or None)
- No config schema changes
- No breaking API changes

### What This Does NOT Change
- Session ID logic (still uses X-SR2-Session-ID or config default)
- Scope detection for non-private scopes (project/global still works the same)
- The YAML `agent_name` config still works as the default fallback
