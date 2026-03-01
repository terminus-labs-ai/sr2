from sr2.cache.registry import CachePolicyRegistry


class ImmutablePolicy:
    """Never recompute mid-session."""
    def should_recompute(self, layer_name, current, previous) -> bool:
        return previous is None


class RefreshOnTopicShiftPolicy:
    """Recompute when intent changes."""
    def should_recompute(self, layer_name, current, previous) -> bool:
        if previous is None:
            return True
        return current.current_intent != previous.current_intent


class RefreshOnStateChangePolicy:
    """Recompute when state hash changes."""
    def should_recompute(self, layer_name, current, previous) -> bool:
        if previous is None:
            return True
        return current.state_hash != previous.state_hash


class AppendOnlyPolicy:
    """Always recompute (content is append-only)."""
    def should_recompute(self, *args) -> bool:
        return True


class AlwaysNewPolicy:
    """Always recompute."""
    def should_recompute(self, *args) -> bool:
        return True


class PerInvocationPolicy:
    """Always recompute (unique per call)."""
    def should_recompute(self, *args) -> bool:
        return True


class TemplateReusePolicy:
    """Recompute only on first call."""
    def should_recompute(self, layer_name, current, previous) -> bool:
        return previous is None


def create_default_cache_registry() -> CachePolicyRegistry:
    """Create a registry pre-loaded with all built-in policies."""
    reg = CachePolicyRegistry()
    reg.register("immutable", ImmutablePolicy())
    reg.register("refresh_on_topic_shift", RefreshOnTopicShiftPolicy())
    reg.register("refresh_on_state_change", RefreshOnStateChangePolicy())
    reg.register("append_only", AppendOnlyPolicy())
    reg.register("always_new", AlwaysNewPolicy())
    reg.register("per_invocation", PerInvocationPolicy())
    reg.register("template_reuse", TemplateReusePolicy())
    return reg
