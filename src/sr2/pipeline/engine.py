import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from sr2.config.models import PipelineConfig, LayerConfig, ContentItemConfig
from sr2.degradation.circuit_breaker import CircuitBreaker
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext, ResolvedContent
from sr2.cache.registry import CachePolicyRegistry, PipelineState
from sr2.pipeline.result import PipelineResult, StageResult, StageTimer
from sr2.pipeline.serializer import ContextSerializer
from sr2.pipeline.prefix_tracker import PrefixTracker, PrefixSnapshot

logger = logging.getLogger(__name__)

# Callback signature for budget overflow handling.
# Receives: (layers, budget, config, resolver_context)
# Returns updated layers dict, or None if no reduction was possible.
BudgetOverflowHandler = Callable[
    [dict[str, list[ResolvedContent]], int, PipelineConfig, ResolverContext],
    Awaitable[dict[str, list[ResolvedContent]] | None],
]


@dataclass
class CompiledContext:
    """The final output: what goes to the LLM."""

    content: str
    tokens: int
    layers: dict[str, list[ResolvedContent]] = field(default_factory=dict)
    pipeline_result: PipelineResult = field(default_factory=PipelineResult)
    prefix_snapshot: PrefixSnapshot | None = None


class PipelineEngine:
    """Core engine: config -> resolve layers -> compile context.

    Integrates circuit breaker for per-layer degradation:
    - If a layer fails, the circuit breaker records the failure
    - After N consecutive failures, the breaker opens and the layer is skipped
    - After cooldown, the breaker closes and the layer is retried
    - The first layer (core/system prompt) is never skipped
    """

    def __init__(
        self,
        resolver_registry: ContentResolverRegistry,
        cache_registry: CachePolicyRegistry,
        circuit_breaker: CircuitBreaker | None = None,
        budget_overflow_handler: BudgetOverflowHandler | None = None,
        trace_collector=None,
    ):
        self._resolvers = resolver_registry
        self._cache_policies = cache_registry
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._budget_overflow_handler = budget_overflow_handler
        self._trace = trace_collector
        self._previous_state: PipelineState | None = None
        self._layer_cache: dict[str, list[ResolvedContent]] = {}
        self._serializer = ContextSerializer()
        self._prefix_tracker = PrefixTracker()
        self._layer_string_cache: dict[str, str] = {}
        self._config_logged = False
        self.truncation_events: int = 0

    def session_prefix_tokens(self, session_layer_name: str) -> int:
        """Count how many session-layer tokens are within the cached prefix.

        The prefix is built from consecutive cached layers starting from the
        first layer.  The session layer is typically last and changes every turn,
        so its prefix contribution is usually 0.
        """
        if session_layer_name not in self._layer_string_cache:
            return 0
        # Session layer is cached — but only counts if ALL prior layers are
        # also cached (prefix is contiguous from the start).
        for layer_name in self._layer_cache:
            if layer_name == session_layer_name:
                return sum(c.tokens for c in self._layer_cache[layer_name])
            if layer_name not in self._layer_string_cache:
                return 0
        return 0

    def _log_resolved_config(self, config: PipelineConfig) -> None:
        """Log the fully resolved config once at INFO level."""
        if self._config_logged:
            return
        self._config_logged = True
        logger.info(
            "Resolved pipeline config: budget=%d, layers=[%s], "
            "compaction=%s, summarization=%s (trigger=%s), kv_cache=%s",
            config.token_budget,
            ", ".join(layer.name for layer in config.layers),
            config.compaction.enabled,
            config.summarization.enabled,
            config.summarization.trigger,
            config.kv_cache.strategy,
        )

    async def compile(
        self,
        config: PipelineConfig,
        context: ResolverContext,
        state: PipelineState | None = None,
    ) -> CompiledContext:
        """Compile context from config."""
        _compile_t0 = time.perf_counter()
        self._log_resolved_config(config)
        result = PipelineResult(config_used=config.extends or "")
        layers: dict[str, list[ResolvedContent]] = {}
        current_state = state or PipelineState()
        strategy = config.kv_cache.strategy

        for i, layer in enumerate(config.layers):
            is_core = i == 0  # First layer is always core — never skip

            # Circuit breaker: skip layer if breaker is open (except core)
            if not is_core and self._circuit_breaker.is_open(layer.name):
                logger.info(f"Circuit breaker open for layer '{layer.name}', skipping")
                result.add_stage(
                    StageResult(
                        stage_name=layer.name,
                        status="degraded",
                        fallback_used=True,
                        cache_status="skipped",
                        error="circuit_breaker_open",
                    )
                )
                continue

            # Check cache policy
            should_recompute = True
            if self._cache_policies.has(layer.cache_policy):
                policy = self._cache_policies.get(layer.cache_policy)
                should_recompute = policy.should_recompute(
                    layer.name, current_state, self._previous_state
                )
            else:
                logger.warning(
                    f"Unknown cache policy '{layer.cache_policy}' for layer "
                    f"'{layer.name}', defaulting to recompute"
                )

            if not should_recompute and layer.name in self._layer_cache:
                # Cache hit — reuse cached objects and serialized string
                layers[layer.name] = self._layer_cache[layer.name]
                result.add_stage(
                    StageResult(
                        stage_name=layer.name,
                        status="success",
                        tokens_used=sum(c.tokens for c in layers[layer.name]),
                        cache_status="hit",
                    )
                )
                self._circuit_breaker.record_success(layer.name)
            else:
                # Recompute path
                resolved = await self._resolve_layer(layer, config, context, result)
                if resolved is not None:
                    # Serialize the resolved content
                    serialized = self._serializer.serialize_layer(resolved)

                    # Strategy: maximize_prefix_reuse — if content is byte-identical
                    # to cached version, reuse the cached string to avoid spurious
                    # invalidation from resolvers producing semantically-identical
                    # but bytewise-different objects
                    if (
                        strategy == "maximize_prefix_reuse"
                        and layer.name in self._layer_string_cache
                        and serialized == self._layer_string_cache[layer.name]
                    ):
                        # Content unchanged despite recompute — reuse cached objects
                        layers[layer.name] = self._layer_cache[layer.name]
                        logger.debug(
                            f"Strategy maximize_prefix_reuse: layer '{layer.name}' "
                            f"content unchanged, reusing cached version"
                        )
                    else:
                        # Strategy: append_only — warn if layer content changed
                        # Skip warning for memory layer when memory_refresh allows
                        # dynamic changes (on_topic_shift, every_n_turns)
                        memory_refresh = config.kv_cache.memory_refresh
                        expects_dynamic_content = layer.name == "memory" and memory_refresh not in (
                            "session_start_only",
                            "disabled",
                        )
                        if (
                            strategy == "append_only"
                            and layer.name in self._layer_string_cache
                            and serialized != self._layer_string_cache[layer.name]
                            and not expects_dynamic_content
                        ):
                            logger.warning(
                                f"Strategy append_only: layer '{layer.name}' "
                                f"content changed unexpectedly, prefix may be invalidated"
                            )

                        layers[layer.name] = resolved
                        self._layer_cache[layer.name] = resolved

                    self._layer_string_cache[layer.name] = serialized
                    self._circuit_breaker.record_success(layer.name)
                else:
                    # Full failure
                    self._circuit_breaker.record_failure(layer.name)
                    if not is_core:
                        logger.warning(f"Layer '{layer.name}' failed, continuing without it")

        self._previous_state = current_state

        # Enforce token budget (with smart reduction before truncation)
        layers = await self._enforce_budget(layers, config, context)

        # Build serialized layer strings for final content assembly
        serialized_layers: dict[str, str] = {}
        layer_hashes: dict[str, str] = {}
        total_tokens = 0

        for layer_name, contents in layers.items():
            # Use cached serialized string if available, otherwise serialize fresh
            if layer_name in self._layer_string_cache:
                layer_str = self._layer_string_cache[layer_name]
            else:
                layer_str = self._serializer.serialize_layer(contents)
            serialized_layers[layer_name] = layer_str
            layer_hashes[layer_name] = self._serializer.hash_content(layer_str)
            total_tokens += sum(c.tokens for c in contents)

        # Assemble final content from serialized layers
        content = self._serializer.serialize_context(serialized_layers)

        # Create prefix snapshot — estimate prefix tokens from immutable layers
        prefix_tokens = 0
        for layer_name, contents in layers.items():
            # Count tokens from layers with immutable cache policy (stable prefix)
            if layer_name in self._layer_string_cache:
                prefix_tokens += sum(c.tokens for c in contents)
            else:
                break  # Stop at first non-cached layer

        full_hash = self._serializer.hash_content(content)
        prefix_snapshot = self._prefix_tracker.snapshot(
            layer_hashes=layer_hashes,
            full_hash=full_hash,
            prefix_tokens=prefix_tokens,
        )

        # Trace: resolve stage with layer details
        _compile_ms = (time.perf_counter() - _compile_t0) * 1000
        if self._trace:
            utilization = total_tokens / config.token_budget if config.token_budget > 0 else 0.0
            cb_status = self._circuit_breaker.status()
            # Build per-layer cache_status from StageResults
            _stage_cache = {s.stage_name: s.cache_status for s in result.stages}
            self._trace.emit("resolve", {
                "total_tokens": total_tokens,
                "budget": config.token_budget,
                "utilization": utilization,
                "cache_efficiency": None,  # populated by runtime via CacheReport
                "layers": [
                    {
                        "name": name,
                        "tokens": sum(c.tokens for c in contents),
                        "items": len(contents),
                        "cache_status": _stage_cache.get(name, ""),
                        "circuit_breaker": "open" if name in cb_status and cb_status[name].get("is_open") else "closed",
                    }
                    for name, contents in layers.items()
                ],
            }, duration_ms=_compile_ms)

        return CompiledContext(
            content=content,
            tokens=total_tokens,
            layers=layers,
            pipeline_result=result,
            prefix_snapshot=prefix_snapshot,
        )

    async def _resolve_layer(
        self,
        layer: LayerConfig,
        config: PipelineConfig,
        ctx: ResolverContext,
        result: PipelineResult,
    ) -> list[ResolvedContent] | None:
        """Resolve all content items in a layer.

        Returns list of resolved content, or None if the layer fully failed.
        """
        resolved: list[ResolvedContent] = []
        had_required_failure = False

        with StageTimer(layer.name) as timer:
            for item in layer.contents:
                try:
                    content = await self._resolve_item(item, ctx)
                    resolved.append(content)
                except Exception as e:
                    if item.optional:
                        logger.error(
                            "Optional item %s in layer %s failed",
                            item.key,
                            layer.name,
                            exc_info=True,
                        )
                        continue
                    had_required_failure = True
                    result.add_stage(
                        timer.result(
                            status="failed",
                            cache_status="miss",
                            error=str(e),
                        )
                    )
                    break

        if had_required_failure:
            if resolved:
                tokens = sum(c.tokens for c in resolved)
                result.add_stage(
                    timer.result(
                        status="degraded", tokens_used=tokens, fallback_used=True, cache_status="miss"
                    )
                )
                logger.info(f"Layer {layer.name} degraded to {tokens} tokens.")
                return resolved
            return None

        tokens = sum(c.tokens for c in resolved)
        result.add_stage(timer.result(status="success", tokens_used=tokens, cache_status="miss"))
        logger.info(f"Layer {layer.name} successfully resolved to {tokens} tokens.")
        return resolved

    async def _resolve_item(
        self,
        item: ContentItemConfig,
        ctx: ResolverContext,
    ) -> ResolvedContent:
        """Resolve a single content item via registry."""
        resolver = self._resolvers.get(item.source)
        return await resolver.resolve(
            key=item.key,
            config=item.model_dump(),
            context=ctx,
        )

    async def _enforce_budget(
        self,
        layers: dict[str, list[ResolvedContent]],
        config: PipelineConfig,
        ctx: ResolverContext,
    ) -> dict[str, list[ResolvedContent]]:
        """Enforce token budget with smart reduction before truncation.

        Strategy (ordered by preference):
        1. Call budget_overflow_handler (compaction + summarization)
        2. Truncate trailing layers as last resort
        """
        budget = config.token_budget
        total = sum(c.tokens for contents in layers.values() for c in contents)
        if total <= budget:
            return layers

        logger.warning(
            "Token budget exceeded (%d/%d tokens, %d over), "
            "attempting smart reduction before truncation",
            total,
            budget,
            total - budget,
        )

        # Phase 1: Try smart reduction via handler (compaction + summarization)
        if self._budget_overflow_handler:
            try:
                reduced = await self._budget_overflow_handler(layers, budget, config, ctx)
                if reduced is not None:
                    layers = reduced
                    total = sum(c.tokens for contents in layers.values() for c in contents)
                    if total <= budget:
                        logger.info(
                            "Budget enforcement: handler reduced context to %d/%d tokens",
                            total,
                            budget,
                        )
                        return layers
                    logger.warning(
                        "Budget enforcement: handler reduced to %d tokens but "
                        "still %d over budget, falling back to truncation",
                        total,
                        total - budget,
                    )
            except Exception:
                logger.error(
                    "Budget overflow handler failed, falling back to truncation",
                    exc_info=True,
                )

        # Phase 2: Priority-aware truncation
        self.truncation_events += 1
        logger.warning(
            "Truncating %d excess tokens using priority-aware strategy (truncation_event #%d)",
            total - budget,
            self.truncation_events,
        )

        excess = total - budget

        # Build priority map from config layers
        layer_priority: dict[str, int] = {}
        layer_preserve: dict[str, bool] = {}
        layer_min_tokens: dict[str, int] = {}
        for layer_cfg in config.layers:
            layer_priority[layer_cfg.name] = layer_cfg.priority
            layer_preserve[layer_cfg.name] = layer_cfg.preserve
            layer_min_tokens[layer_cfg.name] = layer_cfg.min_tokens

        # First layer is always preserved (backwards compat)
        first_layer = next(iter(layers), None)
        if first_layer:
            layer_preserve[first_layer] = True

        # Sort layers by priority ascending (lowest priority truncated first)
        truncatable = [name for name in layers if not layer_preserve.get(name, False)]
        truncatable.sort(key=lambda n: layer_priority.get(n, 0))

        for layer_name in truncatable:
            if excess <= 0:
                break
            contents = layers[layer_name]
            min_tok = layer_min_tokens.get(layer_name, 0)
            current_layer_tokens = sum(c.tokens for c in contents)

            # Don't truncate below min_tokens
            available = max(0, current_layer_tokens - min_tok)
            if available <= 0:
                continue

            for i in range(len(contents) - 1, -1, -1):
                if excess <= 0 or available <= 0:
                    break
                item = contents[i]
                removable = min(item.tokens, available)
                if removable <= excess:
                    excess -= removable
                    available -= removable
                    if removable == item.tokens:
                        contents.pop(i)
                    else:
                        # Partial truncation to respect min_tokens
                        new_tokens = item.tokens - removable
                        char_limit = max(0, new_tokens * 4)
                        contents[i] = ResolvedContent(
                            key=item.key,
                            content=item.content[:char_limit],
                            tokens=new_tokens,
                            metadata=item.metadata,
                        )
                else:
                    new_tokens = item.tokens - excess
                    char_limit = max(0, new_tokens * 4)
                    contents[i] = ResolvedContent(
                        key=item.key,
                        content=item.content[:char_limit],
                        tokens=new_tokens,
                        metadata=item.metadata,
                    )
                    available -= excess
                    excess = 0

        if excess > 0:
            logger.warning(
                "Budget overflow: %d tokens still over budget after truncating all "
                "non-preserved layers. Preserved layers are untouched.",
                excess,
            )

        return layers
