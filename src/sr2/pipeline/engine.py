import logging
from dataclasses import dataclass, field

from sr2.config.models import PipelineConfig, LayerConfig, ContentItemConfig
from sr2.degradation.circuit_breaker import CircuitBreaker
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext, ResolvedContent
from sr2.cache.registry import CachePolicyRegistry, PipelineState
from sr2.pipeline.result import PipelineResult, StageResult, StageTimer
from sr2.pipeline.serializer import ContextSerializer
from sr2.pipeline.prefix_tracker import PrefixTracker, PrefixSnapshot

logger = logging.getLogger(__name__)


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
    ):
        self._resolvers = resolver_registry
        self._cache_policies = cache_registry
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._previous_state: PipelineState | None = None
        self._layer_cache: dict[str, list[ResolvedContent]] = {}
        self._serializer = ContextSerializer()
        self._prefix_tracker = PrefixTracker()
        self._layer_string_cache: dict[str, str] = {}
        self._config_logged = False

    def _log_resolved_config(self, config: PipelineConfig) -> None:
        """Log the fully resolved config once at INFO level."""
        if self._config_logged:
            return
        self._config_logged = True
        logger.info(
            "Resolved pipeline config: budget=%d, layers=[%s], "
            "compaction=%s, summarization=%s (trigger=%s), kv_cache=%s",
            config.token_budget,
            ", ".join(l.name for l in config.layers),
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

            if not should_recompute and layer.name in self._layer_cache:
                # Cache hit — reuse cached objects and serialized string
                layers[layer.name] = self._layer_cache[layer.name]
                result.add_stage(
                    StageResult(
                        stage_name=layer.name,
                        status="success",
                        tokens_used=sum(c.tokens for c in layers[layer.name]),
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
                        if (
                            strategy == "append_only"
                            and layer.name in self._layer_string_cache
                            and serialized != self._layer_string_cache[layer.name]
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

        # Enforce token budget
        layers = self._enforce_budget(layers, config.token_budget)

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
                        logger.debug(
                            f"Optional item '{item.key}' in layer '{layer.name}' failed: {e}"
                        )
                        continue
                    had_required_failure = True
                    result.add_stage(
                        timer.result(
                            status="failed",
                            error=str(e),
                        )
                    )
                    break

        if had_required_failure:
            if resolved:
                tokens = sum(c.tokens for c in resolved)
                result.add_stage(timer.result(status="degraded", tokens=tokens, fallback=True))
                return resolved
            return None

        tokens = sum(c.tokens for c in resolved)
        result.add_stage(timer.result(status="success", tokens=tokens))
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

    def _enforce_budget(
        self,
        layers: dict[str, list[ResolvedContent]],
        budget: int,
    ) -> dict[str, list[ResolvedContent]]:
        """If total tokens exceed budget, trim content from last layers first."""
        total = sum(c.tokens for contents in layers.values() for c in contents)
        if total <= budget:
            return layers

        excess = total - budget
        layer_names = list(layers.keys())

        # Trim from last layer to first, but never trim the first layer
        for layer_name in reversed(layer_names[1:]):
            if excess <= 0:
                break
            contents = layers[layer_name]
            for i in range(len(contents) - 1, -1, -1):
                if excess <= 0:
                    break
                item = contents[i]
                if item.tokens <= excess:
                    excess -= item.tokens
                    contents.pop(i)
                else:
                    new_tokens = item.tokens - excess
                    ratio = new_tokens / item.tokens if item.tokens > 0 else 0
                    new_content = item.content[: int(len(item.content) * ratio)]
                    contents[i] = ResolvedContent(
                        key=item.key,
                        content=new_content,
                        tokens=new_tokens,
                        metadata=item.metadata,
                    )
                    excess = 0

        return layers
