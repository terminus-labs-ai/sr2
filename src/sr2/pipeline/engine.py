"""Pipeline engine — interprets YAML config and compiles context.

This is the internal orchestration layer. The facade (SR2 class) calls this.
The harness never touches this directly — hard public/internal boundary.

Execution flow:
1. Parse layer configs, identify content providers per layer
2. Resolve cross-layer references (summarization scope), determine resolution order
3. For each layer: resolve providers, apply reducers, enforce token budget
4. Enforce total pipeline budget (priority-based shedding if over)
5. Apply cache policies per layer
6. Emit CompiledContext

Design principles:
- SRP: Engine only orchestrates — providers/resolvers do the work.
- OCP: New provider types added via plugins, engine doesn't change.
- DRY: Common resolution/reduction logic extracted to helper methods.
"""

from __future__ import annotations

import asyncio
from typing import Any

from sr2.config.models import LayerConfig, PipelineConfig
from sr2.core.errors import PipelineError, ProviderError
from sr2.core.models import CachePolicy
from sr2.plugins.registry import PluginRegistry
from sr2.pipeline.result import (
    CompiledContext,
    LayerResult,
    PipelineMetrics,
)


class PipelineEngine:
    """Interprets pipeline config and compiles context from layers.

    The engine is the internal workhorse. It knows how to:
    - Resolve providers within each layer
    - Apply reducers to compress content
    - Enforce token budgets per layer and total
    - Handle cache policies (static, ephemeral, none)
    - Shed low-priority layers under budget pressure
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._provider_registry = PluginRegistry("sr2.providers")
        self._reducer_registry = PluginRegistry("sr2.reducers")
        self._cache: dict[str, str] = {}  # layer_name -> cached content

    async def compile(self) -> CompiledContext:
        """Execute the full pipeline and return compiled context.

        Returns:
            CompiledContext with all layers assembled and metrics.
        """
        metrics = PipelineMetrics(total_budget=self.config.total_budget)

        # Step 1: Determine resolution order (cross-layer refs first)
        resolution_order = self._resolve_order()

        # Step 2: Resolve each layer
        layer_results = []
        for layer_name in resolution_order:
            layer_config = self._find_layer(layer_name)
            if layer_config is None:
                raise PipelineError(f"Layer '{layer_name}' referenced but not defined")

            result = await self._resolve_layer(layer_config, metrics)
            layer_results.append(result)

        # Step 3: Sort by original config order
        name_to_result = {r.name: r for r in layer_results}
        ordered_results = [name_to_result[name] for name in [lc.name for lc in self.config.layers]]

        # Step 4: Enforce total budget
        ordered_results = self._enforce_total_budget(ordered_results, metrics)

        # Step 5: Build compiled context
        total_tokens = sum(r.tokens for r in ordered_results)
        metrics.total_tokens = total_tokens

        return CompiledContext(
            layers=ordered_results,
            metrics=metrics,
            total_tokens=total_tokens,
        )

    # --- Internal resolution ---

    def _resolve_order(self) -> list[str]:
        """Determine layer resolution order, respecting cross-layer references.

        Summarization layers that reference other layers must resolve after
        their dependencies. Simple topological sort.
        """
        # Build dependency graph
        deps: dict[str, set[str]] = {}
        for layer in self.config.layers:
            deps[layer.name] = set()
            if layer.summarization and layer.summarization.scope:
                deps[layer.name] = set(layer.summarization.scope)

        # Topological sort (layers with no deps first)
        resolved = []
        visiting = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise PipelineError(f"Circular layer dependency detected: {name}")
            if name in resolved:
                return
            visiting.add(name)
            for dep in deps.get(name, set()):
                visit(dep)
            visiting.discard(name)
            resolved.append(name)

        for layer in self.config.layers:
            visit(layer.name)

        return resolved

    async def _resolve_layer(self, layer: LayerConfig, metrics: PipelineMetrics) -> LayerResult:
        """Resolve all content providers within a single layer."""
        # Check cache
        cache_key = self._cache_key(layer)
        if layer.cache == CachePolicy.STATIC and cache_key in self._cache:
            metrics.cache_hits += 1
            return LayerResult(
                name=layer.name,
                content=self._cache[cache_key],
                tokens=0,  # TODO: cache token count
                cache_hit=True,
            )

        metrics.cache_misses += 1

        # Resolve each provider in the layer
        providers_content: list[str] = []
        provider_tokens: dict[str, int] = {}
        layer_tokens = 0

        # Session history provider
        if layer.session_history:
            content, tokens = await self._resolve_provider(layer.name, "session_history", layer.session_history, metrics)
            providers_content.append(content)
            provider_tokens["session_history"] = tokens
            layer_tokens += tokens

        # Memory provider (read)
        if layer.memory and layer.memory.read:
            if isinstance(layer.memory.read, dict):
                # Will be resolved by memory provider plugin
                pass  # TODO: wire up memory provider

        # Tools provider
        if layer.tools:
            content, tokens = await self._resolve_provider(layer.name, "tools", layer.tools, metrics)
            providers_content.append(content)
            provider_tokens["tools"] = tokens
            layer_tokens += tokens

        # Enforce layer budget
        content = "\n\n".join(providers_content)
        if layer.max_tokens and layer_tokens > layer.max_tokens:
            content = self._truncate_to_budget(content, layer.max_tokens)
            layer_tokens = layer.max_tokens

        # Store in cache if applicable
        if layer.cache == CachePolicy.STATIC:
            self._cache[cache_key] = content

        return LayerResult(
            name=layer.name,
            content=content,
            tokens=layer_tokens,
            cache_hit=False,
            providers=provider_tokens,
        )

    async def _resolve_provider(
        self, layer_name: str, provider_name: str, provider_config: Any, metrics: PipelineMetrics
    ) -> tuple[str, int]:
        """Resolve a single content provider, with circuit breaker and fallback."""
        try:
            # Load provider from registry
            provider = self._provider_registry.get(provider_name)
            # TODO: Pass proper ProviderContext
            result = await provider.resolve({})
            return result.content, result.tokens
        except Exception as e:
            metrics.providers_failed.append(f"{layer_name}.{provider_name}")
            # TODO: Circuit breaker logic and fallback
            return "", 0

    def _enforce_total_budget(
        self, layer_results: list[LayerResult], metrics: PipelineMetrics
    ) -> list[LayerResult]:
        """Shed lowest-priority layers if total budget exceeded."""
        if self.config.total_budget is None:
            return layer_results

        total = sum(r.tokens for r in layer_results)
        if total <= self.config.total_budget:
            return layer_results

        # Sort by priority ascending (lowest priority first for shedding)
        layer_priority = {
            lc.name: lc.priority for lc in self.config.layers
        }

        sorted_results = sorted(layer_results, key=lambda r: layer_priority.get(r.name, 50))

        # Shed layers until under budget
        kept = []
        remaining_budget = self.config.total_budget

        for result in sorted_results:
            if result.tokens <= remaining_budget:
                kept.append(result)
                remaining_budget -= result.tokens
            else:
                # Try partial: truncate this layer
                truncated_content = self._truncate_to_budget(result.content, remaining_budget)
                kept.append(LayerResult(
                    name=result.name,
                    content=truncated_content,
                    tokens=remaining_budget,
                    cache_hit=result.cache_hit,
                    providers=result.providers,
                ))
                remaining_budget = 0

        # Log shed layers
        kept_names = {r.name for r in kept}
        for result in layer_results:
            if result.name not in kept_names:
                metrics.layers_shed.append(result.name)

        # Restore original order
        name_to_result = {r.name: r for r in kept}
        original_order = [lc.name for lc in self.config.layers]
        return [name_to_result[name] for name in original_order if name in name_to_result]

    # --- Helpers ---

    def _find_layer(self, name: str) -> LayerConfig | None:
        for layer in self.config.layers:
            if layer.name == name:
                return layer
        return None

    def _cache_key(self, layer: LayerConfig) -> str:
        return f"static:{layer.name}"

    @staticmethod
    def _truncate_to_budget(content: str, budget: int) -> str:
        """Truncate content to fit token budget.

        Rough estimate: 1 token ≈ 4 chars for English text.
        TODO: Use tiktoken for accurate counting.
        """
        max_chars = budget * 4
        if len(content) <= max_chars:
            return content
        return content[:max_chars] + "\n... [truncated due to token budget]"
