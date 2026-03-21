import logging

from sr2.config.models import PipelineConfig

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when config validation fails."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Config validation failed: {'; '.join(errors)}")


def validate_config(config: PipelineConfig) -> list[str]:
    """Validate cross-field constraints. Returns list of warning strings.
    Raises ConfigValidationError for hard errors."""
    errors: list[str] = []
    warnings: list[str] = []

    # Hard error: no layers defined
    if not config.layers:
        errors.append("No layers defined")

    # Hard error: sum of max_tokens exceeds token_budget
    total_max_tokens = sum(
        item.max_tokens
        for layer in config.layers
        for item in layer.contents
        if item.max_tokens is not None
    )
    if total_max_tokens > config.token_budget:
        errors.append(
            f"Sum of content max_tokens ({total_max_tokens}) exceeds token_budget ({config.token_budget})"
        )

    # Hard error: cache-killing layout — always_new before append_only
    for i, layer in enumerate(config.layers):
        if layer.cache_policy == "always_new":
            for later_layer in config.layers[i + 1 :]:
                if later_layer.cache_policy == "append_only":
                    errors.append(
                        f"Layer '{layer.name}' with cache_policy 'always_new' appears before "
                        f"layer '{later_layer.name}' with cache_policy 'append_only' (cache-killing layout)"
                    )
                    break

    # Warning: compaction enabled but no rules
    if config.compaction.enabled and not config.compaction.rules:
        warnings.append("Compaction is enabled but no compaction rules are defined")

    # Warning: summarization enabled but compaction disabled
    if config.summarization.enabled and not config.compaction.enabled:
        warnings.append(
            "Summarization is enabled but compaction is disabled (should compact first)"
        )

    # Warning: retrieval enabled but no retrieval source
    if config.retrieval.enabled:
        has_retrieval_source = any(
            item.source == "retrieval" for layer in config.layers for item in layer.contents
        )
        if not has_retrieval_source:
            warnings.append("Retrieval is enabled but no layer has source 'retrieval'")

    # Warning: content item without max_tokens
    for layer in config.layers:
        for item in layer.contents:
            if item.max_tokens is None:
                warnings.append(
                    f"Content item '{item.key}' in layer '{layer.name}' has no max_tokens limit"
                )

    if errors:
        raise ConfigValidationError(errors)

    for w in warnings:
        logger.warning("Config validation: %s", w)

    return warnings
