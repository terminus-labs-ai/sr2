"""SR2 Bridge CLI entry point.

Usage:
    sr2-bridge                          # zero-config, sensible defaults
    sr2-bridge bridge.yaml              # custom config
    sr2-bridge bridge.yaml --port 9200  # CLI override
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SR2 Bridge — context optimization proxy for LLM API calls",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to bridge YAML config file (optional, uses defaults if omitted)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on (default: 9200)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--upstream",
        default=None,
        help="Upstream API URL (default: https://api.anthropic.com)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--request-log",
        default=None,
        help="Enable JSONL request/response logging to this file",
    )
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> dict:
    """Load and merge config from YAML file + CLI overrides."""
    raw: dict = {}

    if args.config:
        config_path = os.path.abspath(args.config)
        if not os.path.exists(config_path):
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        # Expand ${VAR} and ${VAR:-default} patterns in all string values
        from sr2.config.loader import expand_env_vars

        raw = expand_env_vars(raw)

    # Ensure bridge section exists
    bridge = raw.setdefault("bridge", {})

    # CLI overrides
    if args.port is not None:
        bridge["port"] = args.port
    if args.host is not None:
        bridge["host"] = args.host
    if args.upstream is not None:
        bridge.setdefault("forwarding", {})["upstream_url"] = args.upstream
    if args.request_log is not None:
        log_cfg = bridge.setdefault("logging", {})
        log_cfg["enabled"] = True
        log_cfg["output_path"] = args.request_log

    return raw


def build_components(raw_config: dict):
    """Build bridge components from raw config dict."""
    from sr2.config.models import PipelineConfig
    from sr2.sr2 import SR2, SR2Config

    from sr2_bridge.config import BridgeConfig
    from sr2_bridge.engine import BridgeEngine
    from sr2_bridge.forwarder import BridgeForwarder
    from sr2_bridge.llm import (
        APIKeyCache,
        make_embedding_callable,
        make_extraction_callable,
        make_summarization_callable,
    )
    from sr2_bridge.request_logger import BridgeRequestLogger
    from sr2_bridge.session_tracker import SessionTracker

    # Bridge config
    bridge_config = BridgeConfig(**(raw_config.get("bridge", {})))

    # Pipeline config (for compaction/summarization settings)
    pipeline_raw = raw_config.get("pipeline", {})
    pipeline_config = PipelineConfig(**pipeline_raw)

    # Shared API key cache — updated from proxied request headers
    key_cache = APIKeyCache()

    # Build LLM callables for SR2's internal use
    upstream_url = bridge_config.forwarding.upstream_url
    fast_complete = None
    embed = None

    if bridge_config.llm.summarization:
        fast_complete = make_summarization_callable(
            bridge_config.llm.summarization, key_cache, upstream_url
        )
    elif bridge_config.llm.extraction:
        # Fall back to extraction model for summarization
        fast_complete = make_summarization_callable(
            bridge_config.llm.extraction, key_cache, upstream_url
        )

    if bridge_config.llm.embedding:
        embed = make_embedding_callable(bridge_config.llm.embedding, key_cache, upstream_url)

    # Build memory store based on backend config.
    # SQLite can be created synchronously; PostgreSQL requires an async pool
    # that gets created during app lifespan (see app.py).
    memory_store = None
    mem_cfg = bridge_config.memory
    if mem_cfg.enabled and mem_cfg.backend == "sqlite":
        from sr2.memory.store import SQLiteMemoryStore

        memory_store = SQLiteMemoryStore(db_path=mem_cfg.db_path)
        logger.info("Bridge memory: using SQLite store (db=%s)", mem_cfg.db_path)

    # Pipeline inspector — ring-buffer trace collector for observability
    from sr2.pipeline.trace import TraceCollector

    trace_collector = TraceCollector(max_turns=100)

    def _log_trace(turn_trace):
        warnings = turn_trace.warnings
        if warnings:
            for w in warnings:
                logger.warning("inspect [%s] turn %d: %s", turn_trace.session_id, turn_trace.turn_number, w)
        stages = [e.stage for e in turn_trace.events]
        logger.info(
            "inspect [%s] turn %d: %s (%.0fms)",
            turn_trace.session_id,
            turn_trace.turn_number,
            " → ".join(stages),
            turn_trace.total_duration_ms,
        )

    trace_collector.on_turn_complete(_log_trace)

    # Build SR2 instance with preloaded PipelineConfig (no agent.yaml needed)
    sr2_config = SR2Config(
        config_dir="",  # Not used when preloaded_config is set
        agent_yaml={},
        preloaded_config=pipeline_config,
        fast_complete=fast_complete,
        embed=embed,
        memory_store=memory_store,
        trace_collector=trace_collector,
    )
    sr2 = SR2(sr2_config)

    # Build engine and wire SR2 into it
    engine = BridgeEngine(pipeline_config, bridge_config=bridge_config, sr2=sr2, key_cache=key_cache)
    forwarder = BridgeForwarder(bridge_config.forwarding)
    session_tracker = SessionTracker(bridge_config.session)

    # Request logger (optional — off by default)
    request_logger = None
    if bridge_config.logging.enabled:
        request_logger = BridgeRequestLogger(bridge_config.logging)

    return bridge_config, engine, forwarder, session_tracker, key_cache, request_logger


def main():
    args = parse_args()

    # Suppress LiteLLM's "Give Feedback / Get Help" banner and info messages.
    # These are raw print() calls that bypass Python logging.
    import litellm

    litellm.suppress_debug_info = True
    litellm.set_verbose = False

    # Blocking mode for piped stderr
    if not os.isatty(sys.stderr.fileno()):
        os.set_blocking(sys.stderr.fileno(), True)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        force=True,
    )

    raw_config = load_config(args)
    bridge_config, engine, forwarder, session_tracker, key_cache, request_logger = build_components(
        raw_config
    )

    from sr2_bridge.app import create_bridge_app

    app = create_bridge_app(
        bridge_config, engine, forwarder, session_tracker, key_cache, request_logger
    )

    import uvicorn

    uvicorn.run(
        app,
        host=bridge_config.host,
        port=bridge_config.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
