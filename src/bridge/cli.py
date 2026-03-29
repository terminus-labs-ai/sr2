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
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
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

    # Ensure bridge section exists
    bridge = raw.setdefault("bridge", {})

    # CLI overrides
    if args.port is not None:
        bridge["port"] = args.port
    if args.host is not None:
        bridge["host"] = args.host
    if args.upstream is not None:
        bridge.setdefault("forwarding", {})["upstream_url"] = args.upstream

    return raw


def build_components(raw_config: dict):
    """Build bridge components from raw config dict."""
    from sr2.config.models import PipelineConfig

    from bridge.config import BridgeConfig
    from bridge.engine import BridgeEngine
    from bridge.forwarder import BridgeForwarder
    from bridge.llm import APIKeyCache
    from bridge.session_tracker import SessionTracker

    # Bridge config
    bridge_config = BridgeConfig(**(raw_config.get("bridge", {})))

    # Pipeline config (for compaction/summarization settings)
    pipeline_raw = raw_config.get("pipeline", {})
    pipeline_config = PipelineConfig(**pipeline_raw)

    # Shared API key cache — updated from proxied request headers
    key_cache = APIKeyCache()

    # Build components (engine creates LLM callables from bridge_config.llm)
    engine = BridgeEngine(pipeline_config, bridge_config=bridge_config, key_cache=key_cache)
    forwarder = BridgeForwarder(bridge_config.forwarding)
    session_tracker = SessionTracker(bridge_config.session)

    return bridge_config, engine, forwarder, session_tracker, key_cache


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
    bridge_config, engine, forwarder, session_tracker, key_cache = build_components(raw_config)

    from bridge.app import create_bridge_app

    app = create_bridge_app(bridge_config, engine, forwarder, session_tracker, key_cache)

    import uvicorn

    uvicorn.run(
        app,
        host=bridge_config.host,
        port=bridge_config.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
