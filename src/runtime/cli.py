"""Generic agent runner. No agent-specific code needed.

Usage:
    sr2-agent <config_dir>
    sr2-agent config/agents/edi
    sr2-agent config/agents/edi --http --port 8008
"""

import argparse
import asyncio
import logging
import os
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()

from runtime.agent import Agent, AgentConfig

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="SR2 Agent Runner")
    parser.add_argument(
        "config_dir",
        help="Path to agent config directory (must contain agent.yaml)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Agent name override (default: read from agent.yaml or directory name)",
    )
    parser.add_argument(
        "--defaults",
        default="configs/defaults.yaml",
        help="Path to library defaults (default: configs/defaults.yaml)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Also start HTTP API (FastAPI with /chat, /health, /metrics)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8008,
        help="HTTP port (default: 8008)",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def resolve_name(config_dir: str, override: str | None = None) -> str:
    """Get agent name from override, agent.yaml, or directory name."""
    if override:
        return override
    yaml_path = os.path.join(config_dir, "agent.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        if "agent_name" in data:
            return data["agent_name"]
    # Fall back to directory name
    return os.path.basename(os.path.normpath(config_dir))


async def run_agent(args):
    """Start and run the agent."""
    name = resolve_name(args.config_dir, args.name)

    agent = Agent(
        AgentConfig(
            name=name,
            config_dir=args.config_dir,
            defaults_path=args.defaults,
        )
    )

    if args.http:
        from runtime.http import create_http_app
        import uvicorn

        app = create_http_app(agent)
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=args.port,
            log_level=args.log_level.lower(),
        )
        server = uvicorn.Server(config)

        # Start agent (heartbeats, MCP connections)
        await agent.start()

        # Run HTTP server (blocks until shutdown)
        await server.serve()
    else:
        # Headless mode with heartbeats
        await agent.start()
        try:
            await asyncio.Event().wait()
        except (KeyboardInterrupt, SystemExit):
            pass

    await agent.shutdown()


def main():
    args = parse_args()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # file_h = logging.FileHandler("run.log")
    # file_h.setFormatter(fmt)
    # file_h.setLevel(logging.DEBUG)

    # console_h = logging.StreamHandler(sys.stdout)
    # console_h.setFormatter(fmt)
    # console_h.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        # handlers=[file_h, console_h],
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        force=True,
    )

    asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()
