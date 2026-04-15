"""Generic agent runner. No agent-specific code needed.

Usage:
    sr2-agent <config_dir>
    sr2-agent config/agents/edi --http --port 8008
    sr2-agent config/agents/tali --single-shot task_runner "implement auth"
    echo "long prompt" | sr2-agent config/agents/tali --single-shot task_runner
"""

import argparse
import asyncio
import logging
import os
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()

from sr2_runtime.agent import Agent, AgentConfig  # noqa: E402

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
        "--single-shot",
        metavar="INTERFACE",
        default=None,
        help="Run a single message through the named interface and exit. "
        "Provide message as positional arg or pipe via stdin.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Message to process in single-shot mode (reads stdin if omitted)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--inspect",
        nargs="?",
        const="default",
        choices=["default", "full", "brief"],
        help="Enable pipeline inspector (default, full, or brief mode)",
    )
    parser.add_argument(
        "--inspect-ui",
        action="store_true",
        help="Enable web inspector UI (default port: 9201)",
    )
    parser.add_argument(
        "--inspect-port",
        type=int,
        default=9201,
        help="Port for inspector web UI (default: 9201)",
    )
    parser.add_argument(
        "--agent-log-path",
        default=None,
        help="Path to agent log file (default: config_dir/agent.log)",
    )
    parser.add_argument(
        "--infra-log-path",
        default=None,
        help="Path to infrastructure log file (default: config_dir/infra.log)",
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

    # Allow SR2_INSPECT env var to enable inspector without CLI flag
    # (useful in Docker where you don't want to modify the CMD)
    if args.inspect is None and not args.inspect_ui:
        env_inspect = os.environ.get("SR2_INSPECT")
        if env_inspect:
            args.inspect = env_inspect if env_inspect in ("default", "full", "brief") else "default"

    # Create trace collector if inspector is enabled
    trace_collector = None
    if args.inspect or args.inspect_ui:
        from sr2.pipeline.trace import TraceCollector

        trace_collector = TraceCollector()

    agent = Agent(
        AgentConfig(
            name=name,
            config_dir=args.config_dir,
            defaults_path=args.defaults,
            trace_collector=trace_collector,
        )
    )

    # Wire CLI trace output if --inspect is set
    if args.inspect and trace_collector:
        from sr2.pipeline.trace_renderer import render_brief, render_default, render_full

        renderers = {"default": render_default, "full": render_full, "brief": render_brief}
        renderer = renderers[args.inspect]

        def _print_trace(trace):
            sys.stderr.write(renderer(trace))
            sys.stderr.write("\n")
            sys.stderr.flush()

        trace_collector.on_turn_complete(_print_trace)

    if args.single_shot:
        interface_name = args.single_shot
        message = args.prompt
        if message is None:
            if sys.stdin.isatty():
                print(
                    "Error: --single-shot requires a prompt argument or stdin input",
                    file=sys.stderr,
                )
                sys.exit(1)
            message = sys.stdin.read().strip()
        if not message:
            print("Error: empty message", file=sys.stderr)
            sys.exit(1)

        await agent.start()
        try:
            plugin = agent._plugins.get(interface_name)
            if plugin is None:
                print(
                    f"Error: interface '{interface_name}' not found. "
                    f"Available: {list(agent._plugins.keys())}",
                    file=sys.stderr,
                )
                sys.exit(1)

            response = await plugin.run(message)
            print(response)
        finally:
            await agent.shutdown()
        return

    if args.http:
        from sr2_runtime.http import create_http_app
        import uvicorn

        app = create_http_app(agent)
        uv_log_level = "warning" if (args.inspect or args.inspect_ui) else args.log_level.lower()
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=args.port,
            log_level=uv_log_level,
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

    # Suppress LiteLLM's "Give Feedback / Get Help" banner and info messages.
    # These are raw print() calls that bypass Python logging.
    import litellm

    litellm.suppress_debug_info = True
    litellm.set_verbose = False

    # When running as a subprocess with piped stderr, the default StreamHandler
    # crashes with BlockingIOError on large log bursts. Force blocking mode.
    if not os.isatty(sys.stderr.fileno()):
        os.set_blocking(sys.stderr.fileno(), True)

    log_handlers: list[logging.Handler] = []
    if args.inspect or args.inspect_ui:
        # When inspector is active, redirect logs to files so stderr stays clean
        # for inspector output. Log files sit next to the agent config by default.
        agent_log_file = args.agent_log_path or os.path.join(args.config_dir, "agent.log")
        infra_log_file = args.infra_log_path or os.path.join(args.config_dir, "infra.log")

        agent_handler = logging.FileHandler(agent_log_file, mode="a")
        agent_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        log_handlers.append(agent_handler)

        infra_handler = logging.FileHandler(infra_log_file, mode="a")
        infra_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )

        # Redirect all LiteLLM loggers to infra log (it creates multiple)
        litellm.suppress_debug_info = True
        for ll_name in ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router", "litellm"):
            ll_logger = logging.getLogger(ll_name)
            ll_logger.handlers = [infra_handler]
            ll_logger.propagate = False

        # Redirect uvicorn's loggers to infra log
        for uv_logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            uv_logger = logging.getLogger(uv_logger_name)
            uv_logger.handlers = [infra_handler]
            uv_logger.propagate = False

        sys.stderr.write(f"[inspect] Logs redirected to {agent_log_file} and {infra_log_file}\n")
        sys.stderr.flush()
    else:
        log_handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=log_handlers,
        force=True,
    )

    asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()
