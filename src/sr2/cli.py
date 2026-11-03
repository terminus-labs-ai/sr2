"""SR2 CLI — load_config, build_llm_clients, chat_loop, main."""

from __future__ import annotations

import asyncio
import sys

import yaml

from sr2.config.models import ConfigError, PipelineConfig, TransformerConfig
from sr2.integrations.litellm import LiteLLMCallable
from sr2.models import TextBlock
from sr2.orchestrator import SR2
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import LLMCallable, StreamEvent


def load_config(path: str) -> dict:
    """Read a YAML file and return its contents as a dict."""
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    models = config.get("models", {})
    for raw in config.get("pipeline", {}).get("transformers", []):
        transformer = TransformerConfig.model_validate(raw)
        if transformer.type != "episode_boundary":
            continue
        judge_model = transformer.config.get("judge_model", "default")
        if judge_model not in models:
            raise ConfigError(
                f"episode_boundary judge_model {judge_model!r} is absent from models"
            )
    return config


def build_llm_clients(models: dict) -> dict[str, LLMCallable]:
    """Build a LiteLLMCallable for each entry in the models dict."""
    clients: dict[str, LLMCallable] = {}
    for name, entry in models.items():
        entry = dict(entry)
        model = entry.pop("model")
        clients[name] = LiteLLMCallable(model=model, **entry)
    return clients


async def chat_loop(sr2) -> None:
    """Interactive REPL: read user input, stream responses from sr2.turn()."""
    while True:
        try:
            user_input = input("> ")
        except KeyboardInterrupt:
            return

        if user_input.strip().lower() in ("exit", "quit"):
            return

        try:
            async for event in sr2.turn([TextBlock(text=user_input)]):
                if event.type == "text":
                    print(event.text, end="", flush=True)
        except Exception as exc:
            print(f"Error: request failed — {exc}")
            continue

        print()


def main() -> None:
    """CLI entry point: load config, build clients, start chat loop."""
    path = sys.argv[1]
    config = load_config(path)
    llm_clients = build_llm_clients(config["models"])
    pipeline_config = PipelineConfig(**config["pipeline"])
    sr2 = SR2(pipeline_config, llm_clients, token_counter=CharacterTokenCounter())
    asyncio.run(chat_loop(sr2))
