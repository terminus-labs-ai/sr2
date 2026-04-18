import pytest
import yaml

from sr2.config.loader import ConfigLoader
from sr2.config.models import PipelineConfig
from sr2.pipeline.router import InterfaceRouter


def test_invalidate_causes_reload_from_disk(tmp_path):
    """After invalidate(), the next route() re-reads from disk instead of serving stale cache."""
    config_file = tmp_path / "chat.yaml"
    config_file.write_text(yaml.dump({"pipeline": {"token_budget": 16000}}))

    loader = ConfigLoader()
    router = InterfaceRouter({"chat": str(config_file)}, loader)

    # First route — caches the result
    first = router.route("chat")
    assert first.token_budget == 16000

    # Mutate the file on disk
    config_file.write_text(yaml.dump({"pipeline": {"token_budget": 32000}}))

    # Route again — should still return stale cached value
    stale = router.route("chat")
    assert stale.token_budget == 16000

    # Invalidate then re-route — should pick up new file content
    router.invalidate("chat")
    refreshed = router.route("chat")
    assert refreshed.token_budget == 32000


def test_invalidate_nonexistent_is_noop():
    """Invalidating an interface that was never cached does not raise."""
    loader = ConfigLoader()
    router = InterfaceRouter({"chat": {"token_budget": 8000}}, loader)

    # Never routed, so nothing cached — should be a silent no-op
    router.invalidate("chat")

    # Totally unknown interface — also a no-op
    router.invalidate("does_not_exist")


def test_invalidate_allows_reroute_to_refresh(tmp_path):
    """After invalidate(), route() returns a new object (not the same identity)."""
    config_file = tmp_path / "agent.yaml"
    config_file.write_text(yaml.dump({"pipeline": {"token_budget": 20000}}))

    loader = ConfigLoader()
    router = InterfaceRouter({"agent": str(config_file)}, loader)

    first = router.route("agent")
    router.invalidate("agent")
    second = router.route("agent")

    # Both are valid configs with the same values, but different objects
    assert first is not second
    assert isinstance(second, PipelineConfig)
    assert second.token_budget == 20000


def test_reload_returns_updated_config(tmp_path):
    """reload_interface() returns a config reflecting the updated YAML on disk."""
    config_file = tmp_path / "chat.yaml"
    config_file.write_text(yaml.dump({"pipeline": {"token_budget": 16000}}))

    loader = ConfigLoader()
    router = InterfaceRouter({"chat": str(config_file)}, loader)

    # Initial route
    original = router.route("chat")
    assert original.token_budget == 16000

    # Edit the file
    config_file.write_text(yaml.dump({"pipeline": {"token_budget": 48000}}))

    # reload_interface invalidates + re-routes, returns the new config
    reloaded = router.reload_interface("chat")
    assert isinstance(reloaded, PipelineConfig)
    assert reloaded.token_budget == 48000

    # Subsequent route() should return the same (now-cached) reloaded config
    assert router.route("chat") is reloaded


def test_reload_invalid_yaml_raises_validation_error(tmp_path):
    """reload_interface() raises when the updated YAML is invalid."""
    config_file = tmp_path / "chat.yaml"
    config_file.write_text(yaml.dump({"pipeline": {"token_budget": 16000}}))

    loader = ConfigLoader()
    router = InterfaceRouter({"chat": str(config_file)}, loader)
    router.route("chat")

    # Replace with invalid config (token_budget must be positive int)
    config_file.write_text(yaml.dump({"pipeline": {"token_budget": "not_a_number"}}))

    with pytest.raises(Exception):
        # Should raise a validation error (pydantic or similar)
        router.reload_interface("chat")


def test_reload_after_invalid_leaves_cache_empty(tmp_path):
    """After a failed reload, the cache must not contain stale data — next route() also fails."""
    config_file = tmp_path / "chat.yaml"
    config_file.write_text(yaml.dump({"pipeline": {"token_budget": 16000}}))

    loader = ConfigLoader()
    router = InterfaceRouter({"chat": str(config_file)}, loader)

    # Cache a valid config
    original = router.route("chat")
    assert original.token_budget == 16000

    # Break the file
    config_file.write_text(yaml.dump({"pipeline": {"token_budget": "garbage"}}))

    # Reload should fail
    with pytest.raises(Exception):
        router.reload_interface("chat")

    # Cache should be empty — route() must NOT serve the old stale config
    with pytest.raises(Exception):
        router.route("chat")
