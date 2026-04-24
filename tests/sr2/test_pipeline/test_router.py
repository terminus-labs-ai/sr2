import pytest
import yaml

from sr2.config.loader import ConfigLoader
from sr2.config.models import PipelineConfig
from sr2.pipeline.router import InterfaceRouter


def test_route_dict_config_returns_valid_pipeline_config():
    """Route with dict config returns a valid PipelineConfig."""
    loader = ConfigLoader()
    router = InterfaceRouter(
        {"chat": {"token_budget": 24000}},
        loader,
    )
    config = router.route("chat")
    assert isinstance(config, PipelineConfig)
    assert config.token_budget == 24000


def test_route_caches_result():
    """Route with same interface twice returns the cached (identical) object."""
    loader = ConfigLoader()
    router = InterfaceRouter(
        {"chat": {"token_budget": 24000}},
        loader,
    )
    first = router.route("chat")
    second = router.route("chat")
    assert first is second


def test_unknown_interface_raises_key_error():
    """Unknown interface type raises KeyError."""
    loader = ConfigLoader()
    router = InterfaceRouter({}, loader)
    with pytest.raises(KeyError, match="Unknown interface type: nope"):
        router.route("nope")


def test_registered_interfaces_lists_all():
    """registered_interfaces returns all registered interface names."""
    loader = ConfigLoader()
    router = InterfaceRouter(
        {"chat": {}, "agent": {}},
        loader,
    )
    assert sorted(router.registered_interfaces) == ["agent", "chat"]


def test_route_from_yaml(tmp_path):
    """Route with a file path loads config from YAML."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump({"pipeline": {"token_budget": 16000}}))

    loader = ConfigLoader()
    router = InterfaceRouter({"test": str(config_file)}, loader)
    config = router.route("test")
    assert isinstance(config, PipelineConfig)
    assert config.token_budget == 16000
