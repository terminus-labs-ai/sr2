import pytest


@pytest.fixture
def sample_layer_config():
    """A minimal layer config dict for testing."""
    return {
        "name": "core",
        "cache_policy": "immutable",
        "contents": [
            {"key": "system_prompt", "source": "config"},
        ],
    }


@pytest.fixture
def sample_interface_config():
    """A minimal complete interface pipeline config dict for testing.

    These are the raw pipeline fields (not wrapped in pipeline:) for direct
    use with PipelineConfig(**sample_interface_config).
    """
    return {
        "token_budget": 32000,
        "pre_rot_threshold": 0.25,
        "compaction": {"enabled": False},
        "summarization": {"enabled": False},
        "retrieval": {"enabled": False},
        "intent_detection": {"enabled": False},
        "layers": [
            {
                "name": "core",
                "cache_policy": "immutable",
                "contents": [
                    {"key": "system_prompt", "source": "config"},
                ],
            },
        ],
    }
