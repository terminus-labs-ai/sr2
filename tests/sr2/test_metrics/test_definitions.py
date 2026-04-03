"""Tests for metric definitions."""

from sr2.metrics.definitions import (
    DEFAULT_THRESHOLDS,
    MetricNames,
    MetricSnapshot,
    MetricThreshold,
    MetricValue,
)


class TestMetricSnapshot:

    def test_add_stores_metric_with_labels(self):
        """MetricSnapshot.add() stores metric with labels."""
        snap = MetricSnapshot(
            invocation_id="inv_1",
            agent_name="test-agent",
            interface_type="user_message",
        )
        snap.add("my_metric", 42.0, "tokens", stage="system_prompt")

        assert len(snap.metrics) == 1
        m = snap.metrics[0]
        assert m.name == "my_metric"
        assert m.value == 42.0
        assert m.labels["stage"] == "system_prompt"

    def test_get_retrieves_by_name(self):
        """MetricSnapshot.get() retrieves by name."""
        snap = MetricSnapshot(
            invocation_id="inv_2",
            agent_name="agent",
            interface_type="api",
        )
        snap.add("alpha", 1.0)
        snap.add("beta", 2.0)

        assert snap.get("beta").value == 2.0

    def test_get_returns_none_for_unknown(self):
        """MetricSnapshot.get() returns None for unknown metric."""
        snap = MetricSnapshot(
            invocation_id="inv_3",
            agent_name="agent",
            interface_type="api",
        )
        assert snap.get("nonexistent") is None

    def test_labels_include_agent_and_interface(self):
        """Labels include agent name and interface type from snapshot."""
        snap = MetricSnapshot(
            invocation_id="inv_4",
            agent_name="my-agent",
            interface_type="slack",
        )
        snap.add("test_metric", 10.0)

        m = snap.metrics[0]
        assert m.labels["agent"] == "my-agent"
        assert m.labels["interface"] == "slack"


class TestMetricThreshold:

    def test_less_than_triggered(self):
        """MetricThreshold.is_triggered() with < condition works."""
        t = MetricThreshold("cache", "<", 0.5)
        assert t.is_triggered(0.3) is True
        assert t.is_triggered(0.7) is False

    def test_greater_than_triggered(self):
        """MetricThreshold.is_triggered() with > condition works."""
        t = MetricThreshold("latency", ">", 500)
        assert t.is_triggered(600) is True
        assert t.is_triggered(400) is False


class TestDefaults:

    def test_default_thresholds_has_critical(self):
        """DEFAULT_THRESHOLDS has entries for critical metrics."""
        names = [t.metric_name for t in DEFAULT_THRESHOLDS]
        assert MetricNames.FULL_DEGRADATION_EVENTS in names
        assert MetricNames.CACHE_HIT_RATE in names

    def test_metric_value_auto_timestamps(self):
        """MetricValue auto-timestamps."""
        mv = MetricValue(name="test", value=1.0)
        assert mv.timestamp is not None
