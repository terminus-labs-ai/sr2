"""Tests for alert rule engine."""

from datetime import UTC, datetime, timedelta

import pytest

from sr2.metrics.alerts import Alert, AlertRuleEngine
from sr2.metrics.definitions import MetricNames, MetricSnapshot, MetricThreshold


def _make_snapshot(metric_name: str, value: float) -> MetricSnapshot:
    snap = MetricSnapshot(
        invocation_id="inv_1",
        agent_name="agent",
        interface_type="api",
    )
    snap.add(metric_name, value)
    return snap


class TestAlertRuleEngine:

    @pytest.mark.asyncio
    async def test_metric_above_threshold_generates_alert(self):
        """Metric above threshold -> alert generated."""
        thresholds = [MetricThreshold(MetricNames.RETRIEVAL_LATENCY_MS, ">", 500)]
        engine = AlertRuleEngine(thresholds=thresholds)

        snap = _make_snapshot(MetricNames.RETRIEVAL_LATENCY_MS, 600.0)
        alerts = await engine.check(snap)

        assert len(alerts) == 1
        assert alerts[0].metric_name == MetricNames.RETRIEVAL_LATENCY_MS

    @pytest.mark.asyncio
    async def test_metric_below_threshold_no_alert(self):
        """Metric below threshold -> no alert."""
        thresholds = [MetricThreshold(MetricNames.RETRIEVAL_LATENCY_MS, ">", 500)]
        engine = AlertRuleEngine(thresholds=thresholds)

        snap = _make_snapshot(MetricNames.RETRIEVAL_LATENCY_MS, 200.0)
        alerts = await engine.check(snap)

        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_alert_includes_severity_and_message(self):
        """Alert includes correct severity and message."""
        thresholds = [
            MetricThreshold(MetricNames.FULL_DEGRADATION_EVENTS, ">", 0, "critical")
        ]
        engine = AlertRuleEngine(thresholds=thresholds)

        snap = _make_snapshot(MetricNames.FULL_DEGRADATION_EVENTS, 1.0)
        alerts = await engine.check(snap)

        assert alerts[0].severity == "critical"
        assert "actual=" in alerts[0].message

    @pytest.mark.asyncio
    async def test_suppression_prevents_re_alert(self):
        """Suppression: same metric doesn't re-alert within window."""
        thresholds = [MetricThreshold(MetricNames.FALLBACK_RATE, ">", 0.1)]
        engine = AlertRuleEngine(thresholds=thresholds)

        snap = _make_snapshot(MetricNames.FALLBACK_RATE, 0.5)
        alerts1 = await engine.check(snap)
        alerts2 = await engine.check(snap)

        assert len(alerts1) == 1
        assert len(alerts2) == 0  # suppressed

    @pytest.mark.asyncio
    async def test_suppression_expires(self):
        """After suppression expires -> alert fires again."""
        thresholds = [MetricThreshold(MetricNames.FALLBACK_RATE, ">", 0.1)]
        engine = AlertRuleEngine(thresholds=thresholds)
        engine._suppression_seconds = 0.0  # Disable suppression window

        snap = _make_snapshot(MetricNames.FALLBACK_RATE, 0.5)
        alerts1 = await engine.check(snap)

        # Force suppression to be in the past
        engine._suppressed[MetricNames.FALLBACK_RATE] = datetime.now(UTC) - timedelta(
            seconds=1
        )
        alerts2 = await engine.check(snap)

        assert len(alerts1) == 1
        assert len(alerts2) == 1

    @pytest.mark.asyncio
    async def test_alert_callback_called(self):
        """alert_callback called when alert fires."""
        received = []

        async def callback(alert):
            received.append(alert)

        thresholds = [MetricThreshold(MetricNames.FALLBACK_RATE, ">", 0.1)]
        engine = AlertRuleEngine(thresholds=thresholds, alert_callback=callback)

        snap = _make_snapshot(MetricNames.FALLBACK_RATE, 0.5)
        await engine.check(snap)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_alert_history_stores_alerts(self):
        """alert_history stores all fired alerts."""
        thresholds = [MetricThreshold(MetricNames.FALLBACK_RATE, ">", 0.1)]
        engine = AlertRuleEngine(thresholds=thresholds)
        engine._suppression_seconds = 0.0

        snap = _make_snapshot(MetricNames.FALLBACK_RATE, 0.5)
        await engine.check(snap)

        # Force suppression expiry for second check
        engine._suppressed.clear()
        await engine.check(snap)

        assert len(engine.alert_history) == 2

    def test_clear_history(self):
        """clear_history() empties history."""
        engine = AlertRuleEngine()
        engine._alert_history.append(
            Alert(
                metric_name="test",
                actual_value=1.0,
                threshold_value=0.5,
                condition=">",
                severity="warning",
            )
        )
        engine.clear_history()
        assert len(engine.alert_history) == 0

    @pytest.mark.asyncio
    async def test_clear_suppressions_allows_re_alerting(self):
        """clear_suppressions() allows immediate re-alerting."""
        thresholds = [MetricThreshold(MetricNames.FALLBACK_RATE, ">", 0.1)]
        engine = AlertRuleEngine(thresholds=thresholds)

        snap = _make_snapshot(MetricNames.FALLBACK_RATE, 0.5)
        await engine.check(snap)
        engine.clear_suppressions()
        alerts = await engine.check(snap)

        assert len(alerts) == 1

    @pytest.mark.asyncio
    async def test_missing_metric_no_alert(self):
        """Missing metric in snapshot -> no alert (not an error)."""
        thresholds = [MetricThreshold(MetricNames.CACHE_HIT_RATE, "<", 0.5)]
        engine = AlertRuleEngine(thresholds=thresholds)

        # Snapshot has a different metric
        snap = _make_snapshot(MetricNames.PIPELINE_TOTAL_TOKENS, 100.0)
        alerts = await engine.check(snap)

        assert len(alerts) == 0
