"""Tests for prefix stability tracking."""

from sr2.pipeline.prefix_tracker import PrefixTracker, PrefixSnapshot


class TestPrefixTracker:
    def test_first_call_no_comparison(self):
        """First call returns prefix_stable=True with first_invocation=True."""
        tracker = PrefixTracker()
        snap = PrefixSnapshot(
            full_hash="abc123", layer_hashes={"core": "h1"}, prefix_tokens=100
        )
        report = tracker.compare(snap, actual_cached_tokens=0)

        assert report.prefix_stable is True
        assert report.first_invocation is True
        assert report.changed_layers == []
        assert report.expected_cached_tokens == 100
        assert report.actual_cached_tokens == 0

    def test_same_prefix_stable(self):
        """Same prefix hash across calls reports prefix_stable=True."""
        tracker = PrefixTracker()
        snap1 = PrefixSnapshot(
            full_hash="abc", layer_hashes={"core": "h1", "mem": "h2"}, prefix_tokens=200
        )
        tracker.compare(snap1, actual_cached_tokens=0)

        snap2 = PrefixSnapshot(
            full_hash="abc", layer_hashes={"core": "h1", "mem": "h2"}, prefix_tokens=200
        )
        report = tracker.compare(snap2, actual_cached_tokens=180)

        assert report.prefix_stable is True
        assert report.first_invocation is False
        assert report.changed_layers == []
        assert report.expected_cached_tokens == 200
        assert report.actual_cached_tokens == 180

    def test_cache_efficiency_computed(self):
        """Efficiency is actual/expected cached tokens."""
        tracker = PrefixTracker()
        snap1 = PrefixSnapshot(full_hash="a", layer_hashes={}, prefix_tokens=200)
        tracker.compare(snap1, actual_cached_tokens=0)

        snap2 = PrefixSnapshot(full_hash="a", layer_hashes={}, prefix_tokens=200)
        report = tracker.compare(snap2, actual_cached_tokens=100)

        assert report.cache_efficiency == 0.5

    def test_changed_prefix_detected(self):
        """Changed prefix hash reports prefix_stable=False with changed layers."""
        tracker = PrefixTracker()
        snap1 = PrefixSnapshot(
            full_hash="old",
            layer_hashes={"core": "h1", "mem": "h2"},
            prefix_tokens=200,
        )
        tracker.compare(snap1, actual_cached_tokens=0)

        snap2 = PrefixSnapshot(
            full_hash="new",
            layer_hashes={"core": "h1", "mem": "h3"},  # mem changed
            prefix_tokens=200,
        )
        report = tracker.compare(snap2, actual_cached_tokens=50)

        assert report.prefix_stable is False
        assert report.changed_layers == ["mem"]

    def test_added_layer_detected(self):
        """New layer appearing is detected as a change."""
        tracker = PrefixTracker()
        snap1 = PrefixSnapshot(
            full_hash="v1", layer_hashes={"core": "h1"}, prefix_tokens=100
        )
        tracker.compare(snap1, actual_cached_tokens=0)

        snap2 = PrefixSnapshot(
            full_hash="v2",
            layer_hashes={"core": "h1", "extra": "h2"},
            prefix_tokens=150,
        )
        report = tracker.compare(snap2, actual_cached_tokens=80)

        assert report.prefix_stable is False
        assert "extra" in report.changed_layers

    def test_reset_clears_state(self):
        """reset() clears state so next compare behaves like first call."""
        tracker = PrefixTracker()
        snap1 = PrefixSnapshot(full_hash="a", layer_hashes={}, prefix_tokens=100)
        tracker.compare(snap1, actual_cached_tokens=0)

        tracker.reset()

        snap2 = PrefixSnapshot(full_hash="b", layer_hashes={}, prefix_tokens=200)
        report = tracker.compare(snap2, actual_cached_tokens=0)

        # After reset, this is treated as first call
        assert report.prefix_stable is True
        assert report.changed_layers == []

    def test_zero_expected_tokens_efficiency(self):
        """Zero expected tokens yields 0.0 efficiency (no division error)."""
        tracker = PrefixTracker()
        snap = PrefixSnapshot(full_hash="a", layer_hashes={}, prefix_tokens=0)
        report = tracker.compare(snap, actual_cached_tokens=0)
        assert report.cache_efficiency == 0.0
