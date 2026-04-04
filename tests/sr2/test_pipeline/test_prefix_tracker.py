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


class TestToolSchemasInPrefixTracking:
    """Tests for tool schema inclusion in prefix hash tracking."""

    def test_tool_schemas_included_in_snapshot(self):
        """Tool schemas are hashed and stored in snapshot."""
        tracker = PrefixTracker()
        tools = [{"name": "search", "parameters": {"type": "object"}}]
        snap = tracker.snapshot(
            layer_hashes={"core": "h1"},
            full_hash="abc",
            prefix_tokens=100,
            tool_schemas=tools,
        )
        assert snap.tools_hash is not None

    def test_stable_tools_no_change(self):
        """Same tool schemas across invocations -> tools_changed=False."""
        tracker = PrefixTracker()
        tools = [{"name": "search", "parameters": {"type": "object"}}]

        snap1 = tracker.snapshot({"core": "h1"}, "abc", 100, tool_schemas=tools)
        tracker.compare(snap1, actual_cached_tokens=0)

        snap2 = tracker.snapshot({"core": "h1"}, "abc", 100, tool_schemas=tools)
        report = tracker.compare(snap2, actual_cached_tokens=90)

        assert report.tools_changed is False
        assert report.prefix_stable is True

    def test_changed_tools_detected(self):
        """Different tool schemas -> tools_changed=True, prefix_stable=False."""
        tracker = PrefixTracker()

        tools1 = [{"name": "search", "parameters": {"type": "object"}}]
        snap1 = tracker.snapshot({"core": "h1"}, "abc", 100, tool_schemas=tools1)
        tracker.compare(snap1, actual_cached_tokens=0)

        tools2 = [{"name": "search_v2", "parameters": {"type": "object"}}]
        snap2 = tracker.snapshot({"core": "h1"}, "abc", 100, tool_schemas=tools2)
        report = tracker.compare(snap2, actual_cached_tokens=50)

        assert report.tools_changed is True
        assert report.prefix_stable is False

    def test_tools_changed_flag_accurate_on_first_invocation(self):
        """First invocation always has tools_changed=False."""
        tracker = PrefixTracker()
        tools = [{"name": "search"}]
        snap = tracker.snapshot({"core": "h1"}, "abc", 100, tool_schemas=tools)
        report = tracker.compare(snap, actual_cached_tokens=0)

        assert report.tools_changed is False
        assert report.first_invocation is True

    def test_no_tools_hash_when_no_schemas(self):
        """When no tool_schemas provided, tools_hash is None."""
        tracker = PrefixTracker()
        snap = tracker.snapshot({"core": "h1"}, "abc", 100)
        assert snap.tools_hash is None

    def test_tool_ordering_static_first(self):
        """suggest_tool_ordering puts static tools first, dynamic last."""
        tools = [
            {"name": "dynamic_tool"},
            {"name": "stable_tool"},
            {"name": "static_tool"},
        ]
        ordered = PrefixTracker.suggest_tool_ordering(
            tools,
            static_tools={"static_tool"},
            dynamic_tools={"dynamic_tool"},
        )
        names = [t["name"] for t in ordered]
        assert names == ["static_tool", "stable_tool", "dynamic_tool"]

    def test_tool_ordering_no_hints(self):
        """Without static/dynamic hints, all tools are session-stable."""
        tools = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        ordered = PrefixTracker.suggest_tool_ordering(tools)
        assert len(ordered) == 3

    def test_deterministic_tool_hash(self):
        """Same tools in same order produce identical hash."""
        tools = [{"name": "a", "params": {"x": 1}}, {"name": "b"}]
        h1 = PrefixTracker._hash_tools(tools)
        h2 = PrefixTracker._hash_tools(tools)
        assert h1 == h2

    def test_different_tools_different_hash(self):
        """Different tools produce different hash."""
        h1 = PrefixTracker._hash_tools([{"name": "a"}])
        h2 = PrefixTracker._hash_tools([{"name": "b"}])
        assert h1 != h2
