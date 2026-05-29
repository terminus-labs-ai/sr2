"""Tests for tightened fact-pattern matching in RuleBasedExtractor.

The fact pattern at extraction.py:29 used an optional subject prefix, meaning
bare verbs (has/is/uses/runs) matched anything — including non-technical prose.
These tests define the required boundary: technical-subject required.

All tests go through the public API: RuleBasedExtractor().extract(text).
No private-attribute assertions.
"""

from __future__ import annotations

import pytest

from sr2.memory import RuleBasedExtractor


class TestFactPatternRejectsGenericProse:
    """Sentences without a technical subject must not produce fact memories."""

    def test_she_has_nice_eyes_produces_no_memory(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("she has nice eyes")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert fact_memories == []

    def test_he_is_tall_produces_no_memory(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("he is tall and athletic")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert fact_memories == []

    def test_it_uses_magic_produces_no_memory(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("it uses magic to make things work")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert fact_memories == []

    def test_bare_has_sentence_produces_no_memory(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("has a really great personality overall")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert fact_memories == []

    def test_bare_runs_sentence_produces_no_memory(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("runs every morning before breakfast")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert fact_memories == []


class TestFactPatternRequiresTechnicalSubject:
    """Sentences with a technical subject must produce a fact memory."""

    def test_project_uses_fastapi_produces_fact(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("the project uses FastAPI for the REST layer")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert len(fact_memories) >= 1

    def test_code_has_memory_leak_produces_fact(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("the code has a memory leak in the event loop")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert len(fact_memories) >= 1

    def test_system_runs_on_python_produces_fact(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("the system runs on Python 3.11")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert len(fact_memories) >= 1

    def test_repo_uses_pytest_produces_fact(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("the repo uses pytest with xdist")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert len(fact_memories) >= 1

    def test_app_is_stateless_produces_fact(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("the app is stateless and horizontally scalable")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert len(fact_memories) >= 1

    def test_service_has_rate_limiting_produces_fact(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("the service has rate limiting on all public endpoints")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert len(fact_memories) >= 1

    def test_database_runs_postgres_produces_fact(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("the database runs Postgres 15")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert len(fact_memories) >= 1

    def test_api_uses_jwt_produces_fact(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("the api uses JWT for authentication")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert len(fact_memories) >= 1

    def test_config_has_feature_flags_produces_fact(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("the config has feature flags for rollout")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert len(fact_memories) >= 1

    def test_server_runs_behind_nginx_produces_fact(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("the server runs behind nginx")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert len(fact_memories) >= 1

    def test_our_code_has_produces_fact(self):
        """'our code' (no 'the') should also be accepted."""
        extractor = RuleBasedExtractor()
        result = extractor.extract("our code has strict type checking enabled")
        fact_memories = [m for m in result.memories if "fact" in m.tags]
        assert len(fact_memories) >= 1


class TestOtherPatternsUnaffected:
    """Verify that non-fact patterns still work after the fix."""

    def test_preference_pattern_still_works(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("I prefer dark mode in all editors")
        pref_memories = [m for m in result.memories if "preference" in m.tags]
        assert len(pref_memories) >= 1

    def test_decision_pattern_still_works(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("decided to use Redis for the cache layer")
        decision_memories = [m for m in result.memories if "decision" in m.tags]
        assert len(decision_memories) >= 1

    def test_correction_pattern_still_works(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract(
            "don't do that again, this was wrong approach for this kind of problem"
        )
        correction_memories = [m for m in result.memories if "correction" in m.tags]
        assert len(correction_memories) >= 1

    def test_tooling_pattern_still_works(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract("installed ruff as the linter and formatter")
        tooling_memories = [m for m in result.memories if "tooling" in m.tags]
        assert len(tooling_memories) >= 1
