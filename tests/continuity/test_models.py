"""sr2-95 / AC3 — immutable continuity records (unit layer, models half of the cage).

Case-table rows covered here:
  AC3-r8  — ContinuityRecord is frozen: attempting to set a field raises
            (pydantic ValidationError or equivalent); the record cannot be
            mutated after creation.
  AC3-r9  — ContinuityView is frozen: attempting to set a field raises.

Red mechanism (deliberate): the model symbols are imported at module level
from their declared landing spot (``sr2.pipeline.continuity.models``, via
tests/continuity/_fixtures.py). Until the implementer adds the module,
collection fails with ImportError naming exactly that module — the expected
red, not scaffolding.

Over-specification guard: the exception type is deliberately NOT pinned
beyond ``Exception`` + mutability-not-taken semantics for attribute sets
(pydantic frozen models raise ValidationError; plain frozen dataclasses
raise FrozenInstanceError — the bead's AC text allows either). What IS
pinned: the attempt raises and the record's field value is unchanged
afterwards. The mutable-default hazard (``meta``) IS pinned to pydantic's
ValidationError because the spec interface declares these models as pydantic
(``BaseModel, frozen=True``), and pydantic wraps ``__setitem__`` on
mutable model attributes with a validation error.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from continuity._fixtures import (
    ContextAuthority,
    ContextScope,
    ContinuityEventKind,
    ContinuityKind,
    ContinuityRecord,
    ContinuityView,
    make_record,
)


def _make_view(record: ContinuityRecord) -> ContinuityView:
    return ContinuityView(
        subject_id=record.subject_id,
        current_record=record,
        status="active",
        history_ids=(record.id,),
    )


# ---------------------------------------------------------------------------
# AC3-r8 — ContinuityRecord immutability
# ---------------------------------------------------------------------------


class TestContinuityRecordIsFrozen:
    def test_record_constructs_with_all_spec_fields(self):
        """Control for AC3-r8: a record with every spec-interface field
        round-trips through construction with the given values."""
        scope = ContextScope(owner_id="owner-1")
        record = ContinuityRecord(
            id="rec-0001",
            subject_id="backend-choice",
            event_kind=ContinuityEventKind.FORMED,
            kind=ContinuityKind.DECISION,
            content="we use Postgres",
            scope=scope,
            authority=ContextAuthority.USER,
            provenance_ids=("prov-1", "prov-2"),
            supersedes=(),
            confidence=0.9,
            salience=0.5,
            created_at=make_record(
                "x", "x", ContinuityEventKind.FORMED
            ).created_at,
        )
        assert record.id == "rec-0001"
        assert record.subject_id == "backend-choice"
        assert record.event_kind is ContinuityEventKind.FORMED
        assert record.kind is ContinuityKind.DECISION
        assert record.content == "we use Postgres"
        assert record.scope == scope
        assert record.authority is ContextAuthority.USER
        assert tuple(record.provenance_ids) == ("prov-1", "prov-2")
        assert tuple(record.supersedes) == ()
        assert record.confidence == 0.9
        assert record.salience == 0.5

    @pytest.mark.parametrize(
        "field_name,new_value",
        [
            ("content", "mutated content"),
            ("subject_id", "someone-elses-subject"),
            ("event_kind", ContinuityEventKind.SUPPRESSED),
            ("kind", ContinuityKind.CORRECTION),
            ("confidence", 0.0001),
            ("salience", 0.9999),
            ("supersedes", ("rec-9999",)),
        ],
    )
    def test_setting_a_field_raises_and_leaves_value_unchanged(
        self, field_name, new_value
    ):
        """AC3-r8: assignment to any ContinuityRecord field raises; the
        stored value is unchanged after the failed attempt."""
        record = make_record(
            "rec-0001", "backend-choice", ContinuityEventKind.FORMED
        )
        before = getattr(record, field_name)
        with pytest.raises(Exception) as excinfo:
            setattr(record, field_name, new_value)
        # A frozen pydantic model must report the mutation attempt as a
        # validation failure — any exception type that is NOT a
        # mutability-related error means the model is not actually frozen.
        assert not isinstance(excinfo.value, (AssertionError, AttributeError)), (
            "expected a mutability error (pydantic ValidationError or "
            "frozen-instance error), got "
            f"{type(excinfo.value).__name__}: {excinfo.value}"
        )
        assert getattr(record, field_name) == before

    def test_mutating_meta_dict_raises(self):
        """AC3-r8 (mutable-hazard side): even the dict-typed ``meta`` field
        must reject mutation through the model — pydantic frozen models
        return read-only mappings, so ``record.meta["k"] = v`` raises
        ValidationError (a TypeError from a custom read-only mapping is
        also acceptable; silently succeeding is not)."""
        record = make_record(
            "rec-0001", "backend-choice", ContinuityEventKind.FORMED
        )
        with pytest.raises((ValidationError, TypeError)):
            record.meta["injected"] = "value"

    def test_records_are_hashable_or_at_least_value_comparable(self):
        """AC3-r8 support: two identically-built records compare equal
        (value semantics) — the byte-identity rows rely on equality of
        unmutated re-reads matching captured snapshots."""
        a = make_record("rec-0001", "s", ContinuityEventKind.FORMED)
        b = make_record("rec-0001", "s", ContinuityEventKind.FORMED)
        assert a == b


# ---------------------------------------------------------------------------
# AC3-r9 — ContinuityView immutability
# ---------------------------------------------------------------------------


class TestContinuityViewIsFrozen:
    def test_view_constructs_with_spec_fields(self):
        """Control for AC3-r9: ContinuityView takes subject_id,
        current_record, status, history_ids and holds them."""
        record = make_record("rec-0001", "backend-choice", ContinuityEventKind.FORMED)
        view = ContinuityView(
            subject_id="backend-choice",
            current_record=record,
            status="active",
            history_ids=("rec-0001",),
        )
        assert view.subject_id == "backend-choice"
        assert view.current_record == record
        assert view.status == "active"
        assert tuple(view.history_ids) == ("rec-0001",)

    @pytest.mark.parametrize(
        "field_name,new_value",
        [
            ("subject_id", "mutated-subject"),
            ("status", "suppressed"),
            ("history_ids", ("rec-0001", "rec-0002")),
        ],
    )
    def test_setting_a_view_field_raises_and_leaves_value_unchanged(
        self, field_name, new_value
    ):
        """AC3-r9: assignment to any ContinuityView field raises; value
        unchanged afterwards."""
        record = make_record("rec-0001", "backend-choice", ContinuityEventKind.FORMED)
        view = _make_view(record)
        before = getattr(view, field_name)
        with pytest.raises(Exception) as excinfo:
            setattr(view, field_name, new_value)
        assert not isinstance(excinfo.value, (AssertionError, AttributeError)), (
            "expected a mutability error (pydantic ValidationError or "
            "frozen-instance error), got "
            f"{type(excinfo.value).__name__}: {excinfo.value}"
        )
        assert getattr(view, field_name) == before

    def test_setting_current_record_raises(self):
        """AC3-r9: current_record is not re-assignable either."""
        record = make_record("rec-0001", "backend-choice", ContinuityEventKind.FORMED)
        other = make_record("rec-0002", "backend-choice", ContinuityEventKind.VERIFIED)
        view = _make_view(record)
        with pytest.raises(Exception):
            view.current_record = other
        assert view.current_record == record
