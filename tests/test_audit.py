"""Gate-override audit events — every force=True past a blocked gate is logged.

Phase 2 start of the control-plane audit trail (Phase 5 builds the multi-user
log on top). Preview/enumeration paths that force gates merely to display
them must NOT record.
"""

from __future__ import annotations

import json

import pytest

from anvil.control.audit import AuditLog, default_log, gate_override_event
from anvil.recipes.profiles import plan_recipe, suggest_for_model

EDGE_VLM = "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.fixture(autouse=True)
def _clean_default_log():
    default_log().clear()
    yield
    default_log().clear()


def test_forced_blocked_plan_records_event() -> None:
    plan = plan_recipe(
        base_model=EDGE_VLM,
        recipe_id="sft_chat_moe",  # blocked for edge_student
        fetch_remote=False,
        force=True,
    )
    assert plan.gate is not None and plan.gate["level"] == "blocked"

    events = default_log().events(kind="gate_override")
    assert len(events) == 1
    e = events[0]
    assert e.kind == "gate_override"
    assert e.recipe_id == "sft_chat_moe"
    assert e.shape == "edge_student"
    assert e.base_model
    assert e.blocked_reasons  # the "why" must be in the trail
    assert e.at  # ISO timestamp present


def test_blocked_without_force_records_nothing() -> None:
    with pytest.raises(ValueError, match="blocked"):
        plan_recipe(
            base_model=EDGE_VLM,
            recipe_id="sft_chat_moe",
            fetch_remote=False,
            force=False,
        )
    assert default_log().events() == []


def test_recommended_plan_records_nothing() -> None:
    plan_recipe(
        base_model=EDGE_VLM,
        recipe_id="vlm_sft_edge",  # recommended for edge_student
        fetch_remote=False,
    )
    assert default_log().events() == []


def test_suggest_preview_does_not_record() -> None:
    out = suggest_for_model(EDGE_VLM, include_blocked=True)
    assert out["recipes"]  # blocked previews were built (they force internally)
    assert default_log().events() == []


def test_record_override_false_suppresses() -> None:
    plan = plan_recipe(
        base_model=EDGE_VLM,
        recipe_id="sft_chat_moe",
        fetch_remote=False,
        force=True,
        record_override=False,
    )
    assert plan.gate is not None and plan.gate["level"] == "blocked"
    assert default_log().events() == []


def test_jsonl_sink_roundtrip(tmp_path) -> None:
    sink = tmp_path / "audit" / "trail.jsonl"
    log = AuditLog(jsonl_path=sink)
    log.record(
        gate_override_event(
            recipe_id="sft_chat_moe",
            base_model=EDGE_VLM,
            shape="edge_student",
            blocked_reasons=("moe recipe on edge vlm",),
            stretch_reasons=(),
        )
    )
    lines = sink.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["kind"] == "gate_override"
    assert row["recipe_id"] == "sft_chat_moe"
    assert row["blocked_reasons"] == ["moe recipe on edge vlm"]
    # in-memory view matches
    assert log.events()[0].recipe_id == "sft_chat_moe"
