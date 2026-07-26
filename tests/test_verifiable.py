"""Unit tests for verifiable rewards — prove scoring, not vibes."""

from __future__ import annotations

from anvil.recipes.verifiable import (
    DEFAULT_HARD_PROBLEMS,
    extract_first_int,
    exact_integer_reward,
    multi_gold_membership_reward,
)


def _ids_from_text(text: str) -> list[int]:
    """Toy detokenize path: encode chars as ordinals (reward only uses detokenize)."""
    return [ord(c) for c in text]


def _detok(tokens) -> str:
    return "".join(chr(int(t)) for t in tokens)


def test_extract_first_int():
    assert extract_first_int("the answer is 45.") == "45"
    assert extract_first_int("  -12  ") == "-12"
    assert extract_first_int("no digits") is None
    assert extract_first_int("4 is smaller than 45") == "4"


def test_exact_integer_reward_match_and_miss():
    reward = exact_integer_reward(_detok, "45")
    assert reward("", _ids_from_text("45")) == 1.0
    assert reward("", _ids_from_text("The answer is 45!")) == 1.0
    assert reward("", _ids_from_text("44")) == 0.0
    assert reward("", _ids_from_text("nope")) == 0.0
    assert reward("", []) == 0.0


def test_exact_integer_reward_rejects_wrong_first_int():
    """First integer wins — leading garbage numbers must not silently score."""
    reward = exact_integer_reward(_detok, "45")
    assert reward("", _ids_from_text("12 then 45")) == 0.0


def test_multi_gold_membership_is_loose():
    """Documents the multi-prompt failure mode: any gold scores 1.0."""
    loose = multi_gold_membership_reward(_detok, ["4", "8", "45"])
    # Always emitting "4" looks perfect even when the problem wanted 45.
    assert loose("", _ids_from_text("4")) == 1.0
    assert loose("", _ids_from_text("45")) == 1.0
    assert loose("", _ids_from_text("7")) == 0.0


def test_hard_problems_have_unique_golds():
    golds = [g for _, g in DEFAULT_HARD_PROBLEMS]
    assert len(golds) == len(set(golds))
    for prompt, gold in DEFAULT_HARD_PROBLEMS:
        assert gold.isdigit() or (gold.startswith("-") and gold[1:].isdigit())
        assert "only the number" in prompt.lower() or "number" in prompt.lower()


def test_select_problem_round_trip():
    # Import from demo script helpers without running main
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "grpo_observe_demo.py"
    spec = importlib.util.spec_from_file_location("grpo_observe_demo", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    u, g = mod.select_problem("hard", 0)
    assert g == "45"
    u2, g2 = mod.select_problem("easy", 0)
    assert g2 == "4"
    assert u != u2
