import pytest

from aiplays.actions import Action, parse_actions


def test_parse_actions_and_reject_duplicates() -> None:
    assert parse_actions(["noop", "a"]) == (Action.NOOP, Action.A)
    with pytest.raises(ValueError):
        parse_actions(["a", "a"])
