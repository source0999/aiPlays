from __future__ import annotations

from enum import StrEnum


class Action(StrEnum):
    NOOP = "noop"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    A = "a"
    B = "b"
    START = "start"
    SELECT = "select"


DEFAULT_ACTIONS = tuple(Action)


def parse_actions(values: list[str]) -> tuple[Action, ...]:
    actions = tuple(Action(value.lower()) for value in values)
    if not actions:
        raise ValueError("emulator.enabled_actions cannot be empty")
    if len(set(actions)) != len(actions):
        raise ValueError("emulator.enabled_actions cannot contain duplicates")
    return actions
