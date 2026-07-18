from __future__ import annotations

from typing import Any

import numpy as np


class PokemonRedRamReader:
    """Conservative Pokemon Red RAM decoder.

    Addresses follow the `pret/pokered` disassembly symbol names (wCurMap, wYCoord,
    wXCoord, wIsInBattle, wPartyCount, wPartyMon1HP, wPartyMon1Level, wObtainedBadges,
    wPlayerMoney). Verify against the user's ROM/version with `verify-ram`; questionable
    event/Pokedex/battle rewards deliberately are not enabled by default.
    """

    CUR_MAP, Y, X, IN_BATTLE = 0xD35E, 0xD361, 0xD362, 0xD057
    PARTY_COUNT, PARTY_HP, PARTY_LEVEL, BADGES = 0xD163, 0xD16C, 0xD18C, 0xD356
    MONEY = 0xD347
    feature_space = 12

    @staticmethod
    def _byte(memory: Any, address: int) -> int:
        return int(memory[address])

    def read(self, memory: Any) -> dict[str, int | list[int]]:
        party_size = min(6, self._byte(memory, self.PARTY_COUNT))
        levels: list[int] = []
        hp: list[int] = []
        # Party structs are 44 bytes; bounds make corrupted/incompatible state visibly benign.
        for index in range(party_size):
            base = index * 44
            levels.append(min(100, self._byte(memory, self.PARTY_LEVEL + base)))
            hp.append(
                min(
                    999,
                    (self._byte(memory, self.PARTY_HP + base) << 8)
                    | self._byte(memory, self.PARTY_HP + base + 1),
                )
            )
        money = min(
            999_999,
            (self._byte(memory, self.MONEY) << 16)
            | (self._byte(memory, self.MONEY + 1) << 8)
            | self._byte(memory, self.MONEY + 2),
        )
        return {
            "map_id": self._byte(memory, self.CUR_MAP),
            "x": self._byte(memory, self.X),
            "y": self._byte(memory, self.Y),
            "in_battle": int(bool(self._byte(memory, self.IN_BATTLE))),
            "party_size": party_size,
            "party_hp": hp,
            "party_levels": levels,
            "party_level_total": sum(levels),
            "badges": self._byte(memory, self.BADGES),
            "money": money,
        }

    def features(self, values: dict[str, int | list[int]]) -> np.ndarray:
        raw_levels = values.get("party_levels", [])
        raw_hp = values.get("party_hp", [])
        levels = raw_levels[:6] if isinstance(raw_levels, list) else []
        hp = raw_hp[:3] if isinstance(raw_hp, list) else []

        def number(name: str) -> int:
            value = values.get(name, 0)
            return value if isinstance(value, int) else 0

        result = (
            [
                number("map_id") / 255,
                number("x") / 255,
                number("y") / 255,
                number("in_battle"),
                number("party_size") / 6,
                number("badges") / 255,
            ]
            + [int(level) / 100 for level in levels]
            + [int(value) / 999 for value in hp]
        )
        return np.asarray(
            (result + [0.0] * self.feature_space)[: self.feature_space], dtype=np.float32
        )
