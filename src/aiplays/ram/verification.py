from __future__ import annotations

from aiplays.ram.pokemon_red import PokemonRedRamReader


def decode_memory(memory: object) -> dict[str, int | list[int]]:
    return PokemonRedRamReader().read(memory)
