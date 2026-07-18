from aiplays.rewards.composite import CompositeReward


def test_rewards_are_deltas_and_coordinates_include_map() -> None:
    reward = CompositeReward({"new_coordinate": 1.0, "step": -0.1})
    first = {"ram": {"map_id": 1, "x": 2, "y": 3}}
    reward.reset(first)
    assert reward.value(first) == -0.1
    assert reward.value({"ram": {"map_id": 2, "x": 2, "y": 3}}) == 0.9
