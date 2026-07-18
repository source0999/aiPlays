import numpy as np

from aiplays.config import ObservationConfig
from aiplays.observations import ObservationProcessor


def test_rgba_buffer_is_copied_and_matches_space() -> None:
    source = np.zeros((144, 160, 4), dtype=np.uint8)
    source[..., 0] = 17
    processor = ObservationProcessor(ObservationConfig(width=8, height=6, frame_stack=2))
    observation = processor.reset(source)
    source[..., 0] = 255
    assert observation.dtype == np.uint8 and observation.shape == (2, 6, 8)
    assert int(observation[0, 0, 0]) != 255 and processor.pixel_space.contains(observation)
