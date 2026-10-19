"""RGBSpace stacks collected camera images channels first at the configured size."""

import numpy as np
from sensor_msgs.msg import Image

from rosnav_rl.observations.data_sources.collectors import ImageColorCollector
from rosnav_rl.spaces.observation_space.spaces.perception import RGBSpace


def _collected(encoding: str, pixels: np.ndarray) -> np.ndarray:
    collector = ImageColorCollector("camera", topic="image")
    height, width, _ = pixels.shape
    collector.update(Image(encoding=encoding, height=height, width=width, data=pixels.astype(np.uint8).tobytes()))
    return collector.get_observation()


def _solid(height: int, width: int, channels: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(np.array(channels, dtype=np.uint8), (height, width, len(channels))).copy()


def test_a_collected_non_square_image_is_resized_into_its_slot():
    pixels = np.random.default_rng(0).integers(1, 256, size=(30, 50, 3), dtype=np.uint8)
    space = RGBSpace(rgb_sources=["wrist_color"], image_height=16, image_width=24)

    out = space.encode_observation(wrist_color=_collected("rgb8", pixels))

    assert out.shape == (1, 3, 16, 24)
    assert out.dtype == np.uint8
    assert (out > 0).all()
    assert space.get_gym_space().contains(out)


def test_an_image_at_the_configured_size_keeps_its_pixels():
    pixels = np.random.default_rng(1).integers(0, 256, size=(12, 20, 3), dtype=np.uint8)
    space = RGBSpace(rgb_sources=["wrist_color"], image_height=12, image_width=20)

    out = space.encode_observation(wrist_color=_collected("rgb8", pixels))

    np.testing.assert_array_equal(out[0], pixels.transpose(2, 0, 1))


def test_alpha_is_dropped_from_channels_first_images():
    space = RGBSpace(rgb_sources=["wrist_color"], image_height=8, image_width=10)

    out = space.encode_observation(wrist_color=_collected("rgba8", _solid(18, 30, (10, 20, 30, 255))))

    assert (out[0] == np.array([10, 20, 30], dtype=np.uint8)[:, None, None]).all()


def test_channels_last_images_with_alpha_are_accepted():
    space = RGBSpace(rgb_sources=["wrist_color"], image_height=8, image_width=10)

    out = space.encode_observation(wrist_color=_solid(18, 30, (40, 50, 60, 255)))

    assert (out[0] == np.array([40, 50, 60], dtype=np.uint8)[:, None, None]).all()


def test_each_source_fills_its_own_sorted_slot_and_missing_ones_stay_zero():
    space = RGBSpace(rgb_sources=["wrist_color", "arm_base_color"], image_height=8, image_width=10)

    out = space.encode_observation(wrist_color=_collected("rgb8", _solid(9, 14, (70, 80, 90))), arm_base_color=None)

    assert out.shape == (2, 3, 8, 10)
    assert (out[0] == 0).all()
    assert (out[1] == np.array([70, 80, 90], dtype=np.uint8)[:, None, None]).all()


def test_single_channel_images_are_skipped():
    space = RGBSpace(rgb_sources=["wrist_color"], image_height=8, image_width=10)

    out = space.encode_observation(wrist_color=np.full((8, 10), 200, dtype=np.uint8))

    assert (out == 0).all()
