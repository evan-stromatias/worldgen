import math

import numpy as np
import pytest

from worldgen.viewer.camera import FlyCamera, FlyInput
from worldgen.viewer.geometry import (
    Frustum,
    forward_vector,
    minimap_camera,
    pick_on_terrain,
    screen_ray,
    view_footprint,
)


def test_directions_follow_the_generator():
    np.testing.assert_allclose(forward_vector(0, 0), [0, 0, -1], atol=1e-12)  # north is -z
    np.testing.assert_allclose(forward_vector(math.pi / 2, 0), [1, 0, 0], atol=1e-12)  # east is +x
    assert forward_vector(0, -math.pi / 4)[1] < 0


def test_frustum_sees_ahead_not_behind():
    frustum = Frustum.from_camera(np.zeros(3), forward_vector(0, 0), 60, 16 / 9)
    assert frustum.sees_box((-1, -1, -11, 1, 1, -9))  # 10 ahead
    assert not frustum.sees_box((-1, -1, 9, 1, 1, 11))  # 10 behind
    assert not frustum.sees_box((50, -1, -11, 52, 1, -9))  # far off to the side
    lo = np.array([[-1, -1, -11], [-1, -1, 9]], float)
    np.testing.assert_array_equal(frustum.sees_boxes(lo, lo + 2), [True, False])


def test_screen_ray_through_the_centre_hits_the_target():
    eye, target = np.array([0.0, 10.0, 10.0]), np.array([0.0, 0.0, 0.0])
    direction = screen_ray(eye, target, 45, 200, 200, 100, 100)
    np.testing.assert_allclose(direction, (target - eye) / np.linalg.norm(target - eye))
    left = screen_ray(eye, target, 45, 200, 200, 0, 100)
    assert left[0] < 0  # the left edge looks towards -x


def test_minimap_pick_finds_the_centre_and_corners():
    world = 1024.0
    cam = minimap_camera(world, 0.0, heading=0.0)

    def flat(x, z):
        return 0.0

    x, z = pick_on_terrain(cam, 256, 128, 128, flat, 4.0, 0.0, world)
    assert x == pytest.approx(world / 2, abs=1) and z == pytest.approx(world / 2, abs=1)
    north = pick_on_terrain(cam, 256, 128, 100, flat, 4.0, 0.0, world)
    assert north[1] < world / 2  # up the minimap is north
    assert pick_on_terrain(cam, 256, 128, 2, flat, 4.0, 0.0, world) is None  # sky beyond the far edge


def test_minimap_pick_accounts_for_raised_terrain():
    world = 1024.0
    cam = minimap_camera(world, 0.0, heading=0.0)
    flat = pick_on_terrain(cam, 256, 128, 128, lambda x, z: 0.0, 4.0, 0.0, world)
    hill = pick_on_terrain(cam, 256, 128, 128, lambda x, z: 30.0, 4.0, 0.0, world)
    assert hill[1] > flat[1]  # seen from the south, a raised surface is hit nearer the camera


def test_footprint_has_four_corners_within_reach():
    corners = view_footprint(np.array([0, 50.0, 0]), forward_vector(0, -0.2), 60, 1.5, 0.0, 500.0)
    assert len(corners) == 4
    assert all(np.hypot(c[0], c[2]) <= 500.0 + 1e-6 for c in corners)


def make_camera() -> FlyCamera:
    return FlyCamera(np.array([0.0, 50.0, 0.0]), 0.0, 0.0, speed=10.0, boost=4.0, sensitivity=0.01)


def test_camera_moves_and_stays_above_ground():
    cam = make_camera()
    cam.update(1.0, FlyInput(move=(0, 0, 1)), lambda x, z: 0.0)
    assert cam.position[2] < 0  # forward is north
    cam.update(10.0, FlyInput(move=(0, -1, 0)), lambda x, z: 20.0)
    assert cam.position[1] == pytest.approx(22.0)  # clamped above the ground


def test_camera_is_faster_higher_up_and_with_boost():
    low, high, boosted = make_camera(), make_camera(), make_camera()
    high.position[1] = 600.0
    low.update(1.0, FlyInput(move=(0, 0, 1)), lambda x, z: 0.0)
    high.update(1.0, FlyInput(move=(0, 0, 1)), lambda x, z: 0.0)
    boosted.update(1.0, FlyInput(move=(0, 0, 1), boost=True), lambda x, z: 0.0)
    assert -high.position[2] > -low.position[2] and -boosted.position[2] == pytest.approx(-low.position[2] * 4)


def test_glide_eases_to_the_target_and_turns_the_short_way():
    cam = make_camera()
    cam.yaw = math.radians(350)
    cam.glide_to(np.array([100.0, 60.0, 0.0]), math.radians(10), -0.5, duration=1.0)
    for _ in range(5):
        cam.update(0.1, FlyInput(), lambda x, z: 0.0)
    assert math.degrees(cam.yaw) % 360 < 10 or math.degrees(cam.yaw) > 350  # through north, not round
    for _ in range(6):
        cam.update(0.1, FlyInput(), lambda x, z: 0.0)
    assert cam.glide is None
    np.testing.assert_allclose(cam.position, [100, 60, 0])
    assert cam.pitch == pytest.approx(-0.5)


def test_moving_cancels_a_glide():
    cam = make_camera()
    cam.glide_to(np.array([500.0, 60.0, 0.0]), 0.0, 0.0, duration=2.0)
    cam.update(0.1, FlyInput(move=(1, 0, 0)), lambda x, z: 0.0)
    assert cam.glide is None
