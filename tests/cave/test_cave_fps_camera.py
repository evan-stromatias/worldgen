import math

import numpy as np
import pytest

from worldgen.cave.ui.model.fps_camera import FpsCamera
from worldgen.viewer.camera import FlyInput


def test_fps_camera_look_turn():
    cam = FpsCamera(position=np.array([10.0, 5.0, 10.0]))
    cam.turn((100.0, 50.0))  # dx, dy
    assert cam.yaw != 0.0
    assert cam.pitch != 0.0

    # Max pitch clamp
    cam.turn((0.0, -10000.0))
    assert cam.pitch <= math.radians(88.0)


def test_fps_camera_movement_and_vertical_clamping():
    cam = FpsCamera(
        position=np.array([10.0, 5.0, 10.0]),
        yaw=0.0,
        pitch=0.0,
        speed=10.0,
        eye_height=1.7,
        head_margin=0.3,
    )

    def floor_at(x, z):
        return 2.0

    def ceiling_at(x, z):
        return 12.0

    def is_navigable(x, z):
        return True

    # Move forward (ahead = 1.0) for dt = 1.0
    inp = FlyInput(move=(0.0, 0.0, 1.0))
    cam.update(1.0, inp, floor_at, ceiling_at, is_navigable)

    # In grounded mode, position[1] sits at floor_at + eye_height
    assert cam.position[1] == pytest.approx(2.0 + 1.7)
    # Forward at yaw 0: x is unchanged, z is -10.0 (north is -z)
    assert cam.position[2] == pytest.approx(0.0)


def test_fps_camera_pillar_collision_and_sliding():
    cam = FpsCamera(position=np.array([5.0, 5.0, 5.0]), yaw=0.0, pitch=0.0, speed=2.0)

    def floor_at(x, z):
        return 0.0

    def ceiling_at(x, z):
        return 20.0

    # Navigable everywhere except where z < 4.0
    def is_navigable(x, z):
        return z >= 4.0

    # Attempt to walk north into blocked z < 4.0
    inp = FlyInput(move=(0.0, 0.0, 1.0))  # forward steps in -z
    cam.update(1.0, inp, floor_at, ceiling_at, is_navigable)

    # Step would have landed at z = 3.0, but blocked!
    assert cam.position[2] >= 4.0


def test_fps_camera_glide():
    cam = FpsCamera(position=np.array([0.0, 0.0, 0.0]), yaw=0.0, pitch=0.0)
    target = np.array([20.0, 5.0, 20.0])
    cam.glide_to(target, yaw=math.pi / 2, pitch=0.1, duration=1.0)

    assert cam.glide is not None

    def floor_at(x, z):
        return 0.0

    def ceiling_at(x, z):
        return 20.0

    def is_navigable(x, z):
        return True

    # Halfway
    cam.update(0.5, FlyInput(), floor_at, ceiling_at, is_navigable)
    assert cam.glide is not None

    # Complete glide
    cam.update(0.6, FlyInput(), floor_at, ceiling_at, is_navigable)
    assert cam.glide is None
    np.testing.assert_allclose(cam.position[[0, 2]], target[[0, 2]], atol=1.0)
