"""The map's viewpoint, shared by 2D and 3D: zoom, pan, the 3D camera and picking on the ground."""

import math

import numpy as np
import pytest

from worldgen.utils.world_gen_ui.model.viewpoint import FOVY, MAX_PITCH, MIN_PITCH, Viewpoint


def test_it_is_kept_until_the_world_changes_size():
    view = Viewpoint()
    view.fit(1000.0)
    assert view.centre == (500.0, 500.0) and view.zoom == 1.0
    view.zoom_about(3)
    view.pan(100.0, -50.0)
    kept = (view.centre, view.zoom)
    view.fit(1000.0)  # a new picture of the same world
    assert (view.centre, view.zoom) == kept
    view.fit(2000.0)
    assert view.centre == (1000.0, 1000.0) and view.zoom == 1.0


def test_zooming_about_a_point_keeps_it_in_place():
    view = Viewpoint()
    view.fit(1000.0)
    point = (800.0, 300.0)
    offset = [p - c for p, c in zip(point, view.centre, strict=True)]
    view.zoom_about(2, point)
    scaled = [p - c for p, c in zip(point, view.centre, strict=True)]
    assert scaled == pytest.approx([o / 1.25 ** 2 for o in offset])  # the same pixels from the middle
    view.zoom_about(-50)
    assert view.zoom == 1.0


def test_the_3d_camera_shows_what_2d_shows_and_faces_its_heading():
    view = Viewpoint()
    view.fit(1000.0)
    view.zoom_about(4)
    eye, target = view.camera(10.0)
    assert tuple(target) == (view.centre[0], 10.0, view.centre[1])
    assert np.linalg.norm(eye - target) == pytest.approx(view.distance())
    assert 2 * view.distance() * math.tan(math.radians(FOVY) / 2) == pytest.approx(1000.0 / view.zoom)
    assert eye[2] > target[2]  # heading 0: from the south, looking north as 2D shows it
    view.orbit(math.pi / 2, 10.0)
    eye, target = view.camera(10.0)
    assert eye[0] < target[0] and view.pitch == MAX_PITCH  # heading east: from the west
    view.orbit(0.0, -10.0)
    assert view.pitch == MIN_PITCH


def test_dragging_moves_the_ground_with_the_mouse():
    view = Viewpoint()
    view.fit(1000.0)
    view.drag(10.0, 0.0, 2.0)  # to the right: the centre goes west
    assert view.centre == pytest.approx((480.0, 500.0))
    view.orbit(math.pi, 0.0)  # facing south, right is west
    view.drag(10.0, 0.0, 2.0)
    assert view.centre == pytest.approx((500.0, 500.0))


def test_the_middle_of_the_3d_view_picks_the_centre():
    view = Viewpoint()
    view.fit(1000.0)
    view.pan(-100.0, 50.0)
    def flat(x, z):
        return 5.0
    assert view.pick(5.0, 800, 600, 400, 300, flat) == pytest.approx((400.0, 550.0), abs=0.01)
    x, z = view.pick(5.0, 800, 600, 400, 200, flat)
    assert x == pytest.approx(400.0) and z < 550.0  # higher on the screen: farther north
    assert view.pick(5.0, 800, 600, 400, 300, lambda x, z: -1e6) is None  # off the world
