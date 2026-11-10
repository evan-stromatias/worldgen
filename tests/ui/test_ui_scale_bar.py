import pytest

from worldgen.ui.view.gl.map_render import bar_length


@pytest.mark.parametrize(("longest", "expected"), [(1.0, 1.0), (1.9, 1.0), (4.99, 2.0), (7.0, 5.0), (120.0, 100.0),
                                                    (0.3, 0.2), (2500.0, 2000.0)])
def test_bar_length_is_round_and_fits(longest, expected):
    assert bar_length(longest) == pytest.approx(expected)
