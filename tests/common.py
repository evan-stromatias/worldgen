"""Small terrains for tests: a recipe, a seed, and a synchronous executor for the UI's workers."""

from concurrent.futures import Executor, Future

# 128² cells, with the defaults scaled down to match, so lakes and rivers still form
SMALL = {
    "world": {"size_cells": 128, "patch_cells": 32},
    "elevation": {"noise": {"frequency": 0.02}, "warp": {"strength": 10, "frequency": 0.01}},
    "hydrology": {"river_threshold": 60, "lake_min_area": 6, "rain": {"frequency": 0.01}},
    "climate": {"moisture": {"water_distance": 8}},
    "biomes": {
        "rules": [
            {"id": "sea_floor", "when": {"water": "sea"}, "ground": {"color": [150, 140, 105]}},
            {"id": "lake_bed", "when": {"water": "lake"}, "ground": {"color": [96, 92, 70]}},
            {"id": "beach", "when": {"elevation": [-1, 1.5]}, "ground": {"color": [205, 190, 140]}},
            {"id": "forest", "when": {"moisture": [0.5, 1.0]}, "ground": {"color": [78, 118, 60], "jitter": 0.07}},
            {"id": "default", "ground": {"color": [112, 152, 82]}},
        ]
    },
}
SEED = 3


def inline_executor() -> Executor:
    """An executor that runs jobs at once on the calling thread (synchronous, for tests)."""

    class Inline(Executor):
        def submit(self, fn, /, *args, **kwargs):
            future: Future = Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as e:  # noqa: BLE001 - the job's own error, raised again by future.result()
                future.set_exception(e)
            return future

    return Inline()
