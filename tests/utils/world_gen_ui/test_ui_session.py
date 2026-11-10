"""The terrain generator UI's session: which stages run, which are stale, failures; and the map's layers."""

import copy
import dataclasses
import threading
import time
from concurrent.futures import Executor, Future

import numpy as np
import pytest

from common import SEED, SMALL, inline_executor
from worldgen.proc_gen.world import pipeline, preview
from worldgen.proc_gen.world.config import parse_config
from worldgen.proc_gen.world.pipeline import Stage, generate
from worldgen.utils.world_gen_ui.model.layers import LAYERS, has_layer, readout, render_layer
from worldgen.utils.world_gen_ui.model.session import Session
from worldgen.utils.world_gen_ui.model.stages import STAGES, Status
from worldgen.utils.world_gen_ui.model.terrain import picture

DONE, SKIPPED, STALE, NOT_RUN = Status.DONE, Status.SKIPPED, Status.STALE, Status.NOT_RUN


def small(**climate):
    data = copy.deepcopy(SMALL)
    data.setdefault("climate", {}).update(climate)
    return parse_config(data)


def run(session: Session, stage: str, config) -> list[str]:
    session.run(stage, config)
    return session.poll()


def test_a_late_stage_runs_the_missing_ones_first():
    config, session = small(), Session(SEED, inline_executor())
    assert session.run("climate", config) == ["elevation", "erosion", "hydrology", "climate"]
    assert session.busy  # until the controller takes in what happened
    assert session.poll() == ["elevation", "erosion", "hydrology", "climate"]
    assert not session.busy and session.running is None
    statuses = session.statuses(config)
    assert [statuses[name] for name in STAGES] == [DONE, SKIPPED, DONE, DONE, NOT_RUN]
    progress = session.states["climate"].progress
    assert progress.heights is not None and progress.hydrology is not None and progress.climate is not None
    assert session.states["climate"].seconds > 0


def test_the_whole_run_matches_generate():
    config, session = small(), Session(SEED, inline_executor())
    assert run(session, "biomes", config) == list(STAGES)
    assert [session.statuses(config)[name] for name in STAGES] == [DONE, SKIPPED, DONE, DONE, DONE]
    terrain, expected = session.states["biomes"].progress.world, generate(config, SEED)
    assert np.array_equal(terrain.elevation, expected.elevation) and np.array_equal(terrain.color, expected.color)


def test_an_edit_makes_its_stage_and_the_later_ones_stale():
    config, session = small(), Session(SEED, inline_executor())
    run(session, "biomes", config)
    edited = small(base_temperature=5.0)
    statuses = session.statuses(edited)
    assert [statuses[name] for name in STAGES] == [DONE, SKIPPED, DONE, STALE, STALE]
    assert session.plan("biomes", edited) == ["climate", "biomes"]
    assert session.plan("hydrology", edited) == ["hydrology"]  # up to date: runs just itself
    assert run(session, "biomes", edited) == ["climate", "biomes"]
    assert STALE not in session.statuses(edited).values()


def test_a_new_seed_makes_everything_stale():
    config, session = small(), Session(SEED, inline_executor())
    run(session, "biomes", config)
    session.seed = SEED + 1
    assert [session.statuses(config)[name] for name in STAGES] == [STALE] * 5
    assert session.plan("biomes", config) == ["elevation", "erosion", "hydrology", "climate", "biomes"]


def test_a_failing_stage_is_reported_and_the_later_ones_stay_as_they_were():
    config, session = small(), Session(SEED, inline_executor())
    run(session, "biomes", config)

    def boom(progress, config, seed):
        raise ValueError("boom")

    session.stages = tuple(dataclasses.replace(s, run=boom) if s.name == "climate" else s for s in pipeline.STAGES)
    edited = small(base_temperature=5.0)
    assert run(session, "biomes", edited) == ["climate"]
    climate = session.states["climate"]
    assert climate.status is Status.FAILED and climate.error == "ValueError: boom"
    assert session.states["biomes"].status is DONE
    assert not session.busy
    assert session.plan("biomes", edited)[0] == "climate"  # tried again next time


def stopping_at(session: Session, name: str) -> tuple[Stage, ...]:
    """The stages, with `name` asking the session to stop, then working on as if it had not heard."""

    def stop_then_work(progress, config, seed):
        assert session.stop()
        while True:
            time.sleep(0.001)

    return tuple(dataclasses.replace(s, run=stop_then_work) if s.name == name else s for s in pipeline.STAGES)


def test_stop_interrupts_the_running_stage_and_starts_no_more():
    config, session = small(), Session(SEED, inline_executor())
    run(session, "biomes", config)
    ran_before = session.states["climate"].progress
    session.stages = stopping_at(session, "climate")
    edited = small(base_temperature=5.0)
    assert run(session, "biomes", edited) == []
    assert not session.busy and not session.stopping and session.running is None
    statuses = session.statuses(edited)
    assert [statuses[name] for name in STAGES] == [DONE, SKIPPED, DONE, STALE, STALE]  # as before the run
    assert session.states["climate"].progress is ran_before
    assert not session.stop()  # nothing left to stop
    session.stages = pipeline.STAGES
    assert run(session, "biomes", edited) == ["climate", "biomes"]  # a new run is not stopped


def test_stop_reaches_a_stage_on_the_worker_thread():
    config, session = small(), Session(SEED)
    started = threading.Event()

    def work_forever(progress, config, seed):
        started.set()
        while True:
            time.sleep(0.001)

    session.stages = tuple(dataclasses.replace(s, run=work_forever) if s.name == "hydrology" else s for s in pipeline.STAGES)
    session.run("climate", config)
    assert started.wait(10)
    assert session.stop() and session.stopping
    deadline = time.perf_counter() + 10
    while session.busy and time.perf_counter() < deadline:
        session.poll()
        time.sleep(0.01)
    assert not session.busy
    assert [session.states[name].status for name in ("elevation", "hydrology", "climate")] == [DONE, NOT_RUN, NOT_RUN]
    session.close()


def test_only_one_run_at_a_time():
    class Never(Executor):
        def submit(self, fn, /, *args, **kwargs):
            return Future()

    session = Session(SEED, Never())
    session.run("elevation", small())
    with pytest.raises(RuntimeError, match="still running"):
        session.run("climate", small())


@pytest.fixture(scope="module")
def small_session():
    """SMALL, run to biomes."""
    config = small()
    session = Session(SEED, inline_executor())
    run(session, "biomes", config)
    return session, config


@pytest.fixture(scope="module")
def terrain_session():
    """SMALL with mountains, terraces and erosion, run to biomes."""
    data = copy.deepcopy(SMALL)
    data["elevation"].update({"max_height": 30.0,
                              "mountains": {"enabled": True, "wavelength": 40.0, "range_wavelength": 120.0},
                              "terraces": {"enabled": True, "where": "mountains", "step": 4.0}})
    data["erosion"] = {"enabled": True, "droplets": 0.3, "levels": 2, "lifetime": 10, "thermal_iterations": 5}
    config = parse_config(data)
    session = Session(SEED, inline_executor())
    run(session, "biomes", config)
    return session, config


def test_every_layer_draws(small_session, terrain_session):
    drawn = set()
    for session, config in (small_session, terrain_session):
        for layer, stage in LAYERS.items():
            progress = session.states[stage].progress
            if progress is None or session.states[stage].status is SKIPPED or not has_layer(layer, progress):
                continue
            image = render_layer(layer, progress, config)
            assert image.shape == (config.world.size_cells,) * 2 + (3,) and image.dtype == np.uint8, layer
            assert image.std() > 0, layer  # not one flat colour
            drawn.add(layer)
    assert drawn == set(LAYERS)


def test_the_new_ground_has_its_layers_and_facts(small_session, terrain_session):
    session, config = terrain_session
    assert session.states["erosion"].status is DONE
    progress = session.states["biomes"].progress
    for layer in ("mountain mask", "terrace mask", "erosion", "erosion change", "sediment"):
        assert has_layer(layer, progress), layer
    facts = dict(readout(progress, config, 70.0, 60.0).facts)
    assert {"mountains", "terraces", "erosion", "sediment", "relief", "slope"} <= set(facts)
    assert facts["erosion"].endswith(" m") and facts["erosion"][0] in "+-"
    plain = small_session[0].states["elevation"].progress
    assert not has_layer("mountain mask", plain) and not has_layer("terrace mask", plain)
    assert small_session[0].states["erosion"].status is SKIPPED


def test_the_biomes_layer_is_the_preview(small_session):
    session, config = small_session
    terrain = session.states["biomes"].progress.world
    assert np.array_equal(render_layer("biomes", session.states["biomes"].progress, config), preview.render(terrain))


def test_large_worlds_are_drawn_smaller(small_session):
    session, config = small_session
    image = render_layer("elevation", session.states["elevation"].progress, config, max_side=100)
    assert image.shape == (64, 64, 3)  # every 2nd cell of 128


def test_readout(small_session):
    session, config = small_session
    progress = session.states["biomes"].progress
    found = readout(progress, config, 64.0, 64.0)
    facts = dict(found.facts)
    assert found.title == "x 64 m, z 64 m" and facts["temperature"].endswith("°C") and "moisture" in facts
    assert facts["biome"] in {rule.id for rule in config.biomes.rules}
    assert found.text().startswith(found.title + " · ground ")  # the status bar's line
    early = readout(session.states["elevation"].progress, config, 64.0, 64.0)
    assert [name for name, _ in early.facts] == ["ground"]  # no water, biome or climate before their stages
    assert early.key == found.key  # the same cell
    assert readout(progress, config, -1.0, 10.0) is None


def test_the_3d_terrain_is_the_layer_on_the_ground_as_far_as_it_is_made(small_session):
    session, config = small_session
    early = picture("elevation", session.states["elevation"].progress, config)
    assert early.terrain.water is None  # no water before hydrology
    assert early.overview.height_at(30.0, 40.0) == pytest.approx(
        float(session.states["elevation"].progress.heights[40, 30]), abs=0.05)  # 1 m cells: vertex (30, 40)
    late = picture("biomes", session.states["biomes"].progress, config)
    terrain = session.states["biomes"].progress.world
    assert late.terrain.water is not None and late.overview.color.shape == (128, 128, 3)
    assert np.array_equal(late.image, render_layer("biomes", session.states["biomes"].progress, config))
    assert np.allclose(late.overview.height, terrain.elevation.astype(np.float32))
