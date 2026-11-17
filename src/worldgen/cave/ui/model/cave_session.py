"""Cave generation session: runs cave generation stages incrementally on worker threads."""

from __future__ import annotations

import ctypes
import hashlib
import json
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from worldgen.cave.config import CaveConfig
from worldgen.cave.export import render_cave_preview
from worldgen.cave.generator import CaveData
from worldgen.cave.mesh import build_cave_mesh, build_cave_water_mesh
from worldgen.cave.pipeline import CAVE_STAGES, CaveProgress, CaveStage, current_floor
from worldgen.mesh.builders import MeshArrays
from worldgen.ui.model.stages import Status

if TYPE_CHECKING:
    pass


class CaveStageStopped(BaseException):
    """Raised in the worker to stop the stage currently running."""


@dataclass(slots=True)
class CaveStageState:
    status: Status = Status.NOT_RUN
    seconds: float = 0.0
    key: str = ""
    progress: CaveProgress | None = None
    config: CaveConfig | None = None
    error: str = ""


class CaveSession:
    """Manages incremental generation of a subterranean cave and its 3D/2D views."""

    def __init__(
        self,
        config: CaveConfig,
        seed: int,
        executor: Executor | None = None,
        stages: tuple[CaveStage, ...] = CAVE_STAGES,
    ):
        self.config = config
        self.seed = seed
        self.stages = stages
        self.states = {stage.name: CaveStageState() for stage in stages}

        self.cave: CaveData | None = None
        self.mesh_arrays: MeshArrays | None = None
        self.water_mesh_arrays: MeshArrays | None = None
        self.preview_image: np.ndarray | None = None
        self.stride: int = 1
        self.cutaway_height: float | None = None
        self.include_ceiling: bool = True

        self._executor = executor or ThreadPoolExecutor(1, thread_name_prefix="cave_stages")
        self._posted: queue.SimpleQueue = queue.SimpleQueue()
        self._planned: list[str] = []
        self._before: dict[str, Status] = {}
        self.running_stage: str | None = None
        self._started = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._in_stage: int | None = None

        # Backward compatibility for legacy thread-based start_generation()
        self._thread: threading.Thread | None = None
        self.status: str = "idle"

    @property
    def running(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return True
        return bool(self._planned)

    @property
    def busy(self) -> bool:
        return self.running

    @property
    def stopping(self) -> bool:
        return self.busy and self._stop.is_set()

    def settled(self) -> bool:
        return not self.running

    def elapsed(self) -> float:
        return time.perf_counter() - self._started if self.running_stage else 0.0

    def keys(self, config: CaveConfig) -> dict[str, str]:
        """Hash-chain inputs for each stage in order."""
        keys, previous = {}, str(self.seed)
        for stage in self.stages:
            sections = {name: getattr(config, name).model_dump(mode="json") for name in stage.sections}
            blob = json.dumps({"previous": previous, "stage": stage.name, "sections": sections}, sort_keys=True)
            keys[stage.name] = previous = hashlib.sha256(blob.encode()).hexdigest()[:20]
        return keys

    def statuses(self, config: CaveConfig | None) -> dict[str, Status]:
        """Return each stage's status, marking stale if recipe inputs changed."""
        keys = self.keys(config) if config is not None else {}
        result = {}
        for name, state in self.states.items():
            status = state.status
            if status in (Status.DONE, Status.SKIPPED) and keys and state.key != keys.get(name):
                status = Status.STALE
            result[name] = status
        return result

    def up_to_date(self, config: CaveConfig) -> bool:
        """Every stage has run with the recipe and seed as they are now."""
        statuses = self.statuses(config)
        return bool(statuses) and all(s in (Status.DONE, Status.SKIPPED) for s in statuses.values())

    def stage_texts(self, config: CaveConfig | None) -> dict[str, tuple[Status, str]]:
        """Human-readable status and time strings for the UI."""
        statuses = self.statuses(config)
        result = {}
        for stage in self.stages:
            name = stage.name
            status = statuses[name]
            state = self.states[name]
            if status == Status.RUNNING:
                text = f"running {self.elapsed():.1f} s"
            elif status == Status.DONE:
                text = f"done {state.seconds:.2f} s"
            elif status == Status.SKIPPED:
                text = stage.skipped or "skipped"
            elif status == Status.FAILED:
                text = "failed"
            elif status == Status.STALE:
                text = "stale"
            elif status == Status.QUEUED:
                text = "waiting"
            else:
                text = "not run"
            result[name] = (status, text)
        return result

    def plan(self, stage: str, config: CaveConfig) -> list[str]:
        """Compute the sequence of stages that need to run up to `stage`."""
        statuses = self.statuses(config)
        names = [s.name for s in self.stages]
        end = names.index(stage)
        first = next((i for i in range(end) if statuses[names[i]] not in (Status.DONE, Status.SKIPPED)), end)
        return names[first : end + 1]

    def run(self, stage: str, config: CaveConfig) -> list[str]:
        """Start running `stage` (and any required prior stages) asynchronously."""
        if self.busy:
            raise RuntimeError(f"cannot run {stage}: {self.running_stage or 'worker'} is still running")
        self.config = config
        plan = self.plan(stage, config)
        names = [s.name for s in self.stages]
        first = names.index(plan[0])
        progress = self.states[names[first - 1]].progress if first else CaveProgress()
        keys = self.keys(config)
        self._planned = list(plan)
        self._before = {name: self.states[name].status for name in plan}
        for name in plan:
            self.states[name].status = Status.QUEUED
        jobs = [(s, keys[s.name]) for s in self.stages if s.name in plan]
        self._stop.clear()
        self.status = "generating"
        self._executor.submit(self._work, jobs, progress, config, self.seed)
        return plan

    def run_all(self, config: CaveConfig) -> list[str]:
        """Run all stages in the pipeline."""
        return self.run(self.stages[-1].name, config)

    def reset_stage(self, stage_name: str) -> None:
        """Mark stage and downstream stages as not run."""
        names = [s.name for s in self.stages]
        if stage_name not in names:
            return
        idx = names.index(stage_name)
        for name in names[idx:]:
            self.states[name] = CaveStageState()

    def stop(self) -> bool:
        """Interrupt any running worker task."""
        if not self.busy:
            return False
        with self._lock:
            if not self._stop.is_set():
                self._stop.set()
                if self._in_stage is not None:
                    _raise_in(self._in_stage, CaveStageStopped)
        return True

    def _work(
        self,
        jobs: list[tuple[CaveStage, str]],
        progress: CaveProgress,
        config: CaveConfig,
        seed: int,
    ) -> None:
        for stage, key in jobs:
            self._posted.put(("started", stage.name, time.perf_counter()))
            start = time.perf_counter()
            try:
                with self._lock:
                    if self._stop.is_set():
                        raise CaveStageStopped
                    self._in_stage = threading.get_ident()
                try:
                    if not stage.applies(config):
                        self._posted.put(("skipped", stage.name, key, progress, config, 0.0))
                        continue
                    progress = stage.run(progress, config, seed)
                finally:
                    with self._lock:
                        self._in_stage = None
                        _raise_in(threading.get_ident(), None)
            except CaveStageStopped:
                self._in_stage = None
                self._posted.put(("stopped", stage.name))
                return
            except Exception as e:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                self._posted.put(("failed", stage.name, f"{type(e).__name__}: {e}"))
                return

            self._posted.put(("done", stage.name, key, progress, config, time.perf_counter() - start))

    def poll(self) -> bool:
        """Poll completed stages and update meshes/previews. Returns True when new data is ready."""
        # Check legacy thread first
        if self._thread is not None:
            if self._thread.is_alive():
                return False
            self._thread = None
            self.status = "settled"
            return True

        updated = False
        while True:
            try:
                kind, name, *rest = self._posted.get_nowait()
            except queue.Empty:
                break

            state = self.states[name]
            if kind == "started":
                state.status, self.running_stage, self._started = Status.RUNNING, name, rest[0]
            elif kind == "failed":
                state.status, state.error = Status.FAILED, rest[0]
                for later in self._planned[1:]:
                    self.states[later].status = self._before[later]
                self._planned, self.running_stage = [], None
                self.status = f"error: {state.error}"
            elif kind == "stopped":
                for later in self._planned:
                    self.states[later].status = self._before[later]
                self._planned, self.running_stage = [], None
                self.status = "stopped"
            else:
                key, progress, config, seconds = rest
                state.status = Status.DONE if kind == "done" else Status.SKIPPED
                state.key, state.progress, state.config, state.seconds, state.error = (
                    key,
                    progress,
                    config,
                    seconds,
                    "",
                )
                self._planned.remove(name)
                if not self._planned:
                    self.running_stage = None
                    self.status = "settled"

                # If cavern or materials ready, update cave data and meshes
                if progress.cave is not None:
                    self.cave = progress.cave
                    self._update_derivatives()
                    updated = True
                elif progress.ceiling_elev is not None and progress.clearance is not None:
                    # Provisional cave data after cavern stage
                    self.cave = self._provisional_cave(progress, config)
                    self._update_derivatives()
                    updated = True

        return updated

    def _provisional_cave(self, p: CaveProgress, config: CaveConfig) -> CaveData:
        floor = current_floor(p)
        assert p.ceiling_elev is not None
        assert p.clearance is not None
        assert p.pillar_mask is not None
        size_cells = config.world.size_cells
        water = p.water if p.water is not None else np.full_like(floor, np.nan)
        sediment = p.sediment if p.sediment is not None else np.zeros((size_cells, size_cells), np.float32)
        temperature = p.temperature if p.temperature is not None else np.full((size_cells, size_cells), 12.0, np.float32)
        moisture = p.moisture if p.moisture is not None else np.full((size_cells, size_cells), 0.8, np.float32)
        material = p.material if p.material is not None else np.zeros((size_cells, size_cells), np.uint8)
        color = p.color if p.color is not None else np.full((size_cells, size_cells, 3), 128, np.uint8)

        return CaveData(
            seed=self.seed,
            config=config,
            floor_elevation=floor,
            ceiling_elevation=p.ceiling_elev,
            clearance=p.clearance,
            pillar_mask=p.pillar_mask,
            water=water,
            material=material,
            color=color,
            temperature=temperature,
            moisture=moisture,
            sediment=sediment,
        )

    def _update_derivatives(self) -> None:
        if self.cave is None:
            return
        self.mesh_arrays = build_cave_mesh(
            self.cave,
            stride=self.stride,
            include_ceiling=self.include_ceiling,
            cutaway_height=self.cutaway_height,
        )
        self.water_mesh_arrays = build_cave_water_mesh(self.cave, stride=self.stride)
        self.preview_image = render_cave_preview(self.cave)

    def rebuild_mesh(self) -> None:
        """Rebuild meshes from existing cave data (e.g. after cutaway toggle)."""
        self._update_derivatives()

    def run_sync(self, config: CaveConfig | None = None) -> None:
        """Run all stages synchronously and build meshes."""
        if config is not None:
            self.config = config
        progress = CaveProgress()
        keys = self.keys(self.config)
        for stage in self.stages:
            t0 = time.perf_counter()
            if stage.applies(self.config):
                progress = stage.run(progress, self.config, self.seed)
                state = self.states[stage.name]
                state.status = Status.DONE
                state.key = keys[stage.name]
                state.progress = progress
                state.config = self.config
                state.seconds = time.perf_counter() - t0
            else:
                state = self.states[stage.name]
                state.status = Status.SKIPPED
                state.key = keys[stage.name]
                state.progress = progress
                state.config = self.config
                state.seconds = 0.0

        assert progress.cave is not None
        self.cave = progress.cave
        self._update_derivatives()
        self.status = "settled"

    def start_generation(self) -> None:
        """Legacy helper: runs run_sync() in a background thread."""
        if self.running:
            return
        self.status = "generating"
        self._thread = threading.Thread(target=self.run_sync, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop background execution and shutdown executor."""
        self.stop()
        if isinstance(self._executor, ThreadPoolExecutor):
            self._executor.shutdown(wait=False, cancel_futures=True)


def _raise_in(thread: int, exception: type[BaseException] | None) -> None:
    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread), ctypes.py_object(exception) if exception else None)
