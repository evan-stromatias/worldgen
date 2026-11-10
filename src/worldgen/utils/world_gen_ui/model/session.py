"""The stages run so far: each one's result, how long it took, and whether it is still up to date.

A stage is up to date while the seed, the recipe sections it reads and every stage before it are as they
were when it ran: its key chains its own inputs onto the previous stage's key. Editing a section makes
the stages that read it stale, and every stage after them.

Stages run on one worker thread. The worker only posts what happened to a queue; `poll`, called by the
controller each frame, applies it, so the session's state only ever changes on the main thread.
A run can be stopped: the stage running is interrupted by an exception raised in the worker thread, which
lands at its next Python instruction (so after the numpy call it is in), and no later stage starts.
"""

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

from worldgen.proc_gen.world import pipeline
from worldgen.proc_gen.world.config import TerrainConfig
from worldgen.proc_gen.world.pipeline import Progress, Stage
from worldgen.utils.world_gen_ui.model.stages import Status


class StageStopped(BaseException):
    """Raised in the worker to stop the stage it runs; not an Exception, so no `except Exception` keeps it."""


@dataclass(slots=True)
class StageState:
    status: Status = Status.NOT_RUN  # as it last ran; `Session.status` adds whether it is stale
    seconds: float = 0.0
    key: str = ""  # the inputs it ran with, chained
    progress: Progress | None = None  # the pipeline's progress once it had run
    config: TerrainConfig | None = None  # the recipe it ran with
    error: str = ""


class Session:
    def __init__(self, seed: int, executor: Executor | None = None, stages: tuple[Stage, ...] = pipeline.STAGES):
        self.seed = seed
        self.stages = stages
        self.states = {stage.name: StageState() for stage in stages}
        self._executor = executor or ThreadPoolExecutor(1, thread_name_prefix="stages")
        self._posted: queue.SimpleQueue = queue.SimpleQueue()
        self._planned: list[str] = []  # the run in progress: stages not finished yet
        self._before: dict[str, Status] = {}  # their statuses before the run, given back if it fails
        self.running: str | None = None
        self._started = 0.0
        self._stop = threading.Event()  # the worker starts no more stages of the run
        self._lock = threading.Lock()  # between Stop and the worker starting or ending a stage
        self._in_stage: int | None = None  # the worker thread's id while it runs a stage

    # --- what the views show ----------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return bool(self._planned)

    @property
    def stopping(self) -> bool:
        """Stop was asked for and the run has not ended yet."""
        return self.busy and self._stop.is_set()

    def keys(self, config: TerrainConfig) -> dict[str, str]:
        """Each stage's key for `config` and the seed, chained through the stages in order."""
        keys, previous = {}, str(self.seed)
        for stage in self.stages:
            sections = {name: getattr(config, name).model_dump(mode="json") for name in stage.sections}
            blob = json.dumps({"previous": previous, "stage": stage.name, "sections": sections}, sort_keys=True)
            keys[stage.name] = previous = hashlib.sha256(blob.encode()).hexdigest()[:20]
        return keys

    def statuses(self, config: TerrainConfig | None) -> dict[str, Status]:
        """Each stage's status, stale where its key no longer matches `config` (none are, without a config)."""
        keys = self.keys(config) if config is not None else {}
        result = {}
        for name, state in self.states.items():
            status = state.status
            if status in (Status.DONE, Status.SKIPPED) and keys and state.key != keys[name]:
                status = Status.STALE
            result[name] = status
        return result

    def elapsed(self) -> float:
        """Seconds the running stage has taken so far."""
        return time.perf_counter() - self._started if self.running else 0.0

    # --- running ----------------------------------------------------------------------------------

    def plan(self, stage: str, config: TerrainConfig) -> list[str]:
        """What running `stage` runs: every stage before it that is missing or stale, then it."""
        statuses = self.statuses(config)
        names = [s.name for s in self.stages]
        end = names.index(stage)
        first = next((i for i in range(end) if statuses[names[i]] not in (Status.DONE, Status.SKIPPED)), end)
        return names[first:end + 1]

    def run(self, stage: str, config: TerrainConfig) -> list[str]:
        """Start running `stage` (and what it needs first) on the worker; returns what will run."""
        if self.busy:
            raise RuntimeError(f"cannot run {stage}: {self.running} is still running")
        plan = self.plan(stage, config)
        names = [s.name for s in self.stages]
        first = names.index(plan[0])
        progress = self.states[names[first - 1]].progress if first else Progress()
        keys = self.keys(config)
        self._planned = list(plan)
        self._before = {name: self.states[name].status for name in plan}
        for name in plan:
            self.states[name].status = Status.QUEUED
        jobs = [(s, keys[s.name]) for s in self.stages if s.name in plan]
        self._stop.clear()
        self._executor.submit(self._work, jobs, progress, config, self.seed)
        return plan

    def stop(self) -> bool:
        """Stop the run going on, the stage running included; returns whether there was one."""
        if not self.busy:
            return False
        with self._lock:
            if not self._stop.is_set():  # raised once a run: the worker's handlers are not interrupted in turn
                self._stop.set()
                if self._in_stage is not None:
                    _raise_in(self._in_stage, StageStopped)
        return True

    def _work(self, jobs: list[tuple[Stage, str]], progress: Progress, config: TerrainConfig, seed: int) -> None:
        """On the worker: run the stages in order, posting each start and outcome."""
        for stage, key in jobs:
            self._posted.put(("started", stage.name, time.perf_counter()))
            start = time.perf_counter()
            try:
                with self._lock:
                    if self._stop.is_set():
                        raise StageStopped
                    self._in_stage = threading.get_ident()
                try:
                    if not stage.applies(config):
                        self._posted.put(("skipped", stage.name, key, progress, config, 0.0))
                        continue
                    progress = pipeline.run_stage(stage, progress, config, seed)
                finally:
                    with self._lock:
                        self._in_stage = None
                        _raise_in(threading.get_ident(), None)  # a stop that came as the stage ended
            except StageStopped:
                self._in_stage = None  # in case it landed in the `finally` above
                self._posted.put(("stopped", stage.name))
                return
            except Exception as e:  # noqa: BLE001 - any failure is the stage's, shown in the UI
                traceback.print_exc(file=sys.stderr)
                self._posted.put(("failed", stage.name, f"{type(e).__name__}: {e}"))
                return
            self._posted.put(("done", stage.name, key, progress, config, time.perf_counter() - start))

    def poll(self) -> list[str]:
        """Apply what the worker has posted; returns the stages that ended since (done, skipped or failed).
        The stage a stop interrupted, and those it kept from starting, are back as they were before the run,
        and not returned."""
        finished = []
        while True:
            try:
                kind, name, *rest = self._posted.get_nowait()
            except queue.Empty:
                return finished
            state = self.states[name]
            if kind == "started":
                state.status, self.running, self._started = Status.RUNNING, name, rest[0]
            elif kind == "failed":
                state.status, state.error = Status.FAILED, rest[0]
                for later in self._planned[1:]:  # never started: back as they were
                    self.states[later].status = self._before[later]
                self._planned, self.running = [], None
                finished.append(name)
            elif kind == "stopped":
                for later in self._planned:
                    self.states[later].status = self._before[later]
                self._planned, self.running = [], None
            else:
                key, progress, config, seconds = rest
                state.status = Status.DONE if kind == "done" else Status.SKIPPED
                state.key, state.progress, state.config, state.seconds, state.error = key, progress, config, seconds, ""
                self._planned.remove(name)
                if not self._planned:
                    self.running = None
                finished.append(name)

    def close(self) -> None:
        """Stop the run going on and drop queued work."""
        self.stop()
        if isinstance(self._executor, ThreadPoolExecutor):
            self._executor.shutdown(wait=False, cancel_futures=True)


def _raise_in(thread: int, exception: type[BaseException] | None) -> None:
    """Raise `exception` in `thread` at its next Python instruction; None takes back one not raised yet."""
    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread), ctypes.py_object(exception) if exception else None)
