"""The generator's stages in pipeline order, and the states the UI shows for each."""

from enum import Enum

from worldgen.proc_gen.world import pipeline

# Each stage is tuned by the recipe section of the same name
STAGES = tuple(stage.name for stage in pipeline.STAGES)


class Status(Enum):
    NOT_RUN = "not run"
    QUEUED = "waiting"
    RUNNING = "running"
    DONE = "done"
    STALE = "stale"
    SKIPPED = "skipped"
    FAILED = "failed"
