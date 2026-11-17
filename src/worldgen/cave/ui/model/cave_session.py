"""Cave generation session: runs cave generation and mesh building on worker threads."""

import threading

import numpy as np

from worldgen.cave.config import CaveConfig
from worldgen.cave.export import render_cave_preview
from worldgen.cave.generator import CaveData, generate_cave
from worldgen.cave.mesh import build_cave_mesh, build_cave_water_mesh
from worldgen.mesh.builders import MeshArrays


class CaveSession:
    """Manages generation of a subterranean cave, its preview, and its 3D meshes."""

    def __init__(self, config: CaveConfig, seed: int):
        self.config = config
        self.seed = seed
        self.cave: CaveData | None = None
        self.mesh_arrays: MeshArrays | None = None
        self.water_mesh_arrays: MeshArrays | None = None
        self.preview_image: np.ndarray | None = None
        self.stride: int = 1
        self.cutaway_height: float | None = None
        self.include_ceiling: bool = True

        self._thread: threading.Thread | None = None
        self._result: tuple[CaveData, MeshArrays, MeshArrays | None, np.ndarray] | None = None
        self._error: str | None = None
        self.status: str = "idle"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def settled(self) -> bool:
        return not self.running

    def start_generation(self) -> None:
        """Start cave generation in background worker thread."""
        if self.running:
            return

        self.status = "generating"
        self._result = None
        self._error = None

        config = self.config
        seed = self.seed
        stride = self.stride
        include_ceiling = self.include_ceiling
        cutaway_height = self.cutaway_height

        def worker() -> None:
            try:
                cave = generate_cave(config, seed)
                mesh = build_cave_mesh(
                    cave,
                    stride=stride,
                    include_ceiling=include_ceiling,
                    cutaway_height=cutaway_height,
                )
                water_mesh = build_cave_water_mesh(cave, stride=stride)
                preview = render_cave_preview(cave)
                self._result = (cave, mesh, water_mesh, preview)
            except Exception as e:  # noqa: BLE001
                self._error = str(e)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def run_sync(self) -> None:
        """Run cave generation and mesh building synchronously."""
        self.cave = generate_cave(self.config, self.seed)
        self.mesh_arrays = build_cave_mesh(
            self.cave,
            stride=self.stride,
            include_ceiling=self.include_ceiling,
            cutaway_height=self.cutaway_height,
        )
        self.water_mesh_arrays = build_cave_water_mesh(self.cave, stride=self.stride)
        self.preview_image = render_cave_preview(self.cave)
        self.status = "settled"

    def rebuild_mesh(self) -> None:
        """Rebuild meshes from existing cave data (e.g. when cutaway height or stride changes)."""
        if self.cave is None:
            return
        self.mesh_arrays = build_cave_mesh(
            self.cave,
            stride=self.stride,
            include_ceiling=self.include_ceiling,
            cutaway_height=self.cutaway_height,
        )
        self.water_mesh_arrays = build_cave_water_mesh(self.cave, stride=self.stride)

    def poll(self) -> bool:
        """Poll background worker. Returns True if fresh results just arrived."""
        if self._result is not None:
            self.cave, self.mesh_arrays, self.water_mesh_arrays, self.preview_image = self._result
            self._result = None
            self._thread = None
            self.status = "settled"
            return True
        if self._error is not None:
            self.status = f"error: {self._error}"
            self._error = None
            self._thread = None
        return False
