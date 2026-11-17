"""The cave recipe being edited in the UI, kept as a nested dict and validated after every change."""

from __future__ import annotations

import copy
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from worldgen.cave.config import CaveConfig
from worldgen.config import VERSION
from worldgen.generator.config import ConfigError, config_errors, yaml_path
from worldgen.ui.model.recipe.recipe import FieldPath, _Dumper, _without_none, renumbered

HEADER = "# Cave recipe, saved by cavegen-ui. Keys left out take their defaults.\n\n"


class CaveRecipe:
    """A draft CaveConfig being edited in the UI."""

    def __init__(self, path: Path | None, data: dict):
        self.path = path
        self.data = data
        self.seed_hint: int | None = None
        self.config: CaveConfig | None = None
        self.problems: dict[str, list[str]] = {}
        self._unreadable: dict[FieldPath, str] = {}
        self._parked: dict[FieldPath, Any] = {}
        self.validate()

    @classmethod
    def load(cls, path: Path) -> CaveRecipe:
        """Read a cave recipe from YAML."""
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as e:
            raise ConfigError(f"cannot read {path}: {e.strerror}") from None
        except yaml.YAMLError as e:
            raise ConfigError(f"{path} is not valid YAML: {e}") from None
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping of sections at the top level")
        seed = data.pop("seed", None)
        recipe = cls(path, data)
        recipe.seed_hint = seed if isinstance(seed, int) and seed >= 0 else None
        if recipe.config is not None:
            recipe.data = recipe.config.model_dump(mode="json")
        return recipe

    @classmethod
    def defaults(cls) -> CaveRecipe:
        """Every value at its model default."""
        return cls(None, copy.deepcopy(defaults()))

    def text(self) -> str:
        """The draft as a YAML recipe file."""
        data = {**_without_none(self.data), "version": VERSION}
        client = self.data.get("client")
        if client:
            data["client"] = client
        else:
            data.pop("client", None)
        return HEADER + yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=110)

    def save(self, path: Path) -> None:
        """Write draft to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text(), encoding="utf-8")
        self.path = path

    @property
    def valid(self) -> bool:
        return self.config is not None

    def get(self, path: FieldPath) -> Any:
        """Get value at path, or None if omitted."""
        value: Any = self.data
        for key in path:
            if isinstance(key, int):
                value = value[key] if isinstance(value, list) and key < len(value) else None
            else:
                value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                return None
        return value

    def set(self, path: FieldPath, value: Any, unreadable: str | None = None) -> None:
        """Set value at path and revalidate."""
        parent: Any = self.data
        for key, next_key in zip(path[:-1], path[1:], strict=True):
            child = parent[key] if _has(parent, key) else None
            if not isinstance(child, (dict, list)):
                child = [None] * (next_key + 1) if isinstance(next_key, int) else {}
                parent[key] = child
            parent = child
        if isinstance(parent, list) and path[-1] >= len(parent):
            parent.extend([None] * (path[-1] + 1 - len(parent)))
        parent[path[-1]] = value
        self._unreadable.pop(path, None)
        if unreadable:
            self._unreadable[path] = unreadable
        self.validate()

    def insert(self, path: FieldPath, index: int, value: Any) -> dict[int, int | None]:
        items = self.get(path)
        items.insert(index, value)
        return self._renumber(path, {i: i + (i >= index) for i in range(len(items) - 1)})

    def remove(self, path: FieldPath, index: int) -> dict[int, int | None]:
        items = self.get(path)
        del items[index]
        return self._renumber(path, {i: None if i == index else i - (i > index) for i in range(len(items) + 1)})

    def move(self, path: FieldPath, index: int, to: int) -> dict[int, int | None]:
        items = self.get(path)
        items.insert(to, items.pop(index))
        order = list(range(len(items)))
        order.insert(to, order.pop(index))
        return self._renumber(path, {old: new for new, old in enumerate(order)})

    def _renumber(self, path: FieldPath, order: dict[int, int | None]) -> dict[int, int | None]:
        for kept in (self._parked, self._unreadable):
            moved = {renumbered(where, path, order): value for where, value in kept.items()}
            moved.pop(None, None)
            kept.clear()
            kept.update(moved)
        self.validate()
        return order

    def turn_off(self, path: FieldPath, drop_key: bool = False) -> None:
        parent = self.get(path[:-1])
        if isinstance(parent, dict) and path[-1] in parent:
            self._parked[path] = parent[path[-1]]
            if drop_key:
                del parent[path[-1]]
            else:
                parent[path[-1]] = None
        self._unreadable.pop(path, None)
        self.validate()

    def turn_on(self, path: FieldPath, blank: Any) -> None:
        self.set(path, self._parked.pop(path, blank))

    def is_default(self, section: str) -> bool:
        return self.data.get(section) == defaults().get(section)

    def reset(self, section: str) -> None:
        """Reset section back to its model defaults."""
        self.data[section] = copy.deepcopy(defaults()[section])
        for kept in (self._parked, self._unreadable):
            for where in [where for where in kept if where and where[0] == section]:
                del kept[where]
        self.validate()

    def problems_at(self, path: FieldPath) -> list[str]:
        return self.problems.get(yaml_path(path), [])

    def validate(self) -> None:
        problems: dict[str, list[str]] = {}
        try:
            self.config = CaveConfig.model_validate(self.data)
        except ValidationError as e:
            self.config = None
            for where, message in config_errors(e):
                problems.setdefault(where, []).append(message)
        for where, message in self._unreadable.items():
            problems[yaml_path(where)] = [message]
        if self._unreadable:
            self.config = None
        self.problems = problems


@cache
def defaults() -> dict:
    """Default values for CaveConfig as JSON-compatible dict."""
    return CaveConfig().model_dump(mode="json")


def _has(container: dict | list, key: str | int) -> bool:
    return key in container if isinstance(container, dict) else 0 <= key < len(container)
