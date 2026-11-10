"""The recipe being edited, kept as the nested dict the forms show, and checked after every change.

The draft may hold values that do not validate (text typed into a number box, a range the wrong way
round): they stay on screen with their problems until fixed. `config` is the validated recipe, or None
while there are problems.
"""

import copy
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from worldgen.config import VERSION
from worldgen.proc_gen.world.config import ConfigError, TerrainConfig, config_errors, yaml_path

FieldPath = tuple[str | int, ...]  # a place in the recipe: ("elevation", "noise", "octaves"), ("biomes", "rules", 3, "id")

HEADER = "# Terrain recipe, saved by worldgen-ui. Keys left out take their defaults.\n\n"


class Recipe:
    def __init__(self, path: Path | None, data: dict):
        self.path = path  # the file it was read from or last saved to; None until it has one
        self.data = data
        self.seed_hint: int | None = None  # a terrain bundle's seed, when the file opened was one
        self.config: TerrainConfig | None = None
        self.problems: dict[str, list[str]] = {}  # YAML path ("" for the whole recipe) -> messages
        self._unreadable: dict[FieldPath, str] = {}  # why the text at a path could not be read
        self._parked: dict[FieldPath, Any] = {}  # values turned off, given back when turned on again
        self.validate()

    @classmethod
    def load(cls, path: Path) -> "Recipe":
        """Read a recipe. A valid one gets every key, defaults included; an invalid one is kept as written,
        with its problems. A terrain bundle's terrain.yaml gives its recipe, and its seed as `seed_hint`.
        Raises ConfigError when the file cannot be read as YAML at all."""
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as e:
            raise ConfigError(f"cannot read {path}: {e.strerror}") from None
        except yaml.YAMLError as e:
            raise ConfigError(f"{path} is not valid YAML: {e}") from None
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping of sections at the top level")
        seed = data.pop("seed", None)  # a bundle's; a recipe's own is rejected by validation
        recipe = cls(path, data)
        recipe.seed_hint = seed if isinstance(seed, int) and seed >= 0 else None
        if recipe.config is not None:
            recipe.data = recipe.config.model_dump(mode="json")
        return recipe

    @classmethod
    def defaults(cls) -> "Recipe":
        """Every value at its model default, as an empty recipe file gives; it has no file until saved."""
        return cls(None, copy.deepcopy(defaults()))

    def text(self) -> str:
        """The draft as a recipe file: YAML, leaving out what is unset. `version` is this worldgen's,
        whatever the file the recipe was read from said."""
        data = {**_without_none(self.data), "version": VERSION}
        return HEADER + yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=110)

    def save(self, path: Path) -> None:
        """Write the draft as a recipe file; from now on the recipe is that file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text(), encoding="utf-8")
        self.path = path

    @property
    def valid(self) -> bool:
        return self.config is not None

    def get(self, path: FieldPath) -> Any:
        """The value at `path`, or None where the recipe leaves it out."""
        value = self.data
        for key in path:
            if isinstance(key, int):
                value = value[key] if isinstance(value, list) and key < len(value) else None
            else:
                value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                return None
        return value

    def set(self, path: FieldPath, value: Any, unreadable: str | None = None) -> None:
        """Put `value` at `path`, then check the whole recipe again. `unreadable` says why typed text
        could not be read; the text is kept as the value so the box still shows it."""
        parent = self.data
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
        """Put `value` into the list at `path` before `index`. Returns how the items after it moved (old index
        -> new), for whatever else keeps things by path."""
        items = self.get(path)
        items.insert(index, value)
        return self._renumber(path, {i: i + (i >= index) for i in range(len(items) - 1)})

    def remove(self, path: FieldPath, index: int) -> dict[int, int | None]:
        """Take item `index` out of the list at `path`. Returns how the items moved (None: removed)."""
        items = self.get(path)
        del items[index]
        return self._renumber(path, {i: None if i == index else i - (i > index) for i in range(len(items) + 1)})

    def move(self, path: FieldPath, index: int, to: int) -> dict[int, int | None]:
        """Move item `index` of the list at `path` to `to`. Returns how the items moved."""
        items = self.get(path)
        items.insert(to, items.pop(index))
        order = list(range(len(items)))
        order.insert(to, order.pop(index))  # order[new index] = old index
        return self._renumber(path, {old: new for new, old in enumerate(order)})

    def _renumber(self, path: FieldPath, order: dict[int, int | None]) -> dict[int, int | None]:
        """Keep values turned off, and text that could not be read, with their list items; then validate."""
        for kept in (self._parked, self._unreadable):
            moved = {renumbered(where, path, order): value for where, value in kept.items()}
            moved.pop(None, None)
            kept.clear()
            kept.update(moved)
        self.validate()
        return order

    def turn_off(self, path: FieldPath, drop_key: bool = False) -> None:
        """Leave out an optional value (None), or with `drop_key` remove a free-form dict's key (a culture,
        a ward height); the value is kept for `turn_on`."""
        parent = self.get(path[:-1])
        if isinstance(parent, dict) and path[-1] in parent:
            self._parked[path] = parent[path[-1]]
            if drop_key:
                del parent[path[-1]]
            else:
                parent[path[-1]] = None
        self._unreadable.pop(path, None)
        self.validate()

    def is_default(self, section: str) -> bool:
        return self.data.get(section) == defaults()[section]

    def reset(self, section: str) -> None:
        """Put a recipe section back to its model defaults, forgetting what was typed or turned off in it."""
        self.data[section] = copy.deepcopy(defaults()[section])
        for kept in (self._parked, self._unreadable):
            for where in [where for where in kept if where[0] == section]:
                del kept[where]
        self.validate()

    def turn_on(self, path: FieldPath, blank: Any) -> None:
        """Give a left-out value back what it had before it was turned off, or `blank`."""
        self.set(path, self._parked.pop(path, blank))

    def problems_at(self, path: FieldPath) -> list[str]:
        return self.problems.get(yaml_path(path), [])

    def validate(self) -> None:
        problems: dict[str, list[str]] = {}
        try:
            self.config = TerrainConfig.model_validate(self.data)
        except ValidationError as e:
            self.config = None
            for where, message in config_errors(e):
                problems.setdefault(where, []).append(message)
        for where, message in self._unreadable.items():
            problems[yaml_path(where)] = [message]  # says more than "Input should be a valid dictionary"
        if self._unreadable:
            self.config = None
        self.problems = problems


@cache
def defaults() -> dict:
    """Every recipe value at its model default, as the forms hold them; copy before changing."""
    return TerrainConfig().model_dump(mode="json")


def renumbered(path: FieldPath, list_path: FieldPath, order: dict[int, int | None]) -> FieldPath | None:
    """`path` once the items of the list at `list_path` have moved as `order` says (old index -> new); None
    when it was inside an item that was removed."""
    n = len(list_path)
    if path[:n] != list_path or len(path) <= n or not isinstance(path[n], int):
        return path
    index = order.get(path[n], path[n])
    return None if index is None else (*list_path, index, *path[n + 1:])


def _has(container: dict | list, key: str | int) -> bool:
    return key in container if isinstance(container, dict) else 0 <= key < len(container)


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_none(v) for v in value]
    return value


class _Dumper(yaml.SafeDumper):
    """Lists of plain values, and small mappings of them, on one line, as world.yaml writes them."""


def _plain(value: Any) -> bool:
    return not isinstance(value, (dict, list))


def _represent_list(dumper: yaml.SafeDumper, value: list):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=all(map(_plain, value)))


def _represent_dict(dumper: yaml.SafeDumper, value: dict):
    small = len(value) <= 4 and all(
        _plain(v) or (isinstance(v, list) and all(map(_plain, v))) for v in value.values()
    )
    return dumper.represent_mapping("tag:yaml.org,2002:map", value, flow_style=small)


_Dumper.add_representer(list, _represent_list)
_Dumper.add_representer(dict, _represent_dict)
