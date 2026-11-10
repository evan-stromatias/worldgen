"""The recipe's forms, built from its pydantic classes: what to draw for each field, and its help text.

The view draws these specs and never looks at pydantic. A field whose type has no widget raises
TypeError here, so a new recipe field cannot go missing from the UI unnoticed.
"""

import inspect
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import annotated_types
import yaml
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from worldgen.proc_gen.world.config import TerrainConfig, WorldSection
from worldgen.utils.gen_ui.model.recipe import FieldPath


class Kind(Enum):
    TEXT = auto()
    INT = auto()
    FLOAT = auto()
    BOOL = auto()
    CHOICE = auto()  # one of `choices`
    PAIR = auto()  # two numbers, each of kind `item`: a range, a direction, a cell
    COLOR = auto()  # RGB, 0-255 each
    COUNTS = auto()  # an int for each of `choices`
    WEIGHTS = auto()  # an optional number for each of `choices`
    PICK = auto()  # some of `choices`, or with none of them the recipe's biome ids
    YAML = auto()  # a free-form mapping, edited as YAML text
    RULES = auto()  # the biome rules, each drawn from `rule`


@dataclass(frozen=True, slots=True)
class Bounds:
    ge: float | None = None
    gt: float | None = None
    le: float | None = None
    lt: float | None = None

    def __str__(self) -> str:
        parts = [
            f"{word} {value:g}"
            for word, value in (("at least", self.ge), ("above", self.gt), ("at most", self.le), ("below", self.lt))
            if value is not None
        ]
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class FieldSpec:
    key: str
    kind: Kind
    description: str
    bounds: Bounds = Bounds()
    choices: tuple[str, ...] = ()
    optional: bool = False  # may be left out (None): a check box turns it on and off
    item: Kind | None = None  # PAIR: INT or FLOAT
    rule: "SectionSpec | None" = None  # RULES: the form of one rule
    step_by: str = ""  # INT: the sibling field whose value its - and + buttons step by


@dataclass(frozen=True, slots=True)
class SectionSpec:
    key: str
    title: str
    description: str
    fields: tuple["FieldSpec | SectionSpec", ...] = field(default=())
    optional: bool = False

    def walk(self, prefix: FieldPath = ()) -> Iterator[tuple[FieldPath, FieldSpec]]:
        """Every field below this section with its path; rules are left to the caller, which knows how many."""
        for spec in self.fields:
            path = (*prefix, spec.key)
            if isinstance(spec, SectionSpec):
                yield from spec.walk(path)
            else:
                yield path, spec

    def find(self, path: FieldPath) -> "FieldSpec | SectionSpec":
        spec = self
        for key in path:
            spec = next(s for s in spec.fields if s.key == key)
        return spec

    def locate(self, path: FieldPath) -> tuple["FieldSpec | SectionSpec", FieldPath]:
        """The spec that draws `path`, and what is left of the path below it (a pair's index, a dict's key).

        A biome rule's index leads into the rule's form, so ("biomes", "rules", 3, "id") is the rule's `id`.
        """
        spec, i = self, 0
        while i < len(path):
            if isinstance(spec, SectionSpec):
                spec = next(s for s in spec.fields if s.key == path[i])
            elif spec.kind is Kind.RULES and isinstance(path[i], int):
                spec = spec.rule
            else:
                break
            i += 1
        return spec, path[i:]


def box_kind(spec: FieldSpec, rest: FieldPath) -> Kind:
    """What one text box of `spec` holds: `rest` picks a pair's or colour's part, or a dict's key."""
    if rest and spec.kind is Kind.PAIR:
        return spec.item
    if rest and spec.kind in (Kind.COLOR, Kind.COUNTS):
        return Kind.INT
    if rest and spec.kind is Kind.WEIGHTS:
        return Kind.FLOAT
    return spec.kind


def parse(kind: Kind, text: str) -> tuple[Any, str | None]:
    """Typed text as a value of `kind`, and a message if it could not be read.

    Numbers that do not parse stay as text, so validation names the problem and the box keeps what was
    typed. An empty box means "left out" (None), which validation rejects where a value is required.
    """
    text = text.strip()
    if kind is Kind.TEXT:
        return text, None
    if not text:
        return None, None
    if kind is Kind.INT:
        try:
            return int(text), None
        except ValueError:
            return text, None
    if kind is Kind.FLOAT:
        try:
            return float(text), None
        except ValueError:
            return text, None
    if kind is Kind.YAML:
        try:
            return yaml.safe_load(text), None
        except yaml.YAMLError as e:
            return text, f"not valid YAML: {getattr(e, 'problem', None) or e}"
    raise ValueError(f"{kind} is not typed into a text box")


def blank(spec: "FieldSpec | SectionSpec", rest: FieldPath = ()) -> Any:
    """A value for a field, or a dict key, that was left out and has just been turned on."""
    if isinstance(spec, SectionSpec):
        return {}
    if rest and spec.kind is Kind.WEIGHTS:
        return 1.0
    zero = 0 if spec.item is Kind.INT else 0.0
    return {
        Kind.TEXT: "",
        Kind.INT: 0,
        Kind.FLOAT: 0.0,
        Kind.BOOL: False,
        Kind.CHOICE: spec.choices[0] if spec.choices else None,
        Kind.PAIR: [zero, zero],
        Kind.COLOR: [128, 128, 128],
        Kind.COUNTS: {},
        Kind.WEIGHTS: {},
        Kind.PICK: [],
        Kind.YAML: {},
        Kind.RULES: [],
    }[spec.kind]


def stepped(value: Any, step: Any, up: bool, bounds: Bounds) -> int | None:
    """The multiple of `step` next above or below `value`, so a value that is not one snaps to one; None when
    either is not a number yet (or `step` is not above zero), or the multiple falls outside `bounds`."""
    if not all(isinstance(n, int) and not isinstance(n, bool) for n in (value, step)) or step <= 0:
        return None
    new = (value // step + 1) * step if up else (value - 1) // step * step
    outside = (
        (bounds.ge is not None and new < bounds.ge)
        or (bounds.gt is not None and new <= bounds.gt)
        or (bounds.le is not None and new > bounds.le)
        or (bounds.lt is not None and new >= bounds.lt)
    )
    return None if outside else new


HIDDEN = {(TerrainConfig, "version")}
KNOWN_KEYS = {}  # dicts whose keys come from elsewhere: one row per key
BIOME_IDS: set = set()  # fields that pick from the recipe's biome ids
STEPS = {(WorldSection, "size_cells"): "patch_cells"}  # ints that must be a multiple of another field


def recipe_form() -> SectionSpec:
    """The form of a whole recipe: one section per recipe section, in the recipe's order."""
    return _section(TerrainConfig, "", "Recipe")


def _section(model: type[BaseModel], key: str, title: str, optional: bool = False) -> SectionSpec:
    fields = tuple(
        _field(model, name, info) for name, info in model.model_fields.items() if (model, name) not in HIDDEN
    )
    return SectionSpec(key, title, inspect.cleandoc(model.__doc__ or ""), fields, optional)


def _title(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _field(owner: type[BaseModel], key: str, info: FieldInfo) -> "FieldSpec | SectionSpec":
    annotation, optional = _unwrap_optional(info.annotation)
    annotation, metadata = _unwrap_annotated(annotation, list(info.metadata))
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _section(annotation, key, _title(key), optional)

    def spec(kind: Kind, **more) -> FieldSpec:
        more.setdefault("bounds", _bounds(metadata))
        return FieldSpec(key, kind, info.description or "", optional=optional, **more)

    scalar = _scalar_kind(annotation)
    if scalar is not None:
        return spec(scalar, step_by=STEPS.get((owner, key), ""))
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Literal:
        return spec(Kind.CHOICE, choices=tuple(str(a) for a in args))
    if origin is tuple:
        items = [_unwrap_annotated(a, []) for a in args]
        kinds = {_scalar_kind(a) for a, _ in items}
        if len(items) == 2 and len(kinds) == 1 and kinds <= {Kind.INT, Kind.FLOAT}:
            return spec(Kind.PAIR, item=kinds.pop())
        if len(items) == 3 and kinds == {Kind.INT}:
            return spec(Kind.COLOR, bounds=_bounds(items[0][1]))
    if origin is dict:
        keys, (value, value_metadata) = args[0], _unwrap_annotated(args[1], [])
        if get_origin(keys) is Literal and _scalar_kind(value) is Kind.INT:
            return spec(Kind.COUNTS, choices=tuple(get_args(keys)), bounds=_bounds(value_metadata))
        if (owner, key) in KNOWN_KEYS and _scalar_kind(value) is Kind.FLOAT:
            return spec(Kind.WEIGHTS, choices=KNOWN_KEYS[owner, key](), bounds=_bounds(value_metadata))
        if keys is str and value is Any:
            return spec(Kind.YAML)
    if origin is list:
        item = _unwrap_annotated(args[0], [])[0]
        if (owner, key) in BIOME_IDS and item is str:
            return spec(Kind.PICK)
        if isinstance(item, type) and issubclass(item, BaseModel):
            return spec(Kind.RULES, rule=_section(item, "", _title(item.__name__)))
    raise TypeError(f"{owner.__name__}.{key}: the terrain generator UI has no widget for {annotation!r}")


def _scalar_kind(annotation: Any) -> Kind | None:
    # bool before int: bool is a subclass of int
    return {bool: Kind.BOOL, int: Kind.INT, float: Kind.FLOAT, str: Kind.TEXT}.get(annotation)


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    if get_origin(annotation) in (Union, types.UnionType):
        rest = [a for a in get_args(annotation) if a is not type(None)]
        if len(rest) == 1:
            return rest[0], True
    return annotation, False


def _unwrap_annotated(annotation: Any, metadata: list) -> tuple[Any, list]:
    """The bare type, and the constraints from any Annotated[...] around it added to `metadata`."""
    if get_origin(annotation) is Annotated:
        base, *extra = get_args(annotation)
        for item in extra:
            metadata = [*metadata, *(item.metadata if isinstance(item, FieldInfo) else [item])]
        return _unwrap_annotated(base, metadata)
    return annotation, metadata


def _bounds(metadata: list) -> Bounds:
    found = {}
    for item in metadata:
        for name, kind in (("ge", annotated_types.Ge), ("gt", annotated_types.Gt), ("le", annotated_types.Le),
                           ("lt", annotated_types.Lt)):
            if isinstance(item, kind):
                found[name] = getattr(item, name)
    return Bounds(**found)
