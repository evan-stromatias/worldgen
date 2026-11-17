"""Form specification for cave recipes, derived automatically from CaveConfig."""

from __future__ import annotations

import inspect

from worldgen.cave.config import CaveConfig
from worldgen.ui.model.recipe.form import SectionSpec, _section

HIDDEN = {"version", "client"}


def cave_recipe_form() -> SectionSpec:
    """Build the SectionSpec tree for CaveConfig, omitting internal/client fields."""
    raw = _section(CaveConfig, "", "Cave Recipe")
    fields = tuple(f for f in raw.fields if f.key not in HIDDEN)
    return SectionSpec(
        key="",
        title="Cave Recipe",
        description=inspect.cleandoc(CaveConfig.__doc__ or ""),
        fields=fields,
        optional=False,
    )
