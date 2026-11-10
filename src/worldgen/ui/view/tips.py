"""A readout's facts laid out as a table of muted names and right-aligned values, for tooltips and menus.
No raylib."""

from rayscreen import SMALL, Label, Table
from worldgen.ui.model.map.readout import Fact


def facts_table(prefix: str, facts: list[Fact]) -> Table:
    """A tooltip's facts: the names muted in one column, the values right-aligned in the next. A name may start
    with spaces, to sit under the fact it breaks down. The labels are a text line tall, so a `Text` row added
    after them sits level."""
    return Table([[Label(f"{prefix}.{i}.name", name, height=SMALL + 4, size=SMALL, tone="muted"),
                   Label(f"{prefix}.{i}.value", value, height=SMALL + 4, size=SMALL)]
                  for i, (name, value) in enumerate(facts)], align=("left", "right"))
