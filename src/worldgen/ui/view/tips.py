"""Tooltips lay out a readout's facts as a table of muted names and right-aligned values, under a title.
No raylib."""

from rayscreen import SMALL, TEXT, Column, Label, Node, Separator, Table, Tooltip
from worldgen.ui.model.map.readout import Fact, Readout


def facts_table(prefix: str, facts: list[Fact]) -> Table:
    """A tooltip's facts: the names muted in one column, the values right-aligned in the next. A name may start
    with spaces, to sit under the fact it breaks down. The labels are a text line tall, so a `Text` row added
    after them sits level."""
    return Table([[Label(f"{prefix}.{i}.name", name, height=SMALL + 4, size=SMALL, tone="muted"),
                   Label(f"{prefix}.{i}.value", value, height=SMALL + 4, size=SMALL)]
                  for i, (name, value) in enumerate(facts)], align=("left", "right"))


def readout_tooltip(found: Readout, footer: str | None = None) -> Tooltip:
    """What lies under the mouse: its title, then its facts as a table, then `footer` (what more there is to do
    there: "right-click: Details... for more")."""
    children: list[Node] = [Label("tip.title", found.title, height=TEXT + 4)]
    if found.facts:
        children += [Separator(), facts_table("tip", found.facts)]
        if footer:
            children += [Separator(), Label("tip.more", footer, height=SMALL + 4, size=SMALL, tone="muted")]
    return Tooltip(found.key, Column(children=children, gap=2))
