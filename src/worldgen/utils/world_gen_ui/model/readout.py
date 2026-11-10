"""What lies under the mouse, as data for the views to lay out: a title and facts (a name and its value each).
The tooltip lays the facts out as a table; the status bar writes them on one line."""

from collections.abc import Hashable
from dataclasses import dataclass

type Fact = tuple[str, str]  # what, and its value: a row of a tooltip's or a details window's table


@dataclass(frozen=True, slots=True)
class Readout:
    """What lies under the mouse: what it is or where, and its facts (none for something with nothing to tell)."""

    key: Hashable  # what is hovered (a building, a cell): the tooltip waits while it stays the same
    title: str
    facts: list[Fact]

    def text(self) -> str:
        """On one line, for the status bar."""
        return " · ".join([self.title, *(f"{name} {value}" for name, value in self.facts)])
