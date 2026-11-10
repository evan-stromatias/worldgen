"""Adding, removing and moving biome rules. Order matters, since the first rule that matches a cell wins, and
the last rule is the catch-all (no `when`): it stays last, so it never moves, is never removed, and no rule
moves past it.
"""

from itertools import count

from worldgen.generator.config import BiomeRule, Ground, When

RULES = ("biomes", "rules")


def new_rule(rules: list) -> dict:
    """A rule to add before the catch-all: valid as it stands, with an id no rule has. It matches low land,
    which the rules before it may already have taken: edit its `when` to say where it applies."""
    taken = {rule.get("id") for rule in rules if isinstance(rule, dict)}
    name = next(f"biome_{n}" for n in count(len(rules) + 1) if f"biome_{n}" not in taken)
    rule = BiomeRule(id=name, when=When(elevation=(0.0, 100.0)), ground=Ground(color=(128, 128, 128)))
    return rule.model_dump(mode="json")  # every key, as a loaded recipe has them


def can_move(rules: list, index: int, to: int) -> bool:
    """Whether rule `index` may move to `to`: neither is the catch-all's place."""
    last = len(rules) - 1
    return index != to and 0 <= index < last and 0 <= to < last


def can_remove(rules: list, index: int) -> bool:
    return 0 <= index < len(rules) - 1
