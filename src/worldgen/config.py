"""Where terrain generation finds its bundled files: the default recipe, kept with the repository's
assets rather than inside the package."""

from pathlib import Path

DEFAULT_RECIPE = Path(__file__).resolve().parents[2] / "assets" / "recipes" / "default.yaml"
